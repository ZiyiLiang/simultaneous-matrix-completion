import sys, os
sys.path.append('../smc')
sys.path.append('../third_party')

import numpy as np   
import pandas as pd
from tqdm import tqdm
from time import time
import sys
import scipy.stats as stats

from utils import *     # contains some useful helper functions 
from models import *    # toy models
from solvers import *   # matrix completion solvers
from methods import *
from missingness_estimation import *


#########################
# Experiment parameters #
#########################
if True:
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 4:
        print("Error: incorrect number of parameters.")
        quit()

    r = int(sys.argv[1])
    mu = int(sys.argv[2])
    seed = int(sys.argv[3])


# Fixed data parameters
max_test_queries = 100            
max_calib_queries = 1000
matrix_generation_seed = 2024    # Data matrix is fixed 

model = "RFM"
solver = "pmf"
r_solver = 30

n1 = n2 = 200
noise_model = "step"
prop_obs = 0.2
gamma_n = 0.5
gamma_m = 0.9

# Other parameters
verbose = True
allow_inf = False
alpha = 0.1

k_list = np.arange(2,9)
repetition = 10



###############
# Output file #
###############
outdir = "./results/exp_est_uniform/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "r" + str(r) + "_mu" + str(mu) + "_seed" + str(seed)
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")
sys.stdout.flush()

# Header for results file
def add_header(df):
    df["n1"] = n1
    df["n2"] = n2
    df['alpha'] = alpha
    df['r_est'] = r
    df['r_solver'] = r_solver
    df['gamma_n'] = gamma_n
    df['gamma_m'] = gamma_m
    df['mu'] = mu
    return df



#################
# Generate Data #
#################
if model == "RFM":
    mm = RandomFactorizationModel(n1 ,n2, 5)
elif model == "ROM":
    mm = RandomOrthogonalModel(n1 ,n2, 5)
else:
    mm = RandomFactorizationModel(n1 ,n2, 5)

if verbose:
    print('Fixing the ground truth matrix generated from the {} model.\n'.format(model))
    sys.stdout.flush()

U, V, M_true = mm.sample_noiseless(matrix_generation_seed)



def run_single_experiment(M_true, k, alpha, prop_obs, max_test_queries, max_calib_queries,
                          r, gamma_n=0, gamma_m=0, mu=1, random_state=0):
    res = pd.DataFrame({})


    #-------Generate masks----------#
    #-------------------------------#
    n1, n2 = M_true.shape
    sampler = QuerySampling(n1,n2)
    mask_obs, mask_test = sampler.sample_submask(sub_size=prop_obs, random_state=random_state)
    n_calib_queries = min(int(0.5 * np.sum(np.sum(mask_obs, axis=1) // k)), max_calib_queries)

    print(f"Estimating missingness with guessed rank {r}...")
    w_obs_est = estimate_P(mask_obs, 1, r=r)
    print("Done estimating!\n")
    sys.stdout.flush()

    #------Sample test queries------#
    #-------------------------------#
    n_test_queries = min(int(0.99 * np.sum(np.sum(mask_test, axis=1) // k)), max_test_queries)
    _, idxs_test, _ = sampler.sample_train_calib(mask_test, k, calib_size=n_test_queries, random_state=random_state)  
    if verbose:
        print("Training size:{}, calib size: {}, test size: {}\n".format(np.sum(mask_obs)-n_calib_queries*k, n_calib_queries, n_test_queries))
        sys.stdout.flush()

    
    #--------Generate noise---------#
    #-------------------------------#
    nm = NoiseModel(random_state)
    M = nm.get_noisy_matrix(M_true, gamma_n=gamma_n, gamma_m=gamma_m, model=noise_model, 
                            mu=mu, alpha=alpha, normalize=False)


    
    #------Split train calib--------#
    #-------------------------------#   
    mask_train, idxs_calib, _ = sampler.sample_train_calib(mask_obs, k, 
                                calib_size=n_calib_queries, random_state=random_state)
   

    #--------Model Training---------#
    #-------------------------------#
    print("Running matrix completion algorithm on the training set...")
    sys.stdout.flush()
    tik = time()
    if solver == "pmf":
        Mhat, _, _ = pmf_solve(M, mask_train, k=r_solver, verbose=verbose, random_state=random_state)
    elif solver == "svt":
        Mhat = svt_solve(M, mask_train, tau=None, verbose = verbose, random_state = random_state)
    elif solver == "nnm":
        Mhat = nnm_solve(M, mask_train, verbose=verbose, random_state=random_state)
    print("Done training!\n")
    sys.stdout.flush() 

    #------Compute intervals--------# 
    #-------------------------------#
    
    # Evaluate the CI and quantile inflation weights using oracle obs sampling weights
    ci_method = SimulCI(M, Mhat, mask_obs, idxs_calib, k)
    df = ci_method.get_CI(idxs_test, alpha, allow_inf=allow_inf, store_weights=True)
    lower, upper, is_inf= df.loc[0].lower, df.loc[0].upper, df.loc[0].is_inf
    res = pd.concat([res, evaluate_SCI(lower, upper, k, M, idxs_test, is_inf=is_inf, method="conformal")])
    
    # Evaluate the CI and quantile inflation weights using estimated obs sampling weights
    ci_est = SimulCI(M, Mhat, mask_obs, idxs_calib, k, w_obs=w_obs_est)
    df = ci_est.get_CI(idxs_test, alpha, allow_inf=allow_inf, store_weights=True)
    lower, upper, is_inf= df.loc[0].lower, df.loc[0].upper, df.loc[0].is_inf
    res = pd.concat([res, evaluate_SCI(lower, upper, k, M, idxs_test, is_inf=is_inf, method="est")])

    # Evaluate the estimation gap
    weights_list = ci_method.weights_list
    est_weights_list = ci_est.weights_list
    est_gaps =[0.5*np.sum(np.abs(weights_list[i]-est_weights_list[i])) for i in range(len(weights_list))]
    avg_gap = np.mean(est_gaps)


    res['k'] = k     
    res['avg_gap'] = avg_gap   
    res['Calib_queries'] = n_calib_queries
    res['Train_entries'] = np.sum(mask_train)
    res['Test_queries'] = n_test_queries
    res['random_state'] = random_state
    return res



#####################
#  Run Experiments  #
#####################
results = pd.DataFrame({})

for i in tqdm(range(1, repetition+1), desc="Repetitions", leave=True, position=0):
    random_state = repetition * (seed-1) + i
    
    for k in tqdm(k_list, desc="k", leave=True, position=0):

        res = run_single_experiment(M_true, k, alpha, prop_obs, max_test_queries, max_calib_queries,
                            r, gamma_n=gamma_n, gamma_m=gamma_m, mu=mu, random_state=random_state)
        
        results = pd.concat([results, res])

add_header(results)



#####################
#    Save Results   #
#####################
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()
