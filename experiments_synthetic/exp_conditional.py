import sys, os
sys.path.append('../smc')

import numpy as np   
import pandas as pd
import sys

from utils import *     # contains some useful helper functions 
from models import *    # toy models
from solvers import *   # matrix completion solvers
from methods import *
from wsc import *



#########################
# Experiment parameters #
#########################
if True:
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 7:
        print("Error: incorrect number of parameters.")
        quit()

    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    r = int(sys.argv[3])
    delta = float(sys.argv[4])
    exp = str(sys.argv[5])
    seed = int(sys.argv[6])


# Fixed data parameters
max_test_queries = 200            
max_calib_queries = 2000
matrix_generation_seed = 2024    # Data matrix is fixed 

methods = ["conditional", 
           "unconditional"]
model = "RFM"
solver = "pmf"
noise_model = "step"
mu = 10
gamma_n = 0.5
gamma_m = 0.9
prop_obs = 0.3

# Other parameters
verbose = True
allow_inf = False
alpha = 0.1

k_list = np.arange(1,5)
repetition = 2



###############
# Output file #
###############
outdir = f"./results/exp_conditional_{exp}/"
os.makedirs(outdir, exist_ok=True)
outfile_name = f"{n1}by{n2}_r{r}_delta{delta:.2f}_seed{seed}"
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")
sys.stdout.flush()

# Header for results file
def add_header(df):
    df["n1"] = n1
    df["n2"] = n2
    df['alpha'] = alpha
    df['r'] = r
    df['delta'] = delta
    df['exp'] = exp
    return df



#################
# Generate Data #
#################
if model == "RFM":
    mm = RandomFactorizationModel(n1 ,n2, r)
elif model == "ROM":
    mm = RandomOrthogonalModel(n1 ,n2, r)
else:
    mm = RandomFactorizationModel(n1 ,n2, r)

if verbose:
    print('Fixing the ground truth matrix generated from the {} model.\n'.format(model))
    sys.stdout.flush()


#---------Noisy Matrix----------#
#-------------------------------#
_, _, M = mm.sample_noiseless(matrix_generation_seed)
nm = NoiseModel(matrix_generation_seed)
M = nm.get_noisy_matrix(M, gamma_n=gamma_n, gamma_m=gamma_m, model=noise_model, 
                        mu=mu, alpha=alpha, normalize=False)



#####################
# Define Experiment #
#####################
def run_single_experiment(M_true, k, alpha, prop_obs, max_test_queries, max_calib_queries,
                          r, delta, random_state=0):
    res = pd.DataFrame({})


    #-------Generate masks----------#
    #-------------------------------#
    n1, n2 = M_true.shape
    sampler = QuerySampling(n1,n2)
    mask_obs, mask_miss = sampler.sample_submask(sub_size=prop_obs, random_state=random_state)


    #---------Calib queries---------#
    #-------------------------------#
    n_calib_queries = min(int(0.5 * np.sum(np.sum(mask_obs, axis=1) // k)), max_calib_queries)
    mask_train, idxs_calib, _ = sampler.sample_train_calib(mask_obs, k, 
                                            calib_size=n_calib_queries, random_state=random_state)


    #--------Model Training---------#
    #-------------------------------#
    print("Running matrix completion algorithm on the splitted training set...")
    sys.stdout.flush()
    if solver == "pmf":
        Mhat, Uhat, Vhat = pmf_solve(M, mask_train, k=r, verbose=verbose, random_state=random_state)
    elif solver == "svt":
        Mhat = svt_solve(M, mask_train, verbose = verbose, random_state = random_state)
    print("Done training!\n")
    sys.stdout.flush()

    
    #---------Test queries----------#
    #-------------------------------#
    if exp == "wsc":
        wsc_param, mask_test = wsc_estimate(M, Mhat, Uhat, Vhat, mask_miss, delta=delta, random_state=random_state)
        bias = SamplingBias(n1,n2)
        w = bias.latent_weights(Uhat, Vhat, r, *wsc_param, scale=(wsc_param[2]-wsc_param[1])/5)
    else:
        print("Unknown type of experiment.")
        quit()

    n_test_queries = min(np.sum(np.sum(mask_test, axis=1) // k), max_test_queries)
    idxs_test= sampler.sample_test(mask_test, k, test_size=n_test_queries, w=w, replace=True, random_state=random_state)
    if verbose:
        print("Training size:{}, calib size: {}, test size: {}\n".format(np.sum(mask_obs)-n_calib_queries*k, n_calib_queries, n_test_queries))
        sys.stdout.flush()
    

    ci_method = SimulCI(M, Mhat, mask_obs, idxs_calib, k)
    for method in methods:
        w_method = w if method == "conditional" else None
        df = ci_method.get_CI(idxs_test, alpha, w_test= w_method, allow_inf=allow_inf)
        lower, upper, is_inf= df.loc[0].lower, df.loc[0].upper, df.loc[0].is_inf
        res = pd.concat([res, evaluate_SCI(lower, upper, k, M, idxs_test, is_inf=is_inf, method=method)])


    res['k'] = k     
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

        res = run_single_experiment(M, k, alpha, prop_obs, max_test_queries, max_calib_queries,
                          r, delta, random_state=random_state)
        
        results = pd.concat([results, res])

add_header(results)



#####################
#    Save Results   #
#####################
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()
