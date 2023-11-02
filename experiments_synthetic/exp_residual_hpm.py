import sys, os
sys.path.append('../pairedRS')

import numpy as np   
import pandas as pd
from tqdm import tqdm
import sys

from utils import *     # contains some useful helper functions 
from models import *    # toy models
from solvers import *   # matrix completion solvers
from methods import *



#########################
# Experiment parameters #
#########################

if True: # Input parameters
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 13:
        print("Error: incorrect number of parameters.")
        quit()

    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    r_true = int(sys.argv[3])
    r_guess = int(sys.argv[4])
    prob_obs = float(sys.argv[5])
    gamma_n = float(sys.argv[6])
    gamma_m = float(sys.argv[7])
    noise_model = sys.argv[8]
    a = int(sys.argv[9])
    b = int(sys.argv[10])
    mu = float(sys.argv[11])
    seed = int(sys.argv[12])


# Fixed data parameters
max_test_pairs = 1000         # Maximum number of test pairs 
matrix_generation_seed = 0    # Data matrix is fixed 
repetition = 2
verbose = True

methods = ["conformal", 
           "naive", 
           "bonferroni", 
           "uncorrected"]
model = "RFM"

# Other parameters
size_obs = int(n1 * n2 * prob_obs)
n_calib_pairs = int(size_obs//4)

allow_inf = False
alpha = 0.1



###############
# Output file #
###############
outdir = "./results/exp_residual_hpm/"
os.makedirs(outdir, exist_ok=True)
outfile_name = str(n1) + "by" + str(n2) + "_rtrue" + str(r_true) + "_rguess" + str(r_guess)
outfile_name += "_prob" + str(prob_obs) +  "_gn" + str(gamma_n) + "_gm" + str(gamma_m) + "_" + noise_model
outfile_name += "_a" + str(a) + "_b" + str(b) + "_mu" + str(mu) + "_seed" + str(seed)
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")
sys.stdout.flush()


# Header for results file
def add_header(df):
    df["n1"] = n1
    df["n2"] = n2
    df['alpha'] = alpha
    df['r_true'] = r_true
    df['r_guess'] = r_guess
    df['gamma_n'] = gamma_n
    df['gamma_m'] = gamma_m
    df['noise_model'] = noise_model
    df['a'] = a
    df['b'] = b
    df['mu'] = mu
    return df



#################
# Generate Data #
#################
if model == "RFM":
    mm = RandomFactorizationModel(n1 ,n2, r_true)
elif model == "ROM":
    mm = RandomOrthogonalModel(n1 ,n2, r_true)
else:
    mm = RandomFactorizationModel(n1 ,n2, r_true)

if verbose:
    print('Fixing the ground truth matrix generated from the {} model.\n'.format(model))
    sys.stdout.flush()

U, V, M_true = mm.sample_noiseless(matrix_generation_seed)



#####################
# Define Experiment #
#####################
def run_single_experiment(M_true, alpha, size_obs, n_calib_pairs, max_test_pairs, 
                          r_true, r_guess, gamma_n=0, gamma_m=0, noise_model='beta', a=1, b=1, mu=1,
                          random_state=0):
    res = pd.DataFrame({})


    #-------Generate masks----------#
    #-------------------------------#
    n1, n2 = M_true.shape
    sampler = PairSampling(n1,n2)
    mask_obs = sampler.sample_submask(size_obs, random_state=random_state)
    mask_test = np.ones_like(mask_obs) - mask_obs
    mask_train, idxs_calib, _, _ = sampler.sample_train_calib(mask_obs, n_calib_pairs,random_state=random_state)
    

    #-------Sample test pairs-------#
    #-------------------------------#
    n_test_pairs = min(int((np.sum(mask_test)-n1)//2), max_test_pairs)
    _, idxs_test, _, _  = sampler.sample_train_calib(mask_test, n_test_pairs, random_state=random_state)    
    
    if verbose:
        print("Training size:{}, calib size: {}, test size: {}\n".format(np.sum(mask_train), n_calib_pairs, n_test_pairs))
        sys.stdout.flush()

    #--------Generate noise---------#
    #-------------------------------#
    nm = NoiseModel(random_state)
    M = nm.get_noisy_matrix(M_true, gamma_n=gamma_n, gamma_m=gamma_m, model=noise_model, 
                            a=a, b=b, mu=mu, alpha=alpha, normalize=False)

    #--------Model Training---------#
    #-------------------------------#
    print("Running matrix completion algorithm on the splitted training set...")
    sys.stdout.flush()
    if solver == "pmf":
        Mhat, _, _ = pmf_solve(M, mask_train, k=r_guess, verbose=verbose, random_state=random_state)
    elif solver == "svt":
        Mhat = svt_solve(M, mask_train, verbose = verbose, random_state = random_state)
    print("Done training!\n")
    sys.stdout.flush()


    #------Conformal methods--------# 
    #-------------------------------#
    for method in methods:
        
        is_inf = np.zeros(n_test_pairs)
        if method == "conformal":
            ci_method = PairedCI_hpm(M, Mhat, mask_obs, idxs_calib)
            lower, upper, is_inf, _ = ci_method.get_CI(idxs_test, alpha, allow_inf)
        elif method == "naive":
            ci_method = PairedCI_hpm(M, Mhat, mask_obs, idxs_calib)
            lower, upper = ci_method.naive_CI(idxs_test, alpha)       
        elif method == "bonferroni":
            lower, upper = benchmark_CI(M, Mhat, idxs_calib, idxs_test, alpha)
        else:
            lower, upper = benchmark_CI(M, Mhat, idxs_calib, idxs_test, 2 * alpha)
        
            
        res = pd.concat([res, evaluate_pairedCI(lower, upper, M, idxs_test, is_inf=is_inf, method=method)])
    
    res['Calib_size'] = n_calib_pairs
    res['Train_size'] = np.sum(mask_train)
    res['Test_size'] = n_test_pairs
    res['random_state'] = random_state
    return res



#####################
#  Run Experiments  #
#####################
results = pd.DataFrame({})

for i in tqdm(range(repetition), desc="Repetitions", leave=True, position=0):
    random_state = repetition * seed + i

    res = run_single_experiment(M_true, alpha, size_obs, n_calib_pairs, max_test_pairs, 
                          r_true, r_guess, gamma_n=gamma_n, gamma_m=gamma_m, model=model, a=a, b=b, mu=mu,
                          random_state=random_state)
    
    results = pd.concat([results, res])

add_header(results)



#####################
#    Save Results   #
#####################
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()
