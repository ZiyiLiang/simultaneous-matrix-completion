import sys, os
sys.path.append('../smc')
sys.path.append('../third_party')

import numpy as np   
import pandas as pd
import pdb
import scipy.stats as stats

from tqdm import tqdm
from utils import *     # contains some useful helper functions 
from utils_data import *
from models import *    # toy models
from solvers import *   # matrix completion solvers
from methods import *
from wsc import *
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
    data_name = str(sys.argv[2])
    seed = int(sys.argv[3])
    
# Fixed data parameters
max_test_queries = 200            
max_calib_queries = 2000
matrix_generation_seed = 2024
max_iterations = 30

methods = ["conformal", 
           "benchmark"]
solver = "pmf"

# Other parameters
verbose = True
allow_inf = False
alpha = 0.1

k_list = np.arange(2,6)
repetition = 1



###############
#  Load data  #
###############
base_path = "../data/"
ds = DataSet(base_path, data_name, matrix_generation_seed)

if data_name == "movielens":
    n1, n2 = ds.shape 
    prop_train = 0.8
    ll, uu = 0, 5

M, mask_avail, mask_miss = ds.sample(n1, n2)


###############
# Output file #
###############
outdir = f"./results/exp_uniform_{data_name}/"
os.makedirs(outdir, exist_ok=True)
outfile_name = f"r{r}_seed{seed}"
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")
sys.stdout.flush()

# Header for results file
def add_header(df):
    df["n1"] = n1
    df["n2"] = n2
    df['alpha'] = alpha
    df['r'] = r
    return df



#####################
# Define Experiment #
#####################

def clip_intervals(lower, upper):
    lower[lower <= ll] = ll
    upper[upper >= uu] = uu
    return lower, upper

def run_single_experiment(M, k, alpha, prop_train, max_test_queries, max_calib_queries,
                          r, random_state=0):
    res = pd.DataFrame({})
    
    #-------Generate masks----------#
    #-------------------------------#
    sampler = QuerySampling(n1,n2)
    # Randomly split the observed set into test set and training set
    mask_obs, mask_test = sampler.sample_submask(mask=mask_avail, sub_size=prop_train, random_state=random_state)
    n_calib_queries = min(int(0.5 * np.sum(np.sum(mask_obs, axis=1) // k)), max_calib_queries)


    #------Sample test queries------#
    #-------------------------------#
    n_test_queries = min(int(0.99 * np.sum(np.sum(mask_test, axis=1) // k)), max_test_queries)
    _, idxs_test, _ = sampler.sample_train_calib(mask_test, k, calib_size=n_test_queries, random_state=random_state)  
    if verbose:
        print("Training size:{}, calib size: {}, test size: {}\n".format(np.sum(mask_obs)-n_calib_queries*k, n_calib_queries, n_test_queries))
        sys.stdout.flush()


    for method in methods:
        #------Split train calib--------#
        #-------------------------------#
        if method == "conformal":
            mask_train, idxs_calib, _ = sampler.sample_train_calib(mask_obs, k, 
                                        calib_size=n_calib_queries, random_state=random_state)
        else: 
            mask_train, idxs_calib, _ = sampler.sample_train_calib(mask_obs, 1, 
                                    calib_size=int(n_calib_queries * k), random_state=random_state)
    

        #--------Model Training---------#
        #-------------------------------#
        print("Running matrix completion algorithm on the splitted training set...")
        sys.stdout.flush()
        if solver == "pmf":
            Mhat, _, _ = pmf_solve(M, mask_train, k=r, max_iterations = max_iterations, verbose=verbose, random_state=random_state)
        elif solver == "svt":
            Mhat = svt_solve(M, mask_train, verbose = verbose, random_state = random_state)
        print("Done training!\n")
        sys.stdout.flush()

        print("Estimating missingness on the splitted training set...")
        w_obs=estimate_P(mask_train, prop_train, r=5)
        print("Done estimating!\n")
        sys.stdout.flush()
    
    
        #------Compute intervals--------# 
        #-------------------------------#
        if method == "conformal":
            ci_method = SimulCI(M, Mhat, mask_obs, idxs_calib, k, w_obs=w_obs)
            df = ci_method.get_CI(idxs_test, alpha, allow_inf=allow_inf)
            lower, upper, is_inf= df.loc[0].lower, df.loc[0].upper, df.loc[0].is_inf
            lower, upper = clip_intervals(lower, upper)
            res = pd.concat([res, evaluate_SCI(lower, upper, k, M, idxs_test, is_inf=is_inf, method=method)])
        else:
            a_list = [alpha, alpha * k]
            ci_method = Bonf_benchmark(M, Mhat, mask_obs, idxs_calib, k, w_obs=w_obs)
            df = ci_method.get_CI(idxs_test, a_list, allow_inf=allow_inf)
            for i, m in enumerate(["Bonferroni", "Uncorrected"]):
                lower, upper, is_inf= df.loc[i].lower, df.loc[i].upper, df.loc[i].is_inf
                lower, upper = clip_intervals(lower, upper)
                res = pd.concat([res, evaluate_SCI(lower, upper, k, M, idxs_test, is_inf=is_inf, method=m)])


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

        res = run_single_experiment(M, k, alpha, prop_train, max_test_queries, max_calib_queries,
                            r, random_state=random_state)
        
        results = pd.concat([results, res])

add_header(results)



#####################
#    Save Results   #
#####################
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()