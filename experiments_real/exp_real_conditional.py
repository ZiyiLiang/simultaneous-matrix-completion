import sys, os
sys.path.append('../smc')
sys.path.append('../third_party')

import numpy as np   
import pandas as pd
import pdb
import scipy.stats as stats
from tqdm import tqdm
import sys

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

    scale = float(sys.argv[1])
    cal = int(sys.argv[2])
    seed = int(sys.argv[3])
    
# Fixed data parameters
max_calib_queries = cal
# If seed is None, choose the rows and cols to minimize missingness
matrix_generation_seed = 2024
max_iterations = 10

methods = ["conditional", 
           "unconditional"]
solver = "pmf"
r_solver = 3

# Other parameters
verbose = True
allow_inf = False
alpha = 0.1
repetition = 2


###############
#  Load data  #
###############
base_path = "../data/ml-100k"

num_columns, num_rows = 800, 1000
prop_train = 0.8
max_test_queries = 100
ll, uu = 1, 5
k_list = np.arange(1,6)

# Load rating matrix 
loader = Load_MovieLens(base_path)
M, mask_avail, _ = loader.load_data(replace_nan=-1, num_columns=num_columns, num_rows=num_rows, 
                                    random_state=matrix_generation_seed)
n1,n2 = M.shape

# Load additional demographic and item information
demo = loader.load_demographics()
genre = loader.load_movie_info()
bias=Movielens_weights(demo,genre)
w_test = bias.demo_weights(var='age', scale=scale)
del loader, demo, genre, bias


###############
# Output file #
###############
outdir = f"./results/exp_movielens_conditional/"
os.makedirs(outdir, exist_ok=True)
outfile_name = f"scale{scale:.2f}_cal{max_calib_queries}_seed{seed}"
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")
sys.stdout.flush()

# Header for results file
def add_header(df):
    df["n1"] = n1
    df["n2"] = n2
    df['scale'] = scale
    df['alpha'] = alpha
    df['r_solver'] = r_solver
    df['prop_test'] = 1-prop_train
    return df



#####################
# Define Experiment #
#####################

def clip_intervals(lower, upper):
    lower[lower <= ll] = ll
    upper[upper >= uu] = uu
    return lower, upper


def run_single_experiment(M, k, alpha, prop_train, w_test, max_test_queries, max_calib_queries,
                          random_state=0):
    res = pd.DataFrame({})
    
    #-------Generate masks----------#
    #-------------------------------#
    sampler = QuerySampling(n1,n2)
    # Randomly split the observed set into test set and training set
    mask_obs, mask_test = sampler.sample_submask(mask=mask_avail, sub_size=prop_train, random_state=random_state)


    #---------Calib queries---------#
    #-------------------------------#
    n_calib_queries = min(int(0.5 * np.sum(np.sum(mask_obs, axis=1) // k)), max_calib_queries)
    mask_train, idxs_calib, _ = sampler.sample_train_calib(mask_obs, k, 
                                            calib_size=n_calib_queries, random_state=random_state)


    #------Sample test queries------#
    #-------------------------------#
    n_test_queries = min(int(0.99 * np.sum(np.sum(mask_test, axis=1) // k)), max_test_queries)
    idxs_test= sampler.sample_test(mask_test, k, test_size=n_test_queries, w=w_test, replace=True, random_state=random_state)  
    del mask_test
    
    n_train = np.sum(mask_obs)-n_calib_queries*k
    if verbose:
        print("Training size:{}, calib size: {}, test size: {}\n".format(n_train, n_calib_queries, n_test_queries))
        sys.stdout.flush()


    #--------Model Training---------#
    #-------------------------------#
    print("Running matrix completion algorithm on the splitted training set...")
    sys.stdout.flush()
    if solver == "pmf":
        Mhat, _, _ = pmf_solve(M, mask_train, k=r_solver, max_iteration = max_iterations, verbose=verbose, random_state=random_state)
    elif solver == "svt":
        Mhat = svt_solve(M, mask_train, max_iteration = max_iterations, verbose = verbose, random_state = random_state)
    del mask_train
    print("Done training!\n")
    sys.stdout.flush()

    print("Estimating missingness on the splitted training set...")
    w_obs=estimate_P(mask_avail, 1, r=5)
    print("Done estimating!\n")
    sys.stdout.flush()

    
    
    #------Compute intervals--------# 
    #-------------------------------#
    ci_method = SimulCI(M, Mhat, mask_obs, idxs_calib, k, w_obs=w_obs)
    for method in methods:
        w_method = w_test if method == "conditional" else None
        df = ci_method.get_CI(idxs_test, alpha, w_test= w_method, allow_inf=allow_inf)
        lower, upper, is_inf= df.loc[0].lower, df.loc[0].upper, df.loc[0].is_inf
        lower, upper = clip_intervals(lower, upper)
        res = pd.concat([res, evaluate_SCI(lower, upper, k, M, idxs_test, is_inf=is_inf, method=method)])

    # free memory
    del ci_method, lower, upper, is_inf

    res['k'] = k     
    res['Calib_queries'] = n_calib_queries
    res['Train_entries'] = n_train
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

        res = run_single_experiment(M, k, alpha, prop_train, w_test, max_test_queries, max_calib_queries,
                                    random_state=random_state)
        
        results = pd.concat([results, res])

add_header(results)



#####################
#    Save Results   #
#####################
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()