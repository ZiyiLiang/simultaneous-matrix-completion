import sys, os
sys.path.append('../')

import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm

from time import time
from largescale_smc.largescale_missingness import *
from largescale_smc.largescale_utils import *
from largescale_smc.largescale_method import *

from surprise import SVD
from surprise import accuracy

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

    max_calib_queries = int(sys.argv[1])
    k = int(sys.argv[2])
    seed = int(sys.argv[3])
    
# Fixed data parameters
max_test_queries = 2000  

# Other parameters
max_iterations = 30   
n_factors = 100       
#max_iterations = 2   
#n_factors = 5 
num_worker = 6
verbose = True
allow_inf = False
alpha = 0.1
repetition = 1


###############
#  Load data  #
###############
file_path = os.path.expanduser("../data/ml-10m/ratings_no_timestamp.dat")
reader = DataReader(line_format="user item rating", sep="::")
splitter = DataSplitter(file_path, reader)
ll, uu = 1, 5


###############
# Output file #
###############i:
outdir = f"./results/largescale/"
os.makedirs(outdir, exist_ok=True)
outfile_name = f"ncal{max_calib_queries}_k{k}_seed{seed}"
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")
sys.stdout.flush()

# Header for results file
def add_header(df):
    df['alpha'] = alpha
    df['max_n_cal'] = max_calib_queries
    df['k'] = k
    return df



#####################
# Define Experiment #
#####################

def clip_intervals(lower, upper):
    lower[lower <= ll] = ll
    upper[upper >= uu] = uu
    return lower, upper

def run_single_experiment(k, alpha, max_test_queries, max_calib_queries, random_state=0):
    res = pd.DataFrame({})
    
    #------Split train calib--------#
    #-------------------------------#
    start_time = time()
    train_samples, calib_samples, _ = splitter.sample_train_calib(k=k, max_n_calib=max_calib_queries, random_state=random_state)
    t_sample_calib = time()-start_time

    #------Sample test queries------#
    #-------------------------------#
    test_idxs =  splitter.sample_test(k, max_n_test=max_test_queries, random_state=0)

    n_train = len(train_samples)
    n_calib_queries = len(calib_samples) // int(k)
    n_test_queries = len(test_idxs) // int(k)
    if verbose:
        print("Training size:{}, calib size: {}, test size: {}\n".format(n_train, n_calib_queries, n_test_queries))
        sys.stdout.flush()
    

    #--------Model Training---------#
    #-------------------------------#
    print("Running matrix completion algorithm on the splitted training set...")
    sys.stdout.flush()

    start_time = time()
    trainset = construct_trainset(train_samples, reader.rating_scale)
    algo = SVD(n_factors=n_factors, n_epochs=max_iterations, biased=False, random_state=random_state, verbose=verbose)
    algo.fit(trainset)
    Mhat = create_batch_rating_predictor(algo)
    t_train = time()-start_time

    print(f"Done training! Took {t_train}s.\n")
    sys.stdout.flush()
    del trainset

    print("Estimating missingness on the splitted training set...")
    sys.stdout.flush()

    start_time = time()
    w_obs=LogisticMFProbs(factors=n_factors, iterations=max_iterations, random_state=random_state)
    w_obs.fit(train_samples+calib_samples)
    t_missing = time()-start_time

    print(f"Done missingness estimation! Took {t_missing}s.\n")
    sys.stdout.flush()
    
    #------Compute intervals--------# 
    #-------------------------------#
    start_time = time()
    ci_method = SimulCI_ls(splitter.config, calib_samples, k, Mhat, w_obs, num_workers=num_worker)
    t_init = time()-start_time

    start_time = time()
    df = ci_method.get_CI(test_idxs, alpha, allow_inf=allow_inf)
    t_inference = time()-start_time
    t_universal = ci_method._t_universal

    # Evaluate the intervals
    lower, upper, is_inf, is_impossible = df.loc[0].lower, df.loc[0].upper, df.loc[0].is_inf, df.loc[0].is_impossible
    lower, upper = clip_intervals(lower, upper)
    res = evaluate_SCI_ls(lower, upper, k, np.zeros(len(test_idxs)), is_inf, is_impossible)

    # free memory
    del ci_method, lower, upper, is_inf, is_impossible

    res['k'] = k     
    res['Calib_queries'] = n_calib_queries
    res['Train_entries'] = n_train
    res['Test_queries'] = n_test_queries
    res['random_state'] = random_state

    res['t_train'] = t_train
    res['t_missing'] = t_missing
    res['t_sample_calib'] = t_sample_calib
    res['t_init'] = t_init
    res['t_inference'] = t_inference
    res['t_universal'] = t_universal
    return res



#####################
#  Run Experiments  #
#####################
results = pd.DataFrame({})

for i in tqdm(range(1, repetition+1), desc="Repetitions", leave=True, position=0):
    random_state = repetition * (seed-1) + i

    res = run_single_experiment(k, alpha, max_test_queries, max_calib_queries, random_state=random_state)
    
    results = pd.concat([results, res])

add_header(results)



#####################
#    Save Results   #
#####################
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()
