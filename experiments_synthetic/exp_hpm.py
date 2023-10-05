import sys, os
sys.path.append('../pairedRS')

import numpy as np   
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
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
    if len(sys.argv) != 10:
        print("Error: incorrect number of parameters.")
        quit()

    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    r_true = int(sys.argv[3])
    r_guess = int(sys.argv[4])
    prob_obs = int(sys.argv[5])
    method = sys.argv[6]
    solver = sys.argv[7]
    model = sys.argv[8]
    seed = int(sys.argv[9])


# Fixed data parameters
n_test = 1000
matrix_generation_seed = 0    # Data matrix is fixed 

# Other parameters
allow_inf = True
alpha = 0.1



###############
# Output file #
###############
outdir = "results/exp_hpm/"
os.makedirs(outdir, exist_ok=True)
outfile_name = str(n1) + "by" + str(n2) + "_rtrue" + str(r_true) + "_rguess" + str(r_guess)
outfile_name += "_prob" + str(prob_obs) + "_" + str(method) + "_" + str(solver)
outfile_name += "_" + str(model) + "_seed" + str(seed)
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")


# Header for results file
def add_header(df):
    df["dimension"] = (n1, n2)
    df["r_true"] = r_true
    df["r_guess"] = r_guess
    df["prob_obs"] = prob_obs
    df["method"] = method
    df["model"] = model
    df["solver"] = solver
    df["seed"] = seed
    return df



#################
# Generate Data #
#################
if model == "RFM":
    model = RandomFactorizationModel(n1 ,n2, r_true)
else:
    model = RandomFactorizationModel(n1 ,n2, r_true)

print('Fixing the ground truth matrix generated from the {} model.\n'.format(model))
sys.stdout.flush()
U, V, M = model.sample_noiseless(0)
