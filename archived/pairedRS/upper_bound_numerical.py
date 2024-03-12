import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *     # contains some useful helper functions
from models import *    # toy models
from solvers import *   # matrix completion solvers
from methods import *
from upper_bound_class import *





def run_single_experiment_max_weight(M_true, alpha, size_obs, n_calib_pairs, max_test_pairs,
                          r_true, r_guess, gamma_n=0, gamma_m=0, noise_model='beta', a=1, b=1, mu=1,
                          random_state=0):
    res = pd.DataFrame({})

    # -------Generate masks----------#
    # -------------------------------#
    n1, n2 = M_true.shape
    sampler = PairSampling(n1, n2)
    mask_obs = sampler.sample_submask(size_obs, random_state=random_state)
    mask_test = np.ones_like(mask_obs) - mask_obs
    mask_train, idxs_calib, _, _ = sampler.sample_train_calib(mask_obs, n_calib_pairs, random_state=random_state)

    # -------Sample test pairs-------#
    # -------------------------------#
    # control the test pairs that we care about in each sampling
    # n_test_pairs = min(int((np.sum(mask_test) - n1) // 2), max_test_pairs)
    n_test_pairs = 1
    _, idxs_test, _, _ = sampler.sample_train_calib(mask_test, n_test_pairs, random_state=random_state) #use the same function to sample n test pairs

    # # --------Generate noise---------#
    # # -------------------------------#
    # nm = NoiseModel(random_state)
    # M = nm.get_noisy_matrix(M_true, gamma_n=gamma_n, gamma_m=gamma_m, model=noise_model,
    #                         a=a, b=b, mu=mu, alpha=alpha, normalize=False)

    # # --------Model Training---------#
    # # -------------------------------#
    # print("Running matrix completion algorithm on the splitted training set...")
    # sys.stdout.flush()
    # if solver == "pmf":
    #     Mhat, _, _ = pmf_solve(M, mask_train, k=r_guess, verbose=verbose, random_state=random_state)
    # elif solver == "svt":
    #     Mhat = svt_solve(M, mask_train, verbose=verbose, random_state=random_state)
    # print("Done training!\n")
    # sys.stdout.flush()

    #-------Get the maximum of the weights-------#
    weights_computation = weights_hpm(n1, n2, mask_obs, idxs_calib)
    weights_list = weights_computation.get_weights(idxs_test, alpha)
    max_weights = []
    for weights in weights_list:
        max_weight = max(weights)
        max_weights.append(max_weight)

    res['max_weight']=max_weights
    return res





