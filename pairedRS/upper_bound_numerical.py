import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_expected_max_weight(M, Mhat, mask_obs, idxs_calib, idxs_test, n_repetitions, alpha):
    expected_max_weights = []

    for _ in range(n_repetitions):
        # Initialize the PairedCI_hpm class
        ci_method = PairedCI_hpm(M, Mhat, mask_obs, idxs_calib)

        # Calculate the weights
        _, _, _, weights_list = ci_method.get_CI(idxs_test, alpha, allow_inf=True)

        # Compute the maximum weight for each test set
        max_weight = max([max(weights) for weights in weights_list])
        expected_max_weights.append(max_weight)

    # Calculate the expected value of the maximum weights
    expected_value = np.mean(expected_max_weights)
    return expected_value

def compute_max_weight_single_test_set(M, Mhat, mask_obs, idxs_calib, idxs_test, alpha):
    # Initialize the PairedCI_hpm class
    ci_method = PairedCI_hpm(M, Mhat, mask_obs, idxs_calib)

    # Calculate the weights
    _, _, _, weights_list = ci_method.get_CI(idxs_test, alpha, allow_inf=True)

    # Compute the maximum weight for the given test set
    max_weight = max([max(weights) for weights in weights_list])

    return max_weight
