import numpy as np
from tqdm import tqdm
import sys

class weights_hpm():
    """
    Conformalized CI for pairs generated with the homogeneous pairs model.
    """

    def __init__(self, mask_obs, idxs_calib,
                 verbose=True, progress=True):
        self.n1, self.n2 = M.shape[0], M.shape[1]
        self.verbose = verbose
        self.progress = progress
        self.mask_obs = mask_obs

        # Number of observations for each user
        self.n_obs = np.sum(mask_obs, axis=1)
        # Number of observations for each user after random dropping
        self.n_obs_drop = self.n_obs - self.n_obs % 2

        # Note that calib indexs should be a tuple
        # idxs_calib[0] is an array of user indexes
        # idxs_calib[1] is an array of movie indexes
        # Two consecutive locations form a pair
        self.idxs_calib = idxs_calib
        self.n_calib_pairs = len(idxs_calib[0]) // 2
        # self.calib_scores = np.zeros(self.n_calib_pairs)

        # # Compute the calibration scores
        # self._get_calib_scores()
        # self.calib_order = np.argsort(self.calib_scores)
        # self.st_calib_scores = self.calib_scores[self.calib_order]

        # get the user index of each calibration pair
        self.users_calib = np.array([self.idxs_calib[0][2 * i] for i in range(self.n_calib_pairs)])

    #         # Compute the universal components shared for all weights
    #         self.users_calib, self.calib_pair_cnt = self._universal_vals()


    #     def _universal_vals(self):
    #         # Users in the calibration pairs
    #         users_calib = np.zeros(self.n_calib_pairs, dtype = int)
    #         # Number of calibration paris for each user
    #         calib_pair_cnt = np.zeros(self.n1, dtype = int)

    #         for i in range(self.n_calib_pairs):
    #             user = self.idxs_calib[0][2 * i]
    #             # get the user index of each calibration pair
    #             users_calib[i] = user
    #             #calib_pair_cnt[user] += 1

    #         return users_calib, calib_pair_cnt

    def _weight_single(self, user_test, users_calib):
        # Initial weights
        weights = np.ones(self.n_calib_pairs + 1)

        # Compute the calibration weights
        for i in range(self.n_calib_pairs):
            user_calib = users_calib[i]
            if user_calib == user_test:
                prob_test = 1 / (self.n2 - self.n_obs[user_calib] - 1)
                prob_drop = 1
                prob_calib = 1
            else:
                prob_test = 1 / (self.n2 - self.n_obs[user_calib] - 3)
                prob_drop = (1 + 2 / (self.n_obs[user_calib] - 2)) ** (self.n_obs[user_calib] % 2 == 1)
                prob_drop *= (1 - 2 / (self.n_obs[user_test] + 2)) ** (self.n_obs[user_test] % 2 == 1)
                prob_calib = (self.n_obs_drop[user_calib] - 1) / (self.n_obs_drop[user_test] + 1)
            weights[i] *= prob_test * prob_drop * prob_calib

        # Compute the test weight
        weights[-1] *= 1 / (self.n2 - self.n_obs[user_test] - 1)

        weights /= np.sum(weights)

        return weights