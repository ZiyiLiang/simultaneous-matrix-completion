import numpy as np   
from tqdm import tqdm
import sys


class PairedCI_hpm():
    """ 
    Conformalized CI for pairs generated with the homogeneous pairs model.
    """
    def __init__(self, M, Mhat, mask_obs, idxs_calib, 
                 verbose=True, progress=True):
        self.n1, self.n2 = M.shape[0], M.shape[1]
        self.Mhat = Mhat
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
        self.n_calib_pairs = len(idxs_calib[0])//2
        self.calib_scores = np.zeros(self.n_calib_pairs)
        
        self.residuals = np.abs(M - self.Mhat)
        
        # Compute the calibration scores
        self._get_calib_scores()
        self.calib_order = np.argsort(self.calib_scores)
        self.st_calib_scores = self.calib_scores[self.calib_order]
        
        # get the user index of each calibration pair
        self.users_calib = np.array([self.idxs_calib[0][2 * i] for i in range(self.n_calib_pairs)])
        
#         # Compute the universal components shared for all weights
#         self.users_calib, self.calib_pair_cnt = self._universal_vals() 

    
    # This function implements the scores as the absolute residuals
    def _get_calib_scores(self):
        for i in range(self.n_calib_pairs):
            score1 = self.residuals[self.idxs_calib[0][2 * i]][self.idxs_calib[1][2 * i]]
            score2 = self.residuals[self.idxs_calib[0][2 * i + 1]][self.idxs_calib[1][2 * i + 1]]
            self.calib_scores[i] = np.max([score1, score2])
    
    
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
                prob_calib = (self.n_obs_drop[user_calib]-1) / (self.n_obs_drop[user_test]+1)
            weights[i] *= prob_test * prob_drop * prob_calib
        
        # Compute the test weight
        weights[-1] *= 1 / (self.n2 - self.n_obs[user_test] - 1)
        
        weights /= np.sum(weights)
        
        return weights[:-1]
            

    def get_CI(self, idxs_test, alpha, allow_inf=True):
        n_test_pairs = len(idxs_test[0])//2
        weights_list = [[]]*n_test_pairs
        
        upper = np.zeros(n_test_pairs * 2)    # upper confidence bound
        lower = np.zeros(n_test_pairs * 2)    # lower confidence bound
        is_inf = np.zeros(n_test_pairs) # positions where interval length is infinity   
        
        if self.verbose:
            print("Computing conformal prediction intervals for {} test pairs...".format(n_test_pairs))
            sys.stdout.flush()
                
        for i in tqdm(range(n_test_pairs), desc="CI", leave=True, position=0, 
                      disable = not self.progress):
            idx_test = (idxs_test[0][2*i : 2*(i+1)], idxs_test[1][2*i : 2*(i+1)])
            user_test = idx_test[0][0]
      
            weights = self._weight_single(user_test, self.users_calib)
            weights_list[i] = weights
            cweights = np.cumsum(weights[self.calib_order])
            
            qnt = np.quantile(self.calib_scores, 1 - alpha, method="lower")
            est = self.Mhat[idx_test]
                        
            # Note that absolute residual scores lead to PI pair with the same length
            if cweights[-1] < 1-alpha:
                is_inf[i] = 1
                if allow_inf:
                    lower[2*i], lower[2*i + 1] = -np.inf, -np.inf
                    upper[2*i], upper[2*i + 1] = np.inf, np.inf
                else:
                    lower[2*i], lower[2*i + 1] = est[0] - self.st_calib_scores[-1], est[1] - self.st_calib_scores[-1]
                    upper[2*i], upper[2*i + 1] = est[0] + self.st_calib_scores[-1], est[1] + self.st_calib_scores[-1]
            else:
                idx = np.argmax(cweights >= 1-alpha)
                lower[2*i], lower[2*i + 1] = est[0] - self.st_calib_scores[idx], est[1] - self.st_calib_scores[idx]
                upper[2*i], upper[2*i + 1] = est[0] + self.st_calib_scores[idx], est[1] + self.st_calib_scores[idx]
        
        if self.verbose:
            print("Done!")
            sys.stdout.flush()
            
        #--- [DEBUG]for deubugging purpose, also return the weights ---#
        # return the confidence interval in the matrix format
        return lower, upper, is_inf, weights_list
        
    def naive_CI(self, idxs_test, alpha):
        n_test_pairs = len(idxs_test[0])//2
        if self.verbose:
            print("Computing naive prediction intervals for {} test pairs...".format(n_test_pairs))
            sys.stdout.flush()

        est = self.Mhat[idxs_test]
        qnt = np.quantile(self.calib_scores, 1 - alpha, method="lower")
        lower, upper = est - qnt, est + qnt   

        if self.verbose:
            print("Done!")
            sys.stdout.flush()
        return lower, upper



def benchmark_CI(M, Mhat, idxs_calib, idxs_test, alpha, verbose = True):
    n_test_pairs = len(idxs_test[0])//2
    alpha_corrected = alpha / 2
    
    if verbose:
        print("Computing prediction intervals with Bonferroni correction with alpha {} for {} test pairs...".format(alpha, n_test_pairs))
        sys.stdout.flush()

    residuals = np.abs(M - Mhat)
    residual_calib = residuals[idxs_calib]

    est = Mhat[idxs_test]
    qnt = np.quantile(residual_calib, 1 - alpha_corrected, method="lower")
    lower, upper = est - qnt, est + qnt   
    
    if verbose:
        print("Done!")
        sys.stdout.flush()
        
    return lower, upper