import numpy as np   
import sys
import pdb

from tqdm import tqdm
from scipy.special import binom
from scipy.optimize import fsolve




class QuerySampling():
    """ 
    This class implements functions to sample the observed index set
    and sample the calibration queries.
    """
    def __init__(self, n1, n2):
        self.n1 = n1   # shape of the output matrix
        self.n2 = n2
        self.shape = (self.n1, self.n2)
        
    
    def sample_submask(self, sub_size, mask=None, w=None, random_state=0):
        rng = np.random.default_rng(random_state)
        
        if mask is None:
            mask = np.ones(self.shape, dtype=int)
        mask = mask.flatten(order='C')   
        obs_idx = np.where(mask == 1)[0]
        sub_size = int(np.clip(sub_size, 1,len(obs_idx)-1))

        if w is not None:
            w = w.flatten(order='C')     
        else:
            w = np.array([1] * (np.prod(self.shape))) / (np.prod(self.shape))
        
        # Make sure the weights sum up to 1 at selected indices
        w /= np.sum(w[obs_idx])
        sub_idx = rng.choice(obs_idx, sub_size, replace=False, p=w[obs_idx])

        sub_mask = np.zeros(np.prod(self.shape), dtype=int)
        sub_mask[sub_idx] = 1
        return sub_mask.reshape(self.shape), (mask-sub_mask).reshape(self.shape)
    

    def sample_train_calib(self, mask_obs, calib_size, k, random_state=0):
        rng = np.random.default_rng(random_state)

        k = int(k) 
        mask_train = np.zeros_like(mask_obs)
        mask_calib = np.zeros_like(mask_obs)
        n_obs = np.sum(mask_obs, axis=1)   # Check the observation# in each row

        n_queries = self._validate_calib_size(self, calib_size, n_obs, k)
        idxs_calib = (-np.ones(k * n_queries, dtype=int),\
                      -np.ones(k * n_queries, dtype=int))
        
        
        avail_idxs, num_idxs = {}, {}
        for i in range(self.n1):
            if n_obs[i] >= k:
                idxs = np.where(mask_obs[i]==1)[0]
                rng.shuffle(idxs)
                residual = int(n_obs[i] % k)
                if residual!= 0:
                    # If the row length is not a multiple of k, trim the indices set
                    idxs = idxs[:-residual]
                avail_idxs[i] = idxs
                num_idxs[i] = len(idxs)
        
        
        # Sample calibration queries
        for i in range(n_queries):
            rows, prob = list(num_idxs.keys()), list(num_idxs.values())
            prob /= np.sum(prob)
            row = rng.choice(rows, p=prob) 
            
            # First k shuffled entries form a calibration query
            idxs_calib[0][k*i : k*(i+1)] = row
            idxs_calib[1][k*i : k*(i+1)] = avail_idxs[row][:k]
            
            # Update the available indices by removing the selected query
            if len(avail_idxs[row]) == k:
                del avail_idxs[row]
                del num_idxs[row]
            else:
                avail_idxs[row] = avail_idxs[row][k:]
                num_idxs[row] -= k
                
        # Get the training mask
        mask_calib[idxs_calib] = 1
        mask_train = mask_obs - mask_calib
        return mask_train, idxs_calib, mask_calib
    

    def _validate_calib_size(self, calib_size, n_obs, k):
        """
        validation helper to check if the calibration size is meaningful
        """
        calib_size_type = np.asarray(calib_size).dtype.kind
        max_calib_num = int(np.sum(n_obs // k))

        if (
            calib_size_type == "i"
            and (calib_size >= max_calib_num or calib_size <= 0)
            or calib_size_type == "f"
            and (calib_size <= 0 or calib_size >= 1)
        ):
            raise ValueError(
                "calib_size={0} should be either positive and smaller"
                " than the maximum possible calibration number {1} or a float in the "
                "(0, 1) range".format(calib_size, max_calib_num)
            )
        
        if calib_size is not None and calib_size_type not in ("i", "f"):
            raise ValueError("Invalid value for calib_size: {}".format(calib_size))
        
        if calib_size_type == "f":
            n_calib = int(calib_size * max_calib_num)
        elif calib_size is None:
            n_calib = int(0.5 * max_calib_num)
        
        return n_calib



class SimulCI():
    """ 
    This class computes the simultaneous conformal prediction region for test query with length k
    """
    def __init__(self, M, Mhat, mask_obs, idxs_calib, k,
                w_obs=None, verbose=True, progress=True):
        self.n1, self.n2 = M.shape[0], M.shape[1]
        self.k = int(k)
        self.Mhat = Mhat
        self.verbose = verbose
        self.progress = progress
        self.mask_obs = mask_obs
        self.mask_miss = np.array(1-mask_obs)

        # If the observation sampling weights is not given, assume it is uniform.
        if w_obs is None:
            w_obs = np.ones_like(Mhat)
        self.w_obs = w_obs
        
        # Number of observations in each row
        self.n_obs = np.sum(mask_obs, axis=1)
        # Number of missing entries in each row
        self.n_miss = self.n2 - self.n_obs
        # Number of observations in each row after random dropping
        self.n_obs_drop = self.n_obs - self.n_obs % self.k
                
        # Note that calib indexs should be a tuple
        # idxs_calib[0] is an array of row indexes
        # idxs_calib[1] is an array of col indexes
        # k consecutive entries from a calibration query
        self.idxs_calib = idxs_calib
        self.n_calib_queries = len(idxs_calib[0])//self.k
        self.calib_scores = np.zeros(self.n_calib_queries)
        
        self.abs_err = np.abs(M - self.Mhat)
        
        # Compute the calibration scores
        self._get_calib_scores()
        self.calib_order = np.argsort(self.calib_scores)
        self.st_calib_scores = self.calib_scores[self.calib_order]
        
        # get the row index of each calibration query
        self.rows_calib = np.array([self.idxs_calib[0][k * i] for i in range(self.n_calib_queries)])


    def _get_calib_scores(self):
        """
        This function implements the scores as the maximum of the absolute prediction errors
        """
        for i in range(self.n_calib_queries):
            scores = self.abs_err[(self.idxs_calib[0][self.k*i: self.k*(i+1)], self.idxs_calib[1][self.k*i: self.k*(i+1)])]
            self.calib_scores[i] = np.max(scores) 


    def _compute_universal(self, w_test):
        # Compute the sum of weights for the pruned missing set over each row
        w_miss = np.multiply(w_test, self.mask_miss)
        sr_prune = np.sum(w_miss, axis=1)
        sr_prune[np.where(self.n_miss < self.k)[0]] = 0
        sw_prune =  np.sum(sr_prune)

        # Compute the sum of weights for the missing entries in each row
        sr_miss = np.sum(w_miss, axis=1)

        # Compute the sum of sampling weights on the missing set
        delta = np.sum(np.multiply(self.w_obs, self.mask_miss))

        # Compute the scaling parameter
        def _z(r,w,d):
            res = d - 1/r
            for i in range(len(w)):
                res -= w[i] / (2**(r*w[i]) - 1)
            return res
        scale = fsolve(_z, 1/delta, (self.w_obs[np.where(self.mask_obs==1)], delta))[0]

        return sr_prune, sw_prune, sr_miss, delta, scale
    

    def _weight_single(self, idx_test, row_test, w_test, 
                       sr_prune, sw_prune, sr_miss, delta, scale):
        weights = np.ones(self.n_calib_queries + 1)

        # Sum of test sampling weights for the test query
        sw_test = np.sum(w_test[idx_test])
        # Sum of observation sampling weights for the test query
        sw_obs_test = np.sum(self.w_obs[idx_test])
        # Augmented index set
        idxs_full = (np.concatenate([self.idxs_calib[0], idx_test[0]]),np.concatenate([self.idxs_calib[1], idx_test[1]]))
        rows_full = np.concatenate([self.rows_calib,[row_test]])

        # Compute the quantile inflation weights
        for i in range(self.n_calib_queries + 1):
            idx_i = (idxs_full[0][self.k*i : self.k*(i+1)], idxs_full[1][self.k*i : self.k*(i+1)])
            row_i = rows_full[i]
            csum_i = np.cumsum(w_test[idx_i])
            
            # Compute the prob of the ith query being the test query
            prob_test = w_test[idx_i[0][0], idx_i[1][0]]
            sw_diff = 0
            if row_i != row_test:
                if self.n_miss[row_i] < self.k:
                    sw_diff += (sr_miss[row_i] - csum_i[-1])
                if self.n_miss[row_test] < 2*self.k:
                    sw_diff -= (sr_miss[row_test] - sw_test)
            prob_test /= (sw_prune - sw_test + csum_i[-1] + sw_diff)

            for j in range(1,self.k):
                prob_test *= w_test[idx_i[0][j], idx_i[1][j]]
                sw_diff = 0
                if row_i == row_test:
                    sw_diff -= sw_test
                if self.n_miss[row_i] < self.k:
                    sw_diff += (sr_miss[row_i] - csum_i[-1])
                prob_test /= (sr_prune[row_i] + csum_i[-1] - csum_i[j-1] + sw_diff)

            # Compute the prob of sampling the given calibration queries
            prob_cal = 1
            if row_i != row_test:
                prob_cal *= binom(self.n_obs[row_i], self.n_obs_drop[row_i])
                prob_cal /= binom(self.n_obs[row_i]-self.k, self.n_obs_drop[row_i]-self.k)
                prob_cal *= binom(self.n_obs[row_test], self.n_obs_drop[row_test])        
                prob_cal /= binom(self.n_obs[row_test]+self.k, self.n_obs_drop[row_test]+self.k)

                for j in range(1, self.k):
                    prob_cal *= (self.n_obs_drop[row_i]-j)/(self.n_obs_drop[row_test]+self.k-j)
        
            # Compute the prob of sampling the given observation entries
            diff = np.sum(self.w_obs[idx_i]) - sw_obs_test
            prob_obs = 0.5**(scale * diff)*(delta + diff)/delta
            for j in range(self.k):
                prob_obs *= 1 - 0.5**(scale * self.w_obs[idx_test[0][j], idx_test[1][j]])
                prob_obs /= 1 - 0.5**(scale * self.w_obs[idx_i[0][j], idx_i[1][j]])
            
            weights[i] = prob_test * prob_cal * prob_obs
        
        weights /= np.sum(weights)
        return weights[:-1]
    

    def get_CI(self, idxs_test, alpha, w_test=None, allow_inf=True):
        if w_test is None:
            w_test = np.ones_like(self.Mhat)
        
        n_test_queries = len(idxs_test[0])//self.k
        weights_list = [[]]*n_test_queries
        
        upper = np.zeros(n_test_queries * self.k)    # upper confidence bound
        lower = np.zeros(n_test_queries * self.k)    # lower confidence bound
        is_inf = np.zeros(n_test_queries) # queries for which interval length is infinity   
        
        if self.verbose:
            print("Computing conformal prediction intervals for {} test queries...".format(n_test_queries))
            sys.stdout.flush()
                
        for i in tqdm(range(n_test_queries), desc="CI", leave=True, position=0, 
                      disable = not self.progress):
            idx_test = (idxs_test[0][self.k*i : self.k*(i+1)], idxs_test[1][self.k*i : self.k*(i+1)])
            row_test = idx_test[0][0]

            # Compute constant components in weights that are invariant to test query
            sr_prune, sw_prune, sr_miss, delta, scale = self._compute_universal(w_test)
            weights = self._weight_single(idx_test, row_test, w_test, sr_prune, sw_prune, sr_miss, delta, scale)

            weights_list[i] = weights
            cweights = np.cumsum(weights[self.calib_order])
            est = np.array(self.Mhat[idx_test])
                        
            if cweights[-1] < 1-alpha:
                is_inf[i] = 1
                if allow_inf:
                    lower[self.k*i : self.k*(i+1)] = -np.inf
                    upper[self.k*i : self.k*(i+1)] = np.inf
                else:
                    lower[self.k*i : self.k*(i+1)] = est - self.st_calib_scores[-1]
                    upper[self.k*i : self.k*(i+1)] = est + self.st_calib_scores[-1]
            else:
                idx = np.argmax(cweights >= 1-alpha)
                lower[self.k*i : self.k*(i+1)] = est - self.st_calib_scores[idx]
                upper[self.k*i : self.k*(i+1)] = est + self.st_calib_scores[idx]
        
        if self.verbose:
            print("Done!")
            sys.stdout.flush()
            
        #--- [DEBUG]for deubugging purpose, also return the weights ---#
        return lower, upper, is_inf, weights_list
    

    def naive_CI(self, idxs_test, alpha, allow_inf=True):
        n_test_queries = len(idxs_test[0])//self.k
        if self.verbose:
            print("Computing naive prediction intervals for {} test queries...".format(n_test_queries))
            sys.stdout.flush()
        
        level_corrected = (1 - alpha) * (1 + 1/self.n_calib_queries)
        est = np.array(self.Mhat[idxs_test])
        
        if level_corrected >= 1:
            is_inf = np.ones(n_test_queries)
            if allow_inf:
                lower = np.repeat(-np.inf, len(idxs_test[0]))
                upper = np.repeat(np.inf, len(idxs_test[0]))
            else:
                lower = est - self.st_calib_scores[-1]
                upper = est + self.st_calib_scores[-1]
        else:
            is_inf = np.zeros(n_test_queries)
            qnt = np.quantile(self.calib_scores, level_corrected, method="higher")
            lower = est - qnt
            upper = est + qnt

        if self.verbose:
            print("Done!")
            sys.stdout.flush()
        return lower, upper, is_inf
    


class Bonf_benchmark():
    """ 
    This class computes the Bonferrroni-style simultaneous confidence region
    """
    def __init__(self, M, Mhat, mask_obs, idxs_calib, k,
                w_obs=None, verbose=True, progress=True):
        self.k = int(k)
        self.verbose = verbose
        self.progress = progress
        
        # Apply simultaneous conformal inference method with k=1
        self.sci = SimulCI(M, Mhat, mask_obs, idxs_calib, 1,
                           w_obs=w_obs, verbose=False, progress=self.progress)
        
    
    def get_CI(self, idxs_test, alpha, w_test=None,  allow_inf=True):
        if w_test is None:
            w_test = np.ones_like(self.Mhat)

        n_test_entries = len(idxs_test[0])
        n_test_queries = n_test_entries//self.k
        alpha_corrected = alpha/self.k

        if self.verbose:
            print("Computing Bonferroni-style intervals for {} test queries...".format(n_test_queries))
            sys.stdout.flush()
                
        lower, upper, is_inf, _ = self.sci.get_CI(self, idxs_test, alpha_corrected, w_test, allow_inf)
        return lower, upper, is_inf
