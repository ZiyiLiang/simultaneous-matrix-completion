import numpy as np   
import pandas as pd
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
        
    
    def sample_submask(self, sub_size=None, mask=None, w=None, random_state=0):
        """
        Sample a subset from any given mask without replacement, sampling weights are optional.
        
        Parameters
        ----------
        sub_size: int or float, optional
            Size of the submask, either a float in (0,1) or a positive integer smaller than 
            size of the parent mask. Default is 0.5.
        mask : ndarray, optional
            The parent mask used for sampling should be an array consisting of 0s and 1s. 
            Default is all 1s.
        w : ndarray, optional
            non-negative sampling weights, dimension should match the
            shape of mask. Default is uniform weights.

        Returns
        -------
        Splits the parent mask into two ndarrays, the sampled submask and its complement.
        """

        rng = np.random.default_rng(random_state)
        
        if mask is None:
            mask = np.ones(self.shape, dtype=int)
        mask = mask.flatten(order='C')   
        obs_idx = np.where(mask == 1)[0]
        n_sub = self._validate_sub_size(sub_size, len(obs_idx))
        
        if w is not None:
            w = w.flatten(order='C')     
        else:
            w = np.array([1] * (np.prod(self.shape))) / (np.prod(self.shape))
        
        # Make sure the weights sum up to 1 at selected indices
        w /= np.sum(w[obs_idx])
        sub_idx = rng.choice(obs_idx, n_sub, replace=False, p=w[obs_idx])

        sub_mask = np.zeros(np.prod(self.shape), dtype=int)
        sub_mask[sub_idx] = 1
        return sub_mask.reshape(self.shape), (mask-sub_mask).reshape(self.shape)
    
    
    def _validate_sub_size(self, sub_size, n):
        """
        validation helper to check if the size of  submask is meaningful
        """
        sub_size_type = np.asarray(sub_size).dtype.kind

        if (
            sub_size_type == "i"
            and (sub_size >= n or sub_size <= 0)
            or sub_size_type == "f"
            and (sub_size <= 0 or sub_size >= 1)
        ):
            raise ValueError(
                "sub_size={0} should be either positive and smaller"
                " than the maximum possible sample number {1} or a float in the "
                "(0, 1) range".format(sub_size, n)
            )
        
        if sub_size is not None and sub_size_type not in ("i", "f"):
            raise ValueError("Invalid value for sub_size: {}".format(sub_size))
        
        if sub_size_type == "f":
            n_sub = int(sub_size * n)
        elif sub_size is None:
            n_sub = int(0.5 * n)
        
        return n_sub


    def sample_train_calib(self, mask_obs, k, calib_size=None, max_n_calib=None, random_state=0):
        """
        Split the observed indices into training set and calibration queries.

        Parameters
        ----------
        mask_obs : ndarray
            The parent mask used for sampling should be an array consisting of 0s and 1s. 
        k : int
            Positive integer smaller than number of cols
        calib_size : int or float, optional
            Size of the submask, either a float in (0,1) or a positive integer smaller than 
            size of the parent mask. Default is 0.5.
        max_n_calib : int, optional
            Optional positive integer, if provided, number of calibration queries will be capped
            at this value.

        Returns
        -------
        mask_train : ndarray
            array of 0s and 1s, with 1 indicating the index is in the training set
        idxs_calib : tuple of arrays
            idxs_calib[0] is an array of row indexes
            idxs_calib[1] is an array of col indexes
            k consecutive entries form a calibration query
        mask_calib : ndarray
            array of 0s and 1s, with 1 indicating the index is in the calibration set
        """
        rng = np.random.default_rng(random_state)

        k = int(k) 
        mask_train = np.zeros_like(mask_obs)
        mask_calib = np.zeros_like(mask_obs)
        n_obs = np.sum(mask_obs, axis=1)   # Check the observation# in each row

        n_queries = self._validate_calib_size(calib_size, n_obs, k, max_n_calib)
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
    

    def _validate_calib_size(self, calib_size, n_obs, k, max_n_calib):
        """
        validation helper to check if the calibration size is meaningful
        """
        calib_size_type = np.asarray(calib_size).dtype.kind
        max_calib_num = int(np.sum(n_obs // k))

        if max_n_calib is not None:
            max_n_calib = int(np.clip(max_n_calib, 1, max_calib_num))
        else:
            max_n_calib = max_calib_num

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
        elif calib_size_type == "i":
            n_calib = calib_size
        elif calib_size is None:
            n_calib = int(0.5 * max_calib_num)
        
        return np.min([n_calib, max_n_calib])



class SimulCI():
    """ 
    This class computes the simultaneous conformal prediction region for test query with length k

    Attributes
    ----------
    M : 2d array
        A partially observed matrix.
    Mhat : 2d array
        An estimation of M.
    mask_obs : 2d array
        Array of 0s and 1s, where 1 indicates the index is observed.
    idxs_calib: tuple of arrays
        idxs_calib[0] is an array of row indexes
        idxs_calib[1] is an array of col indexes
        k consecutive entries form a calibration query
    k : int
        query size 
    w_obs : ndarray, optional
        non-negative sampling weights for the observed entreis, dimension should match the
        shape of the input matrix. Default is uniform weights.
    verbose : bool, optional
        If True, messages will be printed.
    progress : bool, optional
        If True, progress bars will be printed.
    """
    def __init__(self, M, Mhat, mask_obs, idxs_calib, k,
                w_obs=None, verbose=True, progress=True):
        self.k = int(k)
        self.Mhat = np.array(Mhat)
        self.n1, self.n2 = self.Mhat.shape[0], self.Mhat.shape[1]
        self.verbose = verbose
        self.progress = progress
        self.mask_obs = np.array(mask_obs)
        self.mask_miss = np.array(1-mask_obs)
        self.w_obs = self._validate_w(w_obs)
        
        # Number of observations in each row
        self.n_obs = np.sum(mask_obs, axis=1)
        # Number of missing entries in each row
        self.n_miss = self.n2 - self.n_obs
        # Number of observations in each row after random dropping
        self.n_obs_drop = self.n_obs - self.n_obs % self.k
                
        self.idxs_calib = idxs_calib
        self.n_calib_queries = len(idxs_calib[0])//self.k
        self.calib_scores = np.zeros(self.n_calib_queries)
                
        # Compute the calibration scores
        self._get_calib_scores(np.abs(M - self.Mhat))
        self.calib_order = np.argsort(self.calib_scores)
        self.st_calib_scores = self.calib_scores[self.calib_order]
        
        # get the row index of each calibration query
        self.rows_calib = np.array([self.idxs_calib[0][k * i] for i in range(self.n_calib_queries)])


    def _get_calib_scores(self, abs_err):
        """
        This function implements the scores as the maximum of the absolute prediction errors
        """
        for i in range(self.n_calib_queries):
            scores = abs_err[(self.idxs_calib[0][self.k*i: self.k*(i+1)], self.idxs_calib[1][self.k*i: self.k*(i+1)])]
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
    

    def _validate_w(self, w):
        """
        validation helper to check if the sampling weights are valid
        """
        
        if w is None:
            return np.ones_like(self.Mhat)
        
        message = "Weights should be formatted as a numpy.ndarray of non-negative numbers, "+\
                  "and the dimension should match that of the input matrix, i.e. {}.".format(self.Mhat.shape)

        if type(w) != type(self.Mhat):
            raise ValueError(message)
        
        w_type = w.dtype.kind
        if w_type not in ("i", "f"):
            raise ValueError(message)
        
        if (
            w.shape != self.Mhat.shape 
            or np.any(w < 0)
        ):
            raise ValueError(message)
            
        return w
    


    def _initialize_df(self, a_list, n):
        """
        Helper that initializes the confidence interval DataFrame
        """
        df = pd.DataFrame({})

        df["alpha"] = a_list
        df["lower"] = [np.zeros(n) for _ in a_list]
        df["upper"] = [np.zeros(n) for _ in a_list]
        df["is_inf"] = [np.zeros(n) for _ in a_list]
        
        return df
    

    def get_CI(self, idxs_test, alpha, w_test=None, allow_inf=True):
        """
        Compute the confidence intervals for all test queries at any confidence levels

        Parameters
        ----------
        idxs_test: tuple 
            idxs_test[0] is the row indices of the test points and
            idxs_test[1] is the column indices, k consecutive indices are treated as a query
        alpha : float or list of floats
        w_test : ndarray, optional
            non-negative sampling weights for the test points, dimension should match the
            shape of the input matrix. Default is uniform weights.
        allow_inf: bool
            If True, the output intervals might be (-np.inf, np.inf).

        Returns
        -------
        DataFrame
            a pandas DataFrame with 4 columns: alpha, lower, upper, is_inf
            alpha: numeric value of confidence level.
            lower: array of lower confidence bounds.
            upper: array of upper confidence bounds.
            is_inf: array of 0s and 1s where 1 indicates the method outputs (-np.inf, np.inf).
        """
        w_test = self._validate_w(w_test)
        n_test_queries = len(idxs_test[0])//self.k
        a_list = np.array(alpha).reshape(-1)
        
        df = self._initialize_df(a_list=a_list, n = n_test_queries * self.k)
        
        if self.verbose:
            print("Computing conformal prediction intervals for {} test queries...".format(n_test_queries))
            sys.stdout.flush()

        # Compute constant components in weights that are invariant to test query
        sr_prune, sw_prune, sr_miss, delta, scale = self._compute_universal(w_test)      
        
        for i in tqdm(range(n_test_queries), desc="CI", leave=True, position=0, 
                      disable = not self.progress):
            idx_test = (idxs_test[0][self.k*i : self.k*(i+1)], idxs_test[1][self.k*i : self.k*(i+1)])
            row_test = idx_test[0][0]

            weights = self._weight_single(idx_test, row_test, w_test, sr_prune, sw_prune, sr_miss, delta, scale)
            cweights = np.cumsum(weights[self.calib_order])
            est = np.array(self.Mhat[idx_test])

            for row, a in enumerate(a_list):          
                if cweights[-1] < 1-a:
                    df.loc[row, "is_inf"][self.k*i : self.k*(i+1)] = 1
                    if allow_inf:
                        df.loc[row, "lower"][self.k*i : self.k*(i+1)] = -np.inf
                        df.loc[row, "upper"][self.k*i : self.k*(i+1)] = np.inf
                    else:
                        df.loc[row, "lower"][self.k*i : self.k*(i+1)] = est - self.st_calib_scores[-1]
                        df.loc[row, "upper"][self.k*i : self.k*(i+1)] = est + self.st_calib_scores[-1]
                else:
                    idx = np.argmax(cweights >= 1-a)
                    df.loc[row, "lower"][self.k*i : self.k*(i+1)] = est - self.st_calib_scores[idx]
                    df.loc[row, "upper"][self.k*i : self.k*(i+1)] = est + self.st_calib_scores[idx]
        
        if self.verbose:
            print("Done!")
            sys.stdout.flush()
            
        return df
    

    # def naive_CI(self, idxs_test, alpha, allow_inf=True):
    #     n_test_queries = len(idxs_test[0])//self.k
    #     if self.verbose:
    #         print("Computing naive prediction intervals for {} test queries...".format(n_test_queries))
    #         sys.stdout.flush()
        
    #     level_corrected = (1 - alpha) * (1 + 1/self.n_calib_queries)
    #     est = np.array(self.Mhat[idxs_test])
        
    #     if level_corrected >= 1:
    #         is_inf = np.ones(n_test_queries)
    #         if allow_inf:
    #             lower = np.repeat(-np.inf, len(idxs_test[0]))
    #             upper = np.repeat(np.inf, len(idxs_test[0]))
    #         else:
    #             lower = est - self.st_calib_scores[-1]
    #             upper = est + self.st_calib_scores[-1]
    #     else:
    #         is_inf = np.zeros(n_test_queries)
    #         qnt = np.quantile(self.calib_scores, level_corrected, method="higher")
    #         lower = est - qnt
    #         upper = est + qnt

    #     if self.verbose:
    #         print("Done!")
    #         sys.stdout.flush()
    #     return lower, upper, is_inf
    


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
        n_test_entries = len(idxs_test[0])
        n_test_queries = n_test_entries//self.k

        # Make Bonferroni correction on the confidence level
        a_list = np.array(alpha).reshape(-1)
        corrected_list = a_list/self.k

        if self.verbose:
            print("Computing Bonferroni-style intervals for {} test queries...".format(n_test_queries))
            sys.stdout.flush()

        df = self.sci.get_CI(idxs_test, corrected_list, w_test=w_test, allow_inf=allow_inf)

        if self.verbose:
            print("Done!")
            sys.stdout.flush()
        
        df.alpha = a_list
        return df