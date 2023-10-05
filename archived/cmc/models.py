import numpy as np   
from scipy.stats import ortho_group


class RandomFactorizationModel():
    def __init__(self, m, n, r):
        self.m = m   # shape of the output matrix
        self.n = n
        self.r = r   # matrix rank
    
    def sample_noiseless(self, random_state=0):
        """
        Generate noiseless target matrix by sampling factor matrices from 
        standard normal distribution
        """
        np.random.seed(random_state)
        U = np.random.randn(self.m, self.r)
        V = np.random.randn(self.n, self.r)
        M = U @ V.T
        return U, V, M

    def sample_noisy(self, sigma, random_state=0): 
        """
        Generate noisy target matrix by adding a random gaussian noise to the 
        noiseless matrix

        Args
        ------
        sigma: noise level
        """
        U, V, M = self.sample_noiseless(random_state=random_state)
        noisy_M = np.random.randn(self.m, self.n) * sigma + M
        return U, V, M, noisy_M


class RandomOrthogonalModel():
    """ 
    This class implements the random orthogonal model described by Candes and
    Recht (2008) which satisfies the incoherence condition.
    """
    def __init__(self, m, n, r):
        self.m = m   # shape of the output matrix
        self.n = n
        self.r = r   # matrix rank

    def sample_noiseless(self, random_state):
        """
        Generate noiseless target matrix by sampling factor matrices from 
        r latent orthonormal vectors 
        """
        np.random.seed(random_state)
        U = ortho_group.rvs(self.m)[:self.r].T
        V = ortho_group.rvs(self.n)[:self.r].T
        S = np.diag(np.random.uniform(-2 * max(self.m, self.n), 
                                      2 * max(self.m, self.n), self.r))
        M = U @ S @ V.T
        return U, V, S, M
    
    def sample_noisy(self, sigma, random_state):
        """
        Generate noisy target matrix by sampling factor matrices from 
        k latent orthonormal vectors plus gaussian noise
        """
        U, V, S, M = self.sample_noiseless(random_state=random_state)
        noisy_M = np.random.randn(self.m, self.n) * sigma + M
        return U, V, S, M, noisy_M





class RandomSampling():
    """ 
    This class implements functions to sample observed index set and perform data
    splitting 
    """
    def __init__(self, m, n):
        self.m = m   # shape of the output matrix
        self.n = n
        self.shape = (self.m, self.n)
        
    def sample_observed(self, prop = 0.5, w = None, fix_size=True, random_state=0):
        """
        Sample the set of observed indices

        Args:
        ---------
        prop:   proportion of observed entries 
        w:      matrix of weights, each index is sampled with a probability of its weight
                if not provided, the function assumes sampling uniformly at random
        fix_size:  If true, the sample size is fixed to be N*prop where N is the total
                   number of entries in the matrix, otherwise, the sample size is a 
                   random variable following a binomial distribution with Binom(N,prop).
              
        Return:
        ---------
        mask:     a binary mask with 1 denotes observed, and 0 denotes unobserved.
        """
        np.random.seed(random_state)
        mask = np.array([0] * (self.m * self.n))
        
        if fix_size:
            obs_size = int(self.m * self.n * prop)
        else:
            obs_size = np.random.binomial(n=self.m * self.n, p=prop)
        
        if w is not None:
            w = w.flatten(order='C')  # flatten the matrix by rows
        else:
            w = np.array([1] * (self.m * self.n)) / (self.m * self.n)
        
        obs_idx = np.random.choice(self.m * self.n, obs_size, 
                                       replace=False, p=w)
        mask[obs_idx] = 1
        return mask.reshape(self.shape)
    
    def sample_train_calib(self, mask, calib_size, fix_size=True, random_state=0):
        mask = mask.flatten(order='C')   # flatten the matrix by rows
        obs_idx = np.where(mask == 1)[0]
        
        calib_size = int(np.clip(calib_size, 1,len(obs_idx)-1))
        np.random.seed(random_state)
        if fix_size:
            calib_idx = np.random.choice(obs_idx, calib_size, replace=False)
        else:
            prob_calib = calib_size / len(obs_idx)
            calib_idx = obs_idx[np.random.binomial(n=1, p=prob_calib, size=self.shape) == 1]
        
        calib_mask = np.array([0] * (self.m * self.n))
        calib_mask[calib_idx] = 1
        train_mask = mask - calib_mask
        return train_mask.reshape(self.shape), calib_mask.reshape(self.shape)
    
