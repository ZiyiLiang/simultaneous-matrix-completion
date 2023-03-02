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
        k latent orthonormal vectors 
        """
        np.random.seed(random_state)
        U = ortho_group.rvs(self.m)[:self.r].T
        V = ortho_group.rvs(self.n)[:self.r].T
        S = np.diag(np.random.uniform(-2 * max(self.m, self.n), 
                                      2 * max(self.m, self.n), self.r))
        M = U @ S @ V.T
        return U, V, M
    
    def sample_noisy(self, sigma, random_state):
        """
        Generate noisy target matrix by sampling factor matrices from 
        k latent orthonormal vectors plus gaussian noise
        """
        U, V, M = self.sample_noiseless(random_state=random_state)
        noisy_M = np.random.randn(self.m, self.n) * sigma + M
        return U, V, M, noisy_M





class RandomSampling():
    """ 
    This class implements the random orthogonal model described by Candes and
    Recht (2008) which satisfies the incoherence condition.
    """
    def __init__(self, m, n):
        self.m = m   # shape of the output matrix
        self.n = n
        
    def sample_observed(self, prob_masked=0.5, fix_size=True, random_state=0):
        """
        Sample the set of observed indices
        
        Return:
        ---------
        mask:     a binary mask with 1 denotes observed, and 0 denotes unobserved.
        """
        np.random.seed(random_state)
        mask = np.array([1] * (self.m * self.n))
        
        if fix_size:
            missing_size = int(self.m * self.n * prob_masked)
            missing_idx = np.random.choice(self.m * self.n, missing_size, replace=False)
            mask[missing_idx] = 0
        else:
            mask = 1 - np.random.binomial(n=1, p=prob_masked, size=(self.m, self.n))
        return mask.reshape([self.m, self.n])
    
    def sample_train_calib(self, mask, prob_calib=0.5, fix_size=True, random_state=0):
        flat_mask = mask.flatten(order='C')
        obs_idx = np.where(flat_mask == 1)[0]
        
        np.random.seed(random_state)
        if fix_size:
            calib_size = int(len(obs_idx) * prob_calib)
            calib_idx = np.random.choice(obs_idx, calib_size, replace=False)
        else:
            calib_idx = obs_idx[np.random.binomial(n=1, p=prob_calib, size=mask.shape) == 1]
        
        flat_mask[calib_idx] = 0
        train_mask = flat_mask.reshape(mask.shape)
        calib_mask = mask - train_mask

        return train_mask, calib_mask
    
