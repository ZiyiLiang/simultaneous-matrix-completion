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
    


# This class models and adds row-dependent noises to the true data matrix
class NoiseModel():
    def __init__(self, random_state=None):
        if random_state is None: 
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)
    
    def get_noisy_matrix(self, M, gamma_n=0, gamma_m=0, model='beta', a=1, b=1, mu=1, alpha=0.1, normalize=False):
        n1, n2 = M.shape
        
        # baseline noise
        base_noise = self.rng.normal(0,1, M.shape)
        
        # user-dependent noise
        if model == 'beta':
            row_noise = self.rng.beta(a, b, n1)
            row_noise = np.transpose(np.tile(row_noise, (n2,1)))
        elif model == 'step':
            row_noise_small = self.rng.normal(0,1,n1)
            row_noise_large = self.rng.normal(mu,0.1,n1)
            is_large = self.rng.binomial(1,alpha/2,n1)
            row_noise = is_large*row_noise_large + (1-is_large)*row_noise_small
            row_noise = np.transpose(np.tile(row_noise, (n2,1)))
        else:
            raise ValueError('Unknown noise model! Use either \'beta\' or \'step\'!')
            
        if normalize:
            base_noise /= np.max(base_noise)
            row_noise /= np.max(row_noise)
            M /= np.max(M)
        
        # noise mixture
        noise = (1-gamma_m)*base_noise + gamma_m*row_noise
        # noisy matrix
        M = (1-gamma_n)*M + gamma_n*noise
        
        return M   