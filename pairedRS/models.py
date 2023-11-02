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



class PairSampling():
    """ 
    This class implements functions to sample observed index set and perform data
    splitting 
    """
    def __init__(self, n1, n2):
        self.n1 = n1   # shape of the output matrix
        self.n2 = n2
        self.shape = (self.n1, self.n2)
        
    def sample_submask(self, sub_size, mask=None, w=None, random_state=0):
        rng = np.random.default_rng(random_state)
        
        if mask is None:
            mask = np.ones(self.shape)
        mask = mask.flatten(order='C')   # flatten the matrix by rows
        obs_idx = np.where(mask == 1)[0]
        sub_size = int(np.clip(sub_size, 1,len(obs_idx)-1))

        if w is not None:
            w = w.flatten(order='C')  # flatten the matrix by rows    
        else:
            w = np.array([1] * (self.n1 * self.n2)) / (self.n1 * self.n2)
        
        # Make sure the weights sum up to 1 at selected indexes
        w = w / np.sum(w[obs_idx])
        sub_idx = rng.choice(obs_idx, sub_size, replace=False, p=w[obs_idx])

        sub_mask = np.zeros(self.n1 * self.n2)
        sub_mask[sub_idx] = 1
        return sub_mask.reshape(self.shape)
    
    def sample_train_calib(self, mask_obs, n_pairs, random_state=0):
        rng = np.random.default_rng(random_state)
        
        mask_drop = np.zeros_like(mask_obs)
        mask_train = np.zeros_like(mask_obs)
        mask_calib = np.zeros_like(mask_obs)
        n_obs = np.sum(mask_obs, axis=1)   # Check the observation# in each row
        
        assert n_pairs <= np.sum(n_obs // 2), "Too many calibration pairs and not enough observations!"
        idxs_calib = (-np.ones(2 * n_pairs, dtype=int),\
                      -np.ones(2 * n_pairs, dtype=int))
        
        shuffled_pool = {}
        for i in range(self.n1):
            if n_obs[i] > 0:
                idxs = np.where(mask_obs[i]==1)[0]
                rng.shuffle(idxs)
                if n_obs[i] % 2 != 0:
                    # If the row length is odd, take the last shuffled index as the dropout
                    mask_drop[i][idxs[-1]] = 1
                    idxs = idxs[:-1]
                if len(idxs) > 0:
                    shuffled_pool[i] = idxs
                    
        
        # Sample calibration pairs
        for i in range(n_pairs):
            row = rng.choice(list(shuffled_pool.keys()))  # Randomly select a row
            
            # First two shuffled entries form a pair
            idxs_calib[0][2 * i], idxs_calib[0][2 * i + 1] = row, row
            idxs_calib[1][2 * i], idxs_calib[1][2 * i + 1] = shuffled_pool[row][0], shuffled_pool[row][1]
            
            # Update the shuffled pool by removing the selected pair
            if len(shuffled_pool[row]) == 2:
                del shuffled_pool[row]
            else:
                shuffled_pool[row] = shuffled_pool[row][2:]
                
        # Get the training mask
        mask_calib[idxs_calib] = 1
        mask_train = mask_obs - mask_calib
        return mask_train, idxs_calib, mask_calib, mask_drop



# This class models and adds user-dependent noises to the true data matrix
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