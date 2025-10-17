import numpy as np   
import pdb
import scipy.stats as stats
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



class SamplingBias():
    def __init__(self, m, n, normalize=True):
        self.m = m
        self.n = n
        self.shape = (self.m, self.n)
        self.std = normalize
        
    def latent_weights(self, U, V, r, v, a, b, scale=0.5):
        r = int(len(v)/2)
        x = np.tile(np.dot(U, v[:r]), (len(U),1)).T + np.tile(np.dot(V, v[r:]), (len(U),1))
        w = np.zeros_like(x)
        w[np.where((x <= b) & (x >= a))] = 1
        w[np.where(x > b)] = stats.norm.pdf(x[np.where(x > b)], loc=b, scale=scale)/stats.norm.pdf(b, loc=b, scale=scale)
        w[np.where(x < a)] = stats.norm.pdf(x[np.where(x < a)], loc=a, scale=scale)/stats.norm.pdf(a, loc=a, scale=scale)
        return w/np.sum(w) if self.std else w

    def unif_weights(self):
        w = np.ones(self.shape)
        return w/np.sum(w) if self.std else w

    def inc_weights(self, scale=1, logistic=False):
        if logistic:
            w = np.arange(1.0, self.m*self.n+1)*scale/(self.m*self.n)
        else:
            w = np.arange(1.0, self.m*self.n+1)**scale/(self.m*self.n)
        w = w.reshape(self.shape)

        if logistic:
            w -= np.mean(w)
            return 1 / (1 + np.exp(-w))
        return w/np.sum(w) if self.std else w
    
    def block_weights(self, ratio, scale, random_state=0):
        rng = np.random.default_rng(random_state)
        w = np.ones(self.shape)
        is_large = rng.binomial(1,ratio,self.m)
        idxs = np.where(is_large==1)[0]
        w[idxs,] = np.repeat(scale, self.n)
        return w/np.sum(w) if self.std else w


class Movielens_weights():
    def __init__(self, demo, genre, normalize=True, epsilon=1e-2):
        self.demo = demo
        self.genre = genre

        self.m = len(genre)
        self.n = len(demo)
        self.shape = (self.m, self.n)
        self.std = normalize
        self.epsilon = epsilon  # Small constant to avoid zero weights


    def demo_weights(self, var="age", **kwargs):
        # Initialize weights for users (rows)
        user_weights = np.ones(self.n)
        
        if var == "age":
            # Get age data and apply weighting based on scale
            scale = kwargs.get("scale", 1)  # Default scale to 1 if not provided
            ages = self.demo['age'].values
            
            # Compute weights as an inverse exponential function of age
            user_weights = np.exp(-scale * (ages / np.max(ages)))  # normalize age

        elif var == "female":
            # Assign weight 1 to females and 0 to males
            user_weights = np.where(self.demo['gender'] == 'F', 1, 0) + self.epsilon
            
        elif var == "male":
            # Assign weight 1 to males and 0 to females
            user_weights = np.where(self.demo['gender'] == 'M', 1, 0) + self.epsilon
        
        else:
            raise ValueError("Unsupported variable for weighting. Use 'age', 'female', or 'male'.")
        
        # Create the full weight matrix by repeating user weights for each row (movie)
        weight_matrix = np.tile(user_weights, (self.m, 1))  # Shape: (m, n)

        # Optionally normalize weights
        if self.std:
            weight_matrix = weight_matrix / np.sum(weight_matrix)
        
        return weight_matrix
    
    def genre_weights(self, genre):
        if genre not in self.genre.columns[1:]:  # Assuming first column is 'movieid'
            raise ValueError(f"Genre '{genre}' not found in the dataset.")
        
        # Create a column of weights: 1 if movie belongs to the genre, 0 otherwise
        movie_weights = self.genre[genre].values + self.epsilon
        
        # Create the full weight matrix by repeating movie weights for each user
        weight_matrix = np.tile(movie_weights[:, np.newaxis], (1, self.n))  # Shape: (m, n)
        
        # Optionally normalize weights
        if self.std:
            weight_matrix = weight_matrix / np.sum(weight_matrix)
        
        return weight_matrix
        




