import numpy as np
import cvxpy as cp

from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error




def nnm_solve(M, mask, verbose=True, eps=10**-6, random_state=0):
    """ 
    Matrix completion solved by nuclear norm minimizaion

    Args
    ------
    M:      matrix observed
    mask:   bernoulli masking indicating if an entry is observed
    eps:    small constant to control the error

    Returns
    -------
    X:      completed matrix
    """
    np.random.seed(random_state)

    X = cp.Variable(M.shape)
    objective = cp.Minimize(cp.norm(X, 'nuc'))
    constraints = [cp.abs(cp.multiply(mask, M-X)) <= eps]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver = cp.SDPA, verbose=verbose)

    return X.value



def _svd_solve(M, k, algorithm):
    if algorithm == 'random':
        U, S, VT = randomized_svd(
            M, n_components = min(k, min(M.shape)-1)) 
    elif algorithm == 'arpack' or 'propack':
        # NOTE: the sign ambuity for package svds might not be resolved,
        #       it is generally not an issue for matrix completion, but
        #       one might want to fix the issue if needed. 
        U, S, VT = svds(M, k = min(k, min(M.shape)-1))
    else:
        raise ValueError('Unknown algorithm, supported options are: {}, {} and {}.'.format(
                          '\'arpack\'', '\'propack\'', '\'random\'') )
    return U, S, VT


def svt_solve(M, 
              mask,
              tau = None,
              delta = None,
              eps = 10**-4,
              max_iterations = 1000,
              algorithm = 'propack',
              verbose = True,
              random_state = 0
              ):
    """ 
    Matrix completion solved by singular value thresholding

    Args
    ------
    M:      Matrix observed with shape m x n
    mask:   Bernoulli masking indicating if an entry is observed
    tau:    Singular value threshold, advised default value for fast convergence
            is 5 * (m + n) / 2 
    delta:  Step size for each update, advised default value is 1.2 / p
            where p is the missing proportion 
    eps:    Small constant to control the error
    max_iteration:   Maximum number of iterative steps
    algorithm:       Solver used to compute the SVD, options are 'propack', 'random', 
                     'arpack'
                     Default is 'propack' for its efficiency.
    verbose:         Whether to print progress message or not, default is set to 
                     True to keep track of the training.
    random_state:    Random seed to ensure reproducibility.

    Returns
    -------
    X:      completed matrix
    """
    np.random.seed(random_state)
    
    tau = 5 * (np.sum(M.shape)) / 2 if not tau else tau
    delta = 1.2 * (np.prod(M.shape)) / np.sum(np.sum(mask))
    k = 0    # number of principle components to be calculated
    Y = np.zeros_like(M)
    
    for i in range(max_iterations):
        k += 1
        U, S, VT = _svd_solve(Y, k, algorithm)
        # rerun svd with more components if estimated rank is too small 
        while np.min(S) >= tau:
            k += 5
            U, S, VT = _svd_solve(Y, k, algorithm)
        shrink_S = np.maximum(S - tau, 0)
        k = np.count_nonzero(shrink_S)
        X = U @ np.diag(shrink_S) @ VT
        
        Y += delta * mask * (M - X)
        
        recon_error = np.linalg.norm(mask * (M - X)) / np.linalg.norm(mask * M)
        
        # print progress message regularly.
        if verbose and i % 10 == 0:  
            print("Iteration: %i; Rel error: %.4f" % (i + 1, recon_error))
        if recon_error < eps:
            print("Stopping criteria met, training terminated.")
            break

    return X


def pmf_solve(M,
              mask,
              k,
              mu = 1e-2,
              eps = 10**-6,
              max_iteration = 1000,
              random_state = 0
              ):
    """
    Solve probabilistic matrix factorization using alternating least squares.
    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.
    [ Salakhutdinov and Mnih 2008 ]
    [ Hu, Koren, and Volinksy 2009 ]
    Parameters:
    -----------
    M : m x n array
        matrix to complete
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    k : integer
        how many factors to use
    mu : float
        hyper-parameter penalizing norm of factored U, V
    epsilon : float
        convergence condition on the difference between iterative results
    max_iterations: int
        hard limit on maximum number of iterations
    Returns:
    --------
    X: m x n array
        completed matrix
    """
    # logger = logging.getLogger(__name__)
    np.random.seed(random_state)
    m, n = M.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)

    for _ in range(max_iteration):

        for i in range(m):
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([V.T, C_u[i], M[i,:]]))

        for j in range(n):
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([U.T, C_v[j], M[:,j]]))

        X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        # if _ % 1 == 0:
        #     logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if mean_diff < eps:
            break
        prev_X = X

    return X




