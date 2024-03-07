import numpy as np
from sklearn import linear_model
import pdb

from methods import QuerySampling


def wse(X, y, delta=0.1):
    """ Find the worst-slice error section.
    """
    # find v by logistic regression
    lm = linear_model.LinearRegression().fit(X, y)
    v = lm.coef_.T

    n = len(y)       
    z = np.dot(X,v)
    # Compute mass
    z_order = np.argsort(z)
    z_sorted = z[z_order]
    error_ordered = y[z_order]
    ai_max = int(np.round((1.0-delta)*n))
    ai_best = 0
    bi_best = n
    error_max = 0

    for ai in np.arange(0, ai_max):
        bi_min = np.minimum(ai+int(np.round(delta*n)),n)
        avg_error = np.cumsum(error_ordered[ai:n]) / np.arange(1,n-ai+1)
        avg_error[np.arange(0,bi_min-ai)]=0
        bi_star = ai+np.argmax(avg_error)
        error_star = avg_error[bi_star-ai]
        if error_star > error_max:
            ai_best = ai
            bi_best = bi_star
            error_max = error_star
    return error_max, v, z_sorted[ai_best], z_sorted[bi_best]



def wsc_estimate(M, Mhat, U, V, mask_miss, delta=0.1, train_size=0.25, random_state=0):
    """ 
    Randomly split the missing set into training set for finding the worst-slice estimation
    section and evaluation set for esimating the out-of-bag worst-slice coverage. 

    Parameters:
    -----------
    M : ndarray
        Ground Truth matrix.
    U, V : ndarrays
        Factor matrices of M estimated from a matrix completion algorithm.
    mask_miss : ndarray 
        Binary mask with 1 if entry is a test point, 0 otherwise.
    delta : float in (0,1), optional
        The wsc slice should be estimated using at least delta portion of the traning set
    train_size : float in (0,1), optional
        Proportion of the missing indices used for training
        
    Return:
    ---------
    wsc_error : float
        Estimated worst-slice average completion error
    v : ndarray      
        The factor coefficients of the worst-slice error section.
    a, b : float
        Lower and upper bound for statistics used in estimating wsc.
    """

    # Sample the training set uniformly at random from the missing indices
    sampler = QuerySampling(*M.shape)
    mask_train, mask_test = sampler.sample_submask(sub_size=train_size, mask=mask_miss, random_state=random_state)
    train_idx = np.where(mask_train == 1)
    abs_error = np.abs(M-Mhat)

    X_train = np.array([np.concatenate([U[i,:],V.T[:,j]]) for i,j in zip(*train_idx)])
    y_train = abs_error[train_idx]

    # Find adversarial parameters
    _, v, a, b = wse(X_train, y_train, delta=delta)

    return (v, a, b), mask_test