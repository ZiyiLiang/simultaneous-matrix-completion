import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pdb

def wsc(X, y_cov, delta=0.1, random_state=0):
    
    # find v by logistic regression
    lm = linear_model.LogisticRegression(penalty = None , solver = 'sag',
                                         random_state=random_state).fit(X, y_cov)
    v = lm.coef_[0].T

    n = len(y_cov)       
    z = np.dot(X,v)
    # Compute mass
    z_order = np.argsort(z)
    z_sorted = z[z_order]
    cover_ordered = y_cov[z_order]
    ai_max = int(np.round((1.0-delta)*n))
    ai_best = 0
    bi_best = n
    cover_min = 1

    for ai in np.arange(0, ai_max):
        bi_min = np.minimum(ai+int(np.round(delta*n)),n)
        coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
        coverage[np.arange(0,bi_min-ai)]=1
        bi_star = ai+np.argmin(coverage)
        cover_star = coverage[bi_star-ai]
        if cover_star < cover_min:
            ai_best = ai
            bi_best = bi_star
            cover_min = cover_star
    return cover_min, v, z_sorted[ai_best], z_sorted[bi_best]

    

def wsc_estimate(M, U, V, test_mask, PI, delta=0.1, test_size=0.75, random_state=0):
    """ 
    Randomly split the test set into training set for finding the worst-slice coverage
    section and evaluation set for esimating the out-of-bag worst-slice coverage. 

    Args:
    ---------
    M:       The matrix to be completed.
    U, V:    Factor matrices of M, both can be ground truth matrices or estimated
             from a matrix completion algorithm.
    test_mask:  Binary mask with 1 if entry is a test point, 0 otherwise.
    PI:      Prediction interval of the test points

        
    Return:
    ---------
    coverage: Estimated worst-slice coverage
    v:       The factor coefficients of the worst-slice coverage section.
    a, b:    Lower and upper bound for statistics used in estimating wsc.
    """
    def wsc_vab(X, y_cov, v, a, b):
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(y_cov[idx])
        return coverage, idx

    test_idx = np.where(test_mask == 1)
    y_cov = np.array([M[i][j] <= PI[idx][1] and M[i][j] >= PI[idx][0] 
                      for idx, (i,j) in enumerate(zip(*test_idx))])
    X = np.array([np.concatenate([U[i,:],V.T[:,j]]) for i,j in zip(*test_idx)])

    X_train, X_test, y_train, y_test, _, i_test, _, j_test = train_test_split(X, y_cov, test_idx[0], test_idx[1],
                                                    test_size=test_size, random_state=random_state)

    # Find adversarial parameters
    _, v, a, b = wsc(X_train, y_train, delta=delta, random_state=random_state)
    
    # Estimate coverage
    coverage, idx = wsc_vab(X_test, y_test, v, a, b)

    ws_mask = np.zeros_like(test_mask)
    ws_mask[i_test[idx], j_test[idx]] = 1
    return coverage, (v, a, b), ws_mask