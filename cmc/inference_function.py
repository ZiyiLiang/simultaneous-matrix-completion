import pdb

import numpy as np
from scipy.linalg import orth, sqrtm
from scipy.stats import norm
from solvers import _svd_solve


def matrix_decomposition_with_CI(M_obs, A, r, alpha):
    """
    Matrix completion by nonconvex methods (convex part commented) and uncertainty quantification
    Nothing is random here so no need to specify random seed
    Args
    ------
    M_obs:      Matrix observed with shape m x n
    A:   Bernoulli masking indicating if an entry is observed
    r:   rank of the matrix
    random_state:    Random seed to ensure reproducibility.
    alpha:  Desired coverage level is (1-alpha)

    Returns
    -------
    Z_d:      completed matrix
    CI_left:    a matrix of the lower bounds of the confidence intervals
    CI_right:   a matrix of the upper bounds of the confidence intervals
    """

    # Rescale the input
    M_scale = np.linalg.norm(M_obs)
    M_obs /= M_scale

    # Estimating observation probability p

    # # random seed
    # np.random.seed(random_state)

    n1 = M_obs.shape[0]
    n2 = M_obs.shape[1]
    n = max(n1, n2)
    p_est = A.sum() / (n1 * n2) # 1-prob_masked
    # print(p_est)

    # Spectral Initialization
    M_ipw = M_obs / p_est
    U, S, VT = _svd_solve(M_ipw, k = r, algorithm='arpack')
    # print("S",S)
    M_spectral = U @ np.diag(S) @ VT
    S_sqrt = np.sqrt(S)
    S_sqrt_diag = np.diag(S_sqrt)
    # print("S", S_sqrt_diag)

    # Noise level estimation
    sigma_est = np.sqrt((((M_spectral - M_obs) * A) ** 2).sum() / (n1 * n2 * p_est))

    # Nonconvex optimization (option 1)
    eta = 0.1
    max_iter = 1000
    lambda_ = 0

    X = U @ S_sqrt_diag
    Y = VT.T @ S_sqrt_diag

    # print("X", X)

    # gradient descent
    for t in range(max_iter):
        temp = (X @ Y.T - M_obs) * A
        grad_X = temp @ Y + lambda_ * X
        grad_Y = temp.T @ X + lambda_ * Y
        X -= eta * grad_X
        Y -= eta * grad_Y
        grad_norm = np.sum(grad_X ** 2) + np.sum(grad_Y ** 2)
        if grad_norm < 1e-6:
            break

    # Debias (only needed if lambda_ > 0)
    eye_r = np.eye(r)
    # debug
    # print("X",X)
    # print("np.linalg.inv(X.T @ X)",np.linalg.inv(X.T @ X))
    # print("eye_r + lambda_ / p_est * np.linalg.inv(X.T @ X):", eye_r + lambda_ / p_est * np.linalg.inv(X.T @ X))
    # print("eye_r + lambda_ / p_est * np.linalg.inv(Y.T @ Y):", eye_r + lambda_ / p_est * np.linalg.inv(Y.T @ Y))
    # X_ncvx_d = X @ sqrtm(eye_r + lambda_ / p_est * np.linalg.inv(X.T @ X))
    # Y_ncvx_d = Y @ sqrtm(eye_r + lambda_ / p_est * np.linalg.inv(Y.T @ Y))
    X_ncvx_d = X
    Y_ncvx_d = Y
    Z_ncvx_d = X_ncvx_d @ Y_ncvx_d.T

    # # Convex relaxation (option 2)
    # lambda_ = 2 * sigma_est * np.sqrt(n * p_est)
    # max_iter = 1000
    # eta = 1
    # Z = M_spectral.copy()
    #
    # # Proximal gradient method
    # for t in range(max_iter):
    #     temp = Z - eta * (Z - M_obs) * A
    #     U_, S_, VT_ = _svd_solve(temp, k=r, algorithm='arpack')
    #     Z_new = U_ @ np.diag(np.maximum(0, S_ - lambda_ * eta)) @ VT_
    #     grad_norm = np.sqrt(np.sum((Z_new - Z) ** 2)) / eta
    #     Z = Z_new.copy()
    #     if grad_norm < 1e-6:
    #         break
    #
    # # Debiasing
    # U_, S_, VT_ = _svd_solve(Z, k=r, algorithm='arpack')
    #
    # # temp = TruncatedSVD(n_components=r, algorithm='arpack')
    # # temp.fit(Z)
    # X_cvx_d = U_ @ sqrtm(np.diag(S_) + lambda_ / p_est * eye_r)
    # Y_cvx_d = VT_.T @ sqrtm(np.diag(S_) + lambda_ / p_est * eye_r)
    # Z_cvx_d = X_cvx_d @ Y_cvx_d.T


    # Compute entrywise (1-alpha) CI
    alpha = alpha
    X_d = X_ncvx_d  # can also use X_cvx_d
    Y_d = Y_ncvx_d  # can also use Y_cvx_d
    Z_d = Z_ncvx_d  # can also use Z_cvx_d
    #
    # X_d = X_cvx_d
    # Y_d = Y_cvx_d
    # Z_d = Z_cvx_d

    # Create 1D arrays of ones with n1 and n2 elements
    ones_n1 = np.ones(n1)
    ones_n2 = np.ones(n2)

    # Compute the diagonal elements from X_d and Y_d matrix products
    diag_X = np.diag(X_d @ np.linalg.inv(X_d.T @ X_d) @ X_d.T)
    diag_Y = np.diag(Y_d @ np.linalg.inv(Y_d.T @ Y_d) @ Y_d.T)

    # Reshape diag_X and diag_Y for broadcasting
    diag_X_reshaped = diag_X.reshape(n1, 1)
    diag_Y_reshaped = diag_Y.reshape(1, n2)

    # Element-wise multiplication of reshaped diagonal elements
    product_matrix =  diag_X_reshaped @ np.ones((1,n2)) + np.ones((n1,1)) @ diag_Y_reshaped

    # Compute Var by scaling the product_matrix with sigma_est and p_est
    Var = (sigma_est ** 2 / p_est) * product_matrix

    CI_left = Z_d - norm.ppf(1 - alpha / 2) * np.sqrt(Var)
    CI_right = Z_d + norm.ppf(1 - alpha / 2) * np.sqrt(Var)

    # Undo the rescaling
    CI_left *= M_scale
    CI_right *= M_scale

    return CI_left, CI_right


def run_single_experiment_paper(underlying_matrix, mask, r, alpha):
    """
    run a single experiment
    """

    M_obs = underlying_matrix * mask
    CI_left, CI_right = matrix_decomposition_with_CI(M_obs=M_obs, A=mask, r=r, alpha=alpha)

    return CI_left, CI_right