# This file is modified from R codes provided by authors of the paper:
# Inference and uncertainty quantification for noisy matrix completion https://www.pnas.org/doi/10.1073/pnas.1910053116
# We made the code structures more adaptive while maintaining the same methodology.

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm
from solvers import _svd_solve

class noisy_CI():
    def __init__(self, M, mask, r):
        self.M = M          # the observed matrix
        self.mask = mask    # the mmasking matrix
        self.r = r

        # Estimating observation probability p
        self.n1 = M.shape[0]
        self.n2 = M.shape[1]
        self.p_est = mask.sum() / (self.n1 * self.n2) # 1-prob_masked

    def debias_cvx(self, Z, lambda_):
        U_, S_, VT_ = _svd_solve(Z, k=self.r, algorithm='arpack')
        X_d = U_ @ sqrtm(np.diag(S_) + lambda_ / self.p_est * np.eye(self.r))
        Y_d = VT_.T @ sqrtm(np.diag(S_) + lambda_ / self.p_est * np.eye(self.r))
        return X_d, Y_d

    def debias_ncvx(self, X, Y, lambda_):
        X_d = X @ sqrtm(np.eye(self.r) + lambda_ / self.p_est * np.linalg.inv(X.T @ X))
        Y_d = Y @ sqrtm(np.eye(self.r) + lambda_ / self.p_est * np.linalg.inv(Y.T @ Y))
        return X_d, Y_d
    
    def compute_CI(self, X_d, Y_d, alpha, sigma=None):
       # Spectral Initialization
        M_ipw = self.M / self.p_est
        U, S, VT = _svd_solve(M_ipw, k = self.r, algorithm='arpack')
        M_spectral = U @ np.diag(S) @ VT

        if sigma is None:
            # Noise level estimation
            sigma_est = np.sqrt((((M_spectral - self.M) * self.mask) ** 2).sum() /
                                (self.n1 * self.n2 * self.p_est))
        else:
            sigma_est = sigma
        
        # Compute the diagonal elements from X_d and Y_d matrix products
        diag_X = np.diag(X_d @ np.linalg.inv(X_d.T @ X_d) @ X_d.T)
        diag_Y = np.diag(Y_d @ np.linalg.inv(Y_d.T @ Y_d) @ Y_d.T)

        # Reshape diag_X and diag_Y for broadcasting
        diag_X_reshaped = diag_X.reshape(self.n1, 1)
        diag_Y_reshaped = diag_Y.reshape(1, self.n2)

        # Element-wise multiplication of reshaped diagonal elements
        product_matrix =  diag_X_reshaped @ np.ones((1,self.n2)) + np.ones((self.n1,1)) @ diag_Y_reshaped

        # Compute Var by scaling the product_matrix with sigma_est and p_est
        Var = (sigma_est ** 2 / self.p_est) * product_matrix

        Z_d = X_d @ Y_d.T
        CI_left = Z_d - norm.ppf(1 - alpha / 2) * np.sqrt(Var)
        CI_right = Z_d + norm.ppf(1 - alpha / 2) * np.sqrt(Var)

        return CI_left, CI_right
