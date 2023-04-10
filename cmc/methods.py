import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import fsolve
from math import sqrt, pi


class mwnchypg():
    """ 
    This class implements the calculation of probability mass function of the
    multivariate Wallenius' noncentral hypergeometric distribution (mwnchypg).
    """        
    def __init__(self, verbose = True):
        self.verbose = verbose

    def _integrand(self, x, w, s, d):
        """ The integrand from direct computation
        """
        T = 1
        for i in s:
            T *= (1 - x ** (w[i] / d))
        return T


    def _trans_integrand(self, x, w, s, d, r):
        """ 
        Transformed integrand which is suitable for applying the Laplace's method
        """
        T = 1
        for i in s:
            T *= (1 - x ** (w[i] * r))
        return r*d * x**(r*d - 1) * T

    
    def _get_scale(self, w, s, d):
        """
        Compute the scaling parameter r used in the transformed integrand such that
        it reaches the maximum at the center of integration interval, i.e. 1/2.
        """
        def z(r,w,s,d):
            res = d - 1/r
            for i in s:
                res -= w[i] / (2**(r*w[i]) - 1)
            return res
        
        r = fsolve(z, 1/d, (w,s,d))[0]
        return r
    
    def direct_integral(self, w, s):
        """
        This function evaluates the direct integration using numerical methods

        Args
        ------
        w:      list of weights, make sure the weights sum up 1, otherwise the computation
                may be errorneous. 
        s:      sampled indexes
        
        Return:
        ---------
        res:    direct integration evaluted by numerical method
        err:    estimated error
        """
        d = 1-sum(w[s])
        res, err = quad(self._integrand, 0, 1, args=(w,s,d))
        if self.verbose:
            print("Direct integration\n"+"-"*25+"\nResult:" \
                  "{} \nError: {} \nRelative error: {}".format(res, err, err/res))
        
        return res, err
            

    def transformed_integral(self, w, s, r=None):
        """
        This function evaluates the transformed integration using numerical methods

        Args
        ------
        w:      list of weights, make sure the weights sum up 1, otherwise the computation
                may be errorneous. 
        s:      sampled indexes
        r:      any postive scaling parameter used for the transformed integral, if no value
                is provided, r is automatically computed such that the integrand reaches the
                maximum at the center of the integration interval.
        
        Return:
        ---------
        res:    direct integration evaluted by numerical method
        err:    estimated error
        """
        d = 1-sum(w[s])
        if not r:
            r = self._get_scale(w, s, d)

        res, err = quad(self._trans_integrand, 0, 1, args=(w,s,d,r))
        if self.verbose:
            print("Transformed integration\n"+"-"*25+"\nResult:" \
                "{} \nError: {} \nRelative error: {}".format(res, err, err/res))
        return res, err
    
    def laplace_approx(self, w, s, t=0.5, r=None):
        """
        This function applys the Laplace's method to approximate the transformed integral

        Args
        ------
        w:      list of weights, make sure the weights sum up 1, otherwise the computation
                may be errorneous. 
        s:      sampled indexes
        t:      center of the Taylor expansion, ideally should be located at the maximum of 
                the integrand, default value is 0.5 which is the center of the integration 
                interval
        r:      any postive scaling parameter used for the transformed integral, if no value
                is provided, r is automatically computed such that the integrand reaches the
                maximum at t=0.5
        
        
        Return:
        ---------
        res:    approximation of the integral by the Laplace's method
        """
        d = 1-sum(w[s])
        if not r:
            r = self._get_scale(w, s, d)

        res = (1-r*d)/t**2
        for i in s:
            rw = r * w[i]
            res -= ((1-t**(rw)) * rw * (rw - 1) * t**(rw-2) + (rw)**2 * t**(2*rw-2))/(1-t**(rw))**2
        
        res = self._trans_integrand(t,w,s,d,r) * sqrt(-2*pi / (res))

        if self.verbose:
            print("Laplace's method\n"+"-"*25+"\nResult: {}".format(res))
        return res

    
    def check_shape(self, w, s, r=None, bin=1000):
        """
        Check if the transformed integrand is suitable for Laplace's method by 
        examining the shape of the integrand and d
        """
        d = 1-sum(w[s])
        if not r:
            r = self._get_scale(w, s, d)

        x = np.linspace(0, 1, bin, endpoint=True)
        y = self._trans_integrand(x, w, s, d, r)
        plt.plot(x, y)
        plt.axvline(x=0.5, color='r', ls='--')
        plt.ylabel("transformed integrand")
        plt.show()
        print("The reciprocal of d is {}, note if the value is too large,\
               Laplace method is generally not recommended.".format(1/d))
        
    

class Conformal_PI():
    def _get_calib_scores(self,calib_mask, M, Mhat):
        # use the absolute value of estimation residual as the conformity scores
        calib_idx = np.where(calib_mask == 1)
        return abs(M[calib_idx] - Mhat[calib_idx])

    def marginal_PI(self, calib_mask, test_mask, M, Mhat, alpha):
        """
        Caculate conformal prediction interval with marginal coverage.

        Args:
        ------
        calib_mask:   Index set for the calibration data.
        test_mask:    Index set for the test data.
        M:            Matrix to be estimated.
        Mhat:         Estimation of M.
        alpha:        Desired confidence level.

        Return:
        -------
        pi:           Prediction interval(s) for the test point(s).  
        """

        test_idx = np.where(test_mask == 1)
        calib_scores = self._get_calib_scores(calib_mask, M, Mhat)
        n_calib = len(calib_scores)

        qhat = np.quantile(calib_scores, np.ceil((n_calib+1)*(1-alpha))/n_calib,
                            method='higher')

        pi = [[est-qhat, est+qhat] for est in Mhat[test_idx]]
        return pi


