import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import fsolve
from math import sqrt, pi
from tqdm import tqdm


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
        #pdb.set_trace()
        return r*d * x**(r*d - 1) * T

    def _log_trans_integrand(self, x, w, s, d, r, base):
        """ 
        Log-scaled transformed integrand for applying the Laplace's method
        stable to precision loss, suitable for large data set.
        """
        T = 0
        for i in s:
            T += np.emath.logn(base,1 - x ** (w[i] * r))
        return np.emath.logn(base,r*d) + (r*d - 1)*np.emath.logn(base,x) + T

    def _z(self,r,w,s,d):
            res = d - 1/r
            for i in s:
                res -= w[i] / (2**(r*w[i]) - 1)
            return res
     
    def _get_scale(self, w, s, d):
        """
        Compute the scaling parameter r used in the transformed integrand such that
        it reaches the maximum at the center of integration interval, i.e. 1/2.
        """
        r = fsolve(self._z, 1/d, (w,s,d))[0]
        return r
    
    def direct_integral(self, w, s):
        """
        This function evaluates the direct integration using numerical methods

        Args
        ------
        w:      list of weights. 
        s:      sampled indexes
        
        Return:
        ---------
        res:    direct integration evaluted by numerical method
        err:    estimated error
        """
        d = sum(w)-sum(w[s])
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
        w:      list of weights. 
        s:      sampled indexes
        r:      any postive scaling parameter used for the transformed integral, if no value
                is provided, r is automatically computed such that the integrand reaches the
                maximum at the center of the integration interval.
        
        Return:
        ---------
        res:    direct integration evaluted by numerical method
        err:    estimated error
        """
        d = sum(w)-sum(w[s])
        if not r:
            r = self._get_scale(w, s, d)

        res, err = quad(self._trans_integrand, 0, 1, args=(w,s,d,r))
        if self.verbose:
            print("Transformed integration\n"+"-"*25+"\nResult:" \
                "{} \nError: {} \nRelative error: {}".format(res, err, err/res))
        return res, err
    
    def laplace_approx(self, w, s, t=0.5, r=None, log_base="auto"):
        """
        This function applys the Laplace's method to approximate the transformed integral

        Args
        ------
        w:      List of weights. 
        s:      Sampled indexes
        t:      Center of the Taylor expansion, ideally should be located at the maximum of 
                the integrand, default value is 0.5 which is the center of the integration 
                interval
        r:      Any postive scaling parameter used for the transformed integral, if no value
                is provided, r is automatically computed such that the integrand reaches the
                maximum at t=0.5
        log_base:   'auto', None or positive numeric value. If given a positive int, the method
                    produces logscale approximation with the given base, if 'auto', a base value
                    is automatically selected to be N, if None, return unlogged result.
        
        
        Return:
        ---------
        res:    approximation of the integral by the Laplace's method, or logged value 
                if logscale is True.
        """
        d = sum(w)-sum(w[s])
        if not r:
            r = self._get_scale(w, s, d)
        
        if log_base == "auto": 
            log_base = len(w)       

        res = (1-r*d)/t**2
        for i in s:
            rw = r * w[i]
            res -= ((1-t**(rw)) * rw * (rw - 1) * t**(rw-2) + (rw)**2 * t**(2*rw-2))/(1-t**(rw))**2
        
        if log_base:
            res = self._log_trans_integrand(t,w,s,d,r,log_base) + np.emath.logn(log_base, sqrt(-2*pi / (res)))
        else:
            res = self._trans_integrand(t,w,s,d,r) * sqrt(-2*pi / (res))


        if self.verbose:
            print("Laplace's method\n"+"-"*25+"\nResult: {}".format(res)*(log_base is None) +\
                  "\nLog-scaled result: {}".format(res)*(log_base is not None))
        return res

    
    def check_shape(self, w, s, log_base="auto", r=None, bin=1000):
        """
        Check if the transformed integrand is suitable for Laplace's method by 
        examining the shape of the integrand and d
        """
        d = sum(w)-sum(w[s])
        if not r:
            r = self._get_scale(w, s, d)
        
        print('scaling parameter:',r)
        x = np.linspace(0, 1, bin, endpoint=True)

        if log_base == "auto": 
            log_base = len(w)
            print("Showing logscaled integrand with base {}".format(log_base))
        
        if log_base:
            y = self._log_trans_integrand(x, w, s, d, r, log_base)
        else:
            y = self._trans_integrand(x, w, s, d, r)
        plt.plot(x, y)
        plt.axvline(x=0.5, color='r', ls='--')
        plt.ylabel("transformed integrand"+"(Log scale)"*(log_base is not None))
        plt.show()
        print("The reciprocal of d is {}, note if the value is too large, "
               "Laplace method is generally not recommended.".format(1/d))
        
    

class Conformal_PI():
    def __init__(self, M, Mhat, train_mask, calib_mask, 
                 verbose=True, progress=True):
        # Flatten all the matrices to avoid format inconsistency in the following computations
        self.M = M.flatten(order='C')
        self.Mhat = Mhat.flatten(order='C')
        self.train_idxs = np.where(train_mask.flatten(order='C')== 1)[0]
        self.calib_idxs = np.where(calib_mask.flatten(order='C')== 1)[0]
        self.verbose = verbose
        self.progress = progress
        
        self.mw = mwnchypg(verbose=False)
        self.scores = np.abs(self.M - self.Mhat)
        self.calib_scores = self.scores[self.calib_idxs]
        self.calib_order = np.argsort(self.calib_scores)
        self.st_calib_scores = self.calib_scores[self.calib_order]
        
    def _weights_single(self, test_idx, w1, w2, d1, d2, r, log_base):
        """ Fix a test point, compute the weights for conformity scores of the given test point and
            all calibration points

        Args:
        ---------
        calib_idx:   
        test_idx:    
        w1:          Flattened normalized sampling weights(odds) for the calibration points.
        w2:          Flattened normalized sampling weights(odds) for the test point.

        Return:
        ---------
        """
        #n_calib = len(self.calib_idxs)
        #weights = np.zeros(n_calib+1) if log_base else np.ones(n_calib+1) 
        idxs = list(self.calib_idxs) + [test_idx]        
        
        # for i, idx in enumerate(idxs):
        #     if not w2[idx]:
        #         if i == n_calib:
        #             raise Exception( "Probability of observing the given test point must be positive!")
        #         weights[i] = -np.inf if log_base else 0
        #     else:
        #         if log_base:
        #             weights[i] += np.emath.logn(log_base, (d1+w1[idx]-w1[test_idx])/d1 * \
        #                                         0.5**(r*(w1[idx]-w1[test_idx])) * (1-0.5**(r*w1[test_idx])) / (1-0.5**(r*w1[idx])))
        #             weights[i] += np.emath.logn(log_base, w2[idx] / (d2 + w2[idx] - w2[test_idx]))
        #         else:
        #             weights[i] *= (d1+w1[idx]-w1[test_idx])/d1 * 0.5**(r*(w1[idx]-w1[test_idx])) * \
        #                           (1-0.5**(r*w1[test_idx])) / (1-0.5**(r*w1[idx]))
        #             weights[i] *= w2[idx] / (d2 + w2[idx]) 

        if w2[test_idx] == 0:
            raise Exception( "Probability of observing the given test point must be positive!")
        
        if log_base:
            weights = np.emath.logn(log_base, (d1+w1[idxs]-w1[test_idx])/d1 * \
                                                 0.5**(r*(w1[idxs]-w1[test_idx])) * (1-0.5**(r*w1[test_idx])) / (1-0.5**(r*w1[idxs])))
            weights += np.emath.logn(log_base, w2[idxs] / (d2 + w2[idxs] - w2[test_idx]))
        else:
            weights = (d1+w1[idxs]-w1[test_idx])/d1 * 0.5**(r*(w1[idxs]-w1[test_idx])) * \
                                    (1-0.5**(r*w1[test_idx])) / (1-0.5**(r*w1[idxs]))
            weights *= w2[idxs] / (d2 + w2[idxs])
                    
        if log_base:
            ln0 = np.max(weights)
            weights = np.power(log_base, weights-ln0)
        weights /= np.sum(weights)
        
        return weights
    
    def weighted_PI(self, test_idxs, w1, w2, alpha, allow_inf=True, log_base="auto"):
        n_test = len(test_idxs)
        pi = [[]]* n_test
        
        if log_base == "auto":
            log_base = len(self.M)
        
        if self.verbose: 
            print("Computing weighted prediction intervals for {} test points...".format(n_test))
        
        
        # re-standardize w1, w2 by exclude training sampling odds
        w1 = w1 / (1 - np.sum(w1[self.train_idxs]))
        w2 = w2 / (1 - np.sum(w2[self.train_idxs]))

        # Compute universal scaling parameter
        d1 = 1 - np.sum(w1[self.calib_idxs])
        d2 = 1 - np.sum(w2[self.calib_idxs])
        r = self.mw._get_scale(w1, self.calib_idxs, d1)
        
        for i in tqdm(range(n_test), desc="WPI", leave=True, position=0, 
                      disable = not self.progress):
            weights = self._weights_single(test_idxs[i], w1, w2, d1, d2, r, log_base)
            cweights = np.cumsum(weights[self.calib_order])
            est = self.Mhat[test_idxs[i]]
            
            if cweights[-2] < 1-alpha:
                pi[i] = [-np.inf, np.inf] if allow_inf else \
                        [est - self.st_calib_scores[-2], est + self.st_calib_scores[-2]]
            else:
                idx = np.argmax(cweights >= 1-alpha)
                pi[i] = [est - self.st_calib_scores[idx], est + self.st_calib_scores[idx]]
        
        return pi
    
    

    
    def standard_PI(self, test_idxs, alpha):
        """
        Caculate conformal prediction interval with marginal coverage.
        """
        n_test = len(test_idxs)
        n_calib = len(self.calib_scores)
        pi = [[]]* n_test
        
        if self.verbose: 
            print("Computing standard marginal prediction intervals for {} test points...".format(n_test))
            
        qhat = np.quantile(self.calib_scores, np.ceil((n_calib+1)*(1-alpha))/n_calib,
                            method='higher')

        for i in tqdm(range(n_test), desc="SPI", leave=True, position=0, 
                      disable = not self.progress):
            est = self.Mhat[test_idxs[i]]
            pi[i] = [est-qhat, est+qhat]
            
        return pi


