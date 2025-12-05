import numpy as np   
import matplotlib as mpl 
import matplotlib.pyplot as plt
import pandas as pd
import pdb


def plot_before_after_mask(M, mask, vmin=None, vmax=None, bad_color='grey', figsize=(12,5)):
    if vmin is None:
        vmin = np.min(M)
    if vmax is None:
        vmax = np.max(M)
    
    # Plot the masked and unmasked matrix side by side
    fig, axs = plt.subplots(1,2, figsize=figsize)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)

    # imshow plots matrices 
    cmap = mpl.colormaps.get_cmap('viridis')
    cmap.set_bad(color=bad_color)
    im1 = axs[0].imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].title.set_text('Ground truth matrix')
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    # 0-valued entries in the mask are marked in grey
    plot_masked = M.copy()
    plot_masked[np.where(mask==0)]=np.nan
    im2 = axs[1].imshow(plot_masked,vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].title.set_text('Masked matrix')
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    plt.show()



def error_heatmap(M, Mhat, mask, vmin=None, vmax=None, cmap=None, bad_color='white',figsize=(6,4)):
    pred = np.multiply(Mhat, mask)
    truth = np.multiply(M, mask)
    residual = np.abs(pred-truth)
    #residual = residual / np.max(residual)
    
    if cmap is None:
        cmap = plt.cm.get_cmap('viridis').reversed()
        cmap.set_bad(color=bad_color)
    if vmin is None: 
        vmin = 0
    if vmax is None: 
        # filter out some extreme values for better graph
        vmax = np.quantile(residual[np.where(mask==1)].flatten(), 0.95,method='higher')
    
    residual[np.where(mask==0)]=np.nan
    plt.figure(figsize=figsize)
    plt.imshow(residual, cmap=cmap,vmin=vmin, vmax=vmax)
    plt.title("Absolute residuals")
    plt.colorbar()
    plt.show()



def evaluate_SCI(lower, upper, k, M, idxs_test, is_inf=None, 
                 metric='mean', method=None):
    """ This function evaluates the coverage over test queries
    """
    val = M[idxs_test]
    n_test_queries = len(idxs_test[0]) // int(k) 
    
    covered = np.array((lower <= val) & (upper >= val))
    query_covered = [np.prod(covered[k*i:k*(i+1)]) for i in range(n_test_queries)]
    query_coverage = np.mean(query_covered)
    coverage = np.mean(covered)
    sizes = upper - lower
    if metric == 'mean':
        size = np.mean(sizes)
    elif metric == 'median':
        size = np.median(sizes)
    elif metric == 'no_inf':
        if is_inf is None:
            print('Must provide inf locations for the no_inf metric.')
        else:
            query_no_inf = [not np.any(is_inf[k*i:k*(i+1)]) for i in range(n_test_queries)]
            idxs_no_inf = [element for element in query_no_inf for _ in range(int(k))]
        size = np.mean(sizes[idxs_no_inf])
    else:
        print(f'Unknown evaluation metric {metric}')
    results = pd.DataFrame({})
    results["Query_coverage"] = [query_coverage]
    results["Coverage"] = [coverage]
    results["Size"] = [size]
    results["metric"] = [metric]
    if type(is_inf) != type(None):
        query_is_inf = [np.any(is_inf[k*i:k*(i+1)]) for i in range(n_test_queries)]
        results["Inf_prop"] = [np.mean(query_is_inf)]
    if method:
        results["Method"] = [method]
    return results



def compute_error(M, Mhat, mask):
    # Predicted and true values based on mask
    pred = np.multiply(Mhat, mask)
    truth = np.multiply(M, mask)
    
    # Residuals
    residual = np.abs(pred - truth)
    squared_residual = (pred - truth) ** 2

    # Number of observed entries
    n = np.sum(mask)

    # Mean Absolute Error (MAE)
    mae = np.sum(residual) / n

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.sum(squared_residual) / n)

    # Relative Frobenius Norm Error
    frobenius_error = np.linalg.norm(pred - truth, 'fro') / np.linalg.norm(truth, 'fro')

    return mae, rmse, frobenius_error


def compute_extremeness_weights(M, rho=1.0, normalize=True):
    """
    Compute observation weights based on how extreme values are relative to the data distribution.
    More extreme values (farther from center, in the tails of the distribution) have
    higher probability of being observed.

    Parameters
    ----------
    M : ndarray
        The matrix values (possibly noisy observations).
    rho : float, optional
        Base of exponential weighting function controlling MNAR strength.
        - rho = 1: uniform weights (MAR), since 1^x = 1 for all x
        - rho > 1: extreme values have higher weights (MNAR)
        - Continuously transitions from MAR to MNAR as rho increases from 1
        Typical values: 1.0 (MAR), 1.1, 1.2, 1.5
    normalize : bool, optional
        Whether to normalize weights to sum to 1. Default is True.

    Returns
    -------
    w : ndarray
        Observation weights with the same shape as M.

    Notes
    -----
    The weight function measures extremeness relative to the data distribution:
    w_ij = rho^extremeness_ij
    where extremeness_ij = |M_ij - median(M)|

    This gives continuous control over the MNAR mechanism:
    - When rho=1: w_ij = 1 for all i,j (uniform sampling, MAR)
    - When rho>1: rho^(larger extremeness) > rho^(smaller extremeness)
      so more extreme values get exponentially higher weights
    """
    # Compute extremeness (distance from median)
    median_val = np.median(M)
    extremeness = np.abs(M - median_val)

    # Compute weights: rho^extremeness
    w = rho ** extremeness

    if normalize:
        w = w / np.sum(w)

    return w