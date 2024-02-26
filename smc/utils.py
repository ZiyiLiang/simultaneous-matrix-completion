import numpy as np   
import matplotlib as mpl 
import matplotlib.pyplot as plt
import pandas as pd


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



def evaluate_SCI(lower, upper, k, M, idxs_test, is_inf=None, method=None):
    """ This function evaluates the coverage over test queries
    """
    val = M[idxs_test]
    n_test_queries = len(idxs_test[0]) // int(k) 
    
    covered = np.array((lower <= val) & (upper >= val))
    query_covered = [np.prod(covered[k*i:k*(i+1)]) for i in range(n_test_queries)]
    
    query_coverage = np.mean(query_covered)
    coverage = np.mean(covered)
    size = np.mean(upper - lower)
    results = pd.DataFrame({})
    results["Query_coverage"] = [query_coverage]
    results["Coverage"] = [coverage]
    results["Size"] = [size]
    if type(is_inf) != type(None):
        results["Inf_prop"] = [np.mean(is_inf)]
    if method:
        results["Method"] = [method]
    return results

