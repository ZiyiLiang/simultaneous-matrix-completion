import numpy as np   
import matplotlib.pyplot as plt
import pandas as pd


def plot_before_after_mask(M, mask, vmin=-5, vmax=5):
    # Plot the masked and unmasked matrix side by side
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)

    # imshow plots matrices 
    im1 = axs[0].imshow(M, vmin=vmin, vmax=vmax)
    axs[0].title.set_text('Ground truth matrix')
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    plot_masked = -100*(1-mask) + np.multiply(M,mask)
    im2 = axs[1].imshow(plot_masked,vmin=vmin, vmax=vmax)
    axs[1].title.set_text('Masked matrix')
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    plt.show()

    

def error_heatmap(M, Mhat, mask, vmin=None, vmax=None, cmap=None):
    pred = np.multiply(Mhat, mask)
    truth = np.multiply(M, mask)
    residual = np.abs(pred-truth)
    residual = residual / np.max(residual)
    
    if cmap is None:
        cmap = plt.cm.get_cmap('viridis').reversed()
    if vmin is None: 
        vmin = 0
    if vmax is None: 
        # filter out some extreme values for better graph
        vmax = np.quantile(residual.flatten(), 0.95,method='higher')
    
    plt.figure(figsize=(6,4))
    plt.imshow(residual, cmap=cmap,vmin=vmin, vmax=vmax)
    plt.title("Absolute residuals")
    plt.colorbar()
    plt.show()



def evaluate_mse(M, Mhat, idx):
    """
    Calculate RMSE on the given index set, for true matrix UVᵀ.
    
    Args
    ------
    M:      Ground Truth matrix
    M_hat:  Estimated matrix 
    idx:    Binary matrix where target indexed are labelled 1, 0 otherwise
    
    Returns:
    --------
    rmse:   Mean squared error over all unobserved entries
    """
    pred = np.multiply(Mhat, idx)
    truth = np.multiply(M, idx)
    cnt = np.sum(idx)
    return np.linalg.norm(pred - truth, "fro") ** 2 / cnt

def theoretical_bound(m, n, r, C=1):
    """ 
    Approximates the minimal training samples needed for exact matrix recovery,
    following the results from Theoem 1.1 in Candes and Recht[2008].
    """

    n = max(m, n)
    if r <= n ** (1 / 5):
        m = C * n ** (6 / 5) * r * np.log(n)
    else:
        m = C * n ** (5 / 4) * r * np.log(n)

    return np.ceil(m)


def evaluate_CI(lower, upper, M, idxs_test, label=None):
    label = str(label)+'_' if label else ''
    val = M[idxs_test]
    coverage = np.mean((lower < val) & (upper > val))
    size = np.mean(upper - lower)
    
    results = pd.DataFrame({})
    results[label+"Coverage"] = [coverage]
    results[label+"Size"] = [size]
    return results


def evaluate_pairedCI(lower, upper, M, idxs_test, is_inf=None, method=None):
    val = M[idxs_test]
    n_test_pair = len(idxs_test[0]) // 2
    
    covered = (lower < val) & (upper > val)
    pair_covered = [covered[2*i] * covered[2*i+1] for i in range(n_test_pair)]
    
    pair_coverage = np.mean(pair_covered)
    coverage = np.mean(covered)
    size = np.mean(upper - lower)
    results = pd.DataFrame({})
    results["Pair_coverage"] = [pair_coverage]
    results["Coverage"] = [coverage]
    results["Size"] = [size]
    if type(is_inf) != type(None):
        results["Inf_prop"] = [np.mean(is_inf)]
    if method:
        results["Method"] = [method]
    return results

