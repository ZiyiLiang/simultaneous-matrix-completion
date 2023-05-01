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


def evaluate_PI(PI, x):
    coverage = np.mean([x[i] >= PI[i][0] and x[i] <= PI[i][1] for i in range(len(x))])
    size = np.mean([PI[i][1] - PI[i][0] for i in range(len(x))])
    
    results = pd.DataFrame({})
    results["Coverage"] = [coverage]
    results["Size"] = [size]
    return results


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


def residual_hist(M, Mhat, train_mask, calib_mask, test_mask,vmin=0, vmax=None):
    residual = np.abs(M-Mhat)
    #residual = residual / np.max(residual)   # normalize residuals
    res_train = residual[np.where(train_mask==1)]
    res_calib = residual[np.where(calib_mask==1)]
    res_test = residual[np.where(test_mask==1)]
    
    if vmax == None:
        vmax = np.quantile(residual.flatten(), 0.95,method='higher')
    bins = np.linspace(0, vmax, 100)

    plt.hist(res_train, bins, alpha=0.5, label='train')
    plt.hist(res_calib, bins, alpha=0.5, label='calib')
    plt.hist(res_test, bins, alpha=0.5, label='test')
    plt.legend(loc='upper right')
    plt.show()