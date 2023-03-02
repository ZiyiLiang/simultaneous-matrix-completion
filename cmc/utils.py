import numpy as np   
import matplotlib.pyplot as plt


def plot_before_after_mask(M, mask, vmin=-5, vmax=5):
    # Plot the masked and unmasked matrix side by side
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    # imshow plots matrices 
    axs[0].imshow(M, vmin=vmin, vmax=vmax)
    axs[0].title.set_text('Unmasked matrix')

    plot_masked = -100*(1-mask) + np.multiply(M,mask)
    axs[1].imshow(plot_masked,vmin=vmin, vmax=vmax)
    axs[1].title.set_text('Masked matrix')
    plt.show()


def evaluate_mse(M, Mhat, mask):
    """
    Calculate RMSE on all unobserved entries in mask, for true matrix UVᵀ.
    
    Args
    ------
    M:      Ground Truth matrix
    M_hat:  Estimated matrix 
    mask:   Bernoulli masking with 1 indicating observed, 0 unobserved
    
    Returns:
    --------
    rmse:   Mean squared error over all unobserved entries
    """
    pred = np.multiply(Mhat, mask)
    truth = np.multiply(M, mask)
    cnt = np.sum(mask)
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