# Structured Conformal Inference for Matrix Completion with Applications to Group Recommender Systems

This software repository provides a software implementation of the methods described in the following paper:
>"Structured Conformal Inference for Matrix Completion with Applications to Group Recommender Systems"\
>Ziyi Liang, Tianmin Xie, Xin Tong, Matteo Sesia\
>arXiv preprint <https://arxiv.org/pdf/2404.17561>


## Paper abstract
We develop a conformal inference method to construct joint prediction regions for structured groups of missing entries in a sparsely observed matrix, with particular focus on groups drawn from the same column. The method is model-agnostic and can be combined with any black-box matrix completion algorithm. In the context of recommender systems, for example, it is useful to quantify uncertainty in the ratings that all members of a group would assign to the same item, enabling more informed decisions when individual preferences may conflict. Unlike existing conformal techniques that estimate uncertainty for one entry at a time, our approach provides group-level guarantees by assembling calibration data with matching structure. To achieve this, we introduce a generalized weighted conformalization framework that addresses the lack of exchangeability induced by structured calibration, along with computational strategies that make the method practical at scale. We demonstrate the effectiveness of our approach through synthetic experiments under various missing-data mechanisms and applications to MovieLens datasets. 


## Contents
 - `smc/` Python package implementing our methods and some alternative benchmarks.
 - `third_party/` Third-party Python packages imported by our package.
 - `experiments_real/` Codes to replicate the figures for the experiments with the MovieLens datasets.
 - `experiments_synthetic/` Codes to replicate the figures for the synthetic experiments discussed in the accompanying paper.
 - `notebooks/` Contains Jupyter notebooks with introductory usage examples and `wallenius_plot.nb`, a Mathematica notebook (Mathematica v13.2; requires no external packages) to reproduce the plots in Appendix A2.3 of the accompanying paper.

### Results Reproduction
To facilitate immediate reproduction of the figures and tables in the manuscript without re-running the full computational experiments, we provide pre-computed results in the following compressed files:
 - `experiments_synthetic/results_hpc/results/results_synthetic.zip`
 - `experiments_real/results_hpc/results/results_real.zip`

### Dependency Management
To facilitate the reproduction of our computational environment, we provide several specification files:
 - `dependencies.txt` & `R_dependencies.txt` Summarize the major Python and R prerequisites in plain text.
 - `environment.yml` For automatic reconstruction of the Python environment via Conda/Mamba.
 - `renv.lock` To ensure R package version consistency using the `renv` framework.


## Data

### MovieLens 100K

Download from: https://grouplens.org/datasets/movielens/100k/

### MovieLens 10M (for scalability experiments)

Download from: https://grouplens.org/datasets/movielens/10m/

No additional preprocessing is required. The experiment scripts will load and process the data automatically.


## Installation for NCF Solver

The Neural Collaborative Filtering (NCF) solver (`ncf_solve()`) implemented in `smc/solvers.py` is utilized exclusively in the synthetic experiments exploring performance of the proposed framework under different matrix completion solvers (`experiments_synthetic/exp_solver_biased.py` and `experiments_synthetic/exp_solver_uniform.py`). 

Replicating these specific experiments requires the following dependencies (which are included in `dependencies.txt` and `environment.yml`):
* `LibRecommender` (provides the NCF implementation)
* `TensorFlow` (Dependency of `LibRecommender`)
* `Gensim` (Dependency of `LibRecommender`)
* `PyTorch` (Dependency of `LibRecommender`)

For detailed installation instructions and platform-specific guidance, please refer to the official [LibRecommender Installation Guide](https://librecommender.readthedocs.io/en/latest/installation.html). If you do not intend to replicate the NCF-based synthetic experiments, these dependencies can be safely ignored, and the related code in `smc/solvers.py` can be removed.