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
 - `experiments_real/` Codes to replicate the figures for the experiments with the MovieLens data sets.
 - `experiments_synthetic/` Codes to replicate the figures for the synthetic experiments discussed in the accompanying paper.
 - `notebooks/` Jupyter notebooks with introductory usage examples.
 - `dependencies.txt` Prerequisites with version number for the `smc` package. 


## Data

### MovieLens 100K

Download from: https://grouplens.org/datasets/movielens/100k/

### MovieLens 10M (for scalability experiments)

Download from: https://grouplens.org/datasets/movielens/10m/

No additional preprocessing is required. The experiment scripts will load and process the data automatically.
