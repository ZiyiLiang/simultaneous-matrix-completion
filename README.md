# Structured Conformal Inference for Matrix Completion with
Applications to Group Recommender Systems

This software repository provides a software implementation of the methods described in the following paper:
>"Structured Conformal Inference for Matrix Completion with Applications to Group Recommender Systems"
>Ziyi Liang, Tianmin Xie, Xin Tong, Matteo Sesia
>arXiv preprint <https://arxiv.org/pdf/2404.17561>


## Paper abstract
We develop a conformal inference method to construct joint confidence regions for structured groups of missing entries within a sparsely observed matrix. This method is useful to
provide reliable uncertainty estimation for group-level collaborative filtering; for example, it
can be applied to help suggest a movie for a group of friends to watch together. Unlike standard conformal techniques, which make inferences for one individual at a time, our method
achieves stronger group-level guarantees by carefully assembling a structured calibration data
set mimicking the patterns expected among the test group of interest. We propose a generalized weighted conformalization framework to deal with the lack of exchangeability arising from
such structured calibration, and in this process we introduce several innovations to overcome
computational challenges. The practicality and effectiveness of our method are demonstrated
through extensive numerical experiments and an analysis of the MovieLens 100K data set.

## Contents
 - `smc/` Python package implementing our methods and some alternative benchmarks.
 - `third_party/` Third-party Python packages imported by our package.
 - `experiments_real/` Codes to replicate the figures for the experiments with the MovieLens 100K data set.
 - `experiments_synthetic/` Codes to replicate the figures for the synthetic experiments discussed in the accompanying paper.
 - `notebooks/` Jupyter notebooks with introductory usage examples.
 - `dependencies.txt` Prerequisites with version number for the `smc` package.
