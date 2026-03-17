# Third-Party Code

This directory contains code adapted from external sources to support specific experimental functions.

## missingness_estimation.py
- **Source:** [yugjerry/conf-mc](https://github.com/yugjerry/conf-mc/)
- **Version:** Commit `bdfa590f95fc32f5952cf0ed868ef88836529083`
- **Purpose:** Used for estimating the probability of missingness in matrix completion using Spectral Projected Gradient methods.
- **Modifications:** Consolidates SPG solver, Euclidean projections, and Nuclear norm logic into a single file.