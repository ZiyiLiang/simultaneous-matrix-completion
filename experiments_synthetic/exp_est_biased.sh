#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_est_biased.py $1 $2 $3 $4