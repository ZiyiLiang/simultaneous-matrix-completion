#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_est_uniform.py $1 $2 $3