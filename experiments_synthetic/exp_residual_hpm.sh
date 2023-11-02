#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_residual_hpm.py $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12