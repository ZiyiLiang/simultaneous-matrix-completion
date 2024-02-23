#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_uniform.py $1 $2 $3 $4 $5 $6 $7 $8
