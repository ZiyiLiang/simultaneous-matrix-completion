#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_mnar_linear.py $1 $2 $3 $4 $5 $6
