#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_real.py $1 $2 $3 $4 $5