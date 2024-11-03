#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_real_conditional.py $1 $2 $3