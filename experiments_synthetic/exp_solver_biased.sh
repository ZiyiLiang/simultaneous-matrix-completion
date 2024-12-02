#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_solver_biased.py $1 $2 $3 $4