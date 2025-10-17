#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cmc

python3 exp_largescale.py $1 $2 $3