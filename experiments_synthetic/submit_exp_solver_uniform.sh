#!/bin/bash

# Parameters
#MU_LIST=(0)
MU_LIST=(15)
SEED_LIST=$(seq 2 150)
#SOLVER_LIST=("svt" "nnm" "ncf")
SOLVER_LIST=("ncf")
#SEED_LIST=(1)

# Slurm parameters
MEMO=1G                             # Memory required (1 GB)
TIME=00-01:00:00                    # Time required (2 h) 
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs/exp_solver_uniform"
mkdir -p $LOGS

comp=0
incomp=0

OUT_DIR="results/exp_solver_uniform"
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for MU in $MU_LIST; do
        for SOLVER in "${SOLVER_LIST[@]}"; do
            JOBN=$SOLVER"_mu"$MU"_seed"$SEED
            OUT_FILE=$OUT_DIR"/"$JOBN".txt"
            COMPLETE=0
            #ls $OUT_FILE
            if [[ -f $OUT_FILE ]]; then
            COMPLETE=1
            ((comp++))
            fi

            if [[ $COMPLETE -eq 0 ]]; then
            ((incomp++))
            # Script to be run
            SCRIPT="exp_solver_uniform.sh $SOLVER $MU $SEED"
            # Define job name
            OUTF=$LOGS"/"$JOBN".out"
            ERRF=$LOGS"/"$JOBN".err"
            # Assemble slurm order for this job
            ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
            # Print order
            echo $ORD
            # Submit order
            $ORD
            fi
        done
    done
done

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"
