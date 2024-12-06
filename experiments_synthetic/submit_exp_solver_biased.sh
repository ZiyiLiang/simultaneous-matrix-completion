#!/bin/bash

# Parameters

SCALE_LIST=$(seq 0.10 0.1 0.20)
SD_LIST=(0.1)
#SCALE_LIST=$(seq 0.6 0.05 1.0)
#SEED_LIST=(1)
SEED_LIST=$(seq 1 100)
#SOLVER_LIST=("nnm")  # runtime for nnm is longer, set to 1h
SOLVER_LIST=("svt")   # runtime for svt is shorter, set to 30min


# Slurm parameters
MEMO=2G                             # Memory required (1 GB)
TIME=00-01:30:00                    # Time required (2 h)
#TIME=00-01:00:00
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs/exp_solver_biased"
mkdir -p $LOGS

comp=0
incomp=0

OUT_DIR="results/exp_solver_biased"
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for SCALE in $SCALE_LIST; do
        for SD in "${SD_LIST[@]}"; do
            for SOLVER in "${SOLVER_LIST[@]}"; do
                JOBN=$SOLVER"_scale"$SCALE"_sd"$SD"_seed"$SEED
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
                SCRIPT="exp_solver_biased.sh $SOLVER $SCALE $SD $SEED"
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
done

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"
