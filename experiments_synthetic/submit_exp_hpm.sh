#!/bin/bash

# Parameters
# N1_LIST=(100 300 500)
# N2_LIST=(100 300 500)
# R_TRUE_LIST=(5 10 20)
# R_GUESS_LIST=(5 10 20)
# PROB_LIST=(0.1 0.3)
# SOLVER_LIST=(pmf)
# MODEL_LIST=(RFM)
# SEED_LIST=$(seq 1 4)

# test job
N1_LIST=(500)
N2_LIST=(500)
R_TRUE_LIST=(20)
R_GUESS_LIST=(20)
PROB_LIST=(0.1)
SOLVER_LIST=(pmf)
MODEL_LIST=(RFM)
SEED_LIST=$(seq 1 2)


# Slurm parameters
MEMO=12G                             # Memory required (12 GB)
TIME=00-04:00:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs/exp_hpm"
mkdir -p $LOGS

OUT_DIR="results/exp_hpm"
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for N1 in "${N1_LIST[@]}"; do
        for N2 in "${N2_LIST[@]}"; do
            for R_TRUE in "${R_TRUE_LIST[@]}"; do
                for R_GUESS in "${R_GUESS_LIST[@]}"; do
                    for PROB in "${PROB_LIST[@]}"; do
                        for SOLVER in "${SOLVER_LIST[@]}"; do
                            for MODEL in "${MODEL_LIST[@]}"; do
                                JOBN="n1"$N1"_n2"$N2"_rtrue"$R_TRUE"_rguess"$R_GUESS"_prob"$PROB"_"$SOLVER"_"$MODEL"_seed"$SEED
                                OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                                COMPLETE=0
                                #ls $OUT_FILE
                                if [[ -f $OUT_FILE ]]; then
                                COMPLETE=1
                                fi

                                if [[ $COMPLETE -eq 0 ]]; then
                                # Script to be run
                                SCRIPT="exp_hpm.sh $N1 $N2 $R_TRUE $R_GUESS $PROB $SOLVER $MODEL $SEED"
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
            done
        done
    done
done