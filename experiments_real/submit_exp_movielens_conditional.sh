#!/bin/bash

# Parameters
#SCALE_LIST=$(seq 0 0.5 2)
SCALE_LIST=(1 5 10)
SEED_LIST=$(seq 1 30)
#SEED_LIST=(1)
#CAL_LIST=(2000 3000)
CAL_LIST=(2000)


# Slurm parameters
MEMO=13G                             # Memory required (1 GB)
TIME=00-00:30:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs/exp_movielens_conditional"
mkdir -p $LOGS

comp=0
incomp=0

OUT_DIR="results/exp_movielens_conditional"
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for SCALE in "${SCALE_LIST[@]}"; do
        for CAL in "${CAL_LIST[@]}"; do
            JOBN="scale"$SCALE"_cal"$CAL"_seed"$SEED
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
            SCRIPT="exp_real_conditional.sh $SCALE $CAL $SEED"
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
