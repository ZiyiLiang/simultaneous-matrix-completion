#!/bin/bash

# Parameters
N1=200
N2=200
R_LIST=(1 2 3 4 6 8 10)
#MU_LIST=$(seq 0 3 30)
SEED_LIST=$(seq 1 50)
MU_LIST=(30)
#SEED_LIST=(0)

# Slurm parameters
MEMO=1G                             # Memory required (1 GB)
TIME=00-01:00:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs/exp_uniform"
mkdir -p $LOGS

comp=0
incomp=0

OUT_DIR="results/exp_uniform"
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for SCALE in $SCALE_LIST; do
        for R in "${R_LIST[@]}"; do
            JOBN="r"$R"_scale"$SCALE"_seed"$SEED
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
            SCRIPT="exp_uniform.sh $N1 $N2 $R $MU $SEED"
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
