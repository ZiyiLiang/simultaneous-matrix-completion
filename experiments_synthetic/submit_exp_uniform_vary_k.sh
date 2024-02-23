#!/bin/bash

# Parameters
N1=200
N2=200
R=5
EXP="vary_k" 
MU_LIST==(10, 20, 30)
SEED_LIST=$(seq 1 250)

# Slurm parameters
MEMO=12G                             # Memory required (12 GB)
TIME=00-02:00:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs/exp_uniform"
mkdir -p $LOGS

OUT_DIR="results/exp_uniform"
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for MU in $MU_LIST; do
        JOBN=$EXP"_"$N1"by"$N2"_r"$R"_mu"$MU"_seed"$SEED
        OUT_FILE=$OUT_DIR"/"$JOBN".txt"
        COMPLETE=0
        #ls $OUT_FILE
        if [[ -f $OUT_FILE ]]; then
        COMPLETE=1
        fi

        if [[ $COMPLETE -eq 0 ]]; then
        # Script to be run
        SCRIPT="exp_uniform.sh $N1 $N2 $R $EXP $MU $SEED"
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