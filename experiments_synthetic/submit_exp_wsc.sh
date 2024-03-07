#!/bin/bash

# Parameters
N1=400
N2=400
R=8
EXP="wsc"
#SEED_LIST=$(seq 1 200)
SEED_LIST=(1)

# Slurm parameters
MEMO=2G                             # Memory required (1 GB)
TIME=00-01:30:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs/exp_conditional/"$EXP
mkdir -p $LOGS

comp=0
incomp=0

OUT_DIR="results/exp_conditional"$EXP
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    JOBN=$N1"by"$N2"_r"$R"_seed"$SEED
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
    SCRIPT="exp_conditional.sh $N1 $N2 $R $EXP $SEED"
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

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"
