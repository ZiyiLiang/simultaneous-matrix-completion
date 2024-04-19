#!/bin/bash

# Parameters
R_LIST=(3 7)
SEED_LIST=$(seq 1 100)
DATASET="movielens"
#EST=1
EST=0
#FULL_MISS=1
FULL_MISS=0
#R_LIST=(7)
#SEED_LIST=(1)

# Slurm parameters
MEMO=20G                             # Memory required (1 GB)
TIME=00-02:30:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs/"
if [ $EST -eq 0 ]; then
  LOGS="logs/oracle_"$DATASET
  OUT_DIR="results/oracle_"$DATASET
else
  LOGS="logs/est_"$DATASET
  OUT_DIR="results/est_"$DATASET
fi

if [ $FULL_MISS -eq 1 ]; then
  LOGS="${LOGS}_fullmiss"
  OUT_DIR="${OUT_DIR}_fullmiss"
fi

mkdir -p $LOGS

comp=0
incomp=0

mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for R in "${R_LIST[@]}"; do
        JOBN="r"$R"_seed"$SEED
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
        SCRIPT="exp_real.sh $R $DATASET $EST $FULL_MISS $SEED"
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

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"
