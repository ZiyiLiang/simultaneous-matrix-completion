#!/bin/bash

# Parameters
#SCALE_LIST=$(seq 0 0.5 2)
GENRE_LIST=("Drama" "Comedy" "Thriller" "Romance" "Action" "Adventure" "Children's" "Crime" "Horror" "Sci-Fi" "War" "Musical" "Documentary" "Mystery") 
SEED_LIST=$(seq 1 150)
FULL_MISS=1
#SEED_LIST=(1)
CAL_LIST=(2000)


# Slurm parameters
MEMO=13G                             # Memory required (1 GB)
TIME=00-00:40:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs/exp_movielens_conditional"
OUT_DIR="results/exp_movielens_conditional"

if [ $FULL_MISS -eq 1 ]; then
  LOGS="${LOGS}_fullmiss"
  OUT_DIR="${OUT_DIR}_fullmiss"
fi

mkdir -p $LOGS
mkdir -p $OUT_DIR

comp=0
incomp=0

for SEED in $SEED_LIST; do
    for GENRE in "${GENRE_LIST[@]}"; do
        for CAL in "${CAL_LIST[@]}"; do
            JOBN=$GENRE"_cal"$CAL"_seed"$SEED
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
            SCRIPT="exp_real_conditional.sh $GENRE $CAL $FULL_MISS $SEED"
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
