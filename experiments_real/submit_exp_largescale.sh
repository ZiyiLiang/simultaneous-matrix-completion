#!/bin/bash

# Parameters
#K_LIST=(2 3 4 5 6)
K_LIST=(5)
#N_CAL_LIST=(1000 1500 2000 2500 3000)
N_CAL_LIST=(2000)
#SEED_LIST=$(seq 1 10)
SEED_LIST=(1)

# Slurm parameters
MEMO=20G                            # Memory required 
TIME=00-02:00:00                    # Time required 
CORE=6                              # Cores required 

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CORE" --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs/largescale"
OUT_DIR="results/largescale"

mkdir -p $LOGS
mkdir -p $OUT_DIR

comp=0
incomp=0

mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for N_CAL in "${N_CAL_LIST[@]}"; do 
        for K in "${K_LIST[@]}"; do     
            JOBN="ncal"$N_CAL"_k"$K"_seed"$SEED
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
            SCRIPT="exp_largescale.sh $N_CAL $K $SEED"
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
