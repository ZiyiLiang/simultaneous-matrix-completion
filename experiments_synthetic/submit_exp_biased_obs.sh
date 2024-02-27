#!/bin/bash

# Parameters
N1=500
N2=500
R=10
SCALE_LIST=$(seq 0.0 0.2 1.6)
SEED_LIST=$(seq 1 200)
#SCALE_LIST=(0.5)
#SEED_LIST=(1)

# Slurm parameters
MEMO=3G                             # Memory required (1 GB)
TIME=00-01:00:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs/exp_biased_obs"
mkdir -p $LOGS

comp=0
incomp=0

OUT_DIR="results/exp_biased_obs"
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for SCALE in $SCALE_LIST; do
        JOBN=$N1"by"$N2"_r"$R"_scale"$SCALE"_seed"$SEED
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
        SCRIPT="exp_biased_obs.sh $N1 $N2 $R $SCALE $SEED"
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
