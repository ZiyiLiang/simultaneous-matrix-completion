#!/bin/bash

# Parameters
# N1_LIST=(100)
# N2_LIST=(100)
# R_TRUE_LIST= (3)
# R_GUESS_LIST=(3)
# PROB_LIST=(0.2)
# GN_LIST=(0.1 0.2 0.3 0.4 0.5)
# GM_LIST=(0.9)
# NM_LIST=(step)
# A_LIST=(1)     # Values of a, b do not matter if model is 'step'
# B_LIST=(1)     
# MU_LIST=$(seq 1 2 30)  # Value of mu do not matter if model is 'beta'
# SEED_LIST=$(seq 1 2)


test job
N1_LIST=(100)
N2_LIST=(100)
R_TRUE_LIST= (3)
R_GUESS_LIST=(3)
PROB_LIST=(0.2)
GN_LIST=(0.5)
GM_LIST=(0.9)
NM_LIST=(step)
A_LIST=(1)     # Values of a, b do not matter if model is 'step'
B_LIST=(1)     
MU_LIST=$(seq 30)  # Value of mu do not matter if model is 'beta'
SEED_LIST=$(seq 1)


# Slurm parameters
MEMO=12G                             # Memory required (12 GB)
TIME=00-06:00:00                    # Time required (6 h)
CORE=1                              # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Create directory for log files
LOGS="logs/exp_residual_hpm"
mkdir -p $LOGS

    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    r_true = int(sys.argv[3])
    r_guess = int(sys.argv[4])
    prob_obs = float(sys.argv[5])
    gamma_n = float(sys.argv[6])
    gamma_m = float(sys.argv[7])
    noise_model = sys.argv[8]
    a = int(sys.argv[9])
    b = int(sys.argv[10])
    mu = float(sys.argv[11])
    seed = int(sys.argv[12])

OUT_DIR="results/exp_hpm"
mkdir -p $OUT_DIR
for SEED in $SEED_LIST; do
    for N1 in "${N1_LIST[@]}"; do
        for N2 in "${N2_LIST[@]}"; do
            for R_TRUE in "${R_TRUE_LIST[@]}"; do
                for R_GUESS in "${R_GUESS_LIST[@]}"; do
                    for PROB in "${PROB_LIST[@]}"; do
                        for GN in "${GN_LIST[@]}"; do
                            for GM in "${GM_LIST[@]}"; do
                                for A in "${A_LIST[@]}"; do
                                    for B in "${B_LIST[@]}"; do
                                        for MU in "${MU_LIST[@]}"; do
                                            for NM in "${NM_LIST[@]}"; do
                                                JOBN=$N1"by"$N2"_rtrue"$R_TRUE"_rguess"$R_GUESS"_prob"$PROB"_gn"$GN"_gm"$GM"_"$NM"_a"$A"_b"$B"_mu"$MU"_seed"$SEED
                                                OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                                                COMPLETE=0
                                                #ls $OUT_FILE
                                                if [[ -f $OUT_FILE ]]; then
                                                COMPLETE=1
                                                fi

                                                if [[ $COMPLETE -eq 0 ]]; then
                                                # Script to be run
                                                SCRIPT="exp_residual_hpm.sh $N1 $N2 $R_TRUE $R_GUESS $PROB $GN $GM $NM $A $B $MU $SEED"
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
            done
        done
    done
done
