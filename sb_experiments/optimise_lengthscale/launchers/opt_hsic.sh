#!/bin/bash

# Common parameters
CASE_NUMBER=5
SIZE=100
DIM=1
L_BOUND=1e-8
U_BOUND=10
MAX_EVAL=100
N_RESTARTS=10

# Loop over different seeds for training
for SEED in 123 124 125 126 127; do
    # Training step
    echo "ECHO - Training seed: $SEED"
    python sdp_sim_optimise_hsic.py --on "train" --case_number $CASE_NUMBER --size $SIZE --dim $DIM --seed $SEED --l_bound $L_BOUND --u_bound $U_BOUND --max_eval $MAX_EVAL --n_restarts $N_RESTARTS

    # Calibration step for each seed
    for CAL_SEED in 321 322 323; do
        echo "ECHO - Calibration seed: $CAL_SEED with training seed: $SEED"
        python sdp_sim_optimise_hsic.py --on "calibration" --case_number $CASE_NUMBER --size $SIZE --dim $DIM --seed $SEED --l_bound $L_BOUND --u_bound $U_BOUND --max_eval $MAX_EVAL --n_restarts $N_RESTARTS --cal_case_number $CASE_NUMBER --cal_size $SIZE --cal_dim $DIM --cal_seed $CAL_SEED
    done
done