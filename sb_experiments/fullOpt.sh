#!/bin/bash

CASE=5
SIZE=20
DIM=2
SEED=123
CAL_SEED=321
TEST_SEED=987
TEST_SIZE=300

python data/generate_data.py --cases $CASE --sizes $SIZE --dims $DIM --seeds $SEED $CAL_SEED
python gp/train_gp.py --cases $CASE --sizes $SIZE --dims $DIM --seeds $SEED
python data/generate_data.py --cases $CASE --sizes $TEST_SIZE --dims $DIM --seeds $TEST_SEED
python optimise_lengthscale/optimise_Kfolds.py --case_number $CASE --size $SIZE --dim $DIM --seed $SEED --num_folds 10 --l_bound 1e-8 --u_bound 10 --max_eval 100 --n_restarts 2
python optimise_lengthscale/afterTrain.py --train $CASE $SIZE $DIM $SEED --calibration $CASE $SIZE $DIM $CAL_SEED --test $CASE $TEST_SIZE $DIM $TEST_SEED