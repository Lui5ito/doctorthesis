#!/bin/bash

python optimise_Kfolds.py --case_number 5 --size 100 --dim 1 --seed 123 --num_folds 10 --l_bound 1e-8 --u_bound 10 --max_eval 100 --n_restarts 10