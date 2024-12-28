#!/bin/bash

#python data/generate_data.py --cases 8 --sizes 100 200 --seeds 123 321
#python gp/train_gp.py --cases 8 --sizes 100 200 --seeds 123
#python data/generate_data.py --cases 8 --sizes 300 1000 2000 --seeds 987

python optimise_lengthscale/optimise_Kfolds.py --case_number 8 --size 100 --dim 1 --seed 123 --num_folds 10 --l_bound 1e-8 --u_bound 10 --max_eval 100 --n_restarts 10
python optimise_lengthscale/optimise_Kfolds.py --case_number 8 --size 200 --dim 1 --seed 123 --num_folds 10 --l_bound 1e-8 --u_bound 10 --max_eval 100 --n_restarts 10
