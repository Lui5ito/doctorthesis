#!/bin/bash

#python generate_data.py --size 100 --seeds 123 124 125 126 127 321 322 323
#python generate_data.py --size 300 --seeds 987 986 985

#python train_gp.py

python sdp_simultaneous.py
python sdp_sim_calibration.py
#python sdp_sim_inference.py
python plot_hsic.py