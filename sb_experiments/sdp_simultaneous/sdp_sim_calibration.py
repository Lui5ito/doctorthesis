"""
From a trained SDP model, we calibrate:
- On the calibration dataset, we compute:
    - lambda_hat, the empirircal quantile of the scores.
    - the energy-HSIC computed on the calibration data.
"""

import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import json
import pickle
import os
import s3fs


if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Lengthscales to compute
    length_scale_list = np.append(np.array([1e-6, 1e-5, 1e-4, 1e-3]), np.round(np.linspace(0.01, 1, 100), 3))
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    # Which training data
    cases = [5]
    all_sample_sizes = [100]
    all_sample_dims = [1]
    all_sample_seeds = [123, 124, 125, 126, 127]

    for case_number in cases:
        for sample_size in all_sample_sizes:
            for sample_dim in all_sample_dims:
                for seed in all_sample_seeds:
                    for variance_lengthscale in length_scale_list:
                        # Retrieve sdp model path
                        FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/sdp_simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
                        FILE_PATH_IN_S3_MODEL = FOLDER_PATH_IN_S3 + "sdp_model.pkl"

                        # Check if the file already exists
                        if not fs.exists(FILE_PATH_IN_S3_MODEL):
                            print(f"File {FILE_PATH_IN_S3_MODEL} does not exists. Cannot proceed.")
                            continue

                        # Retrieve sdp model
                        with fs.open(FILE_PATH_IN_S3_MODEL, mode="rb") as file_in:
                            sdp_model = pickle.load(file_in)
                        
                        # Which calibration data
                        calibration_cases = [5]
                        calibration_all_sample_sizes = [100]
                        calibration_all_sample_dims = [1]
                        calibration_all_sample_seeds = [321, 322, 323]
                        calibration_all_alphas = [0.05]
                        for calibration_case_number in calibration_cases:
                            for calibration_sample_size in calibration_all_sample_sizes:
                                for calibration_sample_dim in calibration_all_sample_dims:
                                    for calibration_seed in calibration_all_sample_seeds:
                                        for alpha in calibration_all_alphas:
                                            FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/sdp_simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/calibration/data_case_{calibration_case_number}/sample_shape_({calibration_sample_size},{calibration_sample_dim})/seed_{calibration_seed}/alpha_{alpha}/"

                                            # Retrieve data path
                                            FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{calibration_case_number}/sample_shape_({calibration_sample_size},{calibration_sample_dim})/seed_{calibration_seed}/"
                                            FILE_PATH_IN_S3 = FOLDER_PATH_IN_S3 + "data.npz"

                                            # Check if the file already exists
                                            if not fs.exists(FILE_PATH_IN_S3):
                                                print(f"File {FILE_PATH_IN_S3} does not exists. Cannot proceed.")
                                                continue

                                            # Retrieve data
                                            with fs.open(FILE_PATH_IN_S3, mode="rb") as file_in:
                                                data = np.load(file_in)
                                                X_calibration = data["X"]
                                                y_calibration = data["y"]
                                            
                                            # Calibrate the SDP model
                                            sdp_model.calibrate(X=X_calibration, y=y_calibration, alpha=alpha, metric_list=["energy_hsic"])

                                            lambdahat = sdp_model.lambdahat
                                            calibration_hsic = sdp_model.metrics["energy_hsic"]

                                            # Save SDP params
                                            FILE_PATH_OUT_S3_PARAMS = FOLDER_PATH_OUT_S3 + "all_parameters.json"
                                            all_parameters = {
                                                "calibration_data": {
                                                    "data_case": case_number,
                                                    "shape": (sample_size, sample_dim),
                                                    "seed": seed,
                                                },
                                                "input_parameters": {
                                                    "alpha": alpha,
                                                },
                                                "output_parameters": {
                                                    "calibration_hsic": calibration_hsic,
                                                    "lambdahat": lambdahat,
                                                },
                                            }
                                            with fs.open(FILE_PATH_OUT_S3_PARAMS, mode="w") as file_out:
                                                json.dump(all_parameters, file_out)
                                            
                                            # Save whole sdp model
                                            FILE_PATH_OUT_S3_MODEL = FOLDER_PATH_OUT_S3 + "sdp_model_calibrated.pkl"
                                            with fs.open(FILE_PATH_OUT_S3_MODEL, mode="wb") as file_out:
                                                pickle.dump(sdp_model, file_out)

                        
