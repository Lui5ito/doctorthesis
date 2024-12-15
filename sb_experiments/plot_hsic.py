"""
We want to plot the HSIC with regard to the lengthscale for both the training and calibration data.
"""

import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import json
import pickle
import os
import s3fs
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Lengthscales to compute
    length_scale_list = np.round(np.linspace(0.1, 1, 10), 3)
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    # Which training data
    cases = [5]
    all_sample_sizes = [100]
    all_sample_dims = [1]
    all_sample_seeds = [123, 124, 125, 126, 127]

    # Which calibration data
    calibration_cases = [5]
    calibration_all_sample_sizes = [100]
    calibration_all_sample_dims = [1]
    calibration_all_sample_seeds = [321, 322, 323]
    calibration_all_alphas = [0.05]

    # Initilalise the plot
    figs, axs = plt.subplots(len(all_sample_seeds), len(calibration_all_sample_seeds), figsize=(15, 15))
    axs = axs.flatten()
    ax_index = 0

    for case_number in cases:
        for sample_size in all_sample_sizes:
            for sample_dim in all_sample_dims:
                for seed in all_sample_seeds:
                    for calibration_case_number in calibration_cases:
                        for calibration_sample_size in calibration_all_sample_sizes:
                            for calibration_sample_dim in calibration_all_sample_dims:
                                for calibration_seed in calibration_all_sample_seeds:
                                    for alpha in calibration_all_alphas:
                                        train_hsic = []
                                        calibration_hsic = []
                                        for variance_lengthscale in length_scale_list:
                                            # Retrieve all params path
                                            FOLDER_PATH_TRAIN_S3 = f"luisito/these/sb_experiments/sdp_simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
                                            FILE_PATH_TRAIN_S3_PARAMS = FOLDER_PATH_TRAIN_S3 + "all_parameters.json"

                                            # Check if the file already exists
                                            if not fs.exists(FILE_PATH_TRAIN_S3_PARAMS):
                                                print(f"File {FILE_PATH_TRAIN_S3_PARAMS} does not exists. Cannot proceed.")
                                                continue

                                            # Retrieve params
                                            with fs.open(FILE_PATH_TRAIN_S3_PARAMS, mode="r") as file_in:
                                                all_train_params = json.load(file_in)
                                            train_hsic.append(all_train_params["output_parameters"]["training_hsic"])
                    

                                            FOLDER_PATH_CAL_S3 = f"luisito/these/sb_experiments/sdp_simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/calibration/data_case_{calibration_case_number}/sample_shape_({calibration_sample_size},{calibration_sample_dim})/seed_{calibration_seed}/alpha_{alpha}/"
                                            FILE_PATH_CAL_S3 = FOLDER_PATH_CAL_S3 + "all_parameters.json"

                                            # Check if the file already exists
                                            if not fs.exists(FILE_PATH_CAL_S3):
                                                print(f"File {FILE_PATH_CAL_S3} does not exists. Cannot proceed.")
                                                continue

                                            with fs.open(FILE_PATH_CAL_S3, mode="r") as file_in:
                                                all_cal_params = json.load(file_in)
                                            calibration_hsic.append(all_cal_params["output_parameters"]["calibration_hsic"])

                                        # Plot every time we finish all set of lengthscales.

                                        axs[ax_index].scatter(length_scale_list, train_hsic, color="blue", label="Training Data", alpha=0.7, marker='o', s=40)
                                        axs[ax_index].scatter(length_scale_list, calibration_hsic, color="red", label="Calibration Data", alpha=0.7, marker='x', s=40)
                                        # Add vertical lines stopping at the maximum points
                                        if train_hsic:  # Ensure train_hsic is not empty
                                            max_train_hsic_index = np.argmax(train_hsic)
                                            max_train_x = length_scale_list[max_train_hsic_index]
                                            max_train_y = train_hsic[max_train_hsic_index]
                                            axs[ax_index].plot([max_train_x, max_train_x], [0, max_train_y], color="blue", linestyle="--", alpha=0.8, label="Max Train HSIC")

                                        if calibration_hsic:  # Ensure calibration_hsic is not empty
                                            max_cal_hsic_index = np.argmax(calibration_hsic)
                                            max_cal_x = length_scale_list[max_cal_hsic_index]
                                            max_cal_y = calibration_hsic[max_cal_hsic_index]
                                            axs[ax_index].plot([max_cal_x, max_cal_x], [0, max_cal_y], color="red", linestyle="--", alpha=0.8, label="Max Cal HSIC")

                                        axs[ax_index].set_title(f"Training seed: {seed}; Calibration seed: {calibration_seed}")
                                        axs[ax_index].set_xlabel('Lengthscales')
                                        axs[ax_index].set_ylabel('e-HSIC')
                                        axs[ax_index].grid(True, linestyle='--', alpha=0.5)
                                        axs[ax_index].set_xticks(length_scale_list)
                                        axs[ax_index].legend(loc="upper right", fontsize="small", framealpha=0.8)
                                        ax_index += 1
    
    # Finish the figure and save
    figs.tight_layout()

    FILE_PATH_OUT_S3 = "luisito/these/sb_experiments/images/" + "e-HISC_vs_Lengthscales.pdf"
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        figs.savefig(file_out, format="pdf", transparent=True, dpi=600)

                        
