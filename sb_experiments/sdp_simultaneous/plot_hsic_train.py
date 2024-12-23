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
    length_scale_list = np.append(np.array([1e-6, 1e-5, 1e-4, 1e-3]), np.round(np.linspace(0.01, 1, 100), 3))
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    # Which training data
    cases = [5]
    all_sample_sizes = [100]
    all_sample_dims = [1]
    all_sample_seeds = [123]

    # Initilalise the plot
    figs, axs = plt.subplots(len(all_sample_seeds), 1, figsize=(12, 10))
    #axs = axs.flatten()
    ax_index = 0

    for case_number in cases:
        for sample_size in all_sample_sizes:
            for sample_dim in all_sample_dims:
                for seed in all_sample_seeds:
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

                    # Plot every time we finish all set of lengthscales.

                    axs.scatter(length_scale_list, train_hsic, color="blue", label="Training Data", alpha=0.7, marker='o', s=40)
                    # Add vertical lines stopping at the maximum points
                    if train_hsic:  # Ensure train_hsic is not empty
                        max_train_hsic_index = np.argmax(train_hsic)
                        max_train_x = length_scale_list[max_train_hsic_index]
                        max_train_y = train_hsic[max_train_hsic_index]
                        axs.plot([max_train_x, max_train_x], [0, max_train_y], color="blue", linestyle="--", alpha=0.8, label="Max Train HSIC")

                    if calibration_hsic:  # Ensure calibration_hsic is not empty
                        max_cal_hsic_index = np.argmax(calibration_hsic)
                        max_cal_x = length_scale_list[max_cal_hsic_index]
                        max_cal_y = calibration_hsic[max_cal_hsic_index]
                        axs[ax_index].plot([max_cal_x, max_cal_x], [0, max_cal_y], color="red", linestyle="--", alpha=0.8, label="Max Cal HSIC")

                    axs.set_title(f"Training seed: {seed}")
                    axs.set_xlabel('Lengthscales')
                    axs.set_ylabel('e-HSIC')
                    axs.grid(True, linestyle='--', alpha=0.5)
                    axs.set_xticks(length_scale_list)
                    axs.tick_params(axis='x', labelsize=6)
                    plt.xticks(rotation=45)
                    axs.legend(loc="upper right", fontsize="small", framealpha=0.8)
                    ax_index += 1
    
    # Finish the figure and save
    figs.suptitle('e-HSIC vs lengthscales for multiple seeds and both calibration and training data.', x=0.5, y=0.99, size = 16, weight = 'bold')
    figs.tight_layout()

    FILE_PATH_OUT_S3 = "luisito/these/sb_experiments/images/" + f"e-HISC_vs_Lengthscales_training_seed_{seed}.pdf"
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        figs.savefig(file_out, format="pdf", transparent=True, dpi=600)

                        
