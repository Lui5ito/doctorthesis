"""
We compute the SDP model by the following:
- Select a training dataset.
- Divide in K-folds.
- Train on the -k folds and save the AE and width (W) computed on the k-fold.
- Compute the HSIC on AE and W.
- 
"""

import universalbands as ub
from universalbands.metrics.energy_hsic import Energy_HSIC
import numpy as np
from sklearn.gaussian_process import kernels
from sklearn.model_selection import KFold
import json
import pickle
import os
import s3fs
from utils import load_file
import concurrent.futures


def process_fold(fold_data, X_train, y_train, theta_m, variance_lengthscale):
    # Define the model
    sdp_model = ub.UniversalFunctionAndBandsRegressor(
        mean_kernel=kernels.Matern(length_scale=theta_m["posterior_lengthscale"], length_scale_bounds=(1e-5, 1e5), nu=2.5),
        variance_kernel=kernels.Matern(length_scale=variance_lengthscale, length_scale_bounds=(1e-5, 1e5), nu=2.5),
        lambda2=lambda2,
        delta=delta,
        s=theta_m["posterior_training_norm"],
        problem=problem,
        checkSDP=False,
        verbose=True,
    )
    # Train the model
    sdp_model.train(X=X_train[train_index], y=y_train[train_index])
    hsic_value, abs_errors, half_width = sdp_model.compute_HSIC(X=X_train[test_index], y=y_train[test_index], extra=True)

    return abs_errors, half_width

def process_variance_lengthscale(kFold, variance_lengthscale, case_number, sample_size, sample_dim, seed, X_train, y_train, theta_m, fs):
    folded_AE = []
    folded_IW = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_fold, fold_data, X_train, y_train, theta_m, variance_lengthscale)
            for fold_data in enumerate(kFold.split(X_train))
        ]
        # Collect results from all folds
        for future in concurrent.futures.as_completed(futures):
            abs_errors, half_width = future.result()
            folded_AE.append(abs_errors)
            folded_IW.append(half_width)
        
    folded_AE = np.vstack(folded_AE)
    folded_IW = np.vstack(folded_IW)
    kfold_hsic = Energy_HSIC(abs_error=folded_AE, width=folded_IW).compute()

    # Load fully trained model and add kfold_hsic.
    fully_trained_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/ub_models/simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
    fully_trained_FILE_PATH_IN_S3 = fully_trained_FOLDER_PATH_IN_S3 + "sdp_model.pkl"
    fully_trained_sdp_model = load_file(fully_trained_FILE_PATH_IN_S3, fs)
    fully_trained_sdp_model.metrics["kfold_train_hsic"] = kfold_hsic

    with fs.open(fully_trained_FILE_PATH_IN_S3, mode="wb") as file_out:
        pickle.dump(fully_trained_sdp_model, file_out)




if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Lengthscales to compute
    #length_scale_list = np.append(np.array([1e-6, 1e-5, 1e-4, 1e-3]), np.round(np.linspace(0.01, 1, 100), 3))
    length_scale_list = np.round(np.linspace(0.01, 1, 5), 3)
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    number_of_splits = 2

    # Which training data
    cases = [5]
    all_sample_sizes = [100]
    all_sample_dims = [1]
    all_sample_seeds = [123]

    for case_number in cases:
        for sample_size in all_sample_sizes:
            for sample_dim in all_sample_dims:
                for seed in all_sample_seeds:
                    # Retrieve data path
                    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
                    X_train, y_train = load_file(FILE_PATH_IN_S3, fs)

                    # Retrieve gp model path
                    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"
                    theta_m = load_file(gp_FILE_PATH_IN_S3, fs)

                    kf = KFold(n_splits=number_of_splits, shuffle=True, random_state=42)

                    # Use ProcessPoolExecutor to parallelize the loop over length_scale_list
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        futures = [
                            executor.submit(process_variance_lengthscale, kf, variance_lengthscale, case_number, sample_size, sample_dim, seed, X_train, y_train, theta_m, fs)
                            for variance_lengthscale in length_scale_list
                        ]
                        # Wait for all tasks to complete
                        for future in concurrent.futures.as_completed(futures):
                            # You could also handle results here if needed
                            future.result()
                    print("All done.")

