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
from multiprocessing import Pool
from functools import partial


def process_variance_lengthscale(variance_lengthscale, kFold, case_number, sample_size, sample_dim, seed, X_train, y_train, theta_m, lambda2, delta, problem):
    folded_AE = []
    folded_IW = []
    print(variance_lengthscale)

    for k, (train_index, test_index) in enumerate(kFold.split(X_train)):
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
        folded_AE.append(abs_errors)
        folded_IW.append(half_width)

    folded_AE = np.vstack(folded_AE)
    folded_IW = np.vstack(folded_IW)
    kfold_hsic = Energy_HSIC(abs_error=folded_AE, width=folded_IW).compute()

    return kfold_hsic




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Argparse for BOBYQA optimisation.')
    parser.add_argument('--number_of_folds', type=int)
    args = parser.parse_args()
    print(f"Starting optimisation for {args.number_of_folds} folds.")
    
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Lengthscales to compute 900
    list_lengthscales = np.round(np.linspace(1e-3, 1, 900), 4)
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    number_of_splits = args.number_of_folds
    num_processes = 60

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
                    X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)

                    # Retrieve gp model path
                    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"
                    theta_m = load_file(gp_FILE_PATH_IN_S3, fs)

                    kf = KFold(n_splits=number_of_splits, shuffle=True, random_state=42)

                    # Convert my process function from multiple to ONE argument.
                    process_variance_lengthscale_one_param = partial(process_variance_lengthscale, kFold=kf, case_number=case_number, sample_size=sample_size, sample_dim=sample_dim, seed=seed, X_train=X_train, y_train=y_train, theta_m=theta_m, lambda2=lambda2, delta=delta, problem=problem)

                    with Pool(processes=(num_processes)) as pool:
                        results = list(pool.imap(process_variance_lengthscale_one_param, list_lengthscales,))

                    # Save whole SDP model
                    for kfold_hsic, variance_lengthscale in zip(results, list_lengthscales):
                        fully_trained_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/ub_models/simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
                        fully_trained_FILE_PATH_IN_S3 = fully_trained_FOLDER_PATH_IN_S3 + "sdp_model.pkl"
                        fully_trained_sdp_model = load_file(fully_trained_FILE_PATH_IN_S3, fs)
                        fully_trained_sdp_model.metrics[f"hsic_train_kfold_{number_of_splits}"] = kfold_hsic

                        with fs.open(fully_trained_FILE_PATH_IN_S3, mode="wb") as file_out:
                            pickle.dump(fully_trained_sdp_model, file_out)
