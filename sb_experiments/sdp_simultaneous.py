"""
We compute the SDP model by the following:
- On the train dataset, we compute:
    - theta_m (using a Gaussian Process),
    - m_hat and v_hat using the SDP problem,
    - the energy-HSIC computed on the training data.
    (- theta_f which is the lengthscale determined by the highest HSIC.)
"""

import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import json
import pickle
import os
import s3fs


def energy_kernel(x, y):
    # Compute norms of x, y, and their difference
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    norm_diff = np.linalg.norm(x - y)

    # Calculate the kernel value
    return norm_x + norm_y - norm_diff


def energy_gram_matrix(X):
    # Initialize the Gram matrix with zeros
    n = len(X)
    gram = np.zeros((n, n))

    # Compute only the upper triangle (including the diagonal)
    for i in range(n):
        for j in range(i, n):
            gram[i, j] = energy_kernel(X[i], X[j])
            if i != j:
                gram[j, i] = gram[i, j]  # Mirror to the lower triangle

    return gram


def compute_hsic(abs_error, width):
    n = abs_error.shape[0]

    # Compute the centering matrix
    centering_matrix = np.eye(n) - (1 / n) * np.ones((n, n))

    # Compute the Gram matrices
    abs_error_gram_matrix = energy_gram_matrix(abs_error)
    width_gram_matrix = energy_gram_matrix(width)

    # Apply the centering matrix once to each Gram matrix
    centered_abs_error_gram = np.matmul(abs_error_gram_matrix, centering_matrix)
    centered_width_gram = np.matmul(width_gram_matrix, centering_matrix)

    # Compute HSIC using the trace of the product of the centered Gram matrices
    hsic_value = np.trace(np.matmul(centered_abs_error_gram, centered_width_gram)) / (
        n**2
    )

    return hsic_value


def HSIC(X_train, y_train, sdp_model):
    # Compute mean prediction on training data
    mean_gram_matrix = sdp_model.mean_kernel(X=X_train).T
    mean_estimator_prediction = np.matmul(mean_gram_matrix, sdp_model.gammahat)

    # Compute scores on training data
    variance_gram_matrix = sdp_model.variance_kernel(X=X_train)
    Phi_pred = np.linalg.solve(a=sdp_model.Vhat.T, b=variance_gram_matrix)
    variance_estimator_prediction = np.diag(Phi_pred.T @ sdp_model.Ahat @ Phi_pred).reshape(-1, 1)
    predicted_score_function = np.sqrt(variance_estimator_prediction + sdp_model.delta)

    abs_errors = np.abs(y_train - mean_estimator_prediction)
    e_hsic_train = compute_hsic(abs_errors, predicted_score_function)

    return e_hsic_train


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

    for case_number in cases:
        for sample_size in all_sample_sizes:
            for sample_dim in all_sample_dims:
                for seed in all_sample_seeds:
                    # Retrieve data path
                    FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    FILE_PATH_IN_S3 = FOLDER_PATH_IN_S3 + "data.npz"

                    # Check if the file already exists
                    if not fs.exists(FILE_PATH_IN_S3):
                        print(f"File {FILE_PATH_IN_S3} does not exists. Cannot proceed.")
                        continue

                    # Retrieve data
                    with fs.open(FILE_PATH_IN_S3, mode="rb") as file_in:
                        data = np.load(file_in)
                        X_train = data["X"]
                        y_train = data["y"]

                    # Retrieve gp model path
                    FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    FILE_PATH_IN_S3 = FOLDER_PATH_IN_S3 + "optimized_parameters.json"

                    # Check if the file already exists
                    if not fs.exists(FILE_PATH_IN_S3):
                        print(f"File {FILE_PATH_IN_S3} does not exists. Cannot proceed.")
                        continue

                    # Retrieve data
                    with fs.open(FILE_PATH_IN_S3, mode="rb") as file_in:
                        theta_m = json.load(file_in)

                    for variance_lengthscale in length_scale_list:

                        FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/sdp_simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
                        # Check if the file already exists
                        if fs.exists(FOLDER_PATH_OUT_S3):
                            print(f"File {FOLDER_PATH_OUT_S3} already exists. Do not compute.")
                            continue

                        # Define the model
                        sdp_model = ub.UniversalFunctionAndBandsRegressor(
                            mean_kernel=kernels.Matern(length_scale=theta_m["posterior_lengthscale"], length_scale_bounds=(1e-5, 1e5), nu=2.5),
                            variance_kernel=kernels.Matern(length_scale=variance_lengthscale, length_scale_bounds=(1e-5, 1e5), nu=2.5),
                            lambda2=lambda2,
                            delta=delta,
                            s=theta_m["posterior_training_norm"],
                            problem=problem,
                            checkSDP=False,
                            verbose=True
                        )
                        # Train the model
                        sdp_model.train(X=X_train, y=y_train)

                        # Compute HSIC
                        train_hsic = HSIC(X_train, y_train, sdp_model)

                        # Save SDP params
                        FILE_PATH_OUT_S3_PARAMS = FOLDER_PATH_OUT_S3 + "all_parameters.json"
                        all_parameters = {
                            "training_data": {
                                "data_case": case_number,
                                "shape": (sample_size, sample_dim),
                                "seed": seed,
                            },
                            "input_parameters": {
                                "problem": problem,
                                "theta_m": theta_m,
                                "theta_v": variance_lengthscale,
                                "lambda2": lambda2,
                                "delta": delta,
                            },
                            "output_parameters": {
                                "training_hsic": train_hsic,
                                "solver_min": sdp_model.solver_min,
                                "solver_state": sdp_model.solver_state,
                                "solver_time": sdp_model.solver_time,
                                "solver_iter": sdp_model.solver_iter,
                            },
                        }
                        with fs.open(FILE_PATH_OUT_S3_PARAMS, mode="w") as file_out:
                            json.dump(all_parameters, file_out)

                        # Save SDP objects to reconstruct the model
                        FILE_PATH_OUT_S3_OBJECTS = FOLDER_PATH_OUT_S3 + "trained_objects.npz"
                        with fs.open(FILE_PATH_OUT_S3_OBJECTS, mode="wb") as file_out:
                            np.savez(
                                file_out,
                                gammahat=sdp_model.gammahat,
                                Ahat=sdp_model.Ahat,
                                Vhat=sdp_model.Vhat,
                            )

                        # Save whole sdp model
                        FILE_PATH_OUT_S3_MODEL = FOLDER_PATH_OUT_S3 + "sdp_model.pkl"
                        with fs.open(FILE_PATH_OUT_S3_MODEL, mode="wb") as file_out:
                            pickle.dump(sdp_model, file_out)
