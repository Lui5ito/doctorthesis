"""
This file trains a Gaussian Process model using GPy.
The kernel is a Matérn(lengthscale) and uses a nugget to train but not for inference.
Outputs:
    - kernel lengthscale.
    - kernel variance (fixed to 1).
    - likelihood nugget (appears in the coefficients).
    - norm of the function computed on the training data.
    - Also saves a copy of the GPy model in a pickle file.
"""

import numpy as np
import os
import s3fs
import json
from GPy.kern import Matern52
from GPy.models.gp_regression import GPRegression
import pickle


if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Which data to use
    import argparse

    parser = argparse.ArgumentParser(description="Argparse for data generation.")
    parser.add_argument("--cases", nargs="+", type=int)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--dims", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    # Which data to generate
    cases = args.cases
    all_sample_sizes = args.sizes
    all_sample_dims = args.dims
    all_sample_seeds = args.seeds

    # Retrieve the data
    for case_number in cases:
        for sample_size in all_sample_sizes:
            for sample_dim in all_sample_dims:
                for seed in all_sample_seeds:
                    # Retrieve data path
                    FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    FILE_PATH_IN_S3 = FOLDER_PATH_IN_S3 + "data.npz"

                    # Check if the file already exists
                    if not fs.exists(FILE_PATH_IN_S3):
                        print(
                            f"File {FILE_PATH_IN_S3} does not exists. Cannot proceed."
                        )
                        continue

                    # Retrieve data
                    with fs.open(FILE_PATH_IN_S3, mode="rb") as file_in:
                        data = np.load(file_in)
                        X_train = data["X"]
                        y_train = data["y"]

                    # Construct saving path
                    FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    FILE_PATH_OUT_S3 = FOLDER_PATH_OUT_S3 + "optimized_parameters.json"

                    # Check if the model has already been trained.
                    if fs.exists(FILE_PATH_OUT_S3):
                        print(
                            f"File {FILE_PATH_OUT_S3} already exists. Do not compute again."
                        )
                        continue

                    # Train the Gausian Process Regression.
                    kernel = Matern52(input_dim=sample_dim, variance=1.0, lengthscale=None, ARD=True, active_dims=None, name='Mat52')
                    gp_model = GPRegression(
                        X_train,
                        y_train,
                        kernel,
                        Y_metadata=None,
                        normalizer=None,
                        noise_var=1.0,
                        mean_function=None,
                    )
                    gp_model.kern.variance.fix()  # Fixing the variance parameter.
                    gp_model.optimize_restarts(
                        num_restarts=10, messages=False, verbose=True, max_iters=1000
                    )

                    # Retrieve posteriors parameters of the mean kernel.
                    posterior_lengthscale = list(gp_model.kern.lengthscale)
                    posterior_variance = gp_model.kern.variance[0]
                    posterior_nugget = gp_model.Gaussian_noise.variance[0]

                    # Reconstruct the norm of the function on the training parameters.
                    gamma = gp_model.posterior.woodbury_vector
                    K = gp_model.kern.K(X_train)
                    post_training_norm = np.matmul(gamma.T, np.matmul(K, gamma))[0][0]

                    theta_m = {
                        "posterior_lengthscale": posterior_lengthscale,
                        "posterior_variance": posterior_variance,
                        "posterior_nugget": posterior_nugget,
                        "posterior_training_norm": post_training_norm,
                    }

                    # Save the parameters
                    with fs.open(FILE_PATH_OUT_S3, mode="w") as file_out:
                        json.dump(theta_m, file_out)

                    # Save a copy of the model
                    MODEL_PATH_OUT_S3 = FOLDER_PATH_OUT_S3 + "gpy_model.pkl"
                    with fs.open(MODEL_PATH_OUT_S3, "wb") as file_out:
                        pickle.dump(gp_model, file_out)
