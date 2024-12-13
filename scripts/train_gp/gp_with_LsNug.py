"""
This file trains a Gaussian Process on given data, optimising two parameters:
    - lengthscale
    - nugget
and outputs the posterior parameters.
"""

import universalbands as ub
from GPy.kern import Matern52
from GPy.models.gp_regression import GPRegression
import os
import s3fs
import json


def compute(X_train, y_train, save):
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Train the Gausian Process Regression.
    kernel = Matern52(input_dim=1)
    gp_model = GPRegression(
        X_train,
        y_train,
        kernel,
        Y_metadata=None,
        normalizer=None,
        noise_var=1.0,
        mean_function=None,
    )
    gp_model.kern.variance.fix()  # Here we fix the variance parameter of the kernel.
    gp_model.optimize_restarts(
        num_restarts=10, messages=False, verbose=True, max_iters=1000
    )

    # Retrieve posteriors parameters of the mean kernel.
    posterior_lengthscale = gp_model.kern.lengthscale[0]
    posterior_variance = gp_model.kern.variance[0]
    posterior_nugget = gp_model.Gaussian_noise.variance[0]

    theta_m = {
        "posterior_lengthscale": posterior_lengthscale,
        "posterior_variance": posterior_variance,
        "posterior_nugget": posterior_nugget,
    }

    FILE_PATH_OUT_S3 = save + "optimized_parameters.json"
    with fs.open(FILE_PATH_OUT_S3, mode="w") as file_out:
        json.dump(theta_m, file_out)


if __name__ == "__main__":

    # Generate the data
    case_number = 5
    sample_size = 100
    sample_dim = 1
    X_train, y_train = ub.data_generation.synthetic_data(
        case=f"case_{case_number}",
        sample_size=sample_size + 1,
        sample_dim=sample_dim,
        seed=123,
    )

    BUCKET_OUT = "luisito/these/sb_experiments/train_gp/results/gp_with_LsNug/"
    compute(X_train, y_train, BUCKET_OUT)
