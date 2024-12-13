"""
First training a Gaussian Process with two parameters (lengthscale, nugget).
We use the kernel(lengthscale, nugget) to compute s and the kernel (lengthscale) in the SDP problem.
We compute the prediction bands on a wide variety of length scale.
"""

import universalbands as ub
from universalbands.kernels import Matern
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels
import os
import s3fs
import json


def compute(X_train, y_train, theta_v, X_cal, y_cal, X_test, y_test, save):
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Retrieve posterior parameters of Gausian Process Regression.
    FILE_PATH_IN_S3 = (
        "luisito/these/sb_experiments/train_gp/results/gp_with_LsNug/"
        + "optimized_parameters.json"
    )
    with fs.open(FILE_PATH_IN_S3, mode="r") as file_in:
        theta_m = json.load(file_in)

    # Reconstruct the posterior kernel with sklearn.
    posterior_kernel = kernels.Matern(
        length_scale=theta_m["posterior_lengthscale"],
        length_scale_bounds=(1e-5, 1e5),
        nu=2.5,
    ) * kernels.ConstantKernel(constant_value=theta_m["posterior_variance"])

    # Reconstruct the Gram matrix on the training data (using posterior kernel).
    posterior_gram_matrix = posterior_kernel(X=X_train) + theta_m[
        "posterior_nugget"
    ] * np.eye(X_train.shape[0])

    # Compute the s parameter.
    same_posterior_alpha = np.linalg.solve(posterior_gram_matrix, y_train)
    s_parameter = (
        same_posterior_alpha.T @ posterior_gram_matrix @ same_posterior_alpha
    ).item()

    theta_m["s_parameter"] = s_parameter

    # Create the SDP problem
    sdp_model = ub.UniversalFunctionAndBandsRegressor(
        mean_kernel=Matern(
            length_scale=theta_m["posterior_lengthscale"],
            sigma=1,
            nu=2.5,
            nugget=0,
        ),
        variance_kernel=Matern(
            length_scale=theta_v,
            sigma=1,
            nu=2.5,
            nugget=0,
        ),
        problem="Liang",
        lambda2=1.0,
        delta=1e-3,
        s=theta_m["s_parameter"],
        checkSDP=False,
        verbose=False,
    )

    # Actually solve the SDP.
    sdp_model.train(X=X_train, y=y_train)
    FILE_PATH_OUT_S3 = save + "train.npz"
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        np.savez(
            file_out,
            gammahat=sdp_model.gammahat,
            Ahat=sdp_model.Ahat,
            Vhat=sdp_model.Vhat,
        )

    # Calibrate the model.
    sdp_model.calibrate(X=X_cal, y=y_cal, alpha=0.05)
    FILE_PATH_OUT_S3 = save + "calibration.npy"
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        np.save(file_out, arr=np.array(sdp_model.lambdahat))

    # Infer.
    mean_test_prediction, prediction_bands = sdp_model.predict(X=X_test)

    print("Saving metadata.")
    # Metadata
    sdp_metadata = metadata(sdp_model, theta_m, theta_v)
    FILE_PATH_OUT_S3 = save + "metadata.json"
    with fs.open(FILE_PATH_OUT_S3, mode="w") as file_out:
        json.dump(sdp_metadata, file_out)

    # Plot.
    fig = plot(
        X=X_train,
        y=y_train,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        mean_estimation_test=mean_test_prediction,
        bands_estimation_test=prediction_bands,
        limits=(-3, 5),
        title=f"Lengthscale is {theta_v}.",
    )
    FILE_PATH_OUT_S3 = save + "test.pdf"
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        fig.savefig(file_out, format="pdf", transparent=True, dpi=600)

    print(f"Lengthscale {theta_v} is done!")


def metadata(my_model, theta_m, theta_v):
    metadata_dict = {
        "input": {
            "problem": my_model.problem,
            "theta_m": theta_m,
            "theta_v": theta_v,
            "lambda2": my_model.lambda2,
            "delta": my_model.delta,
        },
        "output": {
            "solver_min": my_model.solver_min,
            "solver_state": my_model.solver_state,
            "solver_time": my_model.solver_time,
            "solver_iter": my_model.solver_iter,
        },
    }

    return metadata_dict


def plot(
    X,
    y,
    X_cal,
    y_cal,
    X_test,
    y_test,
    mean_estimation_test,
    bands_estimation_test,
    limits,
    title,
):
    """
    Standard plot for visualising prediction bands in one dimension.
    """
    line_width = 5
    marker_size = 120

    # Make the upper bounds and lower bounds from the mean estimation and the prediction bands.
    upper_bound = mean_estimation_test + bands_estimation_test
    lower_bound = mean_estimation_test - bands_estimation_test

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    X_test_sq = X_test.squeeze()
    mean_est_sq = mean_estimation_test.squeeze()
    sorted_indices = np.argsort(X_test_sq)
    X_test_sorted = X_test_sq[sorted_indices].squeeze()
    mean_est_sorted = mean_est_sq[sorted_indices].squeeze()

    upper_bound_sq = upper_bound.squeeze()
    lower_bound_sq = lower_bound.squeeze()
    upper_bound_sorted = upper_bound_sq[sorted_indices].squeeze()
    lower_bound_sorted = lower_bound_sq[sorted_indices].squeeze()

    ax.plot(X_test_sorted, mean_est_sorted, color="blue", linewidth=line_width)
    ax.fill_between(
        X_test_sorted,
        lower_bound_sorted,
        upper_bound_sorted,
        alpha=0.2,
        color="orange",
    )

    ax.scatter(X, y, s=marker_size, c="blue", alpha=1, marker="^")
    ax.scatter(X_cal, y_cal, s=marker_size, c="orange", alpha=1, marker="v")
    ax.scatter(X_test, y_test, s=35, c="black", alpha=0.6, marker="o")

    ax.plot(
        X_test_sorted,
        upper_bound_sorted,
        color="orange",
        linewidth=line_width,
        ls="--",
    )
    ax.plot(
        X_test_sorted,
        lower_bound_sorted,
        color="orange",
        linewidth=line_width,
        ls="--",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(limits)
    ax.set_title(label=title, loc="center")

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    from multiprocessing import Pool, cpu_count

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
    X_cal, y_cal = ub.data_generation.synthetic_data(
        case=f"case_{case_number}",
        sample_size=sample_size + 2,
        sample_dim=sample_dim,
        seed=321,
    )
    X_test, y_test = ub.data_generation.synthetic_data(
        case=f"case_{case_number}",
        sample_size=3 * sample_size + 3,
        sample_dim=sample_dim,
        seed=987,
    )

    # Set Hyperparameters
    number_of_lengthscale = 10
    all_theta_v = np.round(np.linspace(0.1, 1, number_of_lengthscale), 2)

    pool = Pool(processes=(min(cpu_count() - 1, number_of_lengthscale)))

    for theta_v in all_theta_v:
        BUCKET_OUT = f"luisito/these/sb_experiments/which_kernel/results/LsNug_LsNug_Ls_parameters/lengthscale_{theta_v}/"
        print(f"Launch lengthscale {theta_v}.")
        pool.apply_async(
            compute,
            args=(X_train, y_train, theta_v, X_cal, y_cal, X_test, y_test, BUCKET_OUT),
        )

    pool.close()
    pool.join()
