"""
From a trained and calibrated SDP model, we infere on new data:
- On the inference dataset, we compute:
    - the predicted mean,
    - the predicted bands,
    - the average length,
    - the energy-HSIC?,
    - the marginal coverage,
    - 
"""

import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import json
import pickle
import os
import s3fs
import matplotlib.pyplot as plt


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


def compute_marginal_coverage(y_true, mean_est, bands_est):
    lower_bound = mean_est-bands_est
    upper_bound = mean_est+bands_est
    coverage = ((lower_bound <= y_true) & (y_true <= upper_bound)).mean()
    print(coverage)

    return coverage

def compute_average_length(bands_est):
    return np.mean(bands_est)    


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
                    for variance_lengthscale in length_scale_list:
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

                                            # Retrieve sdp model path
                                            FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/sdp_simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/calibration/data_case_{calibration_case_number}/sample_shape_({calibration_sample_size},{calibration_sample_dim})/seed_{calibration_seed}/alpha_{alpha}/"
                                            FILE_PATH_IN_S3_MODEL = FOLDER_PATH_IN_S3 + "sdp_model_calibrated.pkl"

                                            # Check if the file already exists
                                            if not fs.exists(FILE_PATH_IN_S3_MODEL):
                                                print(f"File {FILE_PATH_IN_S3_MODEL} does not exists. Cannot proceed.")
                                                continue

                                            # Retrieve sdp model
                                            with fs.open(FILE_PATH_IN_S3_MODEL, mode="rb") as file_in:
                                                sdp_model = pickle.load(file_in)

                                            # Which inference data
                                            inference_cases = [5]
                                            inference_all_sample_sizes = [300]
                                            inference_all_sample_dims = [1]
                                            inference_all_sample_seeds = [987, 986, 985]
                                            for inference_case_number in inference_cases:
                                                for inference_sample_size in inference_all_sample_sizes:
                                                    for inference_sample_dim in inference_all_sample_dims:
                                                        for inference_seed in inference_all_sample_seeds:
                                                            FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/sdp_simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/calibration/data_case_{calibration_case_number}/sample_shape_({calibration_sample_size},{calibration_sample_dim})/seed_{calibration_seed}/alpha_{alpha}/inference/data_case_{inference_case_number}/sample_shape_({inference_sample_size},{inference_sample_dim})/seed_{inference_seed}/"

                                                            # Retrieve data path
                                                            FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{inference_case_number}/sample_shape_({inference_sample_size},{inference_sample_dim})/seed_{inference_seed}/"
                                                            FILE_PATH_IN_S3 = FOLDER_PATH_IN_S3 + "data.npz"

                                                            # Check if the file already exists
                                                            if not fs.exists(FILE_PATH_IN_S3):
                                                                print(f"File {FILE_PATH_IN_S3} does not exists. Cannot proceed.")
                                                                continue

                                                            # Retrieve data
                                                            with fs.open(FILE_PATH_IN_S3, mode="rb") as file_in:
                                                                data = np.load(file_in)
                                                                X_test = data["X"]
                                                                y_test = data["y"]

                                                            # Predict
                                                            mean_prediction, bands_prediction = sdp_model.predict(X=X_test)

                                                            # Still to compute some metrics on test data
                                                            FILE_PATH_OUT_S3_PARAMS = FOLDER_PATH_OUT_S3 + "test_metrics.json"
                                                            marginal_coverage = compute_marginal_coverage(y_test, mean_prediction, bands_prediction)
                                                            avg_length = compute_average_length(bands_prediction)
                                                            test_metrics = {
                                                                "test_data": {
                                                                    "data_case": inference_case_number,
                                                                    "shape": (inference_sample_size, inference_sample_dim),
                                                                    "ssed": inference_seed,
                                                                },
                                                                "marginal_coverage": marginal_coverage,
                                                                "average_length": avg_length,
                                                            }
                                                            with fs.open(FILE_PATH_OUT_S3_PARAMS, mode="w") as file_out:
                                                                json.dump(test_metrics, file_out)
                                                            # ...

                                                            # Plot
                                                            figure = plot(
                                                                X=X_train,
                                                                y=y_train,
                                                                X_cal=X_calibration,
                                                                y_cal=y_calibration,
                                                                X_test=X_test,
                                                                y_test=y_test,
                                                                mean_estimation_test=mean_prediction,
                                                                bands_estimation_test=bands_prediction,
                                                                limits=(-3, 5),
                                                                title=f"SDP Model\nLengthscale is {variance_lengthscale}.",
                                                            )
                                                            FILE_PATH_OUT_S3_PLOT = FOLDER_PATH_OUT_S3 + "figure.pdf"
                                                            with fs.open(FILE_PATH_OUT_S3_PLOT, mode="wb") as file_out:
                                                                figure.savefig(file_out, transparent=True, dpi='figure', format="pdf",
                                                                metadata=None,
                                                                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
                                                                plt.close()

                                                            # Save model predictions
                                                            FILE_PATH_OUT_S3_OBJECTS = FOLDER_PATH_OUT_S3 + "predictions.npz"
                                                            with fs.open(FILE_PATH_OUT_S3_OBJECTS, mode="wb") as file_out:
                                                                np.savez(
                                                                    file_out,
                                                                    mean_predictions=mean_prediction,
                                                                    bands_predictions=bands_prediction,
                                                                )
