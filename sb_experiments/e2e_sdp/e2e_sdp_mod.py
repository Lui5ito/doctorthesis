import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import pickle
import os
import s3fs
from ub_models.utils import load_file
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


if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Lengthscales to compute 900
    list_lengthscales = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"
    alpha = 0.05

    # Which training data
    case_number = 5
    sample_size = 100
    sample_dim = 1
    seed = 123

    # Which calibration data
    cal_case_number = 5
    cal_sample_size = 100
    cal_sample_dim = 1
    cal_seed = 321

    # Which calibration data
    test_case_number = 5
    test_sample_size = 300
    test_sample_dim = 1
    test_seed = 987

    # Retrieve data path
    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
    X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)

    # Retrieve GP model path
    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"
    theta_m = load_file(gp_FILE_PATH_IN_S3, fs)

    # Retrieve calibration data
    cal_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/"
    cal_data_FILE_PATH_IN_S3 = cal_data_FOLDER_PATH_IN_S3 + "data.npz"
    X_cal, y_cal = load_file(cal_data_FILE_PATH_IN_S3, fs)

    # Retrieve test data
    test_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{test_case_number}/sample_shape_({test_sample_size},{test_sample_dim})/seed_{test_seed}/"
    test_data_FILE_PATH_IN_S3 = test_data_FOLDER_PATH_IN_S3 + "data.npz"
    X_test, y_test = load_file(test_data_FILE_PATH_IN_S3, fs)

    for variance_lengthscale in list_lengthscales:
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
        print(f"Variance lengthscale {variance_lengthscale}.")

        # Train the model
        sdp_model.train(X=X_train, y=y_train)
        sdp_model.metrics["hsic_train_full"] = sdp_model.compute_HSIC(X=X_train, y=y_train, extra=False)

        # Calibrate the model
        sdp_model.calibrate(X=X_cal, y=y_cal, alpha=alpha)
        sdp_model.metrics["hsic_cal_full"] = sdp_model.compute_HSIC(X=X_cal, y=y_cal, extra=False)

        # Predict
        # Predict
        #X_test = np.concatenate((X_train, X_cal), axis=0)
        #y_test = np.concatenate((y_train, y_cal), axis=0)
        mean_prediction, variance_prediction = sdp_model.predict(X=X_test)
        mean_width = np.mean(variance_prediction)

        # Plot
        figure = plot(
            X=X_train,
            y=y_train,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
            mean_estimation_test=mean_prediction,
            bands_estimation_test=variance_prediction,
            limits=(-3, 5),
            title=f"SDP Model\nLengthscale is {variance_lengthscale}.\nLambdahat is {sdp_model.lambdahat}.\nMean width is {mean_width}.\nHSIC train is {sdp_model.metrics["hsic_train_full"]}.\nHSIC cal is {sdp_model.metrics["hsic_cal_full"]}.",
        )

        FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/e2e_sdp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/data_case_{test_case_number}/sample_shape_({test_sample_size},{test_sample_dim})/seed_{test_seed}/"


        plot_FILE_PATH_OUT_S3 = FOLDER_PATH_OUT_S3 + "figure_bands_test.pdf"
        with fs.open(plot_FILE_PATH_OUT_S3, mode="wb") as file_out:
            figure.savefig(file_out, transparent=True, dpi='figure', format="pdf",
            metadata=None,
            bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
            plt.close()
