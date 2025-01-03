import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import pickle
import os
import s3fs
import matplotlib.pyplot as plt
from utils import load_file
from scipy.spatial.distance import cdist, pdist, squareform


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
    #list_lengthscales = np.round(np.linspace(1e-3, 1, 900), 4)
    list_lengthscales = [0.2821]
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"
    alpha = 0.05

    # Which training data
    case_number = 5
    sample_size = 100
    sample_dim = 1
    seed = 123
    calibration_seed = 321
    test_seed = 987

    for variance_lengthscale in list_lengthscales:
        # Retrieve data path
        data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
        data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
        X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)

        cal_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{calibration_seed}/"
        cal_data_FILE_PATH_IN_S3 = cal_data_FOLDER_PATH_IN_S3 + "data.npz"
        X_cal, y_cal = load_file(cal_data_FILE_PATH_IN_S3, fs)
        
        test_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({300},{sample_dim})/seed_{test_seed}/"
        test_data_FILE_PATH_IN_S3 = test_data_FOLDER_PATH_IN_S3 + "data.npz"
        X_test, y_test = load_file(test_data_FILE_PATH_IN_S3, fs)
                                        
        #FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/e2e_sdp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/data_case_{test_case_number}/sample_shape_({test_sample_size},{test_sample_dim})/seed_{test_seed}/"
        fully_trained_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/ub_models/simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
        model_FILE_PATH_OUT_S3 = fully_trained_FOLDER_PATH_IN_S3 + "sdp_model.pkl"
        sdp_model = load_file(model_FILE_PATH_OUT_S3, fs)
        sdp_model.calibrate(X=X_cal, y=y_cal, alpha=alpha)
        mean_prediction, bands_prediction = sdp_model.predict(X=X_test)

        figure = plot(
            X=X_train,
            y=y_train,
            X_cal=X_cal,
            y_cal=y_cal,
            X_test=X_test,
            y_test=y_test,
            mean_estimation_test=mean_prediction,
            bands_estimation_test=bands_prediction,
            limits=(-3, 5),
            title=f"SDP Model\nLengthscale is {variance_lengthscale}.",
        )
        FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/images/"
        FILE_PATH_OUT_S3_PLOT = FOLDER_PATH_OUT_S3 + f"max_10fold_ls_{variance_lengthscale}.pdf"
        with fs.open(FILE_PATH_OUT_S3_PLOT, mode="wb") as file_out:
            figure.savefig(file_out, transparent=True, dpi='figure', format="pdf",
            metadata=None,
            bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
            plt.close()
