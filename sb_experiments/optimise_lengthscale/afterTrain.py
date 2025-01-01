import json
import pickle
import numpy as np
import os
import s3fs
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate


def plot_2d(
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
    Plot for visualizing prediction bands in two dimensions.
    Saves the interactive plot as an HTML file.
    """
    # Ensure y is a 1D array
    y = y.flatten()
    y_cal = y_cal.flatten()
    y_test = y_test.flatten()

    # Prepare bounds for prediction bands
    upper_bound = mean_estimation_test + bands_estimation_test
    lower_bound = mean_estimation_test - bands_estimation_test

    # Create 3D scatter plot
    fig = go.Figure()

    # Add training points
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=y,
            mode="markers",
            marker=dict(size=4, color="blue"),
            name="Training Data",
        )
    )

    # Add calibration points
    fig.add_trace(
        go.Scatter3d(
            x=X_cal[:, 0],
            y=X_cal[:, 1],
            z=y_cal,
            mode="markers",
            marker=dict(size=4, color="orange"),
            name="Calibration Data",
        )
    )

    # Add test points
    fig.add_trace(
        go.Scatter3d(
            x=X_test[:, 0],
            y=X_test[:, 1],
            z=y_test,
            mode="markers",
            marker=dict(size=2, color="black"),
            name="Test Data",
        )
    )
    # Add mean estimation surface
    fig.add_trace(
        go.Mesh3d(
            x=X_test[:, 0],
            y=X_test[:, 1],
            z=mean_estimation_test.flatten(),
            color="blue",
            opacity=0.7,
            name="Mean Estimation",
        )
    )

    # Add upper bound surface
    fig.add_trace(
        go.Mesh3d(
            x=X_test[:, 0],
            y=X_test[:, 1],
            z=upper_bound.flatten(),
            color="orange",
            opacity=0.5,
            name="Upper Bound",
        )
    )

    # Add lower bound surface
    fig.add_trace(
        go.Mesh3d(
            x=X_test[:, 0],
            y=X_test[:, 1],
            z=lower_bound.flatten(),
            color="orange",
            opacity=0.5,
            name="Lower Bound",
        )
    )


    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X1",
            yaxis_title="X2",
            zaxis_title="y",
        ),
        title=title,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
    )

    return fig



def load_file(file_path, fs):
    """Load a file based on its extension."""
    
    file_extension = file_path.split('.')[-1]  # Get the file extension

    if file_extension == "npz":
        # Load .npz file
        with fs.open(file_path, mode="rb") as file_in:
            data = np.load(file_in)
            X = data["X"]
            y = data["y"]
        return X, y

    elif file_extension == "json":
        # Load .json file
        with fs.open(file_path, mode="r") as file_in:
            data = json.load(file_in)
        return data

    elif file_extension == "pkl":
        # Load .pkl file
        with fs.open(file_path, mode="rb") as file_in:
            data = pickle.load(file_in)
        return data

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


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
    #ax.set_ylim(limits)
    ax.set_title(label=title, loc="center")

    plt.tight_layout()

    return fig


def extract_lengthscales(folder_path, fs):
    # List files in the folder
    file_list = fs.glob(folder_path + 'variance_lengthscale_*')  # Only match files starting with 'try_'
    print(file_list)

    return file_list


if __name__ == "__main__":
    import argparse

    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    parser = argparse.ArgumentParser(description='Argparse for BOBYQA optimisation of the lenthscale using k-folds.')
    parser.add_argument('--case_number', type=int)
    parser.add_argument('--size', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    print(f"Start for case {args.case_number}.")

    case_number = args.case_number
    sample_size = args.size
    sample_dim = args.dim
    seed = args.seed

    cal_case_number = args.case_number
    cal_sample_size = args.size
    cal_sample_dim = args.dim
    cal_seed = 321

    test_case_number = args.case_number
    test_sample_size = 2000
    test_sample_dim = args.dim
    test_seed = 987

    # Retrieve data path
    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
    X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)

    cal_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/"
    cal_data_FILE_PATH_IN_S3 = cal_data_FOLDER_PATH_IN_S3 + "data.npz"
    X_cal, y_cal = load_file(cal_data_FILE_PATH_IN_S3, fs)
    
    test_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{test_case_number}/sample_shape_({test_sample_size},{test_sample_dim})/seed_{test_seed}/"
    test_data_FILE_PATH_IN_S3 = test_data_FOLDER_PATH_IN_S3 + "data.npz"
    X_test, y_test = load_file(test_data_FILE_PATH_IN_S3, fs)

    delta = 1e-3
    lambda2 = 1
    problem = "Liang"
    alpha = 0.05
    
    model_FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/"
    list_lengthscales = extract_lengthscales(model_FOLDER_PATH_OUT_S3, fs)

    for folder_in in list_lengthscales:
        model_FILE_PATH_OUT_S3 = folder_in + "/sdp_model.pkl"
        sdp_model = load_file(model_FILE_PATH_OUT_S3, fs)

        sdp_model.calibrate(X=X_cal, y=y_cal, alpha=alpha)

        mean_prediction, bands_prediction = sdp_model.predict(X=X_test)

        if sample_dim==1:
            figure = plot(
                X=X_train,
                y=y_train,
                X_cal=X_cal,
                y_cal=y_cal,
                X_test=X_test,
                y_test=y_test,
                mean_estimation_test=mean_prediction,
                bands_estimation_test=bands_prediction,
                limits=(-5, 8),
                title=f"SDP Model\nLengthscale is {sdp_model.variance_kernel.get_params()['length_scale'][0]}.",
            )
            FILE_PATH_OUT_S3_PLOT = folder_in + "/figure.pdf"
            with fs.open(FILE_PATH_OUT_S3_PLOT, mode="wb") as file_out:
                figure.savefig(file_out, transparent=True, dpi='figure', format="pdf",
                metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
                plt.close()

        elif sample_dim==2:
            figure = plot_2d(
                X=X_train,
                y=y_train,
                X_cal=X_cal,
                y_cal=y_cal,
                X_test=X_test,
                y_test=y_test,
                mean_estimation_test=mean_prediction,
                bands_estimation_test=bands_prediction,
                limits=(-5, 8),
                title=f"SDP Model\nLengthscale is {sdp_model.variance_kernel.get_params()['length_scale']}.",
            )
            FILE_PATH_OUT_S3_PLOT = folder_in + "/figure.html"
            with fs.open(FILE_PATH_OUT_S3_PLOT, mode="w") as file_out:
                figure.write_html(file_out)
