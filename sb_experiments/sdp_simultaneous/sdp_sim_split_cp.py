"""
This file is here to plot the results of file 'sdp_sim_optimise_hsic.py'
"""
import os
import s3fs
import re  # Import regular expressions for extracting 'try' numbers
import json
import matplotlib.pyplot as plt
import pickle
import numpy as np


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


def extract_try_ids(folder_path):
    # List files in the folder
    file_list = fs.glob(folder_path + 'try_*')  # Only match files starting with 'try_'

    # Create a list to store the extracted 'try' IDs
    try_ids = []

    # Regular expression pattern to extract the 'try_' number
    try_pattern = re.compile(r'try_(\d+)')

    # Iterate through the file list and extract the 'try' number
    for file in file_list:
        # Search for the 'try_' pattern in the file path
        match = try_pattern.search(file)
        if match:
            try_id = int(match.group(1))  # Extract the number and convert it to integer
            try_ids.append(try_id)

    # Sort the try IDs in ascending order
    try_ids.sort()

    # Print the sorted list of 'try' IDs
    return try_ids


def find_max_hsic(all_data):
    max_hsic = -float('inf')  # Start with a very low value for the max HSIC
    best_lengthscale = None

    # Loop through all the restarts and find the last try_id (maximum try_id)
    for restart, data in all_data.items():
        try_ids = data["try_ids"]
        try_lengthscale = data["try_lengthscale"]
        try_hsic = data["try_hsic"]

        # Get the last try_id (i.e., the maximum try_id)
        last_try_id = try_ids[-1]
        last_lengthscale = try_lengthscale[-1]
        last_hsic = try_hsic[-1]

        # Check if this try has the highest HSIC value
        if last_hsic > max_hsic:
            max_hsic = last_hsic
            best_lengthscale = last_lengthscale
            best_try_id = last_try_id
            best_restart = restart

    # Return the results
    return best_restart, best_try_id, best_lengthscale, max_hsic


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

    # Define problem
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    # Which training data
    case_number = 5
    sample_size = 100
    sample_dim = 1
    seed = 123
    restart_num = 10
    all_data = {}
    cal_case_number = 5
    cal_sample_size = 100
    cal_sample_dim = 1
    cal_seed = 321
    alpha = 0.05
    test_case_number = 5
    test_sample_size = 300
    test_sample_dim = 1
    test_seed = 987

    for restart in range(restart_num):
        #FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_training/restart_{restart}/"
        FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_calibration/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/restart_{restart}/"
        
        try_ids = extract_try_ids(FOLDER_PATH_IN_S3)
        try_lengthscale = []
        try_hsic = []
        for num_id in try_ids:
            #try_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_training/restart_{restart}/try_{num_id}/"
            try_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_calibration/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/restart_{restart}/try_{num_id}/"

            FILE_PATH_IN_S3 = try_FOLDER_PATH_IN_S3 + "all_parameters.json"
            with fs.open(FILE_PATH_IN_S3, mode="r") as file_in:
                all_parameters = json.load(file_in)
            try_lengthscale.append(all_parameters["input_parameters"]["theta_v"])
            #try_hsic.append(all_parameters["output_parameters"]["training_hsic"])
            try_hsic.append(all_parameters["output_parameters"]["calibration_hsic"])
        
        all_data[f"restart_{restart}"] = {
            "try_ids": try_ids,
            "try_lengthscale": try_lengthscale,
            "try_hsic": try_hsic,
        }
    
    best_restart, best_try_id, best_lengthscale, max_hsic = find_max_hsic(all_data)

    #best_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_training/{best_restart}/try_{best_try_id}/"
    best_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_calibration/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/{best_restart}/try_{best_try_id}/"
    best_FILE_PATH_IN_S3 = best_FOLDER_PATH_IN_S3 + "sdp_model.pkl"
    sdp_model = load_file(best_FILE_PATH_IN_S3, fs)
    
    #best_FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_training/best_model/calibration/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/"
    best_FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_calibration/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/best_model/"
    
    # Retrieve train data
    train_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    train_data_FILE_PATH_IN_S3 = train_data_FOLDER_PATH_IN_S3 + "data.npz"
    X_train, y_train = load_file(train_data_FILE_PATH_IN_S3, fs)

    # Retrieve calib data
    cal_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/"
    cal_data_FILE_PATH_IN_S3 = cal_data_FOLDER_PATH_IN_S3 + "data.npz"
    X_cal, y_cal = load_file(cal_data_FILE_PATH_IN_S3, fs)

    sdp_model.calibrate(X=X_cal, y=y_cal, alpha=alpha)

    # Retrieve calib data
    test_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{test_case_number}/sample_shape_({test_sample_size},{test_sample_dim})/seed_{test_seed}/"
    test_data_FILE_PATH_IN_S3 = test_data_FOLDER_PATH_IN_S3 + "data.npz"
    X_test, y_test = load_file(test_data_FILE_PATH_IN_S3, fs)

    # Predict
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
        title=f"SDP Model\nLengthscale is {best_lengthscale}.",
    )
    FILE_PATH_OUT_S3_PLOT = best_FOLDER_PATH_OUT_S3 + f"figure_test_seed_{test_seed}.pdf"
    with fs.open(FILE_PATH_OUT_S3_PLOT, mode="wb") as file_out:
        figure.savefig(file_out, transparent=True, dpi='figure', format="pdf",
        metadata=None,
        bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
        plt.close()


    
    







