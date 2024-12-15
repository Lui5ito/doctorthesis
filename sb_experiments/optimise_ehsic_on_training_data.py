"""
The goal of this file is to perform optimisation over the variance lengthscale using the energy-HSIC.
"""

import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import json
import pickle
import os
import s3fs
import nlopt

# Initialize a global list to store the tested values
tested_values = []
associated_hsic = []


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

def objective_function(theta_v, grad):

    # Log the tested value
    tested_values.append(theta_v[0])
    print(f"Lengthscale tested: {theta_v[0]}")

    # Define problem
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    # Define training data
    case_number = 5
    sample_size = 100
    sample_dim = 1
    seed = 123

    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Retrieve data path
    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
    # Check if the file already exists
    if not fs.exists(data_FILE_PATH_IN_S3):
        print(f"File {data_FILE_PATH_IN_S3} does not exists. Cannot proceed.")
        raise FileNotFoundError(f"File {data_FILE_PATH_IN_S3} does not exist in S3.")

    # Retrieve data
    with fs.open(data_FILE_PATH_IN_S3, mode="rb") as file_in:
        data = np.load(file_in)
        X_train = data["X"]
        y_train = data["y"]

    # Retrieve gp model path
    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"
    # Check if the file already exists
    if not fs.exists(gp_FILE_PATH_IN_S3):
        print(f"File {gp_FILE_PATH_IN_S3} does not exists. Cannot proceed.")
        raise FileNotFoundError(f"File {gp_FILE_PATH_IN_S3} does not exist in S3.")
    # Retrieve data
    with fs.open(gp_FILE_PATH_IN_S3, mode="rb") as file_in:
        theta_m = json.load(file_in)

    # Define the model
    sdp_model = ub.UniversalFunctionAndBandsRegressor(
        mean_kernel=kernels.Matern(length_scale=theta_m["posterior_lengthscale"], length_scale_bounds=(1e-5, 1e5), nu=2.5),
        variance_kernel=kernels.Matern(length_scale=theta_v, length_scale_bounds=(1e-5, 1e5), nu=2.5),
        lambda2=lambda2,
        delta=delta,
        s=theta_m["posterior_training_norm"],
        problem=problem,
        checkSDP=False,
        verbose=False
    )
    # Train the model
    sdp_model.train(X=X_train, y=y_train)
    # Compute HSIC
    train_hsic = HSIC(X_train, y_train, sdp_model)
    associated_hsic.append(train_hsic)

    return train_hsic


def last_train(theta_v, problem_definiton):
    # Define problem
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    # Define training data
    case_number = 5
    sample_size = 100
    sample_dim = 1
    seed = 123

    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Retrieve data path
    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"

    # Retrieve data
    with fs.open(data_FILE_PATH_IN_S3, mode="rb") as file_in:
        data = np.load(file_in)
        X_train = data["X"]
        y_train = data["y"]

    # Retrieve gp model path
    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"

    # Retrieve data
    with fs.open(gp_FILE_PATH_IN_S3, mode="rb") as file_in:
        theta_m = json.load(file_in)

    # Define the model
    sdp_model = ub.UniversalFunctionAndBandsRegressor(
        mean_kernel=kernels.Matern(length_scale=theta_m["posterior_lengthscale"], length_scale_bounds=(1e-5, 1e5), nu=2.5),
        variance_kernel=kernels.Matern(length_scale=theta_v, length_scale_bounds=(1e-5, 1e5), nu=2.5),
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

    FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/"

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
            "theta_v": theta_v,
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
    
    # Save whole sdp model
    FILE_PATH_OUT_S3_MODEL = FOLDER_PATH_OUT_S3 + "sdp_model.pkl"
    with fs.open(FILE_PATH_OUT_S3_MODEL, mode="wb") as file_out:
        pickle.dump(sdp_model, file_out)

    # Save all values tested by the solver
    FILE_PATH_OUT_S3_SOLVER_PATH = FOLDER_PATH_OUT_S3 + "solver_path.npz"
    with fs.open(FILE_PATH_OUT_S3_SOLVER_PATH, mode="wb") as file_out:
        np.savez(file_out, lengthscales=np.array(tested_values), ehsic=np.array(associated_hsic))

    return "Best model saved!"


if __name__ == "__main__":

    # Initial guess for theta_v
    initial_theta_v = np.array([1.0])

    # Define bounds for the parameters.
    lower_bounds = np.array([1e-6])
    upper_bounds = np.array([10.0])

    # Create the optimizer.
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(initial_theta_v))

    # Set bounds.
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)

    # Set the objective function, we want the maximum of e-HSIC.
    opt.set_max_objective(objective_function)

    # Set the number max of iterations.
    opt.set_maxeval(100)
    opt.set_xtol_rel(1e-6)

    # Actually solve
    theta_v_opt = opt.optimize(initial_theta_v)

    # Compute and save the best model
    string = last_train(theta_v_opt[0])

    print(string)
