"""
The goal of this file is to perform optimisation over the variance lengthscale using the energy-HSIC.
Key points:
    - All models should be saved.
    - You can choose between two objective functions:
        - e-HSIC computed on the training data
        - e-HSIC computed on a calibration data
    - The path BOBYQA took should be saved in order to reconstruct it.
"""

import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import json
import pickle
import os
import s3fs
import nlopt


def HSIC(X, y, sdp_model):
    """Compute HSIC for given X and y and sdp_model."""

    # Compute mean prediction on training data.
    mean_gram_matrix = sdp_model.mean_kernel(X=X).T
    mean_estimator_prediction = np.matmul(mean_gram_matrix, sdp_model.gammahat)

    # Compute scores on training data.
    variance_gram_matrix = sdp_model.variance_kernel(X=X)
    Phi_pred = np.linalg.solve(a=sdp_model.Vhat.T, b=variance_gram_matrix)
    variance_estimator_prediction = np.diag(Phi_pred.T @ sdp_model.Ahat @ Phi_pred).reshape(-1, 1)
    predicted_score_function = np.sqrt(variance_estimator_prediction + sdp_model.delta)

    # Compute errors and HSIC.
    abs_errors = np.abs(y - mean_estimator_prediction)
    e_hsic_train = ub.metrics.energy_hsic.Energy_HSIC(abs_error=abs_errors, width=predicted_score_function).compute()

    return e_hsic_train


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


def restart_optimise(num_restarts, settings, on):

    # Define bounds for the parameters.
    lower_bounds = np.array([settings["lower_bound"]])
    upper_bounds = np.array([settings["upper_bound"]])

    for restart in range(num_restarts):
        log = {"restart": restart, "try": 0}
        def objective_function_without_settings(theta_v, grad):
            if on == "train":
                return objective_function_train(theta_v, grad, settings, log)
            elif on == "calibration":
                return objective_function_cal(theta_v, grad, settings, log)
        
        initial_theta_v = np.random.uniform(lower_bounds, upper_bounds)
        opt = nlopt.opt(nlopt.LN_BOBYQA, len(initial_theta_v))
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_max_objective(objective_function_without_settings)
        opt.set_maxeval(settings["max_eval"])
        opt.set_ftol_rel(1e-6)
        opt.set_ftol_abs(1e-6)
        opt.set_xtol_rel(1e-6)
        opt.set_xtol_abs(1e-6)
        # Actually solve
        theta_v_opt = opt.optimize(initial_theta_v)


def objective_function_train(theta_v, grad, settings, log):
    # Unpack settings
    delta = settings["delta"]
    lambda2 = settings["lambda2"]
    problem = settings["problem"]

    case_number = settings["case_number"]
    sample_size = settings["sample_size"]
    sample_dim = settings["sample_dim"]
    seed = settings["seed"]

    fs = settings["fs"]

    # Unpack log
    restart_num = log["restart"]
    try_num = log["try"]
    
    # Get train data
    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
    X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)

    # Get theta_m
    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"
    theta_m = load_file(gp_FILE_PATH_IN_S3, fs)

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
    sdp_model.train(X=X_train, y=y_train)
    train_hsic = HSIC(X_train, y_train, sdp_model)

    # Save this try.
    FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_training/restart_{restart_num}/try_{try_num}/"

    params_FILE_PATH_OUT_S3 = FOLDER_PATH_OUT_S3 + "all_parameters.json"
    all_parameters = {
        "training_data": {
            "data_case": case_number,
            "shape": (sample_size, sample_dim),
            "seed": seed,
        },
        "input_parameters": {
            "problem": problem,
            "theta_m": theta_m,
            "theta_v": theta_v[0],
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
    with fs.open(params_FILE_PATH_OUT_S3, mode="w") as file_out:
        json.dump(all_parameters, file_out)

    model_FILE_PATH_OUT_S3 = FOLDER_PATH_OUT_S3 + "sdp_model.pkl"
    with fs.open(model_FILE_PATH_OUT_S3, mode="wb") as file_out:
        pickle.dump(sdp_model, file_out)
    
    log["try"] += 1

    return train_hsic


def objective_function_cal(theta_v, grad, settings, log):
    # Unpack settings
    delta = settings["delta"]
    lambda2 = settings["lambda2"]
    problem = settings["problem"]

    case_number = settings["case_number"]
    sample_size = settings["sample_size"]
    sample_dim = settings["sample_dim"]
    seed = settings["seed"]

    cal_case_number = settings["cal_case_number"]
    cal_sample_size = settings["cal_sample_size"]
    cal_sample_dim = settings["cal_sample_dim"]
    cal_seed = settings["cal_seed"]

    fs = settings["fs"]

    # Unpack log
    restart_num = log["restart"]
    try_num = log["try"]
    
    # Get train data
    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
    X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)

    # Get cal data
    cal_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/"
    cal_data_FILE_PATH_IN_S3 = cal_data_FOLDER_PATH_IN_S3 + "data.npz"
    X_cal, y_cal = load_file(cal_data_FILE_PATH_IN_S3, fs)

    # Get theta_m
    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"
    theta_m = load_file(gp_FILE_PATH_IN_S3, fs)

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
    sdp_model.train(X=X_train, y=y_train)
    cal_hsic = HSIC(X_cal, y_cal, sdp_model)

    # Save this try.
    FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_calibration/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/restart_{restart_num}/try_{try_num}/"

    params_FILE_PATH_OUT_S3 = FOLDER_PATH_OUT_S3 + "all_parameters.json"
    all_parameters = {
        "training_data": {
            "data_case": case_number,
            "shape": (sample_size, sample_dim),
            "seed": seed,
        },
        "calibration_data": {
            "data_case": cal_case_number,
            "shape": (cal_sample_size, cal_sample_dim),
            "seed": cal_seed,
        },
        "input_parameters": {
            "problem": problem,
            "theta_m": theta_m,
            "theta_v": theta_v[0],
            "lambda2": lambda2,
            "delta": delta,
        },
        "output_parameters": {
            "calibration_hsic": cal_hsic,
            "solver_min": sdp_model.solver_min,
            "solver_state": sdp_model.solver_state,
            "solver_time": sdp_model.solver_time,
            "solver_iter": sdp_model.solver_iter,
        },
    }
    with fs.open(params_FILE_PATH_OUT_S3, mode="w") as file_out:
        json.dump(all_parameters, file_out)

    model_FILE_PATH_OUT_S3 = FOLDER_PATH_OUT_S3 + "sdp_model.pkl"
    with fs.open(model_FILE_PATH_OUT_S3, mode="wb") as file_out:
        pickle.dump(sdp_model, file_out)
    
    log["try"] += 1

    return cal_hsic



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Argparse for BOBYQA optimisation.')
    parser.add_argument('--on', type=str)
    parser.add_argument('--case_number', type=int)
    parser.add_argument('--size', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--l_bound', type=float)
    parser.add_argument('--u_bound', type=float)
    parser.add_argument('--max_eval', type=int)
    parser.add_argument('--n_restarts', type=int)
    parser.add_argument('--cal_case_number', type=int)
    parser.add_argument('--cal_size', type=int)
    parser.add_argument('--cal_dim', type=int)
    parser.add_argument('--cal_seed', type=int)
    args = parser.parse_args()

    print(f"Starting optimisation for seed {args.seed}")

    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]

    settings = {
        "delta": 1e-3,
        "lambda2": 1,
        "problem": "Liang",
        "alpha": 0.05,
        "case_number": args.case_number,
        "sample_size": args.size,
        "sample_dim": args.dim,
        "seed": args.seed,
        "fs": s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL}),
        "lower_bound": args.l_bound,
        "upper_bound": args.u_bound,
        "max_eval": args.max_eval,
        "cal_case_number": args.cal_case_number,
        "cal_sample_size": args.cal_size,
        "cal_sample_dim": args.cal_dim,
        "cal_seed": args.cal_seed,
    }

    restart_optimise(num_restarts=args.n_restarts, settings=settings, on=args.on)