import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import json
import pickle
import os
import s3fs
import nlopt
from functools import partial
from sklearn.model_selection import KFold
from universalbands.metrics.energy_hsic import Energy_HSIC
import time


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

def restart_optimise(num_restarts, settings):

    # Define bounds for the parameters.
    lower_bounds = np.array([settings["lower_bound"]])
    upper_bounds = np.array([settings["upper_bound"]])

    last_theta = []
    last_hsic = []

    for restart in range(num_restarts):
        log = {"restart": restart, "try": 0}
        
        objective_function_one_param = partial(objective_function, settings=settings, log=log)
        
        initial_theta_v = np.random.uniform(lower_bounds.flatten(), upper_bounds.flatten(), settings["sample_dim"])
        opt = nlopt.opt(nlopt.LN_BOBYQA, len(initial_theta_v))
        opt.set_lower_bounds(lower_bounds.flatten())
        opt.set_upper_bounds(upper_bounds.flatten())
        opt.set_max_objective(objective_function_one_param)
        opt.set_maxeval(settings["max_eval"])
        opt.set_ftol_rel(1e-6)
        opt.set_ftol_abs(1e-6)
        opt.set_xtol_rel(1e-6)
        opt.set_xtol_abs(1e-6)
        # Actually solve
        theta_v_opt = opt.optimize(initial_theta_v)
        max_hsic = opt.last_optimum_value()

        last_theta.append(theta_v_opt)
        last_hsic.append(max_hsic)
    
    return last_theta, last_hsic

def objective_function(theta_v, grad, settings, log):
    # Unpack settings
    delta = settings["delta"]
    lambda2 = settings["lambda2"]
    problem = settings["problem"]
    case_number = settings["case_number"]
    sample_size = settings["sample_size"]
    sample_dim = settings["sample_dim"]
    seed = settings["seed"]
    num_folds = settings["num_folds"]
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

    # Compute k_fold HSIC
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    folded_AE = []
    folded_IW = []

    for k, (train_index, test_index) in enumerate(kf.split(X_train)):
        # Define the model
        sdp_model = ub.UniversalFunctionAndBandsRegressor(
            mean_kernel=kernels.Matern(length_scale=theta_m["posterior_lengthscale"], length_scale_bounds=(1e-5, 1e5), nu=2.5),
            variance_kernel=kernels.Matern(length_scale=theta_v, length_scale_bounds=(1e-5, 1e5), nu=2.5),
            lambda2=lambda2,
            delta=delta,
            s=theta_m["posterior_training_norm"],
            problem=problem,
            checkSDP=False,
            verbose=True,
        )
        # Train the model
        sdp_model.train(X=X_train[train_index], y=y_train[train_index])
        hsic_value, abs_errors, half_width = sdp_model.compute_HSIC(X=X_train[test_index], y=y_train[test_index], extra=True)
        folded_AE.append(abs_errors)
        folded_IW.append(half_width)

    folded_AE = np.vstack(folded_AE)
    folded_IW = np.vstack(folded_IW)
    kfold_hsic = Energy_HSIC(abs_error=folded_AE, width=folded_IW).compute()
    
    log["try"] += 1

    return kfold_hsic


def save_best_model(theta_v, settings, bobyqa_time):
    # Unpack settings
    delta = settings["delta"]
    lambda2 = settings["lambda2"]
    problem = settings["problem"]
    case_number = settings["case_number"]
    sample_size = settings["sample_size"]
    sample_dim = settings["sample_dim"]
    seed = settings["seed"]
    fs = settings["fs"]
    
    # Get train data
    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
    X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)

    # Get theta_m
    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"
    theta_m = load_file(gp_FILE_PATH_IN_S3, fs)

    sdp_model = ub.UniversalFunctionAndBandsRegressor(
            mean_kernel=kernels.Matern(length_scale=theta_m["posterior_lengthscale"], length_scale_bounds=(1e-5, 1e5), nu=2.5),
            variance_kernel=kernels.Matern(length_scale=theta_v, length_scale_bounds=(1e-5, 1e5), nu=2.5),
            lambda2=lambda2,
            delta=delta,
            s=theta_m["posterior_training_norm"],
            problem=problem,
            checkSDP=False,
            verbose=True,
        )
    sdp_model.train(X=X_train, y=y_train)
    sdp_model.metrics["bobyqa_time"] = bobyqa_time

    model_FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{theta_v}/"
    model_FILE_PATH_OUT_S3 = model_FOLDER_PATH_OUT_S3 + "sdp_model.pkl"
    with fs.open(model_FILE_PATH_OUT_S3, mode="wb") as file_out:
        pickle.dump(sdp_model, file_out)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Argparse for BOBYQA optimisation of the lenthscale using k-folds.')
    parser.add_argument('--case_number', type=int)
    parser.add_argument('--size', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_folds', type=int)
    parser.add_argument('--l_bound', nargs='+', type=float)
    parser.add_argument('--u_bound', nargs='+', type=float)
    parser.add_argument('--max_eval', type=int)
    parser.add_argument('--n_restarts', type=int)
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
        "num_folds": args.num_folds,
        "fs": s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL}),
        "lower_bound": args.l_bound,
        "upper_bound": args.u_bound,
        "max_eval": args.max_eval,
    }

    start = time.time()
    best_theta, best_hsic = restart_optimise(num_restarts=args.n_restarts, settings=settings)
    end = time.time()
    elapsed = (end - start)

    index_hsic = best_hsic.index(max(best_hsic))
    print(f"Best lengthscale found is {best_theta[index_hsic]}.")

    # Save best model
    save_best_model(theta_v=best_theta[index_hsic], settings=settings, bobyqa_time=elapsed)
