import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import pickle
import os
import s3fs
from utils import load_file
from multiprocessing import Pool, cpu_count
from functools import partial


def process_variance_lengthscale(variance_lengthscale, case_number, sample_size, sample_dim, seed, X_train, y_train, theta_m, lambda2, delta, problem, calibration_data):
    
    # Define the model
    sdp_model = ub.UniversalFunctionAndBandsRegressor(
        mean_kernel=kernels.Matern(length_scale=theta_m["posterior_lengthscale"], length_scale_bounds=(1e-5, 1e5), nu=2.5),
        variance_kernel=kernels.Matern(length_scale=variance_lengthscale, length_scale_bounds=(1e-5, 1e5), nu=2.5),
        lambda2=lambda2,
        delta=delta,
        s=theta_m["posterior_training_norm"],
        problem=problem,
        checkSDP=False,
        verbose=False
    )

    # Train the model
    sdp_model.train(X=X_train, y=y_train)
    sdp_model.metrics["hsic_train_full"] = sdp_model.compute_HSIC(X=X_train, y=y_train, extra=False)
    for _, data in calibration_data.items():
        X_cal = data[0]
        y_cal = data[1]
        cal_seed = data[2]
        sdp_model.metrics[f"hsic_cal_{cal_seed}_full"] = sdp_model.compute_HSIC(X=X_cal, y=y_cal, extra=False)

    print(f"Done computing lengthscale {variance_lengthscale}.")

    return sdp_model


if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Lengthscales to compute 900
    list_lengthscales = np.round(np.linspace(1e-3, 1, 900), 4)
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    # Which training data
    cases = [5]
    all_sample_sizes = [100]
    all_sample_dims = [1]
    all_sample_seeds = [123]

    # Which calibration data
    calibration_cases = [5]
    calibration_all_sample_sizes = [100]
    calibration_all_sample_dims = [1]
    calibration_all_sample_seeds = [321, 322, 323]

    for case_number in cases:
        for sample_size in all_sample_sizes:
            for sample_dim in all_sample_dims:
                for seed in all_sample_seeds:
                    # Retrieve data path
                    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
                    X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)

                    # Retrieve GP model path
                    gp_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/gp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    gp_FILE_PATH_IN_S3 = gp_FOLDER_PATH_IN_S3 + "optimized_parameters.json"
                    theta_m = load_file(gp_FILE_PATH_IN_S3, fs)

                    calibration_data = {}
                    for cal_case_number in calibration_cases:
                        for cal_sample_size in calibration_all_sample_sizes:
                            for cal_sample_dim in calibration_all_sample_dims:
                                for cal_seed in calibration_all_sample_seeds:
                                    cal_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/"
                                    cal_data_FILE_PATH_IN_S3 = cal_data_FOLDER_PATH_IN_S3 + "data.npz"
                                    X_cal, y_cal = load_file(cal_data_FILE_PATH_IN_S3, fs)
                                    calibration_data[f"seed_{cal_seed}"] = [X_cal, y_cal, cal_seed]

                    # Convert my process function from multiple to ONE argument.
                    process_variance_lengthscale_one_param = partial(process_variance_lengthscale, case_number=case_number, sample_size=sample_size, sample_dim=sample_dim, seed=seed, X_train=X_train, y_train=y_train, theta_m=theta_m, lambda2=lambda2, delta=delta, problem=problem, calibration_data=calibration_data)

                    with Pool(processes=(cpu_count() - 10)) as pool:
                        results = list(pool.imap(process_variance_lengthscale_one_param, list_lengthscales,))

                    # Save whole SDP model
                    for sdp_model, variance_lengthscale in zip(results, list_lengthscales):
                        model_FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/ub_models/simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
                        model_FILE_PATH_OUT_S3 = model_FOLDER_PATH_OUT_S3 + "sdp_model.pkl"
                        with fs.open(model_FILE_PATH_OUT_S3, mode="wb") as file_out:
                            pickle.dump(sdp_model, file_out)
