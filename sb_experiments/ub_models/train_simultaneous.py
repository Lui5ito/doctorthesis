import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import pickle
import os
import s3fs
from utils import load_file
from multiprocessing import Pool, cpu_count


def process_variance_lengthscale(variance_lengthscale, case_number, sample_size, sample_dim, seed, X_train, y_train, theta_m, lambda2, delta, problem):
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    print("Variance lengthscale: ", variance_lengthscale)
    
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
    
    # Train the model
    sdp_model.train(X=X_train, y=y_train)
    sdp_model.metrics["full_train_hsic"] = sdp_model.compute_HSIC(X=X_train, y=y_train, extra=False)

    # Add calibration HSIC
    calibration_cases = [5]
    calibration_all_sample_sizes = [100]
    calibration_all_sample_dims = [1]
    calibration_all_sample_seeds = [321, 322, 323]
    '''
    for cal_case_number in calibration_cases:
        for cal_sample_size in calibration_all_sample_sizes:
            for cal_sample_dim in calibration_all_sample_dims:
                for cal_seed in calibration_all_sample_seeds:
                    cal_data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/"
                    cal_data_FILE_PATH_IN_S3 = cal_data_FOLDER_PATH_IN_S3 + "data.npz"
                    X_cal, y_cal = load_file(cal_data_FILE_PATH_IN_S3, fs)
                    sdp_model.metrics[f"full_cal_{cal_seed}_hsic"] = sdp_model.compute_HSIC(X=X_cal, y=y_cal, extra=False)
    '''
    # Save whole SDP model
    model_FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/ub_models/simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
    model_FILE_PATH_OUT_S3 = model_FOLDER_PATH_OUT_S3 + "sdp_model.pkl"
    print(f"Starting to save model for lengthscale {variance_lengthscale}...")
    try:
        with fs.open(model_FILE_PATH_OUT_S3, mode="wb") as file_out:
            pickle.dump(sdp_model, file_out)
        print(f"Successfully saved model for lengthscale {variance_lengthscale}.")
    except Exception as e:
        print(f"Error saving model for lengthscale {variance_lengthscale}: {e}")


if __name__ == "__main__":
    # Lengthscales to compute
    length_scale_list = np.round(np.linspace(0.01, 1, 5), 3)
    delta = 1e-3
    lambda2 = 1
    problem = "Liang"

    # Which training data
    cases = [5]
    all_sample_sizes = [100]
    all_sample_dims = [1]
    all_sample_seeds = [123]

    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Create pool of workers
    pool = Pool(processes=(cpu_count() - 2))

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

                    for variance_lengthscale in length_scale_list:
                        print(f"Launch lengthscale {variance_lengthscale}.")
                        pool.apply_async(
                            process_variance_lengthscale,
                            args=(variance_lengthscale, case_number, sample_size, sample_dim, seed, X_train, y_train, theta_m, lambda2, delta, problem),
                        )

    print("All done.")
    pool.close()
    pool.join()
