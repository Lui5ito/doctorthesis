import universalbands as ub
import numpy as np
from sklearn.gaussian_process import kernels
import pickle
import os
import s3fs
import matplotlib.pyplot as plt
from utils import load_file
from scipy.spatial.distance import cdist, pdist, squareform


def retrieve_min(case_number, sample_size, sample_dim, seed):
    # Retrieve data path
    data_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
    data_FILE_PATH_IN_S3 = data_FOLDER_PATH_IN_S3 + "data.npz"
    X_train, y_train = load_file(data_FILE_PATH_IN_S3, fs)
    # Compute the pairwise Euclidean distances
    dists = pdist(X_train, metric="euclidean")

    # Find the minimum distance (excluding 0)
    min_distance = np.min(dists)

    return min_distance


if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Lengthscales to compute 900
    list_lengthscales = np.round(np.linspace(1e-3, 1, 900), 4)
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

    number_of_folds = 10

    width = 1
    height = 1
    figs, axs = plt.subplots(height, width, figsize=(50*width, 25*height))

    train_hsic = []
    calibration_hsic = []
    kfold_hsic = []
    plot_min_dist = True

    for variance_lengthscale in list_lengthscales:
        #FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/e2e_sdp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/data_case_{test_case_number}/sample_shape_({test_sample_size},{test_sample_dim})/seed_{test_seed}/"
        fully_trained_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/ub_models/simultaneous/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/"
        model_FILE_PATH_OUT_S3 = fully_trained_FOLDER_PATH_IN_S3 + "sdp_model.pkl"
        sdp_model = load_file(model_FILE_PATH_OUT_S3, fs)
        train_hsic.append(sdp_model.metrics["hsic_train_full"])
        calibration_hsic.append(sdp_model.metrics[f"hsic_cal_{calibration_seed}_full"])
        kfold_hsic.append(sdp_model.metrics[f"hsic_train_kfold_{number_of_folds}"])


    axs.scatter(list_lengthscales, train_hsic, color="blue", label="Training Data.", alpha=0.7, marker='o', s=35)
    axs.scatter(list_lengthscales, calibration_hsic, color="red", label="Calibration Data.", alpha=0.7, marker='x', s=35)
    axs.scatter(list_lengthscales, kfold_hsic, color="green", label="Training Data, 10-folds.", alpha=0.7, marker='v', s=35)
    
    if train_hsic:
        max_train_hsic_index = np.argmax(train_hsic)
        max_train_x = list_lengthscales[max_train_hsic_index]
        max_train_y = train_hsic[max_train_hsic_index]
        axs.plot([max_train_x, max_train_x], [0, max_train_y], color="blue", linestyle="--", alpha=0.8, label="Max Train HSIC")

    if calibration_hsic:
        max_cal_hsic_index = np.argmax(calibration_hsic)
        max_cal_x = list_lengthscales[max_cal_hsic_index]
        max_cal_y = calibration_hsic[max_cal_hsic_index]
        axs.plot([max_cal_x, max_cal_x], [0, max_cal_y], color="red", linestyle="--", alpha=0.8, label="Max Cal HSIC")
    
    if kfold_hsic:
        max_kfold_hsic_index = np.argmax(kfold_hsic)
        max_kfold_x = list_lengthscales[max_kfold_hsic_index]
        max_kfold_y = kfold_hsic[max_kfold_hsic_index]
        axs.plot([max_kfold_x, max_kfold_x], [0, max_kfold_y], color="green", linestyle="--", alpha=0.8, label="Max 10-folds HSIC")
    
    if plot_min_dist:
        min_dist = retrieve_min(case_number, sample_size, sample_dim, seed)
        axs.axvline(x=min_dist, ymin=0.1, ymax=0.9, color="black", linestyle="-", alpha=0.8, label="Min X distances.")

    axs.set_title(f"Training seed: {seed}; Calibration seed: {calibration_seed}.")
    axs.set_xlabel('Lengthscales')
    axs.set_ylabel('e-HSIC')
    axs.grid(True, linestyle='--', alpha=0.2)
    axs.set_xticks(list_lengthscales)
    axs.tick_params(axis='x', labelsize=4, labelrotation=90)
    axs.legend(loc="upper right", fontsize="small", framealpha=0.8)

    # Finish the figure and save
    figs.suptitle('e-HSIC vs lengthscales for training, calibration and 10fold-training data.', x=0.5, y=0.999, size = 16, weight = 'bold')
    figs.tight_layout()

    FILE_PATH_OUT_S3 = "luisito/these/sb_experiments/images/" + f"e-HISC_vs_Lengthscales_{number_of_folds}fold.pdf"
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        figs.savefig(file_out, format="pdf", transparent=True, dpi=600)