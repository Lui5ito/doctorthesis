import universalbands as ub
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels
import os
import s3fs
import json


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


if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Import data
    case_number = 5
    sample_size = 100
    sample_dim = 1
    X_train, y_train = ub.data_generation.synthetic_data(
        case=f"case_{case_number}",
        sample_size=sample_size + 1,
        sample_dim=sample_dim,
        seed=123,
    )
    X_cal, y_cal = ub.data_generation.synthetic_data(
        case=f"case_{case_number}",
        sample_size=sample_size + 2,
        sample_dim=sample_dim,
        seed=321,
    )

    # Récupération de theta_m
    FILE_PATH_IN_S3 = (
        "luisito/these/sb_experiments/train_gp/results/gp_with_LsNug/"
        # "luisito/these/sb_experiments/train_gp/results/gp_with_LsVarNug/"
        + "optimized_parameters.json"
    )
    with fs.open(FILE_PATH_IN_S3, mode="r") as file_in:
        theta_m = json.load(file_in)

    number_of_lengthscale = 10
    all_theta_v = np.round(np.linspace(0.1, 1, number_of_lengthscale), 2)
    all_hsic_train = []
    all_hsic_cal = []
    for theta_v in all_theta_v:
        # Récupération de gammahat, Ahat et Vhat.
        FILE_PATH_IN_S3 = (
            f"luisito/these/sb_experiments/which_kernel/results/LsNug_LsNug_Ls_parameters/lengthscale_{theta_v}/"
            # f"luisito/these/sb_experiments/which_kernel/results/ls_and_nug_parameters/lengthscale_{theta_v}/"
            # f"luisito/these/sb_experiments/which_kernel/results/oneless_parameter/lengthscale_{theta_v}/"
            # f"luisito/these/sb_experiments/which_kernel/results/three_parameters/lengthscale_{theta_v}/"
            + "train.npz"
        )
        with fs.open(FILE_PATH_IN_S3, mode="rb") as file_in:
            construction_objects = np.load(file_in)
            gammahat = construction_objects["gammahat"]
            Ahat = construction_objects["Ahat"]
            Vhat = construction_objects["Vhat"]

        # Recupération des hyperparametres
        FILE_PATH_IN_S3 = (
            f"luisito/these/sb_experiments/which_kernel/results/LsNug_LsNug_Ls_parameters/lengthscale_{theta_v}/"
            # f"luisito/these/sb_experiments/which_kernel/results/ls_and_nug_parameters/lengthscale_{theta_v}/"
            # f"luisito/these/sb_experiments/which_kernel/results/oneless_parameter/lengthscale_{theta_v}/"
            # f"luisito/these/sb_experiments/which_kernel/results/three_parameters/lengthscale_{theta_v}/"
            + "metadata.json"
        )
        with fs.open(FILE_PATH_IN_S3, mode="r") as file_in:
            metadata = json.load(file_in)

        # Working on X_train and y_train.
        # Reconstruction de la moyenne.
        mean_posterior_kernel = kernels.Matern(
            length_scale=theta_m["posterior_lengthscale"],
            length_scale_bounds=(1e-5, 1e5),
            nu=2.5,
        )
        mean_posterior_gram_matrix = mean_posterior_kernel(X=X_train).T
        mean_estimator_prediction = np.matmul(mean_posterior_gram_matrix, gammahat)

        # Reconstruction de la variance.
        variance_kernel = kernels.Matern(
            length_scale=theta_v,
            length_scale_bounds=(1e-5, 1e5),
            nu=2.5,
        )
        variance_gram_matrix = variance_kernel(X=X_train)
        Phi_pred = np.linalg.solve(a=Vhat.T, b=variance_gram_matrix)
        variance_estimator_prediction = np.diag(Phi_pred.T @ Ahat @ Phi_pred).reshape(
            -1, 1
        )
        predicted_score_function = np.sqrt(
            variance_estimator_prediction + metadata["input"]["delta"]
        )

        abs_errors = np.abs(y_train - mean_estimator_prediction)
        e_hsic_train = compute_hsic(abs_errors, predicted_score_function)
        all_hsic_train.append(e_hsic_train)

        # Working on X_cal and y_cal.
        # Reconstruction de la moyenne.
        mean_posterior_kernel = kernels.Matern(
            length_scale=theta_m["posterior_lengthscale"],
            length_scale_bounds=(1e-5, 1e5),
            nu=2.5,
        )
        mean_posterior_gram_matrix = mean_posterior_kernel(X=X_train, Y=X_cal).T
        mean_estimator_prediction = np.matmul(mean_posterior_gram_matrix, gammahat)

        # Reconstruction de la variance.
        variance_kernel = kernels.Matern(
            length_scale=theta_v,
            length_scale_bounds=(1e-5, 1e5),
            nu=2.5,
        )
        variance_gram_matrix = variance_kernel(X=X_train, Y=X_cal)
        Phi_pred = np.linalg.solve(a=Vhat.T, b=variance_gram_matrix)
        variance_estimator_prediction = np.diag(Phi_pred.T @ Ahat @ Phi_pred).reshape(
            -1, 1
        )
        predicted_score_function = np.sqrt(
            variance_estimator_prediction + metadata["input"]["delta"]
        )

        abs_errors = np.abs(y_cal - mean_estimator_prediction)
        e_hsic_cal = compute_hsic(abs_errors, predicted_score_function)
        all_hsic_cal.append(e_hsic_cal)

    FILE_PATH_OUT_S3 = (
        "luisito/these/sb_experiments/which_kernel/results/LsNug_LsNug_Ls_parameters/"
        # "luisito/these/sb_experiments/which_kernel/results/ls_and_nug_parameters/"
        # "luisito/these/sb_experiments/which_kernel/results/oneless_parameter/"
        # "luisito/these/sb_experiments/which_kernel/results/three_parameters/"
        + "hisc_values.npz"
    )
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        np.savez(
            file_out,
            lengthscales=all_theta_v,
            hsic_train=np.array(all_hsic_train),
            hsic_cal=np.array(all_hsic_cal),
        )

    plt.figure(figsize=(8, 6))
    # Scatter for training data (blue)
    plt.scatter(all_theta_v, all_hsic_train, color="blue", label="Training Data", alpha=0.7, marker='o', s=40)
    # Scatter for calibration data (red)
    plt.scatter(all_theta_v, all_hsic_cal, color="red", label="Calibration Data", alpha=0.7, marker='x', s=40)
    # Adding title and labels
    plt.title('HSIC vs Lengthscales', fontsize=16)
    plt.xlabel('Lengthscales', fontsize=12)
    plt.ylabel('E-HSIC', fontsize=12)
    # Customize grid and ticks
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(all_theta_v)
    # Adding a legend
    plt.legend()
    # Show the plot
    plt.tight_layout()

    FILE_PATH_OUT_S3 = "luisito/these/sb_experiments/which_kernel/results/LsNug_LsNug_Ls_parameters/" + "plot_hsic_vs_ls.pdf"
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        plt.savefig(file_out, format="pdf", transparent=True, dpi=600)

    print("Finished!")
