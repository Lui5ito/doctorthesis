import os

import matplotlib.pyplot as plt

import s3fs

import utils


def extract_lengthscales(folder_path, fs):
    # List files in the folder
    file_list = fs.glob(
        folder_path + "variance_lengthscale_*"
    )  # Only match files starting with 'try_'
    print(file_list)

    return file_list


if __name__ == "__main__":
    import argparse

    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    parser = argparse.ArgumentParser(
        description="Argparse for BOBYQA optimisation of the lenthscale using k-folds."
    )
    parser.add_argument("--train", nargs="+", type=int)
    parser.add_argument("--calibration", nargs="+", type=int)
    parser.add_argument("--test", nargs="+", type=int)
    args = parser.parse_args()

    parameters_train = args.train
    parameters_cal = args.calibration
    parameters_test = args.test

    # Get train data
    X_train, y_train = utils.load_data(
        parameters_train[0],
        parameters_train[1],
        parameters_train[2],
        parameters_train[3],
        fs,
    )

    # Get calibration data
    X_cal, y_cal = utils.load_data(
        parameters_cal[0], parameters_cal[1], parameters_cal[2], parameters_cal[3], fs
    )

    # Get test data
    X_test, y_test = utils.load_data(
        parameters_test[0],
        parameters_test[1],
        parameters_test[2],
        parameters_test[3],
        fs,
    )

    delta = 1e-3
    lambda2 = 1
    problem = "Liang"
    alpha = 0.05

    model_FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{parameters_train[0]}/sample_shape_({parameters_train[1]},{parameters_train[2]})/seed_{parameters_train[3]}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/"
    list_lengthscales = extract_lengthscales(model_FOLDER_PATH_OUT_S3, fs)

    for folder_in in list_lengthscales:
        model_FILE_PATH_OUT_S3 = folder_in + "/sdp_model.pkl"
        sdp_model = utils.load_file(model_FILE_PATH_OUT_S3, fs)

        sdp_model.calibrate(X=X_cal, y=y_cal, alpha=alpha)

        mean_prediction, bands_prediction = sdp_model.predict(X=X_test)

        if parameters_train[2] == 1:
            figure = utils.plot(
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
                figure.savefig(
                    file_out,
                    transparent=True,
                    dpi="figure",
                    format="pdf",
                    metadata=None,
                    bbox_inches=None,
                    pad_inches=0.1,
                    facecolor="auto",
                    edgecolor="auto",
                    backend=None,
                )
                plt.close()

        elif parameters_train[2] == 2:
            figure = utils.plot_2d(
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
