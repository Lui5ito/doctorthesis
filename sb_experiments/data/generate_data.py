"""
This file generates data for all my experiments and stores it in bucket/data.
The data generating functions are in the universalbands package.
"""

import universalbands as ub
import numpy as np
import os
import s3fs

if __name__ == "__main__":

    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    import argparse
    parser = argparse.ArgumentParser(description='Argparse for data generation.')
    parser.add_argument('--size', nargs = '+', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()

    # Which data to generate
    cases = [10, 11]
    all_sample_sizes = args.size
    all_sample_dims = [1]
    all_sample_seeds = args.seeds
    print(all_sample_sizes)
    print(all_sample_seeds)
    # Generate the data
    for case_number in cases:
        for sample_size in all_sample_sizes:
            for sample_dim in all_sample_dims:
                for seed in all_sample_seeds:
                    # Construct file path
                    FOLDER_PATH_OUT_S3 = f"luisito/these/sb_experiments/data/case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/"
                    FILE_PATH_OUT_S3 = FOLDER_PATH_OUT_S3 + "data.npz"

                    # Check if the file already exists
                    if fs.exists(FILE_PATH_OUT_S3):
                        print(f"File {FILE_PATH_OUT_S3} already exists. Skipping...")
                        continue

                    # Generate the data
                    X, y = ub.data_generation.synthetic_data(
                        case=f"case_{case_number}",
                        sample_size=sample_size,
                        sample_dim=sample_dim,
                        seed=seed,
                    )

                    # Save the data
                    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
                        np.savez(file_out, X=X, y=y)
                        print(f"Data saved to {FILE_PATH_OUT_S3}")
