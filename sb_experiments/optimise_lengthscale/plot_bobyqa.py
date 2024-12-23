"""
This file is here to plot the results of file 'sdp_sim_optimise_hsic.py'
"""
import os
import s3fs
import re  # Import regular expressions for extracting 'try' numbers
import json
import matplotlib.pyplot as plt


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


def plot(all_data):

    # Create a figure with 2 subplots: one for lengthscale and one for hsic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Loop through each restart in all_data
    for restart, data in all_data.items():
        try_ids = data["try_ids"]
        try_lengthscale = data["try_lengthscale"]
        try_hsic = data["try_hsic"]
        
        # Left subplot: Lengthscale vs IDs
        ax1.plot(try_ids, try_lengthscale, label=restart, marker='o')

        # Right subplot: HSIC vs IDs
        ax2.plot(try_ids, try_hsic, label=restart, marker='o')

    # Customize left subplot (Lengthscale vs IDs)
    ax1.set_title('Lengthscale vs Try IDs')
    ax1.set_xlabel('Try ID')
    ax1.set_ylabel('Lengthscale')
    ax1.legend(title='Restarts')

    # Customize right subplot (HSIC vs IDs)
    ax2.set_title('HSIC vs Try IDs')
    ax2.set_xlabel('Try ID')
    ax2.set_ylabel('HSIC')
    ax2.legend(title='Restarts')

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()

    # Show the plot
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

    for restart in range(restart_num):

        FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_training/restart_{restart}/"

        try_ids = extract_try_ids(FOLDER_PATH_IN_S3)
        try_lengthscale = []
        try_hsic = []

        for num_id in try_ids:
            try_FOLDER_PATH_IN_S3 = f"luisito/these/sb_experiments/optimise_lengthscale/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/on_training/restart_{restart}/try_{num_id}/"
            FILE_PATH_IN_S3 = try_FOLDER_PATH_IN_S3 + "all_parameters.json"
            with fs.open(FILE_PATH_IN_S3, mode="r") as file_in:
                all_parameters = json.load(file_in)
            try_lengthscale.append(all_parameters["input_parameters"]["theta_v"])
            try_hsic.append(all_parameters["output_parameters"]["training_hsic"])
        
        all_data[f"restart_{restart}"] = {
            "try_ids": try_ids,
            "try_lengthscale": try_lengthscale,
            "try_hsic": try_hsic,
        }
    
    my_fig = plot(all_data)

    FILE_PATH_OUT_S3 = "luisito/these/sb_experiments/images/" + f"bobyqa_path_training_seed_{seed}.pdf"
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        my_fig.savefig(file_out, format="pdf", transparent=True, dpi=600)
    







