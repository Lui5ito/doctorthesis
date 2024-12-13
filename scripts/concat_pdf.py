import os
import s3fs
import matplotlib.pyplot as plt
from pdf2image import convert_from_bytes
import numpy as np


# Function to extract images from PDF and return as a list
def pdf_to_images_from_s3(pdf_paths, fs):
    images = []
    for pdf_path in pdf_paths:
        # Open PDF from S3 using s3fs
        with fs.open(pdf_path, mode="rb") as file_in:
            # Read PDF content as bytes
            pdf_data = file_in.read()
            # Convert the PDF data to images using pdf2image
            images_from_pdf = convert_from_bytes(pdf_data, first_page=1, last_page=1)
            images.append(images_from_pdf[0])  # Append the first page image
    return images


# Function to create a 3x3 grid
def create_image_grid(images, title, FILE_PATH_OUT_S3):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # Create a 3x3 grid of subplots
    fig.suptitle(title, fontsize=16)  # Set the title for the entire grid

    # Flatten the 2D grid of axes to make it easier to iterate
    axs = axs.flatten()

    # Loop through the images and plot them on the grid
    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis("off")  # Remove axis for better visualization

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust title placement
    with fs.open(FILE_PATH_OUT_S3, mode="wb") as file_out:
        fig.savefig(file_out, format="pdf", transparent=True, dpi=600)
    print("I did it!")


# S3 settings (assuming AWS S3 credentials and endpoint are set)
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

# List of file paths in S3 (adjust your file paths as needed)
number_of_lengthscale = 10
all_theta_v = np.round(np.linspace(0.1, 1, number_of_lengthscale), 2)
pdf_paths = [
    f"luisito/these/sb_experiments/which_kernel/results/three_parameters/lengthscale_{i}/test.pdf"
    for i in all_theta_v
]  # Adjust your filenames

# Convert PDF files from S3 to images
images = pdf_to_images_from_s3(pdf_paths, fs)

# Create the image grid with a general title
create_image_grid(
    images,
    title="The kernel used here follows: LsVarNug (GP) - LsVarNug (s) - LsVarNug (SDP).",
    FILE_PATH_OUT_S3="luisito/these/sb_experiments/which_kernel/results/three_parameters/all_ls.pdf",
)
