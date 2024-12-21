import os
import s3fs
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np

# Function to extract images from PDF and return as a list
def pdf_to_images_from_s3(pdf_paths, fs):
    images = []
    for pdf_path in pdf_paths:
        # Open PDF from S3 using s3fs
        with fs.open(pdf_path, mode="rb") as file_in:
            # Read PDF content as bytes
            pdf_data = file_in.read()
            # Convert PDF bytes to image using PyMuPDF (fitz)
            images_from_pdf = pdf_to_images_with_pymupdf(pdf_data)
            images.append(images_from_pdf[0])  # Append the first page image
    return images

# Function to convert PDF data (in bytes) to images using PyMuPDF (fitz)
def pdf_to_images_with_pymupdf(pdf_data):
    # Open the PDF document from bytes
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load each page
        pix = page.get_pixmap()  # Render page to a pixmap (image)
        img = Image.open(io.BytesIO(pix.tobytes()))  # Convert pixmap to PIL Image
        images.append(img)
    return images

# Function to create a 3x3 grid
def create_image_grid(images, title, FILE_PATH_OUT_S3):
    fig, axs = plt.subplots(4, 3, figsize=(15, 15))  # Create a 3x3 grid of subplots
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

# Lengthscales to compute 900
variance_lengthscale = 1e-1
delta = 1e-3
lambda2 = 1
problem = "Liang"
alpha = 0.05

# Which training data
case_number = 5
sample_size = 100
sample_dim = 1
seed = 123

# Which calibration data
cal_case_number = 5
cal_sample_size = 100
cal_sample_dim = 1
cal_seed = 321

# Which calibration data
test_case_number = 5
test_sample_size = 300
test_sample_dim = 1
test_seed = 987

all_theta_v = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
pdf_paths = [
    f"luisito/these/sb_experiments/e2e_sdp/data_case_{case_number}/sample_shape_({sample_size},{sample_dim})/seed_{seed}/problem_{problem}/lambda2_{lambda2}/delta_{delta}/variance_lengthscale_{variance_lengthscale}/data_case_{cal_case_number}/sample_shape_({cal_sample_size},{cal_sample_dim})/seed_{cal_seed}/data_case_{test_case_number}/sample_shape_({test_sample_size},{test_sample_dim})/seed_{test_seed}/figure_bands_test.pdf"
    for variance_lengthscale in all_theta_v
] 

# Convert PDF files from S3 to images
images = pdf_to_images_from_s3(pdf_paths, fs)

# Create the image grid with a general title
create_image_grid(
    images,
    title="For lengthscale from 1e-9 to 1e2.",
    FILE_PATH_OUT_S3="luisito/these/sb_experiments/images/small_ls_test.pdf",
)
