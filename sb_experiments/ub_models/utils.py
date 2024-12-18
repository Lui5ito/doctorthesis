import json
import pickle
import numpy as np

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