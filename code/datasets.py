import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import pickle
import os

def load_array(file_path: str) -> np.ndarray:
    """
    Load a NumPy array from a pickle file.

    Parameters:
    file_path (str): The path to the pickle file containing the NumPy array.

    Returns:
    np.ndarray: The NumPy array loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        array = pickle.load(f)
    return array

def plot_image(array):

    image_to_plot = array[..., :3]


    # Plotting the image
    min_vals = np.min(image_to_plot, axis=(0, 1))
    max_vals = np.max(image_to_plot, axis=(0, 1))

    image_scaled = (image_to_plot - min_vals) / (max_vals - min_vals)
    image_scaled = np.clip(image_scaled, 0, 1)  # Clip values to [0, 1] range

    # Apply gamma correction to enhance contrast
    gamma = 0.4  # Adjust this value to control contrast
    image_corrected = np.power(image_scaled, gamma)

    # Adjust gain to control brightness
    gain = [1.2, 1.0, 0.8]  # Adjust gain for each channel (R, G, B)
    image_corrected = image_corrected * gain

    # Clip values to [0, 1] range
    image_corrected = np.clip(image_corrected, 0, 1)

    # Rearrange channels from BGR to RGB
    image_corrected_rgb = image_corrected[..., ::-1]
    image_corr = image_corrected_rgb[50:100, 50:100]
    # Plotting the corrected image
    plt.figure(figsize=(2,2)) 
    # plt.figure(figsize=(15, 15))
    # plt.imshow(image_corrected_rgb)
    plt.imshow(image_corr)

    plt.axis('off')  # Turn off axis
    plt.show()


def to_patches(file_path: str, patch_size: int, stride: int) -> None:
    """
    Convert an array to patches of a given size and stride.

    Parameters:
    - array (np.ndarray): Input array to convert to patches.
    - patch_size (int): Size of each patch.
    - stride (int): Stride between patches.

    Returns:
    - patches (np.ndarray): Array of patches with shape (num_patches, patch_size, patch_size, array.shape[-1]).
    """
    array = load_array(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Get file name without extension

    height, width, _ =  array.shape
    patch_height, patch_width = patch_size, patch_size
    stride_height, stride_width = stride, stride

    # Calculate the number of patches in each dimension
    num_patches_height = (height - patch_height) // stride_height + 1
    num_patches_width = (width - patch_width) // stride_width + 1

    patches = []
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            start_i = i * stride_height
            start_j = j * stride_width
            patch = array[start_i:start_i + patch_height, start_j:start_j + patch_width, :]

            dir = "C:/Users/ifeol/Desktop/MASTERS/Thesis/multispectral-imagery-segmentation/data/clean_data/"
            if "without_trees" in file_name:
                patch_filename = f"{dir}labels/{file_name}_{i* num_patches_width + j + 1}.pkl"
            else:
                # patch_filename = f"{dir}data_patches/{file_name}_{i * num_patches_width + j + 1}.pkl"
                patch_filename = f"{dir}data_patches/{file_name}_{i * num_patches_width + j + 1}.pkl"
             
            with open(patch_filename, 'wb') as f:
                pickle.dump(patch, f)

def main():
    file_list = []
    folder_path = "C:/Users/ifeol/Desktop/MASTERS/Thesis/multispectral-imagery-segmentation/data/clean_data"
    for file_name in os.listdir(folder_path):
            if file_name.endswith(".pkl"):
                file_list.append(folder_path + "/" + file_name)
    for file_name in file_list:
        to_patches(file_name, patch_size=64, stride=16)

if __name__ == "__main__":
    main()