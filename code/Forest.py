import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from typing import List, Dict, Any

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

def to_patches_old(file_path: str, patch_size: int, stride: int) -> None:
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

    for i in range(num_patches_height):
        for j in range(num_patches_width):
            start_i = i * stride_height
            start_j = j * stride_width
            patch = array[start_i:start_i + patch_height, start_j:start_j + patch_width, :]

            dir = "C:/Users/ifeol/Desktop/MASTERS/Thesis/multispectral-imagery-segmentation/data/"
            if "without_trees" in file_name:
                patch_filename = f"{dir}labels/{file_name}_{i* num_patches_width + j + 1}.pkl"
            else:
                patch_filename = f"{dir}data_patches/{file_name}_{i * num_patches_width + j + 1}.pkl"
            with open(patch_filename, 'wb') as f:
                pickle.dump(patch, f)


def to_patches(array_path, patch_size, stride):
    """
    Split an image into patches of a specified size with a given stride.

    Parameters:
    - array_path: Path to the pickle file containing the image array.
    - patch_size: Tuple representing the size of the patch (height, width).
    - stride: Tuple representing the stride (vertical, horizontal).

    Returns:
    - patch_filenames: List of paths to the saved patches.
    """

    # Load the image array from the pickle file
    with open(array_path, 'rb') as f:
        array = pickle.load(f)

    height, width, channels = array.shape
    patch_height, patch_width = patch_size
    stride_height, stride_width = stride

    # Calculate the number of patches in each dimension
    num_patches_height = (height - patch_height) // stride_height + 1
    num_patches_width = (width - patch_width) // stride_width + 1

    patch_filenames = []

    # Extract region and timestamp from the array_path
    _, filename = os.path.split(array_path)
    region = "_".join(filename.split("_")[:3]) #filename.split('_')[1]
    timestamp = filename.split('_')[3:6]
    timestamp = '_'.join(timestamp)

    # Determine directory based on filename
    if "without_trees" in filename:
        base_dir = "label_patches"
    else:
        base_dir = "data_patches"
    
    # Create directories for region and timestamp
    region_dir = os.path.join(base_dir, region)
    timestamp_dir = os.path.join(region_dir, timestamp)
    
    if not os.path.exists(region_dir):
        os.makedirs(region_dir)
    if not os.path.exists(timestamp_dir):
        os.makedirs(timestamp_dir)

    for i in range(num_patches_height):
        for j in range(num_patches_width):
            start_i = i * stride_height
            start_j = j * stride_width
            patch = array[start_i:start_i + patch_height, start_j:start_j + patch_width, :]

            # Save the patch to a pickle file
            patch_filename = os.path.join(timestamp_dir, f"{i * num_patches_width + j + 1}.pkl")
            with open(patch_filename, 'wb') as file:
                pickle.dump(patch, file)

            patch_filenames.append(patch_filename)
    
    return patch_filenames


class BayernForest(Dataset):
    """PyTorch dataset class for SENTINEL-2 satellite images"""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Get list of all data and label patch filenames
        self.data_files = [os.path.join(root, filename)
                           for root, _, filenames in os.walk(os.path.join(data_dir, 'data_patches'))
                           for filename in filenames if filename.endswith('.pkl')]

        self.label_files = [os.path.join(root, filename)
                            for root, _, filenames in os.walk(os.path.join(data_dir, 'label_patches'))
                            for filename in filenames if filename.endswith('.pkl')]

    def __getitem__(self, idx):
        """Get a single example from the dataset"""
        with open(self.data_files[idx], 'rb') as f:
            data_patch = pickle.load(f)

        with open(self.label_files[idx], 'rb') as f:
            label_patch = pickle.load(f)

        sample = {'image': data_patch, 'mask': label_patch}

        if self.transform:
            sample = self.transform(sample)

        return sample 

    def __len__(self):
        "Get number of samples in the dataset"
        return len(self.data_files)
    
