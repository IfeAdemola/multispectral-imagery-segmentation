import numpy as np
from utils import *

def rotate_90(array):
    """Rotate the array by 90 degrees clockwise."""
    return np.rot90(array, k=-1, axes=(0, 1)).copy()  # k=-1 for 90° clockwise

def rotate_180(array):
    """Rotate the array by 180 degrees."""
    return np.rot90(array, k=2, axes=(0, 1)).copy()  # k=2 for 180°

def rotate_270(array):
    """Rotate the array by 270 degrees clockwise."""
    return np.rot90(array, k=1, axes=(0, 1)).copy()

def add_gaussian_noise(array, mean=0, std=0.006):
    """Add Gaussian noise to the image array (mask is excluded as input to this function)"""
    noise = np.random.normal(mean, std, array.shape)
    noisy_image = (array + noise).copy()

    # Clip the values to ensure they stay within the range (0, 1)
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # Combine with the rest of the array
    # augmented_array = np.concatenate([noisy_image, array[:, :, 10:]], axis=-1).copy()
    return noisy_image.astype(array.dtype)
