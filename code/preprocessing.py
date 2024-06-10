import os
import pickle
import shutil
import numpy as np
import re
import random 
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import *

def get_file_paths(root_dir):
    """
    Retrieve the paths of all .pkl files in a specified directory.

    This function takes a directory path as input and returns a list of 
    full paths to all files in that directory that have a `.pkl` extension.

    Parameters
    ----------
    root_dir : str
        The root directory to search for `.pkl` files.

    Returns
    -------
    file_paths : list of str
        A list containing the full paths of all `.pkl` files in the given 
        directory. If no `.pkl` files are found, the list will be empty.
    """
    file_paths = [os.path.join(root_dir, file_name) for file_name in os.listdir(root_dir) if file_name.endswith(".pkl")]
    return file_paths

def load_array(file_path):
    """
    Load and return a numpy array from a pickle file.

    Parameters
    ----------
    file_path : str
        The path to the pickle file from which the array or object 
        will be loaded.

    Returns
    -------
    np.array
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def combine_data_label(root_dir):
    """Combine the label arrays to the data, so has to create the final label (from 13 channels to 15 or so)
    input  8 files
    output 6 files"""
    file_paths = get_file_paths(root_dir=root_dir)

    arrays = {}
    for path in file_paths:
        array = load_array(path)

        # Extract file name without extension (.pkl) as key
        key = os.path.splitext(os.path.basename(path))[0]

        # Store arrays in dictionary (all data and label files)
        arrays[key] = array

    for key in arrays.keys():
        # Check if the key contains 'without_trees'
        if 'without_trees' in key:
            # Extract cardinal point from the key
            cardinal_point = key.split('_')[-3]
            
            # Find matching keys without 'without_trees' and the same cardinal point
            matching_keys = [k for k in arrays.keys() if cardinal_point in k and 'without_trees' not in k]
            
            # Iterate through matching keys and update arrays
            for match_key in matching_keys:
                # Add the first two channels from the 'without_trees' array to the matching array
                arrays[match_key] = np.concatenate([arrays[match_key], arrays[key][...]], axis=-1)

    mask = 'without_trees'
    arrays = {key: value for key, value in arrays.items() if mask not in key}

    return   # dict

def generate_segmentation_masks(arrays):
    """
    Generate and update segmentation masks for a collection of multidimensional (3D) arrays.

    This function processes each array in the given dictionary by extracting timestamps,
    updating deforestation masks based on specific conditions, adding new vegetation and 
    seasonal masks, and finally removing unnecessary attribute channels.

    Parameters
    ----------
    arrays : dict
        A dictionary where each key is a string identifier (usually containing a timestamp) 
        and each value is a multidimensional NumPy array. The arrays are expected to have 
        at least 15 channels, with specific channels used for spectral data, masks, and 
        attributes.

    Returns
    -------
    dict
        The updated dictionary with processed arrays. Each array will have the new segmentation 
        masks and updated channels. The final arrays will contain 13 channels each.

    Notes
    -----
    - The input arrays must have at least 15 channels.
    - The function expects specific channels: 
      - Channel 10 is the deforestation mask.
      - Channels 11, 12, and 13 are attributes that include a mask status, an old timestamp, 
        and additional data, respectively.
      - Channel 14 is used for vegetation cover types.
    - The processed arrays will have channels reduced to 13:
      - 10 spectral channels.
      - 1 deforestation mask.
      - 1 vegetation (more precise deforestation) mask.
      - 1 seasonal mask.
    - The function does not perform validation on the structure of input arrays beyond checking 
      for the required channels.
    """
     
    for key, array in arrays.items():
    # Extract timestamp from key
        time_stamp_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', key)
        if time_stamp_match:
            year = time_stamp_match.group(1)
            month = time_stamp_match.group(2)
            day = time_stamp_match.group(3)
            ts = float(year[2:] + month + day)  # Convert to float timestamp
        else:
            continue  # Skip if no timestamp found

    # Define deforested condition
    deforested_condition = (array[..., 10] == 1) & (array[..., 11] > 0) & (array[..., 12] <= ts)

    # Step 1: Set all 1s in the 11th channel to 2
    array[array[..., 10] == 1, 10] = 2

    # Step 2: Revert elements that meet the condition back to 1
    array[deforested_condition, 10] = 1

    # Make a new channel where deforested class pixels are split into their different vegetation cover
    # Add a new channel, initialise to zero
    veg_mask = np.zeros(array.shape[:-1] +(1,))
    array = np.concatenate((array, veg_mask), axis=-1)

    condition_1 = (array[..., 10] == 2)
    array[condition_1, 15] = 5

    condition_2 = (array[..., 10] == 1) & (array[..., 14] == 1)
    array[condition_2, 15] = 1

    condition_3 = (array[..., 10] == 1) & (array[..., 14] == 2)
    array[condition_3, 15] = 2

    condition_4 = (array[..., 10] == 1) & (array[..., 14] == 3)
    array[condition_4, 15] = 3

    condition_5 = (array[..., 10] == 1) & (array[..., 14] == 4)
    array[condition_5, 15] = 4

    """At this point array has 16 channels: 10 spectral, 1 mask, 4 attributes and 1 veg mask"""
    # Create a new channel to take into account the weather season, making it 17 channels
    if '2023_05'  in key:
        season_mask = np.zeros(array.shape[:-1]+ (1,))
    elif '2023_09' in key:
        season_mask = np.ones(array.shape[:-1]+ (1,))
    elif '2024_02' in key:
        season_mask = np.full(array.shape[:-1]+ (1,), 2)     

    array = np.concatenate((array, season_mask), axis=-1)

    # Delete the attribute channels (id, timestamp, ...)
    # Should have just 13 channels left
    array = np.delete(array, [11,12,13,14], axis=-1)

    arrays[key] = array

    return arrays

def swap_channels(array):
    """
    Swap specified channels in a multidimensional array.

    This function reorders three specific channels in the given array:
    it swaps the positions of the segmentation mask channel, the extended 
    segmentation mask channel, and the season mask channel. The intended 
    reordering is from `[segmentation_mask, extended_segmentation_mask, season_mask]`
    to `[season_mask, segmentation_mask, extended_segmentation_mask]`.

    Parameters
    ----------
    array : numpy.ndarray
        A multidimensional NumPy array with at least 13 channels. The array is 
        expected to have the following relevant channels:
        - Channel 10: Segmentation mask with values in [0, 1, 2].
        - Channel 11: Extended segmentation mask with values in [0, 1, ..., 5].
        - Channel 12: Season mask with seasonal values.

    Returns
    -------
    numpy.ndarray
        The input array with the specified channels swapped. The reordering 
        will result in the array having the following channel order:
        - Channel 10: Season mask.
        - Channel 11: Segmentation mask.
        - Channel 12: Extended segmentation mask.

    Notes
    -----
    - The function assumes that the input array has at least 13 channels.
    - The channel indices 10, 11, and 12 refer to specific uses in the array 
      (segmentation masks and seasonal mask) and are directly swapped.
    """
    array[..., [11, 12, 10]] = array[..., [10, 11, 12]]
    return array

def unique_values_dict(arrays):
    """Get the unique values in a dictionary"""
    for key, value in arrays.items():
        print(key)
        print(np.unique(value[...,11], return_counts=True))

def to_patches(file_path: str, patch_size: int, stride: int) -> None:
    """
    Convert an array to patches of a given size and stride and save them to train or eval directory.

    Parameters:
    - file_path (str): Path to the input array file.
    - patch_size (int): Size of each patch.
    - stride (int): Stride between patches.
    - train_dir (str): Directory to save train patches.
    - eval_dir (str): Directory to save eval patches.
    """
    array = load_array(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Get file name without extension

    height, width, _ =  array.shape
    patch_height, patch_width = patch_size, patch_size
    stride_height, stride_width = stride, stride

    # Calculate the number of patches in each dimension
    num_patches_height = (height - patch_height) // stride_height + 1
    num_patches_width = (width - patch_width) // stride_width + 1

    # Determine the directory based on a random choice for train or eval
    base_dir =os.path.dirname(file_path)
    train_dir = f"{base_dir}/train/"
    eval_dir = f"{base_dir}/val/"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    for i in range(num_patches_height):
        for j in range(num_patches_width):
            start_i = i * stride_height
            start_j = j * stride_width
            patch = array[start_i:start_i + patch_height, start_j:start_j + patch_width, :]
            if 1 in patch[:,:,11]:
                dir = train_dir if random.random() < 0.8 else eval_dir
                patch_filename = f"{dir}{file_name}_{i* num_patches_width + j + 1}.pkl"
             
                with open(patch_filename, 'wb') as f:
                    pickle.dump(patch, f)

def count_files(folder_path):
    files = os.listdir(folder_path)
    file_count = len(files)
    return file_count

def augment_array(array):
    """
    Augments the input image with the same transformations applied to both
    feature channels and segmentation masks.
    """
    # Define the augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=1, border_mode=cv2.BORDER_CONSTANT, value=0)
    ])
    
    # Separate features and masks
    features = array[:, :, :11]
    masks = array[:, :, 11:]
    
    # Apply transformation
    augmented = transform(image=features, masks=[masks[:, :, 0], masks[:, :, 1]])
    
    # Combine features and masks again
    augmented_image = np.concatenate([augmented['image'], np.stack(augmented['masks'], axis=-1)], axis=-1)
    
    return augmented_image

def process_and_augment_pickles(input_dir, output_dir, quantity):
    """Can add ',num_augmentations' as input argument for more robustness"""
    """
    Process and augment pickle files from an input directory and save the results to an output directory.

    This function reads all `.pkl` files from the specified input directory, performs augmentation 
    on each loaded array, and saves the augmented arrays to the output directory. Each array is augmented 
    multiple times, producing a specified number of augmented versions per original file.

    Parameters
    ----------
    input_dir : str
        The directory path where the input `.pkl` files are located.
    output_dir : str
        The directory path where the augmented `.pkl` files will be saved.
        The directory will be created if it does not already exist.
    quantity: int
        The number of augmentations per array

    Example
    -------
    >>> input_dir = '/path/to/input'
    >>> output_dir = '/path/to/output'
    >>> process_and_augment_pickles(input_dir, output_dir)
    # This will read all `.pkl` files from `/path/to/input`, augment each file 5 times,
    # and save the augmented files to `/path/to/output`.

    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl"):
            input_path = os.path.join(input_dir, filename)
            
            # Load the pickle file
            with open(input_path, 'rb') as f:
                array = pickle.load(f)
            
            for i in range(quantity):
                augmented_array = augment_array(array)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_aug{i}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(augmented_array, f)
            

