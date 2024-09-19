import os
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
import shutil
from utils import *


def ensure_directory_exists(directory):
    """Ensure that the given directory exists."""
    os.makedirs(directory, exist_ok=True)

def get_filename(file_path):
    """Returns filename from the filepath"""
    return  os.path.splitext(os.path.basename(file_path))[0]

def clean_data(input_file_paths, output_dir, cutoff):
    """
    Cleans image arrays by removing borders and concatenating them 
    with the deforested 'areas_without_trees' annotation.
    
    Args:
        files (list): List of file paths to process.
        output_dir (str): Directory to save the processed images.
        cutoff (int): The number of pixels to cut from the borders of the images.

    Returns:
        None
    """
    ensure_directory_exists(output_dir)

    # Find and load the  deforestation map - 'areas_without_trees' array
    deforestation_mask = next((load_array(f) for f in input_file_paths if "areas_without_trees" in f), None)
    if deforestation_mask is None:
        raise FileNotFoundError('Deforestation mask file not found in the provided file list.')

    # Process and save each image file that doesn't contain 'without_trees'
    for file_path in input_file_paths:
        if "without_trees" in file_path:
            continue  # Skip files with 'without_trees'
                
        # Load and concatenate the image array with deforestation map array
        image_data = load_array(file_path)
        print(f"Original shape of {os.path.basename(file_path)}: {image_data.shape}")
        
        concatenated_array = np.concatenate((image_data, deforestation_mask), axis=-1)
        
        # Cut borders based on the cutoff
        cut_data = concatenated_array[cutoff:-cutoff, cutoff:-cutoff, :]
        print(f"Shape after cutoff: {cut_data.shape}")
        
        band_keys = ['blue', 'green', 'red', 'red_edge1', 'red_edge2', 'red_edge3', 'nir',
                 'red_edge4', 'swir1', 'swir2', 'validity', 'id', 'timestamp', 'state', 'vegetation']
        cut_data_dict = {key: cut_data[:, :, idx] for idx, key in enumerate(band_keys)}
        
        # Save the processed dictionary to the output directory
        file_name = f"{get_filename_from_path(file_path)}.pkl" 
        save_file(cut_data_dict, file_name, output_dir)
    # return


def save_file(data, file_name, output_dir):
    """Save processed array to a file."""
    ensure_directory_exists(output_dir)
    save_path = os.path.join(output_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Processed file saved to: {save_path}")
        

def extract_timestamp(filename):
    """
    Extracts a float timestamp from the filename in the format YYYY_MM_DD.
    
    Args:
        filename (str): The filename string containing the date in the format YYYY_MM_DD.
    Returns:
        float: A float representation of the timestamp (YYMMDD).
        None: If no valid timestamp is found.
    """
    time_stamp_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
    if time_stamp_match:
        year = time_stamp_match.group(1)
        month = time_stamp_match.group(2)
        day = time_stamp_match.group(3)
        return float(year[2:] + month + day)
    return None  # Explicitly return None if no match
    
def create_simple_mask(file_name, data):
    """
    Create a simple mask that identifies deforested and forested areas.
    
    Args:
        file_name (str): The file name containing the date information.
        data (dict): Dictionary consisting of the spectral bands (season) and defrestation map channels of an image
    
    Returns:
        dict: Updated data dictionary with the simple segmentation mask added.
    """
    timestamp = extract_timestamp(file_name)
    if timestamp is None:
        raise ValueError(f"Filename '{file_name}' does not contain a valid timestamp.")

    # Initialize a simple_mask channel with zeros
    simple_mask = np.zeros_like(data['id'])

    # Define deforested condition
    deforested_condition = ((data['id'] > 0) & 
                            (data['validity'] == 1) & 
                            (data['timestamp'] <= timestamp))
    
    simple_mask[data['validity'] == 1] = 2  # Forest
    simple_mask[deforested_condition] = 1  # Deforested
    
    data['simple_mask'] = simple_mask
    print(len(data))
    print(data.keys())
    return data

def create_complex_mask(filename, array):
    """
    Create a complex mask based on vegetation types and simple mask values.
    
    Args:
        array (dict): Dictionary representing different channels of the array.
    
    Returns:
        dict: Updated array dictionary with the 'complex_mask' channel added.
    """
    complex_mask = np.zeros_like(array['id'])

    # Define conditions for different vegetation types and simple mask values
    forest = (array['simple_mask']== 2)
    complex_mask[forest] = 5

    soil = (array['simple_mask']== 1) & (array['type'] == 1)
    complex_mask[soil] = 1

    low_grass = (array['simple_mask']== 1) & (array['type'] == 2)
    complex_mask[low_grass] = 2

    high_grass = (array['simple_mask']== 1) & (array['type'] == 3)
    complex_mask[high_grass] = 3

    sparse_trees = (array['simple_mask']== 1) & (array['type'] == 4)
    complex_mask[sparse_trees] = 4

    array['complex_mask'] = complex_mask
    print(array.shape)
    return array

def create_season_mask(file_name, data):
    """
    Create a seasonal mask based on specific dates in the filename.
    
    Args:
        file_name (str): The filename containing date information.
        data (dict): Dictionary representing different channels of the data (spectral bands, deforestation maps, masks).
    
    Returns:
        dict: Updated array dictionary with the season mask channel added.
    """
    # Initialize 'season_mask' based on filename timestamps
    if '2023_05'  in file_name:
        season_mask = np.full(data['id'].shape, -1)
    elif '2023_09' in file_name:
        season_mask = np.zeros(data['id'].shape)
    elif '2024_02' in file_name:
        season_mask = np.ones(data['id'].shape)

    data['season_mask'] = season_mask
    print(len(data))
    return data

def normalise_arrays(data):
    """
    Normalize only the spectral bands by applying zero-centering normalization
    
    Args:
        data (dict): Dictionary of array channels.
    
    Returns:
        dict: Updated array dictionary with normalized channels.
    """
    # Channels to be normalized
    SPECTRAL_BANDS = ['blue', 'green', 'red', 'red_edge1', 'red_edge2', 
                         'red_edge3', 'nir', 'red_edge4', 'swir1', 'swir2']
    
    for band in SPECTRAL_BANDS:
        if band in data:
            data[band] = data[band] * 2 - 1
    return data


def create_patches(data, file_path, patch_size, stride, save_directory):
    """
    Create patches from the input data and save them in the specified directory.
    
    Args:
        data (dict): Dictionary of channels, where each value is a 2D NumPy array (h x w).
        file_path (str): File path of file to be processed
        patch_size (int): Size of each patch (assumed square patches).
        stride (int): Stride to move the patch window.
        save_directory (str): Directory where patches should be saved.
    """    

    height, width = next(iter(data.values())).shape[:2]  

    num_patches_height = (height - patch_size) // stride + 1
    num_patches_width = (width - patch_size) // stride + 1

    spectral_bands = ['blue', 'green', 'red', 'red_edge1', 'red_edge2', 
                         'red_edge3', 'nir', 'red_edge4', 'swir1', 'swir2']
    masks = ['season_mask', 'simple_mask']
    selected_bands = spectral_bands + masks

    for i in range(num_patches_height):
        for j in range(num_patches_width):
            start_i = i * stride
            start_j = j * stride

            patch = extract_patch(data, start_i, start_j, patch_size)
            patch_id = i * num_patches_width + j + 1
            filename = get_filename(file_path)
            patch_filename = f"{filename}_{patch_id}.pkl"
            
            if  1 in patch['simple_mask'] and np.all([np.all(patch[k] <= 1) for k in spectral_bands if k in patch]):
                patch = np.array([patch[selected_band] for selected_band in selected_bands])
                patch = np.transpose(patch, (1, 2, 0))
                deforestation_dir = os.path.join(save_directory, "deforestation")
                save_file(patch, patch_filename, deforestation_dir)

            elif  np.all(patch['simple_mask'] == 2) and np.all([np.all(patch[k] <= 1) for k in spectral_bands if k in patch]):
                patch = np.array([patch[selected_band] for selected_band in selected_bands])
                patch = np.transpose(patch, (1, 2, 0))
                forest_dir = os.path.join(save_directory, "forest")
                save_file(patch, patch_filename, forest_dir)

def extract_patch(array, start_i, start_j, patch_size):
    """
    Extract a patch of the given size from the array at the specified start position.

    Args:
        array (dict): Dictionary of channels (NumPy arrays).
        start_i (int): Starting row index for the patch.
        start_j (int): Starting column index for the patch.
        patch_size (int): Size of the patch (height and width).

    Returns:
        dict: A dictionary containing patches for each channel.
    """
    patch = {}
    for key, channel in array.items():
        patch[key] = channel[start_i:start_i + patch_size, start_j:start_j + patch_size]
    return patch

def train_test_split(root_dir, destination_dir, train_ratio=0.8):
    """
    Split the data into train and test sets based on the specified ratio.

    Args:
        root_dir (str): Root directory containing the data.
        destination_dir (str): Destination directory to save the split data.
        train_ratio (float): Ratio of data to be used for training (default: 0.8).

    Returns:
        None
    """
    train_dir = os.path.join(destination_dir, 'train')
    test_dir = os.path.join(destination_dir, 'test')
    ensure_directory_exists(train_dir)
    ensure_directory_exists(test_dir)

    files = [f for f in os.listdir(root_dir) if f.endswith('.pkl')]

    # Group files by unique filenames (without index)
    file_groups = {}
    for file in files:
        base_name, index = file.rsplit('_', 1)
        index = int(index.replace('.pkl', ''))  # Extract numeric index from the filename
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append((file, index))

    # Sort files by index within each group
    for base_name, file_list in file_groups.items():
        file_list.sort(key=lambda x: x[1])  # Sort by the extracted index

        # Split files into train and test sets based on sorted order
        split_index = int(len(file_list) * train_ratio)  # Calculate the split index

        train_files = file_list[:split_index]
        test_files = file_list[split_index:]

        # Move files to the corresponding directories
        for train_file, _ in train_files:
            shutil.copy(os.path.join(root_dir, train_file), os.path.join(train_dir, train_file))
        for test_file, _ in test_files:
            shutil.copy(os.path.join(root_dir, test_file), os.path.join(test_dir, test_file))

        print(f"Processed {base_name}: {len(train_files)} train files, {len(test_files)} test files.")





def main():
    raw_data_dir = ["/home/k45848/multispectral-imagery-segmentation/data/raw/west",
                    "/home/k45848/multispectral-imagery-segmentation/data/raw/east"]
    cut_data_dir ="/home/k45848/multispectral-imagery-segmentation/data/interim/cut"
    simple_mask_dir = "/home/k45848/multispectral-imagery-segmentation/data/interim/simple_mask"
    patches_dir = "/home/k45848/multispectral-imagery-segmentation/data/processed/patches"
    train_test_dir = "/home/k45848/multispectral-imagery-segmentation/data/processed"
    cutoffs = [18, 25]
    # files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pkl")]
    deforestation_dir = '/home/k45848/multispectral-imagery-segmentation/data/patches/deforestation'

    # for raw_dir, cutoff in zip(raw_data_dir, cutoffs):
    #     file_paths = get_file_paths(raw_dir)
    #     clean_data(file_paths, cut_data_dir, cutoff)
    #     cut_data_paths = get_file_paths(cut_data_dir)
    #     for file_path in cut_data_paths:
    #         data = load_array(file_path)
    #         file_name = f"{get_filename(file_path)}.pkl"
    #         with_season_mask = create_season_mask(file_name, data)
    #         with_simple_mask = create_simple_mask(file_name, with_season_mask)
    #         create_patches(with_simple_mask, file_name, 32, 8, patches_dir)
    #         # with_complex_mask = create_complex_mask(file_name, with_simple_mask)
    #         # save_file(with_simple_mask, file_path, simple_mask_dir)

    train_test_split(deforestation_dir, train_test_dir, train_ratio=0.8)


if __name__ == "__main__":

    main()
