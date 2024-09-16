import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import torch
from utils import *
from typing import List, Dict, Any
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from typing import List, Dict, Any



# Define the dictionary keys instead of indices for the bands
RGB = ['blue', 'green', 'red']
NIR = ['nir']
RED_EDGE = ['red_edge1', 'red_edge2', 'red_edge3', 'red_edge4']
SWIR = ['swir1', 'swir2']
SEASON = ['season_mask']

def compute_num_classes(root_dir, label):
    """
    Compute the number of unique classes in the dataset masks.
    
    Args:
        root_dir (str): Path to the directory containing the dataset files.
        label (bool): Whether to use complex mask (True) or simple mask (False).
    
    Returns:
        int: Number of unique classes in the dataset masks.
    """
    all_masks = []
    file_list = [f for f in os.listdir(root_dir) if f.endswith('.pkl')]
    
    for file_name in file_list:
        with open(os.path.join(root_dir, file_name), 'rb') as f:
            sample = pickle.load(f)  # Assuming the loaded sample is a dictionary of arrays

            # Extract the correct mask depending on label
            if label == False:
                mask = sample['simple_mask']  
            elif label == True:
                mask = sample['complex_mask'] 
            # Flatten the mask and extend the list
            all_masks.extend(mask.flatten().tolist())

    # Get the number of unique classes
    num_classes = len(set(all_masks))
    return num_classes


def load_bands(use_rgb, use_red, use_nir, use_red_edge, use_swir, use_season):
    """
    Load the selected bands based on the user's input preferences using a dictionary mapping.
    
    Args:
        use_rgb (bool): Whether to use RGB bands.
        use_red (bool): Whether to use the red band.
        use_nir (bool): Whether to use NIR band.
        use_red_edge (bool): Whether to use red-edge bands.
        use_swir (bool): Whether to use SWIR bands.
        use_season (bool): Whether to use season information.
    
    Returns:
        list: List of selected bands based on user preferences.
    """
    # Define a dictionary for band groups
    band_groups = {
        'rgb': RGB,
        'red': [RGB[2]],  # Only the 'red' band from RGB
        'nir': NIR,
        'red_edge': RED_EDGE,
        'swir': SWIR,
        'season': SEASON
    }
    
    # Map user preferences to band groups
    user_selection = {
        use_rgb: 'rgb',
        use_red: 'red',
        use_nir: 'nir',
        use_red_edge: 'red_edge',
        use_swir: 'swir',
        use_season: 'season'
    }
    
    # Select bands based on user preferences
    bands_selected = []
    for use_band, band_group in user_selection.items():
        if use_band:
            bands_selected += band_groups[band_group]
    
    # If none selected, default to RGB
    if not bands_selected:
        bands_selected = RGB
    
    # Return unique bands
    return list(set(bands_selected))


def get_ninputs(use_rgb, use_red, use_nir, use_red_edge, use_swir, use_season):
    """
    Get the number of input channels based on user selections.
    
    Args:
        use_rgb (bool): Whether to use RGB bands.
        use_red (bool): Whether to use the red band.
        use_nir (bool): Whether to use NIR band.
        use_red_edge (bool): Whether to use red-edge bands.
        use_swir (bool): Whether to use SWIR bands.
        use_season (bool): Whether to use season information.
    
    Returns:
        int: Number of selected input channels.
    """
    return len(load_bands(use_rgb, use_red, use_nir, use_red_edge, use_swir, use_season))


class Forest(Dataset):
    """PyTorch dataset class for loading numpy arrays from pickle files"""

    def __init__(self, root_dir, use_multiclass=False, transform=None, 
                 use_rgb=False, use_red=False, use_nir=False, use_red_edge=False, use_swir=False, use_season=False):
        """
        Args:
            root_dir (string): Directory with pickle files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # initialise
        super(Forest, self).__init__

        # make sure parameters are okay
        if not (use_rgb or use_red or use_nir or use_red_edge or use_swir or use_season):
            print("No input specified; defaulting to RGB usage.")
            # raise ValueError("No input specified, set at least one of "
            #                  + "use_[rgb, lr, mr, season] to True!")

        # make sure parent dir exists
        assert os.path.exists(root_dir)
        self.root_dir = root_dir

        self.use_rgb = use_rgb
        self.use_red = use_red
        self.use_nir = use_nir
        self.use_red_edge = use_red_edge
        self.use_swir = use_swir
        self.use_season = use_season

        self.use_multiclass = use_multiclass

        self.selected_bands = load_bands(self.use_rgb, self.use_red, self.use_nir, self.use_red_edge, self.use_swir, self.use_season)

        self.n_inputs = get_ninputs(use_rgb, use_red, use_nir, use_red_edge, use_swir, use_season)
        if use_multiclass:
            self.num_classes = 6
            self.use_multiclass = True
        else:
            self.num_classes = 3
            self.use_multiclass = False

        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load pickle file
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        with open(file_name, 'rb') as f:
            sample = pickle.load(f)

        # Extract bands from the numpy array 
        sensor_bands = load_bands(self.use_rgb, self.use_red, self.use_nir, self.use_red_edge, self.use_swir, self.use_season)

        sensor_image = sample[:, :, sensor_bands]


        if self.use_multiclass == False:
            mask = sample[:, :, 11]  # region type [0,1,2]
        elif self.use_multiclass == True:
            mask = sample[:,:,12]  # more specific region type [0,1,...,5]


        input_image = torch.from_numpy(sensor_image).float() # change variable name (it's not a normalisation)
        mask = torch.from_numpy(mask).long()

        # Apply transform if specified
        if self.transform:
            input_image, mask = self.transform((input_image, mask))

        # return {'image':image_nor, 'label':mask}
    
        return input_image, mask
    
    

class ToTensor:
    def __call__(self, sample):
        image, mask = sample
        return image.permute(2, 0, 1), mask