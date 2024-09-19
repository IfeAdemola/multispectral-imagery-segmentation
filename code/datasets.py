import numpy as np
import pickle
import os
import torch
from utils import *
from augmentation import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from typing import List, Dict, Any


RGB = [0, 1, 2]
NIR = [6]
RED_EDGE = [3, 4, 5, 7]
SWIR = [7, 8, 9]
SEASON = [10]

def compute_num_classes(root_dir, label):
        """Compute the number of unique classes in the dataset masks"""
        all_masks = []
        file_list = [f for f in os.listdir(root_dir) if f.endswith('.pkl')]
        for file_name in file_list:
            with open(os.path.join(root_dir, file_name), 'rb') as f:
                sample = pickle.load(f)
                if label == False:
                    mask = sample[:, :, 12]
                elif label == True:
                    mask = sample[:, :, 11]
                all_masks.extend(mask.flatten().tolist())
        num_classes = len(set(all_masks))
        return num_classes

def load_bands(use_rgb, use_red, use_nir, use_red_edge, use_swir, use_season):
    bands_selected = []
    
    if use_rgb:
        bands_selected += RGB        
    if use_red:
        bands_selected += [RGB[0]]
    if use_nir:
        bands_selected += NIR        
    if use_red_edge:
        bands_selected += RED_EDGE
    if use_swir:
        bands_selected += SWIR

    # If none are selected, default to RGB only
    if not (use_rgb or use_nir or use_red_edge or use_swir or use_season):
        bands_selected = RGB

    # Always include SEASON if selected
    if use_season:
        bands_selected += SEASON
        
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
                 use_rgb=False, use_red=False, use_nir=False, use_red_edge=False, 
                 use_swir=False, use_season=False, apply_augmentations=False):
        """
        Args:
            root_dir (string): Directory with pickle files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # initialise
        super(Forest, self).__init__()

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

        self.num_classes = 6 if use_multiclass else 3

        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.pkl')]

        self.apply_augmentations = apply_augmentations
        self.num_augmentations = 8 if apply_augmentations else 1

       
    def __len__(self):
        return len(self.file_list) * self.num_augmentations
    
    def load_sample(self, idx):
        """Loads a sample (sensor image and mask) from the pickle file."""
        file_name = os.path.join(self.root_dir, self.file_list[idx])
        with open(file_name, 'rb') as f:
            sample = pickle.load(f)

        sensor_image = sample[:, :, self.selected_bands]
        mask = sample[:, :, 12] if self.use_multiclass else sample[:, :, 11]

        return sensor_image, mask

    def get_augmentation(self, image, mask, aug_idx):
        """Applies augmentation based on aug_idx."""
        rotations = [
            (image, mask),               # Original image and mask
            (rotate_90(image), rotate_90(mask)),    # 90 degrees
            (rotate_180(image), rotate_180(mask)),   # 180 degrees
            (rotate_270(image), rotate_270(mask))    # 270 degrees
        ]

        # Get base image (original or rotated version)
        base_image, base_mask = rotations[aug_idx % 4]  # The first 4 indices are the rotations

        # Apply Gaussian noise if aug_idx >= 4
        if aug_idx >= 4:
            base_image = add_gaussian_noise(base_image)

        return base_image, base_mask

    def __getitem__(self, idx):
        # Determine original image index and augmentation index
        image_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations

        # Load the original sensor image and mask
        sensor_image, mask = self.load_sample(image_idx)

        # Apply augmentation if necessary
        if self.apply_augmentations:
            sensor_image, mask = self.get_augmentation(sensor_image, mask, aug_idx)

        # Convert to PyTorch tensors
        sensor_image = torch.from_numpy(sensor_image).float()  # Shape: (H, W, Channels)
        mask = torch.from_numpy(mask).long()  # Mask is a 2D tensor

        # Apply any additional transformations (e.g., random crops)
        if self.transform:
            sensor_image, mask = self.transform((sensor_image, mask)) #maybe if I feel like normalising to range -1 and 1
   
        return sensor_image, mask
    

class ToTensor:
    def __call__(self, sample):
        image, mask = sample
        return image.permute(2, 0, 1), mask