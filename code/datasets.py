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
    n_inputs = len(load_bands(use_rgb, use_red, use_nir, use_red_edge, use_swir, use_season))
    return n_inputs

class Forest(Dataset):
    """PyTorch dataset class for loading numpy arrays from pickle files"""

    def __init__(self, root_dir, label_structure=False, transform=None, 
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

        self.label_structure = label_structure

        self.selected_bands = load_bands(self.use_rgb, self.use_red, self.use_nir, self.use_red_edge, self.use_swir, self.use_season)

        self.n_inputs = get_ninputs(use_rgb, use_red, use_nir, use_red_edge, use_swir, use_season)
        if label_structure:
            self.num_classes = 6
            self.label_structure = True
        else:
            self.num_classes = 3
            self.label_structure = False

        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.pkl')]
        # self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

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


        if self.label_structure == False:
            mask = sample[:, :, 11]  # region type [0,1,2]
        elif self.label_structure == True:
            mask = sample[:,:,12]  # more specific region type [0,1,...,5]


        normalised_image = torch.from_numpy(sensor_image).float() # change variable name (it's not a normalisation)
        mask = torch.from_numpy(mask).long()

        # Apply transform if specified
        if self.transform:
            normalised_image, mask = self.transform((normalised_image, mask))

        # return {'image':image_nor, 'label':mask}
    
        return normalised_image, mask
    
    

class ToTensor:
    def __call__(self, sample):
        image, mask = sample
        return image.permute(2, 0, 1), mask