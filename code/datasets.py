import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from typing import List, Dict, Any

S2_BANDS_LR = [0,1,2,6]
# S2_BANDS_LR = [6]
S2_BANDS_MR = [3,4,5,7,8,9]
SEASON_BAND = [10]
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

def load_bands(use_rgb, use_lr, use_mr, use_season):
    bands_selected = []
    if use_rgb:
        bands_selected = bands_selected + S2_BANDS_LR[:-1]
    if use_lr:
        bands_selected = list(set(bands_selected + S2_BANDS_LR))
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_season:
        bands_selected = bands_selected + SEASON_BAND
    return bands_selected

def get_ninputs(use_rgb, use_lr, use_mr, use_season):
    n_inputs = 0
    if use_rgb:
        n_inputs += len(S2_BANDS_LR[:-1])
    if use_lr and not use_rgb:
        n_inputs += len(S2_BANDS_LR)
    if use_lr and use_rgb:
        n_inputs += 1
    if use_mr:
        n_inputs += len(S2_BANDS_MR)
    if use_season:
        n_inputs += len(SEASON_BAND)
    return n_inputs


class Forest(Dataset):
    """PyTorch dataset class for loading numpy arrays from pickle files"""

    def __init__(self, root_dir, label=False, transform=None, 
                 use_rgb=False, use_lr=False, use_mr=False, use_season=False):
        """
        Args:
            root_dir (string): Directory with pickle files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # initialise
        super(Forest, self).__init__

        # make sure parameters are okay
        if not (use_rgb or use_lr or use_mr or use_season):
            raise ValueError("No input specified, set at least one of "
                             + "use_[rgb, lr, mr, season] to True!")

        # make sure parent dir exists
        assert os.path.exists(root_dir)
        self.root_dir = root_dir

        self.use_rgb = use_rgb
        self.use_lr = use_lr
        self.use_mr = use_mr
        self.use_season = use_season

        self.label = label

        self.selected_bands = load_bands(self.use_rgb, self.use_lr, self.use_mr, self.use_season)

        self.n_inputs = get_ninputs(use_rgb, use_lr, use_mr, use_season)
        if label:
            self.num_classes = 6
            self.label = True
        else:
            self.num_classes = 3
            self.label = False

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

        # Extract image and mask from the numpy array
        image = sample[:,:, load_bands(self.use_rgb, self.use_lr, self.use_mr, self.use_season)]

        if self.label == False:
            mask = sample[:, :, 11]  # region type [0,1,2]
        elif self.label == True:
            mask = sample[:,:,12]  # more specific region type [0,1,...,5]

        #Normalise the image
        """this would not work for season band- rectify"""
        min_val = np.min(image)
        max_val = np.max(image)
        image_nor = (image - min_val) / (max_val - min_val)  

        # Convert numpy arrays to PyTorch tensors
        image_nor = torch.from_numpy(image_nor).float()
        mask = torch.from_numpy(mask).long()

        # Normalize the image
        # image = self.normalize(image)

        # Apply transform if specified
        if self.transform:
            image_nor, mask = self.transform((image_nor, mask))

        # return {'image':image_nor, 'label':mask}
    
        return image_nor, mask
    
    

class ToTensor:
    def __call__(self, sample):
        image, mask = sample
        return image.permute(2, 0, 1), mask