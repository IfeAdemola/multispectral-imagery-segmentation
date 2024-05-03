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

class Forest(Dataset):
    """PyTorch dataset class for loading numpy arrays from pickle files"""

    def __init__(self, root_dir, label=False, transform=None):
        """
        Args:
            root_dir (string): Directory with pickle files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        # self.mean = mean
        # self.std = std
        self.label = label
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
        image = sample[:, :, :11]  # first 10 channels are image
        if self.label == False:
            mask = sample[:, :, 12]  # region type [0,1,2]
        elif self.label == True:
            mask = sample[:,:,11]  # more specific region type [0,1,...,5]

        #Normalise the image
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

        return image_nor, mask

class ToTensor:
    def __call__(self, sample):
        image, mask = sample
        return image.permute(2, 0, 1), mask