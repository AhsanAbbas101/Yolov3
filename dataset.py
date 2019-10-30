# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:59:55 2019

@author: Ahsan
"""
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class IIITDataset(Dataset):
    """IIIT dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        
        # Bounding Box Tensor
        labels = self.data_frame.iloc[idx, 4]
        bbx_attr = self.data_frame.iloc[idx, 8:12]
        properties = pd.concat([labels,bbx_attr])

        #properties = self.data_frame.iloc[idx, 1:]
        properties = np.array([properties])
        #properties = properties.astype('float').reshape(-1, 2)
        sample = {'image': image, 'properties': properties}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

"""
    Test code
"""    
dataset = IIITDataset(csv_file="annotations.csv", root_dir="data/images/")
print( dataset[5] )
