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

from skimage.transform import resize
from PIL import Image

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
        self.transform_v2 = transforms.Compose([transforms.ToTensor()]) 
        
        self.data_frame['name'] = self.data_frame['name'] == 'cat'
        self.data_frame['name'] = self.data_frame['name'] == 'dog'
        

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(self.data_frame['filename'][idx])

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        img = Image.open(img_name).convert('RGB')
        img = img.resize((416,416), Image.ANTIALIAS)
        
        
        #image_resized = resize(img, (416,416),
         #              anti_aliasing=True)
        
        # Bounding Box Tensor
        properties = self.data_frame.iloc[idx, [4,7, 8, 9, 10, 11]]
        #properties = properties.astype(np.float64)
        #img = torch.from_numpy(img.transpose(2, 0, 1))
        #properties = self.data_frame.iloc[idx, 1:]
        properties = np.array(properties , dtype=np.float32)
        #print(properties)
        #image_resized = np.array(image_resized , dtype=np.float32)
        #properties = properties.astype('float').reshape(-1, 2)
        #sample = {'image': image_resized, 'properties': properties}
        
        if self.transform:
            sample = self.transform(sample)

        return self.transform_v2(img), torch.Tensor(properties)
        #return self.transform_v2(image_resized) , self.transform_v2(properties)
    

"""
    Test code
"""    
#dataset = IIITDataset(csv_file="annotations.csv", root_dir="data/images/")
#print( dataset[5] )
