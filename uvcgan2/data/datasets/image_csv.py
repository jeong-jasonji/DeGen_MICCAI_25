import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from uvcgan2.consts import SPLIT_TRAIN
from .image_domain_folder import ImageDomainFolder


# add CSV dataset class - specifically for abdominal CT
class CSVDataset(Dataset):
    """Custom pytorch dataset that takes CSV files"""
    def __init__(
        self, path, domain,
        split     = SPLIT_TRAIN,
        transform = None,
        **kwargs
    ):
        """
        Arguments:
            path (string): Path to the csv file with image paths, annotations, and IDs (optional).
            split (string): which split to use
            domain (string): domain to translate to and from
            transform (callable, optional): Optional transform to be applied on a sample.
        Notes:
            csv files should be in the form: {path}/{split}_{domain}.csv
            e.g.:
                path/train.csv
                path/val.csv
            each dataframe has two paths: 'vue_path' and 'iodine_path' and the domain will make
            the dataset load the specific dose.
        """
        super().__init__(**kwargs)
        
        # get the correct csv file
        if 'headct_any' in path:
            df = pd.read_csv(path + '_{}_{}_center.csv'.format(split, domain.upper()))
            df = df[df['0.35_exclude']]
            self._path = df.reset_index(drop=True)
        else:
            self._path = pd.read_csv(path + '_{}_{}.csv'.format(split, domain.upper()))
        self._imgs      = self._path.img_path.to_list()
        self._labels    = self._path.img_label.to_list() # removed for testing
        self._transform = transform

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.image_loader(self._imgs[idx])
        label = self._labels[idx]
        
        if self._transform is not None:
            image = self._transform(image)

        return (image, label)
        
    def image_loader(self, img_path):
        if img_path.endswith(('.png', '.jpg')):
            img = Image.open(img_path)
            if img.mode == 'I': 
                # if image is a 16bit grayscale convert to [0.0, 1.0] array
                img_arr = np.array(img)
                img_arr = img_arr / 65535.0
                img = Image.fromarray(img_arr)
        elif img_path.endswith(('.npy')):
            # keep it the same with sigmoid output - range of [0,1]
            arr = self.window_ct(np.load(img_path))
            img = Image.fromarray(arr)
        return img
    
    def window_ct(self, arr, hu_min=-500, hu_max=1500):
        #convert to float 32
        pixel = np.float32(arr)
        # add the HU intercept
        pixel = pixel*1 + (-1024)
        # clip to the correct HU values
        arr = np.clip(arr, hu_min, hu_max) 
        # normalize
        arr += np.abs(hu_min)
        arr = arr / (np.abs(hu_max) + np.abs(hu_min))
        return arr