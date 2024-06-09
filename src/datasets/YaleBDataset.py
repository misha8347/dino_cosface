import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scipy.io as io


class YaleBDataset(Dataset):
    def __init__(self,path_to_data, split='train', transform = None):
        self.YALE = io.loadmat(path_to_data)
        Y = self.YALE['Y'].squeeze()
        if split == 'train':
            indices = np.where(Y < 30)
        else:
            indices = np.where(Y >= 30)
        
        self.X = self.YALE['X'].T[indices]
        self.Y = Y[indices]
        self.transform = transform
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, index):
        image = self.X[index].reshape(-1, 192).T #np array
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        label = self.Y[index] #int
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(label)
        
        return image, label