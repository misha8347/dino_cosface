import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scipy.io as io
import cv2

class SnowLeopardDataset(Dataset):
    def __init__(self, path_to_data, transform = None):
        self.items = pd.read_csv(path_to_data).to_dict('records') 
        self.transform = transform
        self.class_dict = {
            '1__ID-SLM-1ALB': 0,
            '6__Аксайский': 1,
            '2__ID-SLM-2BA': 2,
            '3__ID-SLM-8KAS': 2,
            '5__SLF-2Shl-Гладиатор': 2, 
        }
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        image_path = self.items[index]['image_path']
        label_txt = self.items[index]['label']
        label_idx = self.class_dict[label_txt]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(label_idx)
        
        return image, label