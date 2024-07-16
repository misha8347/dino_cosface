import torch
import torch.nn as nn  
from torch.utils.data import DataLoader
import albumentations as A
from addict import Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import cv2

from src.models import DinoVisionTransformerClassifier
from src.utils import load_yaml
from src.datasets import SnowLeopardDataset

def main(image_path):
    cfg = Dict(load_yaml('src/configs/config.yaml'))
    
    model = DinoVisionTransformerClassifier(num_classes=cfg.num_classes, num_features=cfg.feature_dim,
                                        s=cfg.s, m=cfg.m)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = A.Compose([
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
            A.Resize(224, 224),
            ToTensorV2()
        ])
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    augmented = transform(image=image)
    image = augmented['image']
    image_tensor = image.unsqueeze(0).to(device)
    
    model.load_state_dict(torch.load(cfg.path_to_checkpoint))
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        embeddings = model.transformer(image_tensor)
        embeddings = embeddings.cpu().tolist()
    
    X = np.array(embeddings)
    knn_loaded = joblib.load('checkpoints/knn_exp2.joblib')
    
    y_pred = knn_loaded.predict(X)
    print('predicted class:', y_pred)
    

if __name__ == "__main__":
    image_path = '/home/jupyter/datasphere/s3/iofzkzcameratraps/BARSY_OSOBI/barsy_osoby/2__ID-SLM-2BA/day/segmented/IMG_0111_207.png'
    main(image_path)