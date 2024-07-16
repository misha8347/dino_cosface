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

from src.models import DinoVisionTransformerClassifier
from src.utils import load_yaml
from src.datasets import SnowLeopardDataset

def generate_embeddings(model, device, loader):
    embeds_arr = []
    labels_arr = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.detach().cpu().tolist()

            embeds = model.transformer(images)

            embeds_arr.extend(embeds.cpu().tolist())
            labels_arr.extend(labels)
    return embeds_arr, labels_arr


def train_knn(embeds_arr_train, labels_arr_train, embeds_arr_val, labels_arr_val):
    print('initialization of knn training')
    X_train = np.array(embeds_arr_train)
    y_train = np.array(labels_arr_train)
    
    X_val = np.array(embeds_arr_val)
    y_val = np.array(labels_arr_val)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Accuracy KNN: {accuracy}')

    print('saving KNN model...')
    joblib.dump(knn, 'checkpoints/knn_exp2.joblib')
    
def main():
    cfg = Dict(load_yaml('src/configs/config.yaml'))
    
    model = DinoVisionTransformerClassifier(num_classes=cfg.num_classes, num_features=cfg.feature_dim,
                                        s=cfg.s, m=cfg.m)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
            A.Resize(224, 224),
            ToTensorV2()
        ])
    
    val_transform = A.Compose([
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
            A.Resize(224, 224),
            ToTensorV2()
        ])
        
    df_train = SnowLeopardDataset(cfg.path_to_data_train, transform=transform)
    df_val = SnowLeopardDataset(cfg.path_to_data_val, transform=val_transform)
    
    train_loader = DataLoader(df_train, batch_size=8, shuffle=True, pin_memory=True, num_workers=6)
    val_loader = DataLoader(df_val, batch_size=8, shuffle=False, pin_memory=True, num_workers=6)
    
    model.load_state_dict(torch.load(cfg.path_to_checkpoint))
    model.to(device)
    
    embeds_arr_train, labels_arr_train = generate_embeddings(model, device, train_loader)
    embeds_arr_val, labels_arr_val = generate_embeddings(model, device, val_loader)
    
    train_knn(embeds_arr_train, labels_arr_train, embeds_arr_val, labels_arr_val)
    
if __name__ == "__main__":
    main()