import torch
import torch.nn as nn
import click
from addict import Dict
import torchvision.transforms as T
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.models import DinoVisionTransformerClassifier
from src.trainers import train_fn
from src.datasets import YaleBDataset, SnowLeopardDataset
from src.utils import load_yaml

@click.command()
@click.argument('cfg_path', type=click.Path(), default='src/configs/config.yaml')
def main(cfg_path: str):
    cfg = Dict(load_yaml(cfg_path))

    model = DinoVisionTransformerClassifier(num_classes=cfg.num_classes, num_features=cfg.feature_dim,
                                            s=cfg.s, m=cfg.m)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
        A.Resize(224, 224),
        ToTensorV2()
    ])

    # df_train = YaleBDataset(cfg.path_to_data, split='train', transform=transform)
    # df_test = YaleBDataset(cfg.path_to_data, split='test', transform=transform)

    df_train = SnowLeopardDataset(cfg.path_to_data_train, transform=transform)
    df_val = SnowLeopardDataset(cfg.path_to_data_val, transform=transform)
    
    train_loader = DataLoader(df_train, batch_size=8, shuffle=True, pin_memory=True, num_workers=6)
    val_loader = DataLoader(df_val, batch_size=8, shuffle=False, pin_memory=True, num_workers=6)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5

    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    train_losses, val_losses, train_accuracies, val_accuracies = train_fn(model, device, train_loader, val_loader, 
                                        optimizer, criterion, num_epochs)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5

    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = True

    train_losses2, val_losses2, train_accuracies2, val_accuracies2 = train_fn(model, device, train_loader, val_loader,
                                        optimizer, criterion, num_epochs)


    print('saving model state dict...')
    torch.save(model.state_dict(), cfg.path_to_checkpoint)
    print('model state dict saved successfully!')

if __name__ == '__main__':
    main()