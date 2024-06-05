import torch
import torch.nn as nn
import click
from addict import Dict
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.models import DinoVisionTransformerClassifier
from src.trainers import train_fn

def main():
    model = DinoVisionTransformerClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor()
    ])
    df_train = torchvision.datasets.LFWPeople(root='/home/jupyter/datasphere/project/LFW_people_train', 
                                              split='train', transform=transform, download=True)
    df_test = torchvision.datasets.LFWPeople(root='/home/jupyter/datasphere/project/LFW_people_test', 
                                              split='test', transform=transform, download=True)

    train_loader = DataLoader(df_train, batch_size=8, shuffle=True, pin_memory=True, num_workers=6)
    test_loader = DataLoader(df_test, batch_size=8, shuffle=False, pin_memory=True, num_workers=6)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10

    losses, train_accuracies, test_accuracies = train_fn(model, device, train_loader, 
                                                         test_loader, optimizer, criterion, num_epochs)

    print('saving model state dict...')
    torch.save(model.state_dict(), '../checkpoints/dinov2_cosface.pth')
    print('model state dict saved successfully!')

if __name__ == '__main__':
    main()