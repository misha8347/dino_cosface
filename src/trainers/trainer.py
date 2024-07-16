import torch
from tqdm import tqdm


def train_fn(model, device, train_loader, val_loader, optimizer, criterion, num_epochs):
    model.to(device)
    losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch_i in range(num_epochs):
        print(f"epoch {epoch_i}")
        epoch_loss = 0.0
        epoch_train_accuracy = 0.0
        
        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0

        model.train()
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            pred_labels = logits.argmax(dim=1)
            epoch_train_accuracy += (torch.sum(pred_labels == labels).item() / labels.shape[0])

        epoch_loss /= len(train_loader)
        epoch_train_accuracy /= len(train_loader)
        print(f'train epoch loss: {epoch_loss}')
        print(f'train epoch accuracy: {epoch_train_accuracy}')
        
        losses.append(epoch_loss)
        train_accuracies.append(epoch_train_accuracy)
        
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                epoch_val_loss += loss.item()
                
                pred_labels = logits.argmax(dim=1)
                epoch_val_accuracy += (torch.sum(pred_labels == labels).item() / labels.shape[0])
        
        epoch_val_loss /= len(val_loader)
        epoch_val_accuracy /= len(val_loader)
        print(f'val epoch loss: {epoch_val_loss}')
        print(f'val epoch accuracy: {epoch_val_accuracy}')
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

    
    return losses, val_losses, train_accuracies, val_accuracies
                