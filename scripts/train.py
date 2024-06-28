import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import ShadowDataset
from unet_model import UNet
from loss import DiceBCELoss
from utility import seeding, create_dir, epoch_time
from early_stopping import EarlyStopping

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("models")

    """ Load dataset """
    train_x = sorted(glob("C:/Users/aimyn/test_model2/data/SBU/SBU-Train/ShadowImages_resized/*"))
    train_y = sorted(glob("C:/Users/aimyn/test_model2/data/SBU/SBU-Train/ShadowMasks_resized/*"))

    valid_x = sorted(glob("C:/Users/aimyn/test_model2/data/SBU/SBU-Val/ShadowImages_resized/*"))
    valid_y = sorted(glob("C:/Users/aimyn/test_model2/data/SBU/SBU-Val/ShadowMasks_resized/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    batch_size = 8  
    num_epochs = 50
    lr = 1e-4  
    checkpoint_path = "models/unet_model.pth"
    patience = 5  # Number of epochs to wait for improvement

    """ Dataset and loader """
    train_dataset = ShadowDataset(train_x, train_y)
    valid_dataset = ShadowDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4  
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4  
    )

    device = torch.device('cuda')  
    model = UNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss()  

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    """ Training the model """
    for epoch in range(num_epochs):
        start_time = time.time()

        print(f"Starting epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Check early stopping """
        early_stopping(valid_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
