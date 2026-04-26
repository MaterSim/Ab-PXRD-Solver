# peak_finder/train.py
"""Training script"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random

from config import Config
from data_utils import PeakDataset
from model import PeakFinderCNN
from loss import FocalLoss, PositionWeightedLoss


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        windows = batch['window'].to(device)
        labels = batch['label'].to(device)
        positions = batch['position'].to(device)
        
        optimizer.zero_grad()
        outputs = model(windows)
        
        # Use position-weighted loss if available
        if hasattr(criterion, 'position_weights'):
            loss = criterion(outputs, labels, positions)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
        # Print average loss and accuracy per epoch
        #print(f"Epoch Training Loss: {total_loss / len(dataloader):.4f}, Accuracy: {100. * correct / total:.2f}%")
    
    return total_loss / len(dataloader), 100. * correct / total


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            windows = batch['window'].to(device)
            labels = batch['label'].to(device)
            positions = batch['position'].to(device)
            
            outputs = model(windows)
            
            # Calculate loss
            if hasattr(criterion, 'position_weights'):
                loss = criterion(outputs, labels, positions)
            else:
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    # Config
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"use device: {device}")
    
    # Data loading
    print("prepare data...")
    full_dataset = PeakDataset(
        cfg.training_data_file,
        window_size=cfg.WINDOW_SIZE,
        samples_per_xrd=cfg.SAMPLES_PER_XRD,
        positive_ratio=cfg.POSITIVE_RATIO
    )
    
    # Split dataset by XRD sample count, seed 42
    n_xrd = len(full_dataset.xrd_list)
    n_val = int(n_xrd * 0.1)
    n_train = n_xrd - n_val

    print(f"Total XRD samples: {n_xrd}")
    print(f"Train XRD samples: {n_train}")
    print(f"Val XRD samples: {n_val}")

    # Randomly sample validation set, seed 42
    random.seed(42)
    indices = list(range(n_xrd))
    random.shuffle(indices)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    print(f"Val indices (first 10): {val_indices[:10]}")

    # Split data
    train_xrd_list = [full_dataset.xrd_list[i] for i in train_indices]
    train_label_list = [full_dataset.label_list[i] for i in train_indices]

    val_xrd_list = [full_dataset.xrd_list[i] for i in val_indices]
    val_label_list = [full_dataset.label_list[i] for i in val_indices]
    
    # Create training dataset
    train_dataset = PeakDataset.__new__(PeakDataset)
    train_dataset.window_size = cfg.WINDOW_SIZE
    train_dataset.samples_per_xrd = cfg.SAMPLES_PER_XRD
    train_dataset.positive_ratio = cfg.POSITIVE_RATIO
    train_dataset.half_window = cfg.WINDOW_SIZE // 2
    train_dataset.xrd_list = train_xrd_list
    train_dataset.label_list = train_label_list
    
    # Create validation dataset
    val_dataset = PeakDataset.__new__(PeakDataset)
    val_dataset.window_size = cfg.WINDOW_SIZE
    val_dataset.samples_per_xrd = cfg.SAMPLES_PER_XRD
    val_dataset.positive_ratio = cfg.POSITIVE_RATIO
    val_dataset.half_window = cfg.WINDOW_SIZE // 2
    val_dataset.xrd_list = val_xrd_list
    val_dataset.label_list = val_label_list
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Train windows per epoch: {len(train_dataset)}")
    print(f"Val windows per epoch: {len(val_dataset)}")
    
    # Model
    print("create model...")
    model = PeakFinderCNN(window_size=cfg.WINDOW_SIZE).to(device)
    
    # Loss and optimizer
    if cfg.USE_POSITION_WEIGHT:
        criterion = PositionWeightedLoss(
            xrd_length=cfg.XRD_LENGTH,
            base_loss='focal'
        ).to(device)
    else:
        criterion = FocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Training loop
    print("begin training...")
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(cfg.EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'{cfg.SAVE_DIR}/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = f'{cfg.SAVE_DIR}/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"New best model! Val Acc: {val_acc:.2f}%")
    
    print(f"\nTraining finished!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")


if __name__ == '__main__':
    main()