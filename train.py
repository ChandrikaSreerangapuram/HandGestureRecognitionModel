import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataloader
from model_hybrid import HybridASLModel
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for features, labels in tqdm(train_loader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

def train_model(epochs=50, batch_size=32, lr=1e-3):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    JSON_PATH = os.path.join('archive', 'nslt_300.json')
    DATA_DIR = os.path.join('archive', 'processed_300')
    CHECKPOINT_DIR = 'checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # DataLoaders
    train_loader = get_dataloader(JSON_PATH, DATA_DIR, subset='train', batch_size=batch_size)
    val_loader = get_dataloader(JSON_PATH, DATA_DIR, subset='val', batch_size=batch_size)
    
    if len(train_loader) == 0:
        print("No training data found. Run preprocess.py first.")
        return
    
    # Model
    model = HybridASLModel(num_classes=300).to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_asl_model.pth'))
            print(f"Checkpoint saved! Best Acc: {best_val_acc:.2f}%")
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_asl_model.pth'))
    print("Training Complete!")

if __name__ == "__main__":
    train_model(epochs=100, batch_size=16)
