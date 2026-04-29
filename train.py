import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import get_dataloader
from model_hybrid import HybridASLModel
from tqdm import tqdm
from sklearn.metrics import f1_score, top_k_accuracy_score
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in sign language datasets."""
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

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

def train_model(epochs=150, batch_size=16, lr=3e-4, num_classes=50):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths - Use 100 JSON but filter to top-K classes with most samples
    JSON_PATH = os.path.join('archive', 'nslt_100.json')
    DATA_DIR = os.path.join('archive', 'processed_300')  # reuse same processed data
    CHECKPOINT_DIR = 'checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # DataLoaders - filter to top 50 classes (more samples per class = better accuracy)
    train_loader = get_dataloader(JSON_PATH, DATA_DIR, subset='train', batch_size=batch_size, 
                                  oversample=True, top_k_classes=num_classes)
    
    # Use same class_map for validation
    class_map = train_loader.dataset.class_map
    val_loader = get_dataloader(JSON_PATH, DATA_DIR, subset='val', batch_size=batch_size, 
                                shuffle=False, class_map=class_map)
    
    if len(train_loader) == 0:
        print("No training data found. Run preprocess.py first.")
        return
    
    actual_classes = len(class_map) if class_map else num_classes
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {actual_classes}")
    
    # Model — smaller, higher dropout to prevent overfitting
    model = HybridASLModel(input_dim=459, hidden_dim=128, num_classes=actual_classes, 
                           num_layers=2, dropout=0.5).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
    # Loss - Focal Loss handles class imbalance better
    criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.05)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    best_val_acc = 0.0
    patience = 30
    patience_counter = 0
    
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
            print(f"✓ Checkpoint saved! Best Acc: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_asl_model.pth'))
    # Save class_map for inference
    import json as json_lib
    if class_map:
        with open(os.path.join(CHECKPOINT_DIR, 'class_map.json'), 'w') as f:
            json_lib.dump({str(k): v for k, v in class_map.items()}, f)
    print(f"\nTraining Complete! Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model(epochs=150, batch_size=16, lr=3e-4, num_classes=50)
