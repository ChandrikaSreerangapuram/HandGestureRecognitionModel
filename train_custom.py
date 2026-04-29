"""
Train model on your custom recorded data.
This gives much better accuracy because the model learns YOUR hands.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import Counter
import json
import random

from model_hybrid import HybridASLModel

CUSTOM_DATA_DIR = "custom_data"
CHECKPOINT_DIR = "checkpoints"


class CustomASLDataset(Dataset):
    def __init__(self, data_dir, max_len=40, subset='train', augment=True, val_ratio=0.2):
        self.max_len = max_len
        self.augment = augment and (subset == 'train')
        self.data_list = []
        self.labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Collect all samples
        all_samples = []
        for label_name in self.labels:
            label_dir = os.path.join(data_dir, label_name)
            files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
            for f in files:
                all_samples.append((os.path.join(label_dir, f), self.label_to_idx[label_name]))
        
        # Split into train/val
        random.seed(42)
        random.shuffle(all_samples)
        split = int(len(all_samples) * (1 - val_ratio))
        
        if subset == 'train':
            self.data_list = all_samples[:split]
        else:
            self.data_list = all_samples[split:]
    
    def _augment(self, landmarks):
        # Gaussian noise
        if random.random() < 0.5:
            landmarks = landmarks + np.random.normal(0, 0.008, landmarks.shape).astype(np.float32)
        
        # Speed variation
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            new_len = max(5, int(len(landmarks) * factor))
            indices = np.linspace(0, len(landmarks) - 1, new_len, dtype=int)
            landmarks = landmarks[indices]
        
        # Mirror flip
        if random.random() < 0.3:
            flipped = landmarks.copy()
            flipped[:, 0:63], flipped[:, 63:126] = landmarks[:, 63:126], landmarks[:, 0:63]
            for start in range(0, 153, 3):
                flipped[:, start] = 1.0 - flipped[:, start]
            landmarks = flipped
        
        # Frame dropout
        if random.random() < 0.3 and len(landmarks) > 8:
            keep = sorted(random.sample(range(len(landmarks)), max(5, int(len(landmarks) * 0.85))))
            landmarks = landmarks[keep]
        
        return landmarks
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        landmarks = np.load(file_path).astype(np.float32)
        
        if self.augment:
            landmarks = self._augment(landmarks)
        
        # Pad/truncate
        if len(landmarks) > self.max_len:
            indices = np.linspace(0, len(landmarks) - 1, self.max_len, dtype=int)
            landmarks = landmarks[indices]
        elif len(landmarks) < self.max_len:
            if len(landmarks) > 0:
                repeats = self.max_len // len(landmarks) + 1
                landmarks = np.tile(landmarks, (repeats, 1))[:self.max_len]
            else:
                landmarks = np.zeros((self.max_len, 153), dtype=np.float32)
        
        # Features: landmarks + velocity + acceleration
        velocity = np.diff(landmarks, axis=0, append=landmarks[-1:])
        acceleration = np.diff(velocity, axis=0, append=velocity[-1:])
        features = np.concatenate([landmarks, velocity, acceleration], axis=-1)
        
        return torch.tensor(features), torch.tensor(label)


def train_custom():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Check data
    if not os.path.exists(CUSTOM_DATA_DIR):
        print(f"ERROR: '{CUSTOM_DATA_DIR}/' not found. Run record_data.py first!")
        return
    
    signs = sorted([d for d in os.listdir(CUSTOM_DATA_DIR) if os.path.isdir(os.path.join(CUSTOM_DATA_DIR, d))])
    if len(signs) == 0:
        print("ERROR: No sign data found!")
        return
    
    # Print dataset info
    print(f"\nSigns found: {signs}")
    for sign in signs:
        count = len([f for f in os.listdir(os.path.join(CUSTOM_DATA_DIR, sign)) if f.endswith('.npy')])
        print(f"  {sign}: {count} samples")
    
    num_classes = len(signs)
    print(f"\nTotal classes: {num_classes}")
    
    # Datasets
    train_dataset = CustomASLDataset(CUSTOM_DATA_DIR, max_len=40, subset='train')
    val_dataset = CustomASLDataset(CUSTOM_DATA_DIR, max_len=40, subset='val', augment=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Oversampling
    labels = [l for _, l in train_dataset.data_list]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[l] for _, l in train_dataset.data_list]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset) * 5, replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Model — small and efficient for custom data
    model = HybridASLModel(input_dim=459, hidden_dim=128, num_classes=num_classes, 
                           num_layers=2, dropout=0.4).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}\n")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_acc = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(100):
        # Train
        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0
        for features, labels_batch in train_loader:
            features, labels_batch = features.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()
        
        scheduler.step()
        train_acc = 100.0 * train_correct / train_total
        
        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features, labels_batch in val_loader:
                features, labels_batch = features.to(device), labels_batch.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
        
        val_acc = 100.0 * val_correct / max(val_total, 1)
        
        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_asl_model.pth'))
    
    # Save metadata
    metadata = {
        "signs": signs,
        "num_classes": num_classes,
        "label_to_idx": {label: idx for idx, label in enumerate(signs)},
        "idx_to_label": {idx: label for idx, label in enumerate(signs)},
        "best_val_acc": best_val_acc
    }
    with open(os.path.join(CHECKPOINT_DIR, 'custom_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Training Complete! Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to: {CHECKPOINT_DIR}/best_asl_model.pth")
    print(f"  Metadata saved to: {CHECKPOINT_DIR}/custom_metadata.json")
    print(f"\n  Next: python3 realtime_custom.py")


if __name__ == "__main__":
    train_custom()
