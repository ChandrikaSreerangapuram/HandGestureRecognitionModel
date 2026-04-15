import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ASLDataset(Dataset):
    def __init__(self, json_path, data_dir, subset='train', max_len=60, transform=None):
        self.data_dir = data_dir
        self.subset = subset
        self.max_len = max_len
        self.transform = transform
        
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        # Filter by subset and check if file exists
        self.data_list = []
        for video_id, info in full_data.items():
            if info['subset'] == subset:
                file_path = os.path.join(data_dir, f"{video_id}.npy")
                if os.path.exists(file_path):
                    self.data_list.append((file_path, info['action'][0])) # (path, label)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        # Load landmarks (Seq_Len, 153)
        landmarks = np.load(file_path).astype(np.float32)
        
        # Simple normalization: center around (0.5, 0.5) if not already
        # Landmarks are already in range [0, 1] usually from MediaPipe
        
        # Padding or Truncation
        if len(landmarks) > self.max_len:
            # Sub-sample instead of just truncating
            indices = np.linspace(0, len(landmarks)-1, self.max_len, dtype=int)
            landmarks = landmarks[indices]
        elif len(landmarks) < self.max_len:
            # Zero padding
            padding = np.zeros((self.max_len - len(landmarks), landmarks.shape[1]), dtype=np.float32)
            landmarks = np.concatenate([landmarks, padding], axis=0)
            
        # Feature Engineering: Add Velocity (difference between frames)
        # Velocity = (Seq_Len, 153)
        velocity = np.diff(landmarks, axis=0, append=landmarks[-1:])
        
        # Combine landmarks and velocity (Seq_Len, 306)
        features = np.concatenate([landmarks, velocity], axis=-1)
        
        return torch.tensor(features), torch.tensor(label)

def get_dataloader(json_path, data_dir, subset='train', batch_size=32, max_len=60, shuffle=True):
    dataset = ASLDataset(json_path, data_dir, subset, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test loading
    JSON_PATH = os.path.join('archive', 'nslt_300.json')
    DATA_DIR = os.path.join('archive', 'processed_300')
    
    train_loader = get_dataloader(JSON_PATH, DATA_DIR, subset='train', batch_size=4)
    if len(train_loader) > 0:
        batch_features, batch_labels = next(iter(train_loader))
        print(f"Batch features shape: {batch_features.shape}") # (Batch, 60, 306)
        print(f"Batch labels shape: {batch_labels.shape}")
    else:
        print("No processed data found. Run preprocess.py first.")
