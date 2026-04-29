import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class ASLDataset(Dataset):
    def __init__(self, json_path, data_dir, subset='train', max_len=60, transform=None, augment=True, top_k_classes=None, class_map=None):
        self.data_dir = data_dir
        self.subset = subset
        self.max_len = max_len
        self.transform = transform
        self.augment = augment and (subset == 'train')
        
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        # Filter by subset and check if file exists
        self.data_list = []
        for video_id, info in full_data.items():
            if info['subset'] == subset:
                file_path = os.path.join(data_dir, f"{video_id}.npy")
                if os.path.exists(file_path):
                    self.data_list.append((file_path, info['action'][0])) # (path, label)
        
        # Filter to top-K classes (those with most samples) for better accuracy
        if top_k_classes and class_map is None:
            from collections import Counter
            label_counts = Counter(label for _, label in self.data_list)
            top_classes = [cls for cls, _ in label_counts.most_common(top_k_classes)]
            self.class_map = {old: new for new, old in enumerate(sorted(top_classes))}
            self.data_list = [(p, l) for p, l in self.data_list if l in self.class_map]
        elif class_map is not None:
            self.class_map = class_map
            self.data_list = [(p, l) for p, l in self.data_list if l in self.class_map]
        else:
            self.class_map = None
        
    def __len__(self):
        return len(self.data_list)
    
    def _augment_landmarks(self, landmarks):
        """Apply data augmentation to landmark sequences."""
        # 1. Temporal speed variation (stretch/compress by 0.8x-1.2x)
        if random.random() < 0.5:
            speed_factor = random.uniform(0.8, 1.2)
            new_len = max(2, int(len(landmarks) * speed_factor))
            indices = np.linspace(0, len(landmarks) - 1, new_len, dtype=int)
            landmarks = landmarks[indices]
        
        # 2. Gaussian noise on landmarks
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.005, landmarks.shape).astype(np.float32)
            landmarks = landmarks + noise
        
        # 3. Mirror flip (swap left and right hand landmarks)
        if random.random() < 0.3:
            # Left hand: indices 0-62, Right hand: indices 63-125
            flipped = landmarks.copy()
            flipped[:, 0:63], flipped[:, 63:126] = landmarks[:, 63:126], landmarks[:, 0:63]
            # Flip x-coordinates (every 3rd value starting from 0)
            for start in range(0, 126, 3):
                flipped[:, start] = 1.0 - flipped[:, start]
            # Flip pose x-coordinates
            for start in range(126, 153, 3):
                flipped[:, start] = 1.0 - flipped[:, start]
            landmarks = flipped
        
        # 4. Random temporal crop and resize
        if random.random() < 0.3 and len(landmarks) > 10:
            crop_ratio = random.uniform(0.7, 0.95)
            crop_len = max(5, int(len(landmarks) * crop_ratio))
            start = random.randint(0, len(landmarks) - crop_len)
            landmarks = landmarks[start:start + crop_len]
        
        # 5. Frame dropout (randomly drop some frames)
        if random.random() < 0.3 and len(landmarks) > 10:
            keep_ratio = random.uniform(0.8, 0.95)
            keep_indices = sorted(random.sample(range(len(landmarks)), max(5, int(len(landmarks) * keep_ratio))))
            landmarks = landmarks[keep_indices]
        
        return landmarks
    
    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        # Remap label if using top-K classes
        if self.class_map is not None:
            label = self.class_map[label]
        
        # Load landmarks (Seq_Len, 153)
        landmarks = np.load(file_path).astype(np.float32)
        
        # Apply augmentation during training
        if self.augment:
            landmarks = self._augment_landmarks(landmarks)
        
        # Padding or Truncation with uniform temporal sampling
        if len(landmarks) > self.max_len:
            # Sub-sample uniformly
            indices = np.linspace(0, len(landmarks)-1, self.max_len, dtype=int)
            landmarks = landmarks[indices]
        elif len(landmarks) < self.max_len:
            # Repeat-pad instead of zero-pad (more natural for sign language)
            if len(landmarks) > 0:
                repeats = self.max_len // len(landmarks) + 1
                landmarks = np.tile(landmarks, (repeats, 1))[:self.max_len]
            else:
                landmarks = np.zeros((self.max_len, 153), dtype=np.float32)
            
        # Feature Engineering: Add Velocity (difference between frames)
        velocity = np.diff(landmarks, axis=0, append=landmarks[-1:])
        
        # Add Acceleration (2nd order difference)
        acceleration = np.diff(velocity, axis=0, append=velocity[-1:])
        
        # Combine: landmarks + velocity + acceleration (Seq_Len, 459)
        features = np.concatenate([landmarks, velocity, acceleration], axis=-1)
        
        return torch.tensor(features), torch.tensor(label)

def get_dataloader(json_path, data_dir, subset='train', batch_size=32, max_len=60, shuffle=True, oversample=False, top_k_classes=None, class_map=None):
    augment = (subset == 'train')
    dataset = ASLDataset(json_path, data_dir, subset, max_len, augment=augment, top_k_classes=top_k_classes, class_map=class_map)
    
    sampler = None
    if oversample and subset == 'train' and len(dataset) > 0:
        # Weighted random sampling to balance classes
        from torch.utils.data import WeightedRandomSampler
        from collections import Counter
        labels = [label for _, label in dataset.data_list]
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[label] for _, label in dataset.data_list]
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset) * 4, replacement=True)
        shuffle = False  # sampler and shuffle are mutually exclusive
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0, pin_memory=True)

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
