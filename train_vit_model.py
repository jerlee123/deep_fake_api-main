"""
Training Script for ViT-based Deepfake Detection Model
=======================================================

This script trains the Vision Transformer model on deepfake detection datasets.

Usage:
    python train_vit_model.py --data_dir /path/to/dataset --epochs 50 --batch_size 8

Requirements:
    - PyTorch
    - torchvision
    - timm (for data loading utilities)
    - tqdm
    - sklearn (for metrics)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from model_definition import Model

class VideoDataset(Dataset):
    """Dataset for loading video files for deepfake detection."""

    def __init__(self, root_dir, transform=None, sequence_length=30, cache_dir=None):
        """
        Args:
            root_dir: Directory containing 'real' and 'fake' subdirectories
            transform: Transform to apply to frames
            sequence_length: Number of frames to extract per video
            cache_dir: Directory to cache processed videos
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()
        self.sequence_length = sequence_length
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

        # Collect video paths and labels
        self.video_paths = []
        self.labels = []

        # Real videos
        real_dir = self.root_dir / 'real'
        if real_dir.exists():
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                self.video_paths.extend(real_dir.glob(ext))
                self.labels.extend([0] * len(list(real_dir.glob(ext))))  # 0 for real

        # Fake videos
        fake_dir = self.root_dir / 'fake'
        if fake_dir.exists():
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                self.video_paths.extend(fake_dir.glob(ext))
                self.labels.extend([1] * len(list(fake_dir.glob(ext))))  # 1 for fake

        print(f"Found {len(self.video_paths)} videos: {sum(1 for l in self.labels if l == 0)} real, {sum(1 for l in self.labels if l == 1)} fake")

    def _default_transform(self):
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract frames from video
        frames = self._extract_frames(video_path)

        # Apply transforms
        transformed_frames = []
        for frame in frames:
            if self.transform:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                transformed_frames.append(self.transform(pil_image))
            else:
                transformed_frames.append(T.ToTensor()(Image.fromarray(frame)))

        # Stack frames
        video_tensor = torch.stack(transformed_frames)  # [seq_len, C, H, W]

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def _extract_frames(self, video_path):
        """Extract frames from video file using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                return [blank_frame for _ in range(self.sequence_length)]

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                cap.release()
                blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                return [blank_frame for _ in range(self.sequence_length)]

            # Uniform sampling
            frame_step = max(frame_count // self.sequence_length, 1)
            frame_indices = range(0, min(frame_count, frame_step * self.sequence_length), frame_step)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()

            # Handle case with fewer frames than needed
            if len(frames) < self.sequence_length:
                last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
                frames.extend([last_frame] * (self.sequence_length - len(frames)))

            # Ensure we have exactly sequence_length frames
            frames = frames[:self.sequence_length]

            return frames

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Return blank frames
            blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            return [blank_frame for _ in range(self.sequence_length)]

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda', save_path='checkpoint.pth'):
    """Train the model."""

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for videos, labels in tqdm(train_loader, desc="Training"):
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            features, outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc="Validation"):
                videos, labels = videos.to(device), labels.to(device)

                features, outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(".4f")
        print(".4f")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step()

    return history

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set with comprehensive metrics."""

    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Evaluating"):
            videos, labels = videos.to(device), labels.to(device)

            features, outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    except:
        roc_auc = None

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'real': {'precision': precision_per_class[0], 'recall': recall_per_class[0], 'f1': f1_per_class[0]},
            'fake': {'precision': precision_per_class[1], 'recall': recall_per_class[1], 'f1': f1_per_class[1]}
        }
    }

    return results

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history."""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train ViT Deepfake Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length (frames per video)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='vit_checkpoint.pth', help='Path to save model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = VideoDataset(os.path.join(args.data_dir, 'train'))
    val_dataset = VideoDataset(os.path.join(args.data_dir, 'val'))
    test_dataset = VideoDataset(os.path.join(args.data_dir, 'test'))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    print("Creating model...")
    model = Model(num_classes=2, attn_max_len=args.seq_len)

    # Train model
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, args.epochs, device, args.save_path)

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader, device)

    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    print("Training completed!")
    print(f"Test Results: Accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1_score']:.4f}")

if __name__ == '__main__':
    main()