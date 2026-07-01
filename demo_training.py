"""
Demo: Test ViT Model Training on Sample Data
=============================================

This script creates a small demo dataset from existing videos and trains the ViT model.

Usage:
    python demo_training.py
"""

import os
import shutil
from pathlib import Path
import torch
from train_vit_model import VideoDataset, DataLoader, train_model, evaluate_model
from model_definition import Model

# here the dataset is used
def create_demo_dataset(source_dir='SHORT_VIDEO_HD', demo_dir='demo_dataset'):
    """Create a small demo dataset from existing videos."""

    source_path = Path(source_dir)
    demo_path = Path(demo_dir)

    # Create directories
    (demo_path / 'train' / 'real').mkdir(parents=True, exist_ok=True)
    (demo_path / 'train' / 'fake').mkdir(parents=True, exist_ok=True)
    (demo_path / 'val' / 'real').mkdir(parents=True, exist_ok=True)
    (demo_path / 'val' / 'fake').mkdir(parents=True, exist_ok=True)

    # Copy some videos from existing datasets
    real_videos = list((source_path / 'REAL_DATA_SET').glob('*'))[:5] if (source_path / 'REAL_DATA_SET').exists() else []
    fake_videos = list((source_path / 'FAKE_DATA_SET').glob('*'))[:5] if (source_path / 'FAKE_DATA_SET').exists() else []

    for video in real_videos:
        shutil.copy(video, demo_path / 'train' / 'real' / video.name)

    for video in fake_videos:
        shutil.copy(video, demo_path / 'train' / 'fake' / video.name)

    # For validation, use different videos or same for small dataset
    val_real = real_videos[3:] if len(real_videos) > 3 else real_videos
    val_fake = fake_videos[3:] if len(fake_videos) > 3 else fake_videos

    for video in val_real:
        shutil.copy(video, demo_path / 'val' / 'real' / video.name)

    for video in val_fake:
        shutil.copy(video, demo_path / 'val' / 'fake' / video.name)

    print(f"Demo dataset created at {demo_dir}")
    print(f"Train: {len(real_videos)} real, {len(fake_videos)} fake")
    print(f"Val: {len(val_real)} real, {len(val_fake)} fake")
    return demo_dir

def main():
    print("ViT Deepfake Detection Training Demo")
    print("=" * 40)

    # Create demo dataset
    demo_dir = create_demo_dataset()

    # Create datasets
    train_dataset = VideoDataset(demo_dir + '/train', sequence_length=10)  # Shorter for demo
    val_dataset = VideoDataset(demo_dir + '/val', sequence_length=10)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    print(f"Train dataset: {len(train_dataset)} videos")
    print(f"Val dataset: {len(val_dataset)} videos")

    # Create model
    model = Model(num_classes=2, attn_max_len=10)

    # Train for few epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    history = train_model(model, train_loader, val_loader, num_epochs=5, device=device, save_path='demo_checkpoint.pth')

    # Evaluate
    test_results = evaluate_model(model, val_loader, device)
    print("\nDemo Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1 Score: {test_results['f1_score']:.4f}")

    print("\nDemo completed! Check demo_checkpoint.pth for the trained model.")

if __name__ == '__main__':
    main()