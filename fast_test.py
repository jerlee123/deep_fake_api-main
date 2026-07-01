"""
Fast Training Test
==================

This script trains on a very small subset for quick testing.
"""

import os
import shutil
from pathlib import Path
import torch
from train_vit_model import VideoDataset, DataLoader, train_model, evaluate_model
from model_definition import Model

def create_fast_test_dataset(source_dir='my_training_dataset', test_dir='fast_test', num_videos=4):
    """Create a very small test dataset."""

    source_path = Path(source_dir)
    test_path = Path(test_dir)

    # Create directories
    (test_path / 'train' / 'real').mkdir(parents=True, exist_ok=True)
    (test_path / 'train' / 'fake').mkdir(parents=True, exist_ok=True)
    (test_path / 'val' / 'real').mkdir(parents=True, exist_ok=True)
    (test_path / 'val' / 'fake').mkdir(parents=True, exist_ok=True)

    # Copy just a few videos
    real_videos = list((source_path / 'train' / 'real').glob('*'))[:num_videos//2]
    fake_videos = list((source_path / 'train' / 'fake').glob('*'))[:num_videos//2]

    for video in real_videos:
        shutil.copy(video, test_path / 'train' / 'real' / video.name)
        shutil.copy(video, test_path / 'val' / 'real' / video.name)  # Same for val

    for video in fake_videos:
        shutil.copy(video, test_path / 'train' / 'fake' / video.name)
        shutil.copy(video, test_path / 'val' / 'fake' / video.name)

    print(f"Fast test dataset created with {len(real_videos)} real and {len(fake_videos)} fake videos")
    return test_dir

def main():
    print("Fast ViT Training Test")
    print("=" * 30)

    # Create small test dataset
    test_dir = create_fast_test_dataset(num_videos=4)

    # Create datasets with shorter sequences
    train_dataset = VideoDataset(test_dir + '/train', sequence_length=5)  # Very short
    val_dataset = VideoDataset(test_dir + '/val', sequence_length=5)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Train dataset: {len(train_dataset)} videos")
    print(f"Val dataset: {len(val_dataset)} videos")

    # Create smaller model for testing
    model = Model(num_classes=2, attn_max_len=5, vit_depth=6, vit_num_heads=6)  # Smaller model

    # Train for very few epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} (this will be slow on CPU)")

    try:
        history = train_model(model, train_loader, val_loader, num_epochs=2, device=device, save_path='fast_test_checkpoint.pth')

        # Evaluate
        test_results = evaluate_model(model, val_loader, device)
        print("\nFast Test Results:")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"F1 Score: {test_results['f1_score']:.4f}")

        print("\nFast test completed! The ViT model works.")
        print("For full training, use GPU or reduce dataset size.")

    except KeyboardInterrupt:
        print("\nTraining interrupted. The model setup is working!")

if __name__ == '__main__':
    main()