"""
Prepare Dataset for Training
============================

This script helps organize your video dataset for training the ViT model.

Usage:
    python prepare_dataset.py --source SHORT_VIDEO_HD --output my_dataset --train_split 0.7 --val_split 0.2
"""

import os
import shutil
import argparse
from pathlib import Path
import random

def prepare_dataset(source_dir, output_dir, train_split=0.7, val_split=0.2, test_split=0.1):
    """
    Prepare dataset by splitting videos into train/val/test sets.

    Args:
        source_dir: Directory containing REAL_DATA_SET and FAKE_DATA_SET
        output_dir: Output directory for organized dataset
        train_split: Fraction for training (default 0.7)
        val_split: Fraction for validation (default 0.2)
        test_split: Fraction for testing (default 0.1)
    """

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directories
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']

    for split in splits:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    # Process real videos
    real_videos = list((source_path / 'REAL_DATA_SET').glob('*'))
    random.shuffle(real_videos)

    n_total = len(real_videos)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    print(f"Real videos: {n_total} total")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    for i, video in enumerate(real_videos):
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'val'
        else:
            split = 'test'

        shutil.copy(video, output_path / split / 'real' / video.name)

    # Process fake videos
    fake_videos = list((source_path / 'FAKE_DATA_SET').glob('*'))
    random.shuffle(fake_videos)

    n_total = len(fake_videos)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    print(f"Fake videos: {n_total} total")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    for i, video in enumerate(fake_videos):
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'val'
        else:
            split = 'test'

        shutil.copy(video, output_path / split / 'fake' / video.name)

    print(f"\nDataset prepared at: {output_dir}")
    print("Structure:")
    print(f"  {output_dir}/")
    print("    train/")
    print("      real/  (real videos for training)")
    print("      fake/  (fake videos for training)")
    print("    val/")
    print("      real/  (real videos for validation)")
    print("      fake/  (fake videos for validation)")
    print("    test/")
    print("      real/  (real videos for testing)")
    print("      fake/  (fake videos for testing)")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for ViT training')
    parser.add_argument('--source', type=str, default='SHORT_VIDEO_HD',
                       help='Source directory containing REAL_DATA_SET and FAKE_DATA_SET')
    parser.add_argument('--output', type=str, default='my_dataset',
                       help='Output directory for prepared dataset')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Fraction of data for training (default 0.7)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Fraction of data for validation (default 0.2)')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Fraction of data for testing (default 0.1)')

    args = parser.parse_args()

    # Validate splits
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 0.01:
        print("Error: train_split + val_split + test_split must equal 1.0")
        return

    prepare_dataset(args.source, args.output, args.train_split, args.val_split, args.test_split)

if __name__ == '__main__':
    main()