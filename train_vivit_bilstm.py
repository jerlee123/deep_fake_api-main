"""
Training Script – ViViT + BiLSTM + Temporal Attention Deepfake Detector
========================================================================

Dataset expected layout (your my_training_dataset):
    <data_dir>/
        train/
            real/   *.mp4  *.MOV  *.avi  *.mov
            fake/   *.mp4  *.MOV  *.avi  *.mov
        val/
            real/
            fake/

Usage:
    py -3 train_vivit_bilstm.py --data_dir my_training_dataset --epochs 30
    py -3 train_vivit_bilstm.py --data_dir my_training_dataset --epochs 30 --device cpu
"""

import os
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from vivit_bilstm_model import DeepfakeViViTBiLSTM


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".MOV"}


class VideoDataset(Dataset):
    """
    Loads videos from:
        <root>/real/*.{mp4,MOV,avi,...}   → label 0
        <root>/fake/*.{mp4,MOV,avi,...}   → label 1

    Extracts T uniformly-sampled frames per video.
    Applies augmentation during training.
    """

    def __init__(self, root_dir: str, T: int = 8,
                 img_size: int = 224, augment: bool = False):
        self.root    = Path(root_dir)
        self.T       = T
        self.img_size = img_size
        self.augment  = augment

        self.paths: list[Path] = []
        self.labels: list[int] = []

        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            folder = self.root / label_name
            if not folder.exists():
                print(f"[WARN] Folder not found, skipping: {folder}")
                continue
            for p in folder.iterdir():
                if p.suffix in VIDEO_EXTS:
                    self.paths.append(p)
                    self.labels.append(label_idx)

        n_real = sum(1 for l in self.labels if l == 0)
        n_fake = sum(1 for l in self.labels if l == 1)
        print(f"  {self.root.name}: {len(self.paths)} videos  "
              f"(real={n_real}, fake={n_fake})")

        if not self.paths:
            raise RuntimeError(f"No videos found under {root_dir}")

        # Pre-processing transforms
        base = [
            TF.Resize((img_size, img_size)),
            TF.ToTensor(),
            TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        aug = [
            TF.RandomHorizontalFlip(),
            TF.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ] if augment else []
        self.transform = TF.Compose(aug + base)

    def __len__(self) -> int:
        return len(self.paths)

    def _extract_frames(self, video_path: Path) -> list:
        """Extract T uniformly-spaced frames as RGB numpy arrays."""
        cap = cv2.VideoCapture(str(video_path))
        n   = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if n == 0 or not cap.isOpened():
            cap.release()
            return [blank] * self.T

        indices = [int(n * i / self.T) for i in range(self.T)]
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(blank)
        cap.release()

        # Pad in case of short video
        while len(frames) < self.T:
            frames.append(frames[-1])
        return frames[: self.T]

    def __getitem__(self, idx: int):
        frames = self._extract_frames(self.paths[idx])
        tensors = [self.transform(Image.fromarray(f)) for f in frames]
        video = torch.stack(tensors)          # [T, C, H, W]
        return video, torch.tensor(self.labels[idx], dtype=torch.long)


# -------------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for videos, labels in tqdm(loader,
                                   desc="Train" if train else "Val",
                                   leave=False):
            videos = videos.to(device)   # [B, T, C, H, W]
            labels = labels.to(device)

            logits = model(videos)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(args):
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("\nLoading datasets...")
    train_ds = VideoDataset(
        os.path.join(args.data_dir, "train"),
        T=args.T, img_size=args.img_size, augment=True,
    )
    val_ds = VideoDataset(
        os.path.join(args.data_dir, "val"),
        T=args.T, img_size=args.img_size, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nBuilding model: ViViT + BiLSTM + Temporal Attention")
    model = DeepfakeViViTBiLSTM(
        num_classes=2,
        img_size=args.img_size,
        patch_size=16,
        T=args.T,
        embed_dim=args.embed_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {total_params:.2f} M")

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    # Weighted loss to handle class imbalance
    n_real = sum(1 for l in train_ds.labels if l == 0)
    n_fake = sum(1 for l in train_ds.labels if l == 1)
    n_total = n_real + n_fake
    w = torch.tensor(
        [n_total / (2 * n_real), n_total / (2 * n_fake)],
        dtype=torch.float, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_count = 0
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [],  "val_acc": []}

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early stop patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion,
                                    optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.1f}%  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.1f}%  "
              f"lr={lr_now:.2e}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"  ✓ Best model saved → {args.save_path}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save training history
    history_path = args.save_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved → {history_path}")
    print(f"Best checkpoint → {args.save_path}  (val_loss={best_val_loss:.4f})")

    return history


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train ViViT+BiLSTM+TemporalAttention deepfake detector")
    p.add_argument("--data_dir",    default="my_training_dataset",
                   help="Root dir with train/ and val/ subfolders")
    p.add_argument("--save_path",   default="ekyc_checkpoint.pth",
                   help="Output checkpoint path")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=2,
                   help="Keep at 2 for small GPU / 1 for CPU")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=8,
                   help="Early-stop patience (epochs without val improvement)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available()
                                    else "cpu")
    # Model hyper-parameters
    p.add_argument("--T",           type=int,   default=8,
                   help="Frames per video clip")
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--embed_dim",   type=int,   default=384,
                   help="ViT embedding dimension")
    p.add_argument("--vit_depth",   type=int,   default=4,
                   help="Number of ViT transformer blocks")
    p.add_argument("--vit_heads",   type=int,   default=8)
    p.add_argument("--lstm_hidden", type=int,   default=256,
                   help="BiLSTM hidden size (output dim = hidden*2)")
    p.add_argument("--lstm_layers", type=int,   default=2)
    p.add_argument("--attn_heads",  type=int,   default=8)
    p.add_argument("--dropout",     type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
