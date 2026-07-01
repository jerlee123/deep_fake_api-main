"""
ViViT (Video Vision Transformer) for EKYC Presentation Attack Detection (PAD)
==============================================================================

This model detects spoofing attacks by analyzing temporal patterns in videos.
Attacks: video replay, printed photos, mask/recognized attacks.

Author: Implementation for EKYC PAD
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedding3D(nn.Module):
    """3D patch embedding for spatiotemporal video patches."""

    def __init__(self, video_size=224, patch_size=16, time_size=8, temporal_patch_size=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.time_size = time_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # Temporal patching size (must be <= time_size)
        self.temporal_patch_size = temporal_patch_size

        # Spatial patches per side
        self.patches_per_side = video_size // patch_size
        # Temporal patches (use temporal_patch_size, not spatial patch size)
        self.temporal_patches = max(1, time_size // self.temporal_patch_size)

        # Total patches (spatial_patches^2 × temporal_patches)
        self.num_patches = (self.patches_per_side ** 2) * self.temporal_patches

        # 3D Convolution for patch embedding
        # Input: [B, C, T, H, W]
        # Output: [B, embed_dim, T', H', W']
        # Use temporal_patch_size for the temporal kernel so kernel <= time_size
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(self.temporal_patch_size, patch_size, patch_size),
            stride=(self.temporal_patch_size, patch_size, patch_size)
        )
        
        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] - video batch
        Returns:
            embeddings: [B, num_patches + 1, embed_dim]
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, embed_dim, t, h, w]
        
        # Flatten spatial and temporal dimensions
        x = x.flatten(2).transpose(1, 2)  # [B, t*h*w, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        return x

class MultiHeadAttention3D(nn.Module):
    """Multi-head attention for ViViT."""

    def __init__(self, embed_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

class TransformerBlock3D(nn.Module):
    """Transformer block for ViViT."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention3D(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViViT(nn.Module):
    """
    Video Vision Transformer for Presentation Attack Detection (PAD).
    
    Detects spoofing attacks:
    - Video replay: repeated frame patterns
    - Printed photos: no temporal motion
    - Masked/recognized: unnatural micro-expressions
    """

    def __init__(self, video_size=224, patch_size=16, time_size=8, in_channels=3,
                 num_classes=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # Ensure temporal patch size is <= time_size; use 2 as a sensible default
        temporal_patch = 2 if time_size >= 2 else 1
        self.patch_embed = PatchEmbedding3D(video_size=video_size, patch_size=patch_size, time_size=time_size, temporal_patch_size=temporal_patch, in_channels=in_channels, embed_dim=embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock3D(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv_proj.weight)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.xavier_uniform_(block.mlp[0].weight)
            nn.init.xavier_uniform_(block.mlp[3].weight)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] video batch
        Returns:
            logits: [B, num_classes]
            cls_feat: [B, embed_dim] for explainability
        """
        x = self.patch_embed(x)  # [B, num_patches + 1, embed_dim]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_feat = x[:, 0]  # CLS token
        logits = self.head(cls_feat)
        
        return logits, cls_feat

class EKYCPADModel(nn.Module):
    """
    EKYC Presentation Attack Detection model using ViViT.
    
    Detects:
    - Real (liveness): natural motion, micro-expressions, blinking
    - Spoofing: video replay, printed, mask attacks
    """

    def __init__(self, num_classes=2, video_size=224, time_size=8, dropout=0.3):
        super().__init__()
        
        self.vivit = ViViT(
            video_size=video_size,
            patch_size=16,
            time_size=time_size,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            dropout=dropout
        )
        
        # Additional head for attack type classification (optional)
        # Types: 0=Real, 1=VideoReplay, 2=PrintedPhoto, 3=Masked
        self.attack_type_head = nn.Linear(768, 4)
        
        # Temporal consistency module (detect frame repetition)
        self.temporal_consistency = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] video batch (T must match time_size)
        Returns:
            logits: [B, 2] binary classification (real vs spoof)
            cls_feat: [B, 768] CLS token features
            attack_type: [B, 4] attack type predictions
            temporal_score: [B, 1] temporal consistency score
        """
        logits, cls_feat = self.vivit(x)
        
        attack_type = self.attack_type_head(cls_feat)
        temporal_score = self.temporal_consistency(cls_feat)
        
        return {
            'logits': logits,
            'features': cls_feat,
            'attack_type': attack_type,
            'temporal_score': temporal_score
        }

class EKYCDetector:
    """High-level interface for EKYC PAD detection."""

    def __init__(self, model_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EKYCPADModel().to(self.device)
        self.model.eval()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained weights."""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    
    def predict_video(self, video_frames):
        """
        Predict on video.
        
        Args:
            video_frames: [B, C, T, H, W] or video path
        
        Returns:
            result: {
                'label': 'REAL' or 'SPOOF',
                'confidence': float,
                'attack_type': str,
                'temporal_consistency': float,
                'probabilities': {real, spoof}
            }
        """
        with torch.no_grad():
            outputs = self.model(video_frames.to(self.device))
            
            logits = outputs['logits']
            features = outputs['features']
            attack_type = outputs['attack_type']
            temporal_score = outputs['temporal_score']
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = logits.argmax(dim=1).item()
            
            attack_names = {0: 'REAL', 1: 'VIDEO_REPLAY', 2: 'PRINTED_PHOTO', 3: 'MASKED'}
            attack_pred = attack_type.argmax(dim=1).item()
            
            label = 'REAL' if pred_class == 0 else 'SPOOF'
            confidence = float(probs[pred_class])
            
            return {
                'label': label,
                'confidence': confidence,
                'attack_type': attack_names.get(attack_pred, 'UNKNOWN'),
                'temporal_consistency': float(temporal_score.squeeze()),
                'probabilities': {
                    'real': float(probs[0]),
                    'spoof': float(probs[1])
                },
                'features': features.cpu().numpy()[0]
            }
