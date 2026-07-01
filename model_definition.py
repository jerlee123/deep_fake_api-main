# model_definition.py
import math
import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_side = img_size // patch_size
        self.num_patches = self.patches_per_side ** 2
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embed  # [B, num_patches + 1, embed_dim]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
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

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = self.patch_embed.cls_token
        self.pos_embed = self.patch_embed.pos_embed
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
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
        x = self.patch_embed(x)  # [B, num_patches + 1, embed_dim]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token = x[:, 0]  # [B, embed_dim]
        return cls_token  # Return CLS token for further processing

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                         # [1, max_len, d_model]
        self.register_buffer("pe", pe)               # => attention.positional_encoding.pe

    def forward(self, x):                             # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, max_len: int = 200):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                             # [B, T, d_model]
        x = self.positional_encoding(x)
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm(x + attn_out)
        return x

class Model(nn.Module):
    def __init__(self, num_classes: int = 2, attn_max_len: int = 200, attn_heads: int = 8,
                 vit_embed_dim=768, vit_depth=12, vit_num_heads=12):
        super().__init__()
        # Vision Transformer backbone
        self.vit = VisionTransformer(
            img_size=224, patch_size=16, in_channels=3, num_classes=vit_embed_dim,
            embed_dim=vit_embed_dim, depth=vit_depth, num_heads=vit_num_heads
        )
        
        self.lstm = nn.LSTM(input_size=vit_embed_dim, hidden_size=512, num_layers=2, bidirectional=True, batch_first=True)

        d_model = 1024  # 512 * 2
        self.attention = TemporalAttention(d_model=d_model, num_heads=attn_heads, max_len=attn_max_len)

        self.fc1 = nn.Linear(d_model, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):                              # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.vit(x)                        # [B*T, 768]
        x = features.view(b, t, -1)                   # [B, T, 768]
        x_lstm, _ = self.lstm(x)                       # [B, T, 1024]
        x_attn = self.attention(x_lstm)                # [B, T, 1024]
        x_seq = torch.mean(x_attn, dim=1)              # [B, 1024]
        x = self.dropout(torch.relu(self.fc1(x_seq)))  # [B, 512]
        logits = self.fc2(x)                           # [B, 2]
        return features, logits
