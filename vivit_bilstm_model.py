"""
ViViT + BiLSTM + Temporal Attention Deepfake Detection Model
=============================================================

Pipeline:
    Video
      │
    Frame Extraction  (T=8 uniform frames)
      │
    ViViT Frame Encoder  (shared 2-D ViT per frame → CLS token)
      │                  output: [B, T, embed_dim]
    BiLSTM             (bidirectional LSTM over time)
      │                  output: [B, T, hidden*2]
    Temporal Attention (multi-head self-attention + mean pool)
      │                  output: [B, hidden*2]
    Classifier         (FC → 2 classes: REAL / FAKE)

Author: auto-generated for project deepfake_api
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except Exception:
    qml = None
    PENNYLANE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2-D patch embedding for a single video frame."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W] → [N, num_patches, embed_dim]
        x = self.proj(x)          # [N, embed_dim, H/p, W/p]
        x = x.flatten(2)          # [N, embed_dim, num_patches]
        x = x.transpose(1, 2)     # [N, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """Standard ViT encoder block (pre-norm)."""

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden     = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Stage 1 – ViViT Frame Encoder
# ---------------------------------------------------------------------------

class ViViTFrameEncoder(nn.Module):
    """
    Shared 2-D ViT applied to each frame independently.

    Input:  [B, T, C, H, W]
    Output: [B, T, embed_dim]   (CLS token per frame)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 384,
                 depth: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        returns: [B, T, embed_dim]
        """
        B, T, C, H, W = x.shape

        # Merge batch and time so we can process all frames at once
        x = x.view(B * T, C, H, W)                    # [B*T, C, H, W]
        tokens = self.patch_embed(x)                   # [B*T, N, D]

        # Prepend CLS token
        cls = self.cls_token.expand(B * T, -1, -1)    # [B*T, 1, D]
        tokens = torch.cat((cls, tokens), dim=1)       # [B*T, N+1, D]
        tokens = self.pos_drop(tokens + self.pos_embed)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Return only the CLS token per frame
        cls_out = tokens[:, 0, :]                      # [B*T, D]
        cls_out = cls_out.view(B, T, -1)               # [B, T, D]
        return cls_out


# ---------------------------------------------------------------------------
# Stage 2 – BiLSTM
# ---------------------------------------------------------------------------

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM over the temporal sequence.

    Input:  [B, T, embed_dim]
    Output: [B, T, hidden*2]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)   # [B, T, hidden*2]
        return self.norm(out)


# ---------------------------------------------------------------------------
# Stage 3 – Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over T time steps, then mean pool.

    Input:  [B, T, hidden_dim]
    Output: [B, hidden_dim]
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]  → returns [B, D]"""
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_out)           # [B, T, D]
        return x.mean(dim=1)                            # [B, D]  mean pool over T


class QuantumClassifier(nn.Module):
    """Optional Quantum Neural Network classifier head via PennyLane."""

    def __init__(self, input_dim: int, num_classes: int = 2,
                 n_qubits: int = 4, quantum_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "pennylane is required for --use_quantum mode. Install with: pip install pennylane"
            )

        self.n_qubits = n_qubits
        self.input_proj = nn.Linear(input_dim, n_qubits)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (quantum_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        hidden = max(8, n_qubits * 2)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to a stable angle range for quantum embedding.
        q_in = torch.tanh(self.input_proj(x)) * np.pi
        q_out = self.quantum_layer(q_in)
        return self.post(q_out)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class DeepfakeViViTBiLSTM(nn.Module):
    """
    Full deepfake-detection pipeline:

        Video frames [B, T, C, H, W]
          │
        ViViT Frame Encoder  → [B, T, embed_dim]
          │
        BiLSTM              → [B, T, hidden*2]
          │
        Temporal Attention  → [B, hidden*2]
          │
        Classifier          → [B, 2]
    """

    def __init__(
        self,
        num_classes: int = 2,
        img_size:    int = 224,
        patch_size:  int = 16,
        T:           int = 8,
        embed_dim:   int = 384,
        vit_depth:   int = 4,
        vit_heads:   int = 8,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        attn_heads:  int = 8,
        dropout:     float = 0.3,
        use_quantum: bool = False,
        n_qubits: int = 4,
        quantum_layers: int = 2,
    ):
        super().__init__()
        self.T = T
        self.use_quantum = use_quantum

        # Stage 1 – ViViT frame encoder
        self.vit_encoder = ViViTFrameEncoder(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=vit_depth,
            num_heads=vit_heads, dropout=dropout,
        )

        # Stage 2 – BiLSTM
        self.bilstm = BiLSTMEncoder(
            input_dim=embed_dim, hidden_dim=lstm_hidden,
            num_layers=lstm_layers, dropout=dropout,
        )
        lstm_out_dim = lstm_hidden * 2

        # Stage 3 – Temporal Attention
        self.temporal_attn = TemporalAttention(
            hidden_dim=lstm_out_dim, num_heads=attn_heads, dropout=dropout,
        )

        # Stage 4 – Classifier
        if use_quantum:
            self.classifier = QuantumClassifier(
                input_dim=lstm_out_dim,
                num_classes=num_classes,
                n_qubits=n_qubits,
                quantum_layers=quantum_layers,
                dropout=dropout,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_dim // 2, num_classes),
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, C, H, W]  — T frames per video
        Returns:
            logits: [B, num_classes]
        """
        # Stage 1
        vit_feats = self.vit_encoder(x)      # [B, T, embed_dim]
        # Stage 2
        lstm_out  = self.bilstm(vit_feats)   # [B, T, hidden*2]
        # Stage 3
        pooled    = self.temporal_attn(lstm_out)  # [B, hidden*2]
        # Stage 4
        logits    = self.classifier(pooled)  # [B, 2]
        return logits


# ---------------------------------------------------------------------------
# High-level Detector  (same interface as EKYCDetector)
# ---------------------------------------------------------------------------

class ViViTBiLSTMDetector:
    """
    High-level inference wrapper.
    Compatible with the API's EKYCDetector interface.

    predict_video(tensor): tensor [B, C, T, H, W] or list of RGB frames
    """

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 T: int = 8):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.T      = T
        self.model  = DeepfakeViViTBiLSTM(T=T).to(self.device)
        self.model.eval()

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        raw = torch.load(model_path, map_location=self.device)

        if isinstance(raw, dict) and 'model_state_dict' in raw:
            state = raw['model_state_dict']
            ckpt_args = raw.get('args', {}) or {}

            model_kwargs = {
                'num_classes': 2,
                'img_size': ckpt_args.get('img_size', 224),
                'patch_size': 16,
                'T': ckpt_args.get('T', self.T),
                'embed_dim': ckpt_args.get('embed_dim', 384),
                'vit_depth': ckpt_args.get('vit_depth', 4),
                'vit_heads': ckpt_args.get('vit_heads', 8),
                'lstm_hidden': ckpt_args.get('lstm_hidden', 256),
                'lstm_layers': ckpt_args.get('lstm_layers', 2),
                'attn_heads': ckpt_args.get('attn_heads', 8),
                'dropout': ckpt_args.get('dropout', 0.3),
                'use_quantum': ckpt_args.get('use_quantum', False),
                'n_qubits': ckpt_args.get('n_qubits', 4),
                'quantum_layers': ckpt_args.get('quantum_layers', 2),
            }

            self.T = model_kwargs['T']
            self.model = DeepfakeViViTBiLSTM(**model_kwargs).to(self.device)
        else:
            state = raw

        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        print(f"[ViViTBiLSTMDetector] Loaded from {model_path} on {self.device}")

    # ------------------------------------------------------------------
    # Pre-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_frames(frames_rgb: list, size: int = 224) -> torch.Tensor:
        """
        Convert list of RGB uint8 numpy frames → float tensor [1, T, C, H, W]
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        processed = []
        for f in frames_rgb:
            f = cv2.resize(f, (size, size)).astype(np.float32) / 255.0
            f = (f - mean) / std          # [H, W, C]
            processed.append(f)

        arr = np.stack(processed)         # [T, H, W, C]
        t   = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T, C, H, W]
        return t.unsqueeze(0)             # [1, T, C, H, W]

    @staticmethod
    def extract_frames(video_path: str, T: int = 8) -> list:
        """Extract T uniformly-sampled RGB frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n < 1:
            cap.release()
            raise ValueError(f"No frames in video: {video_path}")
        indices = [int(n * i / T) for i in range(T)]
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        while len(frames) < T:
            frames.append(frames[-1] if frames else
                          np.zeros((224, 224, 3), dtype=np.uint8))
        return frames[:T]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_video(self, input_data) -> dict:
        """
        Accept either:
          - a pre-built tensor  [B, C, T, H, W]  (from API pipeline)
          - a list of RGB numpy frames (len == T)
          - a video file path (str)

        Returns dict compatible with EKYCDetector output:
          { 'label', 'confidence', 'probabilities': {'real', 'spoof'} }
        """
        # --- Build [1, T, C, H, W] input tensor ---
        if isinstance(input_data, str):
            frames = self.extract_frames(input_data, self.T)
            tensor = self.preprocess_frames(frames).to(self.device)  # [1, T, C, H, W]
        elif isinstance(input_data, list):
            frames = input_data
            tensor = self.preprocess_frames(frames).to(self.device)  # [1, T, C, H, W]
        elif isinstance(input_data, torch.Tensor):
            # Accept [B, C, T, H, W] (API convention) → convert to [B, T, C, H, W]
            if input_data.dim() == 5 and input_data.shape[2] == self.T:
                tensor = input_data.permute(0, 2, 1, 3, 4).to(self.device)
            else:
                tensor = input_data.to(self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        with torch.no_grad():
            logits = self.model(tensor)              # [B, 2]
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred   = int(logits.argmax(dim=1).item())

        # label convention: 0=REAL, 1=FAKE/SPOOF
        label      = 'REAL' if pred == 0 else 'SPOOF'
        confidence = float(probs[pred])

        return {
            'label':       label,
            'confidence':  confidence,
            'probabilities': {
                'real':  float(probs[0]),
                'spoof': float(probs[1]),
            },
        }
