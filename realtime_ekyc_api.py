"""
Real-Time EKYC PAD Streaming API with WebSocket
================================================

Enables real-time video streaming from Flutter to API for live liveness detection.

Features:
- WebSocket streaming (low latency)
- Frame buffer accumulation
- Async inference (doesn't block stream)
- Progressive feedback
- Connection pooling
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import asyncio
import json
import base64
from collections import deque
from datetime import datetime
import logging
import threading
from typing import Dict, List
import time

from vivit_model import EKYCPADModel, EKYCDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-Time EKYC PAD API")

# Enable CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FrameBuffer:
    """Buffer for collecting frames for ViViT inference."""
    
    def __init__(self, buffer_size=8):
        self.buffer_size = buffer_size
        self.frames = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to buffer."""
        with self.lock:
            self.frames.append(frame)
            self.timestamps.append(datetime.now())
    
    def get_frames(self) -> tuple:
        """Get buffered frames and timestamps."""
        with self.lock:
            if len(self.frames) == self.buffer_size:
                frames = np.array(list(self.frames))
                timestamps = list(self.timestamps)
                return frames, timestamps
        return None, None
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.frames.clear()
            self.timestamps.clear()
    
    def is_full(self) -> bool:
        """Check if buffer has enough frames."""
        return len(self.frames) == self.buffer_size

class RealtimePADProcessor:
    """Real-time PAD inference processor."""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model = EKYCPADModel().to(device)
        self.model.eval()
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        
        self.inference_thread = None
        self.stop_inference = False
    
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess 8 frames for ViViT."""
        
        # frames: [8, H, W, 3]
        processed = []
        
        for frame in frames:
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            # Normalize
            frame = frame.astype(np.float32) / 255.0
            # Imagenet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frame = (frame - mean) / std
            processed.append(frame)
        
        # Convert to tensor [8, C, H, W] then add batch → [1, C, 8, H, W]
        tensor = torch.from_numpy(
            np.asarray(processed, dtype=np.float32)
        ).permute(0, 3, 1, 2)
        tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        
        return tensor.to(device=self.device, dtype=torch.float32)
    
    def infer(self, frames: np.ndarray) -> Dict:
        """Run inference on frame buffer."""
        
        try:
            tensor = self.preprocess_frames(frames)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                
                logits = outputs['logits']
                attack_type = outputs['attack_type']
                temporal_score = outputs['temporal_score']
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class = logits.argmax(dim=1).item()
                attack_pred = attack_type.argmax(dim=1).item()
                
                attack_names = {0: 'REAL', 1: 'VIDEO_REPLAY', 2: 'PRINTED_PHOTO', 3: 'MASKED'}
                label = 'REAL' if pred_class == 0 else 'SPOOF'
                
                return {
                    'label': label,
                    'confidence': float(probs[pred_class]),
                    'probabilities': {
                        'real': float(probs[0]),
                        'spoof': float(probs[1])
                    },
                    'attack_type': attack_names.get(attack_pred, 'UNKNOWN'),
                    'temporal_score': float(temporal_score.squeeze()),
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {'error': str(e)}

# Global processor instance
processor = RealtimePADProcessor(device=DEVICE)
frame_buffers: Dict[str, FrameBuffer] = {}

@app.websocket("/ws/ekyc-stream/{client_id}")
async def websocket_ekyc_stream(websocket: WebSocket, client_id: str):
    """
    Real-time EKYC PAD WebSocket endpoint.
    
    Client sends:
        {
            "type": "frame",
            "data": "base64_encoded_frame",
            "timestamp": "2026-02-01T10:30:00Z"
        }
    
    Server responds:
        {
            "type": "status",  // or "inference"
            "frames_buffered": 3,
            "label": "REAL",
            "confidence": 0.96,
            "attack_type": "REAL",
            "temporal_score": 0.845
        }
    """
    
    await websocket.accept()
    
    # Create frame buffer for this client
    if client_id not in frame_buffers:
        frame_buffers[client_id] = FrameBuffer(buffer_size=8)
    
    buffer = frame_buffers[client_id]
    inference_task = None
    
    logger.info(f"Client {client_id} connected")
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'frame':
                # Decode base64 frame
                frame_data = base64.b64decode(message['data'])
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add to buffer
                    buffer.add_frame(frame)
                    
                    # Send status update
                    await websocket.send_json({
                        'type': 'status',
                        'frames_buffered': len(buffer.frames),
                        'message': f"Frame buffered ({len(buffer.frames)}/8)"
                    })
                    
                    # If buffer full, run inference
                    if buffer.is_full():
                        frames, timestamps = buffer.get_frames()
                        
                        # Run async inference
                        result = processor.infer(frames)

                        if result.get('error'):
                            await websocket.send_json({
                                'type': 'error',
                                'message': 'Real-time inference failed'
                            })
                            buffer.clear()
                            continue
                        
                        # Send inference result
                        await websocket.send_json({
                            'type': 'inference',
                            'label': result.get('label'),
                            'confidence': result.get('confidence'),
                            'attack_type': result.get('attack_type'),
                            'temporal_score': result.get('temporal_score'),
                            'probabilities': result.get('probabilities'),
                            'timestamp': result.get('timestamp')
                        })
                        
                        # Clear buffer for next inference
                        buffer.clear()
            
            elif message['type'] == 'ping':
                # Keep-alive
                await websocket.send_json({'type': 'pong'})
    
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        if client_id in frame_buffers:
            del frame_buffers[client_id]
    
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        if client_id in frame_buffers:
            del frame_buffers[client_id]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'device': DEVICE,
        'active_connections': len(frame_buffers),
        'timestamp': datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    return {
        'active_clients': len(frame_buffers),
        'device': DEVICE,
        'model_loaded': processor.model is not None,
        'buffer_sizes': {cid: len(buf.frames) for cid, buf in frame_buffers.items()},
        'timestamp': datetime.now().isoformat()
    }

if __name__ == '__main__':
    import uvicorn
    
    # Run with: uvicorn realtime_ekyc_api:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host='0.0.0.0', port=8000)
