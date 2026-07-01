# Real-Time EKYC PAD Integration Guide

## Architecture Overview

```
Flutter App (Mobile)
    ↓
Camera Stream (30 FPS)
    ↓
YUV420 → RGBA Conversion
    ↓
JPEG Encoding (80% quality)
    ↓
Base64 → WebSocket
    ↓
Backend (realtime_ekyc_api.py)
    ↓
FrameBuffer (8 frames)
    ↓
ViViT Model (Async Inference)
    ↓
Results → WebSocket Response
    ↓
Flutter UI (Real-time Display)
```

## Backend Setup

### 1. **Install Dependencies**
```bash
cd c:\Users\jer-lee.loo\Downloads\deep_fake_api-main

# Install/upgrade required packages
pip install fastapi uvicorn websockets torch torchvision opencv-python numpy asyncio --upgrade
```

### 2. **Start Real-Time API**
```bash
# Start WebSocket server on port 8000
uvicorn realtime_ekyc_api:app --host 0.0.0.0 --port 8000 --reload

# Or specify device IP (for remote access)
uvicorn realtime_ekyc_api:app --host 192.168.1.100 --port 8000
```

### 3. **Verify API Health**
```bash
# Health check endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "model_loaded": true, "gpu_available": true}
```

## Flutter Setup

### 1. **Create Flutter Project**
```bash
flutter create deepfake_ekyc_flutter
cd deepfake_ekyc_flutter

# Get dependencies
flutter pub get
```

### 2. **Add Files**
Copy the following files to your Flutter project:
- `flutter_realtime_service.dart` → `lib/services/realtime_ekyc_service.dart`
- `realtime_ekyc_screen.dart` → `lib/screens/realtime_ekyc_screen.dart`

### 3. **Update main.dart**
```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'services/realtime_ekyc_service.dart';
import 'screens/realtime_ekyc_screen.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EKYC PAD',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        useMaterial3: true,
      ),
      home: ChangeNotifierProvider(
        create: (_) => RealtimeEKYCService(),
        child: RealtimeEKYCScreen(),
      ),
    );
  }
}
```

### 4. **Update Platform-Specific Permissions**

#### Android (`android/app/AndroidManifest.xml`)
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
```

#### iOS (`ios/Runner/Info.plist`)
```xml
<key>NSCameraUsageDescription</key>
<string>App needs camera access for face liveness detection</string>
<key>NSLocalNetworkUsageDescription</key>
<string>App needs local network access for real-time inference</string>
```

### 5. **Configure Server URL**
Edit `realtime_ekyc_screen.dart`:
```dart
final String serverUrl = '192.168.1.100:8000'; // Change to your server IP
```

Get your server IP:
```bash
# Windows
ipconfig

# Linux/Mac
ifconfig
```

### 6. **Run Flutter App**
```bash
# For Android
flutter run -d android

# For iOS
flutter run -d ios

# Or hot reload on connected device
flutter run --hot
```

## Connection Flow

### WebSocket Connection
```
Client connects to: ws://server_ip:8000/ws/ekyc-stream/client_id

Message format (Client → Server):
{
  "type": "frame",
  "data": "base64_encoded_jpeg_frame",
  "timestamp": "2024-02-01T15:30:45.123456Z"
}

Response format (Server → Client):
{
  "type": "inference",
  "label": "REAL|SPOOF",
  "confidence": 0.95,
  "attack_type": "video_replay|printed|masked|none",
  "temporal_score": 0.87,
  "frames_buffered": 8,
  "inference_time_ms": 245.3
}
```

### Status Messages
```json
{
  "type": "status",
  "message": "Buffering frames (5/8)...",
  "frames_buffered": 5
}
```

## Performance Optimization

### 1. **Frame Size**
- Input: 224×224 (ViViT requirement)
- JPEG Quality: 80% (good balance)
- Compression ratio: ~80KB per frame → ~10KB after compression

### 2. **Network Bandwidth**
- Raw: 80KB/frame × 30 FPS = 2.4 MB/s
- Compressed: 10KB/frame × 4 FPS (inference cadence) = 40 KB/s
- Typical 4G: 20+ Mbps (plenty of headroom)

### 3. **Latency Breakdown**
```
Frame capture:          ~33ms (30 FPS)
JPEG encoding:          ~20ms
Base64 encoding:        ~5ms
Network transit:        ~50ms (LAN) / ~200ms (WAN)
Buffer accumulation:    ~267ms (8 frames @ 30 FPS)
ViViT inference:        ~150-300ms (GPU) / ~500-1000ms (CPU)
Response transit:       ~50ms
UI update:              ~16ms

Total (GPU):            ~550-750ms per inference
Total (CPU):            ~1000-1500ms per inference
```

## Troubleshooting

### Issue: Connection Refused
```
Error: Connection refused to ws://192.168.1.100:8000
```
**Solution:**
1. Verify server is running: `uvicorn realtime_ekyc_api:app --host 0.0.0.0 --port 8000`
2. Check firewall allows port 8000
3. Verify correct IP address: `ipconfig`
4. Test connection: `curl http://192.168.1.100:8000/health`

### Issue: Frames Not Processing
```
Buffered frames stuck at 0
```
**Solution:**
1. Check camera permission is granted
2. Verify camera is available: `flutter devices`
3. Check mobile network can reach backend

### Issue: Slow Inference (>2s)
```
inference_time_ms > 2000
```
**Solution:**
1. Enable GPU acceleration on server
2. Use CPU inference locally (on-device ViViT quantized model)
3. Reduce frame resolution temporarily
4. Check server CPU/memory usage: `nvidia-smi` (GPU) or `top`

### Issue: WebSocket Disconnection
```
Disconnected after N frames
```
**Solution:**
1. Add reconnection logic in `flutter_realtime_service.dart`:
   ```dart
   Future<void> reconnect() async {
     disconnect();
     await Future.delayed(Duration(seconds: 2));
     await connect(_serverUrl!, clientId);
   }
   ```
2. Implement heartbeat (ping/pong already in code)
3. Check server logs: `tail -f server.log`

## Testing

### 1. **Local Testing (Same Machine)**
```bash
# Terminal 1: Start server
python -m uvicorn realtime_ekyc_api:app --host 127.0.0.1 --port 8000

# Terminal 2: Run Flutter on emulator
flutter run -d emulator-5554
```

### 2. **Network Testing**
```bash
# Terminal 1: Start server on network IP
python -m uvicorn realtime_ekyc_api:app --host 0.0.0.0 --port 8000

# Terminal 2: Connect physical device to same WiFi
flutter run -d <device_id>
```

### 3. **Load Testing**
```bash
# Test multiple concurrent connections
python scripts/load_test_websocket.py --num_clients 10 --duration 60
```

## Deployment Checklist

- [ ] Backend server running on cloud/VPS
- [ ] SSL/TLS certificate for WSS (WebSocket Secure)
- [ ] Firewall rules allow port 8000
- [ ] Model weights downloaded: `checkpoint.pth`
- [ ] GPU available or CPU inference acceptable latency
- [ ] Flutter APK/IPA built for distribution
- [ ] Server URL hardcoded or configurable via env
- [ ] Error logging configured
- [ ] Rate limiting enabled to prevent abuse
- [ ] Database connection for storing results (optional)
- [ ] Monitoring/alerting set up (optional)

## Production Considerations

### 1. **Security**
- Use `wss://` (WebSocket Secure) instead of `ws://`
- Implement authentication token in WebSocket header
- Add rate limiting per client
- Sanitize/validate all inputs

### 2. **Scalability**
- Load balance multiple API instances
- Use Redis for distributed frame buffers
- Cache model in GPU memory
- Implement connection pooling

### 3. **Monitoring**
- Log all predictions with timestamps
- Track inference latency metrics
- Monitor WebSocket connection health
- Alert on model errors/failures

### 4. **Model Updates**
- Version the ViViT model (v1, v2, etc.)
- Allow hot-swapping models without downtime
- A/B test different model versions
- Track model performance metrics

## Integration with Existing API

To combine with existing REST API endpoints:

```python
# In realtime_ekyc_api.py
@app.post("/predict")  # Existing upload endpoint
async def predict_upload(file: UploadFile):
    # Original code
    pass

@app.websocket("/ws/ekyc-stream/{client_id}")  # New streaming endpoint
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # New real-time code
    pass
```

Both endpoints can run simultaneously on the same FastAPI instance.

## Expected Results

After successful setup:
- Frame capture at 30 FPS
- Buffering 8 frames (~267ms)
- Inference results every 300-500ms (GPU)
- 95%+ accuracy on real/spoof classification
- Attack type classified in 150-200ms
- Real-time visual feedback on mobile screen

---

**Last Updated:** Feb 1, 2024
**Author:** Deep Fake Detection Team
**Framework:** Flutter + FastAPI + PyTorch
