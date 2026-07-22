# Deep-Fake Detection API

FastAPI backend for uploaded-video deepfake analysis and live eKYC presentation-attack detection (PAD).

The backend provides two separate workflows:

| Workflow | Transport | Entry point | Typical use |
| --- | --- | --- | --- |
| Upload and analyze | HTTP REST | `apiForAppNewOne.py` | Analyze an existing video and produce detailed results |
| Live eKYC | WebSocket | `apiForAppNewOne.py` | Stream camera frames and receive ongoing REAL/SPOOF results |

The recommended setup uses the unified `apiForAppNewOne.py` application, so
both workflows share one local port and one Cloudflare Tunnel.

## Main components

- `apiForAppNewOne.py` - upload-analysis REST API
- `realtime_ekyc_api.py` - live eKYC WebSocket API
- `vivit_model.py` - spatiotemporal model used for liveness/PAD inference
- `ekyc_pad_pipeline.py` - eKYC training and inference pipeline
- `deepfake_model.py`, `model_definition.py` - deepfake model loaders
- `load_test_websocket.py` - WebSocket load-test client
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10 is recommended
- A trained checkpoint compatible with the selected model
- Optional NVIDIA GPU and CUDA for lower inference latency
- Cloudflared, if the mobile app connects through a Cloudflare Tunnel

## One-time setup

Open PowerShell in this directory:

```powershell
cd D:\App\deep_fake_detection\backend\deep_fake_api-main
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the environment again whenever a new terminal is opened:

```powershell
cd D:\App\deep_fake_detection\backend\deep_fake_api-main
.\.venv\Scripts\Activate.ps1
```

## Run the unified REST and Live eKYC API

One process serves both uploaded-video analysis and Live eKYC on port `8080`:

```powershell
python -m uvicorn apiForAppNewOne:app --host 127.0.0.1 --port 8080
```

The unified application provides:

```text
GET  /health
POST /predict/
POST /predict_csv/
POST /integrity/sign
POST /integrity/verify
GET  /realtime/info
GET  /realtime/health
GET  /realtime/stats
WS   /ws/ekyc-stream/{client_id}
```

Swagger is available at `http://127.0.0.1:8080/docs`. WebSocket routes are not
part of OpenAPI, so `/ws/ekyc-stream/{client_id}` will not appear in Swagger;
use `/realtime/info` to see its connection details.

Upload a test video:

```powershell
curl.exe -F "file=@C:\path\to\sample.mp4" http://127.0.0.1:8080/predict/
```

### Check backend health

Open this URL in a browser:

```text
http://127.0.0.1:8080/health
```

Example response:

```json
{
  "status": "healthy",
  "device": "cpu",
  "active_connections": 0,
  "timestamp": "2026-07-23T12:00:00"
}
```

Connection statistics are available at:

```text
http://127.0.0.1:8080/realtime/stats
```

## Connect Live eKYC through Cloudflared

Cloudflare Tunnel supports WebSockets, so no separate WebSocket proxy configuration is required.

First, leave the unified API running on port `8080`. Open a second PowerShell window and run:

```powershell
cloudflared tunnel --url http://127.0.0.1:8080
```

Cloudflared prints a temporary address similar to:

```text
https://example-name.trycloudflare.com
```

Verify the tunnel before opening the Flutter camera:

```text
https://example-name.trycloudflare.com/health
```

Then open **Profile > Unified backend** in the Flutter app and paste:

```text
https://example-name.trycloudflare.com
```

This one setting is used for uploaded-video REST requests. The app also converts
it to `wss://` and adds the Live eKYC WebSocket path automatically.

Quick Tunnel URLs change whenever Cloudflared restarts. For a stable production URL, configure a named tunnel and map a hostname such as `ekyc.example.com` to `http://127.0.0.1:8080`.

## Live eKYC WebSocket protocol

Endpoint:

```text
/ws/ekyc-stream/{client_id}
```

Example client frame message:

```json
{
  "type": "frame",
  "data": "base64_encoded_jpeg",
  "timestamp": "2026-07-23T12:00:00.000Z"
}
```

The server collects eight frames and sends progress messages:

```json
{
  "type": "status",
  "frames_buffered": 5,
  "message": "Frame buffered (5/8)"
}
```

After inference, it returns:

```json
{
  "type": "inference",
  "label": "REAL",
  "confidence": 0.95,
  "attack_type": "REAL",
  "temporal_score": 0.87,
  "probabilities": {
    "real": 0.95,
    "spoof": 0.05
  },
  "timestamp": "2026-07-23T12:00:01.000Z"
}
```

Clients can send `{"type":"ping"}` to receive `{"type":"pong"}`.

## REST endpoints

### `POST /predict/`

Analyzes an uploaded video and returns its REAL/FAKE prediction, confidence, processing information, metadata, and generated frame links.

### `POST /predict_csv/`

Analyzes evenly distributed video frames and returns frame-level predictions suitable for detailed review or CSV export.

### `POST /integrity/sign`

Calculates a SHA-256 hash and creates an RSA signature for an uploaded video.

### `POST /integrity/verify`

Verifies a video against a previously generated RSA signature.

## Model checkpoint warning

Do not treat Live eKYC output as trustworthy until a trained checkpoint has been loaded and validated. At present, `realtime_ekyc_api.py` constructs `RealtimePADProcessor` without providing a checkpoint path. Unless that code is configured to load a compatible trained checkpoint, the ViViT model starts with randomly initialized weights.

After connecting a checkpoint, validate it against a held-out PAD dataset and test printed-photo, screen-replay, mask, low-light, and poor-network scenarios before production use.

## Production checklist

- Use a named Cloudflare Tunnel and stable hostname
- Require authentication for WebSocket clients
- Restrict CORS origins instead of using `*`
- Validate decoded image type and maximum frame size
- Add connection and frame-rate limits
- Load and verify the trained model checkpoint
- Aggregate several inference results before approving an identity
- Log model version, latency, and final decision without unnecessarily retaining biometric frames
- Define consent, retention, encryption, and deletion controls for biometric data

## Troubleshooting

### `/health` works locally but not through Cloudflared

Confirm that both the Uvicorn and Cloudflared terminals are still running and that Cloudflared points to port `8080`.

### Flutter cannot connect

- Test the tunnel `/health` URL from the phone browser
- Use `wss://`, not `ws://`, with an HTTPS Cloudflare hostname
- Enter the HTTPS hostname in **Profile > Unified backend** and let the app add the WebSocket path
- Update the Flutter Profile setting when a Quick Tunnel URL changes

### Port 8080 is already in use

Either stop the process using it or run the unified API on another port:

```powershell
python -m uvicorn apiForAppNewOne:app --host 127.0.0.1 --port 8081
cloudflared tunnel --url http://127.0.0.1:8081
```

The Cloudflare public URL does not expose the local port number, so the Flutter app still uses the generated HTTPS/WSS hostname.
