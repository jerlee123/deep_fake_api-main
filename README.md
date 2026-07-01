# Deep-Fake API (updated)

This project is a FastAPI backend for video-based presentation-attack and deepfake detection. It supports:

- Per-video deepfake detection (ViViT / ViT-backed pipelines)
- Spoofing/presentation-attack detection (replay, printed, masked)
- Real-time streaming inference via WebSocket
- LLM-powered explainability reports and dashboards

⚠️ Note: model weights (e.g., `checkpoint.pth`) are not included. Download your trained checkpoint into the repo root before starting the server. I keep checkpoints in Drive: `IMAP > publication > amine > checkpoint`.

---

## 🔧 Tech Stack (updated)

- **FastAPI + Uvicorn** — REST + WebSocket API
- **PyTorch** — model runtime (ViViT / ViT)
- **OpenCV, MediaPipe, dlib** — video decoding and face detection
- **Albumentations / torchvision** — preprocessing pipelines
- **Optional:** CUDA for GPU acceleration

---
---

## 📂 Repo structure (high level)

- `apiForAppNewOne.py` — main REST API (predict, predict_csv, integrity endpoints)
- `realtime_ekyc_api.py` — WebSocket server for real-time streaming inference
- `vivit_model.py` — ViViT (spatio-temporal) model implementation for spoofing/liveness
- `deepfake_model.py`, `model_definition.py` — model loader and compatibility wrappers
- `ekyc_pad_pipeline.py` — training/inference pipeline + LLM report generation
- `llm_report_generator.py` — LLM integration for explainability dashboards
- `train_vit_model.py`, `prepare_dataset.py` — training utilities and dataset prep
- `video_utils.py`, `usage_example.py` — preprocessing and client examples
- `load_test_websocket.py` — load testing tool for WebSocket endpoint
- `requirements.txt`, `requirements.lock.txt` — Python dependencies
- `videos/`, `SHORT_VIDEO_HD/` — demo and training videos
- `static/` — stored frames and cropped faces

Note: update your report/README references: the project now includes ViViT-based spoofing detection and a real-time WebSocket interface.

---

## 🚀 How to run

### 1) One-time setup

Windows PowerShell (recommended):
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows CMD:
```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Place your model checkpoint (for example `checkpoint.pth`) in the repository root.

### 2) Start the REST API

If `apiForAppNewOne.py` is in the `deep_fake_api-main` subfolder, run from repo root:

```powershell
uvicorn apiForAppNewOne:app --app-dir .\deep_fake_api-main --host 0.0.0.0 --port 8080 --reload
```

Or change directory into the inner folder and run:

```powershell
Set-Location -Path .\deep_fake_api-main
uvicorn apiForAppNewOne:app --host 0.0.0.0 --port 8080 --reload
```

### 3) Real-time WebSocket API (optional)

```powershell
uvicorn realtime_ekyc_api:app --app-dir .\deep_fake_api-main --host 0.0.0.0 --port 8000 --reload
```

### Quick test

```bash
curl -F "file=@path/to/sample.mp4" http://127.0.0.1:8080/predict/
```
## API endPoint

## 📡 API Endpoints

### 1. `POST /predict/`
This endpoint runs deep-fake detection on a short video.  
It extracts around 30 frames, selects 4 representative ones, and performs inference using the model.  
The response contains the main prediction (FAKE or REAL), the average confidence score, the processing time, the extracted metadata, and links to the frames and cropped faces stored in the static folder.

---

### 2. `POST /predict_csv/`
This endpoint runs detection on about 30 evenly distributed frames of the video.  
Instead of returning only a global prediction, it provides detailed results for each frame, including the predicted label and the confidence score.  
It is useful for generating frame-level analysis and exporting results in CSV-like format.

---

### 3. `POST /integrity/sign`
This endpoint creates an RSA signature for a video file.  
It calculates the SHA256 hash of the video, signs it with a private RSA key, and returns the signature in base64 format along with the hash and the signing timestamp.  
The goal is to prove later that the file has not been modified.

---

### 4. `POST /integrity/verify`
This endpoint checks if a video is authentic by verifying its RSA signature.  
It takes the video and a base64-encoded signature as input, compares the computed SHA256 hash with the signed one, and returns whether the signature is valid or not.  
It also provides additional details such as the hash value and the verification timestamp.
