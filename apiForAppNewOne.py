from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch, tempfile, os, time, cv2, numpy as np, uuid, subprocess, json, base64
from datetime import datetime
import mediapipe as mp
from collections import Counter
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256

# >>> NEW: utilise la nouvelle def modèle
# Assure-toi que deepfake_model.py est dans le PYTHONPATH (même dossier que l’API)
from deepfake_model import DeepfakeDetector, load_model

# ------------------ CONFIG & APP ------------------
KEYS_DIR = "keys"
PRIVATE_KEY_PATH = os.path.join(KEYS_DIR, "private.pem")
PUBLIC_KEY_PATH  = os.path.join(KEYS_DIR, "public.pem")
os.makedirs(KEYS_DIR, exist_ok=True)

app = FastAPI()

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

CROPPED_DIR      = "static/cropped_faces"
SPLIT_FRAMES_DIR = "static/split_frames"
os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(SPLIT_FRAMES_DIR, exist_ok=True)

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ RSA UTILS ------------------
def rsa_generate_keys_if_needed():
    if not (os.path.exists(PRIVATE_KEY_PATH) and os.path.exists(PUBLIC_KEY_PATH)):
        key = RSA.generate(2048)
        with open(PRIVATE_KEY_PATH, "wb") as f:
            f.write(key.export_key("PEM"))
        with open(PUBLIC_KEY_PATH, "wb") as f:
            f.write(key.publickey().export_key("PEM"))

def sha256_bytes(data: bytes) -> SHA256.SHA256Hash:
    h = SHA256.new(); h.update(data); return h

def rsa_sign_bytes(data: bytes) -> bytes:
    rsa_generate_keys_if_needed()
    with open(PRIVATE_KEY_PATH, "rb") as f:
        priv = RSA.import_key(f.read())
    return pkcs1_15.new(priv).sign(sha256_bytes(data))

def rsa_verify_bytes(data: bytes, signature: bytes) -> bool:
    rsa_generate_keys_if_needed()
    with open(PUBLIC_KEY_PATH, "rb") as f:
        pub = RSA.import_key(f.read())
    try:
        pkcs1_15.new(pub).verify(sha256_bytes(data), signature)
        return True
    except (ValueError, TypeError):
        return False

# --- Strict base64 helpers (reject extra junk) ---
def b64decode_strict(s: str) -> bytes:
    """
    Strict base64 decode:
    - trims surrounding whitespace (copy/paste safety),
    - rejects any non-base64 characters and bad padding.
    Raises ValueError on invalid input.
    """
    try:
        s = s.strip()
        return base64.b64decode(s, validate=True)  # strict mode
    except Exception as e:
        raise ValueError("Invalid base64 signature") from e

def rsa_expected_sig_len() -> int:
    """
    Returns the expected RSA signature length in bytes
    for the current public key (e.g., 256 for 2048-bit).
    """
    rsa_generate_keys_if_needed()
    with open(PUBLIC_KEY_PATH, "rb") as f:
        pub = RSA.import_key(f.read())
    return pub.size_in_bytes()
# ------------------ VIDEO UTILS ------------------
def parse_fps(r_frame_rate):
    try:
        if isinstance(r_frame_rate, str) and "/" in r_frame_rate:
            num, denom = r_frame_rate.split("/")
            return float(num) / float(denom)
        return float(r_frame_rate)
    except Exception:
        return 0.0

def extract_metadata(video_path):
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
        result = subprocess.run(cmd, capture_output=True)
        return json.loads(result.stdout)
    except Exception:
        return {}

def simplify_video_metadata(metadata):
    video_stream = None
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream; break
    info = {}
    if video_stream:
        info["width"]      = video_stream.get("width")
        info["height"]     = video_stream.get("height")
        info["video_codec"]= video_stream.get("codec_name")
        info["frame_rate"] = video_stream.get("r_frame_rate")
        info["nb_frames"]  = video_stream.get("nb_frames")
        info["bit_rate"]   = video_stream.get("bit_rate")
        info["duration"]   = video_stream.get("duration")
    fmt = metadata.get("format", {})
    info["file_size"] = fmt.get("size")
    info["format"]    = fmt.get("format_long_name")
    info["bit_rate"]  = info.get("bit_rate") or fmt.get("bit_rate")
    tags = fmt.get("tags", {})
    if "creation_time" in tags: info["creation_time"] = tags["creation_time"]
    if "com.apple.quicktime.location.ISO6709" in tags:
        info["location"] = tags["com.apple.quicktime.location.ISO6709"]
    elif "location" in tags:
        info["location"] = tags["location"]
    for k in ("com.apple.quicktime.make","com.android.manufacturer","make","device_make"):
        if tags.get(k): info["device_make"] = tags.get(k); break
    for k in ("com.apple.quicktime.model","com.android.model","model","device_model"):
        if tags.get(k): info["device_model"] = tags.get(k); break
    for k in ("com.apple.quicktime.software","com.android.name","software","device_software"):
        if tags.get(k): info["device_software"] = tags.get(k); break
    if tags.get("encoder"): info["encoder"] = tags.get("encoder")
    return info

def correct_rotation(frame, rotation):
    if rotation is None: return frame
    try:
        rot = int(rotation)
        if   rot ==  90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180: return cv2.rotate(frame, cv2.ROTATE_180)
        elif rot == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return frame

# ------------------ FACE DETECT ------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_faces_mediapipe(frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    faces = []
    if results.detections:
        ih, iw, _ = frame_bgr.shape
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = max(int(bbox.xmin * iw), 0)
            y1 = max(int(bbox.ymin * ih), 0)
            x2 = min(int((bbox.xmin + bbox.width) * iw), iw)
            y2 = min(int((bbox.ymin + bbox.height) * ih), ih)
            faces.append({'img': frame_bgr[y1:y2, x1:x2], 'bbox': (x1,y1,x2,y2)})
    return faces

def detect_faces_haar(frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    out = []
    for (x,y,w,h) in faces:
        out.append({'img': frame_bgr[y:y+h, x:x+w], 'bbox': (x,y,x+w,y+h)})
    return out

def detect_best_face_from_frame(frame_bgr: np.ndarray) -> np.ndarray | None:
    best_crop, max_area = None, 0
    for d in detect_faces_mediapipe(frame_bgr):
        x1,y1,x2,y2 = d['bbox']; area = (x2-x1)*(y2-y1)
        if area > max_area: max_area, best_crop = area, d['img']
    if best_crop is None or best_crop.size == 0:
        for d in detect_faces_haar(frame_bgr):
            x1,y1,x2,y2 = d['bbox']; area = (x2-x1)*(y2-y1)
            if area > max_area: max_area, best_crop = area, d['img']
    if best_crop is None or best_crop.size == 0:
        h,w,_ = frame_bgr.shape; cx,cy = w//2, h//2
        x1 = max(cx-56, 0); y1 = max(cy-56, 0); x2 = x1+112; y2 = y1+112
        best_crop = frame_bgr[y1:y2, x1:x2]
    return best_crop

def uniform_frame_indices(nb_frames, n):
    if nb_frames < n: return list(range(nb_frames))
    return [int(nb_frames * i / n) for i in range(n)]

# ------------------ NOUVEAU: modèle & préproc ------------------

ckpt_config = {
    "num_classes": 2,
    "latent_dim": 2048,   
    "lstm_layers": 2,
    "hidden_dim": 512,    
    "num_heads": 4,
    "bidirectional": True,
    "dropout": 0.3
}
DETECTOR = load_model("checkpoint.pth", model_config=ckpt_config, device=DEVICE)
MODEL    = DETECTOR.model
TRANS    = DETECTOR.transform

def infer_frame_with_new_model(frame_bgr: np.ndarray) -> tuple[str, float]:
    """
    Conserve la logique 'une frame -> une prédiction' de tes endpoints,
    mais utilise le preprocessing & la tête du nouveau modèle.
    Retourne (label, confidence[%]).
    """
    # 1) centrer sur le visage (comme avant)
    face = detect_best_face_from_frame(frame_bgr)
    if face is None or face.size == 0:
        face = frame_bgr

    # 2) BGR->RGB + Albumentations (Resize 180, Normalize ImageNet)
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tens = TRANS(rgb)     

    # 3) forme attendue: [B, T, C, H, W] ; ici B=1, T=1
    seq = tens.unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,3,180,180]

    with torch.no_grad():
        _, logits, _ = MODEL(seq)
        probs = torch.softmax(logits, dim=1)[0]      # [2]
        pred  = int(torch.argmax(probs).item())
        conf  = float(torch.max(probs).item() * 100.0)
        label = "REAL" if pred == 1 else "FAKE"      # mapping conforme au code fourni
    return label, conf

def infer_video_with_new_model(video_path: str, sequence_length: int = 30) -> dict:
    """
    Run the model the way it was designed for final classification:
    one temporal sequence of frames -> one video-level prediction.
    """
    result = DETECTOR.predict_video(video_path, sequence_length=sequence_length)
    confidence = float(result.get("confidence", 0.0) * 100.0)
    probabilities = result.get("probabilities") or {}
    return {
        "label": result.get("label", "Unknown"),
        "confidence": confidence,
        "probabilities": {
            "FAKE": float(probabilities.get("FAKE", 0.0) * 100.0),
            "REAL": float(probabilities.get("REAL", 0.0) * 100.0),
        },
    }

# ------------------ ENDPOINTS ------------------
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": DETECTOR is not None,
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return {"error": f"Unsupported format {ext}. Allowed formats: {', '.join(VIDEO_EXTENSIONS)}"}
    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read()); tmp_path = tmp.name
    try:
        video_prediction = infer_video_with_new_model(tmp_path, sequence_length=30)

        metadata_raw = extract_metadata(tmp_path)
        metadata     = simplify_video_metadata(metadata_raw)
        fps          = parse_fps(metadata.get("frame_rate", "0/1"))
        rotation     = metadata.get("rotation", None)
        if not rotation and 'streams' in metadata_raw:
            for sd in metadata_raw.get('streams', [])[0].get('side_data_list', []):
                if 'rotation' in sd: rotation = sd['rotation']

        vid = cv2.VideoCapture(tmp_path)
        nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if nb_frames < 1:
            return {"error": "No frames in video."}

        frame_indices   = uniform_frame_indices(nb_frames, 30)
        display_idx     = [0, 10, 20, 29]
        display_indices = {frame_indices[i] for i in display_idx if i < len(frame_indices)}

        frames_for_display, cropped_faces_info = [], []
        global_confidences, global_labels = [], []

        # Infer on all sampled frames for final verdict, keep only 4 for UI display.
        for idx in frame_indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = vid.read()
            if not ret or frame is None: continue
            frame = correct_rotation(frame, rotation)
            frame_time = idx / fps if fps > 0 else None

            # Prédiction sur CETTE frame (nouveau pipeline)
            label, confidence = infer_frame_with_new_model(frame)
            global_confidences.append(confidence)
            global_labels.append(label)

            if idx in display_indices:
                # Save display frame only for selected preview indices.
                fname = f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(os.path.join(SPLIT_FRAMES_DIR, fname), frame)

                frames_for_display.append({
                    "frame_index": idx,
                    "frame_time_sec": round(frame_time, 2) if frame_time is not None else None,
                    "url": "/static/split_frames/" + fname,
                    "confidence": round(confidence, 2),
                    "label": label
                })

                # crop visage pour UI (comme avant)
                best_crop = detect_best_face_from_frame(frame)
                if best_crop is not None and best_crop.size > 0:
                    try:
                        face_img = cv2.resize(best_crop, (112, 112))
                        f_face = f"{uuid.uuid4().hex}.jpg"
                        cv2.imwrite(os.path.join(CROPPED_DIR, f_face), face_img)
                        cropped_faces_info.append({
                            "frame_index": idx,
                            "url": "/static/cropped_faces/" + f_face
                        })
                    except Exception:
                        pass

        vid.release()
        elapsed    = time.time() - start_time
        mean_conf  = float(np.mean(global_confidences)) if global_confidences else 0.0
        label_counts = Counter(global_labels)
        frame_vote_label = label_counts.most_common(1)[0][0] if global_labels else "Unknown"

        return {
            "frames_for_display": frames_for_display,
            "cropped_faces": cropped_faces_info[:4],
            "frames_csv_indices": frame_indices,
            "inference_frame_count": len(global_confidences),
            "display_frame_count": len(frames_for_display),
            "frame_vote_result": frame_vote_label,
            "frame_vote_confidence": round(mean_conf, 2),
            "frame_vote_counts": dict(label_counts),
            "probabilities": {
                "FAKE": round(video_prediction["probabilities"]["FAKE"], 2),
                "REAL": round(video_prediction["probabilities"]["REAL"], 2),
            },
            "filename": file.filename,
            "analyzed_at": datetime.utcnow().isoformat() + "Z",
            "processing_time_sec": round(elapsed, 2),
            "metadata": metadata,
            "confidence": round(video_prediction["confidence"], 2),
            "result": video_prediction["label"],
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return {"error": f"Unsupported format {ext}. Allowed formats: {', '.join(VIDEO_EXTENSIONS)}"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read()); tmp_path = tmp.name
    try:
        metadata_raw = extract_metadata(tmp_path)
        metadata     = simplify_video_metadata(metadata_raw)
        fps          = parse_fps(metadata.get("frame_rate", "0/1"))
        rotation     = metadata.get("rotation", None)
        if not rotation and 'streams' in metadata_raw:
            for sd in metadata_raw.get('streams', [])[0].get('side_data_list', []):
                if 'rotation' in sd: rotation = sd['rotation']

        vid = cv2.VideoCapture(tmp_path)
        nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if nb_frames == 0:
            return {"error": "No frames detected in video"}

        frame_indices = uniform_frame_indices(nb_frames, 30)
        frame_confidences = []
        for idx in frame_indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = vid.read()
            if not ret or frame is None: continue
            frame = correct_rotation(frame, rotation)
            frame_time = idx / fps if fps > 0 else None

            label, confidence = infer_frame_with_new_model(frame)

            frame_confidences.append({
                "frame_index": idx,
                "frame_time_sec": round(frame_time, 2) if frame_time is not None else None,
                "confidence": round(confidence, 2),
                "label": label
            })

        vid.release()
        return {"frames_csv": frame_confidences, "filename": file.filename}
    except Exception as e:
        return {"error": str(e)}

@app.post("/integrity/sign")
async def integrity_sign(file: UploadFile = File(...)):
    data = await file.read()
    sig = rsa_sign_bytes(data)
    return {
        "video_filename": file.filename,
        "signature_base64": base64.b64encode(sig).decode("utf-8"),
        "sha256_hex": sha256_bytes(data).hexdigest(),
        "signed_at": datetime.utcnow().isoformat() + "Z",
    }

@app.post("/integrity/verify")
async def integrity_verify(
    file: UploadFile = File(...),
    signature_base64: str = Form(...),
):
    # Read video bytes
    data = await file.read()

    # Strictly decode base64 (rejects any extra characters)
    try:
        sig = b64decode_strict(signature_base64)
    except ValueError as e:
        return {"valid": False, "error": str(e)}

    # Optional: reject unexpected signature length early
    exp_len = rsa_expected_sig_len()  # e.g., 256 bytes for RSA-2048
    if len(sig) != exp_len:
        return {
            "video_filename": file.filename,
            "valid": False,
            "error": f"Signature size mismatch (got {len(sig)}, expected {exp_len}).",
            "sha256_hex": sha256_bytes(data).hexdigest(),
            "verified_at": datetime.utcnow().isoformat() + "Z",
        }

    # Verify RSA-PKCS#1 v1.5 signature
    ok = rsa_verify_bytes(data, sig)
    return {
        "video_filename": file.filename,
        "valid": ok,
        "sha256_hex": sha256_bytes(data).hexdigest(),
        "verified_at": datetime.utcnow().isoformat() + "Z",
    }
