from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import tempfile
from model_definition import Model
from torch import nn
from datetime import datetime
import os
import time
import cv2
import numpy as np
import uuid
from fastapi.staticfiles import StaticFiles
import subprocess
import json
import mediapipe as mp
from collections import Counter


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

CROPPED_DIR = "static/cropped_faces"
SPLIT_FRAMES_DIR = "static/split_frames"
os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(SPLIT_FRAMES_DIR, exist_ok=True)

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg"]

def parse_fps(r_frame_rate):
    """Convertit le champ r_frame_rate de ffprobe en float FPS."""
    try:
        if isinstance(r_frame_rate, str):
            num, denom = r_frame_rate.split('/')
            return float(num) / float(denom)
        else:
            return float(r_frame_rate)
    except Exception:
        return 0.0


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_faces_mediapipe(frame: np.ndarray):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    faces = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1 = int(bbox.xmin * iw)
            y1 = int(bbox.ymin * ih)
            x2 = int((bbox.xmin + bbox.width) * iw)
            y2 = int((bbox.ymin + bbox.height) * ih)
            # Clamp aux dimensions de l'image
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, iw)
            y2 = min(y2, ih)
            face_roi = frame[y1:y2, x1:x2]
            faces.append({'img': face_roi, 'bbox': (x1, y1, x2, y2)})
    return faces

def detect_faces_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_imgs = []
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_imgs.append({'img': face_roi, 'bbox': (x, y, x+w, y+h)})
    return face_imgs

def uniform_frame_indices(nb_frames, n):
    if nb_frames < n:
        return list(range(nb_frames))
    return [int(nb_frames * i / n) for i in range(n)]

def extract_metadata(video_path):
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        metadata = json.loads(result.stdout)
        return metadata
    except Exception as e:
        print("Erreur extraction metadata:", e)
        return {}

def simplify_video_metadata(metadata):
    video_stream = None
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    info = {}
    if video_stream:
        info["width"] = video_stream.get("width")
        info["height"] = video_stream.get("height")
        info["video_codec"] = video_stream.get("codec_name")
        info["frame_rate"] = video_stream.get("r_frame_rate")
        info["nb_frames"] = video_stream.get("nb_frames")
        info["bit_rate"] = video_stream.get("bit_rate")
        info["duration"] = video_stream.get("duration")
    fmt = metadata.get("format", {})
    info["file_size"] = fmt.get("size")
    info["format"] = fmt.get("format_long_name")
    info["bit_rate"] = info.get("bit_rate") or fmt.get("bit_rate")
    tags = fmt.get("tags", {})
    if "creation_time" in tags:
        info["creation_time"] = tags["creation_time"]
    if "com.apple.quicktime.location.ISO6709" in tags:
        info["location"] = tags["com.apple.quicktime.location.ISO6709"]
    elif "location" in tags:
        info["location"] = tags["location"]
    device_make = (
        tags.get("com.apple.quicktime.make") or
        tags.get("com.android.manufacturer") or
        tags.get("make") or
        tags.get("device_make")
    )
    device_model = (
        tags.get("com.apple.quicktime.model") or
        tags.get("com.android.model") or
        tags.get("model") or
        tags.get("device_model")
    )
    device_software = (
        tags.get("com.apple.quicktime.software") or
        tags.get("com.android.name") or
        tags.get("software") or
        tags.get("device_software")
    )
    device_encoder = tags.get("encoder")
    if device_make:
        info["device_make"] = device_make
    if device_model:
        info["device_model"] = device_model
    if device_software:
        info["device_software"] = device_software
    if device_encoder:
        info["encoder"] = device_encoder
    return info

def correct_rotation(frame, rotation):
    if rotation is None:
        return frame
    try:
        rot = int(rotation)
        if rot == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rot == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return frame

# Chargement du modèle
#model = Model(2)
#model.load_state_dict(torch.load("checkpoint.pt", map_location=torch.device('cpu')))
#model.eval()

# Chargement du modèle (version optimisée)
# Load the model (latest model)
def load_checkpoint_compat(model: nn.Module, ckpt_path: str, device="cpu"):
    sd = torch.load(ckpt_path, map_location=device)

    # extraire "state_dict" si présent (Lightning, etc.)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # enlever un éventuel préfixe "module." (DataParallel)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    model_sd = model.state_dict()
    remapped = {}

    for k, v in sd.items():
        nk = k

        # 1) feature_extractor.* -> backbone.*
        if k.startswith("feature_extractor."):
            nk = "backbone." + k[len("feature_extractor."):]

        # 2) fc2.* -> linear.*  (on suppose que fc2 est la couche finale)
        elif k.startswith("fc2."):
            nk = "linear." + k[len("fc2."):]

        # 3) ignorer explicitement des blocs non présents dans le modèle
        if k.startswith(("attention.", "fc1.")):
            continue

        # 4) ne garder que si la clé et la forme correspondent
        if nk in model_sd and model_sd[nk].shape == v.shape:
            remapped[nk] = v

    # fusionner avec l'état actuel et charger
    merged = model_sd.copy()
    merged.update(remapped)
    model.load_state_dict(merged, strict=False)

    loaded_keys = sorted(remapped.keys())
    missing = sorted(set(model_sd.keys()) - set(loaded_keys))
    ignored = sorted(set(sd.keys()) - set(remapped.keys()))
    print(f"[load] loaded={len(loaded_keys)} missing={len(missing)} ignored={len(ignored)}")
    return {"loaded": loaded_keys, "missing": missing, "ignored": ignored}

model = Model(num_classes=2, hidden_dim=512, lstm_layers=2, bidirectional=True)
info = load_checkpoint_compat(model, "checkpoint.pth", device="cpu")
model.eval()

def detect_best_face_from_frame(frame: np.ndarray, face_algo: str) -> np.ndarray | None:
    """Détecte et retourne le plus grand visage détecté (ou None)."""
    best_crop = None
    max_area = 0

    if face_algo == "mediapipe":
        faces = detect_faces_mediapipe(frame)
    elif face_algo == "haar":
        faces = detect_faces_haar(frame)
    else:
        faces = []

    for d in faces:
        x1, y1, x2, y2 = d['bbox']
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_crop = d['img']

    # Fallback centre image si aucun visage trouvé
    if best_crop is None or best_crop.size == 0:
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        x1 = max(cx - 56, 0)
        y1 = max(cy - 56, 0)
        x2 = x1 + 112
        y2 = y1 + 112
        best_crop = frame[y1:y2, x1:x2]

    return best_crop



@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    face_algo: str = Query("mediapipe", enum=["mediapipe", "haar"])):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return {"error": f"Unsupported format {ext}. Allowed formats: {', '.join(VIDEO_EXTENSIONS)}"}
    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # 1. Metadata
        metadata_raw = extract_metadata(tmp_path)
        metadata = simplify_video_metadata(metadata_raw)
        fps = parse_fps(metadata.get("frame_rate", "0/1"))
        rotation = metadata.get('rotation', None)
        if not rotation and 'streams' in metadata_raw:
            streams = metadata_raw.get('streams', [])
            if len(streams) > 0:
                for sd in streams[0].get('side_data_list', []):
                    if 'rotation' in sd:
                        rotation = sd['rotation']

        vid = cv2.VideoCapture(tmp_path)
        nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if nb_frames < 1:
            return {"error": "No frames in video."}

        frame_indices = uniform_frame_indices(nb_frames, 30)
        display_idx = [0, 10, 20, 29]
        display_indices = [frame_indices[i] for i in display_idx if i < len(frame_indices)]

        frames_for_display = []
        cropped_faces_info = []
        global_confidences = []

        for idx in display_indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = vid.read()
            if not ret or frame is None:
                continue
            frame = correct_rotation(frame, rotation)
            frame_time = idx / fps if fps > 0 else None
            # Optional : resize la frame avant toute détection (pour normalisation). 
            # frame = cv2.resize(frame, (W, H))  # décommente pour normaliser à W*H, mais la détection fonctionne bien sur la résolution d'origine normalement.

            # --- Save full frame ---
            fname = f"{uuid.uuid4().hex}.jpg"
            fpath = os.path.join(SPLIT_FRAMES_DIR, fname)
            cv2.imwrite(fpath, frame)

            # --- Prediction deepfake ---
            frame_resized = cv2.resize(frame, (112, 112))
            tensor = torch.from_numpy(frame_resized).permute(2,0,1).unsqueeze(0).float() / 255.0
            tensor = tensor.unsqueeze(0)
            if tensor.shape[2] != 3:
                tensor = tensor[:,:,:3,:,:]
            with torch.no_grad():
                _, output = model(tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs).item() * 100
                label = "REAL" if pred_class == 1 else "FAKE"
            frames_for_display.append({
                "frame_index": idx,
                "frame_time_sec": round(frame_time, 2),
                "url": "/static/split_frames/" + fname,
                "confidence": round(confidence, 2),
                "label": label
            })
            global_confidences.append(confidence)
            # --- Détection des visages ---
            best_crop = detect_best_face_from_frame(frame, face_algo)

            # --- Sauvegarde du crop (si OK) ---
            if best_crop is not None and best_crop.size > 0:
                try:
                    face_img = cv2.resize(best_crop, (112, 112))
                    f_face = f"{uuid.uuid4().hex}.jpg"
                    fpath_face = os.path.join(CROPPED_DIR, f_face)
                    cv2.imwrite(fpath_face, face_img)
                    cropped_faces_info.append({
                        "frame_index": idx,
                        "url": "/static/cropped_faces/" + f_face
                    })
                except Exception:
                    pass
            if len(cropped_faces_info) >= 4:
                break

        mean_conf = float(np.mean(global_confidences)) if global_confidences else 0
        
        main_label = Counter([f['label'] for f in frames_for_display]).most_common(1)[0][0] if frames_for_display else "Unknown"
        vid.release()
        elapsed = time.time() - start_time

        return {
            "frames_for_display": frames_for_display,
            "cropped_faces": cropped_faces_info[:4],
            "frames_csv_indices": frame_indices,
            "filename": file.filename,
            "analyzed_at": datetime.utcnow().isoformat() + "Z",
            "processing_time_sec": round(elapsed, 2),
            "metadata": metadata,
            "confidence": round(mean_conf, 2),
            "result": main_label,
        }
    except Exception as e:
        return {"error": str(e)}





@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return {"error": f"Unsupported format {ext}. Allowed formats: {', '.join(VIDEO_EXTENSIONS)}"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # Récupère la rotation
        metadata_raw = extract_metadata(tmp_path)
        metadata = simplify_video_metadata(metadata_raw)
        fps = parse_fps(metadata.get("frame_rate", "0/1"))
        rotation = metadata.get('rotation', None)
        if not rotation and 'streams' in metadata_raw:
            streams = metadata_raw.get('streams', [])
            if len(streams) > 0:
                for sd in streams[0].get('side_data_list', []):
                    if 'rotation' in sd:
                        rotation = sd['rotation']

        vid = cv2.VideoCapture(tmp_path)
        nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if nb_frames == 0:
            return {"error": "No frames detected in video"}
        frame_indices = uniform_frame_indices(nb_frames, 30)
        frame_confidences = []
        for idx in frame_indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = vid.read()
            if not ret or frame is None:
                continue
            frame = correct_rotation(frame, rotation)
            frame_time = idx / fps if fps > 0 else None
            frame_resized = cv2.resize(frame, (112, 112))
            tensor = torch.from_numpy(frame_resized).permute(2,0,1).unsqueeze(0).float() / 255.0
            tensor = tensor.unsqueeze(0)
            if tensor.shape[2] != 3:
                tensor = tensor[:,:,:3,:,:]
            with torch.no_grad():
                _, output = model(tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs).item() * 100
                label = "REAL" if pred_class == 1 else "FAKE"
            frame_confidences.append({
                "frame_index": idx,
                "frame_time_sec": round(frame_time, 2),
                "confidence": round(confidence,2),
                "label": label
            })
        vid.release()
        return {
            "frames_csv": frame_confidences,
            "filename": file.filename,
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/detect_faces/")
async def detect_faces_endpoint(
    file: UploadFile = File(...),
    face_algo: str = Query("mediapipe", enum=["mediapipe", "haar"])):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return {"error": f"Unsupported format {ext}. Allowed: {', '.join(VIDEO_EXTENSIONS)}"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        vid = cv2.VideoCapture(tmp_path)
        nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if nb_frames == 0:
            return {"error": "No frames in video"}

        # Analyse la frame du milieu
        idx = nb_frames // 2
        vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vid.read()
        if not ret or frame is None:
            return {"error": "Failed to read frame"}

        # Détection de visage
        face = detect_best_face_from_frame(frame, face_algo)
        if face is None or face.size == 0:
            return {"faces": [], "message": "No face found."}

        # Enregistrement du crop
        fname = f"{uuid.uuid4().hex}.jpg"
        fpath = os.path.join(CROPPED_DIR, fname)
        cv2.imwrite(fpath, cv2.resize(face, (112, 112)))

        return {
            "faces": [{
                "url": f"/static/cropped_faces/{fname}"
            }],
            "message": "1 face cropped from middle frame",
            "frame_index": idx
        }

    except Exception as e:
        return {"error": str(e)}



def detect_best_face_no_fallback(frame: np.ndarray, face_algo: str):
    """
    Détecte le plus grand visage sur la frame avec l'algo demandé.
    Retourne (face_img, bbox) ou (None, None) si aucun visage détecté.
    Pas de fallback (centre d'image) pour permettre une vraie comparaison des algos.
    """
    if face_algo == "mediapipe":
        faces = detect_faces_mediapipe(frame)
    elif face_algo == "haar":
        faces = detect_faces_haar(frame)
    else:
        faces = []

    if not faces:
        return None, None

    # Choisit le plus grand visage
    best = None
    max_area = 0
    for d in faces:
        x1, y1, x2, y2 = d['bbox']
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best = d
    return (best['img'], best['bbox']) if best else (None, None)


@app.post("/sample_faces/")
async def sample_faces(
    file: UploadFile = File(...),
    face_algo: str = Query("mediapipe", enum=["mediapipe", "haar"]),
    n: int = Query(10, ge=1, le=100)  # par défaut 10 crops
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return {"error": f"Unsupported format {ext}. Allowed: {', '.join(VIDEO_EXTENSIONS)}"}

    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # --- Métadonnées + rotation + fps ---
        metadata_raw = extract_metadata(tmp_path)
        metadata = simplify_video_metadata(metadata_raw)
        fps = parse_fps(metadata.get("frame_rate", "0/1"))

        rotation = metadata.get('rotation', None)
        if not rotation and 'streams' in metadata_raw:
            streams = metadata_raw.get('streams', [])
            if len(streams) > 0:
                for sd in streams[0].get('side_data_list', []):
                    if 'rotation' in sd:
                        rotation = sd['rotation']

        # --- Ouverture vidéo ---
        vid = cv2.VideoCapture(tmp_path)
        nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if nb_frames == 0:
            return {"error": "No frames in video"}

        # --- Indices uniformes ---
        frame_indices = uniform_frame_indices(nb_frames, n)

        results = []
        found = 0

        for idx in frame_indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = vid.read()
            if not ret or frame is None:
                results.append({
                    "frame_index": idx,
                    "detected": False,
                    "reason": "read_error"
                })
                continue

            frame = correct_rotation(frame, rotation)
            face_img, bbox = detect_best_face_no_fallback(frame, face_algo)

            if face_img is None or face_img.size == 0:
                t = (idx / fps) if fps > 0 else None
                results.append({
                    "frame_index": idx,
                    "frame_time_sec": round(t, 2) if t is not None else None,
                    "detected": False
                })
                continue

            # Sauvegarde crop 112x112
            face_resized = cv2.resize(face_img, (112, 112))
            fname = f"{uuid.uuid4().hex}.jpg"
            fpath = os.path.join(CROPPED_DIR, fname)
            cv2.imwrite(fpath, face_resized)

            t = (idx / fps) if fps > 0 else None
            results.append({
                "frame_index": idx,
                "frame_time_sec": round(t, 2) if t is not None else None,
                "detected": True,
                "bbox": [int(v) for v in bbox],
                "url": f"/static/cropped_faces/{fname}"
            })
            found += 1

        vid.release()
        elapsed = time.time() - start_time

        return {
            "filename": file.filename,
            "algorithm": face_algo,
            "n_requested": n,
            "n_returned": found,
            "nb_frames": nb_frames,
            "fps": fps,
            "sampled_indices": frame_indices,
            "faces": results,             # 10 lignes (ou moins si lecture impossible), avec detected True/False
            "metadata": metadata,
            "processing_time_sec": round(elapsed, 2),
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
