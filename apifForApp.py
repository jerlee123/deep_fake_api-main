from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import tempfile
from model_definition import Model
from torch import nn
from collections import OrderedDict
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
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import base64
from fastapi import Form

KEYS_DIR = "keys"
PRIVATE_KEY_PATH = os.path.join(KEYS_DIR, "private.pem")
PUBLIC_KEY_PATH = os.path.join(KEYS_DIR, "public.pem")
os.makedirs(KEYS_DIR, exist_ok=True)

app = FastAPI()

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

CROPPED_DIR = "static/cropped_faces"
SPLIT_FRAMES_DIR = "static/split_frames"
os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(SPLIT_FRAMES_DIR, exist_ok=True)

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RSA FUNCTION 
def rsa_generate_keys_if_needed():
    """Génère une paire RSA 2048 bits si elle n'existe pas encore."""
    if not (os.path.exists(PRIVATE_KEY_PATH) and os.path.exists(PUBLIC_KEY_PATH)):
        key = RSA.generate(2048)
        with open(PRIVATE_KEY_PATH, "wb") as f:
            f.write(key.export_key("PEM"))
        with open(PUBLIC_KEY_PATH, "wb") as f:
            f.write(key.publickey().export_key("PEM"))

def sha256_bytes(data: bytes) -> SHA256.SHA256Hash:
    """Renvoie l'objet hash SHA-256 d'un contenu binaire."""
    h = SHA256.new()
    h.update(data)
    return h

def rsa_sign_bytes(data: bytes) -> bytes:
    """Signe un contenu binaire avec la clé privée."""
    rsa_generate_keys_if_needed()
    with open(PRIVATE_KEY_PATH, "rb") as f:
        priv = RSA.import_key(f.read())
    digest = sha256_bytes(data)
    signature = pkcs1_15.new(priv).sign(digest)
    return signature

def rsa_verify_bytes(data: bytes, signature: bytes) -> bool:
    """Vérifie qu'une signature correspond au contenu binaire."""
    rsa_generate_keys_if_needed()
    with open(PUBLIC_KEY_PATH, "rb") as f:
        pub = RSA.import_key(f.read())
    digest = sha256_bytes(data)
    try:
        pkcs1_15.new(pub).verify(digest, signature)
        return True
    except (ValueError, TypeError):
        return False



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

# Load the model (not the last one )
# model = Model(2)
# model.load_state_dict(torch.load("checkpoint.pth", map_location=torch.device('cpu')))
# model.eval()

# --- Charger le checkpoint et déduire attn_max_len ---
state = torch.load("checkpoint.pth", map_location="cpu")
state = state.get("state_dict", state)

pe = state["attention.positional_encoding.pe"]     # [1, L, 1024]
attn_max_len = pe.shape[1]                         # L = 200 dans ton ckpt
print("ATTN MAX LEN FROM CKPT =", attn_max_len)

# --- Instancier le modèle avec la bonne longueur ---
model = Model(num_classes=2, attn_max_len=attn_max_len)
model.to(DEVICE)
model.eval()

# --- Charger strictement ---
missing, unexpected = model.load_state_dict(state, strict=True)
print("Missing:", missing)
print("Unexpected:", unexpected)

# --- Sanity check ---
with torch.no_grad():
    x = torch.randn(2, 4, 3, 224, 224, device=DEVICE)
    _, logits = model(x)
    print("OK shapes:", logits.shape)


#model = CombinedModel(num_classes=2, max_len=200, num_heads=8).to(DEVICE)
#state = torch.load("checkpoint.pth", map_location=DEVICE)
# Si ton fichier est un state_dict "plat" (ce que tu as indiqué), on peut charger strictement :
#model.load_state_dict(state, strict=True)
#model.eval()


# Load the model (latest model)
# def load_checkpoint_compat(model: nn.Module, ckpt_path: str, device="cpu"):
#     sd = torch.load(ckpt_path, map_location=device)

#     # extraire "state_dict" si présent (Lightning, etc.)
#     if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
#         sd = sd["state_dict"]

#     # enlever un éventuel préfixe "module." (DataParallel)
#     sd = {k.replace("module.", ""): v for k, v in sd.items()}

#     model_sd = model.state_dict()
#     remapped = {}

#     for k, v in sd.items():
#         nk = k

#         # 1) feature_extractor.* -> backbone.*
#         if k.startswith("feature_extractor."):
#             nk = "backbone." + k[len("feature_extractor."):]

#         # 2) fc2.* -> linear.*  (on suppose que fc2 est la couche finale)
#         elif k.startswith("fc2."):
#             nk = "linear." + k[len("fc2."):]

#         # 3) ignorer explicitement des blocs non présents dans le modèle
#         if k.startswith(("attention.", "fc1.")):
#             continue

#         # 4) ne garder que si la clé et la forme correspondent
#         if nk in model_sd and model_sd[nk].shape == v.shape:
#             remapped[nk] = v

#     # fusionner avec l'état actuel et charger
#     merged = model_sd.copy()
#     merged.update(remapped)
#     model.load_state_dict(merged, strict=False)

#     loaded_keys = sorted(remapped.keys())
#     missing = sorted(set(model_sd.keys()) - set(loaded_keys))
#     ignored = sorted(set(sd.keys()) - set(remapped.keys()))
#     print(f"[load] loaded={len(loaded_keys)} missing={len(missing)} ignored={len(ignored)}")
#     return {"loaded": loaded_keys, "missing": missing, "ignored": ignored}


# model = Model(num_classes=2, hidden_dim=512, lstm_layers=2, bidirectional=True)
# info = load_checkpoint_compat(model, "checkpoint.pth", device="cpu")
# model.eval()

# def load_checkpoint_strict(model, ckpt_path: str, device="cpu", min_ok_ratio: float = 0.85):
#     """
#     Charge un checkpoint de façon robuste et *strictement contrôlée*.
#     - Dé-neste 'state_dict' (Lightning) si présent
#     - Retire le préfixe 'module.' (DataParallel)
#     - Remappe quelques alias courants vers les noms attendus par le modèle Option A :
#         backbone.* -> feature_extractor.*
#         linear.* -> fc2.*
#         classifier.* -> fc2.*
#     - Ne garde que les clés dont la *forme* correspond exactement
#     - Echec si trop peu de clés sont chargées, ou si la tête fc2 n'est pas chargée

#     Retourne un dict récapitulatif (loaded/missing/unexpected, ratios).
#     """
#     sd_raw = torch.load(ckpt_path, map_location=device)

#     # 1) Extraire un vrai state_dict si Lightning
#     if isinstance(sd_raw, dict) and "state_dict" in sd_raw and isinstance(sd_raw["state_dict"], dict):
#         sd_raw = sd_raw["state_dict"]

#     # 2) Enlever le préfixe 'module.' si présent
#     def strip_module(k: str) -> str:
#         return k.replace("module.", "", 1) if k.startswith("module.") else k

#     sd_raw = {strip_module(k): v for k, v in sd_raw.items()}

#     # 3) Définir le remapping d'alias possibles vers les noms du modèle Option A
#     alias_rules = [
#         # (prefix_source, prefix_cible)
#         ("backbone.", "feature_extractor."),  # certains entraînements nomment le backbone "backbone"
#         ("feature_extractor.", "feature_extractor."),  # identité (au cas où)
#         ("linear.", "fc2."),                  # anciennes têtes nommées "linear" -> "fc2"
#         ("classifier.", "fc2."),              # alias possible
#         # on garde 'lstm.', 'attention.', 'fc1.', 'fc2.' tels quels
#     ]

#     def remap_key(k: str) -> str:
#         for src, dst in alias_rules:
#             if k.startswith(src):
#                 return dst + k[len(src):]
#         return k

#     # 4) Remapper toutes les clés
#     sd_remapped = OrderedDict((remap_key(k), v) for k, v in sd_raw.items())

#     # 5) Faire correspondre uniquement si shapes identiques
#     model_sd = model.state_dict()
#     to_load = {}
#     mismatched = []
#     for k, v in sd_remapped.items():
#         if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
#             to_load[k] = v
#         elif k in model_sd:
#             mismatched.append((k, tuple(getattr(v, "shape", ())), tuple(model_sd[k].shape)))

#     # 6) Calculer stats & décider si on accepte
#     loaded_keys = set(to_load.keys())
#     expected_keys = set(model_sd.keys())
#     missing = sorted(expected_keys - loaded_keys)
#     unexpected = sorted(set(sd_remapped.keys()) - loaded_keys - (expected_keys - loaded_keys))

#     ok_ratio = (len(loaded_keys) / max(1, len(expected_keys)))

#     # 7) Charger (en deux temps : on part d'un état courant puis on met à jour)
#     merged = model_sd.copy()
#     merged.update(to_load)

#     # Important : strict=False ici, mais on contrôle nous-mêmes via ok_ratio + vérifications
#     model.load_state_dict(merged, strict=False)

#     # 8) Vérif spécifique : la tête doit être *réellement* chargée
#     head_keys = [k for k in ["fc2.weight", "fc2.bias"] if k in model_sd]
#     head_loaded = all(k in loaded_keys for k in head_keys)
#     if not head_loaded:
#         # Donner le détail utile pour diagnostiquer rapidement
#         msg = [
#             "[load_checkpoint_strict] ERREUR: la tête 'fc2' n'a pas été chargée correctement.",
#             f"Clés tête attendues: {head_keys}",
#             f"Clés effectivement chargées (extrait): {sorted(h for h in loaded_keys if h.startswith('fc2'))}",
#             f"Clés mismatch (extrait): {mismatched[:3]}",
#         ]
#         raise RuntimeError("\n".join(msg))

#     # 9) Vérif de ratio global
#     if ok_ratio < min_ok_ratio:
#         msg = [
#             f"[load_checkpoint_strict] ERREUR: trop peu de poids chargés ({ok_ratio:.1%} < {min_ok_ratio:.0%}).",
#             f"Chargés: {len(loaded_keys)} / {len(expected_keys)}",
#             f"Manquants (extrait 10): {missing[:10]}",
#             f"Inattendus (extrait 10): {unexpected[:10]}",
#             f"Mismatch (extrait 5): {mismatched[:5]}",
#         ]
#         raise RuntimeError("\n".join(msg))

#     # 10) Rapport lisible
#     report = {
#         "loaded_count": len(loaded_keys),
#         "expected_count": len(expected_keys),
#         "ok_ratio": ok_ratio,
#         "head_loaded": head_loaded,
#         "missing": missing,
#         "unexpected": unexpected,
#         "mismatched": mismatched,
#     }

#     print("[load_checkpoint_strict] OK")
#     print(f"  - loaded:   {len(loaded_keys)} / {len(expected_keys)}  ({ok_ratio:.1%})")
#     print(f"  - head(fc2) loaded: {head_loaded}")
#     if mismatched:
#         print(f"  - mismatched (showing up to 5): {mismatched[:5]}")
#     if unexpected:
#         print(f"  - unexpected (showing up to 5): {unexpected[:5]}")
#     if missing:
#         print(f"  - missing (showing up to 5): {missing[:5]}")
#     return report

# model = Model(num_classes=2, hidden_dim=512, lstm_layers=2, bidirectional=True)
# report = load_checkpoint_strict(model, "/path/to/combined_best.pth", device="cpu")
# model.eval()


def detect_best_face_from_frame(frame: np.ndarray) -> np.ndarray | None:
    """Détecte et retourne le plus grand visage détecté (ou None)."""
    best_crop = None
    max_area = 0

    # MediaPipe
    mp_faces = detect_faces_mediapipe(frame)
    if len(mp_faces) > 0:
        for d in mp_faces:
            x1, y1, x2, y2 = d['bbox']
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_crop = d['img']

    # Haar fallback
    if best_crop is None or best_crop.size == 0:
        haar_faces = detect_faces_haar(frame)
        for d in haar_faces:
            x1, y1, x2, y2 = d['bbox']
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_crop = d['img']

    # Fallback centre image
    if best_crop is None or best_crop.size == 0:
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        x1 = max(cx - 56, 0)
        y1 = max(cy - 56, 0)
        x2 = x1 + 112
        y2 = y1 + 112
        best_crop = frame[y1:y2, x1:x2]

    return best_crop


# Endpoint definitions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
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
            frame_resized = cv2.resize(frame, (224, 224))
            tensor = torch.from_numpy(frame_resized).permute(2,0,1).unsqueeze(0).float() / 255.0
            tensor = tensor.unsqueeze(0)
            if tensor.shape[2] != 3:
                tensor = tensor[:,:,:3,:,:]
            with torch.no_grad():
                # supprime ceci si tu veux revenir comme avant
                tensor = tensor.to(DEVICE)  # <— important si CUDA dispo
                _, logits = model(tensor)
                #_, output = model(tensor) # enlève le commentaire si tu veux revenir comme avant
                probs = torch.softmax(logits, dim=1)
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
            best_crop = detect_best_face_from_frame(frame)

            # --- Sauvegarde du crop (si OK) ---
            if best_crop is not None and best_crop.size > 0:
                try:
                    face_img = cv2.resize(best_crop, (224, 224))
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




@app.get("/integrity/public-key")
def get_public_key():
    rsa_generate_keys_if_needed()
    with open(PUBLIC_KEY_PATH, "rb") as f:
        return {"public_key_pem": f.read().decode("utf-8")}

# --- imports en plus ---
from fastapi import Form

@app.post("/integrity/sign")
async def integrity_sign(file: UploadFile = File(...)):
    """
    Signe *les octets exacts* de la vidéo.
    Retourne la signature en base64 + le SHA-256 en hex.
    """
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
    signature_base64: str = Form(...)
):
    """
    Vérifie que la signature correspond exactement aux octets de la vidéo.
    """
    data = await file.read()
    try:
        sig = base64.b64decode(signature_base64)
    except Exception:
        return {"valid": False, "error": "signature_base64 invalide"}

    ok = rsa_verify_bytes(data, sig)
    return {
        "video_filename": file.filename,
        "valid": ok,
        "sha256_hex": sha256_bytes(data).hexdigest(),
        "verified_at": datetime.utcnow().isoformat() + "Z",
    }



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
            frame_resized = cv2.resize(frame, (224, 224))
            tensor = torch.from_numpy(frame_resized).permute(2,0,1).unsqueeze(0).float() / 255.0
            tensor = tensor.unsqueeze(0)
            if tensor.shape[2] != 3:
                tensor = tensor[:,:,:3,:,:]
            with torch.no_grad():
                #idem suprime ceci si tu veux revenir comme avant
                tensor = tensor.to(DEVICE)  # <— important si CUDA dispo
                _, logits = model(tensor)  
                #_, output = model(tensor) et remets en commentaire cette ligne si tu veux revenir comme avant
                probs = torch.softmax(logits, dim=1)
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
async def detect_faces_endpoint(file: UploadFile = File(...)):
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
        face = detect_best_face_from_frame(frame)
        if face is None or face.size == 0:
            return {"faces": [], "message": "No face found."}

        # Enregistrement du crop
        fname = f"{uuid.uuid4().hex}.jpg"
        fpath = os.path.join(CROPPED_DIR, fname)
        cv2.imwrite(fpath, cv2.resize(face, (224, 224)))

        return {
            "faces": [{
                "url": f"/static/cropped_faces/{fname}"
            }],
            "message": "1 face cropped from middle frame",
            "frame_index": idx
        }

    except Exception as e:
        return {"error": str(e)}

