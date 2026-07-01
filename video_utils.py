import cv2
import torch
from torchvision import transforms
import numpy as np

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def extract_frames(video_path, sequence_length=60):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(frame_count // sequence_length, 1)
    frames = []

    for i in range(0, frame_count, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))
        if len(frames) == sequence_length:
            break

    cap.release()
    return torch.stack(frames).unsqueeze(0)  # shape: (1, seq_len, C, H, W)
