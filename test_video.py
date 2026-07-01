"""
Quick Test: Process a single video
===================================

This script tests if video processing works with a single video.
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

def test_video_processing(video_path):
    """Test processing a single video."""
    print(f"Testing video: {video_path}")

    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("Could not open video")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Frame count: {frame_count}")

        # Read first frame
        ret, frame = cap.read()
        if ret:
            print(f"Frame shape: {frame.shape}")
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"After BGR2RGB: {frame.shape}")

            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            print(f"PIL Image size: {pil_image.size}")

            # Apply transform
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            tensor = transform(pil_image)
            print(f"Tensor shape: {tensor.shape}")
            print("SUCCESS: Video processing works!")
            return True

        cap.release()
        return False

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    # Test with first real video
    import os
    video_path = "my_training_dataset/train/real/VIDEO_1_HD_SHORT.mp4"
    if os.path.exists(video_path):
        test_video_processing(video_path)
    else:
        print(f"Video not found: {video_path}")
        # Try another one
        video_path = "my_training_dataset/train/real/VIDEO_1_HD_SHORT.MOV"
        if os.path.exists(video_path):
            test_video_processing(video_path)
        else:
            print("No test video found")