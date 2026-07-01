"""
Example Usage of Deepfake Detection Model
=========================================

This file demonstrates how to use the trained deepfake detection model.

Requirements:
- torch
- torchvision
- opencv-python
- albumentations
- numpy
- pathlib

Install with: pip install torch torchvision opencv-python albumentations numpy
"""

import os
from pathlib import Path
from deepfake_model import DeepfakeDetector, load_model

def main():
    """Main example function demonstrating model usage."""
    
    # Configuration - Update these paths for your setup
    MODEL_PATH = "checkpoint.pth"  # Path to your trained model
    VIDEO_DIR = "SHORT_VIDEO_HD"  # Directory containing test videos
    
    print("=== Deepfake Detection Model Usage Example ===\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("Please ensure you have the trained model file (.pth)")
        print("Update MODEL_PATH in this script to point to your model file.")
        return
    
    try:
        # Load the model
        print("🔄 Loading deepfake detection model...")
        detector = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!\n")
        
        # Example 1: Single video prediction
        print("📹 Example 1: Single Video Prediction")
        print("-" * 40)
        
        # You can replace this with any video file path
        test_video = "sample_video.mp4"  # Update with your video path
        
        if os.path.exists(test_video):
            result = detector.predict_video(test_video)
            print(f"Video: {test_video}")
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Probabilities: FAKE={result['probabilities']['FAKE']:.3f}, "
                  f"REAL={result['probabilities']['REAL']:.3f}")
        else:
            print(f"⚠️ Test video not found: {test_video}")
            print("Update 'test_video' path to point to a real video file.")
        
        print()
        
        # Example 2: Batch prediction on directory
        print("📁 Example 2: Batch Prediction")
        print("-" * 40)
        
        if os.path.exists(VIDEO_DIR):
            # Get all video files in directory
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(Path(VIDEO_DIR).glob(f"*{ext}"))
            
            if video_files:
                print(f"Found {len(video_files)} video files in {VIDEO_DIR}")
                
                # Process batch (limit to first 5 for demo)
                batch_files = video_files[:5]
                results = detector.predict_batch([str(f) for f in batch_files])
                
                print("\nBatch Results:")
                for result in results:
                    if 'error' in result:
                        print(f"❌ {Path(result['video_path']).name}: Error - {result['error']}")
                    else:
                        filename = Path(result['video_path']).name
                        print(f"📹 {filename}: {result['label']} (confidence: {result['confidence']:.3f})")
            else:
                print(f"No video files found in {VIDEO_DIR}")
        else:
            print(f"⚠️ Video directory not found: {VIDEO_DIR}")
            print("Create this directory and add some video files to test batch processing.")
        
        print()
        
        # Example 3: Custom configuration
        print("⚙️ Example 3: Custom Model Configuration")
        print("-" * 45)
        
        # If your model was trained with different parameters, specify them here
        custom_config = {
            'num_classes': 2,
            'latent_dim': 2048,
            'lstm_layers': 2,
            'hidden_dim': 1024,
            'num_heads': 4,
            'bidirectional': True,
            'dropout': 0.3
        }
        
        detector_custom = DeepfakeDetector()
        detector_custom.load_model(MODEL_PATH, model_config=custom_config)
        print("✅ Model loaded with custom configuration")
        
        # Get model configuration
        config = detector_custom.model.get_config()
        print("Model configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all dependencies are installed and model path is correct.")

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        if 'cv2' in missing_packages:
            missing_packages[missing_packages.index('cv2')] = 'opencv-python'
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def create_sample_structure():
    """Create sample directory structure for testing."""
    dirs_to_create = ['models', 'test_videos', 'output']
    
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}")
    
    print("\nDirectory structure created!")
    print("Now you can:")
    print("1. Place your trained model (.pth file) in the 'models' directory")
    print("2. Add test videos to the 'test_videos' directory")
    print("3. Run this script again to test the model")

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    
    if check_dependencies():
        main()
    else:
        print("\n" + "="*50)
        print("Please install missing dependencies and run again.")
        
        # Offer to create directory structure
        response = input("\nWould you like to create sample directory structure? (y/n): ")
        if response.lower().startswith('y'):
            create_sample_structure()
