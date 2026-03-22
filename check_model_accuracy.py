"""
Check and display model accuracy information
"""
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from config import VIDEO_MODEL_PATH, AUDIO_MODEL_PATH, EMOTION_LABELS

def check_model_info(model_path, model_type):
    """Check if model exists and get basic info"""
    if not os.path.exists(model_path):
        print(f"❌ {model_type} model not found at: {model_path}")
        return None
    
    try:
        model = load_model(model_path)
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        print(f"\n{'='*60}")
        print(f"{model_type.upper()} MODEL INFO")
        print(f"{'='*60}")
        print(f"Location: {model_path}")
        print(f"File Size: {file_size:.2f} MB")
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        print(f"Number of Parameters: {model.count_params():,}")
        print(f"Number of Layers: {len(model.layers)}")
        print(f"{'='*60}")
        
        return model
    except Exception as e:
        print(f"❌ Error loading {model_type} model: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("MODEL ACCURACY CHECKER")
    print("="*60)
    print("Checking existing models...")
    print("="*60)
    
    video_model = check_model_info(VIDEO_MODEL_PATH, "Video")
    audio_model = check_model_info(AUDIO_MODEL_PATH, "Audio")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if video_model:
        print("✓ Video model is available")
    else:
        print("✗ Video model not found - needs training")
    
    if audio_model:
        print("✓ Audio model is available")
    else:
        print("✗ Audio model not found - needs training")
    
    print("\n" + "="*60)
    print("NOTE: To see training accuracy, run the training script:")
    print("  python train_models.py")
    print("="*60)
    print("\nThe training script will display:")
    print("  - Training Accuracy")
    print("  - Validation Accuracy")
    print("  - Training Loss")
    print("  - Validation Loss")
    print("  - Best epoch accuracies")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()



