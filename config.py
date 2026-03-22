"""
Configuration file for the Real-Time Emotion Detection System
Contains all constants, model paths, and system parameters
"""

import os

# System Configuration
FPS_TARGET = 60  # Increased for higher frame rate
AUDIO_CHUNK_DURATION = 2.0  # seconds
AUDIO_SAMPLE_RATE = 22050
AUDIO_TARGET_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024

# Video Configuration (lower resolution for higher FPS)
VIDEO_WIDTH = 480
VIDEO_HEIGHT = 360
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_IMAGE_SIZE = (48, 48)

# Model Configuration
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
AUDIO_EMOTION_LABELS = ['Angry', 'Calm', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Model Paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, "models", "video_emotion_model.h5")
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "models", "audio_emotion_model.h5")
HAARCASCADE_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")

# Audio Feature Extraction
MFCC_N_COEFFS = 40
N_MELS = 128
HOP_LENGTH = 512

# Fusion Strategy
FUSION_STRATEGY = "weighted_average"  # Options: "majority_voting", "weighted_average", "confidence_weighted"
VIDEO_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4

# Display Configuration
DISPLAY_FONT = 0.8
DISPLAY_THICKNESS = 2
DISPLAY_COLORS = {
    'video': (255, 255, 255),
    'audio': (0, 255, 255),
    'final': (0, 255, 0),
    'background': (0, 0, 0)
}

# Performance Configuration
MAX_QUEUE_SIZE = 10
PREDICTION_BATCH_SIZE = 1
ENABLE_GPU = True
# Detection/tracking tuning
DETECTION_INTERVAL = 5  # run full face detection every N frames (lower = more detection, higher = faster)
TRACKER_TYPE = "KCF"  # Options: KCF (fast), CSRT (accurate)
TEMPORAL_SMOOTHING_WINDOW = 5  # moving average window for smoothing confidences

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
