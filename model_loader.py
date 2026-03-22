"""
Model Loading Module for Real-Time Emotion Detection
Handles loading and management of pre-trained models
"""

import os
import logging
from typing import Optional, Tuple
import numpy as np
from tensorflow.keras.models import load_model
from config import *

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles loading and management of pre-trained models
    """
    
    def __init__(self):
        """Initialize the model loader"""
        self.video_model = None
        self.audio_model = None
        self.models_loaded = False
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(VIDEO_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(AUDIO_MODEL_PATH), exist_ok=True)
    
    def load_video_model(self, model_path: str = None) -> bool:
        """
        Load the video emotion detection model
        
        Args:
            model_path: Path to the video model file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_path is None:
                model_path = VIDEO_MODEL_PATH
            
            if not os.path.exists(model_path):
                logger.warning(f"Video model not found at: {model_path}")
                return False
            
            self.video_model = load_model(model_path)
            logger.info(f"Video model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
            self.video_model = None
            return False
    
    def load_audio_model(self, model_path: str = None) -> bool:
        """
        Load the audio emotion detection model
        
        Args:
            model_path: Path to the audio model file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_path is None:
                model_path = AUDIO_MODEL_PATH
            
            if not os.path.exists(model_path):
                logger.warning(f"Audio model not found at: {model_path}")
                return False
            
            self.audio_model = load_model(model_path)
            logger.info(f"Audio model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            self.audio_model = None
            return False
    
    def load_all_models(self) -> Tuple[bool, bool]:
        """
        Load all available models
        
        Returns:
            Tuple of (video_model_loaded, audio_model_loaded)
        """
        video_loaded = self.load_video_model()
        audio_loaded = self.load_audio_model()
        
        self.models_loaded = video_loaded or audio_loaded
        
        if self.models_loaded:
            logger.info("At least one model loaded successfully")
        else:
            logger.warning("No models could be loaded")
        
        return video_loaded, audio_loaded
    
    def get_model_info(self) -> dict:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        info = {
            'video_model_loaded': self.video_model is not None,
            'audio_model_loaded': self.audio_model is not None,
            'models_loaded': self.models_loaded
        }
        
        if self.video_model:
            info['video_model_input_shape'] = self.video_model.input_shape
            info['video_model_output_shape'] = self.video_model.output_shape
        
        if self.audio_model:
            info['audio_model_input_shape'] = self.audio_model.input_shape
            info['audio_model_output_shape'] = self.audio_model.output_shape
        
        return info
    
    def create_dummy_models(self):
        """
        Create dummy models for testing when real models are not available
        This is useful for development and testing
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Dropout
            from tensorflow.keras.optimizers import Adam
            
            # Create dummy video model
            if not self.video_model:
                self.video_model = Sequential([
                    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                    Conv2D(64, (3, 3), activation='relu'),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(len(EMOTION_LABELS), activation='softmax')
                ])
                self.video_model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("Dummy video model created")
            
            # Create dummy audio model
            if not self.audio_model:
                self.audio_model = Sequential([
                    Dense(128, activation='relu', input_shape=(MFCC_N_COEFFS + 3,)),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(len(EMOTION_LABELS), activation='softmax')
                ])
                self.audio_model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("Dummy audio model created")
            
            self.models_loaded = True
            logger.info("Dummy models created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create dummy models: {e}")
    
    def cleanup(self):
        """Clean up model resources"""
        self.video_model = None
        self.audio_model = None
        self.models_loaded = False
        logger.info("Model loader cleaned up")

