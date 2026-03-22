"""
Audio Processing Module for Real-Time Emotion Detection
Handles audio capture, feature extraction, and emotion prediction
"""

import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
import logging
from typing import Optional, Tuple, List
from config import *

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles real-time audio processing for emotion detection
    """
    
    def __init__(self, model=None):
        """
        Initialize the audio processor
        
        Args:
            model: Pre-trained audio emotion model
        """
        self.model = model
        self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.audio_stream = None
        self.is_recording = False
        self.audio_buffer = []
        
        # Initialize audio stream
        self._setup_audio_stream()
    
    def _setup_audio_stream(self):
        """Setup audio input stream"""
        try:
            self.audio_stream = sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=AUDIO_SAMPLE_RATE,
                blocksize=AUDIO_CHUNK_SIZE
            )
            logger.info("Audio stream initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio stream: {e}")
            self.audio_stream = None
    
    def _audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio input stream
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio input warning: {status}")
        
        if self.is_recording and not self.audio_queue.full():
            self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        """Start audio recording"""
        if self.audio_stream and not self.is_recording:
            try:
                self.audio_stream.start()
                self.is_recording = True
                logger.info("Audio recording started")
            except Exception as e:
                logger.error(f"Failed to start audio recording: {e}")
    
    def stop_recording(self):
        """Stop audio recording"""
        if self.audio_stream and self.is_recording:
            try:
                self.audio_stream.stop()
                self.is_recording = False
                logger.info("Audio recording stopped")
            except Exception as e:
                logger.error(f"Failed to stop audio recording: {e}")
    
    def extract_mfcc_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio data
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            MFCC features as numpy array
        """
        try:
            # Resample to target sample rate
            if len(audio_data) > 0:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=AUDIO_SAMPLE_RATE, 
                    target_sr=AUDIO_TARGET_SAMPLE_RATE
                )
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=AUDIO_TARGET_SAMPLE_RATE,
                n_mfcc=MFCC_N_COEFFS,
                n_mels=N_MELS,
                hop_length=HOP_LENGTH
            )
            
            # Take mean across time dimension
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            return mfccs_mean
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {e}")
            return np.zeros(MFCC_N_COEFFS)
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract additional spectral features
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Spectral features as numpy array
        """
        try:
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=AUDIO_TARGET_SAMPLE_RATE)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=AUDIO_TARGET_SAMPLE_RATE)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Combine features
            features = np.concatenate([
                np.mean(spectral_centroids),
                np.mean(spectral_rolloff),
                np.mean(zero_crossing_rate)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return np.zeros(3)
    
    def get_audio_emotion(self) -> Tuple[str, float]:
        """
        Get emotion prediction from audio
        
        Returns:
            Tuple of (emotion_label, confidence)
        """
        try:
            if not self.model:
                return "No Model", 0.0
            
            # Collect audio data
            audio_chunks = []
            for _ in range(int(AUDIO_CHUNK_DURATION * AUDIO_SAMPLE_RATE / AUDIO_CHUNK_SIZE)):
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    audio_chunks.append(chunk)
                except queue.Empty:
                    break
            
            if not audio_chunks:
                return "No Audio", 0.0
            
            # Concatenate audio data
            audio_data = np.concatenate(audio_chunks, axis=0)
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]
            
            # Extract features
            mfcc_features = self.extract_mfcc_features(audio_data)
            spectral_features = self.extract_spectral_features(audio_data)
            
            # Combine features
            combined_features = np.concatenate([mfcc_features, spectral_features])
            combined_features = np.expand_dims(combined_features, axis=0)
            
            # Make prediction
            prediction = self.model.predict(combined_features, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            
            # Map to emotion labels (assuming model uses same labels as video)
            if len(prediction) == len(EMOTION_LABELS):
                emotion_label = EMOTION_LABELS[emotion_idx]
            else:
                emotion_label = f"Emotion_{emotion_idx}"
            
            return emotion_label, confidence
            
        except Exception as e:
            logger.error(f"Error getting audio emotion: {e}")
            return "Error", 0.0
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        if self.audio_stream:
            self.audio_stream.close()
        logger.info("Audio processor cleaned up")

