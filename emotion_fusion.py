"""
Emotion Fusion Module for Real-Time Emotion Detection
Combines predictions from video and audio models using various fusion strategies
"""

import numpy as np
import logging
from typing import List, Tuple, Dict
from config import *

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class EmotionFusion:
    """
    Handles fusion of emotion predictions from multiple modalities
    """
    
    def __init__(self, strategy: str = FUSION_STRATEGY):
        """
        Initialize the emotion fusion module
        
        Args:
            strategy: Fusion strategy to use
        """
        self.strategy = strategy
        self.emotion_history = []
        self.max_history = 10
        
        logger.info(f"Emotion fusion initialized with strategy: {strategy}")
    
    def add_prediction(self, video_emotion: str, video_confidence: float,
                      audio_emotion: str, audio_confidence: float):
        """
        Add a new prediction to the history
        
        Args:
            video_emotion: Video emotion prediction
            video_confidence: Video confidence score
            audio_emotion: Audio emotion prediction
            audio_confidence: Audio confidence score
        """
        prediction = {
            'video_emotion': video_emotion,
            'video_confidence': video_confidence,
            'audio_emotion': audio_emotion,
            'audio_confidence': audio_confidence,
            'timestamp': len(self.emotion_history)
        }
        
        self.emotion_history.append(prediction)
        
        # Keep only recent history
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
    
    def majority_voting(self, video_emotion: str, audio_emotion: str) -> str:
        """
        Use majority voting to combine predictions
        
        Args:
            video_emotion: Video emotion prediction
            audio_emotion: Audio emotion prediction
            
        Returns:
            Fused emotion prediction
        """
        if video_emotion == audio_emotion:
            return video_emotion
        
        # Count votes from history
        video_votes = sum(1 for p in self.emotion_history if p['video_emotion'] == video_emotion)
        audio_votes = sum(1 for p in self.emotion_history if p['audio_emotion'] == audio_emotion)
        
        if video_votes > audio_votes:
            return video_emotion
        elif audio_votes > video_votes:
            return audio_emotion
        else:
            # Tie - return the one with higher confidence
            return video_emotion if video_emotion != "Error" else audio_emotion
    
    def weighted_average(self, video_emotion: str, video_confidence: float,
                        audio_emotion: str, audio_confidence: float) -> str:
        """
        Use weighted average based on confidence scores
        
        Args:
            video_emotion: Video emotion prediction
            video_confidence: Video confidence score
            audio_emotion: Audio emotion prediction
            audio_confidence: Audio confidence score
            
        Returns:
            Fused emotion prediction
        """
        # If one prediction is invalid, return the other
        if video_emotion in ["Error", "No Model", "No Audio"]:
            return audio_emotion
        if audio_emotion in ["Error", "No Model", "No Audio"]:
            return video_emotion
        
        # If both are the same, return that emotion
        if video_emotion == audio_emotion:
            return video_emotion
        
        # Calculate weighted scores
        video_weight = video_confidence * VIDEO_WEIGHT
        audio_weight = audio_confidence * AUDIO_WEIGHT
        
        if video_weight > audio_weight:
            return video_emotion
        else:
            return audio_emotion
    
    def confidence_weighted(self, video_emotion: str, video_confidence: float,
                           audio_emotion: str, audio_confidence: float) -> str:
        """
        Use confidence-weighted fusion
        
        Args:
            video_emotion: Video emotion prediction
            video_confidence: Video confidence score
            audio_emotion: Audio emotion prediction
            audio_confidence: Audio confidence score
            
        Returns:
            Fused emotion prediction
        """
        # If one prediction is invalid, return the other
        if video_emotion in ["Error", "No Model", "No Audio"]:
            return audio_emotion
        if audio_emotion in ["Error", "No Model", "No Audio"]:
            return video_emotion
        
        # If both are the same, return that emotion
        if video_emotion == audio_emotion:
            return video_emotion
        
        # Use confidence scores directly
        if video_confidence > audio_confidence:
            return video_emotion
        else:
            return audio_emotion
    
    def temporal_fusion(self, video_emotion: str, video_confidence: float,
                       audio_emotion: str, audio_confidence: float) -> str:
        """
        Use temporal fusion considering recent history
        
        Args:
            video_emotion: Video emotion prediction
            video_confidence: Video confidence score
            audio_emotion: Audio emotion prediction
            audio_confidence: Audio confidence score
            
        Returns:
            Fused emotion prediction
        """
        if len(self.emotion_history) < 3:
            return self.weighted_average(video_emotion, video_confidence, 
                                       audio_emotion, audio_confidence)
        
        # Get recent predictions
        recent_predictions = self.emotion_history[-3:]
        
        # Count recent emotions
        emotion_counts = {}
        for pred in recent_predictions:
            for emotion in [pred['video_emotion'], pred['audio_emotion']]:
                if emotion not in ["Error", "No Model", "No Audio"]:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Find most frequent recent emotion
        if emotion_counts:
            most_frequent = max(emotion_counts, key=emotion_counts.get)
            if emotion_counts[most_frequent] >= 2:  # At least 2 out of 3 recent predictions
                return most_frequent
        
        # Fall back to weighted average
        return self.weighted_average(video_emotion, video_confidence, 
                                   audio_emotion, audio_confidence)
    
    def fuse_emotions(self, video_emotion: str, video_confidence: float,
                     audio_emotion: str, audio_confidence: float) -> Tuple[str, float]:
        """
        Fuse emotions using the specified strategy
        
        Args:
            video_emotion: Video emotion prediction
            video_confidence: Video confidence score
            audio_emotion: Audio emotion prediction
            audio_confidence: Audio confidence score
            
        Returns:
            Tuple of (fused_emotion, fused_confidence)
        """
        # Add to history
        self.add_prediction(video_emotion, video_confidence, audio_emotion, audio_confidence)
        
        # Apply fusion strategy
        if self.strategy == "majority_voting":
            fused_emotion = self.majority_voting(video_emotion, audio_emotion)
        elif self.strategy == "weighted_average":
            fused_emotion = self.weighted_average(video_emotion, video_confidence, 
                                                audio_emotion, audio_confidence)
        elif self.strategy == "confidence_weighted":
            fused_emotion = self.confidence_weighted(video_emotion, video_confidence, 
                                                   audio_emotion, audio_confidence)
        elif self.strategy == "temporal_fusion":
            fused_emotion = self.temporal_fusion(video_emotion, video_confidence, 
                                               audio_emotion, audio_confidence)
        else:
            # Default to weighted average
            fused_emotion = self.weighted_average(video_emotion, video_confidence, 
                                                audio_emotion, audio_confidence)
        
        # Calculate fused confidence
        if video_emotion == audio_emotion == fused_emotion:
            fused_confidence = max(video_confidence, audio_confidence)
        else:
            fused_confidence = (video_confidence + audio_confidence) / 2
        
        return fused_emotion, fused_confidence
    
    def get_emotion_statistics(self) -> Dict:
        """
        Get statistics about recent emotion predictions
        
        Returns:
            Dictionary with emotion statistics
        """
        if not self.emotion_history:
            return {}
        
        # Count emotions
        emotion_counts = {}
        for pred in self.emotion_history:
            for emotion in [pred['video_emotion'], pred['audio_emotion']]:
                if emotion not in ["Error", "No Model", "No Audio"]:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate average confidences
        video_confidences = [p['video_confidence'] for p in self.emotion_history 
                           if p['video_confidence'] > 0]
        audio_confidences = [p['audio_confidence'] for p in self.emotion_history 
                           if p['audio_confidence'] > 0]
        
        stats = {
            'emotion_counts': emotion_counts,
            'avg_video_confidence': np.mean(video_confidences) if video_confidences else 0,
            'avg_audio_confidence': np.mean(audio_confidences) if audio_confidences else 0,
            'total_predictions': len(self.emotion_history)
        }
        
        return stats

