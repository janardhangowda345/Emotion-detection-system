"""
Example Usage Script for Real-Time Emotion Detection System
Demonstrates different ways to use the emotion detection system
"""

import cv2
import numpy as np
import logging
from real_time_emotion_detection import RealTimeEmotionDetection
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic usage example"""
    print("Basic Usage Example")
    print("=" * 40)
    
    try:
        # Create emotion detector with default settings
        detector = RealTimeEmotionDetection()
        
        # Run the system
        detector.run()
        
    except Exception as e:
        logger.error(f"Error in basic usage: {e}")

def example_custom_configuration():
    """Example with custom configuration"""
    print("Custom Configuration Example")
    print("=" * 40)
    
    try:
        # Create detector with custom settings
        detector = RealTimeEmotionDetection(
            camera_index=0,
            fusion_strategy="majority_voting"
        )
        
        # Run the system
        detector.run()
        
    except Exception as e:
        logger.error(f"Error in custom configuration: {e}")

def example_single_frame_processing():
    """Example of processing a single frame"""
    print("Single Frame Processing Example")
    print("=" * 40)
    
    try:
        # Create detector
        detector = RealTimeEmotionDetection()
        
        # Start processors
        detector.video_processor.start_capture()
        detector.audio_processor.start_recording()
        
        # Process a single frame
        success, frame = detector.video_processor.read_frame()
        if success:
            # Detect faces
            faces = detector.video_processor.detect_faces(frame)
            print(f"Found {len(faces)} faces")
            
            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = frame[y:y+h, x:x+w]
                emotion, confidence = detector.video_processor.get_face_emotion(face_roi)
                print(f"Face {i+1}: {emotion} (confidence: {confidence:.2f})")
            
            # Get audio emotion
            audio_emotion, audio_confidence = detector.audio_processor.get_audio_emotion()
            print(f"Audio: {audio_emotion} (confidence: {audio_confidence:.2f})")
            
            # Fuse emotions
            if faces:
                video_emotion, video_confidence = detector.video_processor.get_face_emotion(
                    frame[faces[0][1]:faces[0][1]+faces[0][3], 
                          faces[0][0]:faces[0][0]+faces[0][2]]
                )
                fused_emotion, fused_confidence = detector.fusion.fuse_emotions(
                    video_emotion, video_confidence,
                    audio_emotion, audio_confidence
                )
                print(f"Fused: {fused_emotion} (confidence: {fused_confidence:.2f})")
        
        # Cleanup
        detector.cleanup()
        
    except Exception as e:
        logger.error(f"Error in single frame processing: {e}")

def example_batch_processing():
    """Example of batch processing multiple frames"""
    print("Batch Processing Example")
    print("=" * 40)
    
    try:
        # Create detector
        detector = RealTimeEmotionDetection()
        
        # Start processors
        detector.video_processor.start_capture()
        detector.audio_processor.start_recording()
        
        # Process multiple frames
        frame_count = 10
        emotions = []
        
        for i in range(frame_count):
            success, frame = detector.video_processor.read_frame()
            if success:
                # Detect faces
                faces = detector.video_processor.detect_faces(frame)
                
                if faces:
                    # Process first face
                    face_roi = frame[faces[0][1]:faces[0][1]+faces[0][3], 
                                   faces[0][0]:faces[0][0]+faces[0][2]]
                    emotion, confidence = detector.video_processor.get_face_emotion(face_roi)
                    emotions.append(emotion)
                    print(f"Frame {i+1}: {emotion} (confidence: {confidence:.2f})")
                else:
                    print(f"Frame {i+1}: No faces detected")
            else:
                print(f"Frame {i+1}: Failed to read")
        
        # Analyze results
        if emotions:
            from collections import Counter
            emotion_counts = Counter(emotions)
            most_common = emotion_counts.most_common(1)[0]
            print(f"\nMost common emotion: {most_common[0]} ({most_common[1]} times)")
        
        # Cleanup
        detector.cleanup()
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")

def example_custom_fusion_strategy():
    """Example with custom fusion strategy"""
    print("Custom Fusion Strategy Example")
    print("=" * 40)
    
    try:
        from emotion_fusion import EmotionFusion
        
        # Create custom fusion with different strategy
        fusion = EmotionFusion("temporal_fusion")
        
        # Simulate some predictions
        predictions = [
            ("Happy", 0.8, "Sad", 0.6),
            ("Happy", 0.7, "Happy", 0.9),
            ("Angry", 0.6, "Happy", 0.8),
            ("Happy", 0.9, "Happy", 0.7),
            ("Happy", 0.8, "Neutral", 0.5)
        ]
        
        print("Testing temporal fusion strategy:")
        for i, (video_emotion, video_conf, audio_emotion, audio_conf) in enumerate(predictions):
            fused_emotion, fused_conf = fusion.fuse_emotions(
                video_emotion, video_conf, audio_emotion, audio_conf
            )
            print(f"Prediction {i+1}: Video={video_emotion}({video_conf:.2f}), "
                  f"Audio={audio_emotion}({audio_conf:.2f}) -> Fused={fused_emotion}({fused_conf:.2f})")
        
        # Get statistics
        stats = fusion.get_emotion_statistics()
        print(f"\nEmotion statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error in custom fusion strategy: {e}")

def main():
    """Main function to run examples"""
    print("Real-Time Emotion Detection System - Examples")
    print("=" * 60)
    print("Choose an example to run:")
    print("1. Basic Usage")
    print("2. Custom Configuration")
    print("3. Single Frame Processing")
    print("4. Batch Processing")
    print("5. Custom Fusion Strategy")
    print("6. Run All Examples")
    print("7. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                example_basic_usage()
            elif choice == "2":
                example_custom_configuration()
            elif choice == "3":
                example_single_frame_processing()
            elif choice == "4":
                example_batch_processing()
            elif choice == "5":
                example_custom_fusion_strategy()
            elif choice == "6":
                print("Running all examples...")
                example_single_frame_processing()
                example_batch_processing()
                example_custom_fusion_strategy()
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()




