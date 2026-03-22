"""
Test Script for Real-Time Emotion Detection System
Verifies that all components work correctly
"""

import os
import sys
import logging
import numpy as np
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import cv2
        import numpy as np
        import librosa
        import sounddevice as sd
        from tensorflow.keras.models import Sequential
        from sklearn.model_selection import train_test_split
        
        print("✓ All required modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        print("Testing configuration...")
        
        # Test config values
        assert len(EMOTION_LABELS) > 0, "Emotion labels not defined"
        assert FPS_TARGET > 0, "Invalid FPS target"
        assert AUDIO_SAMPLE_RATE > 0, "Invalid audio sample rate"
        assert VIDEO_WIDTH > 0, "Invalid video width"
        assert VIDEO_HEIGHT > 0, "Invalid video height"
        
        print("✓ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_video_processor():
    """Test video processor"""
    try:
        print("Testing video processor...")
        
        from video_processor import VideoProcessor
        
        # Create processor
        processor = VideoProcessor()
        
        # Test frame reading
        success, frame = processor.read_frame()
        if success:
            print("✓ Video capture working")
        else:
            print("⚠ Video capture not available (camera may be in use)")
        
        # Test face detection
        if success:
            faces = processor.detect_faces(frame)
            print(f"✓ Face detection working (found {len(faces)} faces)")
        
        processor.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Video processor error: {e}")
        return False

def test_audio_processor():
    """Test audio processor"""
    try:
        print("Testing audio processor...")
        
        from audio_processor import AudioProcessor
        
        # Create processor
        processor = AudioProcessor()
        
        # Test audio stream
        processor.start_recording()
        print("✓ Audio capture started")
        processor.stop_recording()
        
        # Test feature extraction
        dummy_audio = np.random.randn(16000)
        features = processor.extract_mfcc_features(dummy_audio)
        assert len(features) == MFCC_N_COEFFS, "MFCC features length mismatch"
        print("✓ Audio feature extraction working")
        
        processor.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Audio processor error: {e}")
        return False

def test_emotion_fusion():
    """Test emotion fusion"""
    try:
        print("Testing emotion fusion...")
        
        from emotion_fusion import EmotionFusion
        
        # Create fusion module
        fusion = EmotionFusion("weighted_average")
        
        # Test fusion
        emotion, confidence = fusion.fuse_emotions(
            "Happy", 0.8, "Sad", 0.6
        )
        
        assert emotion in EMOTION_LABELS or emotion in ["Happy", "Sad"], "Invalid fused emotion"
        assert 0 <= confidence <= 1, "Invalid confidence score"
        
        print("✓ Emotion fusion working")
        return True
        
    except Exception as e:
        print(f"✗ Emotion fusion error: {e}")
        return False

def test_model_loader():
    """Test model loader"""
    try:
        print("Testing model loader...")
        
        from model_loader import ModelLoader
        
        # Create loader
        loader = ModelLoader()
        
        # Test model loading
        video_loaded, audio_loaded = loader.load_all_models()
        
        if video_loaded or audio_loaded:
            print("✓ Models loaded successfully")
        else:
            print("⚠ No models found, will use dummy models")
        
        # Test model info
        info = loader.get_model_info()
        assert 'video_model_loaded' in info, "Model info incomplete"
        assert 'audio_model_loaded' in info, "Model info incomplete"
        
        print("✓ Model loader working")
        loader.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Model loader error: {e}")
        return False

def test_main_system():
    """Test main emotion detection system"""
    try:
        print("Testing main system...")
        
        from real_time_emotion_detection import RealTimeEmotionDetection
        
        # Create system
        detector = RealTimeEmotionDetection()
        
        # Test initialization
        assert detector.video_processor is not None, "Video processor not initialized"
        assert detector.audio_processor is not None, "Audio processor not initialized"
        assert detector.fusion is not None, "Fusion module not initialized"
        
        print("✓ Main system initialized successfully")
        
        # Test cleanup
        detector.cleanup()
        print("✓ System cleanup working")
        
        return True
        
    except Exception as e:
        print(f"✗ Main system error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Real-Time Emotion Detection System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Video Processor Test", test_video_processor),
        ("Audio Processor Test", test_audio_processor),
        ("Emotion Fusion Test", test_emotion_fusion),
        ("Model Loader Test", test_model_loader),
        ("Main System Test", test_main_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)




