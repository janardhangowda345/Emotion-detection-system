"""
Camera Diagnostic Script
Helps troubleshoot camera issues for the emotion detection system
"""

import cv2
import numpy as np
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_camera_access():
    """Test if camera can be accessed"""
    print("🔍 Testing camera access...")
    
    # Try different camera indices
    for camera_index in range(5):  # Try cameras 0-4
        print(f"\nTesting camera index {camera_index}...")
        
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                print(f"✅ Camera {camera_index} is accessible")
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera {camera_index} can capture frames")
                    print(f"   Frame shape: {frame.shape}")
                    
                    # Test camera properties
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    print(f"   Resolution: {int(width)}x{int(height)}")
                    print(f"   FPS: {fps}")
                    
                    cap.release()
                    return camera_index
                else:
                    print(f"❌ Camera {camera_index} cannot capture frames")
            else:
                print(f"❌ Camera {camera_index} is not accessible")
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Error with camera {camera_index}: {e}")
    
    return None

def test_opencv_installation():
    """Test OpenCV installation"""
    print("\n🔍 Testing OpenCV installation...")
    
    try:
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV functionality
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print("✅ OpenCV basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return False

def test_camera_with_display():
    """Test camera with live display"""
    print("\n🔍 Testing camera with live display...")
    
    working_camera = test_camera_access()
    
    if working_camera is None:
        print("❌ No working camera found")
        return False
    
    print(f"\n📹 Testing camera {working_camera} with live display...")
    print("Press 'q' to quit, 's' to save a test image")
    
    try:
        cap = cv2.VideoCapture(working_camera)
        
        if not cap.isOpened():
            print("❌ Could not open camera for display test")
            return False
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Camera Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("✅ Camera display test completed successfully")
                break
            elif key == ord('s'):
                cv2.imwrite("test_camera_image.jpg", frame)
                print("✅ Test image saved as 'test_camera_image.jpg'")
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ Camera display test failed: {e}")
        return False

def test_face_detection():
    """Test face detection functionality"""
    print("\n🔍 Testing face detection...")
    
    try:
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("❌ Could not load face detection classifier")
            return False
        
        print("✅ Face detection classifier loaded")
        
        # Test with camera
        working_camera = test_camera_access()
        if working_camera is None:
            print("❌ No camera available for face detection test")
            return False
        
        cap = cv2.VideoCapture(working_camera)
        
        if not cap.isOpened():
            print("❌ Could not open camera for face detection test")
            return False
        
        print("📹 Testing face detection with live camera...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add face count
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Face Detection Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Face detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def check_system_info():
    """Check system information"""
    print("\n🔍 System Information:")
    
    try:
        import platform
        print(f"Operating System: {platform.system()} {platform.release()}")
        print(f"Python Version: {sys.version}")
        print(f"OpenCV Version: {cv2.__version__}")
        
        # Check if running in virtual environment
        import sys
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Running in virtual environment")
        else:
            print("⚠️ Not running in virtual environment")
        
    except Exception as e:
        print(f"❌ Error getting system info: {e}")

def main():
    """Main diagnostic function"""
    print("=" * 60)
    print("🎥 Camera Diagnostic Tool")
    print("=" * 60)
    
    # Check system info
    check_system_info()
    
    # Test OpenCV
    if not test_opencv_installation():
        print("\n❌ OpenCV installation issue detected")
        print("Try: pip install opencv-python")
        return False
    
    # Test camera access
    working_camera = test_camera_access()
    if working_camera is None:
        print("\n❌ No working camera found")
        print("\nTroubleshooting tips:")
        print("1. Check if camera is connected and not being used by another application")
        print("2. Try different camera indices (0, 1, 2, etc.)")
        print("3. Check camera permissions")
        print("4. Restart your computer")
        return False
    
    # Test camera display
    if not test_camera_with_display():
        print("\n❌ Camera display test failed")
        return False
    
    # Test face detection
    if not test_face_detection():
        print("\n❌ Face detection test failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Your camera should work with the emotion detection system.")
    print(f"✅ Working camera index: {working_camera}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)




