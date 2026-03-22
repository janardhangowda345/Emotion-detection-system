"""
Simple Camera Test
Basic test to check if camera is working
"""

import cv2
import sys

def test_camera():
    """Simple camera test"""
    print("Testing camera...")
    
    # Try camera index 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        print("Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("❌ No camera found")
            return False
    
    print("✅ Camera found!")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Add text to frame
        cv2.putText(frame, "Camera Test - Press 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Camera Test", frame)
        
        # Check for 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed")
    return True

if __name__ == "__main__":
    test_camera()




