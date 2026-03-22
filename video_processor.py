"""
Video Processing Module for Real-Time Emotion Detection
Handles video capture, face detection, and emotion prediction
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Tuple
from config import *

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Handles real-time video processing for emotion detection
    """
    
    def __init__(self, model=None, camera_index=0):
        """
        Initialize the video processor
        
        Args:
            model: Pre-trained video emotion model
            camera_index: Camera device index
        """
        self.model = model
        self.camera_index = camera_index
        self.cap = None
        self.face_detector = None
        self.is_capturing = False
        self.trackers = []
        self.last_faces = []
        
        # Initialize video capture and face detector
        self._setup_video_capture()
        self._setup_face_detector()
    
    def _setup_video_capture(self):
        """Setup video capture from camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
            
            logger.info(f"Video capture initialized (Camera {self.camera_index})")
            
        except Exception as e:
            logger.error(f"Failed to initialize video capture: {e}")
            self.cap = None
    
    def _setup_face_detector(self):
        """Setup face detection cascade classifier"""
        try:
            # Try to load custom haarcascade file first
            if os.path.exists(HAARCASCADE_PATH):
                self.face_detector = cv2.CascadeClassifier(HAARCASCADE_PATH)
            else:
                # Use built-in haarcascade
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            
            if self.face_detector.empty():
                raise Exception("Could not load face detection classifier")
            
            logger.info("Face detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            self.face_detector = None
    
    def start_capture(self):
        """Start video capture"""
        if self.cap and not self.is_capturing:
            self.is_capturing = True
            logger.info("Video capture started")
    
    def stop_capture(self):
        """Stop video capture"""
        if self.cap and self.is_capturing:
            self.is_capturing = False
            logger.info("Video capture stopped")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        try:
            if not self.face_detector:
                return []
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION_SCALE_FACTOR,
                minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
                minSize=(30, 30)
            )
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Preprocess face region for emotion prediction
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Preprocessed face image
        """
        try:
            # Convert to grayscale if needed
            if len(face_roi.shape) == 3:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input size
            face_roi = cv2.resize(face_roi, FACE_IMAGE_SIZE)
            
            # Normalize pixel values
            face_roi = face_roi.astype('float32') / 255.0
            
            # Add batch and channel dimensions
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            return face_roi
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return np.zeros((1, *FACE_IMAGE_SIZE, 1))

    def _create_tracker_instance(self):
        """
        Create a tracker instance according to TRACKER_TYPE (best-effort across OpenCV versions)
        """
        tracker_type = TRACKER_TYPE.upper() if 'TRACKER_TYPE' in globals() else 'KCF'
        try:
            if tracker_type == 'KCF':
                return cv2.TrackerKCF_create()
            if tracker_type == 'CSRT':
                return cv2.TrackerCSRT_create()
        except Exception:
            # Try legacy namespace
            try:
                if tracker_type == 'KCF':
                    return cv2.legacy.TrackerKCF_create()
                if tracker_type == 'CSRT':
                    return cv2.legacy.TrackerCSRT_create()
            except Exception:
                pass
        # Fallbacks
        try:
            return cv2.TrackerKCF_create()
        except Exception:
            try:
                return cv2.legacy.TrackerKCF_create()
            except Exception:
                return None

    def _create_trackers(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]):
        """Create cv2 trackers for the given faces and initialize them with the frame."""
        self.trackers = []
        for (x, y, w, h) in faces:
            tracker = self._create_tracker_instance()
            if tracker is None:
                continue
            try:
                tracker.init(frame, (int(x), int(y), int(w), int(h)))
                self.trackers.append(tracker)
            except Exception:
                # Some tracker APIs use init differently; ignore initialization failures
                continue
        self.last_faces = list(faces)

    def _update_trackers(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Update existing trackers and return bounding boxes."""
        faces = []
        new_trackers = []
        for tracker in self.trackers:
            try:
                ok, bbox = tracker.update(frame)
            except Exception:
                # Some tracker objects return (success, bbox) directly
                try:
                    ok, bbox = tracker.update(frame)
                except Exception:
                    ok = False
                    bbox = None
            if ok and bbox is not None:
                x, y, w, h = bbox
                faces.append((int(x), int(y), int(w), int(h)))
                new_trackers.append(tracker)
        # Keep only successfully updated trackers
        self.trackers = new_trackers
        if faces:
            self.last_faces = list(faces)
        return faces

    def get_tracked_faces(self, frame: np.ndarray, detect: bool = False) -> List[Tuple[int, int, int, int]]:
        """
        Return face bounding boxes using trackers between detections.
        If detect is True, run full detection and re-create trackers.
        """
        try:
            if detect or not self.trackers:
                faces = self.detect_faces(frame)
                # convert faces (numpy array) to list of tuples if needed
                if hasattr(faces, 'tolist'):
                    try:
                        faces = [tuple(map(int, f)) for f in faces]
                    except Exception:
                        faces = list(faces)
                faces = list(faces)
                if faces:
                    self._create_trackers(frame, faces)
                return faces
            else:
                faces = self._update_trackers(frame)
                # If trackers failed, fallback to detect
                if not faces:
                    faces = self.detect_faces(frame)
                    if faces:
                        self._create_trackers(frame, faces)
                return faces
        except Exception as e:
            logger.error(f"Error in get_tracked_faces: {e}")
            return []
    
    def get_face_emotion(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Get emotion prediction from face region
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Tuple of (emotion_label, confidence)
        """
        try:
            if not self.model:
                return "No Model", 0.0
            
            # Preprocess face
            processed_face = self.preprocess_face(face_roi)
            
            # Make prediction
            prediction = self.model.predict(processed_face, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            
            emotion_label = EMOTION_LABELS[emotion_idx]
            
            return emotion_label, confidence
            
        except Exception as e:
            logger.error(f"Error getting face emotion: {e}")
            return "Error", 0.0
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera
        
        Returns:
            Tuple of (success, frame)
        """
        try:
            if not self.cap:
                return False, None
            
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return False, None
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def draw_face_boxes(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                       emotions: List[Tuple[str, float]]) -> np.ndarray:
        """
        Draw face bounding boxes and emotion labels on frame
        
        Args:
            frame: Input video frame
            faces: List of face bounding boxes
            emotions: List of emotion predictions
            
        Returns:
            Frame with drawn annotations
        """
        try:
            for i, (x, y, w, h) in enumerate(faces):
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw emotion label
                if i < len(emotions):
                    emotion, confidence = emotions[i]
                    label = f"{emotion} ({confidence:.2f})"
                    
                    # Position label above face
                    label_y = max(y - 10, 20)
                    cv2.putText(
                        frame, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, DISPLAY_FONT,
                        DISPLAY_COLORS['video'], DISPLAY_THICKNESS
                    )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing face boxes: {e}")
            return frame
    
    def cleanup(self):
        """Clean up video resources"""
        self.stop_capture()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Video processor cleaned up")

