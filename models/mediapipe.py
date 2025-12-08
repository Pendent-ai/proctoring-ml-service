"""
MediaPipe Face Analysis
"""

import numpy as np
import mediapipe as mp
from dataclasses import dataclass

from config import settings


@dataclass
class FaceAnalysis:
    """Face analysis results."""
    face_detected: bool = False
    face_count: int = 0
    face_box: list | None = None
    landmarks: list | None = None
    
    # Gaze estimation
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    looking_away: bool = False
    
    # Head pose
    pitch: float = 0.0  # Up/down
    yaw: float = 0.0    # Left/right
    roll: float = 0.0   # Tilt
    
    # Eye tracking
    left_eye_open: float = 1.0
    right_eye_open: float = 1.0
    eyes_closed: bool = False


class MediaPipeAnalyzer:
    """MediaPipe wrapper for face and pose analysis."""
    
    # Key landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EAR = 234
    RIGHT_EAR = 454
    
    def __init__(self):
        """Initialize MediaPipe models."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        
        # Face mesh for detailed landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,  # Include iris landmarks
            min_detection_confidence=settings.face_confidence,
            min_tracking_confidence=0.5,
        )
        
        # Face detection for counting
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full range model
            min_detection_confidence=settings.face_confidence,
        )
        
        print("✅ MediaPipe initialized")
    
    def analyze(self, frame: np.ndarray) -> FaceAnalysis:
        """
        Analyze face in frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            FaceAnalysis with detection results.
        """
        result = FaceAnalysis()
        
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = frame
        else:
            return result
        
        h, w = frame.shape[:2]
        
        # Run face detection for counting
        detection_results = self.face_detection.process(rgb_frame)
        
        if detection_results.detections:
            result.face_count = len(detection_results.detections)
            result.face_detected = result.face_count > 0
            
            if result.face_detected:
                # Get first face box
                detection = detection_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                result.face_box = [
                    int(bbox.xmin * w),
                    int(bbox.ymin * h),
                    int((bbox.xmin + bbox.width) * w),
                    int((bbox.ymin + bbox.height) * h),
                ]
        
        # Run face mesh for detailed analysis
        mesh_results = self.face_mesh.process(rgb_frame)
        
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            
            # Store landmarks
            result.landmarks = [
                (lm.x, lm.y, lm.z) for lm in landmarks.landmark
            ]
            
            # Calculate gaze direction
            result.gaze_x, result.gaze_y = self._calculate_gaze(landmarks, w, h)
            result.looking_away = self._is_looking_away(result.gaze_x, result.gaze_y)
            
            # Calculate head pose
            result.pitch, result.yaw, result.roll = self._calculate_head_pose(landmarks, w, h)
            
            # Calculate eye openness
            result.left_eye_open = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
            result.right_eye_open = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
            result.eyes_closed = result.left_eye_open < 0.2 and result.right_eye_open < 0.2
        
        return result
    
    def _calculate_gaze(self, landmarks, w: int, h: int) -> tuple[float, float]:
        """Calculate gaze direction from iris landmarks."""
        try:
            # Get iris centers
            left_iris = np.mean([
                [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                for i in self.LEFT_IRIS
            ], axis=0)
            
            right_iris = np.mean([
                [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                for i in self.RIGHT_IRIS
            ], axis=0)
            
            # Get eye corners for reference
            left_eye_outer = np.array([
                landmarks.landmark[33].x * w,
                landmarks.landmark[33].y * h,
            ])
            left_eye_inner = np.array([
                landmarks.landmark[133].x * w,
                landmarks.landmark[133].y * h,
            ])
            
            right_eye_inner = np.array([
                landmarks.landmark[362].x * w,
                landmarks.landmark[362].y * h,
            ])
            right_eye_outer = np.array([
                landmarks.landmark[263].x * w,
                landmarks.landmark[263].y * h,
            ])
            
            # Calculate relative position within eye
            left_eye_width = np.linalg.norm(left_eye_inner - left_eye_outer)
            right_eye_width = np.linalg.norm(right_eye_inner - right_eye_outer)
            
            left_gaze_x = (left_iris[0] - left_eye_outer[0]) / left_eye_width - 0.5
            right_gaze_x = (right_iris[0] - right_eye_inner[0]) / right_eye_width - 0.5
            
            gaze_x = (left_gaze_x + right_gaze_x) / 2
            gaze_y = (left_iris[1] + right_iris[1]) / 2 / h - 0.5
            
            return float(gaze_x), float(gaze_y)
            
        except Exception:
            return 0.0, 0.0
    
    def _is_looking_away(self, gaze_x: float, gaze_y: float) -> bool:
        """Determine if person is looking away from screen."""
        # Thresholds for "looking away"
        return abs(gaze_x) > 0.3 or abs(gaze_y) > 0.25
    
    def _calculate_head_pose(self, landmarks, w: int, h: int) -> tuple[float, float, float]:
        """Estimate head pose from landmarks."""
        try:
            nose = np.array([
                landmarks.landmark[self.NOSE_TIP].x * w,
                landmarks.landmark[self.NOSE_TIP].y * h,
                landmarks.landmark[self.NOSE_TIP].z * w,
            ])
            
            chin = np.array([
                landmarks.landmark[self.CHIN].x * w,
                landmarks.landmark[self.CHIN].y * h,
                landmarks.landmark[self.CHIN].z * w,
            ])
            
            left_ear = np.array([
                landmarks.landmark[self.LEFT_EAR].x * w,
                landmarks.landmark[self.LEFT_EAR].y * h,
                landmarks.landmark[self.LEFT_EAR].z * w,
            ])
            
            right_ear = np.array([
                landmarks.landmark[self.RIGHT_EAR].x * w,
                landmarks.landmark[self.RIGHT_EAR].y * h,
                landmarks.landmark[self.RIGHT_EAR].z * w,
            ])
            
            # Yaw: difference between ear positions
            yaw = np.arctan2(right_ear[2] - left_ear[2], right_ear[0] - left_ear[0])
            
            # Pitch: nose to chin angle
            pitch = np.arctan2(chin[1] - nose[1], chin[2] - nose[2])
            
            # Roll: ear tilt
            roll = np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
            
            return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _eye_aspect_ratio(self, landmarks, eye_indices: list) -> float:
        """Calculate eye aspect ratio (EAR) for blink detection."""
        try:
            points = np.array([
                [landmarks.landmark[i].x, landmarks.landmark[i].y]
                for i in eye_indices
            ])
            
            # Vertical distances
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            
            # Horizontal distance
            h = np.linalg.norm(points[0] - points[3])
            
            ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
            return float(ear)
            
        except Exception:
            return 1.0
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()
        self.face_detection.close()
