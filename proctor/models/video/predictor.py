from __future__ import annotations
"""
Video Predictor

Handles video frame inference with YOLO and MediaPipe.
"""

import asyncio
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import numpy as np

from proctor.engine.predictor import BasePredictor
from proctor.engine.results import AnalysisResults, DetectionResults, FaceResults
from proctor.cfg import VideoConfig

# Lazy imports
YOLO = None
mp = None


def _import_yolo():
    """Lazy import YOLO."""
    global YOLO
    if YOLO is None:
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO
    return YOLO


def _import_mediapipe():
    """Lazy import MediaPipe."""
    global mp
    if mp is None:
        import mediapipe as _mp
        mp = _mp
    return mp


# COCO class IDs
PERSON_CLASS = 0
PHONE_CLASS = 67
LAPTOP_CLASS = 63
BOOK_CLASS = 73
TV_CLASS = 62

TARGET_CLASSES = {
    PERSON_CLASS: "person",
    PHONE_CLASS: "phone",
    LAPTOP_CLASS: "laptop",
    BOOK_CLASS: "book",
    TV_CLASS: "tv",
}


class VideoPredictor(BasePredictor):
    """
    Video frame predictor combining YOLO and MediaPipe.
    
    Performs:
    - Object detection (phone, person, laptop, book)
    - Face detection and mesh
    - Gaze estimation
    - Head pose estimation
    """
    
    # MediaPipe landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EAR = 234
    RIGHT_EAR = 454
    
    def __init__(self, cfg: VideoConfig):
        """
        Initialize video predictor.
        
        Args:
            cfg: Video configuration
        """
        super().__init__(cfg)
        
        self.yolo = None
        self.face_mesh = None
        self.face_detection = None
        
        # Feature extractor for classifier
        self._feature_history = []
        self._classifier = None
    
    def setup_model(self):
        """Load YOLO and MediaPipe models."""
        import torch
        
        # Load YOLO
        YOLO = _import_yolo()
        
        model_path = self.cfg.yolo_model_path
        
        if not Path(model_path).exists():
            print("📥 Downloading YOLO11 model...")
            model_path = "yolo11n.pt"
        
        self.yolo = YOLO(model_path)
        
        # Detect best available device
        if self.cfg.use_gpu:
            if torch.backends.mps.is_available():
                # Apple Silicon (M1/M2/M3/M4)
                self.yolo.to("mps")
                self.device = "mps"
                print("🍎 Using Apple MPS (Metal Performance Shaders)")
            elif torch.cuda.is_available():
                # NVIDIA GPU
                self.yolo.to("cuda")
                self.device = "cuda"
                print("🎮 Using NVIDIA CUDA")
            else:
                # CPU fallback
                self.device = "cpu"
                print("💻 Using CPU (no GPU available)")
        else:
            self.device = "cpu"
            print("💻 Using CPU (GPU disabled in config)")
        
        print(f"✅ YOLO11 loaded: {model_path} on {self.device}")
        
        # Load MediaPipe
        mp = _import_mediapipe()
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=self.cfg.face_confidence,
            min_tracking_confidence=0.5,
        )
        
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.cfg.face_confidence,
        )
        
        print("✅ MediaPipe initialized")
        
        # Load classifier
        self._load_classifier()
    
    def _load_classifier(self):
        """Load cheating classifier."""
        try:
            import xgboost as xgb
            
            path = Path(self.cfg.classifier_model_path)
            if path.exists():
                self._classifier = xgb.XGBClassifier()
                self._classifier.load_model(str(path))
                print(f"✅ Classifier loaded: {path}")
        except ImportError:
            print("⚠️ XGBoost not installed. Using rule-based detection.")
        except Exception as e:
            print(f"⚠️ Could not load classifier: {e}")
    
    def preprocess(self, source: Any) -> np.ndarray:
        """
        Preprocess input frame.
        
        Args:
            source: Image frame (numpy array or path)
            
        Returns:
            Preprocessed frame
        """
        if isinstance(source, (str, Path)):
            import cv2
            frame = cv2.imread(str(source))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif isinstance(source, np.ndarray):
            frame = source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        # Resize if needed
        h, w = frame.shape[:2]
        if w != self.cfg.frame_width or h != self.cfg.frame_height:
            import cv2
            frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))
        
        return frame
    
    def inference(self, frame: np.ndarray) -> dict:
        """
        Run inference on frame.
        
        Args:
            frame: Preprocessed frame
            
        Returns:
            Raw inference results
        """
        # Run YOLO
        yolo_results = self.yolo(
            frame,
            conf=self.cfg.yolo_confidence,
            classes=list(TARGET_CLASSES.keys()),
            verbose=False,
        )
        
        # Run MediaPipe
        h, w = frame.shape[:2]
        detection_results = self.face_detection.process(frame)
        mesh_results = self.face_mesh.process(frame)
        
        return {
            "yolo": yolo_results,
            "face_detection": detection_results,
            "face_mesh": mesh_results,
            "frame_size": (w, h),
        }
    
    def postprocess(self, preds: dict, source: Any) -> AnalysisResults:
        """
        Postprocess inference results.
        
        Args:
            preds: Raw inference results
            source: Original input
            
        Returns:
            AnalysisResults with processed predictions
        """
        # Parse YOLO results
        detection = self._parse_yolo(preds["yolo"])
        
        # Parse face results
        face = self._parse_face(
            preds["face_detection"],
            preds["face_mesh"],
            preds["frame_size"],
        )
        
        # Run classifier
        cheating_prob, is_cheating, factors = self._classify(detection, face)
        
        # Determine if alert needed
        should_alert, alert_type, severity = self._check_alert(
            detection, face, cheating_prob
        )
        
        return AnalysisResults(
            timestamp=datetime.utcnow(),
            source=source,
            detection=detection,
            face=face,
            cheating_probability=cheating_prob,
            is_cheating=is_cheating,
            top_factors=factors,
            should_alert=should_alert,
            alert_type=alert_type,
            alert_severity=severity,
        )
    
    def _parse_yolo(self, results) -> DetectionResults:
        """Parse YOLO detection results."""
        detection = DetectionResults()
        
        if not results or len(results) == 0:
            return detection
        
        result = results[0]
        if result.boxes is None:
            return detection
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].cpu().numpy()
            
            box_data = {
                "class": TARGET_CLASSES.get(cls, "unknown"),
                "confidence": conf,
                "box": xyxy.tolist(),
            }
            
            detection.boxes.append(box_data)
            
            if cls == PERSON_CLASS:
                detection.person_count += 1
            elif cls == PHONE_CLASS:
                detection.phone_detected = True
                detection.phone_boxes.append(box_data)
            elif cls == LAPTOP_CLASS:
                detection.laptop_detected = True
            elif cls == BOOK_CLASS:
                detection.book_detected = True
        
        return detection
    
    def _parse_face(
        self,
        detection_results,
        mesh_results,
        frame_size: tuple,
    ) -> FaceResults:
        """Parse MediaPipe face results."""
        face = FaceResults()
        w, h = frame_size
        
        # Face detection
        if detection_results.detections:
            face.face_count = len(detection_results.detections)
            face.face_detected = face.face_count > 0
            
            if face.face_detected:
                det = detection_results.detections[0]
                bbox = det.location_data.relative_bounding_box
                face.face_box = [
                    int(bbox.xmin * w),
                    int(bbox.ymin * h),
                    int((bbox.xmin + bbox.width) * w),
                    int((bbox.ymin + bbox.height) * h),
                ]
        
        # Face mesh
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            
            face.landmarks = [
                (lm.x, lm.y, lm.z) for lm in landmarks.landmark
            ]
            
            # Gaze
            face.gaze_x, face.gaze_y = self._calculate_gaze(landmarks, w, h)
            face.looking_away = self._is_looking_away(face.gaze_x, face.gaze_y)
            
            # Head pose
            face.pitch, face.yaw, face.roll = self._calculate_head_pose(landmarks, w, h)
            
            # Eye openness
            face.left_eye_open = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
            face.right_eye_open = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
            face.eyes_closed = face.left_eye_open < 0.2 and face.right_eye_open < 0.2
        
        return face
    
    def _calculate_gaze(self, landmarks, w: int, h: int) -> tuple[float, float]:
        """Calculate gaze direction from iris landmarks."""
        try:
            left_iris = np.mean([
                [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                for i in self.LEFT_IRIS
            ], axis=0)
            
            right_iris = np.mean([
                [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                for i in self.RIGHT_IRIS
            ], axis=0)
            
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
        return (
            abs(gaze_x) > self.cfg.gaze_away_threshold_x or
            abs(gaze_y) > self.cfg.gaze_away_threshold_y
        )
    
    def _calculate_head_pose(
        self,
        landmarks,
        w: int,
        h: int,
    ) -> tuple[float, float, float]:
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
            
            yaw = np.arctan2(right_ear[2] - left_ear[2], right_ear[0] - left_ear[0])
            pitch = np.arctan2(chin[1] - nose[1], chin[2] - nose[2])
            roll = np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
            
            return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _eye_aspect_ratio(self, landmarks, eye_indices: list) -> float:
        """Calculate eye aspect ratio for blink detection."""
        try:
            points = np.array([
                [landmarks.landmark[i].x, landmarks.landmark[i].y]
                for i in eye_indices
            ])
            
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            h = np.linalg.norm(points[0] - points[3])
            
            ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
            return float(ear)
        except Exception:
            return 1.0
    
    def _classify(
        self,
        detection: DetectionResults,
        face: FaceResults,
    ) -> tuple[float, bool, list]:
        """Run cheating classifier."""
        factors = []
        
        # Rule-based classification
        score = 0.0
        
        if detection.phone_detected:
            score += 0.4
            factors.append("phone_use")
        
        if detection.person_count > 1:
            score += 0.3
            factors.append("multiple_faces")
        
        if face.looking_away:
            score += 0.2
            factors.append("looking_away")
        
        if not face.face_detected:
            score += 0.1
            factors.append("face_not_visible")
        
        return min(1.0, score), score >= self.cfg.cheating_threshold, factors[:3]
    
    def _check_alert(
        self,
        detection: DetectionResults,
        face: FaceResults,
        cheating_prob: float,
    ) -> tuple[bool, Optional[str], str]:
        """Check if alert should be triggered."""
        if detection.phone_detected:
            return True, "phone_detected", "high"
        
        if detection.person_count > 1:
            return True, "multiple_faces", "high"
        
        if not face.face_detected:
            return True, "face_not_visible", "medium"
        
        if face.looking_away:
            return True, "looking_away", "low"
        
        return False, None, "low"
    
    def detect_objects(self, frame: np.ndarray) -> DetectionResults:
        """Run only object detection."""
        frame = self.preprocess(frame)
        results = self.yolo(
            frame,
            conf=self.cfg.yolo_confidence,
            classes=list(TARGET_CLASSES.keys()),
            verbose=False,
        )
        return self._parse_yolo(results)
    
    def analyze_face(self, frame: np.ndarray) -> FaceResults:
        """Run only face analysis."""
        frame = self.preprocess(frame)
        h, w = frame.shape[:2]
        
        detection_results = self.face_detection.process(frame)
        mesh_results = self.face_mesh.process(frame)
        
        return self._parse_face(detection_results, mesh_results, (w, h))
    
    def export(self, format: str = "onnx", **kwargs) -> Path:
        """Export YOLO model."""
        return self.yolo.export(format=format, **kwargs)
    
    def close(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()
        if self.face_detection:
            self.face_detection.close()
