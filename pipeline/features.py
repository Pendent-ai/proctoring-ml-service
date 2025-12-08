"""
Feature Extraction from Frame History
"""

import numpy as np
from typing import Any


class FeatureExtractor:
    """Extract features from sliding window of frame data."""
    
    def __init__(self, window_size: int = 60):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Number of frames in sliding window
        """
        self.window_size = window_size
    
    def extract(self, history: list[dict]) -> dict:
        """
        Extract features from frame history.
        
        Args:
            history: List of frame data dictionaries
            
        Returns:
            Dictionary of features for classifier.
        """
        if not history:
            return self._empty_features()
        
        features = {}
        
        # Gaze features
        gaze_x = [h["face"].gaze_x for h in history if h["face"].face_detected]
        gaze_y = [h["face"].gaze_y for h in history if h["face"].face_detected]
        
        features["gaze_x_mean"] = np.mean(gaze_x) if gaze_x else 0.0
        features["gaze_y_mean"] = np.mean(gaze_y) if gaze_y else 0.0
        features["gaze_variance"] = np.var(gaze_x) + np.var(gaze_y) if gaze_x else 0.0
        
        looking_away = [h["face"].looking_away for h in history if h["face"].face_detected]
        features["gaze_away_ratio"] = np.mean(looking_away) if looking_away else 0.0
        
        # Head pose features
        yaw = [h["face"].yaw for h in history if h["face"].face_detected]
        pitch = [h["face"].pitch for h in history if h["face"].face_detected]
        
        features["head_yaw_mean"] = np.mean(yaw) if yaw else 0.0
        features["head_yaw_variance"] = np.var(yaw) if yaw else 0.0
        features["head_pitch_mean"] = np.mean(pitch) if pitch else 0.0
        
        # Head movement frequency (changes in yaw)
        if len(yaw) > 1:
            yaw_diff = np.abs(np.diff(yaw))
            features["head_movement_freq"] = np.sum(yaw_diff > 5) / len(yaw_diff)
        else:
            features["head_movement_freq"] = 0.0
        
        # Face visibility
        face_visible = [h["face"].face_detected for h in history]
        features["face_visible_ratio"] = np.mean(face_visible) if face_visible else 0.0
        
        # Face count
        face_counts = [h["face"].face_count for h in history]
        features["face_count_max"] = max(face_counts) if face_counts else 0
        
        # Multiple faces ratio
        features["multiple_faces_ratio"] = np.mean([c > 1 for c in face_counts]) if face_counts else 0.0
        
        # Phone detection
        phone_detected = [h["yolo"]["phone_detected"] for h in history]
        features["phone_detected_ratio"] = np.mean(phone_detected) if phone_detected else 0.0
        
        # Phone duration (consecutive frames)
        features["phone_duration"] = self._max_consecutive(phone_detected)
        
        # Eyes closed
        eyes_closed = [h["face"].eyes_closed for h in history if h["face"].face_detected]
        features["eyes_closed_ratio"] = np.mean(eyes_closed) if eyes_closed else 0.0
        
        # Looking away duration
        features["looking_away_duration"] = self._max_consecutive(looking_away) if looking_away else 0
        
        return features
    
    def _empty_features(self) -> dict:
        """Return empty feature dictionary."""
        return {
            "gaze_x_mean": 0.0,
            "gaze_y_mean": 0.0,
            "gaze_variance": 0.0,
            "gaze_away_ratio": 0.0,
            "head_yaw_mean": 0.0,
            "head_yaw_variance": 0.0,
            "head_pitch_mean": 0.0,
            "head_movement_freq": 0.0,
            "face_visible_ratio": 1.0,
            "face_count_max": 1,
            "multiple_faces_ratio": 0.0,
            "phone_detected_ratio": 0.0,
            "phone_duration": 0,
            "eyes_closed_ratio": 0.0,
            "looking_away_duration": 0,
        }
    
    def _max_consecutive(self, arr: list[bool]) -> int:
        """Find maximum consecutive True values."""
        if not arr:
            return 0
        
        max_count = 0
        current_count = 0
        
        for val in arr:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
