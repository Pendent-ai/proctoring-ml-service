"""
Enhanced Feature Engineering

Advanced features for cheating detection including:
- Temporal gaze patterns
- Head movement analysis
- Behavioral patterns
- Audio-video correlation
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import numpy as np


@dataclass
class FrameFeatures:
    """Features extracted from a single frame."""
    timestamp: float = 0.0
    
    # Gaze
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    looking_away: bool = False
    
    # Head pose
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    
    # Face
    face_detected: bool = False
    face_count: int = 0
    left_eye_open: float = 1.0
    right_eye_open: float = 1.0
    
    # Detections
    phone_detected: bool = False
    phone_confidence: float = 0.0
    person_count: int = 0
    
    # Audio (if available)
    voice_detected: bool = False
    speaker_count: int = 1


# 25+ Enhanced Features
ENHANCED_FEATURES = [
    # === Gaze Features (8) ===
    "gaze_x_mean",
    "gaze_y_mean", 
    "gaze_variance",
    "gaze_away_ratio",
    "gaze_oscillation_freq",      # NEW: Rapid left-right (reading notes)
    "gaze_return_pattern",        # NEW: Looking away then back
    "gaze_fixation_duration",     # NEW: Time on single point
    "gaze_path_length",           # NEW: Total gaze movement distance
    
    # === Head Features (8) ===
    "head_yaw_mean",
    "head_yaw_variance",
    "head_pitch_mean",
    "head_pitch_variance",
    "head_movement_freq",
    "head_turn_count",            # NEW: Number of head turns
    "head_jerk_count",            # NEW: Sudden movements
    "head_tilt_angle",            # NEW: Angle toward potential 2nd screen
    
    # === Face Features (6) ===
    "face_visible_ratio",
    "face_count_max",
    "multiple_faces_ratio",
    "blink_rate",                 # NEW: Blinks per minute
    "eyes_closed_ratio",
    "eye_aspect_ratio_variance",  # NEW: Irregular blinking
    
    # === Detection Features (4) ===
    "phone_detected_ratio",
    "phone_duration",
    "phone_confidence_max",       # NEW: Peak phone confidence
    "person_count_variance",      # NEW: People appearing/disappearing
    
    # === Temporal Patterns (6) ===
    "looking_away_duration",
    "suspicious_sequence_count",  # NEW: Repeated suspicious patterns
    "behavior_entropy",           # NEW: Randomness of behavior
    "attention_consistency",      # NEW: How consistent is attention
    "activity_level",             # NEW: Overall movement level
    "stillness_ratio",            # NEW: Unusually still (reading)
    
    # === Audio Features (5) ===
    "voice_activity_ratio",       # NEW: Speaking vs silent
    "speaker_change_count",       # NEW: Different voices
    "audio_visual_sync",          # NEW: Lip sync score
    "background_noise_level",     # NEW: Environmental noise
    "whisper_ratio",              # NEW: Whispering detected
    
    # === Context Features (3) ===
    "time_in_session_pct",        # NEW: Position in interview
    "alert_history_count",        # NEW: Previous alerts
    "cumulative_suspicion",       # NEW: Running suspicion score
]


class EnhancedFeatureExtractor:
    """
    Extract enhanced features from frame history.
    
    Computes 40+ features from sliding window of frames.
    """
    
    def __init__(self, window_size: int = 60, fps: int = 10):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Number of frames in sliding window
            fps: Frames per second
        """
        self.window_size = window_size
        self.fps = fps
        self.window_seconds = window_size / fps
        
        # State
        self.frame_history: deque[FrameFeatures] = deque(maxlen=window_size)
        self.alert_count = 0
        self.session_start_time: Optional[float] = None
        self.cumulative_suspicion = 0.0
    
    def add_frame(self, features: FrameFeatures):
        """Add frame features to history."""
        if self.session_start_time is None:
            self.session_start_time = features.timestamp
        self.frame_history.append(features)
    
    def extract(self) -> dict:
        """
        Extract all features from current window.
        
        Returns:
            Dictionary of feature name -> value
        """
        if len(self.frame_history) < 5:
            return {f: 0.0 for f in ENHANCED_FEATURES}
        
        frames = list(self.frame_history)
        
        features = {}
        
        # === Gaze Features ===
        features.update(self._extract_gaze_features(frames))
        
        # === Head Features ===
        features.update(self._extract_head_features(frames))
        
        # === Face Features ===
        features.update(self._extract_face_features(frames))
        
        # === Detection Features ===
        features.update(self._extract_detection_features(frames))
        
        # === Temporal Patterns ===
        features.update(self._extract_temporal_features(frames))
        
        # === Audio Features ===
        features.update(self._extract_audio_features(frames))
        
        # === Context Features ===
        features.update(self._extract_context_features(frames))
        
        return features
    
    def _extract_gaze_features(self, frames: list[FrameFeatures]) -> dict:
        """Extract gaze-related features."""
        gaze_x = [f.gaze_x for f in frames if f.face_detected]
        gaze_y = [f.gaze_y for f in frames if f.face_detected]
        looking_away = [f.looking_away for f in frames]
        
        if not gaze_x:
            return {
                "gaze_x_mean": 0.0,
                "gaze_y_mean": 0.0,
                "gaze_variance": 0.0,
                "gaze_away_ratio": 0.0,
                "gaze_oscillation_freq": 0.0,
                "gaze_return_pattern": 0.0,
                "gaze_fixation_duration": 0.0,
                "gaze_path_length": 0.0,
            }
        
        gaze_x = np.array(gaze_x)
        gaze_y = np.array(gaze_y)
        
        # Basic stats
        gaze_x_mean = float(np.mean(gaze_x))
        gaze_y_mean = float(np.mean(gaze_y))
        gaze_variance = float(np.var(gaze_x) + np.var(gaze_y))
        gaze_away_ratio = sum(looking_away) / len(looking_away)
        
        # Oscillation frequency (rapid left-right movement)
        gaze_diff = np.diff(gaze_x)
        sign_changes = np.sum(np.diff(np.sign(gaze_diff)) != 0)
        gaze_oscillation_freq = sign_changes / self.window_seconds
        
        # Return pattern (look away then back)
        away_periods = self._find_periods(looking_away, True)
        return_count = sum(1 for start, end in away_periods if end - start < self.fps)  # Short away periods
        gaze_return_pattern = return_count / max(1, len(away_periods))
        
        # Fixation duration (time spent looking at one point)
        stable_threshold = 0.05
        stable_count = np.sum(np.abs(gaze_diff) < stable_threshold) if len(gaze_diff) > 0 else 0
        gaze_fixation_duration = stable_count / self.fps
        
        # Path length (total movement)
        if len(gaze_x) > 1:
            dx = np.diff(gaze_x)
            dy = np.diff(gaze_y)
            gaze_path_length = float(np.sum(np.sqrt(dx**2 + dy**2)))
        else:
            gaze_path_length = 0.0
        
        return {
            "gaze_x_mean": gaze_x_mean,
            "gaze_y_mean": gaze_y_mean,
            "gaze_variance": gaze_variance,
            "gaze_away_ratio": gaze_away_ratio,
            "gaze_oscillation_freq": gaze_oscillation_freq,
            "gaze_return_pattern": gaze_return_pattern,
            "gaze_fixation_duration": gaze_fixation_duration,
            "gaze_path_length": gaze_path_length,
        }
    
    def _extract_head_features(self, frames: list[FrameFeatures]) -> dict:
        """Extract head movement features."""
        yaw = np.array([f.head_yaw for f in frames if f.face_detected])
        pitch = np.array([f.head_pitch for f in frames if f.face_detected])
        
        if len(yaw) < 2:
            return {
                "head_yaw_mean": 0.0,
                "head_yaw_variance": 0.0,
                "head_pitch_mean": 0.0,
                "head_pitch_variance": 0.0,
                "head_movement_freq": 0.0,
                "head_turn_count": 0.0,
                "head_jerk_count": 0.0,
                "head_tilt_angle": 0.0,
            }
        
        # Basic stats
        head_yaw_mean = float(np.mean(yaw))
        head_yaw_variance = float(np.var(yaw))
        head_pitch_mean = float(np.mean(pitch))
        head_pitch_variance = float(np.var(pitch))
        
        # Movement frequency
        yaw_diff = np.abs(np.diff(yaw))
        significant_moves = np.sum(yaw_diff > 5)  # > 5 degrees
        head_movement_freq = significant_moves / self.window_seconds
        
        # Head turns (large rotations)
        head_turn_threshold = 20  # degrees
        head_turn_count = float(np.sum(yaw_diff > head_turn_threshold))
        
        # Jerks (sudden movements)
        jerk_threshold = 15  # degrees per frame
        head_jerk_count = float(np.sum(yaw_diff > jerk_threshold))
        
        # Tilt angle (potential second screen)
        roll = np.array([f.head_roll for f in frames if f.face_detected])
        head_tilt_angle = float(np.mean(np.abs(roll))) if len(roll) > 0 else 0.0
        
        return {
            "head_yaw_mean": head_yaw_mean,
            "head_yaw_variance": head_yaw_variance,
            "head_pitch_mean": head_pitch_mean,
            "head_pitch_variance": head_pitch_variance,
            "head_movement_freq": head_movement_freq,
            "head_turn_count": head_turn_count,
            "head_jerk_count": head_jerk_count,
            "head_tilt_angle": head_tilt_angle,
        }
    
    def _extract_face_features(self, frames: list[FrameFeatures]) -> dict:
        """Extract face-related features."""
        face_detected = [f.face_detected for f in frames]
        face_counts = [f.face_count for f in frames]
        left_eye = [f.left_eye_open for f in frames if f.face_detected]
        right_eye = [f.right_eye_open for f in frames if f.face_detected]
        
        # Face visibility
        face_visible_ratio = sum(face_detected) / len(face_detected)
        face_count_max = max(face_counts) if face_counts else 0
        multiple_faces_ratio = sum(1 for c in face_counts if c > 1) / len(face_counts)
        
        # Eye features
        if left_eye and right_eye:
            ear = [(l + r) / 2 for l, r in zip(left_eye, right_eye)]
            
            # Blink detection (EAR drops below 0.2)
            blinks = sum(1 for e in ear if e < 0.2)
            blink_rate = blinks * (60 / self.window_seconds)  # Per minute
            
            eyes_closed_ratio = sum(1 for e in ear if e < 0.15) / len(ear)
            eye_aspect_ratio_variance = float(np.var(ear))
        else:
            blink_rate = 0.0
            eyes_closed_ratio = 0.0
            eye_aspect_ratio_variance = 0.0
        
        return {
            "face_visible_ratio": face_visible_ratio,
            "face_count_max": float(face_count_max),
            "multiple_faces_ratio": multiple_faces_ratio,
            "blink_rate": blink_rate,
            "eyes_closed_ratio": eyes_closed_ratio,
            "eye_aspect_ratio_variance": eye_aspect_ratio_variance,
        }
    
    def _extract_detection_features(self, frames: list[FrameFeatures]) -> dict:
        """Extract object detection features."""
        phone_detected = [f.phone_detected for f in frames]
        phone_confidence = [f.phone_confidence for f in frames if f.phone_detected]
        person_counts = [f.person_count for f in frames]
        
        phone_detected_ratio = sum(phone_detected) / len(phone_detected)
        
        # Phone duration (consecutive frames)
        phone_periods = self._find_periods(phone_detected, True)
        phone_duration = max((end - start for start, end in phone_periods), default=0) / self.fps
        
        phone_confidence_max = max(phone_confidence) if phone_confidence else 0.0
        
        person_count_variance = float(np.var(person_counts))
        
        return {
            "phone_detected_ratio": phone_detected_ratio,
            "phone_duration": phone_duration,
            "phone_confidence_max": phone_confidence_max,
            "person_count_variance": person_count_variance,
        }
    
    def _extract_temporal_features(self, frames: list[FrameFeatures]) -> dict:
        """Extract temporal pattern features."""
        looking_away = [f.looking_away for f in frames]
        
        # Looking away duration
        away_periods = self._find_periods(looking_away, True)
        looking_away_duration = max((end - start for start, end in away_periods), default=0) / self.fps
        
        # Suspicious sequence count (patterns like: look away -> phone -> look back)
        suspicious_count = 0
        for i in range(len(frames) - 10):
            window = frames[i:i+10]
            if (any(f.looking_away for f in window[:3]) and
                any(f.phone_detected for f in window[3:7]) and
                not any(f.looking_away for f in window[7:])):
                suspicious_count += 1
        
        # Behavior entropy (randomness)
        behavior_sequence = [
            int(f.looking_away) + int(f.phone_detected) * 2 + int(f.face_count > 1) * 4
            for f in frames
        ]
        behavior_entropy = self._calculate_entropy(behavior_sequence)
        
        # Attention consistency
        gaze_x = [f.gaze_x for f in frames if f.face_detected]
        if len(gaze_x) > 1:
            attention_consistency = 1.0 - min(1.0, np.std(gaze_x))
        else:
            attention_consistency = 0.0
        
        # Activity level (overall movement)
        movements = []
        for i in range(1, len(frames)):
            if frames[i].face_detected and frames[i-1].face_detected:
                dx = abs(frames[i].gaze_x - frames[i-1].gaze_x)
                dy = abs(frames[i].gaze_y - frames[i-1].gaze_y)
                dh = abs(frames[i].head_yaw - frames[i-1].head_yaw)
                movements.append(dx + dy + dh/100)
        activity_level = np.mean(movements) if movements else 0.0
        
        # Stillness ratio (unusually still = reading notes)
        stillness_threshold = 0.02
        stillness_count = sum(1 for m in movements if m < stillness_threshold)
        stillness_ratio = stillness_count / len(movements) if movements else 0.0
        
        return {
            "looking_away_duration": looking_away_duration,
            "suspicious_sequence_count": float(suspicious_count),
            "behavior_entropy": behavior_entropy,
            "attention_consistency": attention_consistency,
            "activity_level": float(activity_level),
            "stillness_ratio": stillness_ratio,
        }
    
    def _extract_audio_features(self, frames: list[FrameFeatures]) -> dict:
        """Extract audio-related features."""
        voice_detected = [f.voice_detected for f in frames]
        speaker_counts = [f.speaker_count for f in frames]
        
        voice_activity_ratio = sum(voice_detected) / len(voice_detected)
        
        # Speaker changes
        speaker_changes = sum(
            1 for i in range(1, len(speaker_counts))
            if speaker_counts[i] != speaker_counts[i-1]
        )
        
        return {
            "voice_activity_ratio": voice_activity_ratio,
            "speaker_change_count": float(speaker_changes),
            "audio_visual_sync": 0.5,  # TODO: Implement lip sync
            "background_noise_level": 0.0,  # TODO: From audio pipeline
            "whisper_ratio": 0.0,  # TODO: From audio pipeline
        }
    
    def _extract_context_features(self, frames: list[FrameFeatures]) -> dict:
        """Extract context features."""
        if frames and self.session_start_time:
            elapsed = frames[-1].timestamp - self.session_start_time
            time_in_session_pct = min(1.0, elapsed / 3600)  # Assume 1 hour max
        else:
            time_in_session_pct = 0.0
        
        return {
            "time_in_session_pct": time_in_session_pct,
            "alert_history_count": float(self.alert_count),
            "cumulative_suspicion": self.cumulative_suspicion,
        }
    
    def _find_periods(self, values: list, target: bool) -> list[tuple[int, int]]:
        """Find consecutive periods where value equals target."""
        periods = []
        start = None
        
        for i, v in enumerate(values):
            if v == target and start is None:
                start = i
            elif v != target and start is not None:
                periods.append((start, i))
                start = None
        
        if start is not None:
            periods.append((start, len(values)))
        
        return periods
    
    def _calculate_entropy(self, sequence: list) -> float:
        """Calculate Shannon entropy of sequence."""
        if not sequence:
            return 0.0
        
        counts = {}
        for v in sequence:
            counts[v] = counts.get(v, 0) + 1
        
        total = len(sequence)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return float(entropy)
    
    def update_alert(self):
        """Called when an alert is generated."""
        self.alert_count += 1
    
    def update_suspicion(self, score: float):
        """Update cumulative suspicion score."""
        self.cumulative_suspicion = 0.9 * self.cumulative_suspicion + 0.1 * score
    
    def reset(self):
        """Reset extractor state."""
        self.frame_history.clear()
        self.alert_count = 0
        self.session_start_time = None
        self.cumulative_suspicion = 0.0
