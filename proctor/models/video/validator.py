"""
Video Validator

Validates video proctoring model performance.
"""

from pathlib import Path
from typing import Any, Optional
import numpy as np

from proctor.engine.validator import BaseValidator
from proctor.cfg import VideoConfig


class VideoValidator(BaseValidator):
    """
    Validator for video proctoring model.
    
    Evaluates detection and face analysis accuracy.
    
    Example:
        >>> validator = VideoValidator(cfg)
        >>> metrics = validator(data_path)
    """
    
    def __init__(self, cfg: VideoConfig):
        """Initialize video validator."""
        super().__init__(cfg)
        
        self.predictor = None
    
    def setup(self, data: str | Path | None = None):
        """Setup validation components."""
        from proctor.models.video.predictor import VideoPredictor
        
        self.predictor = VideoPredictor(self.cfg)
        
        if data:
            self.data = self._load_data(Path(data))
    
    def _load_data(self, path: Path) -> list:
        """Load validation data."""
        import json
        
        if path.suffix == ".json":
            with open(path) as f:
                return json.load(f)
        else:
            # Assume directory of images
            data = []
            for img_path in path.glob("*.jpg"):
                label_path = img_path.with_suffix(".json")
                if label_path.exists():
                    with open(label_path) as f:
                        labels = json.load(f)
                    data.append({
                        "image": str(img_path),
                        "labels": labels,
                    })
            return data
    
    def evaluate(self) -> dict:
        """Run evaluation on validation data."""
        if not self.data:
            return {"error": "No validation data"}
        
        results = {
            "phone_tp": 0, "phone_fp": 0, "phone_fn": 0,
            "face_tp": 0, "face_fp": 0, "face_fn": 0,
            "total": len(self.data),
        }
        
        for sample in self.data:
            import cv2
            
            frame = cv2.imread(sample["image"])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pred = self.predictor(frame)
            labels = sample["labels"]
            
            # Phone detection
            if pred.detection.phone_detected:
                if labels.get("phone_present", False):
                    results["phone_tp"] += 1
                else:
                    results["phone_fp"] += 1
            elif labels.get("phone_present", False):
                results["phone_fn"] += 1
            
            # Face detection
            if pred.face.face_detected:
                if labels.get("face_visible", True):
                    results["face_tp"] += 1
                else:
                    results["face_fp"] += 1
            elif labels.get("face_visible", True):
                results["face_fn"] += 1
        
        return results
    
    def compute_metrics(self, results: dict) -> dict:
        """Compute metrics from evaluation results."""
        metrics = {}
        
        # Phone detection metrics
        phone_tp = results.get("phone_tp", 0)
        phone_fp = results.get("phone_fp", 0)
        phone_fn = results.get("phone_fn", 0)
        
        phone_precision = phone_tp / (phone_tp + phone_fp) if (phone_tp + phone_fp) > 0 else 0
        phone_recall = phone_tp / (phone_tp + phone_fn) if (phone_tp + phone_fn) > 0 else 0
        
        metrics["phone_precision"] = phone_precision
        metrics["phone_recall"] = phone_recall
        metrics["phone_f1"] = (
            2 * phone_precision * phone_recall / (phone_precision + phone_recall)
            if (phone_precision + phone_recall) > 0 else 0
        )
        
        # Face detection metrics
        face_tp = results.get("face_tp", 0)
        face_fp = results.get("face_fp", 0)
        face_fn = results.get("face_fn", 0)
        
        face_precision = face_tp / (face_tp + face_fp) if (face_tp + face_fp) > 0 else 0
        face_recall = face_tp / (face_tp + face_fn) if (face_tp + face_fn) > 0 else 0
        
        metrics["face_precision"] = face_precision
        metrics["face_recall"] = face_recall
        metrics["face_f1"] = (
            2 * face_precision * face_recall / (face_precision + face_recall)
            if (face_precision + face_recall) > 0 else 0
        )
        
        metrics["total_samples"] = results.get("total", 0)
        
        return metrics
