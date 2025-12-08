"""
YOLO11 Object Detection Model

Uses Ultralytics YOLO11 - the latest and most performant YOLO version.
https://docs.ultralytics.com/models/yolo11/
"""

from pathlib import Path
from ultralytics import YOLO
import numpy as np

from config import settings


# COCO class IDs we care about
PERSON_CLASS = 0
PHONE_CLASS = 67
LAPTOP_CLASS = 63
BOOK_CLASS = 73
TV_CLASS = 62


class YOLODetector:
    """YOLO11 wrapper for object detection."""
    
    # Classes we want to detect
    TARGET_CLASSES = {
        PERSON_CLASS: "person",
        PHONE_CLASS: "phone",
        LAPTOP_CLASS: "laptop",
        BOOK_CLASS: "book",
        TV_CLASS: "tv",
    }
    
    def __init__(self, model_path: str | None = None):
        """
        Initialize YOLO11 detector.
        
        Args:
            model_path: Path to model weights. Uses default if None.
        """
        path = model_path or settings.yolo_model_path
        
        # Download if not exists
        if not Path(path).exists():
            print(f"📥 Downloading YOLO11 model...")
            path = "yolo11n.pt"  # Will download automatically
        
        self.model = YOLO(path)
        
        # Configure device
        if settings.use_gpu:
            self.model.to("cuda")
        
        print(f"✅ YOLO11 loaded: {path}")
    
    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Detection results with counts and bounding boxes.
        """
        # Run inference
        results = self.model(
            frame,
            conf=settings.yolo_confidence,
            classes=list(self.TARGET_CLASSES.keys()),
            verbose=False,
        )
        
        # Parse results
        detections = {
            "person_count": 0,
            "phone_detected": False,
            "phone_boxes": [],
            "laptop_detected": False,
            "book_detected": False,
            "all_boxes": [],
        }
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].cpu().numpy()
            
            box_data = {
                "class": self.TARGET_CLASSES.get(cls, "unknown"),
                "confidence": conf,
                "box": xyxy.tolist(),
            }
            
            detections["all_boxes"].append(box_data)
            
            if cls == PERSON_CLASS:
                detections["person_count"] += 1
            elif cls == PHONE_CLASS:
                detections["phone_detected"] = True
                detections["phone_boxes"].append(box_data)
            elif cls == LAPTOP_CLASS:
                detections["laptop_detected"] = True
            elif cls == BOOK_CLASS:
                detections["book_detected"] = True
        
        return detections
    
    def detect_batch(self, frames: list[np.ndarray]) -> list[dict]:
        """Run detection on multiple frames."""
        return [self.detect(frame) for frame in frames]


class YOLOFineTuner:
    """Fine-tune YOLOv8 for interview-specific detection."""
    
    def __init__(self, base_model: str = "yolov8n.pt"):
        self.model = YOLO(base_model)
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "runs/detect",
        name: str = "interview_yolo",
    ):
        """
        Fine-tune the model on custom dataset.
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            project: Project directory
            name: Run name
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            patience=10,
            save=True,
            device=0 if settings.use_gpu else "cpu",
        )
        
        return results
    
    def export(self, format: str = "onnx"):
        """Export model to different format."""
        return self.model.export(format=format)
