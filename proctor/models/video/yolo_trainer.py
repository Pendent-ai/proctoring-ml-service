"""
Custom YOLO training for interview-specific object detection.

This module provides training utilities for fine-tuning YOLO11 on
interview proctoring-specific classes beyond standard COCO classes.

Custom Classes:
- phone_in_hand: Phone being held by candidate
- phone_on_desk: Phone visible on desk
- earbuds: Wireless earbuds/headphones
- smartwatch: Smartwatch on wrist
- second_screen: Additional monitor/screen
- notes: Paper notes or cheat sheets
- book: Open book or reference material
- tablet: Tablet device
- another_person: Second person in frame

Training supports:
- Transfer learning from pretrained YOLO11
- Custom data augmentation for interview scenarios
- Class-balanced sampling
- Export to optimized formats (ONNX, TensorRT)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from proctor.utils.logger import get_logger

logger = get_logger(__name__)


# Custom classes for interview proctoring
PROCTORING_CLASSES = {
    0: "phone_in_hand",
    1: "phone_on_desk",
    2: "earbuds",
    3: "smartwatch",
    4: "second_screen",
    5: "notes",
    6: "book",
    7: "tablet",
    8: "another_person",
    9: "laptop",
    10: "calculator",
}

# Severity mapping for each class
CLASS_SEVERITY = {
    "phone_in_hand": "critical",
    "phone_on_desk": "high",
    "earbuds": "critical",
    "smartwatch": "medium",
    "second_screen": "high",
    "notes": "high",
    "book": "medium",
    "tablet": "high",
    "another_person": "critical",
    "laptop": "low",
    "calculator": "low",
}


class AugmentationType(str, Enum):
    """Data augmentation types."""

    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


@dataclass
class YOLOTrainingConfig:
    """Configuration for YOLO training."""

    # Model settings
    base_model: str = "yolo11n.pt"  # Pretrained model to start from
    model_size: str = "n"  # n, s, m, l, x

    # Dataset settings
    data_yaml: str = "data.yaml"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    patience: int = 50

    # Augmentation
    augmentation: AugmentationType = AugmentationType.MEDIUM
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5

    # Class balance
    use_class_weights: bool = True
    class_weights: dict[str, float] = field(default_factory=dict)

    # Hardware
    device: str = "auto"
    workers: int = 8
    amp: bool = True  # Automatic mixed precision

    # Output
    project: str = "runs/train"
    name: str = "proctoring_yolo"
    save_period: int = 10
    cache: bool = True

    def get_augmentation_config(self) -> dict[str, float]:
        """Get augmentation parameters based on level."""
        if self.augmentation == AugmentationType.NONE:
            return {
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
            }
        elif self.augmentation == AugmentationType.LIGHT:
            return {
                "mosaic": 0.5,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "degrees": 5.0,
                "translate": 0.05,
                "scale": 0.2,
                "shear": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
            }
        elif self.augmentation == AugmentationType.HEAVY:
            return {
                "mosaic": 1.0,
                "mixup": 0.3,
                "copy_paste": 0.3,
                "degrees": 15.0,
                "translate": 0.2,
                "scale": 0.9,
                "shear": 5.0,
                "flipud": 0.1,
                "fliplr": 0.5,
            }
        else:  # MEDIUM (default)
            return {
                "mosaic": self.mosaic,
                "mixup": self.mixup,
                "copy_paste": self.copy_paste,
                "degrees": self.degrees,
                "translate": self.translate,
                "scale": self.scale,
                "shear": self.shear,
                "flipud": self.flipud,
                "fliplr": self.fliplr,
            }


class DatasetPreparer:
    """Prepare dataset for YOLO training."""

    def __init__(self, config: YOLOTrainingConfig):
        """Initialize preparer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.classes = list(PROCTORING_CLASSES.values())

    def create_data_yaml(
        self,
        dataset_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Create YOLO data.yaml configuration.

        Args:
            dataset_path: Path to dataset root
            output_path: Path to save data.yaml

        Returns:
            Path to created data.yaml
        """
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)

        data_config = {
            "path": str(dataset_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(self.classes)},
            "nc": len(self.classes),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        logger.info(f"Created data.yaml at {output_path}")
        return output_path

    def prepare_directory_structure(self, dataset_path: str | Path) -> None:
        """
        Create YOLO-compatible directory structure.

        Args:
            dataset_path: Path to dataset root
        """
        dataset_path = Path(dataset_path)

        # Create directories
        for split in ["train", "val", "test"]:
            (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directory structure at {dataset_path}")

    def calculate_class_weights(
        self, labels_path: str | Path
    ) -> dict[int, float]:
        """
        Calculate class weights based on frequency.

        Args:
            labels_path: Path to labels directory

        Returns:
            Dictionary of class_id -> weight
        """
        import glob

        labels_path = Path(labels_path)
        class_counts: dict[int, int] = {i: 0 for i in range(len(self.classes))}

        # Count instances per class
        for label_file in glob.glob(str(labels_path / "**/*.txt"), recursive=True):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1

        # Calculate weights (inverse frequency)
        total = sum(class_counts.values())
        if total == 0:
            return {i: 1.0 for i in range(len(self.classes))}

        weights = {}
        for class_id, count in class_counts.items():
            if count > 0:
                weights[class_id] = total / (len(self.classes) * count)
            else:
                weights[class_id] = 1.0

        # Normalize weights
        max_weight = max(weights.values())
        weights = {k: v / max_weight for k, v in weights.items()}

        logger.info(f"Class weights calculated: {weights}")
        return weights


class YOLOTrainer:
    """
    YOLO trainer for interview proctoring detection.

    Handles training, validation, and export of custom YOLO models.
    """

    def __init__(self, config: YOLOTrainingConfig | None = None):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or YOLOTrainingConfig()
        self.model = None
        self.results = None
        self._yolo = None

    def _load_ultralytics(self):
        """Lazy load ultralytics."""
        if self._yolo is None:
            try:
                from ultralytics import YOLO

                self._yolo = YOLO
            except ImportError:
                raise ImportError(
                    "ultralytics package required. Install with: pip install ultralytics"
                )
        return self._yolo

    def train(
        self,
        data_yaml: str | Path,
        resume: bool = False,
    ) -> dict[str, Any]:
        """
        Train YOLO model on custom dataset.

        Args:
            data_yaml: Path to data.yaml configuration
            resume: Whether to resume from last checkpoint

        Returns:
            Training results
        """
        YOLO = self._load_ultralytics()

        # Load base model
        self.model = YOLO(self.config.base_model)

        logger.info(f"Starting YOLO training with {self.config.base_model}")
        logger.info(f"Data config: {data_yaml}")

        # Get augmentation config
        aug_config = self.config.get_augmentation_config()

        # Train
        self.results = self.model.train(
            data=str(data_yaml),
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            imgsz=self.config.image_size,
            lr0=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            warmup_epochs=self.config.warmup_epochs,
            patience=self.config.patience,
            device=self.config.device if self.config.device != "auto" else None,
            workers=self.config.workers,
            amp=self.config.amp,
            project=self.config.project,
            name=self.config.name,
            save_period=self.config.save_period,
            cache=self.config.cache,
            resume=resume,
            # Augmentation
            **aug_config,
            hsv_h=self.config.hsv_h,
            hsv_s=self.config.hsv_s,
            hsv_v=self.config.hsv_v,
        )

        logger.info("Training complete")

        return {
            "best_model": str(self.results.save_dir / "weights/best.pt"),
            "last_model": str(self.results.save_dir / "weights/last.pt"),
            "results_dir": str(self.results.save_dir),
            "metrics": self._extract_metrics(),
        }

    def _extract_metrics(self) -> dict[str, float]:
        """Extract metrics from training results."""
        if self.results is None:
            return {}

        try:
            return {
                "precision": float(self.results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(self.results.results_dict.get("metrics/recall(B)", 0)),
                "mAP50": float(self.results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(self.results.results_dict.get("metrics/mAP50-95(B)", 0)),
            }
        except Exception:
            return {}

    def validate(
        self,
        model_path: str | Path | None = None,
        data_yaml: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Validate model on test set.

        Args:
            model_path: Path to model weights
            data_yaml: Path to data.yaml

        Returns:
            Validation metrics
        """
        YOLO = self._load_ultralytics()

        if model_path is not None:
            self.model = YOLO(model_path)
        elif self.model is None:
            raise ValueError("No model loaded. Provide model_path or train first.")

        results = self.model.val(data=str(data_yaml) if data_yaml else None)

        return {
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "speed": {
                "preprocess": results.speed.get("preprocess", 0),
                "inference": results.speed.get("inference", 0),
                "postprocess": results.speed.get("postprocess", 0),
            },
        }

    def export(
        self,
        model_path: str | Path | None = None,
        format: str = "onnx",
        optimize: bool = True,
        half: bool = False,
        dynamic: bool = True,
    ) -> Path:
        """
        Export model to deployment format.

        Args:
            model_path: Path to model weights
            format: Export format (onnx, torchscript, tflite, etc.)
            optimize: Whether to optimize for inference
            half: Whether to use FP16
            dynamic: Whether to use dynamic input shapes

        Returns:
            Path to exported model
        """
        YOLO = self._load_ultralytics()

        if model_path is not None:
            self.model = YOLO(model_path)
        elif self.model is None:
            raise ValueError("No model loaded. Provide model_path or train first.")

        export_path = self.model.export(
            format=format,
            optimize=optimize,
            half=half,
            dynamic=dynamic,
        )

        logger.info(f"Exported model to {export_path}")
        return Path(export_path)

    def predict(
        self,
        source: str | Path,
        model_path: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Run inference on images/video.

        Args:
            source: Path to image/video or directory
            model_path: Path to model weights
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Whether to save annotated results

        Returns:
            List of detection results
        """
        YOLO = self._load_ultralytics()

        if model_path is not None:
            self.model = YOLO(model_path)
        elif self.model is None:
            raise ValueError("No model loaded. Provide model_path or train first.")

        results = self.model.predict(
            source=str(source),
            conf=conf,
            iou=iou,
            save=save,
        )

        detections = []
        for result in results:
            frame_detections = []
            for box in result.boxes:
                det = {
                    "class_id": int(box.cls),
                    "class_name": PROCTORING_CLASSES.get(int(box.cls), "unknown"),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                    "severity": CLASS_SEVERITY.get(
                        PROCTORING_CLASSES.get(int(box.cls), ""), "low"
                    ),
                }
                frame_detections.append(det)
            detections.append({"detections": frame_detections})

        return detections


class LabelConverter:
    """Convert labels between different formats."""

    @staticmethod
    def coco_to_yolo(
        coco_json: str | Path,
        output_dir: str | Path,
        class_mapping: dict[str, int] | None = None,
    ) -> None:
        """
        Convert COCO format to YOLO format.

        Args:
            coco_json: Path to COCO JSON annotation
            output_dir: Output directory for YOLO labels
            class_mapping: Optional mapping from COCO category to YOLO class
        """
        import json

        coco_json = Path(coco_json)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(coco_json) as f:
            coco = json.load(f)

        # Build category mapping
        if class_mapping is None:
            class_mapping = {
                cat["name"]: i for i, cat in enumerate(coco["categories"])
            }

        # Build image id to annotations mapping
        image_annotations: dict[int, list] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        # Convert each image
        for image in coco["images"]:
            img_id = image["id"]
            img_width = image["width"]
            img_height = image["height"]
            img_name = Path(image["file_name"]).stem

            labels = []
            if img_id in image_annotations:
                for ann in image_annotations[img_id]:
                    cat_name = next(
                        (c["name"] for c in coco["categories"] if c["id"] == ann["category_id"]),
                        None,
                    )
                    if cat_name and cat_name in class_mapping:
                        class_id = class_mapping[cat_name]
                        x, y, w, h = ann["bbox"]

                        # Convert to YOLO format (center x, center y, width, height - normalized)
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width = w / img_width
                        height = h / img_height

                        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Write label file
            label_file = output_dir / f"{img_name}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(labels))

        logger.info(f"Converted {len(coco['images'])} images to YOLO format")

    @staticmethod
    def voc_to_yolo(
        xml_dir: str | Path,
        output_dir: str | Path,
        class_list: list[str],
    ) -> None:
        """
        Convert Pascal VOC format to YOLO format.

        Args:
            xml_dir: Directory with VOC XML annotations
            output_dir: Output directory for YOLO labels
            class_list: List of class names in order
        """
        import glob
        import xml.etree.ElementTree as ET

        xml_dir = Path(xml_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class_to_id = {name: i for i, name in enumerate(class_list)}

        for xml_file in glob.glob(str(xml_dir / "*.xml")):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get image size
            size = root.find("size")
            img_width = int(size.find("width").text)
            img_height = int(size.find("height").text)

            labels = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in class_to_id:
                    continue

                class_id = class_to_id[class_name]
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # Convert to YOLO format
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Write label file
            label_file = output_dir / f"{Path(xml_file).stem}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(labels))

        logger.info(f"Converted VOC annotations to YOLO format in {output_dir}")
