"""
YOLOv8 Fine-Tuning Script

Fine-tune YOLOv8 for interview-specific object detection:
- Phones (handheld, on desk)
- Multiple people
- Cheat sheets / notes
- Secondary screens
"""

import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


def create_dataset_yaml(data_dir: Path, output_path: Path):
    """Create dataset YAML configuration."""
    yaml_content = f"""
# Interview Object Detection Dataset
path: {data_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: person
  1: phone
  2: phone_in_hand
  3: cheat_sheet
  4: secondary_screen
  5: book
  6: laptop

# Number of classes
nc: 7
"""
    output_path.write_text(yaml_content)
    print(f"✅ Dataset YAML created: {output_path}")
    return output_path


def train(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 10,
    project: str = "runs/detect",
    name: str | None = None,
    device: str = "0",
    resume: bool = False,
):
    """
    Fine-tune YOLOv8 on custom dataset.
    
    Args:
        data_yaml: Path to dataset YAML file
        base_model: Base model to fine-tune from
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        patience: Early stopping patience
        project: Project directory for outputs
        name: Run name (auto-generated if None)
        device: Device to use (0 for GPU, cpu for CPU)
        resume: Resume from last checkpoint
    """
    print("🚀 Starting YOLOv8 fine-tuning...")
    print(f"   Base model: {base_model}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Image size: {imgsz}")
    
    # Load base model
    model = YOLO(base_model)
    
    # Generate run name if not provided
    if name is None:
        name = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=project,
        name=name,
        device=device,
        resume=resume,
        
        # Augmentation settings optimized for interview context
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=10,   # Rotation (limited for webcam frames)
        translate=0.1,
        scale=0.3,
        flipud=0.0,   # No vertical flip
        fliplr=0.5,   # Horizontal flip
        mosaic=0.5,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
        
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Output
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {project}/{name}/weights/best.pt")
    print(f"   Last model: {project}/{name}/weights/last.pt")
    
    return results


def export_model(model_path: str, format: str = "onnx"):
    """Export trained model to different format."""
    model = YOLO(model_path)
    model.export(format=format)
    print(f"✅ Model exported to {format}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for interview detection")
    
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--export", type=str, default=None, help="Export format after training")
    
    args = parser.parse_args()
    
    # Validate data path
    if not Path(args.data).exists():
        print(f"❌ Dataset YAML not found: {args.data}")
        print("\nTo create a dataset, organize your images as:")
        print("  data/")
        print("  ├── images/")
        print("  │   ├── train/")
        print("  │   ├── val/")
        print("  │   └── test/")
        print("  └── labels/")
        print("      ├── train/")
        print("      ├── val/")
        print("      └── test/")
        print("\nLabel format (YOLO): class x_center y_center width height")
        return
    
    # Train
    results = train(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
        resume=args.resume,
    )
    
    # Export if requested
    if args.export:
        best_model = Path(args.project) / (args.name or "train") / "weights" / "best.pt"
        if best_model.exists():
            export_model(str(best_model), args.export)


if __name__ == "__main__":
    main()
