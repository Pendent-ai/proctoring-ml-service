"""
Download Pre-trained Models
"""

import os
from pathlib import Path


def download_yolo():
    """Download YOLOv8 nano model."""
    from ultralytics import YOLO
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    model_path = weights_dir / "yolov8n.pt"
    
    if not model_path.exists():
        print("📥 Downloading YOLOv8n...")
        model = YOLO("yolov8n.pt")
        # Model is auto-downloaded, copy to weights dir
        import shutil
        default_path = Path.home() / ".config" / "Ultralytics" / "yolov8n.pt"
        if default_path.exists():
            shutil.copy(default_path, model_path)
        print(f"✅ YOLOv8n saved to: {model_path}")
    else:
        print(f"✅ YOLOv8n already exists: {model_path}")


def main():
    """Download all required models."""
    print("🔧 Downloading models...")
    
    # Create directories
    Path("weights").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    download_yolo()
    
    print("\n✅ All models ready!")
    print("\nTo fine-tune YOLOv8 on custom data:")
    print("  1. Prepare your dataset in YOLO format")
    print("  2. Run: python scripts/train_yolo.py --data data/dataset.yaml")


if __name__ == "__main__":
    main()
