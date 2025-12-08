"""
Download Pre-trained Models

Downloads YOLO11 - the latest Ultralytics YOLO model.
https://docs.ultralytics.com/models/yolo11/
"""

import os
from pathlib import Path


def download_yolo():
    """Download YOLO11 nano model."""
    from ultralytics import YOLO
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    model_path = weights_dir / "yolo11n.pt"
    
    if not model_path.exists():
        print("📥 Downloading YOLO11n (latest)...")
        model = YOLO("yolo11n.pt")
        # Model is auto-downloaded, copy to weights dir
        import shutil
        default_path = Path.home() / ".config" / "Ultralytics" / "yolo11n.pt"
        if default_path.exists():
            shutil.copy(default_path, model_path)
        print(f"✅ YOLO11n saved to: {model_path}")
    else:
        print(f"✅ YOLO11n already exists: {model_path}")


def main():
    """Download all required models."""
    print("🔧 Downloading models...")
    
    # Create directories
    Path("weights").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    download_yolo()
    
    print("\n✅ All models ready!")
    print("\nTo fine-tune YOLO11 on custom data:")
    print("  1. Prepare your dataset in YOLO format")
    print("  2. Run: python scripts/train_yolo.py --data data/dataset.yaml")


if __name__ == "__main__":
    main()
