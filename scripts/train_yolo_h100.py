"""
YOLO11 Fine-Tuning Script - Optimized for NVIDIA H100 80GB GPU

Fine-tune YOLO11 (latest Ultralytics) for interview-specific object detection:
- Phones (handheld, on desk)
- Multiple people  
- Cheat sheets / notes
- Secondary screens

Optimizations for H100 80GB:
- Maximum batch sizes for 80GB VRAM
- Full mixed precision (BF16) support
- Multi-threaded data loading
- Fastest training possible

https://docs.ultralytics.com/models/yolo11/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from ultralytics import YOLO


# H100 80GB optimized configurations - MAXIMUM PERFORMANCE
H100_CONFIGS = {
    # Fast training - for quick experiments
    "fast": {
        "model": "yolo11s.pt",
        "epochs": 100,
        "batch": 64,          # H100 can handle large batches
        "imgsz": 640,
        "patience": 20,
    },
    # Balanced - good accuracy with fast training
    "balanced": {
        "model": "yolo11m.pt",
        "epochs": 150,
        "batch": 32,
        "imgsz": 640,
        "patience": 30,
    },
    # Accurate - high accuracy
    "accurate": {
        "model": "yolo11l.pt",
        "epochs": 250,
        "batch": 24,
        "imgsz": 800,
        "patience": 50,
    },
    # Best - very high accuracy
    "best": {
        "model": "yolo11x.pt",
        "epochs": 300,
        "batch": 16,
        "imgsz": 800,
        "patience": 60,
    },
    # Ultimate - MAXIMUM accuracy with YOLO11x (default)
    "ultimate": {
        "model": "yolo11x.pt",
        "epochs": 400,
        "batch": 16,          # H100 80GB handles batch=16 with imgsz=1024
        "imgsz": 1024,
        "patience": 80,
    },
}


def check_gpu():
    """Check GPU availability and memory."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   Memory: {gpu_mem:.1f} GB")
        
        if "H100" in gpu_name:
            print("   🚀 H100 detected - MAXIMUM PERFORMANCE MODE!")
        elif "A100" in gpu_name:
            print("   ⚡ A100 detected - using optimized settings!")
        return True
    else:
        print("⚠️  No CUDA GPU detected - training will be slow on CPU")
        return False


def get_optimal_workers():
    """Get optimal number of dataloader workers for H100."""
    cpu_count = os.cpu_count() or 8
    # H100 instances typically have 24+ CPUs
    # Use more workers for faster data loading
    return min(cpu_count, 16)


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
    base_model: str = "yolo11x.pt",
    epochs: int = 400,
    imgsz: int = 1024,
    batch: int = 16,
    patience: int = 80,
    project: str = "runs/detect",
    name: Optional[str] = None,
    device: str = "0",
    resume: bool = False,
    preset: Optional[str] = None,
):
    """
    Fine-tune YOLO11x on custom dataset - Optimized for H100 80GB GPU.
    
    Args:
        data_yaml: Path to dataset YAML file
        base_model: Base model (default: yolo11x - ultimate accuracy)
        epochs: Number of training epochs (default: 400 for maximum accuracy)
        imgsz: Image size for training (default: 1024 for best detection)
        batch: Batch size (default: 16 for H100 80GB)
        patience: Early stopping patience (default: 80 for thorough training)
        project: Project directory for outputs
        name: Run name (auto-generated if None)
        device: Device to use (0 for GPU)
        resume: Resume from last checkpoint
        preset: Training preset (fast, balanced, accurate, best, ultimate)
    """
    # Check GPU
    has_gpu = check_gpu()
    
    # Apply preset if specified
    if preset and preset in H100_CONFIGS:
        config = H100_CONFIGS[preset]
        base_model = config["model"]
        epochs = config["epochs"]
        batch = config["batch"]
        imgsz = config["imgsz"]
        patience = config["patience"]
        print(f"\n📋 Using '{preset}' preset for H100 80GB")
    
    # Get optimal workers
    workers = get_optimal_workers()
    
    print(f"\n🚀 Starting YOLO11 fine-tuning (H100 80GB MAXIMUM PERFORMANCE)...")
    print(f"   Base model: {base_model}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Image size: {imgsz}")
    print(f"   Workers: {workers}")
    print(f"   Mixed Precision (AMP): Enabled")
    print(f"   Device: cuda:{device}")
    
    # Load base model
    model = YOLO(base_model)
    
    # Generate run name if not provided
    if name is None:
        name = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # H100 80GB optimized training configuration - MAXIMUM PERFORMANCE
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=project,
        name=name,
        device=device if has_gpu else "cpu",
        resume=resume,
        
        # H100 Performance Optimization
        amp=True,              # Mixed precision - BF16 on H100
        cache="ram",           # Cache in RAM - H100 has plenty (200GB)
        workers=workers,       # Max workers for data loading
        
        # Strong augmentation for better generalization & accuracy
        hsv_h=0.015,           # Hue augmentation
        hsv_s=0.7,             # Saturation augmentation
        hsv_v=0.4,             # Value augmentation
        degrees=15.0,          # Increased rotation
        translate=0.15,        # More translation
        scale=0.5,             # More scale variation
        shear=2.0,             # Add shear
        perspective=0.0001,    # Slight perspective
        flipud=0.0,            # No vertical flip
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Full mosaic augmentation
        mixup=0.2,             # Increased mixup
        copy_paste=0.15,       # More copy-paste augmentation
        erasing=0.3,           # Random erasing for robustness
        
        # Optimization - tuned for maximum accuracy
        optimizer="AdamW",
        lr0=0.001,             # Slightly higher LR for H100
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,       # Longer warmup
        warmup_momentum=0.8,
        cos_lr=True,           # Cosine LR scheduler
        
        # Training stability & accuracy
        nbs=64,                # Nominal batch size for loss scaling
        close_mosaic=15,       # Disable mosaic for last 15 epochs
        
        # Multi-scale training for better accuracy
        rect=False,            # Disable rect training for better augmentation
        
        # Output
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        exist_ok=True,
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {project}/{name}/weights/best.pt")
    print(f"   Last model: {project}/{name}/weights/last.pt")
    
    # Print final metrics
    if results:
        print(f"\n📈 Training Results:")
        try:
            print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        except:
            print("   Check results in the output folder")
    
    return results


def export_model(model_path: str, format: str = "onnx"):
    """Export trained model to different format."""
    model = YOLO(model_path)
    model.export(format=format)
    print(f"✅ Model exported to {format}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11x for interview detection (H100 80GB - MAXIMUM PERFORMANCE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # RECOMMENDED: Ultimate accuracy with YOLO11x (default)
    python scripts/train_yolo_h100.py --data data.yaml
    
    # Or explicitly use ultimate preset
    python scripts/train_yolo_h100.py --data data.yaml --preset ultimate
    
    # Faster training if needed
    python scripts/train_yolo_h100.py --data data.yaml --preset balanced

Presets (optimized for H100 80GB - MAXIMUM PERFORMANCE):
    fast      - yolo11s, batch=64, 100 epochs  (~15-20 min)
    balanced  - yolo11m, batch=32, 150 epochs  (~30-45 min)
    accurate  - yolo11l, batch=24, 250 epochs  (~1-1.5 hours)
    best      - yolo11x, batch=16, 300 epochs, imgsz=800 (~1.5-2 hours)
    ultimate  - yolo11x, batch=16, 400 epochs, imgsz=1024 (~2-2.5 hours) [DEFAULT]

Estimated cost on H100 (₹213/hr):
    fast      - ₹70
    balanced  - ₹160
    accurate  - ₹320
    best      - ₹430
    ultimate  - ₹530
        """
    )
    
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Base model (yolo11s, yolo11m, yolo11l, yolo11x)")
    parser.add_argument("--preset", type=str, choices=["fast", "balanced", "accurate", "best", "ultimate"], default="ultimate", help="Training preset (default: ultimate)")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs (default: 400)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16 for H100)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size (default: 1024)")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--export", type=str, default=None, help="Export format after training (onnx, tensorrt)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 YOLO11 Training - Optimized for NVIDIA H100 80GB GPU")
    print("=" * 70)
    
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
        preset=args.preset,
    )
    
    # Export if requested
    if args.export:
        best_model = Path(args.project) / (args.name or "train") / "weights" / "best.pt"
        if best_model.exists():
            export_model(str(best_model), args.export)


if __name__ == "__main__":
    main()
