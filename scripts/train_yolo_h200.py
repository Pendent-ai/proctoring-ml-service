#!/usr/bin/env python3
"""
YOLO11 Training Script - Optimized for NVIDIA H200 (141GB VRAM)

H200 Specs:
- 141GB HBM3e VRAM (MASSIVE!)
- 1,979 TFLOPs FP16/BF16
- 989 TFLOPs TF32
- 3,958 TOPS INT8
- ~2x faster than H100, ~4x faster than A100

Cost: 3.62 Lightning credits/hour

Usage:
    # Ultimate preset (recommended for max accuracy)
    python train_yolo_h200.py --data data/training/merged/data.yaml --preset ultimate
    
    # Fast training
    python train_yolo_h200.py --data data/training/merged/data.yaml --preset fast
    
    # Custom epochs
    python train_yolo_h200.py --data data/training/merged/data.yaml --epochs 500 --batch 64
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Training presets optimized for H200 141GB VRAM
PRESETS = {
    "fast": {
        "epochs": 100,
        "batch": 128,        # H200 can handle massive batches
        "imgsz": 640,
        "patience": 30,
        "description": "Fast training (~25-30 min, ~2 credits)"
    },
    "balanced": {
        "epochs": 200,
        "batch": 128,
        "imgsz": 640,
        "patience": 40,
        "description": "Balanced speed/accuracy (~45 min, ~3 credits)"
    },
    "accurate": {
        "epochs": 300,
        "batch": 128,
        "imgsz": 640,
        "patience": 50,
        "description": "High accuracy (~1 hr, ~4 credits)"
    },
    "best": {
        "epochs": 400,
        "batch": 128,
        "imgsz": 640,
        "patience": 60,
        "description": "Best accuracy (~1.5 hrs, ~6 credits)"
    },
    "ultimate": {
        "epochs": 1000,
        "batch": 128,
        "imgsz": 640,
        "patience": 150,
        "description": "Ultimate accuracy 1000 epochs (~4-5 hrs, ~18 credits)"
    },
    "extreme": {
        "epochs": 600,
        "batch": 128,
        "imgsz": 800,       # Higher resolution for extreme accuracy
        "patience": 100,
        "description": "EXTREME accuracy (~3 hrs, ~12 credits)"
    },
}

# 9 unified OBJECT classes for AI interview proctoring
# (Removed behavioral classes - use MediaPipe for gaze/pose detection)
INTERVIEW_CLASSES = [
    "phone",           # Mobile phone usage
    "earbuds",         # Earbuds/AirPods
    "smartwatch",      # Smartwatch
    "notes",           # Notes/cheat sheets/books
    "another_person",  # Extra person in frame
    "laptop",          # Secondary laptop/tablet
    "second_screen",   # TV/monitor in background
    "calculator",      # Calculator device
    "face_hiding",     # Face covered by hand/scarf/mask
]


def check_gpu():
    """Check GPU availability and specs."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   VRAM: {vram_gb:.1f} GB")
            
            if "H200" in gpu_name or vram_gb > 130:
                print("   🔥 H200 BEAST MODE ACTIVATED!")
            elif "H100" in gpu_name or vram_gb > 70:
                print("   ⚡ H100 detected - consider using train_yolo_h100.py")
            elif "A100" in gpu_name or vram_gb > 35:
                print("   💪 A100 detected - consider using train_yolo_a100_40gb.py")
            
            return True, vram_gb
        else:
            print("❌ No GPU detected! Training will be extremely slow.")
            return False, 0
    except ImportError:
        print("⚠️  PyTorch not installed. Install with: pip install torch")
        return False, 0


def train(
    data_yaml: str,
    epochs: int = 500,
    batch: int = 64,
    imgsz: int = 640,
    patience: int = 80,
    model: str = "yolo11m.pt",
    name: str = None,
    resume: bool = False,
    workers: int = 12,
):
    """
    Train YOLO11 model optimized for H200.
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        batch: Batch size (H200 can handle 64-128)
        imgsz: Image size (640 or 800 for extreme)
        patience: Early stopping patience
        model: Base model to use
        name: Run name
        resume: Resume from last checkpoint
        workers: Number of data loading workers
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed!")
        print("   Run: pip install ultralytics")
        sys.exit(1)
    
    # Check GPU
    has_gpu, vram = check_gpu()
    
    # Auto-adjust batch size based on VRAM
    if has_gpu and vram < 130:
        print(f"⚠️  VRAM ({vram:.0f}GB) less than expected for H200 (141GB)")
        print("   Reducing batch size for safety...")
        if vram < 80:
            batch = min(batch, 32)
        elif vram < 100:
            batch = min(batch, 48)
        else:
            batch = min(batch, 64)
        print(f"   Adjusted batch size: {batch}")
    
    # Validate data path
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_yaml}")
        print("   Run download script first:")
        print("   python scripts/download_datasets.py --priority 1 --merge")
        sys.exit(1)
    
    # Generate run name
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"proctoring_h200_{timestamp}"
    
    print("\n" + "=" * 60)
    print("🚀 YOLO11m Training - H200 BEAST MODE")
    print("=" * 60)
    print(f"   Model: {model}")
    print(f"   Data: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Image size: {imgsz}")
    print(f"   Patience: {patience}")
    print(f"   Workers: {workers}")
    print(f"   Run name: {name}")
    print("=" * 60)
    
    # Estimated time and cost
    est_time_hrs = (epochs / 500) * 2  # ~2 hrs for 500 epochs
    est_credits = est_time_hrs * 3.62
    print(f"\n⏱️  Estimated time: ~{est_time_hrs:.1f} hours")
    print(f"💰 Estimated cost: ~{est_credits:.1f} Lightning credits")
    print()
    
    # Load model
    print("📦 Loading YOLO11m model...")
    yolo = YOLO(model)
    
    # Training configuration optimized for H200
    train_args = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "patience": patience,
        "name": name,
        "workers": workers,
        "device": 0,
        "resume": resume,
        
        # H200 optimizations
        "amp": True,                    # Mixed precision (FP16/BF16)
        "cache": "disk",                # Cache images to disk (RAM not enough for 157K images)
        "cos_lr": True,                 # Cosine learning rate scheduler
        "close_mosaic": 20,             # Disable mosaic last 20 epochs
        
        # Augmentation for interview proctoring
        "hsv_h": 0.015,                 # Hue augmentation
        "hsv_s": 0.7,                   # Saturation augmentation
        "hsv_v": 0.4,                   # Value augmentation
        "degrees": 5.0,                 # Rotation (small for webcam)
        "translate": 0.1,               # Translation
        "scale": 0.3,                   # Scale
        "flipud": 0.0,                  # No vertical flip (webcam)
        "fliplr": 0.5,                  # Horizontal flip
        "mosaic": 1.0,                  # Mosaic augmentation
        "mixup": 0.1,                   # Mixup augmentation
        
        # Logging
        "verbose": True,
        "plots": True,
        "save": True,
        "save_period": 1,               # Save checkpoint every epoch
    }
    
    # Train
    print("🚀 Starting training...")
    print("   Press Ctrl+C to stop early (model will be saved)\n")
    
    try:
        results = yolo.train(**train_args)
        
        print("\n" + "=" * 60)
        print("✅ Training completed!")
        print("=" * 60)
        print(f"   Best model: runs/detect/{name}/weights/best.pt")
        print(f"   Last model: runs/detect/{name}/weights/last.pt")
        print(f"   Results: runs/detect/{name}/")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"   Checkpoint saved to: runs/detect/{name}/weights/last.pt")
        print("   Resume with: --resume flag")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 for AI Interview Proctoring (H200 optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ultimate accuracy (recommended)
    python train_yolo_h200.py --data data/training/merged/data.yaml --preset ultimate
    
    # Fast training for testing
    python train_yolo_h200.py --data data/training/merged/data.yaml --preset fast
    
    # Extreme accuracy (longer training)
    python train_yolo_h200.py --data data/training/merged/data.yaml --preset extreme
    
    # Custom configuration
    python train_yolo_h200.py --data data/training/merged/data.yaml --epochs 400 --batch 64
    
    # Resume interrupted training
    python train_yolo_h200.py --data data/training/merged/data.yaml --resume

Presets (H200 141GB VRAM):
    fast      - 100 epochs, batch=96  (~25-30 min, ~2 credits)
    balanced  - 200 epochs, batch=80  (~45 min, ~3 credits)
    accurate  - 300 epochs, batch=64  (~1 hr, ~4 credits)
    best      - 400 epochs, batch=48  (~1.5 hrs, ~6 credits)
    ultimate  - 500 epochs, batch=32  (~2 hrs, ~8 credits)
    extreme   - 600 epochs, batch=24, imgsz=800 (~3 hrs, ~12 credits)
        """
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to data.yaml file"
    )
    
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=list(PRESETS.keys()),
        default="ultimate",
        help="Training preset (default: ultimate)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Number of epochs (overrides preset)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=None,
        help="Batch size (overrides preset)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size (overrides preset)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11m.pt",
        help="Base model (default: yolo11m.pt for balanced accuracy/speed)"
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Run name"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from last checkpoint"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., runs/detect/run_name/weights/last.pt)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=12,
        help="Number of data loading workers (default: 12)"
    )
    
    args = parser.parse_args()
    
    # Get preset configuration
    preset = PRESETS[args.preset]
    print(f"\n📋 Using preset: {args.preset}")
    print(f"   {preset['description']}")
    
    # Apply preset with overrides
    epochs = args.epochs if args.epochs else preset["epochs"]
    batch = args.batch if args.batch else preset["batch"]
    imgsz = args.imgsz if args.imgsz else preset["imgsz"]
    patience = preset["patience"]
    
    # Handle resume - auto-detect checkpoint if not specified
    model = args.model
    if args.resume:
        if args.resume_from:
            model = args.resume_from
            print(f"📂 Resuming from: {model}")
        else:
            # Try to find the most recent last.pt
            from pathlib import Path
            runs_dir = Path("runs/detect")
            if runs_dir.exists():
                # Find most recent run with last.pt
                checkpoints = sorted(runs_dir.glob("*/weights/last.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
                if checkpoints:
                    model = str(checkpoints[0])
                    print(f"📂 Auto-detected checkpoint: {model}")
                else:
                    print("❌ No checkpoint found! Starting fresh training.")
                    print("   Use --resume-from to specify checkpoint path")
            else:
                print("❌ No runs directory found! Starting fresh training.")
    
    # Run training
    train(
        data_yaml=args.data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        model=model,  # Use the updated model path (checkpoint if resuming)
        name=args.name,
        resume=args.resume,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
