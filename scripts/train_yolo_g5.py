#!/usr/bin/env python3
"""
YOLO11 Training Script - Optimized for AWS g5.2xlarge (NVIDIA A10G 24GB)

g5.2xlarge Specs:
- 1x NVIDIA A10G GPU (24GB VRAM)
- 8 vCPUs (AMD EPYC)
- 32 GB RAM
- Up to 10 Gbps network
- Supports CUDA, cuDNN, TensorRT

Cost: ~$1.21/hour (on-demand), ~$0.48/hour (spot)

Usage:
    # Balanced preset (recommended)
    python train_yolo_g5.py --data data/training/merged/data.yaml --preset balanced
    
    # Fast training for testing
    python train_yolo_g5.py --data data/training/merged/data.yaml --preset fast
    
    # Custom epochs
    python train_yolo_g5.py --data data/training/merged/data.yaml --epochs 200 --batch 16
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Training presets optimized for A10G 24GB VRAM
PRESETS = {
    "fast": {
        "epochs": 200,
        "batch": 16,          # Safe for 24GB VRAM
        "imgsz": 640,
        "patience": 40,
        "description": "Fast training (~4 hrs, ~$4.80)"
    },
    "balanced": {
        "epochs": 400,
        "batch": 16,
        "imgsz": 640,
        "patience": 60,
        "description": "Balanced speed/accuracy (~8 hrs, ~$9.60)"
    },
    "accurate": {
        "epochs": 600,
        "batch": 16,
        "imgsz": 640,
        "patience": 80,
        "description": "High accuracy (~12 hrs, ~$14.50)"
    },
    "best": {
        "epochs": 800,
        "batch": 12,
        "imgsz": 640,
        "patience": 100,
        "description": "Best accuracy (~16-20 hrs, ~$20.00)"
    },
    "ultimate": {
        "epochs": 1000,
        "batch": 12,
        "imgsz": 640,
        "patience": 120,
        "description": "Ultimate accuracy (~24-30 hrs, ~$30.00)"
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
            
            if "A10G" in gpu_name or (20 < vram_gb < 26):
                print("   ⚡ A10G detected - g5.2xlarge optimized mode!")
            elif "A10" in gpu_name:
                print("   ⚡ A10 variant detected")
            elif vram_gb > 30:
                print("   💪 Larger GPU detected - you can increase batch size!")
            elif vram_gb < 20:
                print("   ⚠️  Smaller GPU - reducing batch size recommended")
            
            return True, vram_gb
        else:
            print("❌ No GPU detected! Training will be extremely slow.")
            return False, 0
    except ImportError:
        print("⚠️  PyTorch not installed. Install with: pip install torch")
        return False, 0


def train(
    data_yaml: str,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 25,
    model: str = "yolo11m.pt",
    name: str = None,
    resume: bool = False,
    workers: int = 4,
):
    """
    Train YOLO11 model optimized for AWS g5.2xlarge (A10G 24GB).
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        batch: Batch size (A10G 24GB handles 8-16)
        imgsz: Image size (640 recommended)
        patience: Early stopping patience
        model: Base model to use
        name: Run name
        resume: Resume from last checkpoint
        workers: Number of data loading workers (4 for 8 vCPUs)
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
    if has_gpu:
        if vram < 20:
            print(f"⚠️  VRAM ({vram:.0f}GB) is limited")
            batch = min(batch, 8)
            print(f"   Adjusted batch size: {batch}")
        elif vram < 24:
            batch = min(batch, 12)
            print(f"   Adjusted batch size: {batch}")
        elif vram > 30:
            print(f"   💪 Extra VRAM available - you could increase batch size")
    
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
        name = f"proctoring_g5_{timestamp}"
    
    print("\n" + "=" * 60)
    print("🚀 YOLO11m Training - AWS g5.2xlarge (A10G 24GB)")
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
    est_time_hrs = (epochs / 100) * 2  # ~2 hrs for 100 epochs on A10G
    est_cost = est_time_hrs * 1.21  # $1.21/hr on-demand
    est_cost_spot = est_time_hrs * 0.48  # ~$0.48/hr spot
    print(f"\n⏱️  Estimated time: ~{est_time_hrs:.1f} hours")
    print(f"💰 Estimated cost: ~${est_cost:.2f} (on-demand) / ~${est_cost_spot:.2f} (spot)")
    print()
    
    # Load model
    print("📦 Loading YOLO11m model...")
    yolo = YOLO(model)
    
    # Training configuration optimized for A10G 24GB
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
        
        # A10G optimizations
        "amp": True,                    # Mixed precision (FP16) - essential for A10G
        "cache": False,                 # Don't cache - only 32GB RAM
        "cos_lr": True,                 # Cosine learning rate scheduler
        "close_mosaic": 10,             # Disable mosaic last 10 epochs
        
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
        "mixup": 0.0,                   # Disable mixup to save VRAM
        
        # Logging
        "verbose": True,
        "plots": True,
        "save": True,
        "save_period": 1,               # Save checkpoint every epoch (important for long training)
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
        description="Train YOLO11 for AI Interview Proctoring (AWS g5.2xlarge A10G optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Balanced preset (recommended)
    python train_yolo_g5.py --data data/training/merged/data.yaml --preset balanced
    
    # Fast training for testing
    python train_yolo_g5.py --data data/training/merged/data.yaml --preset fast
    
    # Best accuracy
    python train_yolo_g5.py --data data/training/merged/data.yaml --preset best
    
    # Custom configuration
    python train_yolo_g5.py --data data/training/merged/data.yaml --epochs 150 --batch 12
    
    # Resume interrupted training
    python train_yolo_g5.py --data data/training/merged/data.yaml --resume

Presets (A10G 24GB VRAM, 32GB RAM):
    fast      - 200 epochs,  batch=16 (~4 hrs,     ~$4.80)
    balanced  - 400 epochs,  batch=16 (~8 hrs,     ~$9.60) [DEFAULT]
    accurate  - 600 epochs,  batch=16 (~12 hrs,    ~$14.50)
    best      - 800 epochs,  batch=12 (~16-20 hrs, ~$20.00)
    ultimate  - 1000 epochs, batch=12 (~24-30 hrs, ~$30.00)

Tips for g5.2xlarge:
    - Use spot instances for ~60% cost savings
    - batch=16 is optimal for yolo11m on 24GB VRAM
    - cache=False because only 32GB RAM (not enough for 157K images)
    - workers=4 matches the 8 vCPU count well
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
        default="balanced",
        help="Training preset (default: balanced)"
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
        default=4,
        help="Number of data loading workers (default: 4 for 8 vCPUs)"
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
        model=model,
        name=args.name,
        resume=args.resume,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
