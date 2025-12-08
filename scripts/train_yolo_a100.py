#!/usr/bin/env python3
"""
YOLO11 Training Script - Optimized for A100 64GB GPU
=====================================================

This script is optimized to fully utilize A100 64GB GPU for maximum accuracy.

Usage:
    # On cloud with A100 (Google Colab Pro+, RunPod, Lambda Labs, etc.)
    python scripts/train_yolo_a100.py
    
    # With custom data path
    python scripts/train_yolo_a100.py --data /path/to/data.yaml
    
    # Resume training
    python scripts/train_yolo_a100.py --resume
    
    # Quick test run
    python scripts/train_yolo_a100.py --test

Estimated Training Times on A100 64GB:
    - yolo11n: ~8-10 min
    - yolo11s: ~12-15 min  
    - yolo11m: ~20-25 min (recommended)
    - yolo11l: ~35-45 min
    - yolo11x: ~60-80 min (highest accuracy)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Training configurations optimized for A100 64GB
CONFIGS = {
    # Fast training, good accuracy - recommended for initial experiments
    "fast": {
        "model": "yolo11s.pt",
        "epochs": 100,
        "batch": 64,
        "imgsz": 640,
        "patience": 20,
    },
    
    # Balanced - good accuracy with reasonable time
    "balanced": {
        "model": "yolo11m.pt",
        "epochs": 150,
        "batch": 48,
        "imgsz": 640,
        "patience": 30,
    },
    
    # High accuracy - recommended for production
    "accurate": {
        "model": "yolo11m.pt",
        "epochs": 200,
        "batch": 32,
        "imgsz": 800,
        "patience": 40,
    },
    
    # Maximum accuracy - for best results (longer training)
    "best": {
        "model": "yolo11l.pt",
        "epochs": 300,
        "batch": 24,
        "imgsz": 800,
        "patience": 50,
    },
    
    # Ultimate - absolute best accuracy (longest training)
    "ultimate": {
        "model": "yolo11x.pt",
        "epochs": 400,
        "batch": 16,
        "imgsz": 1024,
        "patience": 60,
    },
}


def check_gpu():
    """Check GPU availability and type."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   Memory: {gpu_mem:.1f} GB")
            
            if "A100" in gpu_name:
                print("   🚀 A100 detected - using optimized settings!")
                return "a100"
            elif "V100" in gpu_name:
                print("   ⚡ V100 detected")
                return "v100"
            elif "T4" in gpu_name:
                print("   💡 T4 detected - will use smaller batch size")
                return "t4"
            else:
                return "other"
        else:
            print("❌ No CUDA GPU detected")
            return None
    except ImportError:
        print("❌ PyTorch not installed")
        return None


def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    
    packages = [
        "ultralytics>=8.3.0",  # YOLO11 support
        "torch>=2.0.0",
        "torchvision",
        "albumentations>=1.3.0",  # Advanced augmentations
        "wandb",  # Experiment tracking (optional)
    ]
    
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
    
    print("✅ Dependencies installed")


def get_optimal_config(gpu_type: str, preset: str = "balanced"):
    """Get optimal training config based on GPU type."""
    config = CONFIGS.get(preset, CONFIGS["balanced"]).copy()
    
    # Adjust batch size based on GPU
    if gpu_type == "a100":
        # A100 64GB can handle large batches
        pass  # Use default config
    elif gpu_type == "v100":
        # V100 32GB - reduce batch slightly
        config["batch"] = min(config["batch"], 32)
    elif gpu_type == "t4":
        # T4 16GB - smaller batches
        config["batch"] = min(config["batch"], 16)
        config["imgsz"] = min(config["imgsz"], 640)
    else:
        # Unknown GPU - conservative settings
        config["batch"] = min(config["batch"], 16)
    
    return config


def train_yolo(
    data_path: str,
    preset: str = "balanced",
    resume: bool = False,
    test_run: bool = False,
    project_name: str = "proctoring_yolo11",
    use_wandb: bool = False,
):
    """
    Train YOLO11 model with optimized settings.
    
    Args:
        data_path: Path to data.yaml
        preset: Training preset (fast, balanced, accurate, best, ultimate)
        resume: Resume from last checkpoint
        test_run: Quick test with minimal epochs
        project_name: Project name for saving runs
        use_wandb: Enable Weights & Biases logging
    """
    from ultralytics import YOLO
    
    # Check GPU
    gpu_type = check_gpu()
    if not gpu_type:
        print("\n⚠️  No GPU detected. Training on CPU will be very slow.")
        print("   Consider using Google Colab, RunPod, or Lambda Labs for GPU access.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Get optimal config
    config = get_optimal_config(gpu_type, preset)
    
    if test_run:
        config["epochs"] = 3
        config["batch"] = 8
        print("\n🧪 Test run mode - 3 epochs only")
    
    print(f"\n📋 Training Configuration:")
    print(f"   Preset: {preset}")
    print(f"   Model: {config['model']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch']}")
    print(f"   Image size: {config['imgsz']}")
    print(f"   Patience: {config['patience']}")
    print(f"   Data: {data_path}")
    
    # Load model
    if resume:
        # Find last run
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            last_run = sorted(runs_dir.glob("*/weights/last.pt"))[-1] if list(runs_dir.glob("*/weights/last.pt")) else None
            if last_run:
                print(f"\n📂 Resuming from: {last_run}")
                model = YOLO(str(last_run))
            else:
                print("❌ No checkpoint found to resume from")
                return
        else:
            print("❌ No runs directory found")
            return
    else:
        model = YOLO(config["model"])
    
    # Training arguments optimized for maximum accuracy
    train_args = {
        "data": data_path,
        "epochs": config["epochs"],
        "batch": config["batch"],
        "imgsz": config["imgsz"],
        "patience": config["patience"],
        "device": 0 if gpu_type else "cpu",
        "project": project_name,
        "name": f"{preset}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        
        # Optimizer settings
        "optimizer": "AdamW",
        "lr0": 0.001,         # Initial learning rate
        "lrf": 0.01,          # Final learning rate factor
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        
        # Augmentation (strong for better generalization)
        "hsv_h": 0.015,       # Hue augmentation
        "hsv_s": 0.7,         # Saturation augmentation
        "hsv_v": 0.4,         # Value augmentation
        "degrees": 10.0,      # Rotation
        "translate": 0.1,     # Translation
        "scale": 0.5,         # Scale
        "shear": 2.0,         # Shear
        "perspective": 0.0001, # Perspective
        "flipud": 0.0,        # Vertical flip (disabled for interview)
        "fliplr": 0.5,        # Horizontal flip
        "mosaic": 1.0,        # Mosaic augmentation
        "mixup": 0.15,        # Mixup augmentation
        "copy_paste": 0.1,    # Copy-paste augmentation
        
        # Training behavior
        "close_mosaic": 10,   # Disable mosaic last N epochs
        "amp": True,          # Mixed precision (faster on A100)
        "cos_lr": True,       # Cosine learning rate scheduler
        "label_smoothing": 0.0,
        "nbs": 64,            # Nominal batch size
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,       # No dropout (use augmentation instead)
        
        # Validation
        "val": True,
        "plots": True,
        "save": True,
        "save_period": 10,    # Save checkpoint every N epochs
        
        # Multi-GPU (if available)
        "workers": 8,
        "cache": True,        # Cache images in RAM for speed
        
        # Resume
        "resume": resume,
        
        # Experiment tracking
        "exist_ok": True,
    }
    
    # Add wandb tracking if enabled
    if use_wandb:
        try:
            import wandb
            train_args["project"] = project_name
            print("\n📊 Weights & Biases logging enabled")
        except ImportError:
            print("⚠️  wandb not installed, skipping...")
    
    print(f"\n🚀 Starting training...")
    print("-" * 50)
    
    # Train!
    results = model.train(**train_args)
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print("=" * 50)
    
    # Print results
    if results:
        print(f"\n📈 Best Results:")
        print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"   Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"   Recall: {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
    
    # Get best weights path
    best_weights = Path(train_args["project"]) / train_args["name"] / "weights" / "best.pt"
    print(f"\n📁 Best weights saved to: {best_weights}")
    
    # Validation on best model
    print("\n🔍 Running final validation...")
    best_model = YOLO(str(best_weights))
    val_results = best_model.val(data=data_path)
    
    print("\n📊 Final Validation Results:")
    print(f"   mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
    # Export to different formats
    print("\n📦 Exporting model...")
    
    # ONNX export for deployment
    best_model.export(format="onnx", dynamic=True, simplify=True)
    print("   ✅ Exported to ONNX")
    
    # TensorRT if on GPU (for fastest inference)
    if gpu_type:
        try:
            best_model.export(format="engine", half=True)  # TensorRT FP16
            print("   ✅ Exported to TensorRT (FP16)")
        except Exception as e:
            print(f"   ⚠️  TensorRT export failed: {e}")
    
    # CoreML for iOS/macOS
    try:
        best_model.export(format="coreml", nms=True)
        print("   ✅ Exported to CoreML")
    except Exception as e:
        print(f"   ⚠️  CoreML export skipped: {e}")
    
    print("\n🎉 Training pipeline complete!")
    print(f"\nNext steps:")
    print(f"  1. Review training plots in: {train_args['project']}/{train_args['name']}/")
    print(f"  2. Test model: yolo predict model={best_weights} source=your_image.jpg")
    print(f"  3. Deploy with: {best_weights.with_suffix('.onnx')} or TensorRT engine")
    
    return str(best_weights)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11 Training Script - Optimized for A100 64GB GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default balanced training
    python scripts/train_yolo_a100.py
    
    # Fast training for quick experiments
    python scripts/train_yolo_a100.py --preset fast
    
    # Best accuracy (longer training)
    python scripts/train_yolo_a100.py --preset best
    
    # Ultimate accuracy (longest training)
    python scripts/train_yolo_a100.py --preset ultimate
    
    # Custom data path
    python scripts/train_yolo_a100.py --data /path/to/data.yaml
    
    # Resume interrupted training
    python scripts/train_yolo_a100.py --resume
    
    # Quick test (3 epochs)
    python scripts/train_yolo_a100.py --test

Presets:
    fast      - yolo11s, 100 epochs, batch=64, ~10-15 min
    balanced  - yolo11m, 150 epochs, batch=48, ~20-25 min (default)
    accurate  - yolo11m, 200 epochs, batch=32, imgsz=800, ~30-40 min
    best      - yolo11l, 300 epochs, batch=24, imgsz=800, ~60-80 min
    ultimate  - yolo11x, 400 epochs, batch=16, imgsz=1024, ~2-3 hours
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/datasets/merged_proctoring/data.yaml",
        help="Path to data.yaml file"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "accurate", "best", "ultimate"],
        default="balanced",
        help="Training preset (default: balanced)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test run (3 epochs)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="proctoring_yolo11",
        help="Project name for saving runs"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install dependencies before training"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 YOLO11 Training Script - Optimized for A100 64GB GPU")
    print("=" * 60)
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"\n📂 Working directory: {project_root}")
    
    # Install dependencies if requested
    if args.install:
        install_dependencies()
    
    # Check if data exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n❌ Data file not found: {data_path}")
        print("\n   Run the download script first:")
        print("   python scripts/download_and_prepare.py")
        return
    
    # Start training
    train_yolo(
        data_path=str(data_path),
        preset=args.preset,
        resume=args.resume,
        test_run=args.test,
        project_name=args.project,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
