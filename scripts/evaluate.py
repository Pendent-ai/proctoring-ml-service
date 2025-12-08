"""
Model Evaluation Script
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
import numpy as np


def evaluate_yolo(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.5,
    iou: float = 0.5,
    device: str = "0",
):
    """
    Evaluate YOLOv8 model on dataset.
    
    Args:
        model_path: Path to model weights
        data_yaml: Path to dataset YAML
        split: Dataset split to evaluate (train, val, test)
        imgsz: Image size
        conf: Confidence threshold
        iou: IoU threshold
        device: Device to use
    """
    print(f"📊 Evaluating model: {model_path}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Split: {split}")
    
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        plots=True,
        save_json=True,
    )
    
    # Print metrics
    print("\n📈 Evaluation Results:")
    print(f"   mAP50: {results.box.map50:.4f}")
    print(f"   mAP50-95: {results.box.map:.4f}")
    print(f"   Precision: {np.mean(results.box.p):.4f}")
    print(f"   Recall: {np.mean(results.box.r):.4f}")
    
    # Per-class metrics
    if hasattr(results.box, 'ap_class_index'):
        print("\n📋 Per-Class Results:")
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = results.names[cls_idx]
            ap50 = results.box.ap50[i]
            print(f"   {cls_name}: mAP50={ap50:.4f}")
    
    return results


def benchmark_speed(
    model_path: str,
    imgsz: int = 640,
    device: str = "0",
):
    """Benchmark model inference speed."""
    import time
    
    print(f"\n⚡ Benchmarking speed...")
    
    model = YOLO(model_path)
    
    # Warmup
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        model(dummy, verbose=False)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model(dummy, verbose=False)
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000
    fps = 1000 / avg_time
    
    print(f"   Average inference: {avg_time:.2f}ms")
    print(f"   FPS: {fps:.1f}")
    
    return avg_time, fps


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    if not Path(args.data).exists():
        print(f"❌ Dataset YAML not found: {args.data}")
        return
    
    # Evaluate
    evaluate_yolo(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    
    # Benchmark
    if args.benchmark:
        benchmark_speed(args.model, args.imgsz, args.device)


if __name__ == "__main__":
    main()
