#!/usr/bin/env python3
"""
Download and Prepare Dataset for YOLO11 Training

This script:
1. Downloads all datasets from Roboflow
2. Merges them into unified format
3. Creates data.yaml ready for training

Usage:
    python scripts/download_and_prepare.py

Output:
    data/datasets/merged_proctoring/
    ├── train/images/
    ├── train/labels/
    ├── valid/images/
    ├── valid/labels/
    ├── test/images/
    ├── test/labels/
    └── data.yaml
"""

import os
import shutil
import yaml
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

ROBOFLOW_API_KEY = "WYYAK1sLCjjwaXhv82T3"

# Datasets to download
DATASETS_TO_DOWNLOAD = [
    {
        "name": "cheating_devices",
        "workspace": "sumanth-kumar-c",
        "project": "cheating-detection-qc4rk",
        "description": "earphone, mobile_phone, smartwatch, peeking"
    },
    {
        "name": "classroom_cheating",
        "workspace": "classrom-assistance",
        "project": "cheating-detection-u2v47",
        "description": "cell phone, talking, normal"
    },
    {
        "name": "mobile_phone",
        "workspace": "tusker-ai",
        "project": "mobile-phone-detection-2vads",
        "description": "mobile phone detection - large dataset"
    },
    {
        "name": "phone_in_hand",
        "workspace": "phone-in-hand-detection",
        "project": "phone-in-hand-detection",
        "description": "phone specifically in hand"
    },
]

# Unified classes for interview proctoring
UNIFIED_CLASSES = [
    'phone',           # 0 - any phone (in hand, on desk)
    'earphone',        # 1 - earbuds, headphones
    'smartwatch',      # 2 - smartwatch
    'person_talking',  # 3 - person talking/cheating behavior
    'peeking',         # 4 - looking away suspiciously  
    'hand_gesture',    # 5 - suspicious hand movement
    'paper',           # 6 - notes/paper
]

# Class remapping per dataset
DATASET_MAPPINGS = {
    'cheating_devices': {
        'earphone': 1,
        'mobile_phone': 0,
        'smartwatch': 2,
        'peeking': 4,
        'hand_gestures': 5,
        'paper_passing': 6,
        'head_rotation': 4,
        'normal': None,  # skip
    },
    'classroom_cheating': {
        'cell phone': 0,
        'Talking': 3,
        'Normal': None,  # skip
    },
    'mobile_phone': {
        '0': 0,
        '1': 0,
    },
    'phone_in_hand': {
        'Phone in hand detection - v8 2024-11-09 6-19am': 0,
    },
}


def install_roboflow():
    """Install roboflow if not present"""
    try:
        import roboflow
        print("✅ roboflow already installed")
    except ImportError:
        print("📦 Installing roboflow...")
        os.system("pip install roboflow -q")


def download_datasets(output_dir: Path):
    """Download all datasets from Roboflow"""
    from roboflow import Roboflow
    
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    for dataset_info in DATASETS_TO_DOWNLOAD:
        name = dataset_info["name"]
        workspace = dataset_info["workspace"]
        project_name = dataset_info["project"]
        desc = dataset_info["description"]
        
        dataset_dir = output_dir / name
        
        # Skip if already downloaded
        if dataset_dir.exists() and (dataset_dir / "data.yaml").exists():
            print(f"⏭️  {name} already downloaded, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"📥 Downloading: {name}")
        print(f"   {desc}")
        print(f"{'='*60}")
        
        try:
            project = rf.workspace(workspace).project(project_name)
            versions = project.versions()
            
            if versions:
                print(f"   Found {len(versions)} versions, using latest...")
                dataset = versions[0].download("yolov8", location=str(dataset_dir))
                print(f"✅ Downloaded {name}!")
            else:
                print(f"❌ No versions found for {name}")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")


def remap_labels(src_dir: Path, dst_dir: Path, class_map: dict, original_classes: list, prefix: str) -> int:
    """Remap labels from original classes to unified classes"""
    src_labels = src_dir / 'labels'
    src_images = src_dir / 'images'
    
    if not src_labels.exists() or not src_images.exists():
        return 0
    
    count = 0
    for img_file in src_images.glob('*'):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        
        label_file = src_labels / f'{img_file.stem}.txt'
        
        # Copy image with prefix to avoid duplicates
        dst_img = dst_dir / 'images' / f'{prefix}_{img_file.name}'
        shutil.copy(img_file, dst_img)
        
        # Remap labels
        new_labels = []
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_class_id = int(parts[0])
                        bbox = parts[1:5]
                        
                        if orig_class_id < len(original_classes):
                            orig_class = original_classes[orig_class_id]
                            new_class_id = class_map.get(orig_class)
                            if new_class_id is not None:
                                new_labels.append(f'{new_class_id} {" ".join(bbox)}')
        
        dst_label = dst_dir / 'labels' / f'{prefix}_{img_file.stem}.txt'
        with open(dst_label, 'w') as f:
            f.write('\n'.join(new_labels))
        
        count += 1
    return count


def merge_datasets(datasets_dir: Path, output_dir: Path):
    """Merge all datasets into unified format"""
    print(f"\n{'='*60}")
    print("🔀 Merging datasets...")
    print(f"{'='*60}")
    
    # Clean and create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Create directory structure
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    stats = {}
    
    # Process each dataset
    for dataset_name, class_map in DATASET_MAPPINGS.items():
        dataset_dir = datasets_dir / dataset_name
        if not dataset_dir.exists():
            print(f"⚠️  {dataset_name} not found, skipping...")
            continue
        
        # Load original class names
        data_yaml_path = dataset_dir / 'data.yaml'
        if not data_yaml_path.exists():
            print(f"⚠️  No data.yaml for {dataset_name}, skipping...")
            continue
        
        with open(data_yaml_path) as f:
            config = yaml.safe_load(f)
        original_classes = config.get('names', [])
        
        print(f"\n📦 Processing {dataset_name}...")
        print(f"   Original classes: {original_classes}")
        
        dataset_total = 0
        for split in ['train', 'valid', 'test']:
            src_split = dataset_dir / split
            if src_split.exists():
                count = remap_labels(
                    src_split,
                    output_dir / split,
                    class_map,
                    original_classes,
                    dataset_name[:10]  # prefix
                )
                if count > 0:
                    print(f"   {split}: {count} images")
                dataset_total += count
        
        stats[dataset_name] = dataset_total
        total_images += dataset_total
    
    # Create data.yaml for training
    data_yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(UNIFIED_CLASSES),
        'names': UNIFIED_CLASSES,
    }
    
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("✅ DATASET READY FOR TRAINING!")
    print(f"{'='*60}")
    print(f"\n📊 Summary:")
    for name, count in stats.items():
        print(f"   {name}: {count} images")
    print(f"\n   TOTAL: {total_images} images")
    print(f"\n📁 Output: {output_dir}")
    print(f"📄 Config: {output_dir}/data.yaml")
    print(f"\n🏷️  Classes:")
    for i, cls in enumerate(UNIFIED_CLASSES):
        print(f"   {i}: {cls}")
    
    return total_images


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     PROCTORING DATASET DOWNLOAD & PREPARATION SCRIPT          ║
║                                                               ║
║  This will download ~30,000+ labeled images for training      ║
║  YOLO11 to detect: phone, earphone, smartwatch, etc.          ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "data" / "datasets"
    merged_dir = datasets_dir / "merged_proctoring"
    
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Install roboflow
    print("\n📦 Step 1: Checking dependencies...")
    install_roboflow()
    
    # Step 2: Download datasets
    print("\n📥 Step 2: Downloading datasets from Roboflow...")
    download_datasets(datasets_dir)
    
    # Step 3: Merge datasets
    print("\n🔀 Step 3: Merging datasets...")
    total = merge_datasets(datasets_dir, merged_dir)
    
    # Final instructions
    print(f"""
{'='*60}
🚀 NEXT STEP: START TRAINING
{'='*60}

Run this command to train YOLO11:

  # On Mac M4 (MPS):
  yolo detect train data={merged_dir}/data.yaml model=yolo11s.pt epochs=100 imgsz=640 device=mps batch=16

  # On NVIDIA GPU:
  yolo detect train data={merged_dir}/data.yaml model=yolo11s.pt epochs=100 imgsz=640 device=0 batch=64

  # On A100 64GB (fastest):
  yolo detect train data={merged_dir}/data.yaml model=yolo11m.pt epochs=100 imgsz=640 device=0 batch=64

{'='*60}
    """)


if __name__ == "__main__":
    main()
