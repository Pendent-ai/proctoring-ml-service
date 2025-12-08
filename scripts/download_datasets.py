#!/usr/bin/env python3
"""
Dataset Downloader for AI Interview Proctoring

Downloads labeled datasets for training models for LIVE INTERVIEW monitoring:

PRIORITY OBJECTS FOR INTERVIEW PROCTORING:
==========================================
1. Phone detection (in hand, on desk)
2. Earbuds/Headphones/AirPods  
3. Smartwatch
4. Calculator
5. Notes/Cheat sheets
6. Second person in frame
7. Second screen (monitor)

DATASETS TO DOWNLOAD:
====================
- Phone: ~20,000 images from multiple sources
- Cheating objects: ~10,000 images (earbuds, smartwatch, etc.)
- Face detection: Use pretrained (MediaPipe)
- Person detection: Use pretrained (YOLO COCO)

Usage:
    # With Roboflow API key (RECOMMENDED):
    python download_datasets.py --api-key YOUR_ROBOFLOW_API_KEY --category all

    # Download specific category:
    python download_datasets.py --api-key KEY --category phone
    python download_datasets.py --api-key KEY --category cheating_objects
    
    # List available datasets:
    python download_datasets.py --list
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error


# Dataset registry with Roboflow workspace/project info
DATASETS = {
    # === PHONE DETECTION ===
    "phone_hi": {
        "workspace": "hi-wzisi",
        "project": "phone-du6hl",
        "version": 1,
        "images": 9720,
        "classes": ["phone"],
        "priority": 1,
        "description": "Large phone detection dataset"
    },
    "phone_person": {
        "workspace": "phonedt",
        "project": "phone-person",
        "version": 1,
        "images": 3120,
        "classes": ["phone"],
        "priority": 2,
        "description": "Phone with person context"
    },
    "mobile_phone_tusker": {
        "workspace": "tusker-ai",
        "project": "mobile-phone-detection-2vads",
        "version": 1,
        "images": 9900,
        "classes": ["phone"],
        "priority": 1,
        "description": "Large mobile phone dataset"
    },
    "phone_in_hand": {
        "workspace": "phone-in-hand-detection",
        "project": "phone-in-hand-detection",
        "version": 8,
        "images": 338,
        "classes": ["CellPhone", "phone_in_hand"],
        "priority": 3,
        "description": "Phone specifically in hand"
    },
    "hand_holding_phone": {
        "workspace": "data-annotation-pwjtq",
        "project": "detect-hand-holding-phone",
        "version": 5,
        "images": 983,
        "classes": ["hand_phone", "phone_usage"],
        "priority": 2,
        "description": "Hand holding phone detection"
    },
    
    # === CHEATING DETECTION ===
    "cheating_sumanth": {
        "workspace": "sumanth-kumar-c",
        "project": "cheating-detection-qc4rk",
        "version": 1,
        "images": 2070,
        "classes": ["normal", "earphone", "hand_gestures", "head_rotation", 
                   "mobile_phone", "paper_passing", "peeking", "smartwatch"],
        "priority": 1,
        "description": "Comprehensive cheating behaviors with devices"
    },
    "online_exam_cheating": {
        "workspace": "fraud-detection-using-cnn",
        "project": "online-exam-cheating-detection",
        "version": 1,
        "images": 7140,
        "classes": ["normal", "cheating"],
        "priority": 2,
        "description": "Large binary cheating/normal dataset"
    },
    "online_exam_cheating_2": {
        "workspace": "fraud-detection-using-cnn",
        "project": "online-exam-cheating-detection-2",
        "version": 1,
        "images": 8180,
        "classes": ["normal", "cheating"],
        "priority": 2,
        "description": "Extended cheating dataset"
    },
    "cheating_rakshitha": {
        "workspace": "rakshitha-beeranhalli-louuy",
        "project": "cheating-detection-wmju5",
        "version": 1,
        "images": 4690,
        "classes": ["book", "person", "phone"],
        "priority": 2,
        "description": "Books, person, phone in exam context"
    },
    "exam_cheating_datatrain": {
        "workspace": "datatrain-lbott",
        "project": "exam-cheating-9iz1y",
        "version": 1,
        "images": 3410,
        "classes": ["laptop", "mobile", "pen", "watch", "headphones"],
        "priority": 1,
        "description": "Multiple exam cheating objects"
    },
    
    # === ADDITIONAL USEFUL ===
    "cheating_unt": {
        "workspace": "unt",
        "project": "cheating-detection-ppnrs",
        "version": 1,
        "images": 50,
        "classes": ["apuntes", "auriculares", "celular", "lapicera", "persona"],
        "priority": 4,
        "description": "Spanish labels - notes, earphones, phone, pen, person"
    },
    "classroom_cheating": {
        "workspace": "classrom-assistance",
        "project": "cheating-detection-u2v47",
        "version": 1,
        "images": 2110,
        "classes": ["cell phone", "Normal", "Talking"],
        "priority": 2,
        "description": "Classroom setting cheating"
    },
    "exam_monitoring": {
        "workspace": "college-hhnwl",
        "project": "exam-cheating-315",
        "version": 1,
        "images": 315,
        "classes": ["examiner", "learning to copy", "looking around", 
                   "normal sitting", "talking", "using phone"],
        "priority": 3,
        "description": "Behavior classification in exam"
    },
}


# Class mapping to unified proctoring classes
CLASS_MAPPING = {
    # Phone variants -> phone_in_hand
    "phone": "phone_in_hand",
    "cellphone": "phone_in_hand",
    "mobile": "phone_in_hand",
    "mobile_phone": "phone_in_hand",
    "cell phone": "phone_in_hand",
    "celular": "phone_in_hand",
    "handphone": "phone_in_hand",
    "phone_in_hand": "phone_in_hand",
    "hand_phone": "phone_in_hand",
    "phone_usage": "phone_in_hand",
    "using phone": "phone_in_hand",
    
    # Earphones/headphones -> earbuds
    "earphone": "earbuds",
    "earphones": "earbuds",
    "headphones": "earbuds",
    "auriculares": "earbuds",
    
    # Smartwatch
    "smartwatch": "smartwatch",
    "watch": "smartwatch",
    
    # Notes/books
    "book": "notes",
    "apuntes": "notes",
    "paper_passing": "notes",
    
    # Another person
    "person": "another_person",
    "persona": "another_person",
    
    # Laptop
    "laptop": "laptop",
    
    # Behaviors (for classification training)
    "normal": "normal",
    "cheating": "cheating",
    "peeking": "looking_away",
    "looking around": "looking_away",
    "head_rotation": "looking_away",
    "learning to copy": "cheating",
    "talking": "talking_to_someone",
}


# Final unified classes
UNIFIED_CLASSES = [
    "phone_in_hand",   # 0
    "earbuds",         # 1
    "smartwatch",      # 2
    "notes",           # 3
    "another_person",  # 4
    "laptop",          # 5
    "normal",          # 6 (for classification)
    "cheating",        # 7 (for classification)
    "looking_away",    # 8
    "talking_to_someone"  # 9
]


class DatasetDownloader:
    """Downloads and processes datasets from Roboflow"""
    
    def __init__(
        self,
        output_dir: str = "data/training",
        api_key: Optional[str] = None,
        format: str = "yolov8"
    ):
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.format = format
        
        # Create directories
        self.downloads_dir = self.output_dir / "downloads"
        self.merged_dir = self.output_dir / "merged"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "downloaded": 0,
            "failed": 0,
            "total_images": 0,
            "class_counts": {}
        }
    
    def download_all(
        self,
        datasets: Optional[List[str]] = None,
        max_priority: int = 3
    ) -> Dict:
        """
        Download all or selected datasets.
        
        Args:
            datasets: List of dataset keys to download. If None, downloads all.
            max_priority: Only download datasets with priority <= this value
        """
        if datasets is None:
            datasets = [
                k for k, v in DATASETS.items() 
                if v.get("priority", 5) <= max_priority
            ]
        
        print(f"\n📥 Downloading {len(datasets)} datasets...")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Format: {self.format}\n")
        
        for dataset_key in datasets:
            if dataset_key not in DATASETS:
                print(f"⚠️  Unknown dataset: {dataset_key}")
                continue
            
            dataset_info = DATASETS[dataset_key]
            print(f"\n{'='*60}")
            print(f"📦 {dataset_key}: {dataset_info['description']}")
            print(f"   Images: ~{dataset_info['images']}")
            print(f"   Classes: {dataset_info['classes']}")
            
            try:
                self._download_dataset(dataset_key, dataset_info)
                self.stats["downloaded"] += 1
            except Exception as e:
                print(f"❌ Failed to download {dataset_key}: {e}")
                self.stats["failed"] += 1
        
        print(f"\n{'='*60}")
        print(f"✅ Download complete!")
        print(f"   Downloaded: {self.stats['downloaded']}")
        print(f"   Failed: {self.stats['failed']}")
        
        return self.stats
    
    def _download_dataset(self, key: str, info: Dict):
        """Download a single dataset"""
        dataset_dir = self.downloads_dir / key
        
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"   ⏭️  Already downloaded, skipping...")
            return
        
        if self.api_key:
            self._download_with_api(key, info, dataset_dir)
        else:
            self._download_public(key, info, dataset_dir)
    
    def _download_with_api(self, key: str, info: Dict, output_dir: Path):
        """Download using Roboflow API"""
        try:
            from roboflow import Roboflow
        except ImportError:
            print("   ⚠️  roboflow package not installed. Installing...")
            os.system("pip install roboflow -q")
            from roboflow import Roboflow
        
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(info["workspace"]).project(info["project"])
        version = project.version(info["version"])
        
        print(f"   ⬇️  Downloading via API...")
        dataset = version.download(self.format, location=str(output_dir))
        print(f"   ✅ Downloaded to {output_dir}")
    
    def _download_public(self, key: str, info: Dict, output_dir: Path):
        """Download from public URL if available"""
        # Construct Roboflow public download URL
        # Note: This only works for public datasets
        workspace = info["workspace"]
        project = info["project"]
        version = info["version"]
        
        url = f"https://universe.roboflow.com/ds/{workspace}/{project}/{version}/{self.format}"
        
        print(f"   ⬇️  Attempting public download...")
        print(f"   📌 For reliable downloads, use --api-key option")
        print(f"   📌 Get free API key at: https://app.roboflow.com/settings/api")
        
        # Create placeholder with instructions
        output_dir.mkdir(parents=True, exist_ok=True)
        instructions = output_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions, 'w') as f:
            f.write(f"""Dataset: {key}
Workspace: {workspace}
Project: {project}
Version: {version}

To download this dataset:

1. Go to: https://universe.roboflow.com/{workspace}/{project}
2. Click "Download Dataset"
3. Select format: {self.format}
4. Download and extract to this folder

Or use the Roboflow API:
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("{workspace}").project("{project}")
    version = project.version({version})
    dataset = version.download("{self.format}")
""")
        print(f"   📝 Created download instructions at {instructions}")
    
    def merge_datasets(self) -> Path:
        """
        Merge all downloaded datasets into unified training set.
        
        Returns:
            Path to merged dataset
        """
        print(f"\n🔀 Merging datasets...")
        
        # Create merged structure
        train_images = self.merged_dir / "train" / "images"
        train_labels = self.merged_dir / "train" / "labels"
        val_images = self.merged_dir / "valid" / "images"
        val_labels = self.merged_dir / "valid" / "labels"
        
        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)
        
        total_images = 0
        class_counts = {c: 0 for c in UNIFIED_CLASSES}
        
        # Process each downloaded dataset
        for dataset_dir in self.downloads_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_key = dataset_dir.name
            if dataset_key not in DATASETS:
                continue
            
            print(f"\n   Processing {dataset_key}...")
            
            # Find train and valid folders
            for split in ["train", "valid", "test"]:
                split_dir = dataset_dir / split
                if not split_dir.exists():
                    continue
                
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"
                
                if not images_dir.exists():
                    continue
                
                # Determine target split (test -> valid)
                target_split = "train" if split == "train" else "valid"
                target_images = train_images if target_split == "train" else val_images
                target_labels = train_labels if target_split == "train" else val_labels
                
                # Get original class mapping
                data_yaml = dataset_dir / "data.yaml"
                original_classes = self._load_classes(data_yaml)
                
                # Copy and remap
                for img_path in images_dir.glob("*"):
                    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                        continue
                    
                    # Generate unique filename
                    new_name = f"{dataset_key}_{img_path.name}"
                    
                    # Copy image
                    shutil.copy(img_path, target_images / new_name)
                    
                    # Process and copy label
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        new_labels = self._remap_labels(
                            label_path, original_classes, class_counts
                        )
                        new_label_path = target_labels / f"{dataset_key}_{img_path.stem}.txt"
                        with open(new_label_path, 'w') as f:
                            f.write('\n'.join(new_labels))
                    
                    total_images += 1
        
        # Create data.yaml
        data_yaml = self.merged_dir / "data.yaml"
        with open(data_yaml, 'w') as f:
            f.write(f"""# Merged Proctoring Dataset
# Generated by download_datasets.py

path: {self.merged_dir.absolute()}
train: train/images
val: valid/images

nc: {len(UNIFIED_CLASSES)}
names: {UNIFIED_CLASSES}

# Class mapping used:
# 0: phone_in_hand - Any phone being held
# 1: earbuds - Earphones, headphones, airpods
# 2: smartwatch - Smartwatches, watches
# 3: notes - Books, papers, notes
# 4: another_person - Additional people in frame
# 5: laptop - Laptops, notebooks
# 6: normal - Normal behavior (classification)
# 7: cheating - Cheating behavior (classification)
# 8: looking_away - Looking away, peeking
# 9: talking_to_someone - Talking to someone else
""")
        
        # Save statistics
        stats = {
            "total_images": total_images,
            "train_images": len(list(train_images.glob("*"))),
            "val_images": len(list(val_images.glob("*"))),
            "class_counts": class_counts
        }
        
        stats_path = self.merged_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Merged dataset created!")
        print(f"   Location: {self.merged_dir}")
        print(f"   Total images: {total_images}")
        print(f"   Train: {stats['train_images']}")
        print(f"   Valid: {stats['val_images']}")
        print(f"\n   Class distribution:")
        for cls, count in class_counts.items():
            if count > 0:
                print(f"      {cls}: {count}")
        
        return self.merged_dir
    
    def _load_classes(self, data_yaml: Path) -> List[str]:
        """Load class names from data.yaml"""
        if not data_yaml.exists():
            return []
        
        try:
            import yaml
            with open(data_yaml) as f:
                data = yaml.safe_load(f)
            return data.get("names", [])
        except:
            return []
    
    def _remap_labels(
        self,
        label_path: Path,
        original_classes: List[str],
        class_counts: Dict[str, int]
    ) -> List[str]:
        """Remap labels from original classes to unified classes"""
        new_labels = []
        
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                orig_class_id = int(parts[0])
                bbox = parts[1:5]
                
                # Get original class name
                if orig_class_id < len(original_classes):
                    orig_class = original_classes[orig_class_id].lower()
                else:
                    continue
                
                # Map to unified class
                unified_class = CLASS_MAPPING.get(orig_class)
                if unified_class is None:
                    continue
                
                # Get new class ID
                if unified_class in UNIFIED_CLASSES:
                    new_class_id = UNIFIED_CLASSES.index(unified_class)
                    new_labels.append(f"{new_class_id} {' '.join(bbox)}")
                    class_counts[unified_class] = class_counts.get(unified_class, 0) + 1
        
        return new_labels


def main():
    parser = argparse.ArgumentParser(
        description="Download labeled datasets for proctoring ML training"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key. Get free at https://app.roboflow.com/settings/api"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/training",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="yolov8",
        choices=["yolov8", "yolov5", "yolov11", "coco"],
        help="Dataset format"
    )
    parser.add_argument(
        "--datasets", "-d",
        type=str,
        nargs="+",
        help="Specific datasets to download (default: all priority 1-3)"
    )
    parser.add_argument(
        "--priority", "-p",
        type=int,
        default=3,
        help="Max priority level to download (1=highest, 4=lowest)"
    )
    parser.add_argument(
        "--merge", "-m",
        action="store_true",
        help="Merge downloaded datasets after download"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Use public download (creates instructions if API unavailable)"
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("\n📋 Available Datasets:\n")
        print(f"{'Key':<25} {'Images':>8} {'Priority':>8}  Classes")
        print("-" * 80)
        for key, info in sorted(DATASETS.items(), key=lambda x: x[1].get("priority", 5)):
            classes = ", ".join(info["classes"][:3])
            if len(info["classes"]) > 3:
                classes += f"... (+{len(info['classes'])-3})"
            print(f"{key:<25} {info['images']:>8} {info.get('priority', 5):>8}  {classes}")
        print(f"\nTotal: {sum(d['images'] for d in DATASETS.values())} images across {len(DATASETS)} datasets")
        return
    
    # Check API key
    if not args.api_key and not args.public:
        print("\n⚠️  No Roboflow API key provided!")
        print("   Get a free API key at: https://app.roboflow.com/settings/api")
        print("\n   Options:")
        print("   1. Set ROBOFLOW_API_KEY environment variable")
        print("   2. Use --api-key YOUR_KEY")
        print("   3. Use --public for manual download instructions")
        print("\n   Example:")
        print("   python download_datasets.py --api-key rf_xxxxxxxxxxxx")
        return
    
    # Download
    downloader = DatasetDownloader(
        output_dir=args.output,
        api_key=args.api_key if not args.public else None,
        format=args.format
    )
    
    stats = downloader.download_all(
        datasets=args.datasets,
        max_priority=args.priority
    )
    
    # Merge if requested
    if args.merge:
        downloader.merge_datasets()


if __name__ == "__main__":
    main()
