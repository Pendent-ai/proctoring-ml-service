#!/usr/bin/env python3
"""
Dataset Downloader for AI Interview Proctoring

Downloads labeled datasets for training models for LIVE AI INTERVIEW monitoring:

PRIORITY OBJECTS FOR INTERVIEW PROCTORING:
==========================================
1. Phone detection (in hand, on desk) - ~50,000+ images
2. Earbuds/Headphones/AirPods - ~6,000+ images  
3. Smartwatch - ~3,000+ images
4. Calculator - ~1,000+ images
5. Notes/Cheat sheets/Books - ~5,000+ images
6. Second person in frame - ~45,000+ images
7. Second screen (TV/monitor) - ~2,000+ images
8. Gaze direction (looking away) - ~12,000+ images
9. Hand gestures (suspicious) - ~4,000+ images

TOTAL AVAILABLE: ~130,000+ IMAGES across 30+ datasets!

NEW DATASETS ADDED:
==================
✅ online_proctoring_system - 9.9K images (gaze + phone)
✅ headphones_wired_wireless - 1.15K (wired vs wireless)
✅ headphones_person - 1.42K (person wearing headphones)
✅ cheating_cheatingways - 10.9K (laptop, mobile, pen, watch, headphones)
✅ cheating_zeina - 9.46K (binary cheating detection)
✅ kidsentry_master - 26.8K (adult/child + objects)
✅ ddv2_person - 17.9K (multiple objects around person)
✅ fyp_exam_proctoring - Extra person detection (CRITICAL)
✅ And many more...

Usage:
    # With Roboflow API key (RECOMMENDED):
    python download_datasets.py --api-key YOUR_ROBOFLOW_API_KEY --category all

    # Download specific category:
    python download_datasets.py --api-key KEY --category phone
    python download_datasets.py --api-key KEY --category earbuds
    python download_datasets.py --api-key KEY --category interview
    python download_datasets.py --api-key KEY --category cheating
    python download_datasets.py --api-key KEY --category gaze
    
    # Download only priority 1 datasets (largest/most useful):
    python download_datasets.py --api-key KEY --priority 1
    
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

# ============================================
# CONFIGURATION - Roboflow API Key
# ============================================
# Get your own at: https://app.roboflow.com/settings/api
ROBOFLOW_API_KEY = "WYYAK1sLCjjwaXhv82T3"


# Dataset registry with Roboflow workspace/project info
DATASETS = {
    # ============================================================
    # === PHONE DETECTION (Essential for interview monitoring) ===
    # ============================================================
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
    
    # ============================================================
    # === COMPREHENSIVE CHEATING DETECTION ===
    # ============================================================
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
    "classroom_cheating": {
        "workspace": "classrom-assistance",
        "project": "cheating-detection-u2v47",
        "version": 1,
        "images": 2110,
        "classes": ["cell phone", "Normal", "Talking"],
        "priority": 2,
        "description": "Classroom setting cheating"
    },
    
    # ============================================================
    # === NEW: ONLINE PROCTORING SPECIFIC (AI Interview) ===
    # ============================================================
    "online_proctoring_system": {
        "workspace": "project-2morrow-software-limited",
        "project": "online-proctoring-system-x27ou",
        "version": 1,
        "images": 9900,
        "classes": ["face", "looking_down", "looking_left", "looking_right", 
                   "looking_straight", "looking_up", "mobile_phone"],
        "priority": 1,
        "category": "interview",
        "description": "⭐ LARGE - Gaze detection + phone for ONLINE interviews"
    },
    "online_proctoring_deep_learning": {
        "workspace": "deep-learning-jdjnf",
        "project": "online-proctoring-system",
        "version": 1,
        "images": 1970,
        "classes": ["book", "cell_phone", "laptop", "person", "headphone", "tv"],
        "priority": 1,
        "category": "interview",
        "description": "⭐ Laptop, headphones, TV/second screen detection"
    },
    "fyp_exam_proctoring": {
        "workspace": "fyp-lyzmw",
        "project": "fyp-exam-proctoring-robot",
        "version": 1,
        "images": 473,
        "classes": ["book", "laptop", "phone", "extra_person", "student"],
        "priority": 2,
        "category": "interview",
        "description": "Extra person detection - CRITICAL for interviews"
    },
    "proctoring_obj": {
        "workspace": "proctoring-pjt1g",
        "project": "proctoring-obj",
        "version": 1,
        "images": 548,
        "classes": ["bottle", "mobile_phone", "notebook", "pen", "phone", 
                   "watch", "ipad"],
        "priority": 2,
        "category": "interview",
        "description": "iPad/tablet and notebook detection"
    },
    "proctoring_samya": {
        "workspace": "samya-eiiy9",
        "project": "proctoring-rzkos",
        "version": 1,
        "images": 782,
        "classes": ["Kertas", "Menoleh", "Menunduk", "Smartphone", "Tidak_Ada_Kecurangan"],
        "priority": 3,
        "category": "interview",
        "description": "Looking away (Menoleh), head down (Menunduk)"
    },
    "exam_proctoring_urhzv": {
        "workspace": "fyp-lyzmw",
        "project": "exam-proctoring-urhzv",
        "version": 1,
        "images": 463,
        "classes": ["phone", "cheating", "not_cheating", "note_passing", 
                   "objects", "possible_cheating"],
        "priority": 3,
        "category": "interview",
        "description": "Cheating likelihood classification"
    },
    
    # ============================================================
    # === NEW: HEADPHONES/EARBUDS DETECTION (Interview Critical) ===
    # ============================================================
    "headphones_wired_wireless": {
        "workspace": "erzhan",
        "project": "headphones-yp3kq",
        "version": 1,
        "images": 1150,
        "classes": ["wired_headphones", "wireless"],
        "priority": 1,
        "category": "earbuds",
        "description": "⭐ Wired vs Wireless headphones classification"
    },
    "headphones_large": {
        "workspace": "headphones-9uy0k",
        "project": "headphones-fwhbt",
        "version": 1,
        "images": 1420,
        "classes": ["headphones"],
        "priority": 1,
        "category": "earbuds",
        "description": "⭐ Large headphones-only dataset"
    },
    "headphones_person": {
        "workspace": "swati-karot-kti9y",
        "project": "headphones-1etei",
        "version": 1,
        "images": 1420,
        "classes": ["person_with_headphone", "person_without_headphone"],
        "priority": 1,
        "category": "earbuds",
        "description": "⭐ Person wearing headphones detection"
    },
    "headphone_earphone_v2": {
        "workspace": "just-a-guy-bqxzq",
        "project": "headphone-earphone-detection-v2",
        "version": 1,
        "images": 459,
        "classes": ["not_wearing", "wearing_earphones", "wearing_headphones"],
        "priority": 2,
        "category": "earbuds",
        "description": "Earphones vs Headphones distinction"
    },
    "pen_calculator_earbuds": {
        "workspace": "aquamanj",
        "project": "pen-calculator-earbuds-ecdso",
        "version": 1,
        "images": 947,
        "classes": ["Calculator", "Ear_Device", "Pen", "earbuds", "objects"],
        "priority": 2,
        "category": "earbuds",
        "description": "Earbuds + Calculator + Pen detection"
    },
    
    # ============================================================
    # === NEW: GAZE/LOOKING DIRECTION DETECTION ===
    # ============================================================
    "gaze_direction": {
        "workspace": "fraud-detection-using-cnn",
        "project": "cheating-detection-ohiq5",
        "version": 1,
        "images": 197,
        "classes": ["look_down", "look_forward", "look_left", "look_right", 
                   "look_up", "mouth_close", "mouth_open"],
        "priority": 2,
        "category": "gaze",
        "description": "Gaze + mouth detection for interview monitoring"
    },
    "looking_direction": {
        "workspace": "project-6jgqt",
        "project": "face-detection-slqcm",
        "version": 1,
        "images": 2060,
        "classes": ["Back", "Front", "Left", "FrontRight", "Right", "Phone"],
        "priority": 2,
        "category": "gaze",
        "description": "Face orientation with phone detection"
    },
    
    # ============================================================
    # === NEW: CHEATING BEHAVIORS & GESTURES ===
    # ============================================================
    "cheating_cheatingways": {
        "workspace": "cheatingways",
        "project": "cheating-zkje8",
        "version": 1,
        "images": 10900,
        "classes": ["laptop", "mobile", "pen", "watch", "headphones"],
        "priority": 1,
        "category": "cheating",
        "description": "⭐ MASSIVE - All cheating objects combined"
    },
    "cheating_zeina": {
        "workspace": "zeina-niyuf",
        "project": "cheating-aswze",
        "version": 1,
        "images": 9460,
        "classes": ["normal", "cheating"],
        "priority": 1,
        "category": "cheating",
        "description": "⭐ LARGE binary classification"
    },
    "cheating_jo": {
        "workspace": "jo-qyhte",
        "project": "cheating-gjiev",
        "version": 1,
        "images": 5760,
        "classes": ["normal", "phone"],
        "priority": 2,
        "category": "cheating",
        "description": "Large normal vs phone usage"
    },
    "cheating_system_3k": {
        "workspace": "chetaing-detection",
        "project": "cheating-detection-system-oy177",
        "version": 1,
        "images": 3550,
        "classes": ["Hand_Gestures", "Passing_of_Paper", "Peeking", "Posture"],
        "priority": 2,
        "category": "cheating",
        "description": "Gestures, paper passing, peeking, posture"
    },
    "cheating_analyze": {
        "workspace": "analyze",
        "project": "cheating-detection-obw3c",
        "version": 1,
        "images": 2790,
        "classes": ["Fokus", "Mencontek"],
        "priority": 3,
        "category": "cheating",
        "description": "Focused vs Cheating (Indonesian)"
    },
    
    # ============================================================
    # === NEW: EXTRA PERSON / MULTIPLE PEOPLE DETECTION ===
    # ============================================================
    "kidsentry_master": {
        "workspace": "thesis-meyy8",
        "project": "kidsentry_master_dataset-twnfr",
        "version": 1,
        "images": 26800,
        "classes": ["book", "cup", "knife", "potted_plant", "remote", 
                   "scissors", "adult", "child", "outlet"],
        "priority": 2,
        "category": "person",
        "description": "⭐ MASSIVE - Adult/child detection + objects"
    },
    "ddv2_person": {
        "workspace": "ddv2",
        "project": "ddv2-zeg5e",
        "version": 1,
        "images": 17900,
        "classes": ["backpack", "book", "door", "mobile_phone", "notebook", 
                   "pen", "plastic_bottle"],
        "priority": 2,
        "category": "person",
        "description": "⭐ LARGE - Multiple objects around person"
    },
    
    # ============================================================
    # === NEW: CALCULATOR DETECTION (Interview cheating) ===
    # ============================================================
    "calculator_detection": {
        "workspace": "aquamanj",
        "project": "pen-calculator-earbuds-ecdso",
        "version": 1,
        "images": 947,
        "classes": ["Calculator", "Ear_Device", "Pen"],
        "priority": 2,
        "category": "calculator",
        "description": "Calculator + Pen + Earbuds"
    },
    
    # ============================================================
    # === EXISTING: ADDITIONAL USEFUL ===
    # ============================================================
    "cheating_unt": {
        "workspace": "unt",
        "project": "cheating-detection-ppnrs",
        "version": 1,
        "images": 50,
        "classes": ["apuntes", "auriculares", "celular", "lapicera", "persona"],
        "priority": 4,
        "description": "Spanish labels - notes, earphones, phone, pen, person"
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


# Class mapping to unified proctoring classes for AI Interview Monitoring
CLASS_MAPPING = {
    # === Phone variants -> phone ===
    "phone": "phone",
    "cellphone": "phone",
    "mobile": "phone",
    "mobile_phone": "phone",
    "cell phone": "phone",
    "cell_phone": "phone",
    "celular": "phone",
    "handphone": "phone",
    "phone_in_hand": "phone",
    "hand_phone": "phone",
    "phone_usage": "phone",
    "using phone": "phone",
    "smartphone": "phone",
    
    # === Earphones/headphones -> earbuds ===
    "earphone": "earbuds",
    "earphones": "earbuds",
    "headphones": "earbuds",
    "headphone": "earbuds",
    "auriculares": "earbuds",
    "ear_device": "earbuds",
    "ear_piece": "earbuds",
    "wired_headphones": "earbuds",
    "wireless": "earbuds",
    "person_with_headphone": "earbuds",
    "wearing_earphones": "earbuds",
    "wearing_headphones": "earbuds",
    
    # === Smartwatch ===
    "smartwatch": "smartwatch",
    "watch": "smartwatch",
    
    # === Notes/books/paper ===
    "book": "notes",
    "apuntes": "notes",
    "paper_passing": "notes",
    "notebook": "notes",
    "kertas": "notes",  # Indonesian for paper
    "passing_of_paper": "notes",
    
    # === Another person (CRITICAL for interviews) ===
    "person": "another_person",
    "persona": "another_person",
    "extra_person": "another_person",
    "student": "another_person",
    "adult": "another_person",
    "child": "another_person",
    
    # === Laptop/tablet/second screen ===
    "laptop": "laptop",
    "ipad": "laptop",
    "tv": "second_screen",
    
    # === Calculator ===
    "calculator": "calculator",
    
    # === Pen (less critical but useful) ===
    "pen": "pen",
    "lapicera": "pen",
    
    # === Gaze direction (CRITICAL for interview monitoring) ===
    "looking_down": "looking_away",
    "looking_left": "looking_away",
    "looking_right": "looking_away",
    "looking_up": "looking_away",
    "look_down": "looking_away",
    "look_left": "looking_away",
    "look_right": "looking_away",
    "look_up": "looking_away",
    "menoleh": "looking_away",  # Indonesian: looking away
    "menunduk": "looking_away",  # Indonesian: head down
    "back": "looking_away",
    "left": "looking_away",
    "right": "looking_away",
    "looking_straight": "looking_forward",
    "looking_forward": "looking_forward",
    "look_forward": "looking_forward",
    "front": "looking_forward",
    "frontright": "looking_forward",
    
    # === Behaviors (for classification training) ===
    "normal": "normal",
    "tidak_ada_kecurangan": "normal",  # Indonesian: no cheating
    "fokus": "normal",  # Indonesian: focused
    "not_cheating": "normal",
    "person_without_headphone": "normal",
    "not_wearing": "normal",
    
    "cheating": "cheating",
    "mencontek": "cheating",  # Indonesian: cheating
    "possible_cheating": "cheating",
    
    "peeking": "peeking",
    "looking around": "peeking",
    "head_rotation": "peeking",
    "learning to copy": "peeking",
    
    "talking": "talking",
    "talking_to_someone": "talking",
    "mouth_open": "talking",
    
    # === Hand gestures ===
    "hand_gestures": "hand_gesture",
    "hand_gesture": "hand_gesture",
    "posture": "hand_gesture",
    
    # === Objects less relevant but keep for context ===
    "bottle": "other_object",
    "cup": "other_object",
    "remote": "other_object",
    "objects": "other_object",
}


# Final unified classes for AI Interview Proctoring Model
UNIFIED_CLASSES = [
    "phone",           # 0 - Phone in hand or visible
    "earbuds",         # 1 - Earphones, headphones, AirPods
    "smartwatch",      # 2 - Smartwatch/Apple Watch
    "notes",           # 3 - Books, papers, cheat sheets
    "another_person",  # 4 - Second person in frame (CRITICAL)
    "laptop",          # 5 - Laptop/tablet/iPad
    "second_screen",   # 6 - TV/monitor as second screen
    "calculator",      # 7 - Calculator
    "pen",             # 8 - Pen (low priority)
    "looking_away",    # 9 - Looking left/right/down/up
    "looking_forward", # 10 - Looking at camera (normal)
    "peeking",         # 11 - Peeking at something
    "talking",         # 12 - Talking to someone
    "hand_gesture",    # 13 - Suspicious hand movements
    "normal",          # 14 - Normal behavior
    "cheating",        # 15 - General cheating behavior
]


class DatasetDownloader:
    """Downloads and processes datasets from Roboflow"""
    
    def __init__(
        self,
        output_dir: str = "data/training",
        api_key: Optional[str] = ROBOFLOW_API_KEY,
        format: str = "yolov8"
    ):
        self.output_dir = Path(output_dir)
        self.api_key = api_key or ROBOFLOW_API_KEY
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
        """Download using Roboflow API with progress indicator"""
        try:
            from roboflow import Roboflow
        except ImportError:
            print("   ⚠️  roboflow package not installed. Installing...")
            os.system("pip install roboflow -q")
            from roboflow import Roboflow
        
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(info["workspace"]).project(info["project"])
        version = project.version(info["version"])
        
        expected_images = info.get("images", "unknown")
        print(f"   ⬇️  Downloading ~{expected_images} images via Roboflow API...")
        print(f"   ⏳ This may take 2-5 minutes for large datasets...")
        
        # Download with progress
        import time
        start_time = time.time()
        dataset = version.download(self.format, location=str(output_dir))
        elapsed = time.time() - start_time
        
        # Count actual downloaded images
        images_count = 0
        for split in ["train", "valid", "test"]:
            split_images = output_dir / split / "images"
            if split_images.exists():
                images_count += len(list(split_images.glob("*")))
        
        print(f"   ✅ Downloaded {images_count} images in {elapsed:.1f}s to {output_dir}")
    
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
        default=os.environ.get("ROBOFLOW_API_KEY", ROBOFLOW_API_KEY),
        help="Roboflow API key (default: embedded key)"
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
    
    # API key is now embedded by default, no need to check
    api_key = args.api_key
    
    # Download
    downloader = DatasetDownloader(
        output_dir=args.output,
        api_key=api_key if not args.public else None,
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
