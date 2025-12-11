#!/usr/bin/env python3
"""
Dataset Downloader for AI Interview Proctoring

Downloads labeled datasets for training YOLO models for LIVE AI INTERVIEW monitoring.

9 OBJECT CLASSES (Behavioral detection done by MediaPipe):
==========================================================
1. phone - Phone in hand or visible (~35,000+ images)
2. earbuds - Earphones, headphones, AirPods (~8,000+ images)
3. smartwatch - Smartwatch/Apple Watch (~5,000+ images)
4. notes - Books, papers, cheat sheets (~15,000+ images)
5. another_person - Second person in frame (~30,000+ images)
6. laptop - Laptop/tablet/iPad (~15,000+ images)
7. second_screen - TV/monitor in background (~2,000 images)
8. calculator - Calculator device (~1,000 images)
9. face_hiding - Face covered by hand/scarf/mask (~3,600 images)

TOTAL: ~100,000+ IMAGES across 20+ datasets!

NOTE: Behavioral classes (looking_away, cheating, talking, etc.) are REMOVED.
      Use MediaPipe for gaze/pose detection instead.

PRIORITY 1 DATASETS (Core Training):
====================================
✅ online_proctoring_system - 87K images (gaze + phone)
✅ cheating_cheatingways - 10.9K (laptop, mobile, pen, watch, headphones)
✅ mobile_phone_tusker - 9.9K (phone detection)
✅ phone_hi - 9.7K (phone detection)
✅ And more...

PRIORITY 2 EDGE CASE DATASETS (Fine-tuning):
============================================
✅ face_hiding_detection - 3.63K (face covered by hand/scarf)
✅ notes_detection_large - 9.97K (additional notes/paper)
✅ sticky_notes_desk - 150 (sticky notes on desk)

Usage:
    # With Roboflow API key (RECOMMENDED):
    python download_datasets.py --api-key YOUR_ROBOFLOW_API_KEY --category all

    # Download specific category:
    python download_datasets.py --api-key KEY --category phone
    python download_datasets.py --api-key KEY --category earbuds
    python download_datasets.py --api-key KEY --category interview
    python download_datasets.py --api-key KEY --category cheating
    python download_datasets.py --api-key KEY --category edge_case
    
    # Download only priority 1 datasets (largest/most useful):
    python download_datasets.py --api-key KEY --priority 1
    
    # Download priority 2 edge case datasets:
    python download_datasets.py --api-key KEY --priority 2 --category edge_case
    
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
    # REMOVED: online_exam_cheating - only has [normal, cheating] classes (both skipped)
    # REMOVED: online_exam_cheating_2 - only has [normal, cheating] classes (both skipped)
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
    # REMOVED: classroom_cheating - Normal/Talking skipped, only phone useful (redundant)
    
    # ============================================================
    # === NEW: ONLINE PROCTORING SPECIFIC (AI Interview) ===
    # ============================================================
    # REMOVED: online_proctoring_system - mostly gaze classes (use MediaPipe instead)
    #          Only mobile_phone useful but we have 30K+ phone images already
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
    # REMOVED: proctoring_samya - mostly behavioral (Menoleh, Menunduk, Tidak_Ada_Kecurangan skipped)
    # REMOVED: exam_proctoring_urhzv - mostly behavioral (cheating, not_cheating, possible_cheating skipped)
    
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
    # === GAZE/LOOKING DIRECTION - REMOVED (use MediaPipe instead)
    # ============================================================
    # REMOVED: gaze_direction - all gaze classes skipped (use MediaPipe)
    # REMOVED: looking_direction - gaze classes skipped (only Phone useful but redundant)
    
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
    # cheating_zeina removed - no versions available on Roboflow
    # phone_hi moved to end of Priority 1 (version detection issue)
    "phone_hi": {
        "workspace": "hi-wzisi",
        "project": "phone-du6hl",
        "version": 1,
        "images": 9720,
        "classes": ["phone"],
        "priority": 1,
        "category": "phone",
        "description": "Large phone detection dataset (download last)"
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
    # REMOVED: cheating_system_3k - all behavioral classes skipped (Hand_Gestures, Peeking, Posture)
    # REMOVED: cheating_analyze - only behavioral classes (Fokus, Mencontek skipped)
    
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
    # REMOVED: exam_monitoring - mostly behavioral classes (only 'using phone' useful but small dataset)
    
    # ============================================================
    # === PRIORITY 2: EDGE CASE DATASETS (For Fine-tuning) ===
    # ============================================================
    "face_hiding_detection": {
        "workspace": "university-8zqnr",
        "project": "face-hiding-detection",
        "version": 1,
        "images": 3630,
        "classes": ["hand", "balaclava", "concealing_glasses", "medicine_mask", 
                   "non-concealing_glasses", "nothing", "scarf"],
        "priority": 2,
        "category": "edge_case",
        "description": "⭐ Face covered by hand/scarf/mask detection"
    },
    "notes_detection_large": {
        "workspace": "note-detection-2024",
        "project": "notes-dpwrs",
        "version": 1,
        "images": 9970,
        "classes": ["note"],
        "priority": 2,
        "category": "edge_case",
        "description": "⭐ LARGE notes/paper detection for edge cases"
    },
    "sticky_notes_desk": {
        "workspace": "qwerty-k5pgk",
        "project": "desk-ir3be",
        "version": 1,
        "images": 150,
        "classes": ["bag", "book", "bottle", "keyboard", "laptop", "mouse", 
                   "mug", "vase", "StickyNotes", "calender", "desktop", 
                   "file", "filetray", "headphone", "nameplate", "page"],
        "priority": 2,
        "category": "edge_case",
        "description": "⭐ Sticky notes on desk detection"
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
    
    # === Gaze direction -> SKIP (use MediaPipe instead) ===
    # These behavioral classes cannot be reliably detected by YOLO
    # Use MediaPipe face mesh for gaze detection instead
    "looking_down": None,  # Skip - use MediaPipe
    "looking_left": None,  # Skip - use MediaPipe
    "looking_right": None, # Skip - use MediaPipe
    "looking_up": None,    # Skip - use MediaPipe
    "look_down": None,
    "look_left": None,
    "look_right": None,
    "look_up": None,
    "menoleh": None,       # Indonesian: looking away
    "menunduk": None,      # Indonesian: head down
    "back": None,
    "left": None,
    "right": None,
    "looking_straight": None,
    "looking_forward": None,
    "look_forward": None,
    "front": None,
    "frontright": None,
    
    # === Behaviors -> SKIP (cannot be detected by object detection) ===
    "normal": None,
    "tidak_ada_kecurangan": None,  # Indonesian: no cheating
    "fokus": None,                  # Indonesian: focused
    "not_cheating": None,
    "person_without_headphone": None,
    "not_wearing": None,
    
    "cheating": None,
    "mencontek": None,              # Indonesian: cheating
    "possible_cheating": None,
    
    "peeking": None,
    "looking around": None,
    "head_rotation": None,
    "learning to copy": None,
    
    "talking": None,
    "talking_to_someone": None,
    "mouth_open": None,
    
    # === Hand gestures -> SKIP ===
    "hand_gestures": None,
    "hand_gesture": None,
    "posture": None,
    
    # === Pen -> SKIP (too small, unreliable) ===
    "pen": None,
    "lapicera": None,
    
    # === Edge Case: Face Hiding ===
    "hand": "face_hiding",
    "balaclava": "face_hiding",
    "concealing_glasses": "face_hiding",
    "medicine_mask": "face_hiding",
    "scarf": "face_hiding",
    "non-concealing_glasses": None,  # Regular glasses are OK - skip
    "nothing": None,  # Skip normal behavior
    
    # === Edge Case: Sticky Notes ===
    "stickynotes": "notes",
    "StickyNotes": "notes",
    "sticky_notes": "notes",
    "calender": "notes",
    "page": "notes",
    "filetray": None,   # Skip irrelevant objects
    "nameplate": None,
    "desktop": None,
    "vase": None,
    "mug": None,
    
    # === Objects less relevant -> SKIP ===
    "bottle": None,
    "cup": None,
    "remote": None,
    "objects": None,
}


# Final unified classes for AI Interview Proctoring Model
# ONLY OBJECT CLASSES - Use MediaPipe for behavioral detection (gaze, pose)
UNIFIED_CLASSES = [
    "phone",            # 0 - Phone in hand or visible
    "earbuds",          # 1 - Earphones, headphones, AirPods
    "smartwatch",       # 2 - Smartwatch/Apple Watch
    "notes",            # 3 - Books, papers, cheat sheets, sticky notes
    "another_person",   # 4 - Second person in frame (CRITICAL)
    "laptop",           # 5 - Laptop/tablet/iPad
    "second_screen",    # 6 - TV/monitor as second screen
    "calculator",       # 7 - Calculator
    "face_hiding",      # 8 - Face covered by hand/scarf/mask
]
# NOTE: Removed behavioral classes (looking_away, looking_forward, peeking,
#       talking, hand_gesture, normal, cheating, pen)
#       These are detected using MediaPipe pose/face mesh instead.


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
        
        import time
        import threading
        import sys
        
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(info["workspace"]).project(info["project"])
        
        # Auto-detect latest version if specified version fails
        specified_version = info.get("version", 1)
        try:
            version = project.version(specified_version)
        except Exception as e:
            print(f"   ⚠️  Version {specified_version} not found, trying to get latest...")
            versions = project.versions()
            if versions:
                version = versions[0]
                print(f"   📌 Using version {version.version}")
            else:
                raise Exception(f"No versions available for {info['project']}")
        
        expected_images = info.get("images", "unknown")
        print(f"   ⬇️  Downloading ~{expected_images} images via Roboflow API...")
        print(f"   ⏳ This may take 2-5 minutes for large datasets...", flush=True)
        
        # Progress spinner in background
        stop_spinner = threading.Event()
        def spinner():
            chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            i = 0
            start = time.time()
            while not stop_spinner.is_set():
                elapsed = int(time.time() - start)
                sys.stdout.write(f"\r   {chars[i % len(chars)]} Downloading... {elapsed}s elapsed")
                sys.stdout.flush()
                i += 1
                time.sleep(0.1)
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()
        
        spinner_thread = threading.Thread(target=spinner)
        spinner_thread.start()
        
        # Download
        start_time = time.time()
        try:
            dataset = version.download(self.format, location=str(output_dir))
        finally:
            stop_spinner.set()
            spinner_thread.join()
        
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
