"""
Dataset Preparation for YOLO11 Fine-Tuning

Converts various annotation formats to YOLO format.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional
import random


def prepare_directory_structure(output_dir: Path):
    """Create YOLO dataset directory structure."""
    dirs = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "images" / "test",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
        output_dir / "labels" / "test",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory structure in: {output_dir}")


def convert_coco_to_yolo(
    coco_json: Path,
    images_dir: Path,
    output_dir: Path,
    class_mapping: Optional[dict] = None,
):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json: Path to COCO JSON file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO dataset
        class_mapping: Optional mapping from COCO category IDs to YOLO class IDs
    """
    with open(coco_json) as f:
        coco = json.load(f)
    
    # Build category mapping
    if class_mapping is None:
        class_mapping = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
    
    # Build image ID to filename mapping
    image_map = {img["id"]: img for img in coco["images"]}
    
    # Group annotations by image
    annotations_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert each image
    for img_id, img_info in image_map.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        img_file = img_info["file_name"]
        
        # Get annotations for this image
        anns = annotations_by_image.get(img_id, [])
        
        # Convert to YOLO format
        yolo_lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in class_mapping:
                continue
            
            class_id = class_mapping[cat_id]
            
            # COCO bbox: [x, y, width, height] (absolute)
            x, y, w, h = ann["bbox"]
            
            # YOLO: [class x_center y_center width height] (normalized)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Write label file
        label_file = output_dir / "labels" / Path(img_file).stem + ".txt"
        label_file.write_text("\n".join(yolo_lines))
        
        # Copy image
        src_img = images_dir / img_file
        if src_img.exists():
            dst_img = output_dir / "images" / img_file
            shutil.copy(src_img, dst_img)
    
    print(f"✅ Converted {len(image_map)} images to YOLO format")


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split dataset into train/val/test sets.
    
    Args:
        images_dir: Directory with all images
        labels_dir: Directory with all labels
        output_dir: Output directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all images
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    random.shuffle(images)
    
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]
    
    # Copy files to appropriate directories
    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]:
        for img_path in split_images:
            # Copy image
            dst_img = output_dir / "images" / split_name / img_path.name
            shutil.copy(img_path, dst_img)
            
            # Copy label
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                dst_label = output_dir / "labels" / split_name / label_path.name
                shutil.copy(label_path, dst_label)
    
    print(f"✅ Split dataset: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")


def create_dataset_yaml(output_dir: Path, class_names: list[str]):
    """Create dataset.yaml for YOLO training."""
    yaml_content = f"""# Interview Detection Dataset
# Auto-generated by prepare_dataset.py

path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_content += f"\nnc: {len(class_names)}\n"
    
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"✅ Created: {yaml_path}")
    
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO training")
    
    parser.add_argument("--input", type=str, required=True, help="Input directory with images/annotations")
    parser.add_argument("--output", type=str, required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--format", type=str, default="yolo", choices=["yolo", "coco"], 
                       help="Input annotation format")
    parser.add_argument("--split", action="store_true", help="Split into train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Create directory structure
    prepare_directory_structure(output_dir)
    
    # Default class names for interview detection
    class_names = [
        "person",
        "phone", 
        "phone_in_hand",
        "cheat_sheet",
        "secondary_screen",
        "book",
        "laptop",
    ]
    
    if args.format == "coco":
        coco_json = input_dir / "annotations.json"
        if not coco_json.exists():
            print(f"❌ COCO annotations not found: {coco_json}")
            return
        
        convert_coco_to_yolo(
            coco_json=coco_json,
            images_dir=input_dir / "images",
            output_dir=output_dir,
        )
    
    if args.split:
        split_dataset(
            images_dir=input_dir / "images" if not args.format == "coco" else output_dir / "images",
            labels_dir=input_dir / "labels" if not args.format == "coco" else output_dir / "labels",
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    
    # Create dataset YAML
    create_dataset_yaml(output_dir, class_names)
    
    print(f"\n✅ Dataset prepared in: {output_dir}")
    print(f"\nTo train YOLO11:")
    print(f"  python scripts/train_yolo_h200.py --data {output_dir}/dataset.yaml")


if __name__ == "__main__":
    main()
