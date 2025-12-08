# YOLO11 Fine-Tuning Guide

This guide covers fine-tuning **YOLO11** (Ultralytics latest) for interview-specific object detection.

https://docs.ultralytics.com/models/yolo11/

## Overview

We fine-tune YOLO11 to detect:
- **Phones** (handheld and on desk)
- **Multiple people** in frame
- **Cheat sheets / notes**
- **Secondary screens**
- **Books and reference materials**

## Dataset Requirements

### Minimum Dataset Size
- 500+ images per class
- At least 1000 total images recommended
- 80% train, 15% validation, 5% test split

### Image Collection Guidelines

1. **Capture variety:**
   - Different lighting conditions
   - Various camera angles
   - Multiple phone models and colors
   - Different skin tones and genders
   - Various backgrounds

2. **Representative scenarios:**
   - Phone in hand (holding, typing)
   - Phone on desk (visible, partially hidden)
   - Cheat sheets on desk or held
   - Second monitor visible
   - Multiple people in frame

### Annotation Format (YOLO)

Each image needs a corresponding `.txt` file:

```
# format: class x_center y_center width height
# values are normalized (0-1)
0 0.5 0.4 0.3 0.5    # person
1 0.7 0.8 0.1 0.15   # phone
```

## Directory Structure

```
data/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── dataset.yaml
```

## dataset.yaml

```yaml
path: /path/to/data
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: phone
  2: phone_in_hand
  3: cheat_sheet
  4: secondary_screen
  5: book
  6: laptop

nc: 7
```

## Training

### Quick Start

```bash
# Download base model
python scripts/download_models.py

# Prepare dataset (if converting from COCO format)
python scripts/prepare_dataset.py \
    --input data/raw \
    --output data/prepared \
    --format coco \
    --split

# Train
python scripts/train_yolo.py \
    --data data/prepared/dataset.yaml \
    --epochs 50 \
    --batch 16
```

### Training Options

```bash
python scripts/train_yolo.py \
    --data data/dataset.yaml \    # Dataset config
    --model yolo11n.pt \          # Base model (n, s, m, l, x)
    --epochs 50 \                 # Training epochs
    --batch 16 \                  # Batch size
    --imgsz 640 \                 # Image size
    --patience 10 \               # Early stopping patience
    --device 0 \                  # GPU device
    --name my_model               # Run name
```

### Model Variants (YOLO11)

| Model   | Size   | mAP   | Speed (T4) | Use Case |
|---------|--------|-------|------------|----------|
| yolo11n | 2.6M   | 39.5% | 100+ FPS   | Real-time, edge |
| yolo11s | 9.4M   | 47.0% | 60 FPS     | Balanced |
| yolo11m | 20.1M  | 51.5% | 30 FPS     | Higher accuracy |
| yolo11l | 25.3M  | 53.4% | 20 FPS     | Best accuracy |
| yolo11x | 56.9M  | 54.7% | 11 FPS     | Maximum accuracy |

For real-time processing at 10 FPS, **yolo11n** or **yolo11s** recommended.

## Evaluation

```bash
python scripts/evaluate.py \
    --model runs/detect/my_model/weights/best.pt \
    --data data/dataset.yaml \
    --split test \
    --benchmark
```

### Expected Metrics

After fine-tuning on interview data:

| Class          | Expected mAP50 |
|----------------|----------------|
| person         | 95%+           |
| phone          | 85%+           |
| phone_in_hand  | 80%+           |
| cheat_sheet    | 75%+           |
| secondary_screen| 90%+          |

## Using the Fine-tuned Model

```python
from models.yolo import YOLODetector

# Use fine-tuned model
detector = YOLODetector(model_path="weights/yolo11_interview.pt")

# Detect objects
results = detector.detect(frame)
print(results["phone_detected"])
print(results["person_count"])
```

## Tips for Better Results

1. **Balance your dataset** - Equal samples per class
2. **Include hard negatives** - Similar objects that aren't phones
3. **Augment carefully** - Avoid unrealistic transformations
4. **Start with nano** - yolo11n trains faster for experiments
5. **Monitor val loss** - Stop if overfitting occurs
6. **Test on real interviews** - Validate with actual video frames

## Labeling Tools

Recommended tools for annotation:
- [Label Studio](https://labelstud.io/)
- [CVAT](https://cvat.ai/)
- [Roboflow](https://roboflow.com/)
- [LabelImg](https://github.com/heartexlabs/labelImg)

## Transfer Learning Strategy

1. Start with COCO pre-trained yolo11n
2. Freeze backbone for first 10 epochs
3. Unfreeze and train full model
4. Use lower learning rate (0.001 → 0.0001)

```python
# Advanced training with frozen backbone
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="dataset.yaml",
    epochs=50,
    freeze=10,  # Freeze first 10 layers for 10 epochs
)
```
