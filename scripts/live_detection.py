#!/usr/bin/env python3
"""
Live Proctoring Detection using MacBook Camera

This script opens your webcam and runs real-time detection
for cheating behaviors using the trained YOLO model.

Usage:
    python scripts/live_detection.py
    python scripts/live_detection.py --model runs/detect/proctoring_h200_20251209_020408/weights/best.pt
    python scripts/live_detection.py --conf 0.5

Controls:
    q - Quit
    s - Save screenshot
    p - Pause/Resume
"""

import argparse
import cv2
import time
from pathlib import Path
from datetime import datetime

# Colors for different class categories (BGR format)
COLORS = {
    # Devices - Red
    "phone": (0, 0, 255),
    "earbuds": (0, 0, 255),
    "smartwatch": (0, 0, 255),
    "laptop": (0, 0, 255),
    "second_screen": (0, 0, 255),
    "calculator": (0, 0, 255),
    
    # Cheating behaviors - Orange
    "notes": (0, 165, 255),
    "cheating": (0, 165, 255),
    "peeking": (0, 165, 255),
    
    # People - Yellow
    "another_person": (0, 255, 255),
    
    # Gaze - Blue/Green
    "looking_away": (255, 100, 0),
    "looking_forward": (0, 255, 0),
    
    # Other
    "talking": (255, 0, 255),
    "hand_gesture": (255, 0, 255),
    "pen": (128, 128, 128),
    "normal": (0, 255, 0),
    "face_hiding": (0, 0, 255),
}

# High-risk classes that should trigger alerts
HIGH_RISK_CLASSES = [
    "phone", "earbuds", "smartwatch", "notes", "another_person",
    "laptop", "second_screen", "cheating", "peeking", "face_hiding"
]

# Classes that need higher confidence to reduce false positives
# (e.g., book covers, posters detected as "another_person")
HIGH_CONF_CLASSES = {
    "another_person": 0.75,  # Requires 75% confidence (reduce false positives on book covers)
    "cheating": 0.55,
    "peeking": 0.55,
}

# Classes that can use lower confidence (gaze/behavior detection)
LOW_CONF_CLASSES = {
    "looking_away": 0.20,
    "looking_forward": 0.20,
    "talking": 0.25,
    "hand_gesture": 0.25,
    "normal": 0.20,
}


def get_color(class_name):
    """Get color for a class, default to white if not found."""
    return COLORS.get(class_name, (255, 255, 255))


def draw_detections(frame, results, conf_threshold=0.25):
    """Draw bounding boxes and labels on frame."""
    alerts = []
    
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls_id]
            
            # Use class-specific confidence thresholds
            if class_name in HIGH_CONF_CLASSES:
                min_conf = HIGH_CONF_CLASSES[class_name]
            elif class_name in LOW_CONF_CLASSES:
                min_conf = LOW_CONF_CLASSES[class_name]
            else:
                min_conf = conf_threshold
            
            if conf < min_conf:
                continue
            
            # Get color
            color = get_color(class_name)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Check for high-risk detection
            if class_name in HIGH_RISK_CLASSES:
                alerts.append((class_name, conf))
    
    return frame, alerts


def draw_status_bar(frame, fps, alerts, paused=False):
    """Draw status bar at top of frame."""
    h, w = frame.shape[:2]
    
    # Draw status bar background
    cv2.rectangle(frame, (0, 0), (w, 40), (50, 50, 50), -1)
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw pause status
    if paused:
        cv2.putText(frame, "PAUSED", (w//2 - 50, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw alert count
    if alerts:
        alert_text = f"⚠️ ALERTS: {len(alerts)}"
        cv2.putText(frame, alert_text, (w - 180, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "✓ OK", (w - 80, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw alert banner if high-risk detected
    if alerts:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 200), -1)
        alert_names = ", ".join([f"{a[0]} ({a[1]:.0%})" for a in alerts[:3]])
        cv2.putText(frame, f"DETECTED: {alert_names}", (10, h - 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="Live Proctoring Detection")
    parser.add_argument("--model", "-m", type=str, 
                       default="runs/detect/proctoring_h200_20251209_020408/weights/best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--conf", "-c", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                       help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                       help="Frame height (default: 720)")
    parser.add_argument("--imgsz", type=int, default=480,
                       help="Inference image size (default: 480, smaller = faster)")
    args = parser.parse_args()
    
    # Detect device
    import torch
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("🍎 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🟢 Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("⚪ Using CPU")
    
    # Load model
    print(f"📦 Loading model: {args.model}")
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        model.to(device)
        print(f"✅ Model loaded! Classes: {len(model.names)}")
        print(f"📐 Inference size: {args.imgsz}px")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Open camera
    print(f"📷 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return
    
    print("✅ Camera opened!")
    print("\n🎮 Controls:")
    print("   q - Quit")
    print("   s - Save screenshot")
    print("   p - Pause/Resume")
    print("\n🚀 Starting live detection...\n")
    
    # Create screenshots directory
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame!")
                    break
                
                # Run detection with smaller image size for speed
                results = model(frame, imgsz=args.imgsz, verbose=False, device=device)
                
                # Draw detections
                frame, alerts = draw_detections(frame, results, args.conf)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed
                
                # Reset FPS counter every 2 seconds
                if elapsed > 2:
                    frame_count = 0
                    start_time = time.time()
            
            # Draw status bar
            display_frame = draw_status_bar(frame.copy(), fps, alerts if not paused else [], paused)
            
            # Show frame
            cv2.imshow("Proctoring Live Detection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = screenshots_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(filepath), display_frame)
                print(f"📸 Screenshot saved: {filepath}")
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")


if __name__ == "__main__":
    main()
