#!/usr/bin/env python3
"""
Inference script for testing proctoring models on images/videos.

Usage:
    python scripts/inference.py --image path/to/image.jpg
    python scripts/inference.py --video path/to/video.mp4
    python scripts/inference.py --webcam
"""

import argparse
import cv2
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.yolo import YOLODetector
from models.mediapipe import MediaPipeAnalyzer
from models.classifier import CheatingClassifier
from pipeline.processor import FrameProcessor


def process_image(image_path: str, processor: FrameProcessor, output_path: str = None):
    """Process a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Process frame
    results = processor.process(frame)
    
    # Draw results
    annotated = draw_results(frame, results)
    
    # Print results
    print("\n=== Proctoring Results ===")
    print(f"Cheating Probability: {results.get('cheating_probability', 0):.2%}")
    print(f"Person Count: {results.get('person_count', 0)}")
    print(f"Phone Detected: {results.get('phone_detected', False)}")
    print(f"Face Visible: {results.get('face_visible', False)}")
    if results.get('face_visible'):
        print(f"Gaze Direction: {results.get('gaze_direction', 'unknown')}")
        print(f"Looking Away: {results.get('looking_away', False)}")
    
    if results.get('alerts'):
        print(f"\n⚠️ Alerts: {', '.join(results['alerts'])}")
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"\nSaved to: {output_path}")
    else:
        cv2.imshow("Proctoring Analysis", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(video_path: str, processor: FrameProcessor, output_path: str = None):
    """Process a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    alert_count = 0
    
    print(f"Processing video at {fps} FPS...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = processor.process(frame)
        
        # Draw results
        annotated = draw_results(frame, results)
        
        # Count alerts
        if results.get('alerts'):
            alert_count += 1
        
        # Write or display
        if writer:
            writer.write(annotated)
        else:
            cv2.imshow("Proctoring Analysis", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    if writer:
        writer.release()
        print(f"Saved to: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"\n=== Summary ===")
    print(f"Total Frames: {frame_count}")
    print(f"Frames with Alerts: {alert_count}")
    print(f"Alert Rate: {alert_count / frame_count:.1%}")


def process_webcam(processor: FrameProcessor):
    """Process webcam feed in real-time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Process frame
        results = processor.process(frame)
        
        # Draw results
        annotated = draw_results(frame, results)
        
        # Draw FPS
        cv2.putText(
            annotated, f"FPS: {fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        cv2.imshow("Proctoring Analysis", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def draw_results(frame, results):
    """Draw proctoring results on frame."""
    annotated = frame.copy()
    
    # Draw cheating probability
    prob = results.get('cheating_probability', 0)
    color = (0, 255, 0) if prob < 0.5 else (0, 165, 255) if prob < 0.7 else (0, 0, 255)
    cv2.putText(
        annotated, f"Cheat Prob: {prob:.1%}",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )
    
    # Draw status
    status_y = 90
    
    # Phone detection
    phone_status = "📱 Phone: YES" if results.get('phone_detected') else "📱 Phone: No"
    phone_color = (0, 0, 255) if results.get('phone_detected') else (0, 255, 0)
    cv2.putText(
        annotated, phone_status,
        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 2
    )
    
    # Person count
    count = results.get('person_count', 0)
    count_color = (0, 0, 255) if count > 1 else (0, 255, 0)
    cv2.putText(
        annotated, f"👥 People: {count}",
        (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, count_color, 2
    )
    
    # Gaze
    gaze = results.get('gaze_direction', 'unknown')
    looking_away = results.get('looking_away', False)
    gaze_color = (0, 0, 255) if looking_away else (0, 255, 0)
    cv2.putText(
        annotated, f"👁️ Gaze: {gaze}",
        (10, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_color, 2
    )
    
    # Alerts
    alerts = results.get('alerts', [])
    if alerts:
        cv2.rectangle(annotated, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
        cv2.putText(
            annotated, f"⚠️ ALERT: {', '.join(alerts)}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
    
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Test proctoring models")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--output", "-o", type=str, help="Output path")
    parser.add_argument("--model", type=str, default="weights/yolov8n.pt", 
                       help="YOLOv8 model path")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold")
    args = parser.parse_args()
    
    if not any([args.image, args.video, args.webcam]):
        parser.print_help()
        return
    
    # Initialize processor
    print("Loading models...")
    processor = FrameProcessor(
        yolo_model_path=args.model,
        yolo_confidence=args.confidence
    )
    print("Models loaded!")
    
    # Process input
    if args.image:
        process_image(args.image, processor, args.output)
    elif args.video:
        process_video(args.video, processor, args.output)
    elif args.webcam:
        process_webcam(processor)


if __name__ == "__main__":
    main()
