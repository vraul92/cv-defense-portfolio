"""
Ghost-Tracker Demo - Video File Processing

Usage:
    python demo_video.py --input video.mp4 --output output/
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tracker import GhostTracker
from visualizer import visualize_tracks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_detector(model_name: str = "yolov8n"):
    """Initialize YOLO detector."""
    try:
        from ultralytics import YOLO
        model = YOLO(f"{model_name}.pt")
        logger.info(f"Loaded detector: {model_name}")
        return model
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        raise


def detect_frame(model, frame):
    """Run detection on single frame."""
    results = model(frame, verbose=False)
    
    detections = []
    confidences = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bbox coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            
            # Convert to center format
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            detections.append(np.array([x, y, w, h]))
            confidences.append(conf)
    
    return detections, confidences


def process_video(input_path: str, output_dir: str, detector_model: str = "yolov8n"):
    """Process video file with tracking."""
    
    # Initialize
    detector = get_detector(detector_model)
    tracker = GhostTracker(max_age=30, min_hits=3)
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video
    output_path = Path(output_dir) / f"tracked_{Path(input_path).name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections, confidences = detect_frame(detector, frame)
        
        # Track
        tracks = tracker.update(detections, confidences)
        
        # Visualize
        vis_frame = visualize_tracks(
            frame, tracks,
            show_ghosts=True,
            show_uncertainty=True,
            show_trajectories=True
        )
        
        # Write output
        writer.write(vis_frame)
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    writer.release()
    
    logger.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ghost-Tracker Video Demo")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input video file")
    parser.add_argument("--output", "-o", type=str, default="output/",
                       help="Output directory")
    parser.add_argument("--model", "-m", type=str, default="yolov8n",
                       help="YOLO model (yolov8n, yolov8s, yolov8m)")
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
