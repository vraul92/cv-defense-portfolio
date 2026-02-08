"""
Recon-Map Demo - Visual Odometry and Trajectory Mapping

Usage:
    python demo.py --input video.mp4 --output output/
    python demo.py --camera 0  # Live webcam
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

from vo_pipeline import VisualOdometry, FeatureExtractor
from visualizer import TrajectoryVisualizer, create_dashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_video(input_source, output_dir: str, is_camera: bool = False):
    """
    Process video for visual odometry.
    
    Args:
        input_source: Video file path or camera index
        output_dir: Output directory
        is_camera: Whether input is live camera
    """
    # Initialize
    vo = VisualOdometry(focal_length=800.0)
    visualizer = TrajectoryVisualizer()
    feature_extractor = FeatureExtractor()
    
    # Open video source
    if is_camera:
        cap = cv2.VideoCapture(int(input_source))
        logger.info(f"Opened camera {input_source}")
    else:
        cap = cv2.VideoCapture(input_source)
        logger.info(f"Opened video: {input_source}")
    
    if not cap.isOpened():
        logger.error("Failed to open video source")
        return
    
    # Get properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_camera else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not is_camera:
        out_file = output_path / "recon_map_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width * 2, height * 2))
    
    frame_count = 0
    
    logger.info("Processing frames... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with VO
        pose = vo.process_frame(frame)
        
        if pose is not None:
            # Get current position
            position = pose[:3, 3]
            visualizer.update(position)
            
            # Extract features for visualization
            keypoints, _ = feature_extractor.extract(frame)
            feature_img = frame.copy()
            cv2.drawKeypoints(feature_img, keypoints, feature_img, 
                            color=(0, 255, 0), 
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Get visualizations
            topdown = visualizer.draw_2d_topdown()
            view3d = visualizer.draw_3d_view()
            
            # Create dashboard
            dashboard = create_dashboard(frame, feature_img, topdown, view3d)
            
            # Resize for display
            display = cv2.resize(dashboard, (1280, 720))
            
            # Show
            cv2.imshow("Recon-Map", display)
            
            # Save
            if not is_camera:
                writer.write(cv2.resize(dashboard, (width * 2, height * 2)))
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            logger.info(f"Processed {frame_count} frames")
    
    # Cleanup
    cap.release()
    if not is_camera:
        writer.release()
    cv2.destroyAllWindows()
    
    # Save trajectory
    trajectory = vo.get_trajectory()
    if len(trajectory) > 0:
        np.save(output_path / "trajectory.npy", trajectory)
        logger.info(f"Saved trajectory with {len(trajectory)} points")
        logger.info(f"Total distance: {np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))):.2f}m")


def main():
    parser = argparse.ArgumentParser(description="Recon-Map Visual Odometry Demo")
    parser.add_argument("--input", "-i", type=str, 
                       help="Input video file")
    parser.add_argument("--camera", "-c", type=int, default=None,
                       help="Camera index (e.g., 0 for webcam)")
    parser.add_argument("--output", "-o", type=str, default="output/",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.camera is not None:
        process_video(args.camera, args.output, is_camera=True)
    elif args.input:
        process_video(args.input, args.output, is_camera=False)
    else:
        logger.error("Please specify --input or --camera")


if __name__ == "__main__":
    main()
