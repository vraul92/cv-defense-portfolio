"""
SAM 3D Demo - Multi-View 3D Reconstruction

Usage:
    python demo_sam3d.py --images samples/view*.jpg --output output/
    python demo_sam3d.py --video samples/object_rotation.mp4 --output output/
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sam3d_reconstructor import SAM3DReconstructor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_pointcloud_matplotlib(points: np.ndarray, 
                                     colors: np.ndarray,
                                     save_path: Optional[str] = None):
    """
    Visualize point cloud using matplotlib.
    
    Args:
        points: Point cloud (N, 3)
        colors: Colors (N, 3)
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for visualization
    n_points = min(len(points), 5000)
    indices = np.random.choice(len(points), n_points, replace=False)
    
    pts = points[indices]
    cols = colors[indices]
    
    # Plot
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
              c=cols, s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction ({len(points)} points)')
    
    # Equal aspect ratio
    max_range = np.array([pts[:, 0].max() - pts[:, 0].min(),
                         pts[:, 1].max() - pts[:, 1].min(),
                         pts[:, 2].max() - pts[:, 2].min()]).max() / 2.0
    
    mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
    mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
    mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def process_images(image_paths: List[str], 
                   checkpoint: str,
                   output_dir: str):
    """Process multiple images for 3D reconstruction."""
    
    # Initialize reconstructor
    logger.info("Initializing SAM 3D Reconstructor...")
    reconstructor = SAM3DReconstructor(
        sam_checkpoint=checkpoint,
        device="cpu",
        voxel_size=0.02
    )
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        logger.info(f"Processing {img_path}...")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Failed to load {img_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate camera pose (rotate around object)
        angle = 2 * np.pi * i / len(image_paths)
        
        # Simple circular camera path
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        t = np.array([np.sin(angle) * 2, 0, np.cos(angle) * 2])
        
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        
        # Add view
        reconstructor.add_view(image, camera_pose=pose)
    
    # Reconstruct
    logger.info("Reconstructing 3D point cloud...")
    points, colors = reconstructor.reconstruct()
    
    if len(points) == 0:
        logger.error("No points reconstructed")
        return
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    reconstructor.save_pointcloud(str(output_path / "reconstruction.ply"))
    
    # Visualize
    visualize_pointcloud_matplotlib(
        points, colors,
        save_path=str(output_path / "visualization.png")
    )
    
    logger.info(f"Output saved to {output_dir}")


def process_video(video_path: str,
                 checkpoint: str,
                 output_dir: str,
                 frame_skip: int = 10):
    """Process video for 3D reconstruction."""
    
    logger.info(f"Processing video: {video_path}")
    
    # Initialize
    reconstructor = SAM3DReconstructor(
        sam_checkpoint=checkpoint,
        device="cpu",
        voxel_size=0.02
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        logger.info(f"Processing frame {frame_count}")
        
        # Convert
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add view (no camera pose - will use identity)
        reconstructor.add_view(frame_rgb)
        
        frame_count += 1
        
        # Limit views
        if len(reconstructor.views) >= 8:
            break
    
    cap.release()
    
    # Reconstruct
    points, colors = reconstructor.reconstruct()
    
    if len(points) == 0:
        logger.error("No points reconstructed")
        return
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    reconstructor.save_pointcloud(str(output_path / "reconstruction.ply"))
    
    visualize_pointcloud_matplotlib(
        points, colors,
        save_path=str(output_path / "visualization.png")
    )


def create_synthetic_views(output_dir: str, n_views: int = 8):
    """Create synthetic multi-view images for testing."""
    logger.info(f"Creating {n_views} synthetic views...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    
    for i in range(n_views):
        # Create synthetic scene with object
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Gray background
        
        # Add object (colored cube-like shape)
        angle = 2 * np.pi * i / n_views
        
        # Main object
        cx, cy = 320, 240
        size = 100
        
        # Draw cube faces
        pts = np.array([
            [cx - size, cy - size],
            [cx + size, cy - size],
            [cx + size, cy + size],
            [cx - size, cy + size]
        ], dtype=np.int32)
        
        color = tuple(np.random.randint(50, 200, 3).tolist())
        cv2.fillPoly(img, [pts], color)
        
        # Add depth cue (smaller face)
        offset_x = int(30 * np.cos(angle))
        offset_y = int(20 * np.sin(angle))
        
        pts_back = np.array([
            [cx - size + offset_x, cy - size + offset_y],
            [cx + size + offset_x, cy - size + offset_y],
            [cx + size + offset_x, cy + size + offset_y],
            [cx - size + offset_x, cy + size + offset_y]
        ], dtype=np.int32)
        
        color_back = tuple(np.clip(np.array(color) * 0.7, 0, 255).astype(np.uint8).tolist())
        cv2.fillPoly(img, [pts_back], color_back)
        
        # Save
        img_path = str(output_path / f"view_{i:02d}.jpg")
        cv2.imwrite(img_path, img)
        image_paths.append(img_path)
    
    logger.info(f"Created {len(image_paths)} synthetic views")
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="SAM 3D Reconstruction Demo")
    parser.add_argument("--images", "-i", nargs="+", 
                       help="Input image files")
    parser.add_argument("--video", "-v", type=str,
                       help="Input video file")
    parser.add_argument("--checkpoint", "-c", type=str,
                       default="weights/sam_vit_b_01ec64.pth",
                       help="SAM checkpoint path")
    parser.add_argument("--output", "-o", type=str, default="output/sam3d/",
                       help="Output directory")
    parser.add_argument("--create-synthetic", action="store_true",
                       help="Create synthetic test views")
    parser.add_argument("--frame-skip", type=int, default=10,
                       help="Frame skip for video processing")
    
    args = parser.parse_args()
    
    # Download weights if needed
    if not Path(args.checkpoint).exists():
        logger.info("Downloading SAM weights...")
        import urllib.request
        Path(args.checkpoint).parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            args.checkpoint
        )
    
    # Process
    if args.create_synthetic:
        image_paths = create_synthetic_views("samples/sam3d/")
        process_images(image_paths, args.checkpoint, args.output)
    elif args.images:
        process_images(args.images, args.checkpoint, args.output)
    elif args.video:
        process_video(args.video, args.checkpoint, args.output, args.frame_skip)
    else:
        logger.error("Please specify --images, --video, or --create-synthetic")


if __name__ == "__main__":
    from typing import List, Optional
    main()
