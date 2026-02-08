"""
SAM 3D Lite - Multi-View 3D Reconstruction using SAM

This module implements a lightweight 3D reconstruction pipeline using:
1. SAM for segmentation across multiple views
2. Depth estimation (MiDaS) for 3D projection
3. Point cloud fusion from multiple angles
4. Interactive 3D visualization

Author: Rahul Vuppalapati
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class View:
    """Single view with image, mask, and camera pose."""
    image: np.ndarray
    mask: np.ndarray
    depth: Optional[np.ndarray] = None
    camera_pose: Optional[np.ndarray] = None  # 4x4 transformation matrix
    

class DepthEstimator:
    """Depth estimation using MiDaS."""
    
    def __init__(self, model_type: str = "DPT_Large", device: str = "cpu"):
        """
        Initialize MiDaS depth estimator.
        
        Args:
            model_type: MiDaS model variant
            device: Device for inference
        """
        self.device = device
        self.model_type = model_type
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model."""
        logger.info(f"Loading MiDaS model: {self.model_type}")
        
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
            
            logger.info("MiDaS loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS: {e}")
            logger.info("Falling back to simple depth estimation")
            self.model = None
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth for image.
        
        Args:
            image: Input RGB image
            
        Returns:
            Depth map (H, W)
        """
        if self.model is None:
            # Fallback: simple gradient-based depth
            return self._fallback_depth(image)
        
        # Prepare input
        input_batch = self.transform(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        
        return depth
    
    def _fallback_depth(self, image: np.ndarray) -> np.ndarray:
        """Simple fallback depth estimation."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Use gradient magnitude as proxy for depth edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Invert and normalize
        depth = 255 - (gradient / gradient.max() * 255).astype(np.uint8)
        
        return depth


class SAM3DReconstructor:
    """
    3D reconstruction using SAM + Depth estimation.
    
    Takes multiple views of an object, segments with SAM,
    estimates depth, and fuses into 3D point cloud.
    """
    
    def __init__(self, 
                 sam_checkpoint: str,
                 device: str = "cpu",
                 voxel_size: float = 0.01):
        """
        Initialize SAM 3D reconstructor.
        
        Args:
            sam_checkpoint: Path to SAM checkpoint
            device: Device for inference
            voxel_size: Voxel size for point cloud downsampling
        """
        self.device = device
        self.voxel_size = voxel_size
        
        # Load SAM
        self._load_sam(sam_checkpoint)
        
        # Load depth estimator
        self.depth_estimator = DepthEstimator(device=device)
        
        # Storage
        self.views: List[View] = []
        self.point_cloud: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        
        logger.info("SAM 3D Reconstructor initialized")
    
    def _load_sam(self, checkpoint_path: str):
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            logger.info(f"Loading SAM from {checkpoint_path}")
            
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.9,
                stability_score_thresh=0.95,
                min_mask_region_area=100,
            )
            
            logger.info("SAM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            raise
    
    def add_view(self, 
                 image: np.ndarray,
                 camera_pose: Optional[np.ndarray] = None):
        """
        Add a new view for reconstruction.
        
        Args:
            image: Input RGB image
            camera_pose: 4x4 camera transformation matrix (optional)
        """
        logger.info(f"Processing view {len(self.views) + 1}")
        
        # Generate SAM mask
        masks = self.mask_generator.generate(image)
        
        if not masks:
            logger.warning("No objects detected in view")
            return
        
        # Get largest mask
        largest_mask = max(masks, key=lambda x: x['area'])
        mask = largest_mask['segmentation']
        
        # Estimate depth
        depth = self.depth_estimator.estimate(image)
        
        # Create view
        view = View(
            image=image,
            mask=mask,
            depth=depth,
            camera_pose=camera_pose if camera_pose is not None else np.eye(4)
        )
        
        self.views.append(view)
        logger.info(f"View added. Total views: {len(self.views)}")
    
    def reconstruct(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct 3D point cloud from all views.
        
        Returns:
            (points, colors): Point cloud and corresponding colors
        """
        if not self.views:
            logger.error("No views to reconstruct")
            return np.array([]), np.array([])
        
        logger.info(f"Reconstructing from {len(self.views)} views...")
        
        all_points = []
        all_colors = []
        
        for i, view in enumerate(self.views):
            points, colors = self._view_to_pointcloud(view)
            
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
                logger.info(f"View {i+1}: {len(points)} points")
        
        if not all_points:
            logger.warning("No points generated")
            return np.array([]), np.array([])
        
        # Concatenate
        self.point_cloud = np.vstack(all_points)
        self.colors = np.vstack(all_colors)
        
        # Voxel downsampling
        self.point_cloud, self.colors = self._voxel_downsample(
            self.point_cloud, self.colors
        )
        
        logger.info(f"Final point cloud: {len(self.point_cloud)} points")
        
        return self.point_cloud, self.colors
    
    def _view_to_pointcloud(self, view: View) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert single view to point cloud.
        
        Args:
            view: View object
            
        Returns:
            (points, colors)
        """
        h, w = view.mask.shape
        
        # Get masked pixels
        y, x = np.where(view.mask > 0)
        
        if len(x) == 0:
            return np.array([]), np.array([])
        
        # Get depth values
        depths = view.depth[y, x]
        
        # Normalize depths
        depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
        depths = depths * 2.0  # Scale to reasonable range
        
        # Back-project to 3D
        # Simple pinhole camera model
        fx = fy = 500.0  # Focal length
        cx, cy = w / 2, h / 2
        
        X = (x - cx) * depths / fx
        Y = (y - cy) * depths / fy
        Z = depths
        
        points = np.stack([X, Y, Z], axis=1)
        
        # Apply camera pose transformation
        if view.camera_pose is not None:
            # Convert to homogeneous coordinates
            points_homo = np.hstack([points, np.ones((len(points), 1))])
            points = (view.camera_pose @ points_homo.T).T[:, :3]
        
        # Get colors
        colors = view.image[y, x] / 255.0
        
        return points, colors
    
    def _voxel_downsample(self, points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsample point cloud using voxel grid.
        
        Args:
            points: Point cloud (N, 3)
            colors: Colors (N, 3)
            
        Returns:
            Downsampled points and colors
        """
        if len(points) == 0:
            return points, colors
        
        # Simple voxel downsampling
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        # Unique voxels
        unique_indices, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
        
        # Average points in each voxel
        downsampled_points = np.zeros((len(unique_indices), 3))
        downsampled_colors = np.zeros((len(unique_indices), 3))
        
        for i in range(len(unique_indices)):
            mask = inverse == i
            downsampled_points[i] = points[mask].mean(axis=0)
            downsampled_colors[i] = colors[mask].mean(axis=0)
        
        return downsampled_points, downsampled_colors
    
    def save_pointcloud(self, filepath: str):
        """
        Save point cloud to file.
        
        Args:
            filepath: Output file path (.ply or .pcd)
        """
        if self.point_cloud is None or len(self.point_cloud) == 0:
            logger.error("No point cloud to save")
            return
        
        if filepath.endswith('.ply'):
            self._save_ply(filepath)
        elif filepath.endswith('.pcd'):
            self._save_pcd(filepath)
        else:
            logger.error("Unsupported format. Use .ply or .pcd")
    
    def _save_ply(self, filepath: str):
        """Save as PLY format."""
        with open(filepath, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Data
            for point, color in zip(self.point_cloud, self.colors):
                r, g, b = (color * 255).astype(np.uint8)
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")
        
        logger.info(f"Saved PLY to {filepath}")
    
    def _save_pcd(self, filepath: str):
        """Save as PCD format."""
        with open(filepath, 'w') as f:
            # Header
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {len(self.point_cloud)}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {len(self.point_cloud)}\n")
            f.write("DATA ascii\n")
            
            # Data
            for point, color in zip(self.point_cloud, self.colors):
                rgb = (int(color[0] * 255) << 16) | (int(color[1] * 255) << 8) | int(color[2] * 255)
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {rgb}\n")
        
        logger.info(f"Saved PCD to {filepath}")
    
    def reset(self):
        """Reset all views and reconstruction."""
        self.views = []
        self.point_cloud = None
        self.colors = None
        logger.info("Reconstructor reset")
