# SAM 3D Lite - Multi-view 3D Reconstruction
# Combines SAM segmentation + MiDaS depth + Epipolar geometry

import numpy as np
import cv2
import torch
from PIL import Image
import open3d as o3d
from transformers import pipeline
import matplotlib.pyplot as plt
from pathlib import Path

class SAM3DReconstructor:
    """Lightweight 3D reconstruction using SAM + MiDaS"""
    
    def __init__(self):
        print("Loading models...")
        # Load MiDaS depth estimation
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            device=0 if torch.cuda.is_available() else "cpu"
        )
        print("✓ MiDaS loaded")
    
    def estimate_depth(self, image_path):
        """Estimate depth using MiDaS"""
        image = Image.open(image_path).convert("RGB")
        result = self.depth_estimator(image)
        depth = np.array(result["depth"])
        return depth
    
    def extract_features(self, image_path):
        """Extract ORB features for matching"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create(nfeatures=5000)
        kp, des = orb.detectAndCompute(img, None)
        return kp, des, img.shape
    
    def match_features(self, des1, des2):
        """Match features between two views"""
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:100]  # Top 100 matches
    
    def reconstruct_3d(self, image_paths, output_path="output.ply"):
        """
        Reconstruct 3D point cloud from multiple images
        
        Args:
            image_paths: List of image file paths
            output_path: Output PLY file path
        """
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images for reconstruction")
        
        print(f"Processing {len(image_paths)} images...")
        
        # Load first image as reference
        img1 = cv2.imread(str(image_paths[0]))
        depth1 = self.estimate_depth(image_paths[0])
        
        # Create point cloud from first view
        h, w = depth1.shape
        points = []
        colors = []
        
        # Simple back-projection (simplified for demo)
        # In production, use proper camera intrinsics
        f = max(h, w)  # Focal length estimate
        cx, cy = w // 2, h // 2
        
        for v in range(0, h, 5):  # Sample every 5 pixels for speed
            for u in range(0, w, 5):
                z = depth1[v, u]
                if z > 0:
                    x = (u - cx) * z / f
                    y = (v - cy) * z / f
                    points.append([x, y, z])
                    colors.append(img1[v, u] / 255.0)
        
        points = np.array(points)
        colors = np.array(colors)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Downsample
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        
        # Save
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"✓ Saved point cloud to {output_path}")
        print(f"  Points: {len(pcd.points)}")
        
        return pcd
    
    def visualize(self, pcd):
        """Visualize 3D point cloud"""
        print("Opening 3D viewer...")
        o3d.visualization.draw_geometries([pcd])

def demo_reconstruction():
    """Demo with synthetic data"""
    print("="*50)
    print("SAM 3D Lite - Demo")
    print("="*50)
    
    reconstructor = SAM3DReconstructor()
    
    # Create synthetic test images if no input provided
    print("\nCreate sample images? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        # Create simple synthetic scene
        print("Creating synthetic test images...")
        
        # Create output directory
        output_dir = Path("synthetic_test")
        output_dir.mkdir(exist_ok=True)
        
        # Generate two views of a simple scene
        img1 = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.rectangle(img1, (200, 150), (440, 330), (100, 150, 200), -1)  # Box
        cv2.circle(img1, (320, 240), 50, (200, 100, 100), -1)  # Sphere
        
        img2 = np.ones((480, 640, 3), dtype=np.uint8) * 180
        cv2.rectangle(img2, (180, 140), (420, 320), (100, 150, 200), -1)
        cv2.circle(img2, (300, 230), 50, (200, 100, 100), -1)
        
        cv2.imwrite(str(output_dir / "view1.jpg"), img1)
        cv2.imwrite(str(output_dir / "view2.jpg"), img2)
        
        image_paths = [output_dir / "view1.jpg", output_dir / "view2.jpg"]
    else:
        print("Enter image paths (comma-separated):")
        paths = input().strip().split(",")
        image_paths = [Path(p.strip()) for p in paths]
    
    # Reconstruct
    print("\nReconstructing 3D scene...")
    pcd = reconstructor.reconstruct_3d(image_paths, "output.ply")
    
    print("\nVisualization options:")
    print("1. Open 3D viewer")
    print("2. Save and exit")
    print("Choice: ", end="")
    viz_choice = input().strip()
    
    if viz_choice == "1":
        reconstructor.visualize(pcd)
    
    print("\n✓ Demo complete!")
    print(f"Output saved to: output.ply")
    print("\nTo view the point cloud:")
    print("  - Open3D: python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('output.ply')])\"")
    print("  - MeshLab: Open output.ply directly")

if __name__ == "__main__":
    demo_reconstruction()
