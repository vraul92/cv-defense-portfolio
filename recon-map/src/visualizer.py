"""
Visualization and dashboard for Recon-Map.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Optional


class TrajectoryVisualizer:
    """Real-time trajectory visualization."""
    
    def __init__(self, max_history: int = 500):
        self.trajectory_history = []
        self.max_history = max_history
    
    def update(self, position: np.ndarray):
        """Add new position to trajectory."""
        self.trajectory_history.append(position)
        if len(self.trajectory_history) > self.max_history:
            self.trajectory_history.pop(0)
    
    def draw_2d_topdown(self, size: Tuple[int, int] = (400, 400)) -> np.ndarray:
        """
        Draw 2D top-down view of trajectory.
        
        Returns:
            BGR image
        """
        if len(self.trajectory_history) < 2:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Convert to array
        traj = np.array(self.trajectory_history)
        
        # Project to 2D (X-Z plane for top-down)
        x = traj[:, 0]
        z = traj[:, 2]
        
        # Normalize to image coordinates
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()
        
        # Add padding
        margin = 0.1
        x_range = max(x_max - x_min, 1.0)
        z_range = max(z_max - z_min, 1.0)
        
        x_norm = (x - x_min) / x_range
        z_norm = (z - z_min) / z_range
        
        # Scale to image size
        img_x = (x_norm * (size[0] - 40) + 20).astype(np.int32)
        img_z = (size[1] - 20 - z_norm * (size[1] - 40)).astype(np.int32)
        
        # Create image
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Draw trajectory with gradient
        for i in range(1, len(img_x)):
            color_val = int(255 * i / len(img_x))
            color = (0, color_val, 255 - color_val)
            cv2.line(img, (img_x[i-1], img_z[i-1]), (img_x[i], img_z[i]), color, 2)
        
        # Draw current position
        cv2.circle(img, (img_x[-1], img_z[-1]), 5, (0, 255, 0), -1)
        
        # Add labels
        cv2.putText(img, "Top-Down View (X-Z)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Distance: {np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))):.1f}m",
                   (10, size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img
    
    def draw_3d_view(self, size: Tuple[int, int] = (400, 400)) -> np.ndarray:
        """
        Draw 3D view of trajectory.
        
        Returns:
            BGR image
        """
        if len(self.trajectory_history) < 2:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        traj = np.array(self.trajectory_history)
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
               'b-', linewidth=2, label='Trajectory')
        
        # Plot current position
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                  c='red', s=50, marker='o', label='Current')
        
        # Plot start position
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                  c='green', s=50, marker='s', label='Start')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img


def create_dashboard(frame: np.ndarray,
                    feature_img: np.ndarray,
                    topdown_map: np.ndarray,
                    view3d: np.ndarray) -> np.ndarray:
    """
    Create combined dashboard view.
    
    Args:
        frame: Current video frame
        feature_img: Frame with features drawn
        topdown_map: 2D trajectory map
        view3d: 3D trajectory view
    
    Returns:
        Combined dashboard image
    """
    # Resize all to same height
    h = 400
    
    frame = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
    feature_img = cv2.resize(feature_img, (int(feature_img.shape[1] * h / feature_img.shape[0]), h))
    topdown_map = cv2.resize(topdown_map, (400, h))
    view3d = cv2.resize(view3d, (400, h))
    
    # Create grid
    top_row = np.hstack([frame, feature_img])
    bottom_row = np.hstack([topdown_map, view3d])
    
    dashboard = np.vstack([top_row, bottom_row])
    
    # Add titles
    cv2.putText(dashboard, "Live Feed", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(dashboard, "Feature Tracking", (frame.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(dashboard, "2D Trajectory", (10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(dashboard, "3D View", (410, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return dashboard
