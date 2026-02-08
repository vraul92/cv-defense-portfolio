# Visualization utilities for ThermalSAM-Lite
"""
Advanced visualization functions for thermal imaging results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def create_dashboard(image: np.ndarray,
                     results: list,
                     temperature_map: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a comprehensive dashboard view.
    
    Args:
        image: Original thermal image
        results: Detection results
        temperature_map: Optional temperature map
        
    Returns:
        Dashboard image
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Original with detections
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Thermal')
    ax1.axis('off')
    
    # False color
    ax2 = plt.subplot(2, 3, 2)
    false_color = cv2.applyColorMap(
        ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    ax2.imshow(cv2.cvtColor(false_color, cv2.COLOR_BGR2RGB))
    ax2.set_title('False Color')
    ax2.axis('off')
    
    # Temperature map
    if temperature_map is not None:
        ax3 = plt.subplot(2, 3, 3)
        im = ax3.imshow(temperature_map, cmap='hot')
        ax3.set_title('Temperature Map (°C)')
        plt.colorbar(im, ax=ax3)
        ax3.axis('off')
    
    # Detection overlay
    ax4 = plt.subplot(2, 3, 4)
    overlay = cv2.cvtColor(false_color, cv2.COLOR_BGR2RGB)
    for r in results:
        x1, y1, x2, y2 = r.bbox
        color = (0, 255, 0) if 'human' in r.class_name else (255, 0, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    ax4.imshow(overlay)
    ax4.set_title(f'Detections ({len(results)} objects)')
    ax4.axis('off')
    
    # Mask overlay
    ax5 = plt.subplot(2, 3, 5)
    mask_overlay = np.zeros_like(overlay)
    for i, r in enumerate(results):
        color = plt.cm.tab20(i % 20)[:3]
        color = tuple(int(c * 255) for c in color)
        mask_overlay[r.mask > 0] = color
    blended = cv2.addWeighted(overlay, 0.5, mask_overlay, 0.5, 0)
    ax5.imshow(blended)
    ax5.set_title('Segmentation Masks')
    ax5.axis('off')
    
    # Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    Detection Statistics:
    ====================
    Total Objects: {len(results)}
    
    By Class:
    """
    
    class_counts = {}
    for r in results:
        class_counts[r.class_name] = class_counts.get(r.class_name, 0) + 1
    
    for cls, count in class_counts.items():
        stats_text += f"  {cls}: {count}\n"
    
    if results:
        temps = [r.temperature for r in results if r.temperature is not None]
        if temps:
            stats_text += f"""
    Temperature Range:
      Min: {min(temps):.1f}°C
      Max: {max(temps):.1f}°C
      Mean: {np.mean(temps):.1f}°C
            """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    # Convert to image
    fig.canvas.draw()
    dashboard = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    dashboard = dashboard.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return dashboard


def create_comparison_grid(images: list,
                          titles: list,
                          cols: int = 3) -> np.ndarray:
    """
    Create a grid of images for comparison.
    
    Args:
        images: List of images
        titles: List of titles
        cols: Number of columns
        
    Returns:
        Grid image
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        
        if len(img.shape) == 2:
            axes[row, col].imshow(img, cmap='gray')
        else:
            axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    fig.canvas.draw()
    grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return grid
