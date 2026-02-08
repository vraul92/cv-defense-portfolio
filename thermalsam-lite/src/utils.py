"""
Utility functions for ThermalSAM-Lite.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def resize_image(image: np.ndarray, 
                 max_size: int = 1024) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension
        
    Returns:
        Resized image and scale factor
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image, 1.0
    
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized, scale


def enhance_contrast(image: np.ndarray, 
                     clip_limit: float = 2.0, 
                     grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.
    
    Args:
        image: Input grayscale image
        clip_limit: CLAHE clip limit
        grid_size: CLAHE grid size
        
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(image.astype(np.uint8))
    
    return enhanced


def denoise_thermal(image: np.ndarray, 
                    h: float = 10, 
                    template_window: int = 7,
                    search_window: int = 21) -> np.ndarray:
    """
    Denoise thermal image using Non-local Means Denoising.
    
    Args:
        image: Input image
        h: Filter strength
        template_window: Template patch size
        search_window: Search window size
        
    Returns:
        Denoised image
    """
    if image.dtype == np.uint16:
        # Convert to 8-bit for denoising
        image_8bit = (image / 256).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(image_8bit, None, h, 
                                           template_window, search_window)
        # Convert back
        return (denoised.astype(np.uint16) * 256)
    else:
        return cv2.fastNlMeansDenoising(image.astype(np.uint8), None, h,
                                       template_window, search_window)


def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate image noise using Laplacian variance.
    
    Args:
        image: Input image
        
    Returns:
        Noise estimate (lower is less noisy)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    
    return float(np.var(laplacian))


def get_image_stats(image: np.ndarray) -> dict:
    """
    Get image statistics.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(image.min()),
        'max': float(image.max()),
        'mean': float(image.mean()),
        'std': float(image.std()),
    }
    
    if len(image.shape) == 2:
        stats['channels'] = 1
    else:
        stats['channels'] = image.shape[2]
    
    return stats


def save_results(results: list, 
                 output_path: str,
                 format: str = 'txt'):
    """
    Save detection results to file.
    
    Args:
        results: List of DetectionResult objects
        output_path: Output file path
        format: Output format ('txt', 'json', 'csv')
    """
    if format == 'txt':
        with open(output_path, 'w') as f:
            f.write("Detection Results\n")
            f.write("=" * 50 + "\n\n")
            
            for i, r in enumerate(results):
                f.write(f"Object {i+1}:\n")
                f.write(f"  Class: {r.class_name}\n")
                f.write(f"  Confidence: {r.confidence:.3f}\n")
                f.write(f"  Temperature: {r.temperature:.2f}°C\n")
                f.write(f"  Area: {r.area} pixels\n")
                f.write(f"  Bounding Box: {r.bbox}\n")
                f.write("\n")
    
    elif format == 'json':
        import json
        data = []
        for r in results:
            data.append({
                'class': r.class_name,
                'confidence': r.confidence,
                'temperature': r.temperature,
                'area': r.area,
                'bbox': r.bbox
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format == 'csv':
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'class', 'confidence', 'temperature', 
                           'area', 'x1', 'y1', 'x2', 'y2'])
            
            for i, r in enumerate(results):
                x1, y1, x2, y2 = r.bbox
                writer.writerow([i+1, r.class_name, r.confidence, r.temperature,
                               r.area, x1, y1, x2, y2])
    
    logger.info(f"Saved results to {output_path}")
