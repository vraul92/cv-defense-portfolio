"""
ThermalSAM-Lite: Core Implementation
Zero-shot object detection for thermal imagery using SAM.

Author: Rahul Vuppalapati
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for detection results."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray
    confidence: float
    temperature: Optional[float] = None
    class_name: str = "object"
    area: int = 0


class ThermalPreprocessor:
    """
    Preprocess thermal images for SAM compatibility.
    
    Thermal cameras output 16-bit grayscale, but SAM expects RGB.
    This class handles the conversion with false-color mapping.
    """
    
    def __init__(self, 
                 colormap: int = cv2.COLORMAP_JET,
                 normalize: bool = True,
                 temperature_range: Tuple[float, float] = (20.0, 150.0)):
        """
        Args:
            colormap: OpenCV colormap for false-color conversion
            normalize: Whether to normalize intensity values
            temperature_range: Min/max temperature in Celsius for calibration
        """
        self.colormap = colormap
        self.normalize = normalize
        self.temperature_range = temperature_range
        
    def thermal_to_rgb(self, thermal_img: np.ndarray) -> np.ndarray:
        """
        Convert 16-bit thermal grayscale to 3-channel RGB.
        
        Args:
            thermal_img: Input thermal image (H, W) or (H, W, 1)
            
        Returns:
            RGB image (H, W, 3) normalized for SAM
        """
        # Ensure grayscale
        if len(thermal_img.shape) == 3:
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 0-255 range
        if self.normalize:
            # Handle 16-bit input
            if thermal_img.dtype == np.uint16:
                # Map to effective temperature range
                min_val = thermal_img.min()
                max_val = thermal_img.max()
                normalized = ((thermal_img - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)
            else:
                normalized = thermal_img.astype(np.uint8)
        else:
            normalized = thermal_img.astype(np.uint8)
        
        # Apply false-color mapping
        colored = cv2.applyColorMap(normalized, self.colormap)
        
        # Convert BGR to RGB for SAM
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return rgb
    
    def calibrate_temperature(self, pixel_value: float, 
                             offset: float = 27315.0,
                             gain: float = 100.0) -> float:
        """
        Convert pixel intensity to temperature (Celsius).
        
        Default parameters calibrated for FLIR Lepton 3.5.
        
        Args:
            pixel_value: Raw pixel intensity
            offset: Sensor offset
            gain: Sensor gain
            
        Returns:
            Temperature in Celsius
        """
        # FLIR calibration formula (simplified)
        temp_kelvin = (pixel_value - offset) / gain + 273.15
        temp_celsius = temp_kelvin - 273.15
        return temp_celsius
    
    def extract_temperature_map(self, thermal_img: np.ndarray) -> np.ndarray:
        """
        Extract temperature map from thermal image.
        
        Args:
            thermal_img: 16-bit thermal image
            
        Returns:
            Temperature map in Celsius
        """
        if thermal_img.dtype == np.uint16:
            # Vectorized calibration
            temp_map = (thermal_img.astype(np.float32) - 27315.0) / 100.0
        else:
            temp_map = thermal_img.astype(np.float32)
        
        return temp_map


class ThermalSAM:
    """
    Main class for thermal object detection using SAM.
    
    This class wraps Meta's SAM model with thermal-specific preprocessing
    and post-processing for defense applications.
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 model_type: str = "vit_b",
                 device: str = "cpu",
                 temperature_range: Tuple[float, float] = (20.0, 150.0),
                 confidence_threshold: float = 0.7):
        """
        Initialize ThermalSAM detector.
        
        Args:
            checkpoint_path: Path to SAM checkpoint (.pth file)
            model_type: SAM model variant (vit_b, vit_l, vit_h)
            device: Device for inference (cpu or cuda)
            temperature_range: Valid temperature range in Celsius
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.model_type = model_type
        self.temperature_range = temperature_range
        self.confidence_threshold = confidence_threshold
        
        # Initialize preprocessor
        self.preprocessor = ThermalPreprocessor(
            colormap=cv2.COLORMAP_JET,
            temperature_range=temperature_range
        )
        
        # Load SAM model
        self._load_model(checkpoint_path)
        
        logger.info(f"ThermalSAM initialized: {model_type} on {device}")
    
    def _load_model(self, checkpoint_path: str):
        """Load SAM model from checkpoint."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load model
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            
            # Create mask generator with optimal settings for thermal
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.9,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            
            self.model = sam
            logger.info(f"Loaded SAM model from {checkpoint_path}")
            
        except ImportError:
            logger.error("segment_anything not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git")
            raise
    
    def detect(self, 
               image: np.ndarray,
               min_area: int = 100,
               max_area: Optional[int] = None) -> List[DetectionResult]:
        """
        Detect objects in thermal image.
        
        Args:
            image: Input thermal image (grayscale or BGR)
            min_area: Minimum detection area in pixels
            max_area: Maximum detection area in pixels
            
        Returns:
            List of DetectionResult objects
        """
        # Store original for temperature extraction
        if len(image.shape) == 3:
            original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = image.copy()
        
        # Convert to RGB for SAM
        rgb_image = self.preprocessor.thermal_to_rgb(image)
        
        # Generate masks with SAM
        masks = self.mask_generator.generate(rgb_image)
        
        # Post-process results
        results = []
        for mask_data in masks:
            # Extract mask
            mask = mask_data['segmentation'].astype(np.uint8)
            
            # Calculate area
            area = int(mask_data['area'])
            
            # Filter by area
            if area < min_area:
                continue
            if max_area and area > max_area:
                continue
            
            # Get confidence
            confidence = mask_data.get('stability_score', mask_data.get('predicted_iou', 0.5))
            
            if confidence < self.confidence_threshold:
                continue
            
            # Get bounding box
            bbox = mask_data['bbox']  # x, y, w, h format from SAM
            x, y, w, h = bbox
            bbox_xyxy = (int(x), int(y), int(x + w), int(y + h))
            
            # Extract temperature
            temp_map = self.preprocessor.extract_temperature_map(original_gray)
            masked_temp = temp_map[mask > 0]
            avg_temperature = float(np.mean(masked_temp)) if len(masked_temp) > 0 else None
            
            # Filter by temperature range
            if avg_temperature is not None:
                if not (self.temperature_range[0] <= avg_temperature <= self.temperature_range[1]):
                    continue
            
            # Classify object type based on temperature and size
            class_name = self._classify_object(area, avg_temperature)
            
            result = DetectionResult(
                bbox=bbox_xyxy,
                mask=mask,
                confidence=confidence,
                temperature=avg_temperature,
                class_name=class_name,
                area=area
            )
            results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Detected {len(results)} objects")
        return results
    
    def _classify_object(self, area: int, temperature: Optional[float]) -> str:
        """
        Simple heuristic classification based on size and temperature.
        
        Args:
            area: Object area in pixels
            temperature: Average temperature in Celsius
            
        Returns:
            Class name string
        """
        # Size-based classification (approximate for 512x512 images)
        if temperature is not None:
            if temperature > 60:
                if area > 50000:
                    return "vehicle_hot"
                elif area > 5000:
                    return "human"
                else:
                    return "heat_source"
            else:
                if area > 50000:
                    return "vehicle"
                elif area > 5000:
                    return "human_cool"
                else:
                    return "object"
        else:
            if area > 50000:
                return "large_object"
            elif area > 5000:
                return "medium_object"
            else:
                return "small_object"
    
    def visualize(self,
                  image: np.ndarray,
                  results: List[DetectionResult],
                  mode: str = "combined",
                  save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detection results.
        
        Args:
            image: Original thermal image
            results: Detection results from detect()
            mode: Visualization mode ("combined", "mask", "bbox", "heatmap")
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Convert to displayable format
        if len(image.shape) == 2:
            display_img = self.preprocessor.thermal_to_rgb(image)
        else:
            display_img = image.copy()
        
        if mode == "combined":
            output = self._viz_combined(display_img, results)
        elif mode == "mask":
            output = self._viz_masks(display_img, results)
        elif mode == "bbox":
            output = self._viz_bboxes(display_img, results)
        elif mode == "heatmap":
            output = self._viz_heatmap(image, results)
        else:
            output = display_img
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization to {save_path}")
        
        return output
    
    def _viz_combined(self, image: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """Combined visualization with masks and bounding boxes."""
        output = image.copy()
        h, w = image.shape[:2]
        
        # Create overlay for masks
        overlay = np.zeros_like(image)
        
        # Color map for different classes
        colors = {
            "human": (0, 255, 0),  # Green
            "human_cool": (0, 200, 0),
            "vehicle": (255, 0, 0),  # Blue
            "vehicle_hot": (255, 100, 0),
            "heat_source": (255, 0, 255),  # Magenta
            "object": (255, 255, 0),  # Yellow
        }
        
        for result in results:
            # Get color
            color = colors.get(result.class_name, (255, 255, 255))
            
            # Draw mask
            mask = result.mask
            overlay[mask > 0] = color
            
            # Draw bounding box
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{result.class_name}: {result.confidence:.2f}"
            if result.temperature is not None:
                label += f" | {result.temperature:.1f}°C"
            
            # Draw label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(y1, label_size[1] + 10)
            cv2.rectangle(output, (x1, y_label - label_size[1] - 10), 
                         (x1 + label_size[0], y_label), color, -1)
            cv2.putText(output, label, (x1, y_label - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend overlay
        output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
        
        return output
    
    def _viz_masks(self, image: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """Visualization showing only segmentation masks."""
        overlay = np.zeros_like(image)
        
        for i, result in enumerate(results):
            # Assign color based on index
            color = tuple(np.random.randint(100, 255, 3).tolist())
            overlay[result.mask > 0] = color
        
        # Blend
        output = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)
        return output
    
    def _viz_bboxes(self, image: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """Visualization showing only bounding boxes."""
        output = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result.bbox
            color = (0, 255, 0) if "human" in result.class_name else (255, 0, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            label = f"{result.class_name}: {result.confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    def _viz_heatmap(self, image: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """Visualization with temperature heatmap overlay."""
        # Create temperature map
        temp_map = self.preprocessor.extract_temperature_map(image)
        
        # Normalize for display
        temp_norm = ((temp_map - temp_map.min()) / 
                    (temp_map.max() - temp_map.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(temp_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
