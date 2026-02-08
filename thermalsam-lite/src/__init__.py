"""
ThermalSAM-Lite: Zero-Shot Object Detection for Thermal Imagery
"""

from .thermal_sam import ThermalSAM, ThermalPreprocessor, DetectionResult
from .utils import (
    resize_image,
    enhance_contrast,
    denoise_thermal,
    estimate_noise,
    get_image_stats,
    save_results
)
from .visualization import create_dashboard, create_comparison_grid

__version__ = "1.0.0"
__author__ = "Rahul Vuppalapati"

__all__ = [
    'ThermalSAM',
    'ThermalPreprocessor',
    'DetectionResult',
    'resize_image',
    'enhance_contrast',
    'denoise_thermal',
    'estimate_noise',
    'get_image_stats',
    'save_results',
    'create_dashboard',
    'create_comparison_grid',
]
