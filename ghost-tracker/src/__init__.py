"""Ghost-Tracker: Multi-Object Tracking with Occlusion Handling"""

from .kalman_filter import ExtendedKalmanFilter
from .tracker import GhostTracker, Track
from .visualizer import (
    visualize_tracks,
    draw_bbox,
    draw_uncertainty_ellipse,
    draw_trajectory
)

__version__ = "1.0.0"
__author__ = "Rahul Vuppalapati"

__all__ = [
    'ExtendedKalmanFilter',
    'GhostTracker',
    'Track',
    'visualize_tracks',
    'draw_bbox',
    'draw_uncertainty_ellipse',
    'draw_trajectory',
]
