"""
Visualization utilities for Ghost Tracker.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from .tracker import Track


def draw_bbox(image: np.ndarray,
              bbox: Tuple[int, int, int, int],
              track_id: int,
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2,
              confidence: float = 1.0,
              is_ghost: bool = False) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        track_id: Track ID
        color: Box color (BGR)
        thickness: Line thickness
        confidence: Detection confidence
        is_ghost: Whether this is a ghost track
    
    Returns:
        Image with drawn bbox
    """
    x1, y1, x2, y2 = bbox
    
    if is_ghost:
        # Ghost box - dashed line effect
        draw_dashed_rectangle(image, (x1, y1), (x2, y2), color, thickness)
        alpha = 0.5
    else:
        # Normal box - solid line
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        alpha = 1.0
    
    # Draw label
    label = f"ID: {track_id}"
    if confidence < 1.0:
        label += f" ({confidence:.2f})"
    
    if is_ghost:
        label += " [GHOST]"
    
    # Label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_label = max(y1, label_size[1] + 10)
    
    cv2.rectangle(image, 
                 (x1, y_label - label_size[1] - 5),
                 (x1 + label_size[0], y_label + 5),
                 color, -1)
    
    # Label text
    cv2.putText(image, label, (x1, y_label),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def draw_dashed_rectangle(image: np.ndarray,
                          pt1: Tuple[int, int],
                          pt2: Tuple[int, int],
                          color: Tuple[int, int, int],
                          thickness: int = 2,
                          dash_length: int = 10):
    """Draw dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw dashed lines
    for i in range(x1, x2, dash_length * 2):
        cv2.line(image, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        cv2.line(image, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
    
    for i in range(y1, y2, dash_length * 2):
        cv2.line(image, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        cv2.line(image, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)


def draw_uncertainty_ellipse(image: np.ndarray,
                             track: Track,
                             color: Tuple[int, int, int] = (255, 255, 0),
                             n_std: float = 2.0):
    """
    Draw uncertainty ellipse for track.
    
    Args:
        image: Input image
        track: Track object
        color: Ellipse color
        n_std: Number of standard deviations
    """
    try:
        center, axes, angle = track.get_uncertainty_ellipse()
        
        center = (int(center[0]), int(center[1]))
        axes = (int(axes[0] / 2), int(axes[1] / 2))
        
        cv2.ellipse(image, center, axes, angle, 0, 360, color, 2)
        
        # Draw center point
        cv2.circle(image, center, 3, color, -1)
        
    except Exception as e:
        # Silently skip if ellipse can't be drawn
        pass


def draw_trajectory(image: np.ndarray,
                    history: List[Tuple[int, int, int, int]],
                    color: Tuple[int, int, int] = (0, 255, 255),
                    max_points: int = 30):
    """
    Draw trajectory trail.
    
    Args:
        image: Input image
        history: List of past bounding boxes
        color: Trajectory color
        max_points: Maximum number of points to draw
    """
    if len(history) < 2:
        return
    
    # Get centers from bboxes
    points = []
    for bbox in history[-max_points:]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        points.append((cx, cy))
    
    # Draw trajectory with fading effect
    for i in range(1, len(points)):
        alpha = i / len(points)
        thickness = max(1, int(alpha * 3))
        color_fade = tuple(int(c * alpha) for c in color)
        cv2.line(image, points[i-1], points[i], color_fade, thickness)


def draw_velocity_vector(image: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         velocity: Tuple[float, float],
                         scale: float = 5.0,
                         color: Tuple[int, int, int] = (0, 0, 255)):
    """
    Draw velocity vector arrow.
    
    Args:
        image: Input image
        bbox: Bounding box
        velocity: (dx, dy) velocity
        scale: Scale factor for arrow length
        color: Arrow color
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    end_x = int(cx + velocity[0] * scale)
    end_y = int(cy + velocity[1] * scale)
    
    cv2.arrowedLine(image, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)


def visualize_tracks(image: np.ndarray,
                    tracks: List[Track],
                    show_ghosts: bool = True,
                    show_uncertainty: bool = True,
                    show_trajectories: bool = True,
                    show_velocity: bool = False) -> np.ndarray:
    """
    Visualize all tracks on image.
    
    Args:
        image: Input image
        tracks: List of tracks
        show_ghosts: Whether to show ghost tracks
        show_uncertainty: Whether to show uncertainty ellipses
        show_trajectories: Whether to show trajectories
        show_velocity: Whether to show velocity vectors
    
    Returns:
        Visualization image
    """
    output = image.copy()
    
    # Assign colors to track IDs
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 255, 0),  # Lime
        (255, 128, 0),  # Orange
    ]
    
    for track in tracks:
        state = track.get_state()
        track_id = state['track_id']
        is_ghost = state['is_ghost']
        bbox = state['bbox']
        confidence = state['confidence']
        
        # Skip ghost tracks if not showing
        if is_ghost and not show_ghosts:
            continue
        
        # Get color (cycle through palette)
        color = colors[track_id % len(colors)]
        
        # Draw bbox
        draw_bbox(output, bbox, track_id, color, 
                 confidence=confidence, is_ghost=is_ghost)
        
        # Draw uncertainty ellipse
        if show_uncertainty and is_ghost:
            draw_uncertainty_ellipse(output, track, color)
        
        # Draw trajectory
        if show_trajectories and len(state['history']) > 1:
            draw_trajectory(output, state['history'], color)
        
        # Draw velocity
        if show_velocity:
            velocity = (state['dx'], state['dy'])
            draw_velocity_vector(output, bbox, velocity, color=color)
    
    # Draw statistics
    n_confirmed = sum(1 for t in tracks if not t.is_ghost)
    n_ghosts = sum(1 for t in tracks if t.is_ghost)
    
    stats_text = f"Tracks: {n_confirmed} | Ghosts: {n_ghosts}"
    cv2.putText(output, stats_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output
