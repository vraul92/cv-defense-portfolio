"""
Ghost Tracker - Multi-Object Tracking with Occlusion Handling

Main tracker class that manages multiple tracks, handles data association,
and implements ghost mode for occlusions.

Author: Rahul Vuppalapati
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

from .kalman_filter import ExtendedKalmanFilter

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """
    Single track representation.
    """
    track_id: int
    kalman_filter: ExtendedKalmanFilter
    
    # Track state
    hits: int = 0                    # Number of successful detections
    time_since_update: int = 0       # Frames since last detection
    is_ghost: bool = False           # Is this a ghost track?
    
    # History
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Detection info
    last_detection: Optional[np.ndarray] = None
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    
    # Appearance features (for re-identification)
    appearance_features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize track."""
        self.age = 0
        self.start_frame = 0
    
    def predict(self) -> np.ndarray:
        """Predict next state using Kalman filter."""
        return self.kalman_filter.predict()
    
    def update(self, detection: np.ndarray, confidence: float = 1.0):
        """Update track with new detection."""
        self.kalman_filter.update(detection)
        self.hits += 1
        self.time_since_update = 0
        self.is_ghost = False
        self.confidence = confidence
        self.last_detection = detection
        
        # Update history
        state = self.kalman_filter.get_state()
        self.history.append(state['bbox'])
        self.last_bbox = state['bbox']
    
    def mark_missed(self):
        """Mark track as missed (no detection this frame)."""
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.is_ghost = True
    
    def get_state(self) -> Dict:
        """Get current track state."""
        state = self.kalman_filter.get_state()
        state.update({
            'track_id': self.track_id,
            'is_ghost': self.is_ghost,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'age': self.age,
            'confidence': self.confidence,
            'history': list(self.history)
        })
        return state
    
    def get_uncertainty_ellipse(self) -> Tuple:
        """Get uncertainty ellipse for visualization."""
        return self.kalman_filter.get_uncertainty_ellipse(n_std=2.0)


class GhostTracker:
    """
    Multi-object tracker with occlusion handling (ghost mode).
    
    This tracker uses:
    - Extended Kalman Filter for state estimation
    - Mahalanobis distance for data association
    - Ghost mode for handling occlusions
    """
    
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 mahalanobis_threshold: float = 9.4877,  # 95% confidence for 4-DOF
                 process_noise: float = 0.05,
                 measurement_noise: float = 0.1):
        """
        Initialize Ghost Tracker.
        
        Args:
            max_age: Maximum frames to keep ghost track alive
            min_hits: Minimum detections to confirm track
            iou_threshold: IoU threshold for initial association
            mahalanobis_threshold: Threshold for Mahalanobis distance
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.mahalanobis_threshold = mahalanobis_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Track management
        self.tracks: List[Track] = []
        self.track_count = 0
        self.frame_count = 0
        
        logger.info(f"GhostTracker initialized: max_age={max_age}, min_hits={min_hits}")
    
    def update(self, detections: List[np.ndarray], 
               confidences: Optional[List[float]] = None) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection bounding boxes [x, y, w, h, confidence]
            confidences: Optional list of detection confidences
        
        Returns:
            List of confirmed tracks
        """
        self.frame_count += 1
        
        if confidences is None:
            confidences = [1.0] * len(detections)
        
        # Step 1: Predict all existing tracks
        for track in self.tracks:
            track.predict()
            track.age += 1
        
        # Step 2: Associate detections to tracks
        matches, unmatched_detections, unmatched_tracks = self._associate(
            detections, self.tracks
        )
        
        # Step 3: Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], confidences[det_idx])
        
        # Step 4: Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Step 5: Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._initiate_track(detections[det_idx], confidences[det_idx])
        
        # Step 6: Remove dead tracks
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update <= self.max_age]
        
        # Step 7: Return confirmed tracks
        confirmed_tracks = [t for t in self.tracks 
                          if t.hits >= self.min_hits or t.time_since_update == 0]
        
        return confirmed_tracks
    
    def _associate(self, detections: List[np.ndarray], 
                  tracks: List[Track]) -> Tuple[List, List, List]:
        """
        Associate detections to tracks using Mahalanobis distance.
        
        Args:
            detections: List of detection vectors
            tracks: List of track objects
        
        Returns:
            matches: List of (track_idx, det_idx) tuples
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Compute cost matrix using Mahalanobis distance
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Use Mahalanobis distance
                distance = track.kalman_filter.mahalanobis_distance(det)
                cost_matrix[i, j] = distance
        
        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for track_idx, det_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, det_idx] < self.mahalanobis_threshold:
                matches.append((track_idx, det_idx))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _initiate_track(self, detection: np.ndarray, confidence: float = 1.0):
        """
        Initialize new track from detection.
        
        Args:
            detection: Detection bounding box [x, y, w, h]
            confidence: Detection confidence
        """
        # Convert bbox to state vector [x, y, w, h, dx, dy]
        x, y, w, h = detection[:4]
        initial_state = np.array([x, y, w, h, 0, 0])  # Zero initial velocity
        
        # Create Kalman filter
        kf = ExtendedKalmanFilter(
            initial_state=initial_state,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        
        # Create track
        self.track_count += 1
        track = Track(
            track_id=self.track_count,
            kalman_filter=kf,
            hits=1,
            confidence=confidence,
            last_detection=detection
        )
        track.start_frame = self.frame_count
        
        self.tracks.append(track)
        logger.debug(f"Initiated new track {track.track_id}")
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (including tentative)."""
        return self.tracks
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks if t.hits >= self.min_hits]
    
    def get_ghost_tracks(self) -> List[Track]:
        """Get tracks currently in ghost mode."""
        return [t for t in self.tracks if t.is_ghost]
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_count = 0
        self.frame_count = 0
        logger.info("GhostTracker reset")
