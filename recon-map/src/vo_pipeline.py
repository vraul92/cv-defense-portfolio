# Visual Odometry Core Implementation
"""
Real-time visual odometry for GPS-denied navigation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Single frame with features."""
    id: int
    image: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    pose: Optional[np.ndarray] = None  # 4x4 transformation matrix


class FeatureExtractor:
    """Extract ORB features from images."""
    
    def __init__(self, n_features: int = 2000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def extract(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract keypoints and descriptors."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors


class FeatureMatcher:
    """Match features between frames."""
    
    def __init__(self, ratio_threshold: float = 0.7):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_threshold = ratio_threshold
    
    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """Find good matches between descriptors."""
        if desc1 is None or desc2 is None:
            return []
        
        # KNN match
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches


class MotionEstimator:
    """Estimate camera motion from feature matches."""
    
    def __init__(self, focal_length: float = 800.0, 
                 principal_point: Tuple[float, float] = (320.0, 240.0)):
        """
        Args:
            focal_length: Camera focal length in pixels
            principal_point: Principal point (cx, cy)
        """
        self.K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.focal = focal_length
        self.pp = principal_point
    
    def estimate_motion(self, kp1: List[cv2.KeyPoint], 
                       kp2: List[cv2.KeyPoint],
                       matches: List[cv2.DMatch]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate relative motion between two frames.
        
        Returns:
            (R, t) rotation matrix and translation vector
        """
        if len(matches) < 8:
            return None
        
        # Get matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.focal, self.pp,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is None:
            return None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        return R, t


class VisualOdometry:
    """Main visual odometry pipeline."""
    
    def __init__(self, focal_length: float = 800.0):
        self.feature_extractor = FeatureExtractor()
        self.feature_matcher = FeatureMatcher()
        self.motion_estimator = MotionEstimator(focal_length=focal_length)
        
        self.prev_frame: Optional[Frame] = None
        self.trajectory: List[np.ndarray] = []
        self.current_pose = np.eye(4)
        
        self.frame_id = 0
    
    def process_frame(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a new frame and update trajectory.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Current pose (4x4 matrix) or None
        """
        # Extract features
        keypoints, descriptors = self.feature_extractor.extract(image)
        
        # Create frame
        frame = Frame(
            id=self.frame_id,
            image=image,
            keypoints=keypoints,
            descriptors=descriptors,
            pose=self.current_pose.copy()
        )
        
        if self.prev_frame is not None:
            # Match features
            matches = self.feature_matcher.match(
                self.prev_frame.descriptors,
                frame.descriptors
            )
            
            # Estimate motion
            motion = self.motion_estimator.estimate_motion(
                self.prev_frame.keypoints,
                frame.keypoints,
                matches
            )
            
            if motion is not None:
                R, t = motion
                
                # Update pose: T = T_prev * [R|t]
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()
                
                self.current_pose = self.current_pose @ T
                frame.pose = self.current_pose.copy()
                
                # Add to trajectory
                self.trajectory.append(self.current_pose[:3, 3].copy())
        
        self.prev_frame = frame
        self.frame_id += 1
        
        return self.current_pose
    
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory as Nx3 array."""
        if not self.trajectory:
            return np.array([])
        return np.array(self.trajectory)
    
    def reset(self):
        """Reset VO state."""
        self.prev_frame = None
        self.trajectory = []
        self.current_pose = np.eye(4)
        self.frame_id = 0
