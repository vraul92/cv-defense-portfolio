"""
Extended Kalman Filter Implementation for Object Tracking

This module implements a full EKF from scratch for bounding box tracking.
No external Kalman filter libraries used - pure NumPy implementation.

Author: Rahul Vuppalapati
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for bounding box tracking.
    
    State Vector (6x1): [x, y, w, h, dx, dy]^T
        x, y: Center position of bounding box
        w, h: Width and height of bounding box
        dx, dy: Velocity in x and y directions
    
    Measurement Vector (4x1): [x, y, w, h]^T
        Direct observation of bounding box from detector
    """
    
    def __init__(self,
                 initial_state: np.ndarray,
                 process_noise: float = 0.05,
                 measurement_noise: float = 0.1,
                 initial_covariance: float = 1.0):
        """
        Initialize EKF.
        
        Args:
            initial_state: Initial state vector [x, y, w, h, dx, dy]
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
            initial_covariance: Initial state covariance (P)
        """
        # State vector
        self.x = initial_state.reshape(6, 1).astype(np.float32)
        
        # State covariance matrix (6x6)
        self.P = np.eye(6, dtype=np.float32) * initial_covariance
        
        # Process noise covariance (6x6)
        self.Q = np.eye(6, dtype=np.float32) * process_noise
        
        # Measurement noise covariance (4x4)
        self.R = np.eye(4, dtype=np.float32) * measurement_noise
        
        # Time step
        self.dt = 1.0
        
        logger.debug(f"EKF initialized with state: {self.x.flatten()}")
    
    def predict(self) -> np.ndarray:
        """
        Prediction step - predict next state using motion model.
        
        Uses constant velocity model:
            x_{k+1} = x_k + dx_k * dt
            y_{k+1} = y_k + dy_k * dt
            w_{k+1} = w_k
            h_{k+1} = h_k
            dx_{k+1} = dx_k
            dy_{k+1} = dy_k
        
        Returns:
            Predicted state vector
        """
        # State transition matrix (Jacobian of motion model) - 6x6
        F = np.array([
            [1, 0, 0, 0, self.dt, 0],
            [0, 1, 0, 0, 0, self.dt],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Predict state: x_pred = F @ x
        self.x = F @ self.x
        
        # Predict covariance: P_pred = F @ P @ F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        # Ensure covariance stays symmetric
        self.P = (self.P + self.P.T) / 2
        
        return self.x.flatten()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step - correct prediction with measurement.
        
        Args:
            measurement: Measurement vector [x, y, w, h]
        
        Returns:
            Updated state vector
        """
        measurement = measurement.reshape(4, 1).astype(np.float32)
        
        # Measurement matrix (Jacobian of measurement function) - 4x6
        # We measure x, y, w, h directly
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Predicted measurement: z_pred = H @ x
        z_pred = H @ self.x
        
        # Innovation (measurement residual): y = z - z_pred
        innovation = measurement - z_pred
        
        # Innovation covariance: S = H @ P @ H^T + R
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain: K = P @ H^T @ S^(-1)
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in Kalman gain computation")
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        # Update state: x_new = x + K @ y
        self.x = self.x + K @ innovation
        
        # Update covariance: P_new = (I - K @ H) @ P
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        
        # Ensure covariance stays symmetric and positive semi-definite
        self.P = (self.P + self.P.T) / 2
        
        return self.x.flatten()
    
    def get_state(self) -> dict:
        """
        Get current state as dictionary.
        
        Returns:
            Dictionary with position, size, velocity, and uncertainty
        """
        x, y, w, h, dx, dy = self.x.flatten()
        
        # Get position uncertainty (2x2 covariance submatrix)
        pos_uncertainty = self.P[:2, :2]
        
        return {
            'x': float(x),
            'y': float(y),
            'w': float(w),
            'h': float(h),
            'dx': float(dx),
            'dy': float(dy),
            'bbox': self._state_to_bbox(),
            'uncertainty': pos_uncertainty,
            'trace_covariance': float(np.trace(self.P))
        }
    
    def _state_to_bbox(self) -> Tuple[int, int, int, int]:
        """
        Convert state vector to bounding box (x1, y1, x2, y2).
        
        Returns:
            Bounding box coordinates
        """
        x, y, w, h, _, _ = self.x.flatten()
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        return (x1, y1, x2, y2)
    
    def get_uncertainty_ellipse(self, n_std: float = 2.0) -> Tuple[float, float, float]:
        """
        Get parameters for uncertainty ellipse visualization.
        
        Args:
            n_std: Number of standard deviations for ellipse
        
        Returns:
            (center_x, center_y), (width, height), angle_degrees
        """
        # Position covariance (2x2)
        cov = self.P[:2, :2]
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Calculate ellipse parameters
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        center_x, center_y = self.x[0, 0], self.x[1, 0]
        
        return (center_x, center_y), (width, height), angle
    
    def mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance between measurement and prediction.
        
        Used for data association - matching detections to tracks.
        
        Args:
            measurement: Measurement vector [x, y, w, h]
        
        Returns:
            Mahalanobis distance
        """
        measurement = measurement.reshape(4, 1).astype(np.float32)
        
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)
        
        z_pred = H @ self.x
        innovation = measurement - z_pred
        
        S = H @ self.P @ H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            distance = np.sqrt(innovation.T @ S_inv @ innovation)
            return float(distance)
        except np.linalg.LinAlgError:
            return float('inf')
