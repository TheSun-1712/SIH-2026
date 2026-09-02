"""
Shared Data Contract: Frame & Telemetry dataclasses for AERO MESH pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np

@dataclass
class CameraPose:
    """Represents a 6-DoF camera pose in world coordinates."""
    position: np.ndarray          # 3-element vector [x, y, z] in meters
    orientation: np.ndarray       # 4-element quaternion [w, x, y, z] or 3x3 rotation matrix
    covariance: Optional[np.ndarray] = None  # 6x6 pose covariance matrix

    def transform_matrix(self) -> np.ndarray:
        """Returns 4x4 homogenous transformation matrix (Camera to World)."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self.position
        
        # If quaternion
        if self.orientation.size == 4:
            w, x, y, z = self.orientation
            # Normalize quaternion
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            if norm > 0:
                w, x, y, z = w/norm, x/norm, y/norm, z/norm
            R = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
                [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
            ], dtype=np.float64)
            T[:3, :3] = R
        elif self.orientation.shape == (3, 3):
            T[:3, :3] = self.orientation
        return T

@dataclass
class Frame:
    """Core Frame data structure passed across all 8 pipeline modules."""
    frame_id: int
    timestamp: float                       # Timestamp in seconds
    image_path: Optional[str] = None       # Path to saved frame image
    image_data: Optional[np.ndarray] = None # RGB image array (H, W, 3)
    sharpness_score: float = 0.0           # Variance of Laplacian score
    estimated_pose: Optional[CameraPose] = None # Fused 6-DoF pose from ESKF
    gps_lat_lon_alt: Optional[Tuple[float, float, float]] = None # (lat, lon, alt)
    is_keyframe: bool = False
    depth_map: Optional[np.ndarray] = None # Monocular / MVS depth map
    depth_confidence: Optional[np.ndarray] = None # Per-pixel depth confidence
    metadata: Dict[str, Any] = field(default_factory=dict)
