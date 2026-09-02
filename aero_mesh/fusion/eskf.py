"""
Stage 2: Extended Error-State Kalman Filter (ESKF) Sensor Fusion
Fuses high-rate IMU (accel + gyro) with lower-rate GPS & Barometric readings to estimate smooth 6-DoF camera poses.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from aero_mesh.core.frame import CameraPose, Frame

class ESKFFusion:
    """15-State Error-State Kalman Filter for UAV pose tracking."""

    def __init__(self):
        # Nominal state vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bgx, bgy, bgz, bax, bay, baz]
        self.p = np.zeros(3)  # Position in meters
        self.v = np.zeros(3)  # Velocity in m/s
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # Quaternion [w, x, y, z]
        self.bg = np.zeros(3) # Gyro bias
        self.ba = np.zeros(3) # Accel bias

        # Error state covariance matrix P (15x15)
        self.P = np.eye(15, dtype=np.float64) * 0.1

        # Process noise Q and Measurement noise R
        self.Q = np.eye(15, dtype=np.float64) * 0.01
        self.R_gps = np.eye(3, dtype=np.float64) * 0.5  # GPS noise in meters

        self.gravity = np.array([0.0, 0.0, -9.81])

    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """High-rate propagation using accelerometer and gyroscope measurements."""
        if dt <= 0:
            return

        # Unbias measurements
        unbiased_accel = accel - self.ba
        unbiased_gyro = gyro - self.bg

        # Quaternion rotation matrix
        w, x, y, z = self.q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])

        # State integration
        accel_world = R @ unbiased_accel + self.gravity
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v = self.v + accel_world * dt

        # Update orientation quaternion via gyro integration
        omega = unbiased_gyro
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-8:
            axis = omega / np.linalg.norm(omega)
            dq = np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)])
            # Quaternion multiplication dq * q
            w1, x1, y1, z1 = dq
            w2, x2, y2, z2 = self.q
            self.q = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
            self.q /= np.linalg.norm(self.q)

        # Covariance propagation
        self.P += self.Q * dt

    def update_gps(self, gps_pos: np.ndarray):
        """Low-rate measurement update using metric GPS position (x, y, z)."""
        # Innovation
        y_residual = gps_pos - self.p
        
        # Measurement matrix H (3x15) mapping position state error
        H = np.zeros((3, 15))
        H[:3, :3] = np.eye(3)

        # Innovation covariance S
        S = H @ self.P @ H.T + self.R_gps

        # Kalman gain K
        K = self.P @ H.T @ np.linalg.inv(S)

        # Error state correction
        delta_x = K @ y_residual

        # Correct nominal state
        self.p += delta_x[:3]
        self.v += delta_x[3:6]
        
        # Orientation error vector
        delta_theta = delta_x[6:9]
        if np.linalg.norm(delta_theta) > 1e-8:
            angle = np.linalg.norm(delta_theta)
            axis = delta_theta / angle
            dq = np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)])
            # Multiply quaternion
            w1, x1, y1, z1 = dq
            w2, x2, y2, z2 = self.q
            self.q = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
            self.q /= np.linalg.norm(self.q)

        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        # Update covariance
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

    def get_current_pose(self) -> CameraPose:
        """Returns the current estimated 6-DoF pose."""
        return CameraPose(
            position=self.p.copy(),
            orientation=self.q.copy(),
            covariance=self.P[:6, :6].copy()
        )

    def synchronize_frames(self, frames: List[Frame], imu_data: List[Dict], gps_data: List[Dict]) -> List[Frame]:
        """Synchronizes IMU and GPS telemetry stream with frame timestamps."""
        imu_idx = 0
        gps_idx = 0

        # Sort telemetry by timestamp
        imu_sorted = sorted(imu_data, key=lambda x: x["timestamp"])
        gps_sorted = sorted(gps_data, key=lambda x: x["timestamp"])

        last_time = imu_sorted[0]["timestamp"] if imu_sorted else 0.0

        for frame in sorted(frames, key=lambda f: f.timestamp):
            target_t = frame.timestamp

            # Process IMU readings up to target timestamp
            while imu_idx < len(imu_sorted) and imu_sorted[imu_idx]["timestamp"] <= target_t:
                item = imu_sorted[imu_idx]
                dt = item["timestamp"] - last_time
                self.predict(accel=np.array(item["accel"]), gyro=np.array(item["gyro"]), dt=dt)
                last_time = item["timestamp"]
                imu_idx += 1

                # Apply GPS update if available
                while gps_idx < len(gps_sorted) and gps_sorted[gps_idx]["timestamp"] <= item["timestamp"]:
                    g_item = gps_sorted[gps_idx]
                    self.update_gps(gps_pos=np.array(g_item["pos_xyz"]))
                    gps_idx += 1

            frame.estimated_pose = self.get_current_pose()

        return frames
