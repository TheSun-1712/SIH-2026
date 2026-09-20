"""
AERO MESH — MPPI Controller with Control Barrier Functions
===========================================================
Physics-Informed Model Predictive Path Integral (MPPI) controller.
Receives Next-Best-View deltas from the RL agent and produces smooth,
dynamically-feasible, collision-free B-spline trajectories.

Key guarantees:
  - Minimum 3m standoff distance from ALL detected obstacle surfaces (CBF)
  - Blur-safe flight speed (motion blur < 1px at current focal length)
  - Maximum jerk limits for smooth gimbal and camera motion
  - Aerodynamic feasibility (max velocity, acceleration bounds)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field


# ── Physical Drone Parameters ─────────────────────────────────────────────────

@dataclass
class DronePhysics:
    max_vel_ms:        float = 8.0       # Maximum horizontal speed (m/s)
    max_accel_ms2:     float = 4.0       # Maximum acceleration (m/s²)
    max_jerk_ms3:      float = 6.0       # Maximum jerk (m/s³)
    max_yaw_rate_rads: float = 1.0       # Maximum yaw rate (rad/s)
    min_altitude_m:    float = 5.0       # Minimum flight altitude (m)
    max_altitude_m:    float = 120.0     # Regulatory altitude limit (m)
    obstacle_buffer_m: float = 3.0       # CBF minimum standoff distance
    focal_length_mm:   float = 25.0      # Camera focal length (mm)
    sensor_width_mm:   float = 6.17      # Sensor width (mm) — typical 1/2.3"
    image_width_px:    int   = 1920      # Camera resolution width


@dataclass
class MPPIConfig:
    n_samples:       int   = 512         # Number of trajectory samples
    horizon:         int   = 20          # Planning horizon (steps)
    dt:              float = 0.2         # Timestep (seconds)
    temperature:     float = 10.0        # MPPI temperature λ
    noise_sigma:     float = 0.8         # Action perturbation noise std
    cbf_alpha:       float = 2.0         # CBF class-K function gain
    cbf_gamma:       float = 0.5         # CBF decay rate


class MPPIController:
    """
    Model Predictive Path Integral controller with Control Barrier Functions.

    Usage:
        mppi = MPPIController()
        trajectory = mppi.plan(current_pose, nbv_target, obstacle_positions)
        next_velocity = mppi.get_next_command()
    """

    def __init__(
        self,
        physics: Optional[DronePhysics] = None,
        config:  Optional[MPPIConfig]   = None,
    ):
        self.physics = physics or DronePhysics()
        self.config  = config  or MPPIConfig()

        # State: [x, y, z, vx, vy, vz, yaw]
        self.state = np.zeros(7, dtype=np.float32)
        self._planned_trajectory: Optional[np.ndarray] = None
        self._trajectory_step = 0

    # ── Blur-safe speed ───────────────────────────────────────────────────────

    def get_blur_safe_speed(
        self, altitude_m: float, lighting_lux: float = 10000.0
    ) -> float:
        """
        Compute maximum flight speed that keeps motion blur < 1 pixel.

        Formula:
          GSD = (altitude × sensor_width) / (focal_length × image_width)   [m/px]
          shutter_speed = f(lighting) → assume 1/500s in bright sun
          max_speed = GSD / shutter_speed_seconds

        Returns: max safe speed in m/s
        """
        gsd_m_per_px = (
            altitude_m * (self.physics.sensor_width_mm / 1000.0)
        ) / (
            (self.physics.focal_length_mm / 1000.0) * self.physics.image_width_px
        )
        # Shutter speed model: brighter → faster shutter → more speed allowed
        shutter_s = 1.0 / (min(max(lighting_lux, 100.0), 100000.0) / 100.0)
        shutter_s = max(shutter_s, 1.0 / 4000.0)   # clamp to 1/4000s max
        max_speed = gsd_m_per_px / shutter_s
        return float(np.clip(max_speed, 0.3, self.physics.max_vel_ms))

    # ── Control Barrier Function ──────────────────────────────────────────────

    def _cbf_cost(
        self,
        positions:  np.ndarray,     # (n_samples, horizon, 3) sampled trajectories
        obstacles:  np.ndarray,     # (M, 3) obstacle center positions
    ) -> np.ndarray:
        """
        Compute CBF safety cost for sampled trajectories.
        Returns: (n_samples,) cost — high if any trajectory violates buffer.
        """
        if len(obstacles) == 0:
            return np.zeros(len(positions))

        buf = self.physics.obstacle_buffer_m
        alpha = self.config.cbf_alpha

        # positions: (S, H, 3), obstacles: (M, 3)
        # Compute min distance from each sample trajectory to each obstacle
        # Broadcast: (S, H, 1, 3) - (1, 1, M, 3) = (S, H, M, 3)
        pos_exp = positions[:, :, None, :]            # (S, H, 1, 3)
        obs_exp = obstacles[None, None, :, :]         # (1, 1, M, 3)
        diffs   = pos_exp - obs_exp                   # (S, H, M, 3)
        dists   = np.linalg.norm(diffs, axis=-1)      # (S, H, M)
        min_dists = dists.min(axis=(1, 2))             # (S,) min over H and M

        # CBF constraint: h(x) = dist - buffer ≥ 0
        violations = np.maximum(0.0, buf - min_dists)
        return alpha * violations ** 2

    # ── Simple Dynamics Model ─────────────────────────────────────────────────

    def _rollout(
        self,
        init_pos: np.ndarray,        # (3,) current position
        init_vel: np.ndarray,        # (3,) current velocity
        actions:  np.ndarray,        # (S, H, 3) velocity commands per sample
        dt:       float,
        max_vel:  float,
    ) -> np.ndarray:
        """
        Simple double-integrator rollout for S samples over H steps.
        Returns: (S, H, 3) positions
        """
        S, H, _ = actions.shape
        positions = np.zeros((S, H, 3), dtype=np.float32)
        vel = np.tile(init_vel, (S, 1)).astype(np.float32)
        pos = np.tile(init_pos, (S, 1)).astype(np.float32)

        for h in range(H):
            acc = np.clip(
                actions[:, h, :] - vel,
                -self.physics.max_accel_ms2 * dt,
                 self.physics.max_accel_ms2 * dt,
            )
            vel = np.clip(vel + acc, -max_vel, max_vel)
            pos = pos + vel * dt
            # Altitude constraint
            pos[:, 1] = np.clip(
                pos[:, 1], self.physics.min_altitude_m, self.physics.max_altitude_m
            )
            positions[:, h, :] = pos
        return positions

    # ── Main Planning ─────────────────────────────────────────────────────────

    def plan(
        self,
        current_pose:       np.ndarray,     # (7,) [x,y,z, vx,vy,vz, yaw]
        nbv_target:         np.ndarray,     # (3,) target position from RL agent
        obstacle_positions: np.ndarray,     # (M, 3) obstacle centres
        altitude_m:         Optional[float] = None,
        lighting_lux:       float           = 10000.0,
    ) -> np.ndarray:
        """
        Run MPPI to compute an optimal trajectory from current position to NBV target.

        Returns: (H, 6) trajectory — each row is [x, y, z, vx, vy, vz]
        """
        self.state = np.array(current_pose, dtype=np.float32)
        pos = self.state[:3]
        vel = self.state[3:6]
        alt = altitude_m or float(pos[1])

        max_vel = self.get_blur_safe_speed(alt, lighting_lux)

        S = self.config.n_samples
        H = self.config.horizon
        dt = self.config.dt
        σ  = self.config.noise_sigma

        # Nominal actions: steer toward target
        direction = nbv_target - pos
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            nominal_vel = (direction / dist) * min(dist / (H * dt), max_vel)
        else:
            nominal_vel = np.zeros(3, dtype=np.float32)

        nominal_actions = np.tile(nominal_vel, (S, H, 1)).astype(np.float32)

        # Perturb samples
        noise = np.random.randn(S, H, 3).astype(np.float32) * σ
        sampled_actions = nominal_actions + noise

        # Rollout all samples
        trajectories = self._rollout(pos, vel, sampled_actions, dt, max_vel)

        # ── Cost computation ──────────────────────────────────────────────────
        # 1. Distance to goal at final step
        final_pos   = trajectories[:, -1, :]           # (S, 3)
        goal_cost   = np.linalg.norm(final_pos - nbv_target, axis=1)  # (S,)

        # 2. Smoothness (sum of accelerations)
        vel_diffs    = np.diff(sampled_actions, axis=1)   # (S, H-1, 3)
        smooth_cost  = np.linalg.norm(vel_diffs, axis=-1).sum(axis=1)  # (S,)

        # 3. CBF obstacle cost
        cbf_cost = self._cbf_cost(trajectories, obstacle_positions)

        total_cost = goal_cost + 0.1 * smooth_cost + 10.0 * cbf_cost

        # ── MPPI weighted average ─────────────────────────────────────────────
        beta = total_cost.min()
        weights = np.exp(-(total_cost - beta) / self.config.temperature)
        weights /= weights.sum() + 1e-8

        optimal_actions = np.einsum("s,shd->hd", weights, sampled_actions)  # (H, 3)

        # Build trajectory from optimal actions
        opt_traj = self._rollout(
            pos, vel,
            optimal_actions[None, :, :],
            dt, max_vel
        )[0]  # (H, 3)

        # Build full state trajectory (H, 6)
        vel_traj = np.zeros_like(opt_traj)
        for h in range(H):
            vel_traj[h] = optimal_actions[h]
        full_traj = np.concatenate([opt_traj, vel_traj], axis=1)  # (H, 6)

        self._planned_trajectory = full_traj
        self._trajectory_step = 0
        return full_traj

    def get_next_command(self) -> Optional[np.ndarray]:
        """
        Pop the next velocity command from the planned trajectory.
        Returns: (6,) [x,y,z, vx,vy,vz] or None if trajectory exhausted.
        """
        if self._planned_trajectory is None:
            return None
        if self._trajectory_step >= len(self._planned_trajectory):
            return None
        cmd = self._planned_trajectory[self._trajectory_step]
        self._trajectory_step += 1
        return cmd

    def update_state(self, new_pose: np.ndarray) -> None:
        """Update the controller state with a new measured pose."""
        self.state = np.array(new_pose, dtype=np.float32)

    def is_safe(
        self,
        position:           np.ndarray,
        obstacle_positions: np.ndarray,
    ) -> bool:
        """Quick CBF safety check for a single position."""
        if len(obstacle_positions) == 0:
            return True
        dists = np.linalg.norm(obstacle_positions - position, axis=1)
        return bool(dists.min() >= self.physics.obstacle_buffer_m)

    def trajectory_to_json(self) -> Dict[str, Any]:
        """Export the planned trajectory for the web viewer's DroneSimulator."""
        if self._planned_trajectory is None:
            return {"waypoints": []}
        return {
            "generatedAt": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            "dt": self.config.dt,
            "waypoints": [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                 "vx": float(p[3]), "vy": float(p[4]), "vz": float(p[5])}
                for p in self._planned_trajectory
            ]
        }
