"""
AERO MESH — NBV RL Agent (PPO)
==============================
Deep Reinforcement Learning Next-Best-View agent using Proximal Policy
Optimisation (PPO). Learns micro-inspection trajectories that maximise
reconstruction quality by resolving per-Gaussian and per-entity uncertainty.

The agent DOES NOT directly control the drone. It outputs target NBV
waypoints, which are passed to the MPPI controller for smooth, safe execution.

Observation vector (88-dimensional):
  [drone_pose(6)] + [gauss_mi_top10(30)] + [ougs_entity_scores(20)] +
  [failure_heat(16)] + [battery_pct(1)] + [sector_completion_8(8)] +
  [global_uncertainty(1)] + [step_count_norm(1)] + [action_history(5)]

Action space (continuous, 5-dimensional):
  [Δx, Δy, Δz, Δyaw, Δgimbal_pitch]
  All in range [-1, +1], scaled to physical limits by MPPI.

Checkpoint format:
  checkpoints/rl_agent/ckpt_ep{N}.pt → {
    episode, policy_state, value_state, optimizer_state,
    scheduler_state, best_reward, running_mean_std, config
  }

See: train_rl_agent.py for full training loop with --resume support.
"""

import os
import json
import time
import signal
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    print("[RLAgent] PyTorch not available. Install: pip install torch")


# ── Constants ─────────────────────────────────────────────────────────────────
OBS_DIM    = 88
ACTION_DIM = 5
ACTION_SCALE = np.array([5.0, 3.0, 5.0, np.pi, np.radians(40)], dtype=np.float32)
# Physical limits: ±5m horizontal, ±3m vertical, ±π yaw, ±40° gimbal


# ── Running Normaliser ────────────────────────────────────────────────────────

class RunningMeanStd:
    """Welford online mean/variance for observation normalisation."""
    def __init__(self, shape: Tuple):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch = x.reshape(-1, *self.mean.shape)
        for xi in batch:
            self.count += 1
            delta = xi - self.mean
            self.mean += delta / self.count
            self.var  += delta * (xi - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var / self.count) + 1e-8)

    def state_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: Dict):
        self.mean  = np.array(d["mean"],  dtype=np.float64)
        self.var   = np.array(d["var"],   dtype=np.float64)
        self.count = float(d["count"])


# ── Policy & Value Networks ───────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Actor network: obs → (mean_action, log_std).
    Uses separate heads for mean and log_std with tanh output squashing.
    """
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256),     nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 128),     nn.Tanh(),
        )
        self.mean_head    = nn.Linear(128, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: "torch.Tensor"):
        x    = self.shared(obs)
        mean = torch.tanh(self.mean_head(x))
        log_std = self.log_std_head.expand_as(mean).clamp(-4, 0)
        return mean, log_std

    def get_action(
        self, obs: "torch.Tensor", deterministic: bool = False
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        mean, log_std = self.forward(obs)
        if deterministic:
            return mean, torch.zeros(1)
        std  = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1, 1), log_prob


class ValueNetwork(nn.Module):
    """Critic network: obs → scalar value estimate."""
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256),     nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 128),     nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
        return self.net(obs)


# ── NBV RL Agent ──────────────────────────────────────────────────────────────

class NBVRLAgent:
    """
    PPO-based Next-Best-View agent.

    Runtime usage (inference only — no training):
        agent = NBVRLAgent(checkpoint_path="checkpoints/rl_agent/best.pt")
        obs   = build_observation(drone_pose, gauss_mi, ougs_scores, ...)
        delta = agent.act(obs)   # (5,) action in physical units

    Training usage: see train_rl_agent.py
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device:          str = "auto",
    ):
        if device == "auto":
            self.device = "cuda" if (_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.policy: Optional[PolicyNetwork] = None
        self.value:  Optional[ValueNetwork]  = None
        self.obs_rms = RunningMeanStd((OBS_DIM,))
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)

        if _TORCH:
            self.policy = PolicyNetwork().to(self.device)
            self.value  = ValueNetwork().to(self.device)
            if checkpoint_path:
                self._load_checkpoint(checkpoint_path)
        else:
            print("[RLAgent] Running without PyTorch — random policy fallback.")

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _load_checkpoint(self, path: str) -> int:
        """Load checkpoint. Returns episode number."""
        if not Path(path).exists():
            print(f"[RLAgent] Checkpoint not found: {path}")
            return 0
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.value.load_state_dict(ckpt["value_state"])
        if "running_mean_std" in ckpt:
            self.obs_rms.load_state_dict(ckpt["running_mean_std"])
        episode = ckpt.get("episode", 0)
        print(f"[RLAgent] Loaded checkpoint — episode {episode}, "
              f"best_reward={ckpt.get('best_reward', 'N/A')}")
        return episode

    def save_checkpoint(
        self,
        save_dir:       str,
        episode:        int,
        optimizer_state: Any,
        scheduler_state: Any,
        best_reward:    float,
        config:         Dict,
    ) -> str:
        """Save a full resumable checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"ckpt_ep{episode:05d}.pt"
        torch.save({
            "episode":          episode,
            "policy_state":     self.policy.state_dict(),
            "value_state":      self.value.state_dict(),
            "optimizer_state":  optimizer_state,
            "scheduler_state":  scheduler_state,
            "best_reward":      best_reward,
            "running_mean_std": self.obs_rms.state_dict(),
            "config":           config,
        }, path)
        # Also keep a "latest.pt" symlink/copy for easy --resume
        latest = save_dir / "latest.pt"
        torch.save(torch.load(path, weights_only=False), latest)
        return str(path)

    # ── Observation Builder ────────────────────────────────────────────────────

    @staticmethod
    def build_observation(
        drone_pose:        np.ndarray,      # (6,) [x,y,z,roll,pitch,yaw]
        gauss_mi_top10:    np.ndarray,      # (10,3) top-10 uncertain positions
        ougs_scores:       Dict[str, float],# entity_id → score
        failure_heat:      np.ndarray,      # (16,) flat heat grid (4×4 cells)
        battery_pct:       float,           # 0–1
        sector_completions: np.ndarray,     # (8,) per-sector completion
        global_uncertainty: float,
        step_norm:         float,           # current_step / max_steps
        last_action:       np.ndarray,      # (5,) previous action
    ) -> np.ndarray:
        """
        Build the 88-dimensional observation vector for the RL policy.
        All values normalised to approximately [-1, 1] or [0, 1].
        """
        drone_norm   = np.array(drone_pose, dtype=np.float32)
        drone_norm[:3] /= 200.0      # Normalise position by survey radius
        drone_norm[3:] /= np.pi      # Normalise angles

        gauss_flat   = gauss_mi_top10.flatten().astype(np.float32)[:30]
        if len(gauss_flat) < 30:
            gauss_flat = np.pad(gauss_flat, (0, 30 - len(gauss_flat)))
        gauss_flat  /= 100.0         # Rough position scale

        ougs_flat    = np.array(list(ougs_scores.values())[:20], dtype=np.float32)
        if len(ougs_flat) < 20:
            ougs_flat = np.pad(ougs_flat, (0, 20 - len(ougs_flat)), constant_values=0.5)

        heat_flat    = failure_heat.flatten().astype(np.float32)[:16]
        if len(heat_flat) < 16:
            heat_flat = np.pad(heat_flat, (0, 16 - len(heat_flat)))

        sector_flat  = np.array(sector_completions, dtype=np.float32)[:8]
        if len(sector_flat) < 8:
            sector_flat = np.pad(sector_flat, (0, 8 - len(sector_flat)), constant_values=0.0)

        obs = np.concatenate([
            drone_norm,                                              # 6
            gauss_flat,                                              # 30
            ougs_flat,                                               # 20
            heat_flat,                                               # 16
            np.array([battery_pct], dtype=np.float32),               # 1
            sector_flat,                                             # 8
            np.array([global_uncertainty, step_norm], dtype=np.float32),  # 2
            last_action.astype(np.float32),                          # 5
        ]).astype(np.float32)                                         # total 88

        assert len(obs) == OBS_DIM, f"Obs dim mismatch: {len(obs)} != {OBS_DIM}"
        return obs

    # ── Inference ─────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Forward pass: obs → physical action delta.
        Returns: (5,) array [Δx_m, Δy_m, Δz_m, Δyaw_rad, Δgimbal_rad]
        """
        if not _TORCH or self.policy is None:
            # Random fallback
            return (np.random.uniform(-1, 1, ACTION_DIM) * ACTION_SCALE * 0.2)

        norm_obs = self.obs_rms.normalize(obs)
        with torch.no_grad():
            t = torch.from_numpy(norm_obs).float().unsqueeze(0).to(self.device)
            action, _ = self.policy.get_action(t, deterministic=deterministic)
            raw = action.squeeze(0).cpu().numpy()
        physical = raw * ACTION_SCALE
        self._last_action = raw
        return physical

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()
