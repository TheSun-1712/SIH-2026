"""
AERO MESH — GauSS-MI: Live Per-Gaussian Uncertainty Oracle
============================================================
Implements the GauSS-MI (Gaussian Splatting Shannon Mutual Information)
framework for real-time, view-driven uncertainty quantification over the
3D Gaussian Splatting scene representation.

In contrast to TSDF-based confidence (which is computed post-flight),
GauSS-MI computes per-Gaussian uncertainty WHILE the drone is flying,
enabling the RL agent to act proactively instead of reactively.

Architecture:
  - Each Gaussian g has an uncertainty score U_g ∈ [0, 1]
  - U_g is high for Gaussians that have been seen from few viewpoints
    or that have high colour variance across views (ambiguous appearance)
  - U_g is updated incrementally as new frames arrive
  - Shannon MI is used to evaluate expected information gain from candidate
    next viewpoints without actually flying to them

Training:
  A lightweight MLP uncertainty head is trained on pairs of
  (Gaussian state, view history) → uncertainty label derived from
  multi-view colour consistency ground truth.

See: train_gauss_mi.py for the training script.
"""

import os
import json
import time
import signal
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("[GauSS-MI] PyTorch not available. Install: pip install torch")


# ── Uncertainty Head Network ──────────────────────────────────────────────────

class GaussianUncertaintyHead(nn.Module):
    """
    Lightweight MLP that predicts per-Gaussian uncertainty from
    the Gaussian's view history embedding.

    Input:  (N, input_dim) — per-Gaussian feature vector:
              [mean_pos(3), scale(3), rotation_q(4),
               view_count, mean_opacity, colour_var(3),
               sh_coefficient_var(3)]   = 20-dim

    Output: (N, 1)  — uncertainty score in [0, 1]
    """

    INPUT_DIM  = 20
    HIDDEN_DIM = 64

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.HIDDEN_DIM),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class GaussMI:
    """
    Live GauSS-MI uncertainty oracle for the NBV RL agent.

    Usage (runtime):
        gmi = GaussMI(checkpoint_path="checkpoints/gauss_mi/best.pt")
        gmi.initialize_from_splat(gaussians_dict)
        gmi.update_from_new_view(image_np, camera_pose)
        scores = gmi.get_uncertainty_map()          # (N_gaussians,) array
        top_k  = gmi.get_top_k_uncertain_positions(k=10)  # (k, 3) xyz

    Gaussian dict format:
        {
          "positions":  (N, 3) float32,
          "scales":     (N, 3) float32,
          "rotations":  (N, 4) float32  (quaternion),
          "opacities":  (N, 1) float32,
          "sh_coeffs":  (N, C) float32  (spherical harmonics),
        }
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = "cuda" if (_TORCH_AVAILABLE and
                                     torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model: Optional[GaussianUncertaintyHead] = None
        self._gaussians: Optional[Dict[str, np.ndarray]] = None
        self._view_counts: Optional[np.ndarray] = None
        self._colour_history: Dict[int, List[np.ndarray]] = {}
        self._uncertainty_cache: Optional[np.ndarray] = None
        self._cache_dirty = True

        if _TORCH_AVAILABLE:
            self.model = GaussianUncertaintyHead().to(self.device)
            if checkpoint_path and Path(checkpoint_path).exists():
                self._load_checkpoint(checkpoint_path)
                print(f"[GauSS-MI] Loaded uncertainty head from {checkpoint_path}")
            else:
                print("[GauSS-MI] No checkpoint found — using un-trained model. "
                      "Run train_gauss_mi.py to train first.")
        else:
            print("[GauSS-MI] PyTorch unavailable — falling back to view-count heuristic.")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
        elif "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize_from_splat(self, gaussians: Dict[str, np.ndarray]) -> None:
        """
        Initialise the uncertainty tracker from a 3DGS Gaussian dict.
        Called once when the 3DGS model is first built or loaded.
        """
        self._gaussians = gaussians
        N = len(gaussians["positions"])
        self._view_counts    = np.zeros(N, dtype=np.float32)
        self._colour_history = {}
        self._cache_dirty    = True
        print(f"[GauSS-MI] Initialised with {N} Gaussians.")

    def update_from_new_view(
        self,
        image:       np.ndarray,    # (H, W, 3) RGB uint8
        camera_pose: np.ndarray,    # (4, 4) camera-to-world matrix
    ) -> None:
        """
        Update uncertainty estimates given a newly captured frame.
        Increments view counts for Gaussians visible in this frustum
        and accumulates colour observations.
        This is a fast incremental update — O(N) Gaussian operations.
        """
        if self._gaussians is None:
            return

        positions = self._gaussians["positions"]  # (N, 3)

        # Simplified visibility: project Gaussians into camera frustum
        visible_mask = self._get_visible_gaussians(positions, camera_pose, image.shape)
        self._view_counts[visible_mask] += 1.0

        # Record colour observations for visible Gaussians (sampled, not all)
        vis_indices = np.where(visible_mask)[0]
        for idx in vis_indices[::max(1, len(vis_indices) // 50)]:  # sample 50 per frame
            # Sample image colour at Gaussian's projected pixel
            col = self._sample_colour_at_gaussian(
                positions[idx], camera_pose, image
            )
            if col is not None:
                if idx not in self._colour_history:
                    self._colour_history[idx] = []
                self._colour_history[idx].append(col)
                # Keep last 10 observations per Gaussian
                if len(self._colour_history[idx]) > 10:
                    self._colour_history[idx].pop(0)

        self._cache_dirty = True

    def _get_visible_gaussians(
        self,
        positions:   np.ndarray,   # (N, 3)
        camera_pose: np.ndarray,   # (4, 4)
        image_shape: Tuple,
    ) -> np.ndarray:
        """
        Fast frustum-based visibility test.
        Returns boolean mask (N,) of which Gaussians are in the camera frustum.
        """
        if positions is None or len(positions) == 0:
            return np.array([], dtype=bool)

        # World → camera transform
        R = camera_pose[:3, :3].T
        t = -R @ camera_pose[:3, 3]
        cam_pos = (R @ positions.T).T + t    # (N, 3) in camera space

        # Frustum: in front of camera (z > 0) and within FOV cone (~60°)
        in_front = cam_pos[:, 2] > 0.5
        # Simple ±45 degree frustum test
        fov_x = np.abs(cam_pos[:, 0]) < cam_pos[:, 2] * 0.95
        fov_y = np.abs(cam_pos[:, 1]) < cam_pos[:, 2] * 0.75
        return in_front & fov_x & fov_y

    def _sample_colour_at_gaussian(
        self,
        position:    np.ndarray,   # (3,)
        camera_pose: np.ndarray,   # (4, 4)
        image:       np.ndarray,   # (H, W, 3)
    ) -> Optional[np.ndarray]:
        """Project a Gaussian onto image and sample its colour (simplified pinhole)."""
        try:
            H, W = image.shape[:2]
            R = camera_pose[:3, :3].T
            t = -R @ camera_pose[:3, 3]
            cam = R @ position + t
            if cam[2] < 0.1:
                return None
            fx = fy = W * 0.8
            u = int(cam[0] / cam[2] * fx + W / 2)
            v = int(cam[1] / cam[2] * fy + H / 2)
            if not (0 <= u < W and 0 <= v < H):
                return None
            return image[v, u].astype(np.float32) / 255.0
        except Exception:
            return None

    # ── Uncertainty Computation ───────────────────────────────────────────────

    def _build_features(self) -> np.ndarray:
        """
        Build the per-Gaussian feature matrix for the uncertainty head.
        Returns: (N, INPUT_DIM) float32
        """
        g = self._gaussians
        N = len(g["positions"])

        view_count_norm = np.clip(self._view_counts / 20.0, 0, 1).reshape(-1, 1)

        # Colour variance per Gaussian
        colour_var = np.zeros((N, 3), dtype=np.float32)
        for idx, hist in self._colour_history.items():
            if len(hist) >= 2:
                colour_var[idx] = np.var(np.stack(hist), axis=0)[:3]

        # SH coefficient variance (use std as proxy if full SH not available)
        sh = g.get("sh_coeffs", np.zeros((N, 3), dtype=np.float32))
        sh_var = np.var(sh, axis=1, keepdims=True).repeat(3, axis=1) \
            if sh.shape[1] > 3 else np.zeros((N, 3), dtype=np.float32)
        sh_var = np.clip(sh_var, 0, 1)

        opacities = g.get("opacities", np.ones((N, 1), dtype=np.float32))
        if opacities.ndim == 1:
            opacities = opacities.reshape(-1, 1)

        features = np.concatenate([
            g["positions"],                             # 3
            g.get("scales", np.ones((N, 3)) * 0.05),   # 3
            g.get("rotations", np.tile([1,0,0,0], (N,1))),  # 4
            view_count_norm,                            # 1
            np.clip(opacities, 0, 1),                  # 1
            colour_var,                                 # 3
            sh_var,                                     # 3 → total 18+2 = 20
            np.clip(colour_var.mean(axis=1, keepdims=True), 0, 1),  # 1  mean colour var
            (self._view_counts < 3).astype(np.float32).reshape(-1, 1),  # 1 barely seen
        ], axis=1).astype(np.float32)

        return features[:, :GaussianUncertaintyHead.INPUT_DIM]

    def get_uncertainty_map(self) -> np.ndarray:
        """
        Compute per-Gaussian uncertainty scores.
        Returns: (N,) float32 array, values in [0, 1].
        Higher = more uncertain = higher priority for RL inspection.
        """
        if self._gaussians is None:
            return np.array([], dtype=np.float32)

        if not self._cache_dirty and self._uncertainty_cache is not None:
            return self._uncertainty_cache

        N = len(self._gaussians["positions"])

        if _TORCH_AVAILABLE and self.model is not None:
            features = self._build_features()   # (N, 20)
            with torch.no_grad():
                x = torch.from_numpy(features).to(self.device)
                scores = self.model(x).squeeze(-1).cpu().numpy()
        else:
            # Fallback heuristic: uncertainty ∝ 1 / (1 + view_count)
            scores = 1.0 / (1.0 + self._view_counts)

        self._uncertainty_cache = scores.astype(np.float32)
        self._cache_dirty = False
        return self._uncertainty_cache

    def get_top_k_uncertain_positions(
        self,
        k: int = 10,
        min_uncertainty: float = 0.3,
    ) -> np.ndarray:
        """
        Return 3D world positions of the k most uncertain Gaussians.
        Returns: (k, 3) float32 array.
        Directly fed into the RL agent's observation vector.
        """
        scores = self.get_uncertainty_map()
        if len(scores) == 0:
            return np.zeros((k, 3), dtype=np.float32)

        mask = scores >= min_uncertainty
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(scores))

        top_k_local = np.argsort(scores[valid_indices])[::-1][:k]
        top_k_global = valid_indices[top_k_local]
        positions = self._gaussians["positions"][top_k_global]

        # Pad with zeros if fewer than k
        if len(positions) < k:
            pad = np.zeros((k - len(positions), 3), dtype=np.float32)
            positions = np.vstack([positions, pad])
        return positions.astype(np.float32)

    def global_scene_uncertainty(self) -> float:
        """Mean uncertainty across all Gaussians — used as flight termination check."""
        scores = self.get_uncertainty_map()
        return float(np.mean(scores)) if len(scores) > 0 else 1.0

    def export_uncertainty_json(
        self, out_path: str = "web/public/uncertainty_map.json"
    ) -> str:
        """Export top-100 uncertain positions for web viewer heatmap overlay."""
        scores = self.get_uncertainty_map()
        top_100 = np.argsort(scores)[::-1][:100]
        positions = self._gaussians["positions"][top_100]
        out = {
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "globalUncertainty": float(np.mean(scores)),
            "hotspots": [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                 "uncertainty": float(scores[i])}
                for p, i in zip(positions, top_100)
            ]
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        return out_path
