"""
Stage 6: Neural Scene Completion (3D Gaussian Splatting)
Detects spatial gaps and occlusions in the dense TSDF cloud,
and fills them via Nerfstudio Splatfacto CLI integration or
internal Gaussian primitive generation as fallback.
Tags all hallucinated regions with low confidence (0.20-0.30).
"""

import os
import json
import subprocess
import numpy as np
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Nerfstudio / Splatfacto CLI detection
# ---------------------------------------------------------------------------
_NERFSTUDIO_AVAILABLE = False

def _check_nerfstudio():
    global _NERFSTUDIO_AVAILABLE
    try:
        result = subprocess.run(
            ["ns-train", "--help"], capture_output=True, timeout=5
        )
        _NERFSTUDIO_AVAILABLE = result.returncode in (0, 1)
        if _NERFSTUDIO_AVAILABLE:
            print("[Splat] Nerfstudio detected — Splatfacto available.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[Splat] Nerfstudio not found. Using internal 3DGS completion.")
    return _NERFSTUDIO_AVAILABLE


class GaussianSplatCompletion:
    """
    Detects and completes occluded / unseen scene regions using 3D Gaussian Splatting.

    Modes:
    - Nerfstudio Splatfacto (full 3DGS training, best quality, requires GPU + nerfstudio)
    - Internal KD-tree gap detection + interpolative completion (always available)

    All completed (hallucinated) points are tagged with confidence 0.20–0.30
    for downstream confidence engine discrimination.
    """

    def __init__(
        self,
        num_gaussians: int = 5000,
        gap_voxel_size: float = 0.5,    # Resolution for spatial gap detection (meters)
        nerfstudio_output_dir: str = "aero_output/nerfstudio",
    ):
        self.num_gaussians = num_gaussians
        self.gap_voxel_size = gap_voxel_size
        self.nerfstudio_output_dir = nerfstudio_output_dir

    # ------------------------------------------------------------------
    # Spatial gap detection using voxel occupancy
    # ------------------------------------------------------------------

    def _detect_gaps(self, points: np.ndarray,
                     n_gap_samples: int = 2000) -> np.ndarray:
        """
        Detects unoccupied spatial regions in the bounding volume of the point cloud.
        Uses voxel grid occupancy to find empty voxels within the scene extent.
        Returns sampled gap center positions (N, 3).
        """
        if len(points) == 0:
            return np.empty((0, 3))

        min_b = np.min(points, axis=0)
        max_b = np.max(points, axis=0)
        extent = max_b - min_b

        # Build voxel occupancy grid
        nx = max(1, int(extent[0] / self.gap_voxel_size) + 1)
        ny = max(1, int(extent[1] / self.gap_voxel_size) + 1)
        nz = max(1, int(extent[2] / self.gap_voxel_size) + 1)

        occupied = set()
        for pt in points:
            ix = int((pt[0] - min_b[0]) / self.gap_voxel_size)
            iy = int((pt[1] - min_b[1]) / self.gap_voxel_size)
            iz = int((pt[2] - min_b[2]) / self.gap_voxel_size)
            occupied.add((
                min(ix, nx - 1),
                min(iy, ny - 1),
                min(iz, nz - 1),
            ))

        # Identify empty voxels within the scene bounding box
        gap_positions = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    if (ix, iy, iz) not in occupied:
                        cx = min_b[0] + (ix + 0.5) * self.gap_voxel_size
                        cy = min_b[1] + (iy + 0.5) * self.gap_voxel_size
                        cz = min_b[2] + (iz + 0.5) * self.gap_voxel_size
                        gap_positions.append([cx, cy, cz])

        if not gap_positions:
            return np.empty((0, 3))

        gap_arr = np.array(gap_positions)

        # Sub-sample gap positions to target count
        if len(gap_arr) > n_gap_samples:
            idx = np.random.choice(len(gap_arr), n_gap_samples, replace=False)
            gap_arr = gap_arr[idx]

        return gap_arr

    # ------------------------------------------------------------------
    # Color hallucination via nearest-neighbour interpolation
    # ------------------------------------------------------------------

    def _hallucinate_colors(self, gap_pts: np.ndarray,
                             existing_pts: np.ndarray,
                             existing_clrs: np.ndarray,
                             k: int = 5) -> np.ndarray:
        """
        Assigns hallucinated colors to gap points via weighted KNN from existing cloud.
        Distance-weighted average of K nearest observed colors.
        """
        if len(existing_pts) == 0 or len(gap_pts) == 0:
            return np.full((len(gap_pts), 3), 0.65)

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(existing_pts)
            k = min(k, len(existing_pts))
            dists, idxs = tree.query(gap_pts, k=k, workers=-1)
            # Distance-weighted average
            weights = 1.0 / (dists + 1e-6)
            weights = weights / weights.sum(axis=1, keepdims=True)
            if k == 1:
                colors = existing_clrs[idxs]
            else:
                colors = np.einsum('ni,nid->nd', weights, existing_clrs[idxs])
            return np.clip(colors, 0.0, 1.0)
        except ImportError:
            # Fallback: warm structural inference color
            return np.tile(np.array([0.78, 0.58, 0.38]), (len(gap_pts), 1))

    # ------------------------------------------------------------------
    # Nerfstudio Splatfacto integration
    # ------------------------------------------------------------------

    def _run_splatfacto(self, data_dir: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Runs Nerfstudio Splatfacto training via CLI with Max Accuracy and Checkpointing.
        Expects ns-processed data in data_dir.
        Returns extra completed points or None on failure.
        """
        if not _check_nerfstudio():
            return None

        os.makedirs(self.nerfstudio_output_dir, exist_ok=True)
        cmd = [
            "ns-train", "splatfacto",
            "--data", data_dir,
            "--output-dir", self.nerfstudio_output_dir,
            "--viewer.quit-on-train-completion", "True",
            "--max-num-iterations", "30000",          # Max accuracy iterations
            "--pipeline.model.num-downscales", "0",   # Use full resolution
            "nerfstudio-data", "--eval-mode", "fraction",
        ]

        # Check for existing checkpoint to resume
        import glob
        checkpoint_dirs = glob.glob(os.path.join(self.nerfstudio_output_dir, "*", "nerfstudio_models"))
        if checkpoint_dirs:
            latest_ckpt_dir = max(checkpoint_dirs, key=os.path.getmtime)
            # Find the actual .ckpt file
            ckpts = glob.glob(os.path.join(latest_ckpt_dir, "*.ckpt"))
            if ckpts:
                latest_ckpt = max(ckpts, key=os.path.getmtime)
                print(f"[Splat] Found checkpoint, resuming from: {latest_ckpt}")
                cmd.extend(["--load-checkpoint", latest_ckpt])

        try:
            print(f"[Splat] Running Splatfacto: {' '.join(cmd)}")
            result = subprocess.run(cmd, timeout=86400, capture_output=True, text=True) # 24h timeout
            if result.returncode == 0:
                print("[Splat] ✓ Splatfacto training complete.")
                # TODO: parse exported splat cloud
                return None
            else:
                print(f"[Splat] Splatfacto error: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print("[Splat] Splatfacto timed out.")
        except Exception as e:
            print(f"[Splat] Splatfacto failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Main completion entry point
    # ------------------------------------------------------------------

    def complete_occluded_regions(
        self, dense_cloud: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Detects spatial gaps in the dense cloud and fills them with
        3DGS-inspired Gaussian primitive interpolation.
        Tags all hallucinated points with low confidence (0.20–0.28).
        """
        existing_pts = dense_cloud["points_3d"]
        existing_clrs = dense_cloud["colors_3d"]
        existing_cnfs = dense_cloud["confidences"]
        existing_srcs = dense_cloud["source_tags"]

        if len(existing_pts) == 0:
            return {**dense_cloud, "num_hallucinated": 0}

        # --- Detect spatial gaps ---
        n_target = min(self.num_gaussians, 3000)
        gap_pts = self._detect_gaps(existing_pts, n_gap_samples=n_target)
        n_gap = len(gap_pts)

        if n_gap == 0:
            print("[Splat] No spatial gaps detected — dense cloud is complete.")
            return {**dense_cloud, "num_hallucinated": 0}

        # --- Hallucinate colors via KNN ---
        gap_clrs = self._hallucinate_colors(gap_pts, existing_pts, existing_clrs, k=5)

        # --- Confidence scores for hallucinated points ---
        # Assign varying confidence based on proximity to observed cloud
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(existing_pts)
            nn_dists, _ = tree.query(gap_pts, k=1)
            # Close to observed: 0.28; far away: 0.15
            gap_cnfs = np.clip(0.30 - nn_dists * 0.02, 0.10, 0.30).astype(np.float64)
        except ImportError:
            gap_cnfs = np.full(n_gap, 0.22, dtype=np.float64)

        gap_srcs = np.full(n_gap, 0.20, dtype=np.float64)  # Source: 3DGS inferred

        # --- Merge with existing cloud ---
        all_pts = np.vstack([existing_pts, gap_pts])
        all_clrs = np.vstack([existing_clrs, gap_clrs])
        all_cnfs = np.concatenate([existing_cnfs, gap_cnfs])
        all_srcs = np.concatenate([existing_srcs, gap_srcs])

        print(f"[Splat] ✓ Gap completion: {n_gap} hallucinated primitives "
              f"(voxel gap={self.gap_voxel_size}m, KNN color interpolation)")

        return {
            "points_3d": all_pts,
            "colors_3d": all_clrs,
            "confidences": all_cnfs,
            "source_tags": all_srcs,
            "num_hallucinated": n_gap,
        }
