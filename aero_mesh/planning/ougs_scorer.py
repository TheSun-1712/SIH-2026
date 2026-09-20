"""
AERO MESH — OUGS: Object-Aware Uncertainty Gaussian Splatting Scorer
======================================================================
Propagates Gaussian covariance through the rendering Jacobian to produce
per-entity uncertainty scores (not just voxel-level).

Each building, tree, and vehicle entity detected by YOLO-seg gets its own
uncertainty score U_k ∈ [0, 1]. The RL agent uses these scores to:
  - Prioritise inspecting buildings with high uncertainty
  - Trigger orbital/helical inspection patterns for U_k > threshold
  - Report per-entity completion back to the Frontier Planner

References:
  OUGS (Object-aware Uncertainty Gaussian Splatting) — CVPR 2025
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class OUGSScorer:
    """
    Per-entity uncertainty scorer using OUGS methodology.

    Usage:
        scorer = OUGSScorer()
        entity_scores = scorer.score_entities(entity_manifest, gaussians, camera_poses)
        priority_targets = scorer.get_priority_inspection_targets(entity_scores)
    """

    def __init__(self, uncertainty_threshold: float = 0.55):
        self.uncertainty_threshold = uncertainty_threshold

    # ── Gaussian → Entity Association ────────────────────────────────────────

    def _associate_gaussians_to_entities(
        self,
        gaussians:       Dict[str, np.ndarray],
        entity_manifest: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Associate each Gaussian to the closest entity using spatial bounding box tests.
        Returns: { entity_id → boolean mask (N,) of member Gaussians }
        """
        positions = gaussians["positions"]    # (N, 3)
        N = len(positions)
        associations: Dict[str, np.ndarray] = {}

        for bldg in entity_manifest.get("buildings", []):
            fp = bldg.get("footprint", {})
            cx   = float(fp.get("cx", 0.0))
            cz   = float(fp.get("cz", 0.0))
            hw   = float(fp.get("w", 5.0)) / 2.0 + 2.0  # + 2m tolerance
            hd   = float(fp.get("d", 5.0)) / 2.0 + 2.0
            hh   = float(bldg.get("measuredHeight", 10.0)) + 3.0
            mask = (
                (np.abs(positions[:, 0] - cx) < hw) &
                (np.abs(positions[:, 2] - cz) < hd) &
                (positions[:, 1] < hh) &
                (positions[:, 1] > -2.0)
            )
            associations[bldg["id"]] = mask

        for tree in entity_manifest.get("vegetation", []):
            pos  = tree.get("position", [0, 0, 0])
            r    = float(tree.get("radius", 3.0)) + 1.0
            h    = float(tree.get("height", 5.0)) + 2.0
            mask = (
                (np.abs(positions[:, 0] - pos[0]) < r) &
                (np.abs(positions[:, 2] - pos[2]) < r) &
                (positions[:, 1] < h)
            )
            associations[tree["id"]] = mask

        for veh in entity_manifest.get("vehicles", []):
            pos  = veh.get("position", [0, 0, 0])
            mask = (
                (np.abs(positions[:, 0] - pos[0]) < 4.0) &
                (np.abs(positions[:, 2] - pos[2]) < 4.0) &
                (positions[:, 1] < 4.0)
            )
            associations[veh["id"]] = mask

        return associations

    # ── Rendering Jacobian Covariance Propagation ─────────────────────────────

    def _propagate_covariance(
        self,
        positions:    np.ndarray,   # (M, 3) Gaussians belonging to entity
        scales:       np.ndarray,   # (M, 3)
        camera_poses: List[np.ndarray],  # list of (4,4) cam-to-world matrices
    ) -> float:
        """
        Propagate 3D Gaussian covariance through the rendering Jacobian
        to estimate per-entity 2D rendering uncertainty.

        Simplified OUGS formula:
          U_entity = mean( var( J @ Σ_3D @ J.T ) over all views )

        where J is the affine Jacobian of the perspective projection.
        Returns scalar uncertainty in [0, 1].
        """
        if len(positions) == 0 or len(camera_poses) == 0:
            return 1.0

        view_uncertainties = []
        for pose in camera_poses:
            R = pose[:3, :3]
            t = pose[:3, 3]
            cam_positions = (R @ positions.T).T + t  # (M, 3)

            # Only consider Gaussians in front of camera
            valid = cam_positions[:, 2] > 0.1
            if not np.any(valid):
                continue
            cam_valid = cam_positions[valid]
            scales_valid = scales[valid]

            # Perspective Jacobian (simplified — focal length normalised)
            z = cam_valid[:, 2].clip(0.1)
            # J = [[1/z, 0, -x/z²], [0, 1/z, -y/z²]]
            # Approximate: uncertainty ∝ scale_projected / z²
            projected_scale = (scales_valid[:, :2] / z[:, None]).clip(0, 10)
            view_unc = projected_scale.mean()
            view_uncertainties.append(float(view_unc))

        if not view_uncertainties:
            return 1.0

        # More uncertainty when the entity is seen from few views
        n_views = len(view_uncertainties)
        coverage_factor = 1.0 / (1.0 + 0.1 * n_views)
        raw_unc = np.mean(view_uncertainties) * coverage_factor
        return float(np.clip(raw_unc, 0.0, 1.0))

    # ── Main Scoring API ──────────────────────────────────────────────────────

    def score_entities(
        self,
        entity_manifest: Dict[str, Any],
        gaussians:       Dict[str, np.ndarray],
        camera_poses:    List[np.ndarray],
        gauss_mi_scores: Optional[np.ndarray] = None,  # from GauSS-MI
    ) -> Dict[str, float]:
        """
        Compute per-entity uncertainty score U_k for every entity in the manifest.

        Args:
            entity_manifest: dict with 'buildings', 'vegetation', 'vehicles' lists
            gaussians: dict with 'positions', 'scales', 'rotations', etc.
            camera_poses: list of (4,4) camera-to-world matrices from prior views
            gauss_mi_scores: optional (N,) per-Gaussian uncertainty from GauSS-MI
                             If provided, these are fused with the OUGS covariance score.

        Returns:
            { entity_id → uncertainty_score (float in [0, 1]) }
        """
        positions = gaussians.get("positions", np.zeros((0, 3), dtype=np.float32))
        scales    = gaussians.get("scales",    np.ones((0, 3), dtype=np.float32) * 0.05)

        if len(positions) == 0:
            return {}

        entity_scores: Dict[str, float] = {}
        associations = self._associate_gaussians_to_entities(
            gaussians, entity_manifest
        )

        for entity_id, mask in associations.items():
            member_positions = positions[mask]
            member_scales    = scales[mask]
            n_members = int(mask.sum())

            if n_members == 0:
                # Entity has no associated Gaussians — fully uncertain
                entity_scores[entity_id] = 1.0
                continue

            # OUGS covariance propagation score
            cov_score = self._propagate_covariance(
                member_positions, member_scales, camera_poses
            )

            # GauSS-MI integration: if available, average in per-Gaussian MI scores
            if gauss_mi_scores is not None and len(gauss_mi_scores) == len(positions):
                mi_scores_entity = gauss_mi_scores[mask]
                mi_mean = float(np.mean(mi_scores_entity))
                # Weighted fusion: 60% covariance propagation, 40% MI uncertainty
                final_score = 0.6 * cov_score + 0.4 * mi_mean
            else:
                final_score = cov_score

            # Extra penalty for entities seen from very few views
            view_coverage = min(len(camera_poses), 10) / 10.0
            final_score = min(1.0, final_score * (1.0 + 0.3 * (1.0 - view_coverage)))

            entity_scores[entity_id] = float(np.clip(final_score, 0.0, 1.0))

        return entity_scores

    def get_priority_inspection_targets(
        self,
        entity_scores:     Dict[str, float],
        entity_manifest:   Dict[str, Any],
        threshold:         Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a priority-sorted list of entities that need detailed inspection.
        Each entry contains entity_id, uncertainty, position, and inspection type.

        inspection_type:
          - 'orbital'  : fly 360-degree orbit (for buildings with U > 0.7)
          - 'oblique'  : 4 cardinal oblique passes (for buildings with U > 0.55)
          - 'helical'  : ascending helix (for tall buildings, U > 0.7)
        """
        threshold = threshold or self.uncertainty_threshold
        targets = []

        # Build position lookup
        pos_lookup: Dict[str, Any] = {}
        for b in entity_manifest.get("buildings", []):
            fp = b.get("footprint", {})
            pos_lookup[b["id"]] = {
                "type":     "building",
                "position": [fp.get("cx", 0), 0, fp.get("cz", 0)],
                "height":   b.get("measuredHeight", 10.0),
                "arch_type": b.get("type", "apartment_block"),
            }
        for t in entity_manifest.get("vegetation", []):
            pos_lookup[t["id"]] = {
                "type": "vegetation", "position": t.get("position", [0,0,0]),
                "height": t.get("height", 5.0), "arch_type": "tree",
            }
        for v in entity_manifest.get("vehicles", []):
            pos_lookup[v["id"]] = {
                "type": "vehicle", "position": v.get("position", [0,0,0]),
                "height": 2.0, "arch_type": v.get("type", "Car"),
            }

        for entity_id, score in entity_scores.items():
            if score < threshold:
                continue
            info = pos_lookup.get(entity_id, {})
            height = info.get("height", 5.0)

            if info.get("type") == "building":
                if score > 0.7 and height > 15.0:
                    insp_type = "helical"
                elif score > 0.7:
                    insp_type = "orbital"
                else:
                    insp_type = "oblique"
            else:
                insp_type = "oblique"

            targets.append({
                "entity_id":       entity_id,
                "uncertainty":     score,
                "position":        info.get("position", [0, 0, 0]),
                "height":          height,
                "arch_type":       info.get("arch_type", ""),
                "inspection_type": insp_type,
            })

        targets.sort(key=lambda x: x["uncertainty"], reverse=True)
        return targets

    def export_scores(
        self,
        entity_scores:  Dict[str, float],
        out_path: str = "web/public/ougs_scores.json",
    ) -> str:
        """Export per-entity uncertainty scores for the web viewer heat overlay."""
        out = {
            "generatedAt":    __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            "threshold":      self.uncertainty_threshold,
            "entity_scores":  entity_scores,
            "high_priority":  [eid for eid, s in entity_scores.items()
                                if s >= self.uncertainty_threshold],
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        return out_path
