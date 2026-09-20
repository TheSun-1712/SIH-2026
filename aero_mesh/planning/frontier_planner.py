"""
AERO MESH — Global Frontier Planner (Shannon MI + HGS Bidirectional)
======================================================================
Global macro-coverage planner that:
  1. Divides the survey area into sectors
  2. Scores sectors by Shannon Mutual Information (unexplored voxels)
  3. Prioritises sectors by GIS building density (from GISPrior)
  4. Receives quality feedback from the RL agent (HGS bidirectional link)
  5. Re-queues underperforming sectors automatically

This runs BEFORE the RL agent — it produces the global macro waypoints
that the RL agent then refines into precise micro-inspection orbits.
"""

import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum


class SectorStatus(str, Enum):
    PENDING    = "PENDING"
    IN_FLIGHT  = "IN_FLIGHT"
    COMPLETE   = "COMPLETE"
    RESCHEDULED = "RESCHEDULED"    # RL agent flagged as incomplete


@dataclass
class Sector:
    """A rectangular survey sector."""
    id:          str
    center_x:    float
    center_z:    float
    width:       float
    depth:       float
    priority:    float              # 0–1, higher = fly first
    status:      SectorStatus = SectorStatus.PENDING
    completion:  float = 0.0       # 0–1 quality score from RL feedback
    altitude_m:  float = 30.0      # Planned survey altitude for this sector
    visits:      int = 0


@dataclass
class MacroWaypoint:
    """A single global macro waypoint for the drone."""
    x: float
    y: float                        # altitude (up)
    z: float
    sector_id: str
    heading_deg: float = 0.0
    survey_type: str = "nadir"      # nadir | oblique | orbital


class FrontierPlanner:
    """
    Shannon Mutual Information Frontier Planner with HGS-style bidirectional
    quality feedback from the RL agent.

    Usage:
        planner = FrontierPlanner(survey_radius_m=300, grid_cells=8)
        planner.seed_from_gis(manifest)
        waypoints = planner.compute_global_waypoints(altitude_m=30)
        # ... drone flies ...
        planner.receive_quality_feedback("sector_3", completion_score=0.42)
        # sector_3 will be re-queued if completion < threshold
    """

    COMPLETION_THRESHOLD = 0.70     # Sectors below this are re-queued

    def __init__(
        self,
        survey_radius_m: float = 300.0,
        grid_cells:      int   = 8,
        altitude_m:      float = 30.0,
        output_dir:      str   = "aero_mesh_output",
    ):
        self.survey_radius_m = survey_radius_m
        self.grid_cells      = grid_cells
        self.altitude_m      = altitude_m
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sectors:   List[Sector]        = []
        self.waypoints: List[MacroWaypoint] = []
        self._sector_map: Dict[str, Sector] = {}

    # ── GIS seeding ───────────────────────────────────────────────────────────

    def seed_from_gis(self, manifest) -> None:
        """
        Seed sector priorities from a GISManifest (from gis_prior.py).
        Sectors with higher building density get higher priority.
        """
        priority_grid = manifest.get_sector_priorities(
            manifest, self.grid_cells
        ) if hasattr(manifest, "get_sector_priorities") else np.ones(
            self.grid_cells * self.grid_cells
        )
        self._build_sectors(priority_grid)
        print(f"[FrontierPlanner] Seeded {len(self.sectors)} sectors from GIS prior.")

    def _build_sectors(self, priority_flat: Optional[np.ndarray] = None) -> None:
        """Build the sector grid from the survey area."""
        cell_w = (2 * self.survey_radius_m) / self.grid_cells
        cell_d = (2 * self.survey_radius_m) / self.grid_cells

        if priority_flat is None:
            priority_flat = np.ones(self.grid_cells * self.grid_cells)

        self.sectors = []
        for iz in range(self.grid_cells):
            for ix in range(self.grid_cells):
                cx = -self.survey_radius_m + (ix + 0.5) * cell_w
                cz = -self.survey_radius_m + (iz + 0.5) * cell_d
                idx = iz * self.grid_cells + ix
                priority = float(priority_flat[idx])
                sid = f"sector_{ix}_{iz}"
                s = Sector(
                    id=sid, center_x=cx, center_z=cz,
                    width=cell_w, depth=cell_d, priority=priority,
                    altitude_m=self.altitude_m,
                )
                self.sectors.append(s)
                self._sector_map[sid] = s

        # Sort by priority descending
        self.sectors.sort(key=lambda s: s.priority, reverse=True)

    # ── Shannon MI ────────────────────────────────────────────────────────────

    def _compute_sector_mi(
        self,
        sector: Sector,
        voxel_occupancy: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Shannon Mutual Information score for a sector.
        Uses voxel occupancy if available, otherwise falls back to
        1 - completion (uncertainty proxy).

        In a live system, voxel_occupancy would be the TSDF voxel grid
        slice for this sector's bounding box.
        """
        if voxel_occupancy is not None:
            # Shannon entropy of the free/occupied/unknown voxel distribution
            p_occupied = np.mean(voxel_occupancy > 0)
            p_free     = np.mean(voxel_occupancy < 0)
            p_unknown  = 1.0 - p_occupied - p_free
            probs = np.array([p_occupied, p_free, p_unknown])
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            return float(entropy / 2.0)  # Normalise to [0, 1]
        else:
            # Fallback: information gain ∝ unexplored fraction
            return max(0.0, 1.0 - sector.completion)

    # ── Waypoint Generation ───────────────────────────────────────────────────

    def compute_global_waypoints(
        self,
        altitude_m:       float = 30.0,
        voxel_map:        Optional[np.ndarray] = None,
        oblique_altitude: float = 15.0,
    ) -> List[MacroWaypoint]:
        """
        Generate ordered macro waypoints covering all pending/rescheduled sectors.
        Returns list of MacroWaypoint objects sorted by priority.

        The drone follows these as the coarse flight skeleton; the RL agent
        generates fine micro-inspection movements around each waypoint.
        """
        if not self.sectors:
            self._build_sectors()

        self.waypoints = []
        pending = [s for s in self.sectors
                   if s.status in (SectorStatus.PENDING, SectorStatus.RESCHEDULED)]

        # Score sectors by MI
        scored = []
        for s in pending:
            mi_score = self._compute_sector_mi(s, voxel_map)
            total_score = 0.6 * s.priority + 0.4 * mi_score
            scored.append((total_score, s))
        scored.sort(key=lambda x: x[0], reverse=True)

        for _, sector in scored:
            # Primary nadir pass
            self.waypoints.append(MacroWaypoint(
                x=sector.center_x, y=altitude_m, z=sector.center_z,
                sector_id=sector.id, heading_deg=0.0, survey_type="nadir",
            ))
            # If sector has high priority (likely has buildings), add oblique approach
            if sector.priority > 0.5:
                # 4 cardinal oblique approaches around sector
                for angle_deg in [0, 90, 180, 270]:
                    rad = np.radians(angle_deg)
                    ox = sector.center_x + np.cos(rad) * sector.width * 0.7
                    oz = sector.center_z + np.sin(rad) * sector.depth * 0.7
                    self.waypoints.append(MacroWaypoint(
                        x=ox, y=oblique_altitude, z=oz,
                        sector_id=sector.id, heading_deg=angle_deg + 180,
                        survey_type="oblique",
                    ))

        print(f"[FrontierPlanner] Generated {len(self.waypoints)} macro waypoints "
              f"from {len(pending)} pending sectors.")
        return self.waypoints

    # ── Bidirectional Quality Feedback (HGS-Planner) ─────────────────────────

    def receive_quality_feedback(
        self, sector_id: str, completion_score: float
    ) -> bool:
        """
        Called by the RL agent after it has finished inspecting a sector.
        If completion_score < COMPLETION_THRESHOLD, the sector is re-queued.
        Returns True if the sector was re-queued.
        """
        sector = self._sector_map.get(sector_id)
        if sector is None:
            return False

        sector.completion = float(completion_score)
        sector.visits    += 1

        if completion_score < self.COMPLETION_THRESHOLD and sector.visits < 3:
            sector.status = SectorStatus.RESCHEDULED
            sector.priority = min(1.0, sector.priority * 1.5)  # Boost priority
            print(f"[FrontierPlanner] [!] Sector {sector_id} re-queued "
                  f"(completion={completion_score:.2f}, visits={sector.visits})")
            return True
        else:
            sector.status = SectorStatus.COMPLETE
            print(f"[FrontierPlanner] [+] Sector {sector_id} complete "
                  f"(completion={completion_score:.2f})")
            return False

    def mark_sector_in_flight(self, sector_id: str) -> None:
        if sector_id in self._sector_map:
            self._sector_map[sector_id].status = SectorStatus.IN_FLIGHT

    # ── Flight Plan Export ────────────────────────────────────────────────────

    def export_flight_plan(
        self, out_path: str = "web/public/flight_plan_planned.json"
    ) -> str:
        """Export the macro waypoints to a JSON flight plan for the web viewer."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        plan = {
            "generatedAt":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "totalWaypoints": len(self.waypoints),
            "surveyRadius_m": self.survey_radius_m,
            "gridCells":      self.grid_cells,
            "waypoints": [
                {
                    "x":          w.x, "y":     w.y, "z": w.z,
                    "sectorId":   w.sector_id,
                    "headingDeg": w.heading_deg,
                    "surveyType": w.survey_type,
                }
                for w in self.waypoints
            ],
            "sectors": [
                {
                    "id":         s.id,
                    "centerX":    s.center_x,
                    "centerZ":    s.center_z,
                    "priority":   s.priority,
                    "completion": s.completion,
                    "status":     s.status.value,
                    "visits":     s.visits,
                }
                for s in self.sectors
            ],
        }
        with open(out, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"[FrontierPlanner] Flight plan exported → {out}")
        return str(out)

    def coverage_summary(self) -> Dict[str, Any]:
        """Return a summary of coverage progress."""
        total   = len(self.sectors)
        done    = sum(1 for s in self.sectors if s.status == SectorStatus.COMPLETE)
        pending = sum(1 for s in self.sectors if s.status == SectorStatus.PENDING)
        resched = sum(1 for s in self.sectors if s.status == SectorStatus.RESCHEDULED)
        return {
            "total":          total,
            "complete":       done,
            "pending":        pending,
            "rescheduled":    resched,
            "coverage_pct":   round(done / max(total, 1) * 100, 1),
            "mean_completion": round(
                np.mean([s.completion for s in self.sectors]) if self.sectors else 0.0, 3
            ),
        }


if __name__ == "__main__":
    print("[FrontierPlanner] Standalone test: generating synthetic flight plan")
    planner = FrontierPlanner(survey_radius_m=200, grid_cells=4, altitude_m=30)
    planner._build_sectors()
    waypoints = planner.compute_global_waypoints()
    planner.export_flight_plan("aero_mesh_output/test_flight_plan.json")
    print(f"Generated {len(waypoints)} waypoints.")
    print("Coverage summary:", planner.coverage_summary())
