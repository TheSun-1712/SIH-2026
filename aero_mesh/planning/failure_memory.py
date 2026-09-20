"""
AERO MESH — Episodic Failure Memory
=====================================
Persists and retrieves historical reconstruction failure events across
all flight sessions. The RL agent queries this memory to proactively
avoid repeating known mistake patterns on similar geometries.

Failure modes tracked:
  - LOW_CONFIDENCE  : confidence_engine score < 0.35
  - SFM_FAILURE     : SfM triangulation failure (grazing angle / low parallax)
  - OCCLUSION_VOID  : 3DGS gap detection found an unresolved hollow
  - GRAZING_ANGLE   : camera incidence > 75 degrees from surface normal

Persistence: Append-only failures.jsonl — safe to interrupt at any time.
"""

import os
import json
import time
import hashlib
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum


class FailureMode(str, Enum):
    LOW_CONFIDENCE  = "LOW_CONFIDENCE"
    SFM_FAILURE     = "SFM_FAILURE"
    OCCLUSION_VOID  = "OCCLUSION_VOID"
    GRAZING_ANGLE   = "GRAZING_ANGLE"


class FailureEvent:
    """A single recorded reconstruction failure."""

    def __init__(
        self,
        position_xyz:   np.ndarray,       # 3D world position of the failure
        mode:           FailureMode,
        view_direction: np.ndarray,        # Unit vector: camera look-at at time of failure
        entity_id:      Optional[str],     # Building/entity ID if applicable
        confidence:     float = 0.0,       # Confidence score at failure point
        session_id:     str  = "",
    ):
        self.position_xyz   = np.array(position_xyz, dtype=np.float32)
        self.mode           = FailureMode(mode)
        self.view_direction = np.array(view_direction, dtype=np.float32)
        self.entity_id      = entity_id
        self.confidence     = float(confidence)
        self.session_id     = session_id
        self.timestamp      = time.time()
        self.fingerprint    = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """64-char hex fingerprint of position + mode for dedup."""
        raw = f"{self.position_xyz.tolist()}{self.mode.value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint":    self.fingerprint,
            "timestamp":      self.timestamp,
            "session_id":     self.session_id,
            "mode":           self.mode.value,
            "position_xyz":   self.position_xyz.tolist(),
            "view_direction": self.view_direction.tolist(),
            "entity_id":      self.entity_id,
            "confidence":     self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FailureEvent":
        evt = cls(
            position_xyz   = np.array(d["position_xyz"]),
            mode           = FailureMode(d["mode"]),
            view_direction = np.array(d["view_direction"]),
            entity_id      = d.get("entity_id"),
            confidence     = d.get("confidence", 0.0),
            session_id     = d.get("session_id", ""),
        )
        evt.timestamp    = d.get("timestamp", time.time())
        evt.fingerprint  = d.get("fingerprint", evt.fingerprint)
        return evt


class FailureMemory:
    """
    Append-only episodic failure store with spatial nearest-neighbour query.

    Usage:
        mem = FailureMemory("aero_mesh_output/failures.jsonl")
        mem.record_failure(pos, FailureMode.LOW_CONFIDENCE, view_dir, entity_id="bldg_3")
        similar = mem.query_similar(pos, k=5)
        heat_grid = mem.export_for_rl_obs(center_xyz, radius_m=50, grid_size=32)
    """

    def __init__(
        self,
        store_path: str = "aero_mesh_output/failures.jsonl",
        session_id: str = "",
    ):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"sess_{int(time.time())}"
        self._events: List[FailureEvent] = []
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self):
        """Load all historical events from the append-only JSONL file."""
        if not self.store_path.exists():
            return
        with open(self.store_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._events.append(FailureEvent.from_dict(json.loads(line)))
                except Exception:
                    pass
        print(f"[FailureMemory] Loaded {len(self._events)} historical events from {self.store_path}")

    def _append_to_store(self, event: FailureEvent):
        """Atomically append one event to the JSONL store."""
        with open(self.store_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ── Record ─────────────────────────────────────────────────────────────────

    def record_failure(
        self,
        position_xyz:   np.ndarray,
        mode:           FailureMode,
        view_direction: np.ndarray,
        entity_id:      Optional[str] = None,
        confidence:     float         = 0.0,
    ) -> FailureEvent:
        """
        Record a reconstruction failure event. Immediately persisted to disk.
        Safe to call during active flight — append is atomic.
        """
        event = FailureEvent(
            position_xyz   = position_xyz,
            mode           = mode,
            view_direction = view_direction,
            entity_id      = entity_id,
            confidence     = confidence,
            session_id     = self.session_id,
        )
        self._events.append(event)
        self._append_to_store(event)
        return event

    # ── Query ──────────────────────────────────────────────────────────────────

    def query_similar(
        self,
        position_xyz: np.ndarray,
        k:            int = 5,
        max_radius_m: float = 30.0,
        mode_filter:  Optional[FailureMode] = None,
    ) -> List[FailureEvent]:
        """
        Return up to k nearest historical failures within max_radius_m.
        Optionally filter by failure mode.
        """
        if not self._events:
            return []

        query = np.array(position_xyz, dtype=np.float32)
        candidates = self._events
        if mode_filter is not None:
            candidates = [e for e in candidates if e.mode == mode_filter]

        dists = [np.linalg.norm(e.position_xyz - query) for e in candidates]
        sorted_pairs = sorted(zip(dists, candidates), key=lambda x: x[0])
        return [evt for dist, evt in sorted_pairs if dist <= max_radius_m][:k]

    def export_for_rl_obs(
        self,
        center_xyz: np.ndarray,
        radius_m:   float = 50.0,
        grid_size:  int   = 32,
    ) -> np.ndarray:
        """
        Export a 2D failure-density heat grid for the RL agent's observation vector.
        Returns: (grid_size × grid_size) float32 array, values in [0, 1].
        Values represent normalised failure event density in each grid cell.
        """
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        center = np.array(center_xyz, dtype=np.float32)
        cell_size = (2 * radius_m) / grid_size

        for evt in self._events:
            dx = evt.position_xyz[0] - center[0]
            dz = evt.position_xyz[2] - center[2]
            if abs(dx) > radius_m or abs(dz) > radius_m:
                continue
            cx = int((dx + radius_m) / cell_size)
            cz = int((dz + radius_m) / cell_size)
            cx = min(cx, grid_size - 1)
            cz = min(cz, grid_size - 1)
            grid[cz, cx] += 1.0

        max_val = grid.max()
        if max_val > 0:
            grid /= max_val
        return grid.flatten()  # (grid_size*grid_size,) flat vector for RL obs

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Summary statistics across all stored failures."""
        mode_counts = {}
        for evt in self._events:
            mode_counts[evt.mode.value] = mode_counts.get(evt.mode.value, 0) + 1
        return {
            "total_failures": len(self._events),
            "by_mode": mode_counts,
            "sessions": len(set(e.session_id for e in self._events)),
            "store_path": str(self.store_path),
        }

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"FailureMemory({len(self._events)} events, session={self.session_id})"
