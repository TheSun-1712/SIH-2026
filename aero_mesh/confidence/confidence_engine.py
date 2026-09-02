"""
Stage 8: Confidence Mapping & Georeferencing Engine
Calculates observed vs inferred confidence heatmaps and applies georeferencing transforms.
"""

import numpy as np
from typing import Dict, Tuple

class ConfidenceEngine:
    """Computes multi-factor spatial confidence maps and georeferenced coordinates."""

    def __init__(self, w_view: float = 0.3, w_mvs: float = 0.3, w_source: float = 0.4):
        self.w_view = w_view
        self.w_mvs = w_mvs
        self.w_source = w_source

    def calculate_confidence_map(self, scene_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Computes normalized confidence score C(p) in [0, 1] for every point/vertex.
        Generates RGB confidence heatmaps (Blue = High, Yellow = Mid, Red = Low).
        """
        confidences = scene_data["confidences"]
        source_tags = scene_data["source_tags"]
        num_pts = len(confidences)

        # Composite score
        final_scores = (self.w_mvs * confidences) + (self.w_source * source_tags)
        final_scores = np.clip(final_scores, 0.0, 1.0)

        # Heatmap RGB generation
        # Red (Low, 0.0) -> Yellow (Mid, 0.5) -> Blue/Cyan (High, 1.0)
        heatmap_rgb = np.zeros((num_pts, 3), dtype=np.float64)

        for i, score in enumerate(final_scores):
            if score >= 0.75:
                # High confidence observed (Cyan / Royal Blue)
                t = (score - 0.75) / 0.25
                heatmap_rgb[i] = [0.0, 0.4 * (1 - t) + 0.8 * t, 0.9 * (1 - t) + 1.0 * t]
            elif score >= 0.40:
                # Medium confidence (Amber / Yellow)
                t = (score - 0.40) / 0.35
                heatmap_rgb[i] = [0.9 * (1 - t) + 0.2 * t, 0.8 * (1 - t) + 0.8 * t, 0.1 * (1 - t) + 0.4 * t]
            else:
                # Low confidence (Crimson Red)
                t = score / 0.40
                heatmap_rgb[i] = [0.95, 0.15 * t, 0.15 * t]

        scene_data["final_confidence_scores"] = final_scores
        scene_data["heatmap_colors"] = heatmap_rgb

        # Summary statistics
        high_obs_pct = float(np.mean(final_scores >= 0.75) * 100.0)
        mono_infill_pct = float(np.mean((final_scores >= 0.40) & (final_scores < 0.75)) * 100.0)
        hallucinated_pct = float(np.mean(final_scores < 0.40) * 100.0)

        scene_data["stats"] = {
            "high_confidence_observed_pct": round(high_obs_pct, 1),
            "monocular_infill_pct": round(mono_infill_pct, 1),
            "hallucinated_inferred_pct": round(hallucinated_pct, 1),
            "mean_confidence": round(float(np.mean(final_scores)), 3)
        }

        return scene_data

    def georeference_point_cloud(self, points_3d: np.ndarray, origin_lat_lon_alt: Tuple[float, float, float]) -> np.ndarray:
        """Applies GPS origin transform to convert local metric coords to UTM/WGS84 offsets."""
        lat0, lon0, alt0 = origin_lat_lon_alt
        
        # Approximate local meters to GPS degree conversion (WGS84 ellipsoid near equator/mid-latitudes)
        meters_per_deg_lat = 111132.92
        meters_per_deg_lon = 111412.84 * np.cos(np.radians(lat0))

        geo_coords = np.zeros_like(points_3d)
        geo_coords[:, 0] = lon0 + (points_3d[:, 0] / meters_per_deg_lon)  # Longitude
        geo_coords[:, 1] = lat0 + (points_3d[:, 1] / meters_per_deg_lat)  # Latitude
        geo_coords[:, 2] = alt0 + points_3d[:, 2]                         # Altitude (meters)

        return geo_coords
