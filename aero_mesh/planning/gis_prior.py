"""
AERO MESH — GIS Prior Bootstrapper
=====================================
Pre-flight OSM (OpenStreetMap) query that seeds the reconstruction system
with real building footprints, approximate heights, road networks, and
land-use tags BEFORE the drone takes off.

This gives every downstream component a head start:
  - Shape Grammar Engine: LoD1 box proxies immediately visible
  - Frontier Planner: pre-allocated sectors by building density
  - TSDF: soft building boundary injection
  - VLM Classifier: land-use hints for asset selection

Dependencies:
    pip install overpy shapely

No API key required — uses the public OSM Overpass API.
"""

import json
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict

try:
    import overpy
    _OVERPY_AVAILABLE = True
except ImportError:
    _OVERPY_AVAILABLE = False
    print("[GISPrior] overpy not installed. Install: pip install overpy")
    print("[GISPrior] Will operate in offline/synthetic mode.")

try:
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False


# ── OSM Tag → Architectural Type Mapping ─────────────────────────────────────
OSM_BUILDING_TYPE_MAP = {
    # Specific OSM building types → our VLM architectural categories
    "apartments":    "high_rise_residential",
    "residential":   "high_rise_residential",
    "house":         "villa_house",
    "detached":      "villa_house",
    "semidetached":  "villa_house",
    "bungalow":      "villa_house",
    "office":        "office_tower",
    "commercial":    "commercial_shopfront",
    "retail":        "commercial_shopfront",
    "shop":          "commercial_shopfront",
    "supermarket":   "commercial_shopfront",
    "industrial":    "industrial_warehouse",
    "warehouse":     "industrial_warehouse",
    "factory":       "industrial_warehouse",
    "civic":         "office_tower",
    "hospital":      "office_tower",
    "school":        "apartment_block",
    "university":    "apartment_block",
    "yes":           "apartment_block",       # generic fallback
}

# Average height per floor by building type (metres)
FLOOR_HEIGHT_ESTIMATE = {
    "high_rise_residential": 2.85,
    "office_tower":          3.50,
    "commercial_shopfront":  4.50,
    "industrial_warehouse":  8.00,
    "villa_house":           3.00,
    "apartment_block":       3.00,
}

# Fallback floor count where OSM has no height/levels data
DEFAULT_FLOORS_BY_TYPE = {
    "high_rise_residential": 6,
    "office_tower":          8,
    "commercial_shopfront":  2,
    "industrial_warehouse":  1,
    "villa_house":           2,
    "apartment_block":       4,
}


@dataclass
class BuildingPrior:
    """A single building footprint from OSM with estimated height."""
    id:           str
    osm_id:       int
    arch_type:    str                    # VLM architectural category
    lat_lon_ring: List[Tuple[float, float]]   # footprint polygon in lat/lon
    local_ring:   List[Tuple[float, float]]   # footprint in local metres (x, z)
    center_local: Tuple[float, float]    # (x, z) centre in local coords
    width_m:      float
    depth_m:      float
    heading_rad:  float
    est_height_m: float
    floor_count:  int
    land_use:     str = ""
    osm_tags:     Dict[str, str] = field(default_factory=dict)


@dataclass
class RoadSegment:
    """A road polyline from OSM."""
    id:        str
    osm_id:    int
    polyline:  List[Tuple[float, float]]   # local (x, z) coords
    width_m:   float
    road_type: str                         # primary | secondary | residential | path


@dataclass
class GISManifest:
    """Full preflight GIS data package for a survey area."""
    query_lat:     float
    query_lon:     float
    radius_m:      float
    buildings:     List[BuildingPrior]
    roads:         List[RoadSegment]
    origin_lat:    float
    origin_lon:    float
    queried_at:    str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_lat":  self.query_lat,
            "query_lon":  self.query_lon,
            "radius_m":   self.radius_m,
            "origin_lat": self.origin_lat,
            "origin_lon": self.origin_lon,
            "queried_at": self.queried_at,
            "buildings":  [asdict(b) for b in self.buildings],
            "roads":      [asdict(r) for r in self.roads],
        }


class GISPrior:
    """
    Pre-flight GIS bootstrapper.

    Usage:
        gis = GISPrior()
        manifest = gis.query(lat=28.6139, lon=77.2090, radius_m=300)
        gis.export_entity_manifest(manifest, "web/public/gis_entities.json")
        gis.seed_lod1_entities(manifest)  → returns LoD1 entity dict for viewer
    """

    # Overpass API endpoints (rotated for resilience)
    OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
    ]

    def __init__(self, cache_dir: str = "aero_mesh_output/gis_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    @staticmethod
    def latlon_to_local(lat: float, lon: float,
                        origin_lat: float, origin_lon: float) -> Tuple[float, float]:
        """Convert lat/lon to local flat-earth metres (x=East, z=South)."""
        metres_per_deg_lat = 111132.92
        metres_per_deg_lon = 111132.92 * math.cos(math.radians(origin_lat))
        x = (lon - origin_lon) * metres_per_deg_lon
        z = (lat - origin_lat) * metres_per_deg_lat
        return float(x), float(z)

    # ── OSM Query ──────────────────────────────────────────────────────────────

    def _build_overpass_query(self, lat: float, lon: float, radius_m: float) -> str:
        return f"""
        [out:json][timeout:30];
        (
          way["building"](around:{radius_m},{lat},{lon});
          way["highway"](around:{radius_m},{lat},{lon});
        );
        (._;>;);
        out body;
        """

    def _try_query_osm(self, lat: float, lon: float, radius_m: float) -> Optional[Any]:
        """Attempt Overpass API query with endpoint rotation."""
        if not _OVERPY_AVAILABLE:
            return None
        for endpoint in self.OVERPASS_ENDPOINTS:
            try:
                api = overpy.Overpass(url=endpoint)
                result = api.query(self._build_overpass_query(lat, lon, radius_m))
                print(f"[GISPrior] OSM query OK via {endpoint}")
                return result
            except Exception as e:
                print(f"[GISPrior] Endpoint {endpoint} failed: {e}. Trying next…")
                time.sleep(1)
        return None

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_building(
        self, way, origin_lat: float, origin_lon: float, index: int
    ) -> Optional[BuildingPrior]:
        """Parse an OSM way into a BuildingPrior."""
        try:
            tags = way.tags
            btype_raw = tags.get("building", "yes")
            arch_type = OSM_BUILDING_TYPE_MAP.get(btype_raw, "apartment_block")

            # Lat/lon ring
            ring_latlon = [(float(n.lat), float(n.lon)) for n in way.nodes]
            if len(ring_latlon) < 3:
                return None

            # Local coordinates
            ring_local = [
                self.latlon_to_local(la, lo, origin_lat, origin_lon)
                for la, lo in ring_latlon
            ]

            # Bounding box
            xs = [p[0] for p in ring_local]
            zs = [p[1] for p in ring_local]
            cx = (min(xs) + max(xs)) / 2.0
            cz = (min(zs) + max(zs)) / 2.0
            width_m = max(xs) - min(xs)
            depth_m = max(zs) - min(zs)

            # Height estimation
            try:
                est_height = float(tags.get("height", 0))
            except (ValueError, TypeError):
                est_height = 0.0
            try:
                floor_count = int(tags.get("building:levels", 0))
            except (ValueError, TypeError):
                floor_count = 0

            if floor_count == 0 and est_height == 0.0:
                floor_count = DEFAULT_FLOORS_BY_TYPE.get(arch_type, 3)

            if est_height == 0.0:
                est_height = floor_count * FLOOR_HEIGHT_ESTIMATE.get(arch_type, 3.0)
            elif floor_count == 0:
                fh = FLOOR_HEIGHT_ESTIMATE.get(arch_type, 3.0)
                floor_count = max(1, round(est_height / fh))

            return BuildingPrior(
                id           = f"osm_bldg_{index}",
                osm_id       = int(way.id),
                arch_type    = arch_type,
                lat_lon_ring = ring_latlon,
                local_ring   = ring_local,
                center_local = (cx, cz),
                width_m      = max(width_m, 4.0),
                depth_m      = max(depth_m, 4.0),
                heading_rad  = 0.0,
                est_height_m = max(est_height, 3.0),
                floor_count  = max(floor_count, 1),
                land_use     = tags.get("landuse", ""),
                osm_tags     = dict(tags),
            )
        except Exception as e:
            print(f"[GISPrior] Warning: could not parse building way {way.id}: {e}")
            return None

    def _parse_road(
        self, way, origin_lat: float, origin_lon: float, index: int
    ) -> Optional[RoadSegment]:
        """Parse an OSM highway way into a RoadSegment."""
        try:
            tags = way.tags
            highway = tags.get("highway", "")
            if highway in ("footway", "cycleway", "path", "steps", "service"):
                return None  # Skip pedestrian/service paths

            road_type_map = {
                "motorway": "primary", "trunk": "primary",
                "primary": "primary", "secondary": "secondary",
                "tertiary": "secondary", "residential": "residential",
                "living_street": "residential",
            }
            road_type = road_type_map.get(highway, "residential")

            road_widths = {
                "primary": 10.0, "secondary": 7.0, "residential": 5.0
            }
            width_m = road_widths.get(road_type, 5.0)
            try:
                width_m = float(tags.get("width", width_m))
            except (ValueError, TypeError):
                pass

            polyline = [
                self.latlon_to_local(float(n.lat), float(n.lon), origin_lat, origin_lon)
                for n in way.nodes
            ]

            if len(polyline) < 2:
                return None

            return RoadSegment(
                id        = f"osm_road_{index}",
                osm_id    = int(way.id),
                polyline  = polyline,
                width_m   = width_m,
                road_type = road_type,
            )
        except Exception as e:
            print(f"[GISPrior] Warning: could not parse road way {way.id}: {e}")
            return None

    # ── Public Interface ───────────────────────────────────────────────────────

    def query(
        self,
        lat:      float,
        lon:      float,
        radius_m: float = 300.0,
        use_cache: bool = True,
    ) -> GISManifest:
        """
        Query OSM for buildings and roads around (lat, lon) within radius_m.
        Results are cached to disk — subsequent calls for the same area are instant.
        Falls back to an empty manifest if OSM is unreachable.
        """
        cache_key = f"{lat:.5f}_{lon:.5f}_{int(radius_m)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_file.exists():
            print(f"[GISPrior] Loading cached GIS data: {cache_file}")
            with open(cache_file) as f:
                raw = json.load(f)
            return self._manifest_from_cache(raw)

        print(f"[GISPrior] Querying OSM around ({lat:.5f}, {lon:.5f}) radius={radius_m}m…")
        origin_lat, origin_lon = lat, lon
        buildings: List[BuildingPrior] = []
        roads:     List[RoadSegment]   = []

        osm_result = self._try_query_osm(lat, lon, radius_m)

        if osm_result is not None:
            bldg_idx = 0
            road_idx = 0
            for way in osm_result.ways:
                if "building" in way.tags:
                    b = self._parse_building(way, origin_lat, origin_lon, bldg_idx)
                    if b:
                        buildings.append(b)
                        bldg_idx += 1
                elif "highway" in way.tags:
                    r = self._parse_road(way, origin_lat, origin_lon, road_idx)
                    if r:
                        roads.append(r)
                        road_idx += 1
        else:
            print("[GISPrior] OSM unavailable — operating without GIS prior.")

        manifest = GISManifest(
            query_lat  = lat,
            query_lon  = lon,
            radius_m   = radius_m,
            buildings  = buildings,
            roads      = roads,
            origin_lat = origin_lat,
            origin_lon = origin_lon,
            queried_at = time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # Cache result
        with open(cache_file, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        print(f"[GISPrior] Found {len(buildings)} buildings, {len(roads)} roads. "
              f"Cached to {cache_file}")
        return manifest

    def _manifest_from_cache(self, raw: Dict[str, Any]) -> GISManifest:
        """Reconstruct manifest from cached JSON."""
        buildings = [BuildingPrior(**b) for b in raw.get("buildings", [])]
        roads     = [RoadSegment(**r) for r in raw.get("roads", [])]
        return GISManifest(
            query_lat  = raw["query_lat"],
            query_lon  = raw["query_lon"],
            radius_m   = raw["radius_m"],
            buildings  = buildings,
            roads      = roads,
            origin_lat = raw["origin_lat"],
            origin_lon = raw["origin_lon"],
            queried_at = raw.get("queried_at", ""),
        )

    def export_entity_manifest(
        self,
        manifest:  GISManifest,
        out_path:  str = "web/public/gis_entities.json",
    ) -> str:
        """
        Export GIS manifest as entity JSON for the web viewer.
        Produces LoD1 box representations of all buildings.
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        entities = {
            "buildings": [
                {
                    "id":            b.id,
                    "type":          b.arch_type,
                    "footprint":     {
                        "cx":      b.center_local[0],
                        "cz":      b.center_local[1],
                        "w":       b.width_m,
                        "d":       b.depth_m,
                        "heading": b.heading_rad,
                    },
                    "measuredHeight": b.est_height_m,
                    "floorCount":     b.floor_count,
                    "confidence":     0.40,           # OSM prior = medium confidence
                    "lod":            1,              # Start at LoD1 — upgraded by drone
                    "source":         "osm_prior",
                }
                for b in manifest.buildings
            ],
            "roads": [
                {
                    "id":       r.id,
                    "polyline": r.polyline,
                    "width":    r.width_m,
                    "type":     r.road_type,
                    "source":   "osm_prior",
                }
                for r in manifest.roads
            ],
            "vegetation": [],
            "vehicles":   [],
            "meta": {
                "source":     "openstreetmap",
                "queried_at": manifest.queried_at,
                "origin_lat": manifest.origin_lat,
                "origin_lon": manifest.origin_lon,
                "radius_m":   manifest.radius_m,
            }
        }

        with open(out, "w") as f:
            json.dump(entities, f, indent=2)
        print(f"[GISPrior] Entity manifest exported → {out} "
              f"({len(manifest.buildings)} buildings, {len(manifest.roads)} roads)")
        return str(out)

    def get_sector_priorities(
        self, manifest: GISManifest, grid_cells: int = 8
    ) -> np.ndarray:
        """
        Compute a grid_cells×grid_cells priority map for the Frontier Planner.
        Cells with higher building density get higher priority values.
        Returns: (grid_cells*grid_cells,) float32 array, values in [0, 1].
        """
        if not manifest.buildings:
            return np.ones(grid_cells * grid_cells, dtype=np.float32)

        xs = [b.center_local[0] for b in manifest.buildings]
        zs = [b.center_local[1] for b in manifest.buildings]
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(zs), max(zs)
        x_range = max(x_max - x_min, 1.0)
        z_range = max(z_max - z_min, 1.0)

        grid = np.zeros((grid_cells, grid_cells), dtype=np.float32)
        for b in manifest.buildings:
            cx = int((b.center_local[0] - x_min) / x_range * (grid_cells - 1))
            cz = int((b.center_local[1] - z_min) / z_range * (grid_cells - 1))
            # Weight by estimated height — taller buildings get higher priority
            grid[cz, cx] += min(b.est_height_m / 30.0, 1.0)

        max_val = grid.max()
        if max_val > 0:
            grid /= max_val
        return grid.flatten()


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AERO MESH GIS Prior Query")
    parser.add_argument("--lat",    type=float, default=28.6139, help="Latitude")
    parser.add_argument("--lon",    type=float, default=77.2090, help="Longitude")
    parser.add_argument("--radius", type=float, default=300,     help="Radius in metres")
    parser.add_argument("--out",    type=str,   default="web/public/gis_entities.json")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    gis = GISPrior()
    manifest = gis.query(args.lat, args.lon, args.radius, use_cache=not args.no_cache)
    gis.export_entity_manifest(manifest, args.out)
    print(f"\nGIS Summary:")
    print(f"  Buildings : {len(manifest.buildings)}")
    print(f"  Roads     : {len(manifest.roads)}")
    print(f"  Output    : {args.out}")
