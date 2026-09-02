"""
Tiered Exporter: PLY (binary/ASCII), GeoTIFF DSM, LAS, and JSON deliverables.
Supports Open3D for Poisson surface mesh reconstruction.
"""

import os
import json
import struct
import numpy as np
from typing import Dict, Any, Optional

# Optional dependencies
_OPEN3D_AVAILABLE = False
_LASPY_AVAILABLE = False
_RASTERIO_AVAILABLE = False

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    pass

try:
    import laspy
    _LASPY_AVAILABLE = True
except ImportError:
    pass

try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    pass


class SceneExporter:
    """
    Exports georeferenced point clouds, meshes, and confidence maps.

    Formats:
      - PLY (binary, fast) — Tier 1 (sparse) and Tier 2 (dense confidence)
      - LAS/LAZ — for GIS import (QGIS, CloudCompare)
      - GeoTIFF DSM — rasterized Digital Surface Model
      - OBJ + Poisson mesh — textured surface mesh
      - JSON pipeline summary — for web viewer and logging
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # PLY Export (Binary for speed)
    # ------------------------------------------------------------------

    def export_ply(
        self,
        filename: str,
        points: np.ndarray,
        colors: np.ndarray,
        confidences: np.ndarray,
        heatmaps: np.ndarray,
        use_binary: bool = True,
    ) -> str:
        """
        Exports a PLY file with XYZ, RGB, confidence, and heatmap RGB channels.
        Binary format is ~5x faster than ASCII for large clouds (>100K pts).
        """
        file_path = os.path.join(self.output_dir, filename)
        num_verts = len(points)

        # Clamp and convert to uint8
        rgb_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        hm_uint8 = (np.clip(heatmaps, 0, 1) * 255).astype(np.uint8)
        confs_f32 = confidences.astype(np.float32)
        pts_f32 = points.astype(np.float32)

        if use_binary:
            self._write_ply_binary(file_path, num_verts, pts_f32,
                                    rgb_uint8, confs_f32, hm_uint8)
        else:
            self._write_ply_ascii(file_path, num_verts, pts_f32,
                                   rgb_uint8, confs_f32, hm_uint8)

        size_mb = os.path.getsize(file_path) / 1e6
        print(f"[Export] PLY ({'binary' if use_binary else 'ASCII'}): "
              f"{filename} — {num_verts:,} pts, {size_mb:.1f} MB")
        return file_path

    def _write_ply_binary(self, path: str, n: int,
                           pts: np.ndarray, rgb: np.ndarray,
                           confs: np.ndarray, hm: np.ndarray):
        """Write binary little-endian PLY for fast I/O."""
        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"comment AERO MESH — Binary PLY Export\n"
            f"element vertex {n}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property uchar red\n"
            f"property uchar green\n"
            f"property uchar blue\n"
            f"property float confidence\n"
            f"property uchar hm_red\n"
            f"property uchar hm_green\n"
            f"property uchar hm_blue\n"
            f"end_header\n"
        )
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            # Pack all vertices as a structured array for speed
            dt = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                ('conf', '<f4'),
                ('hm_r', 'u1'), ('hm_g', 'u1'), ('hm_b', 'u1'),
            ])
            arr = np.empty(n, dtype=dt)
            arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
            arr['r'], arr['g'], arr['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            arr['conf'] = confs
            arr['hm_r'], arr['hm_g'], arr['hm_b'] = hm[:, 0], hm[:, 1], hm[:, 2]
            f.write(arr.tobytes())

    def _write_ply_ascii(self, path: str, n: int,
                          pts: np.ndarray, rgb: np.ndarray,
                          confs: np.ndarray, hm: np.ndarray):
        """ASCII PLY for human-readability / compatibility."""
        header = (
            f"ply\nformat ascii 1.0\n"
            f"comment AERO MESH — ASCII PLY Export\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"property float confidence\n"
            f"property uchar hm_red\nproperty uchar hm_green\nproperty uchar hm_blue\n"
            f"end_header\n"
        )
        with open(path, "w") as f:
            f.write(header)
            for i in range(n):
                px, py, pz = pts[i]
                r, g, b = rgb[i]
                conf = confs[i]
                hmr, hmg, hmb = hm[i]
                f.write(f"{px:.4f} {py:.4f} {pz:.4f} "
                        f"{r} {g} {b} {conf:.4f} {hmr} {hmg} {hmb}\n")

    # ------------------------------------------------------------------
    # Poisson Surface Mesh (Open3D)
    # ------------------------------------------------------------------

    def export_mesh_obj(
        self,
        filename: str,
        points: np.ndarray,
        colors: np.ndarray,
        normals: Optional[np.ndarray] = None,
        depth: int = 9,
    ) -> Optional[str]:
        """
        Generates a watertight Poisson surface mesh using Open3D and exports as OBJ.
        Requires Open3D. Depth=9 is a good trade-off between detail and speed.
        """
        if not _OPEN3D_AVAILABLE:
            print("[Export] Open3D not available — skipping OBJ mesh export.")
            return None

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        else:
            print("[Export] Estimating normals for Poisson mesh...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(30)

        print(f"[Export] Running Poisson surface reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )

        # Remove low-density vertices (outlier triangles)
        dens = np.asarray(densities)
        threshold = np.quantile(dens, 0.05)
        vertices_to_remove = dens < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        file_path = os.path.join(self.output_dir, filename)
        o3d.io.write_triangle_mesh(file_path, mesh, write_ascii=False)
        n_tris = len(mesh.triangles)
        print(f"[Export] OBJ mesh: {filename} — {n_tris:,} triangles")
        return file_path

    # ------------------------------------------------------------------
    # LAS Point Cloud Export
    # ------------------------------------------------------------------

    def export_las(
        self,
        filename: str,
        points: np.ndarray,
        colors: np.ndarray,
        confidences: np.ndarray,
    ) -> Optional[str]:
        """
        Exports georeferenced LAS 1.4 point cloud for QGIS / CloudCompare.
        Requires laspy: pip install laspy
        """
        if not _LASPY_AVAILABLE:
            print("[Export] laspy not installed — skipping LAS export. "
                  "Install with: pip install laspy")
            return None

        import laspy
        header = laspy.LasHeader(point_format=2, version="1.4")
        las = laspy.LasData(header=header)

        # Scale to integer (LAS stores integers with scale offsets)
        las.header.offsets = np.min(points, axis=0)
        las.header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision

        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        rgb_uint16 = (np.clip(colors, 0, 1) * 65535).astype(np.uint16)
        las.red = rgb_uint16[:, 0]
        las.green = rgb_uint16[:, 1]
        las.blue = rgb_uint16[:, 2]

        file_path = os.path.join(self.output_dir, filename)
        las.write(file_path)
        print(f"[Export] LAS: {filename} — {len(points):,} points")
        return file_path

    # ------------------------------------------------------------------
    # GeoTIFF DSM Export
    # ------------------------------------------------------------------

    def export_dsm_geotiff(
        self,
        filename: str,
        points: np.ndarray,
        origin_lat_lon: tuple,
        resolution: float = 0.5,  # meters per pixel
    ) -> Optional[str]:
        """
        Rasterizes the point cloud into a GeoTIFF Digital Surface Model.
        Requires rasterio: pip install rasterio
        """
        if not _RASTERIO_AVAILABLE:
            print("[Export] rasterio not installed — skipping GeoTIFF export. "
                  "Install with: pip install rasterio")
            return None

        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        lat0, lon0 = origin_lat_lon
        meters_per_deg_lat = 111132.92
        meters_per_deg_lon = 111412.84 * np.cos(np.radians(lat0))

        # Local metric extents
        min_x, max_x = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
        min_y, max_y = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
        min_z, max_z = float(np.min(points[:, 2])), float(np.max(points[:, 2]))

        width = max(1, int((max_x - min_x) / resolution))
        height = max(1, int((max_y - min_y) / resolution))

        dsm = np.full((height, width), np.nan, dtype=np.float32)

        # Rasterize max-height per cell
        for pt in points:
            col = int((pt[0] - min_x) / resolution)
            row = int((max_y - pt[1]) / resolution)
            col = min(col, width - 1)
            row = min(row, height - 1)
            if np.isnan(dsm[row, col]) or pt[2] > dsm[row, col]:
                dsm[row, col] = pt[2]

        # Fill NaN with minimum (no-data)
        dsm = np.where(np.isnan(dsm), min_z - 1.0, dsm)

        # Convert bounds to geographic degrees
        west = lon0 + min_x / meters_per_deg_lon
        east = lon0 + max_x / meters_per_deg_lon
        south = lat0 + min_y / meters_per_deg_lat
        north = lat0 + max_y / meters_per_deg_lat

        transform = from_bounds(west, south, east, north, width, height)
        crs = CRS.from_epsg(4326)  # WGS84

        file_path = os.path.join(self.output_dir, filename)
        with rasterio.open(
            file_path, "w",
            driver="GTiff", height=height, width=width,
            count=1, dtype="float32",
            crs=crs, transform=transform,
            nodata=min_z - 1.0,
        ) as dst:
            dst.write(dsm, 1)

        print(f"[Export] GeoTIFF DSM: {filename} — {width}x{height}px @ {resolution}m/px")
        return file_path

    # ------------------------------------------------------------------
    # JSON Summary
    # ------------------------------------------------------------------

    def export_summary_json(
        self,
        filename: str,
        scene_stats: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """Exports pipeline summary JSON for web viewer and logging."""
        file_path = os.path.join(self.output_dir, filename)
        payload = {
            "project": "AERO MESH — Single-Pass UAV 3D Reconstruction",
            "version": "2.0",
            "stats": scene_stats,
            "metadata": metadata,
        }
        with open(file_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Export] Summary JSON: {filename}")
        return file_path
