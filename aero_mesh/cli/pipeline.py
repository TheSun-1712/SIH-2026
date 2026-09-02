"""
Pipeline Orchestrator & CLI Runner for AERO MESH.
Executes Stages 1-8 sequentially and generates Tier 1 & Tier 2 georeferenced outputs.
"""

import os
import sys
import json
import argparse
import numpy as np

from aero_mesh.core.frame import Frame
from aero_mesh.ingest.frame_extractor import FrameExtractor, laplacian_sharpness
from aero_mesh.fusion.eskf import ESKFFusion
from aero_mesh.sfm.colmap_wrapper import SparseReconstruction
from aero_mesh.depth.scale_aligner import MonocularScaleAligner
from aero_mesh.mvs.tsdf_fusion import TSDFVolumetricFusion
from aero_mesh.splat.splat_completion import GaussianSplatCompletion
from aero_mesh.segmentation.motion_filter import DynamicObjectFilter
from aero_mesh.confidence.confidence_engine import ConfidenceEngine
from aero_mesh.export.exporter import SceneExporter
from aero_mesh.dataset.synthetic_generator import generate_synthetic_uav_dataset

class AeroMeshPipeline:
    """Master pipeline execution engine for AERO MESH."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.exporter = SceneExporter(output_dir)

    def run_pipeline(self, dataset_dir: str) -> dict:
        print("==========================================================")
        print("   AERO MESH: UAV 3D Spatial Reconstruction & Confidence ")
        print("==========================================================")

        # 1. Load telemetry and images
        imu_file = os.path.join(dataset_dir, "imu_telemetry.json")
        gps_file = os.path.join(dataset_dir, "gps_telemetry.json")
        images_dir = os.path.join(dataset_dir, "images")

        with open(imu_file, "r") as f:
            imu_data = json.load(f)
        with open(gps_file, "r") as f:
            gps_data = json.load(f)

        image_files = sorted([os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith(('.jpg', '.png'))])

        # Stage 1: Frame Extraction & Quality Filtering
        print("\n[Stage 1] Ingest & Laplacian Sharpness Filtering...")
        extractor = FrameExtractor(sharpness_threshold=30.0)
        frames = []
        for idx, img_p in enumerate(image_files):
            import cv2
            mat = cv2.imread(img_p)
            sharpness = laplacian_sharpness(mat)
            f = Frame(
                frame_id=idx,
                timestamp=idx * 0.1,
                image_path=img_p,
                image_data=mat,
                sharpness_score=sharpness
            )
            frames.append(f)

        # Stage 2: Sensor Fusion (ESKF)
        print("[Stage 2] 15-State ESKF Sensor Fusion (GPS + IMU + Baro)...")
        fusion = ESKFFusion()
        frames = fusion.synchronize_frames(frames, imu_data, gps_data)

        # Select keyframes after pose estimation
        keyframes = extractor.select_keyframes(frames, min_baseline_meters=1.0)
        print(f" -> Selected {len(keyframes)} high-overlap keyframes out of {len(frames)} total frames.")

        # Stage 3: Sparse Reconstruction (SfM)
        print("[Stage 3] Sparse SfM & Pose-Prior Triangulation...")
        sfm = SparseReconstruction()
        sfm_workspace = os.path.join(self.output_dir, "sfm_workspace")
        sparse_cloud = sfm.run_sfm(keyframes, sfm_workspace)
        print(f" -> Triangulated {len(sparse_cloud['points_3d'])} metric 3D sparse points with pose priors.")

        # Stage 4: Monocular Depth Infill & Scale Alignment
        print("[Stage 4] Monocular Depth Scale-Shift Alignment...")
        aligner = MonocularScaleAligner()
        for kf in keyframes:
            kf = aligner.infill_frame_depth(kf, sparse_cloud['points_3d'])

        # Stage 7: Motion Masking (Dynamic object removal)
        print("[Stage 7] RAFT Optical Flow & Ego-Motion Dynamic Object Filter...")
        motion_filter = DynamicObjectFilter()
        keyframes = motion_filter.filter_dynamic_objects(keyframes)

        # Stage 5: Dense TSDF Volumetric Fusion
        print("[Stage 5] Multi-View Stereo & TSDF Volumetric Integration...")
        tsdf = TSDFVolumetricFusion(voxel_size=0.05)
        dense_cloud = tsdf.fuse_frames(keyframes)
        print(f" -> TSDF integrated {len(dense_cloud['points_3d'])} dense metric points.")

        # Stage 6: Neural Scene Completion (3D Gaussian Splatting)
        print("[Stage 6] 3D Gaussian Splatting Scene Completion for Occluded Surfaces...")
        splat = GaussianSplatCompletion(num_gaussians=1500)
        completed_scene = splat.complete_occluded_regions(dense_cloud)
        print(f" -> Generated {completed_scene.get('num_hallucinated', 0)} 3DGS novelty completion primitives.")

        # Stage 8: Confidence Mapping & Georeferencing
        print("[Stage 8] Confidence Heatmap Scoring & Georeferencing Engine...")
        engine = ConfidenceEngine()
        final_scene = engine.calculate_confidence_map(completed_scene)
        
        origin_lat_lon_alt = gps_data[0]["lat_lon_alt"]
        geo_points = engine.georeference_point_cloud(final_scene["points_3d"], origin_lat_lon_alt)

        # Export Deliverables
        print("\n[Exporting Tier 1 & Tier 2 Deliverables]")
        tier1_ply = self.exporter.export_ply(
            "tier1_sparse_cloud.ply",
            sparse_cloud["points_3d"],
            sparse_cloud["colors_3d"],
            sparse_cloud["confidences"],
            np.tile(np.array([[0.0, 0.8, 1.0]]), (len(sparse_cloud["points_3d"]), 1))
        )
        
        tier2_ply = self.exporter.export_ply(
            "tier2_dense_confidence_mesh.ply",
            final_scene["points_3d"],
            final_scene["colors_3d"],
            final_scene["final_confidence_scores"],
            final_scene["heatmap_colors"]
        )

        summary_json = self.exporter.export_summary_json(
            "pipeline_summary.json",
            final_scene["stats"],
            {
                "num_frames_processed": len(frames),
                "num_keyframes_selected": len(keyframes),
                "total_points_3d": len(final_scene["points_3d"]),
                "origin_lat_lon_alt": origin_lat_lon_alt,
                "tier1_file": os.path.basename(tier1_ply),
                "tier2_file": os.path.basename(tier2_ply)
            }
        )

        print(f" -> Tier 1 Output: {tier1_ply}")
        print(f" -> Tier 2 Output: {tier2_ply}")
        print(f" -> Summary JSON: {summary_json}")
        print("\nPipeline Statistics:")
        print(json.dumps(final_scene["stats"], indent=2))
        print("\n==========================================================")
        print("   AERO MESH Pipeline Run Completed Successfully!        ")
        print("==========================================================")

        return final_scene

def main():
    parser = argparse.ArgumentParser(description="AERO MESH UAV 3D Reconstruction Pipeline")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to input flight dataset directory")
    parser.add_argument("--output_dir", type=str, default="aero_mesh_output", help="Path to output deliverables directory")
    args = parser.parse_args()

    if args.dataset_dir is None:
        print("[Info] No dataset directory specified. Generating synthetic flight dataset...")
        dataset_info = generate_synthetic_uav_dataset("sample_flight_data", num_frames=15)
        args.dataset_dir = dataset_info["dataset_dir"]

    pipeline = AeroMeshPipeline(args.output_dir)
    pipeline.run_pipeline(args.dataset_dir)

if __name__ == "__main__":
    main()
