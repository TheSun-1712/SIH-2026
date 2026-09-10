import os
import sys
import argparse
import glob
import numpy as np
from typing import List

# Ensure aero_mesh is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aero_mesh.core.frame import Frame
from aero_mesh.core.pose import Pose6D
from aero_mesh.sfm.colmap_wrapper import SparseReconstruction
from aero_mesh.splat.splat_completion import GaussianSplatCompletion

def run_pipeline(image_dir: str, workspace_dir: str):
    print(f"--- Starting 3DGS Pipeline ---")
    print(f"Image Directory: {image_dir}")
    print(f"Workspace Directory: {workspace_dir}")

    # 1. Create Frames
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        print("Error: No .jpg images found in the specified directory.")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images.")
    frames: List[Frame] = []
    for i, path in enumerate(image_paths):
        dummy_pose = Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        frame = Frame(
            timestamp=float(i),
            image_path=path,
            telemetry={"lat": 0, "lon": 0, "alt": 50},
            estimated_pose=dummy_pose
        )
        frame.is_keyframe = True
        frames.append(frame)

    # 2. Run COLMAP
    print("\n--- Running COLMAP (Sparse Reconstruction) ---")
    sfm = SparseReconstruction(use_colmap_if_available=True, use_lightglue=True)
    sfm_results = sfm.run_sfm(frames, workspace_dir)

    if sfm_results.get("colmap_ba"):
        print("COLMAP bundle adjustment successful.")
    else:
        print("COLMAP failed or not used. Will use internal matches.")

    # 3. Run Nerfstudio Splatfacto
    print("\n--- Running Nerfstudio Splatfacto ---")
    splat = GaussianSplatCompletion(
        nerfstudio_output_dir=os.path.join(workspace_dir, "nerfstudio_output")
    )
    
    # We call _run_splatfacto explicitly to train the scene
    print(f"Triggering ns-train splatfacto on {workspace_dir}...")
    splat._run_splatfacto(data_dir=workspace_dir)

    print("\n--- Pipeline Finished ---")
    print("If training completed successfully, .splat files can be exported from the checkpoint.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3DGS pipeline with Max Accuracy and Checkpointing")
    parser.add_argument("--image_dir", type=str, default="VisDrone2019-DET-val/VisDrone2019-DET-val/images", help="Path to drone images")
    parser.add_argument("--workspace_dir", type=str, default="aero_mesh_output/3dgs_workspace", help="Path for COLMAP and Nerfstudio outputs")
    args = parser.parse_args()

    run_pipeline(args.image_dir, args.workspace_dir)
