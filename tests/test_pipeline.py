"""
Unit tests for AERO MESH modules.
"""

import unittest
import numpy as np
import os
import shutil
import tempfile

from aero_mesh.core.frame import Frame, CameraPose
from aero_mesh.ingest.frame_extractor import laplacian_sharpness, FrameExtractor
from aero_mesh.fusion.eskf import ESKFFusion
from aero_mesh.sfm.colmap_wrapper import SparseReconstruction
from aero_mesh.depth.scale_aligner import MonocularScaleAligner
from aero_mesh.confidence.confidence_engine import ConfidenceEngine
from aero_mesh.dataset.synthetic_generator import generate_synthetic_uav_dataset

class TestAeroMesh(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_laplacian_sharpness(self):
        img_sharp = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        sharpness = laplacian_sharpness(img_sharp)
        self.assertGreater(sharpness, 0.0)

    def test_eskf_propagation(self):
        eskf = ESKFFusion()
        accel = np.array([0.1, 0.0, 0.0])
        gyro = np.array([0.0, 0.0, 0.05])
        eskf.predict(accel, gyro, dt=0.1)
        pose = eskf.get_current_pose()
        self.assertIsNotNone(pose.position)
        self.assertEqual(len(pose.orientation), 4)

    def test_scale_aligner(self):
        aligner = MonocularScaleAligner()
        mono_depth = np.ones((50, 50)) * 0.5
        sparse_depth = np.zeros((50, 50))
        sparse_depth[10, 10] = 15.0
        mask = np.zeros((50, 50), dtype=bool)
        mask[10, 10] = True

        aligned, scale, shift = aligner.align_scale_and_shift(mono_depth, sparse_depth, mask)
        self.assertGreater(scale, 0.0)
        self.assertEqual(aligned.shape, (50, 50))

    def test_confidence_engine(self):
        engine = ConfidenceEngine()
        scene_data = {
            "confidences": np.array([0.9, 0.5, 0.2]),
            "source_tags": np.array([1.0, 0.6, 0.2])
        }
        res = engine.calculate_confidence_map(scene_data)
        self.assertIn("stats", res)
        self.assertEqual(len(res["heatmap_colors"]), 3)

    def test_synthetic_dataset_and_pipeline(self):
        ds_info = generate_synthetic_uav_dataset(self.temp_dir, num_frames=5)
        self.assertTrue(os.path.exists(ds_info["imu_file"]))
        self.assertTrue(os.path.exists(ds_info["gps_file"]))
        self.assertEqual(len(os.listdir(ds_info["images_dir"])), 5)

if __name__ == "__main__":
    unittest.main()
