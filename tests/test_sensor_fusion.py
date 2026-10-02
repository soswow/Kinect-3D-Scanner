import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
import unittest
from dataclasses import replace

import numpy as np
import open3d as o3d

from scanner_server.engine import ScanEngine
from shared.confidence import depth_confidence
from shared.settings import CameraCalibration
from shared.surface_metrics import surface_metrics


def plate(width):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [-width / 2, -0.5, 1],
            [width / 2, -0.5, 1],
            [width / 2, 0.5, 1],
            [-width / 2, 0.5, 1],
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    return mesh


class SensorFusionTests(unittest.TestCase):
    def test_weighted_tracking_and_final_mesh_on_asymmetric_scene(self):
        from tests.test_quality import scene_frames

        engine = ScanEngine()
        engine.reset(
            settings=replace(engine.settings, confidence_fusion=True, final_weight=0.5)
        )
        for rgb, depth, _ in scene_frames(5):
            engine.store_frame(rgb, depth)
        engine.process_frames()
        self.assertEqual(5, engine.frame_count, engine.diagnostics)
        self.assertTrue(engine.build_mesh()[0])
        self.assertGreater(len(engine.mesh.triangles), 100)
        self.assertGreater(
            engine.diagnostics[-1]["fusion_confidence"]["mean_observation_weight"], 0
        )

    def test_confidence_preserves_unknown_space_and_downweights_range(self):
        camera = CameraCalibration()
        depth = np.full((480, 640), 1000, np.uint16)
        near = depth_confidence(depth, camera)
        far = depth_confidence(depth * 3, camera)
        self.assertGreater(near[240, 320], far[240, 320] * 10)
        depth[200:210, 200:210] = 0
        depth[:, 400:] = 2000
        weighted = depth_confidence(depth, camera)
        self.assertTrue(np.all(weighted[200:210, 200:210] == 0))
        self.assertTrue(np.all(weighted[:, 399:401] == 0))
        self.assertGreater(weighted[240, 390], 0)

    def test_weighted_fusion_reduces_controlled_far_observation_bias(self):
        medians = []
        for mode in (False, True):
            engine = ScanEngine()
            engine.reset(
                settings=replace(
                    engine.settings,
                    confidence_fusion=mode,
                    truncation_m=0.06,
                    filter_depth=False,
                )
            )
            rgb = np.full((480, 640, 3), 100, np.uint8)
            engine._integrate_vbg(rgb, np.full((480, 640), 1012, np.uint16), np.eye(4))
            extrinsic = np.eye(4)
            extrinsic[2, 3] = 1
            engine._integrate_vbg(rgb, np.full((480, 640), 2052, np.uint16), extrinsic)
            points = engine.vbg.extract_point_cloud(
                weight_threshold=0.01
            ).point.positions.numpy()
            center = points[(np.abs(points[:, 0]) < 0.1) & (np.abs(points[:, 1]) < 0.1)]
            medians.append(float(np.median(center[:, 2])))
            self.assertGreater(len(center), 100)
        self.assertLess(abs(medians[1] - 1.012), abs(medians[0] - 1.012) * 0.5, medians)

    def test_low_error_does_not_hide_missing_surface(self):
        report = surface_metrics(plate(0.5), plate(1.0), threshold_m=0.01, samples=4000)
        self.assertLess(report["surface_rmse_m"], 1e-5)
        self.assertGreater(report["precision"], 0.99)
        self.assertLess(report["completeness"], 0.55)
        self.assertGreater(report["completeness"], 0.45)
