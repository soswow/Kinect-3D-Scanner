"""Withheld-view scoring must catch missing geometry and preserve a fixed anchor."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_quality import aggregate, disjoint_views, input_hash
from replay_scan import ReplayFrame, load_dataset

from shared.recording import RecordingWriter
from shared.settings import ScanSettings
from shared.view_metrics import heldout_view_metrics


def plane(right=1):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [[-1, -1, 1], [right, -1, 1], [right, 1, 1], [-1, 1, 1]]
    )
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    return mesh


def frames():
    return [
        (
            np.zeros((480, 640, 3), np.uint8),
            np.full((480, 640), 1000, np.uint16),
            1.0,
            np.eye(4),
        )
    ]


class ViewMetricTests(unittest.TestCase):
    def test_exact_surface_and_rigid_anchor_preserve_metric_depth(self):
        pose = np.eye(4)
        pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.2, -0.1, 0.05))
        pose[:3, 3] = [3, 4, 5]
        report = heldout_view_metrics(
            plane().transform(pose), frames(), ScanSettings(), pose
        )
        self.assertAlmostEqual(1, report["completeness_within_threshold"])
        self.assertLess(report["depth_rmse_over_hits_m"], 1e-5)
        self.assertEqual(480 * 640 // 16, report["reference_pixels"])

    def test_missing_geometry_cannot_improve_error_without_coverage_penalty(self):
        partial = heldout_view_metrics(plane(0), frames(), ScanSettings(), np.eye(4))
        self.assertLess(partial["depth_rmse_over_hits_m"], 1e-6)
        self.assertLess(partial["hit_fraction"], 0.55)
        self.assertGreater(partial["capped_rmse_including_missing_m"], 0.065)
        empty = heldout_view_metrics(None, frames(), ScanSettings(), np.eye(4))
        self.assertEqual(0, empty["completeness_within_threshold"])
        self.assertIsNone(empty["depth_rmse_over_hits_m"])
        self.assertAlmostEqual(0.1, empty["capped_rmse_including_missing_m"])

    def test_depth_shift_and_missing_reference_samples_are_visible(self):
        values = frames()
        values[0][1][:240] = 0
        wrong = heldout_view_metrics(
            plane().translate([0, 0, 0.03]), values, ScanSettings(), np.eye(4)
        )
        self.assertAlmostEqual(0.03, wrong["depth_rmse_over_hits_m"], places=5)
        self.assertEqual(0, wrong["completeness_within_threshold"])
        self.assertEqual(480 * 640 // 32, wrong["reference_pixels"])
        with self.assertRaises(ValueError):
            heldout_view_metrics(
                plane(), [(None, None, 1, None)], ScanSettings(), np.eye(4)
            )

    def test_disjoint_reader_offsets_preserve_recording_order(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = RecordingWriter(
                Path(folder) / "recording", ScanSettings().to_dict()
            )
            rgb, depth = frames()[0][:2]
            for i in range(6):
                writer.append(rgb, depth, {"timestamp_s": float(i)})
            _, train = load_dataset("recording", writer.path, stride=2)
            _, heldout = load_dataset("recording", writer.path, stride=2, offset=1)
            disjoint_views(train, heldout)
            self.assertEqual([0.0, 2.0, 4.0], [f[2] for f in train])
            self.assertEqual([1.0, 3.0, 5.0], [f[2] for f in heldout])
            with self.assertRaises(ValueError):
                disjoint_views(train, train)

    def test_repeat_summary_retains_failure_and_coverage(self):
        runs = [
            {
                "variant": "uniform",
                "mesh_built": False,
                "accepted": 0,
                "trajectory": {},
                "heldout_views": {
                    "depth_rmse_over_hits_m": None,
                    "hit_fraction": 0.0,
                    "completeness_within_threshold": 0.0,
                    "capped_rmse_including_missing_m": 0.1,
                },
            }
        ]
        summary = aggregate(runs)["uniform"]
        self.assertEqual(0, summary["mesh_successes"])
        self.assertIsNone(summary["depth_rmse_over_hits_m"])
        self.assertEqual(0.1, summary["capped_rmse_including_missing_m"]["median"])

    def test_input_hash_includes_pairing_metadata_and_is_order_stable(self):
        rgb, depth, stamp, pose = frames()[0]
        a = ReplayFrame(
            rgb, depth, stamp, pose, {"rgb_depth_delta_ms": 5, "capture_id": 1}
        )
        b = ReplayFrame(
            rgb, depth, stamp, pose, {"capture_id": 1, "rgb_depth_delta_ms": 5}
        )
        c = ReplayFrame(
            rgb, depth, stamp, pose, {"rgb_depth_delta_ms": 25, "capture_id": 1}
        )
        self.assertEqual(
            input_hash(ScanSettings(), [a], []), input_hash(ScanSettings(), [b], [])
        )
        self.assertNotEqual(
            input_hash(ScanSettings(), [a], []), input_hash(ScanSettings(), [c], [])
        )
