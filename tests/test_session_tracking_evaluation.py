"""Recorded benchmark comparisons must preserve inputs and expose disagreements."""

import io
import os
import sys
import tempfile
from pathlib import Path
import unittest
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_session_tracking import compare_camera, compare_server, frames, validate_output


class RecordedTrackingEvaluationTests(unittest.TestCase):
    def test_report_destination_cannot_overwrite_any_archive_or_hardlink(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.zip"
            source.write_bytes(b"original")
            alias = Path(folder) / "report.json"
            os.link(source, alias)
            for destination in (source, alias):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    validate_output(destination, [source])
            validate_output(Path(folder) / "separate.json", [source])
            self.assertEqual(b"original", source.read_bytes())

    def test_zip_reader_preserves_raw_depth_and_timing_without_truth_seeds(self):
        rgb = np.full((3, 4, 3), 73, np.uint8)
        depth = np.array([[0, 600, 1024, 2047]] * 3, np.uint16)
        data = io.BytesIO()
        entry = {"rgb": "rgb.png", "depth": "depth.png", "timestamp_s": 12.5,
                 "metadata": {"rgb_depth_delta_ms": 31, "captured_monotonic_s": 42,
                              "reference_pose": np.eye(4).tolist()}}
        with zipfile.ZipFile(data, "w") as archive:
            for name, array in (("rgb.png", rgb), ("depth.png", depth)):
                encoded = io.BytesIO()
                Image.fromarray(array).save(encoded, format="PNG")
                archive.writestr(name, encoded.getvalue())
        with zipfile.ZipFile(data) as archive:
            loaded_rgb, loaded_depth, metadata = next(frames(archive, [entry]))
        np.testing.assert_array_equal(rgb, loaded_rgb)
        np.testing.assert_array_equal(depth, loaded_depth)
        self.assertEqual(np.uint16, loaded_depth.dtype)
        self.assertEqual(31, metadata["rgb_depth_delta_ms"])
        self.assertEqual(42, metadata["captured_monotonic_s"])
        self.assertEqual(12.5, metadata["timestamp_s"])
        self.assertNotIn("reference_pose", metadata)
        self.assertIn("reference_pose", entry["metadata"])

    def test_camera_origin_is_not_measured_motion_and_pose_changes_are_exposed(self):
        origin = {"valid": True, "steps": 0, "camera_to_local": np.eye(4).tolist()}
        motion = {"valid": True, "steps": 1, "camera_to_local": np.eye(4).tolist(), "inliers": 50}
        changed = {**motion, "camera_to_local": np.eye(4).tolist(), "inliers": 49}
        changed["camera_to_local"][0][3] = 0.01
        result = compare_camera({"reports": [origin, motion]}, {"reports": [origin, changed]})
        self.assertEqual([1], result["support_disagreement_indices"])
        self.assertEqual(1, result["measured_motion_reports_after"])
        self.assertAlmostEqual(0.01, result["max_pose_matrix_difference"])

    def test_server_comparison_retains_frame_identity_and_numerical_tolerance(self):
        a = {"poses": [(0, np.eye(4)), (2, np.eye(4))],
             "diagnostics": [{"success": True, "fitness": 0.9}, {"success": False},
                             {"success": True, "method": "icp"}]}
        b = {"poses": [(0, np.eye(4))],
             "diagnostics": [{"success": True, "fitness": 0.9 + 1e-13}, {"success": False},
                             {"success": False}]}
        result = compare_server(a, b)
        self.assertEqual([0, 2], result["accepted_before"])
        self.assertEqual([0], result["accepted_after"])
        self.assertEqual([2], result["diagnostic_disagreement_indices"])

    def test_pose_comparison_reports_physical_translation_and_rotation(self):
        before = np.eye(4)
        after = np.eye(4)
        angle = np.deg2rad(2)
        after[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        after[0, 3] = 0.003
        result = compare_server({"poses": [(0, before)], "diagnostics": []},
                                {"poses": [(0, after)], "diagnostics": []})
        self.assertAlmostEqual(0.003, result["max_common_pose_translation_difference_m"])
        self.assertAlmostEqual(2, result["max_common_pose_rotation_difference_deg"])


if __name__ == "__main__":
    unittest.main()
