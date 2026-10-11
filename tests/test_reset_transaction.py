"""Failed reset preparation must leave the active scan fully usable."""

import copy
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scanner_server.engine import ScanEngine
from shared.settings import ScanSettings


class ResetTransactionTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "KINECT_NATIVE": "off", "KINECT_MODEL_REFRESH": "eager",
            "KINECT_KEYFRAME_CACHE": "off", "KINECT_LIVE_RECOVERY": "full",
            "KINECT_VISUAL_REFINEMENT": "icp", "KINECT_FINAL_VISUAL_FIRST": "off",
            "KINECT_FINAL_LOCAL_REFINEMENT": "icp",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        with patch.object(ScanEngine, "BLOCK_COUNT", 100):
            self.engine = ScanEngine(device="cpu")
        self.engine.BLOCK_COUNT = 100
        self.addCleanup(self.engine.shutdown)
        self.engine.store_frame(
            np.zeros((480, 640, 3), np.uint8),
            np.full((480, 640), 1000, np.uint16),
            {"frame_id": "retained", "timestamp_s": 1},
        )
        self.engine.poses = [(0, np.eye(4))]
        self.engine.diagnostics = [{"index": 0, "success": True}]
        self.engine._processed_count = self.engine.frame_count = 1
        self.engine.mesh = object()
        self.engine._final_vbg = object()
        self.engine._apriltag_observations = {0: "retained"}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.archive = Path(self.directory.name) / "opened.zip"
        self.archive.write_bytes(b"retained project source")
        self.engine._project_archive_path = str(self.archive)
        self.previous = self.engine.__dict__.copy()
        self.backend = copy.deepcopy(self.engine.backend)

    def assert_preserved(self):
        self.assertEqual(self.previous.keys(), self.engine.__dict__.keys())
        for name, value in self.previous.items():
            self.assertIs(value, self.engine.__dict__[name], name)
        self.assertEqual(self.backend, self.engine.backend)
        self.assertEqual(b"retained project source", self.archive.read_bytes())

    def test_volume_allocation_failure_preserves_every_active_field(self):
        with patch.object(ScanEngine, "_create_vbg", side_effect=RuntimeError("allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                self.engine.reset(settings=replace(self.engine.settings, voxel_m=.02))
        self.assert_preserved()

    def test_late_invalid_environment_policy_preserves_every_active_field(self):
        with patch.dict(os.environ, {"KINECT_FINAL_LOCAL_REFINEMENT": "invalid"}):
            with self.assertRaisesRegex(ValueError, "KINECT_FINAL_LOCAL_REFINEMENT"):
                self.engine.reset(settings=replace(self.engine.settings, voxel_m=.02))
        self.assert_preserved()

    def test_detector_initialization_failure_preserves_every_active_field(self):
        with patch("shared.apriltag.AprilTagDetector", side_effect=RuntimeError("detector failed")):
            with self.assertRaisesRegex(RuntimeError, "detector failed"):
                self.engine.reset(settings=replace(self.engine.settings, apriltag_tracking=True))
        self.assert_preserved()

    def test_archive_cleanup_failure_preserves_every_active_field(self):
        with patch("scanner_server.engine.os.unlink", side_effect=PermissionError("archive in use")):
            with self.assertRaisesRegex(PermissionError, "archive in use"):
                self.engine.reset(settings=replace(self.engine.settings, voxel_m=.02))
        self.assert_preserved()

    def test_success_commits_complete_new_state_and_discards_old_archive(self):
        settings = replace(self.engine.settings, voxel_m=.02)
        self.engine.reset(settings=settings)
        self.assertIs(settings, self.engine.settings)
        self.assertNotEqual(self.previous["session_id"], self.engine.session_id)
        self.assertIsNot(self.previous["vbg"], self.engine.vbg)
        self.assertFalse(self.archive.exists())
        self.assertIsNone(self.engine._project_archive_path)
        self.assertEqual(0, self.engine.stored_count)
        self.assertEqual([], self.engine.poses)
        self.assertEqual([], self.engine.diagnostics)
        self.assertIsNone(self.engine.mesh)
        self.assertIsNone(self.engine._final_vbg)
        self.assertEqual({}, self.engine._apriltag_observations)
        self.assertEqual(.02, self.engine.voxel_size)


if __name__ == "__main__":
    unittest.main()
