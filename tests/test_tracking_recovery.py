"""Tracking loss must freeze fusion until an observed anchor is verified."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtWidgets import QApplication

from kinect_scanner.gui.live_view import LiveView
from kinect_scanner.gui.tracking_overview import TrackingOverview
from scanner_server.engine import ScanEngine
from tests import test_scanner_workflow as workflow
from tests.test_quality import scene_frames


class RecoveryEngineTests(unittest.TestCase):
    def test_bad_depth_freezes_pose_and_fusion_then_matching_view_recovers(self):
        engine = ScanEngine(device="cpu")
        rgb, depth, _ = scene_frames(1)[0]
        engine.store_frame(rgb, depth)
        engine.process_frames()
        pose = engine.cumulative_T.copy()
        engine.store_frame(rgb, np.zeros_like(depth))
        with patch.object(engine, "_integrate_vbg", wraps=engine._integrate_vbg) as fuse:
            engine.process_frames()
            fuse.assert_not_called()
        snapshot = engine.live_snapshot(max_points=10)
        self.assertEqual("recovering", snapshot["tracking_state"])
        self.assertTrue(snapshot["fusion_paused"])
        self.assertEqual(0, snapshot["last_tracked_index"])
        self.assertEqual(1, snapshot["lost_at_index"])
        self.assertTrue(snapshot["last_tracked_rgb_png"])
        self.assertEqual([0], [p["index"] for p in snapshot["trajectory"]])
        np.testing.assert_array_equal(pose, engine.cumulative_T)
        # A nearby raw camera observation can resume, without seeding a new map.
        engine.store_frame(rgb, depth)
        engine.process_frames()
        self.assertTrue(engine.diagnostics[-1]["success"], engine.diagnostics[-1])
        self.assertEqual("anchor+icp", engine.diagnostics[-1]["method"])
        self.assertEqual(2, engine.frame_count)
        snapshot = engine.live_snapshot(max_points=10)
        self.assertFalse(snapshot["fusion_paused"])
        self.assertIsNone(snapshot["lost_at_index"])
        self.assertIsNone(snapshot["last_tracked_rgb_png"])
        self.assertEqual([0, 2], [p["index"] for p in snapshot["trajectory"]])

    def test_model_icp_cannot_resume_after_failed_anchor_verification(self):
        engine = ScanEngine(device="cpu")
        engine._tracking_lost_frames = 1
        with patch.object(engine, "_recover_anchor", return_value=None), \
                patch.object(engine, "_relocalize", return_value=None), \
                patch.object(engine, "_icp") as icp, \
                patch.object(engine, "_fpfh_fallback") as global_match:
            result, message = engine._register(None)
        self.assertIsNone(result)
        self.assertIn("last good view", message)
        icp.assert_not_called()
        global_match.assert_not_called()

    def test_planar_anchor_cannot_authorize_recovery(self):
        engine = ScanEngine(device="cpu")
        rgb = np.full((480, 640, 3), 128, np.uint8)
        depth = np.full((480, 640), 1000, np.uint16)
        engine._last_rgbd = engine._make_rgbd(rgb, depth)
        engine.poses = [(0, np.eye(4))]
        cloud = engine._make_reg_pcd(engine._last_rgbd)
        self.assertIsNone(engine._recover_anchor(cloud))


class RecoveryViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_map_projects_using_estimated_up_and_never_invents_lost_pose(self):
        view = TrackingOverview()
        pose = np.eye(4)
        up = np.array([0., -0.8, -0.6])
        view.set_snapshot({"world_up": up.tolist(), "fusion_paused": True,
                           "last_tracked_index": 13, "trajectory": [
                               {"index": 13, "camera_to_world": pose.tolist()}]})
        basis = view.projection_basis()
        np.testing.assert_allclose(up @ basis, [0, 0], atol=1e-10)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)
        self.assertEqual([13], [i for i, _ in view.accepted_poses()])
        view.resize(360, 180)
        self.assertFalse(view.grab().isNull())
        view.close()

    def test_large_loss_notice_and_reference_clear_after_recovery(self):
        view = LiveView()
        view.resize(640, 600)
        view.show()
        view.set_snapshot({"fusion_paused": True, "last_tracked_index": 3,
                           "trajectory": [{"index": 3, "camera_to_world": np.eye(4).tolist()}]})
        self.app.processEvents()
        self.assertTrue(view.recovery_label.isVisible())
        self.assertTrue(view.overview.isVisible())
        view.set_snapshot({"fusion_paused": False, "trajectory": []})
        self.assertFalse(view.recovery_label.isVisible())
        self.assertFalse(view.overview.isVisible())
        view.close()


class RecoveryWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    setUp = workflow.ScannerWorkflowTests.setUp
    tearDown = workflow.ScannerWorkflowTests.tearDown
    fresh_frame = workflow.ScannerWorkflowTests.fresh_frame
    retain_scan = workflow.ScannerWorkflowTests.retain_scan
    task_types = workflow.ScannerWorkflowTests.task_types

    def test_recovery_probes_wait_for_pending_check_even_without_adaptive_capture(self):
        self.retain_scan()
        self.window.adaptive_capture_cb.setChecked(False)
        self.window.live_view.snapshot = {"fusion_paused": True, "pending_count": 1}
        self.window._auto_capture_tick()
        self.assertEqual([], self.task_types())
        self.window.live_view.snapshot["pending_count"] = 0
        self.window._auto_capture_tick()
        self.assertEqual(1, len(self.task_types()))
