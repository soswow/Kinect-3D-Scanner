"""Tracking loss must freeze fusion until an observed anchor is verified."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PyQt6.QtWidgets import QApplication

from kinect_scanner.gui.live_view import LiveView
from kinect_scanner.gui.tracking_overview import TrackingOverview
from scanner_server import refinement
from scanner_server.engine import ScanEngine
from tests import test_scanner_workflow as workflow
from tests.test_quality import scene_frames


class RecoveryEngineTests(unittest.TestCase):
    def test_anchor_verification_is_probed_then_sampled_during_a_long_outage(self):
        """A long unrecoverable outage must stop repeating its most expensive check."""
        engine = ScanEngine(device="cpu")
        rgb, depth, _ = scene_frames(1)[0]
        engine.store_frame(rgb, depth, {"timestamp_s": 0})
        engine.process_frames()
        source = engine._make_reg_pcd(engine._make_rgbd(
            *engine._prepare_input(rgb, depth, engine.settings)))

        probed = []
        with patch.object(engine, "_visual_register", return_value=None), \
             patch.object(engine, "_relocalize", return_value=None), \
             patch.object(engine, "_recover_anchor", return_value=None) as anchor:
            anchor.side_effect = lambda *_: probed.append(engine._tracking_lost_frames)
            for lost in range(1, 13):
                engine._tracking_lost_frames = lost
                result, message = engine._register(source)
        self.assertIsNone(result)
        self.assertIn("Tracking lost", message)
        engine._tracking_lost_frames = 0

        frames = engine.ANCHOR_PROBE_FRAMES
        interval = engine.ANCHOR_PROBE_INTERVAL
        expected = [lost for lost in range(1, 13)
                    if lost <= frames or lost % interval == 0]
        self.assertEqual(expected, probed)
        self.assertLess(len(probed), 12, "anchors must be sampled, not re-tried every frame")

    def test_recovery_prepares_each_anchor_cloud_only_once(self):
        """Recovery revisits the same accepted views; a stored frame never changes."""
        engine = ScanEngine(device="cpu")
        frames = scene_frames(6)
        for index, (rgb, depth, _) in enumerate(frames):
            engine.store_frame(rgb, depth, {"timestamp_s": float(index)})
        engine.process_frames()
        self.assertGreaterEqual(len(engine.poses), 3, engine.diagnostics)

        anchor = engine.poses[0][0]
        with patch.object(engine, "_make_reg_pcd", wraps=engine._make_reg_pcd) as prepare:
            first = engine._anchor_cloud(anchor)
            second = engine._anchor_cloud(anchor)
        self.assertIs(first, second)
        self.assertEqual(1, prepare.call_count,
                         "an accepted anchor cloud must be reused, not rebuilt per lost frame")

    def test_a_confident_model_match_after_a_gap_cannot_bypass_raw_view_verification(self):
        engine = ScanEngine(device="cpu")
        rgb, depth, _ = scene_frames(1)[0]
        engine.store_frame(rgb, depth, {"timestamp_s": 0})
        engine.process_frames()
        old_pose = engine.cumulative_T.copy()
        engine.store_frame(rgb, depth, {"timestamp_s": 12})
        wrong = np.eye(4)
        wrong[0, 3] = 0.18
        result = SimpleNamespace(transformation=wrong, fitness=0.99, inlier_rmse=0.004)
        with patch.object(engine, "_register", return_value=(result, "icp")), \
             patch.object(engine, "_integrate_vbg", wraps=engine._integrate_vbg) as fusion:
            engine.process_frames()
        fusion.assert_not_called()
        self.assertFalse(engine.diagnostics[-1]["success"])
        self.assertIn("Unverified tracking transition", engine.diagnostics[-1]["message"])
        np.testing.assert_array_equal(old_pose, engine.cumulative_T)
        self.assertTrue(engine.live_snapshot()["fusion_paused"])

    def test_a_gap_with_verified_geometric_motion_can_continue(self):
        engine = ScanEngine(device="cpu")
        frames = scene_frames(11)
        engine.store_frame(*frames[0][:2], {"timestamp_s": 0})
        engine.process_frames()
        engine.store_frame(*frames[10][:2], {"timestamp_s": 12})
        engine.process_frames()
        self.assertEqual(2, engine.frame_count, engine.diagnostics)
        self.assertLess(np.linalg.norm(engine.cumulative_T[:3, 3] - frames[10][2][:3, 3]), 0.02)

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

    def _verified_transition(self, *, step, correction, cycle):
        """Drive the transition verifier with chosen reciprocal evidence.

        Returns the verdict for a step of `step` metres whose forward match
        corrects the tracked pose by `correction` metres and whose reverse match
        closes the reciprocal loop after `cycle` metres. Both directions are
        declared trustworthy, so the verdict is decided purely by the cycle and
        agreement bounds.
        """
        engine = ScanEngine(device="cpu")
        rgb, depth, _ = scene_frames(1)[0]
        engine.store_frame(rgb, depth, {"timestamp_s": 0})
        engine.process_frames()

        def translate(metres):
            transform = np.eye(4)
            transform[0, 3] = metres
            return transform

        _, last = engine.poses[-1]
        relative = translate(step)
        forward = relative @ translate(correction)
        reverse = np.linalg.inv(forward) @ translate(cycle)
        results = [SimpleNamespace(transformation=forward),
                   SimpleNamespace(transformation=reverse)]

        # A fresh frame makes the timestamp gap alone require verification.
        engine._processed_count = 1
        engine.frame_metadata.append({"timestamp_s": 12})
        source = engine._make_reg_pcd(engine._make_rgbd(
            *engine._prepare_input(rgb, depth, engine.settings)))
        with patch.object(refinement, "_match", side_effect=lambda *_: results.pop(0)), \
             patch.object(refinement, "_trustworthy", return_value=True):
            return engine._verify_tracking_transition(
                source, SimpleNamespace(transformation=last @ relative))

    def test_a_large_correction_after_a_large_step_is_still_unverified(self):
        """Chest 8 frame 6: 43 mm of correction on a 30 cm step is drift.

        Widening the agreement bound to 20% of the step accepted this and the
        twelve frames behind it, but Finish then excluded every one of them and
        produced a bit-identical reconstruction, so the bound stays flat.
        """
        self.assertIsNotNone(
            self._verified_transition(step=0.30, correction=0.043, cycle=0.005))

    def test_reciprocal_closure_must_stay_inside_its_ten_millimetre_corridor(self):
        """13 mm of cycle disagreement means the two directions disagree.

        Widening this to the 3 cm ICP corridor accepted the same unverifiable
        chest 8 chain, so the 10 mm bound stays.
        """
        self.assertIsNotNone(
            self._verified_transition(step=0.30, correction=0.005, cycle=0.013))
        # The reciprocal evidence chest 8 does satisfy is still accepted.
        self.assertIsNone(
            self._verified_transition(step=0.30, correction=0.005, cycle=0.003))

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
                           "color_assistance_requested": True,
                           "result": {"metadata": {"rgb_depth_delta_ms": -43}},
                           "trajectory": [{"index": 3, "camera_to_world": np.eye(4).tolist()}]})
        self.app.processEvents()
        self.assertTrue(view.recovery_label.isVisible())
        self.assertTrue(view.overview.isVisible())
        self.assertTrue(view.color_warning_label.isVisible())
        self.assertIn("43 ms", view.color_warning_label.text())
        self.assertIn("Color-assisted tracking", view.color_warning_label.text())
        self.assertIn("still recorded", view.color_warning_label.toolTip())
        view.set_snapshot({"color_assistance_requested": True,
                           "result": {"metadata": {"rgb_depth_delta_ms": 20}}})
        self.assertFalse(view.color_warning_label.isVisible())
        view.set_snapshot({"fusion_paused": False, "trajectory": []})
        self.assertFalse(view.recovery_label.isVisible())
        self.assertFalse(view.overview.isVisible())
        self.assertFalse(view.color_warning_label.isVisible())
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
