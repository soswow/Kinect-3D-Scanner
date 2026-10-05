"""Measured disconnected scans must be verified before they enter the TSDF."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scanner_server.engine import ScanEngine
from scanner_server.fragments import (
    Fragment,
    _prepare_fragment,
    _verify_bridge,
    _view,
    propose_fragment_poses,
)
from scanner_server.session import export_session
from scripts.reconnect_session import load_session
from shared.settings import ScanSettings
from tests.test_quality import scene_frames


class FragmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = scene_frames(14)

    def disconnected(self):
        engine = ScanEngine(device="cpu")
        engine.reset(settings=ScanSettings(reconnect_fragments=True, final_weight=0.5, max_translation_m=0.05))
        chosen = [0, 1, 2, 10, 11, 12]
        for index, source in enumerate(chosen):
            rgb, depth, _ = self.scene[source]
            engine.store_frame(rgb, depth, {"timestamp_s": index if index < 3 else index + 10,
                                           "rgb_depth_delta_ms": 30})
        engine.process_frames()
        return engine, chosen

    def test_reconnects_disconnected_asymmetric_geometry_without_reference_poses(self):
        engine, chosen = self.disconnected()
        self.assertEqual(3, engine.frame_count, engine.diagnostics)
        original = [(i, p.copy()) for i, p in engine.poses]
        with patch("scanner_server.fragments.extract_features", side_effect=AssertionError("Unsynced RGB")):
            poses, report = propose_fragment_poses(engine)
        self.assertIsNotNone(poses, report)
        self.assertEqual(3, report["recovered_frames"])
        self.assertTrue(report["verified_bridges"])
        self.assertEqual([], report["unconnected_fragments"])
        for index, pose in poses:
            truth = self.scene[chosen[index]][2]
            self.assertLess(np.linalg.norm(pose[:3, 3] - truth[:3, 3]), 0.03)
            self.assertLess(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1))), 3)
        for (index, pose), (old_index, old_pose) in zip(engine.poses, original):
            self.assertEqual(index, old_index)
            np.testing.assert_array_equal(pose, old_pose)
        success, result = engine.build_mesh()
        self.assertTrue(success, result)
        self.assertEqual(6, result["frame_count"])
        self.assertEqual(0, result["skipped_count"])
        self.assertTrue(result["fragment_reconnection"]["applied"])
        self.assertTrue(engine.diagnostics[-1]["recovered_offline"])
        self.assertFalse(engine.live_snapshot(max_points=10)["fusion_paused"])
        with patch("scanner_server.fragments.propose_fragment_poses", side_effect=AssertionError("repeated search")):
            self.assertTrue(engine.build_mesh()[0])
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "session.zip"
            export_session(engine, destination)
            with zipfile.ZipFile(destination) as archive:
                saved = json.loads(archive.read("reconstruction.json"))
                self.assertEqual(3, saved["fragment_reconnection"]["recovered_frames"])
                self.assertEqual(3, len(saved["original_poses"]))
                self.assertTrue(saved["frames"][-1]["recovered_offline"])
            replay = ScanEngine(device="cpu")
            load_session(replay, destination)
            self.assertEqual(6, replay.stored_count)
            self.assertEqual([], replay.poses)  # Exported estimates are never replay authority.
            self.assertTrue(replay.settings.reconnect_fragments)

    def test_native_fusion_failure_preserves_poses_volume_and_rejected_frames(self):
        engine, _ = self.disconnected()
        volume = engine.vbg
        original = [(i, p.copy()) for i, p in engine.poses]
        with patch.object(engine, "_create_vbg", side_effect=RuntimeError("allocation failed")):
            engine._reconnect_volume()
        self.assertFalse(engine.fragment_reconnection["applied"])
        self.assertIs(engine.vbg, volume)
        self.assertEqual(3, engine.frame_count)
        self.assertFalse(engine.diagnostics[-1]["success"])
        for (_, pose), (_, old_pose) in zip(engine.poses, original):
            np.testing.assert_array_equal(pose, old_pose)
        self.assertIsNone(engine._reconnection_count)

    def test_pruned_bridge_does_not_authorize_disconnected_fusion(self):
        engine, _ = self.disconnected()
        with patch("scanner_server.fragments.REG.global_optimization", side_effect=lambda graph, *args: graph.edges.clear()) as optimizer:
            poses, report = propose_fragment_poses(engine)
        optimizer.assert_called_once()
        self.assertIsNone(poses)
        self.assertTrue(report["verified_bridges"])
        self.assertEqual([1], report["unconnected_fragments"])
        self.assertEqual(3, engine.frame_count)

    def test_conflicting_bridge_proposals_remain_separate(self):
        engine, _ = self.disconnected()

        def proposal(source, target, seed):
            pose = np.eye(4)
            pose[0, 3] = 0.2 if seed < 10000 else 0.4
            return pose

        def verified(source, target, pose):
            return {"source": source.index, "target": target.index, "transform": pose,
                    "information": np.eye(6), "support": [(0, 3), (1, 4)], "validation": {}}

        with patch("scanner_server.fragments._global_seed", side_effect=proposal), \
             patch("scanner_server.fragments._verify_bridge", side_effect=verified):
            poses, report = propose_fragment_poses(engine)
        self.assertIsNone(poses)
        self.assertEqual([[0, 1]], report["ambiguous_pairs"])
        self.assertEqual([], report["verified_bridges"])

    def test_search_budget_reports_retained_views_and_new_frame_invalidates_cache(self):
        engine, _ = self.disconnected()
        with patch("scanner_server.fragments.MAX_FRAGMENTS", 1):
            engine._reconnect_volume()
        self.assertTrue(engine.fragment_reconnection["budget_limited"])
        self.assertEqual([3, 4, 5], engine.fragment_reconnection["unassigned_indices"])
        self.assertEqual(6, engine._reconnection_count)
        engine.store_frame(*self.scene[12][:2], {"timestamp_s": 17})
        self.assertIsNone(engine._reconnection_count)
        engine.reset()
        self.assertEqual("Not requested", engine.fragment_reconnection["reason"])

    def test_planes_cannot_connect_and_reports_local_fragments(self):
        engine = ScanEngine(device="cpu")
        engine.reset(settings=replace(engine.settings, reconnect_fragments=True))
        rgb = np.full((480, 640, 3), 100, np.uint8)
        depth = np.full((480, 640), 1200, np.uint16)
        for i in range(5):
            engine.store_frame(rgb, depth, {"timestamp_s": i})
        engine.process_frames()
        poses, report = propose_fragment_poses(engine)
        self.assertIsNone(poses)
        self.assertEqual(0, report["recovered_frames"])
        self.assertTrue(report["unconnected_fragments"])
        self.assertTrue(all(f["fragment_to_world"] is None for f in report["fragments"] if not f["connected"]))
        self.assertTrue(all(f["camera_to_fragment"] for f in report["fragments"]))

    def test_repeated_stationary_captures_are_not_independent_bridge_evidence(self):
        engine = ScanEngine(device="cpu")
        for _ in range(4):
            engine.store_frame(*self.scene[0][:2])
        source = Fragment(0, views=[_view(engine, 0), _view(engine, 1)])
        target = Fragment(1, views=[_view(engine, 2), _view(engine, 3)])
        _prepare_fragment(source)
        _prepare_fragment(target)
        self.assertIsNone(_verify_bridge(source, target, np.eye(4)))

    def test_connected_scan_skips_fragment_search_and_invalid_setting_rejected(self):
        engine = ScanEngine(device="cpu")
        rgb, depth, _ = self.scene[0]
        engine.store_frame(rgb, depth)
        engine.process_frames()
        with patch("scanner_server.fragments._view", side_effect=AssertionError("unneeded search")):
            poses, report = propose_fragment_poses(engine)
        self.assertIsNone(poses)
        self.assertIn("No skipped", report["reason"])
        with self.assertRaises(ValueError):
            ScanSettings(reconnect_fragments="yes")
        self.assertTrue(ScanSettings.from_dict(ScanSettings(reconnect_fragments=True).to_dict()).reconnect_fragments)
