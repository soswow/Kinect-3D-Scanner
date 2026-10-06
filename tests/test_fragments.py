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
import open3d as o3d

from scanner_server.engine import ScanEngine
from scanner_server.fragments import (
    Fragment,
    _matches,
    _prepare_fragment,
    _verify_bridge,
    _view,
    propose_fragment_poses,
)
from scanner_server.session import export_session
from scripts.reconnect_session import load_pose_seeds, load_session
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
        self.assertIsNone(engine._last_reg_pcd)
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
        self.assertIn("fragments", engine.fragment_reconnection)
        self.assertTrue(engine.fragment_reconnection["failed"])

    def test_accepted_drifted_fragments_are_reestimated_without_ground_truth_inputs(self):
        engine, chosen = self.disconnected()
        drift = np.eye(4)
        drift[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0.65, 0))
        drift[:3, 3] = [0.7, 0, 0.2]
        # Mimic a confident but wrong live reconnection. Reference poses below
        # create the faulty test input and score output; registration reads raw
        # observations and must independently verify all pose guesses.
        engine.poses += [(i, drift @ self.scene[source][2]) for i, source in enumerate(chosen) if i >= 3]
        for i in range(3, 6):
            engine.diagnostics[i].update(success=True, pose=engine.poses[i][1].tolist())
        engine.frame_count = 6
        with patch("scanner_server.fragments.extract_features", side_effect=AssertionError("Unsynced RGB")):
            poses, report = propose_fragment_poses(engine)
        self.assertIsNotNone(poses, report)
        self.assertEqual(0, report["recovered_frames"])
        self.assertGreaterEqual(report["corrected_frames"], 3)
        self.assertEqual([], report["excluded_frames"])
        self.assertTrue(report["verified_bridges"])
        for index, pose in poses:
            truth = self.scene[chosen[index]][2]
            self.assertLess(np.linalg.norm(pose[:3, 3] - truth[:3, 3]), 0.03)
        self.assertGreater(np.linalg.norm(engine.poses[-1][1][:3, 3] - poses[-1][1][:3, 3]), 0.5)

    def test_budget_failure_is_reported_before_fusion_and_does_not_export_old_mesh_as_success(self):
        engine, _ = self.disconnected()
        engine.settings = replace(engine.settings, final_block_count=1)
        volume = engine.vbg
        with patch.object(engine, "_integrate_vbg", side_effect=AssertionError("Fusion before budget check")):
            ok, result = engine.build_mesh()
        self.assertFalse(ok, result)
        self.assertIn("increase the final block budget", result["message"])
        self.assertGreater(engine.fragment_reconnection["fusion_required_blocks"], 1)
        self.assertTrue(engine.fragment_reconnection["verified_bridges"])
        self.assertIs(engine.vbg, volume)
        self.assertEqual(3, engine.frame_count)

    def test_unverified_accepted_fragment_is_excluded_with_original_pose_retained(self):
        engine, chosen = self.disconnected()
        engine.poses += [(i, self.scene[source][2].copy()) for i, source in enumerate(chosen) if i >= 3]
        for i in range(3, 6):
            engine.diagnostics[i].update(success=True, pose=engine.poses[i][1].tolist())
        engine.frame_count = 6
        with patch("scanner_server.fragments._verify_bridge", return_value=None):
            engine._reconnect_volume()
        self.assertTrue(engine.fragment_reconnection["applied"])
        self.assertEqual([3, 4, 5], engine.fragment_reconnection["excluded_frames"])
        self.assertEqual(3, engine.frame_count)
        self.assertEqual(6, len(engine.original_poses))
        for result in engine.diagnostics[3:]:
            self.assertFalse(result["success"])
            self.assertTrue(result["excluded_offline"])
            self.assertIn("pose_before_reconnection", result)

    def test_archived_seeds_require_geometry_revalidation_before_mesh_creation(self):
        engine, _ = self.disconnected()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.zip"
            export_session(engine, path)
            replay = ScanEngine(device="cpu")
            load_session(replay, path)
            load_pose_seeds(replay, path)
            self.assertIsNone(replay.model_pcd)
            self.assertEqual(6, replay._processed_count)
            success, result = replay.build_mesh()
            self.assertTrue(success, result)
            self.assertEqual(6, replay.frame_count)
            self.assertTrue(replay.fragment_reconnection["verified_bridges"])

    def test_slow_regular_capture_is_not_split_into_unconnectable_single_views(self):
        engine = ScanEngine(device="cpu")
        for i, (rgb, depth, _) in enumerate(self.scene[:6]):
            engine.store_frame(rgb, depth, {"timestamp_s": i * 4.0})
        engine.process_frames()
        self.assertEqual(6, engine.frame_count, engine.diagnostics)
        poses, report = propose_fragment_poses(engine)
        self.assertIsNone(poses, report)
        self.assertEqual(1, len(report["fragments"]))
        self.assertEqual([], report["excluded_frames"])
        self.assertEqual(12.0, report["capture_gap_limit_s"])

    def test_size_boundary_preserves_measured_continuity_without_global_matching(self):
        engine = ScanEngine(device="cpu")
        # Shorten only the storage boundary; all registration and validation
        # use noisy, raycast measurements. Only the first world pose is supplied.
        for i, (rgb, depth, _) in enumerate(self.scene[:8]):
            engine.store_frame(rgb, depth, {"timestamp_s": i * 0.2})
        engine.poses = [(0, np.eye(4))]
        with patch("scanner_server.fragments.MAX_FRAGMENT_VIEWS", 4), \
             patch("scanner_server.fragments._global_seed", side_effect=AssertionError("unnecessary global search")):
            poses, report = propose_fragment_poses(engine)
        self.assertEqual(1, report["sequential_bridges"], report)
        self.assertEqual(8, len(poses), report)
        self.assertEqual(0, report["tested_pairs"])
        self.assertEqual([], report["unconnected_fragments"])
        for index, pose in poses:
            truth = self.scene[index][2]
            self.assertLess(np.linalg.norm(pose[:3, 3] - truth[:3, 3]), 0.015)
        self.assertEqual("sequential camera pair", report["verified_bridges"][0]["validation_scope"])
        self.assertEqual([2, 3], report["fragments"][1]["context_frame_indices"])
        self.assertEqual([4, 5, 6, 7], report["fragments"][1]["frame_indices"])
        self.assertEqual(8, len({i for i, _ in poses}), "Context must not duplicate fusion")

    def test_size_boundary_does_not_bridge_a_real_tracking_failure(self):
        engine = ScanEngine(device="cpu")
        engine.reset(settings=ScanSettings(max_translation_m=0.05))
        for i, source in enumerate([0, 1, 2, 10, 11, 12]):
            engine.store_frame(*self.scene[source][:2], {"timestamp_s": i * 0.2})
        engine.poses = [(0, np.eye(4))]
        with patch("scanner_server.fragments.MAX_FRAGMENT_VIEWS", 3), \
             patch("scanner_server.fragments._verify_bridge", return_value=None):
            poses, report = propose_fragment_poses(engine)
        self.assertEqual(0, report["sequential_bridges"], report)
        self.assertEqual([1], report["unconnected_fragments"])
        self.assertEqual([0, 1, 2], [i for i, _ in poses])
        self.assertEqual([], report["fragments"][1]["context_frame_indices"])

    def test_short_boundary_fragment_keeps_witnesses_for_later_reconnection(self):
        engine = ScanEngine(device="cpu")
        chosen = [0, 1, 2, 3, 4, 10, 11, 12]
        for i, source in enumerate(chosen):
            engine.store_frame(*self.scene[source][:2], {"timestamp_s": i * 0.2 + (10 if i >= 5 else 0)})
        engine.poses = [(0, np.eye(4))]

        def boundary_overlap_only(source, target, proposal, *args):
            if (source.index, target.index) != (1, 2):
                return None
            return _verify_bridge(source, target, proposal, *args)

        with patch("scanner_server.fragments.MAX_FRAGMENT_VIEWS", 4), \
             patch("scanner_server.fragments._verify_bridge", side_effect=boundary_overlap_only):
            poses, report = propose_fragment_poses(engine)
        self.assertEqual([4], report["fragments"][1]["frame_indices"])
        self.assertEqual([2, 3], report["fragments"][1]["context_frame_indices"])
        self.assertEqual([], report["unconnected_fragments"], report)
        self.assertEqual(list(range(8)), [i for i, _ in poses])
        for index, pose in poses:
            self.assertLess(np.linalg.norm(pose[:3, 3] - self.scene[chosen[index]][2][:3, 3]), 0.02)

    def test_corrupt_optimization_restores_revalidated_measured_boundary(self):
        engine = ScanEngine(device="cpu")
        for i, (rgb, depth, _) in enumerate(self.scene[:8]):
            engine.store_frame(rgb, depth, {"timestamp_s": i * 0.2})
        engine.poses = [(0, np.eye(4))]

        def corrupt(graph, *args):
            pose = graph.nodes[1].pose.copy()
            pose[0, 3] += 0.2
            graph.nodes[1].pose = pose

        with patch("scanner_server.fragments.MAX_FRAGMENT_VIEWS", 4), \
             patch("scanner_server.fragments.REG.global_optimization", side_effect=corrupt):
            poses, report = propose_fragment_poses(engine)
        self.assertEqual(8, len(poses), report)
        self.assertEqual([[0, 1]], report["rejected_optimized_bridges"])
        self.assertIn("Revalidated", report["optimization_fallback"])
        for index, pose in poses:
            self.assertLess(np.linalg.norm(pose[:3, 3] - self.scene[index][2][:3, 3]), 0.015)

    def test_fallback_cannot_restore_an_unverified_bridge(self):
        engine = ScanEngine(device="cpu")
        for i, (rgb, depth, _) in enumerate(self.scene[:8]):
            engine.store_frame(rgb, depth, {"timestamp_s": i * 0.2})
        engine.poses = [(0, np.eye(4))]
        from scanner_server.fragments import _heldout

        changed = False

        def corrupt(graph, *args):
            nonlocal changed
            changed = True

        def heldout(*args, **kwargs):
            return (False, {}) if changed else _heldout(*args, **kwargs)

        with patch("scanner_server.fragments.MAX_FRAGMENT_VIEWS", 4), \
             patch("scanner_server.fragments.REG.global_optimization", side_effect=corrupt), \
             patch("scanner_server.fragments._heldout", side_effect=heldout):
            poses, report = propose_fragment_poses(engine)
        self.assertEqual([0, 1, 2, 3], [i for i, _ in poses], report)
        self.assertEqual([1], report["unconnected_fragments"])
        self.assertFalse(report["verified_bridges"][0]["connected_to_scan"])

    def test_search_does_not_spend_budget_on_unreachable_fragment_pairs(self):
        engine = ScanEngine(device="cpu")
        for i, source in enumerate([0, 1, 2, 5, 6, 7, 10, 11, 12]):
            engine.store_frame(*self.scene[source][:2], {"timestamp_s": i * 0.2 + (i // 3) * 10})
        engine.poses = [(0, np.eye(4))]
        with patch("scanner_server.fragments._global_seed", return_value=None), \
             patch("scanner_server.fragments._verify_bridge", return_value=None) as verify:
            poses, report = propose_fragment_poses(engine)
        self.assertEqual([0, 1, 2], [i for i, _ in poses])
        self.assertEqual(3, report["candidate_pairs"])
        self.assertEqual(2, report["tested_pairs"])
        self.assertEqual(1, report["unreachable_candidate_pairs"])
        self.assertTrue(all(call.args[0].index == 0 for call in verify.call_args_list))

    def test_search_expands_through_newly_connected_fragments(self):
        engine = ScanEngine(device="cpu")
        for i, source in enumerate([0, 1, 2, 5, 6, 7, 10, 11, 12]):
            engine.store_frame(*self.scene[source][:2], {"timestamp_s": i * 0.2 + (i // 3) * 10})
        engine.poses = [(0, np.eye(4))]
        original = _verify_bridge

        def chain_only(source, target, proposal, camera=None):
            if target.index != source.index + 1:
                return None
            return original(source, target, proposal, camera)

        with patch("scanner_server.fragments._verify_bridge", side_effect=chain_only):
            poses, report = propose_fragment_poses(engine)
        self.assertEqual(9, len(poses), report)
        self.assertEqual([], report["unconnected_fragments"])
        self.assertTrue(any(e["source"] == 1 and e["target"] == 2 for e in report["verified_bridges"]))

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

        def verified(source, target, pose, camera=None):
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

    def test_matching_cache_preserves_mutual_match_order_and_invalidates_new_features(self):
        from scanner_server.appearance import correspondences

        engine = ScanEngine(device="cpu")
        for frame in self.scene[:2]:
            engine.store_frame(*frame[:2])
        a, b = _view(engine, 0), _view(engine, 1)
        expected = correspondences(a.features, b.features)
        reverse = correspondences(b.features, a.features)
        self.assertGreater(len(expected), 40)
        with patch("scanner_server.fragments.correspondences", wraps=correspondences) as matcher:
            np.testing.assert_array_equal(expected, _matches(a, b))
            np.testing.assert_array_equal(reverse, _matches(b, a))
            np.testing.assert_array_equal(expected, _matches(replace(a, pose=np.eye(4)), b))
            self.assertEqual(1, matcher.call_count)
            fresh = _view(engine, 1)
            np.testing.assert_array_equal(expected, _matches(a, fresh))
            self.assertEqual(2, matcher.call_count)
            # Dataclass copies share a context cache. Replacing either side's
            # feature object must invalidate it even if the frame ID is reused.
            empty = replace(a.features, descriptors=None)
            replaced_source = replace(a, features=empty)
            self.assertEqual(0, len(_matches(replaced_source, fresh)))
            self.assertEqual(3, matcher.call_count)
            np.testing.assert_array_equal(expected, _matches(a, fresh))
            self.assertEqual(4, matcher.call_count)
            with patch("scanner_server.fragments.MAX_MATCH_CACHE", 2):
                for index in range(10, 14):
                    np.testing.assert_array_equal(expected, _matches(a, replace(b, index=index, match_cache={})))
                    self.assertLessEqual(len(a.match_cache), 2)

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
