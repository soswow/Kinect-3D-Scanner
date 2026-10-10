"""World marker recovery, separate depth rejection, and accepted-only authority."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scanner_server.apriltag_tracking import observation, register
from scanner_server.engine import ScanEngine
from scanner_server.live_tag_map import LiveTagMap, depth_witness
from shared.apriltag import TagFrame
from shared.settings import ScanSettings
from tests.test_apriltag import FAMILIES, tag_scene


def three_tags(shift=0):
    rgb, depth = tag_scene(shift)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 9, 80)
    rgb[90:170, 290-shift:370-shift] = marker[..., None]
    depth[:] = 2003
    return rgb, depth


class LiveTagMapTests(unittest.TestCase):
    def engine(self):
        engine = ScanEngine(device="cpu")
        self.addCleanup(engine.shutdown)
        engine.reset(settings=ScanSettings(apriltag_tracking=True, apriltag_dictionaries=FAMILIES,
                                          confidence_fusion=False))
        return engine

    def test_world_map_recovers_valid_motion_beyond_ordinary_motion_limit(self):
        engine = self.engine()
        for i, shift in enumerate((0, 100)):
            engine.store_frame(*three_tags(shift), {"timestamp_s": i * 3})
            engine.process_frames()
        self.assertEqual(2, engine.frame_count, engine.diagnostics)
        self.assertEqual("apriltag_map", engine.diagnostics[-1]["visual_evidence"]["kind"])
        self.assertGreater(engine.cumulative_T[0, 3], engine.settings.max_translation_m)
        self.assertAlmostEqual(100 * 2.003 / 525, engine.cumulative_T[0, 3], delta=.003)
        self.assertTrue(engine.frame_metadata[1]["apriltag_live_map"]["depth_witness"]["accepted"])

    def test_visible_known_tags_do_not_require_sixty_percent_whole_scene_overlap(self):
        engine = self.engine()
        rgb, depth = three_tags()
        sparse = np.zeros_like(depth)
        sparse[80:401, 90:551] = depth[80:401, 90:551]
        engine.store_frame(rgb, sparse, {"timestamp_s": 0})
        engine.process_frames()
        engine.store_frame(rgb, depth, {"timestamp_s": 3})
        rgbd = engine._make_rgbd(rgb, depth)
        source = engine._make_reg_pcd(rgbd, normals=False)
        with patch("scanner_server.live_tag_map.register", return_value=None):
            self.assertIsNone(register(engine, source, rgbd))
        engine.process_frames()
        self.assertEqual(2, engine.frame_count, engine.diagnostics)
        self.assertEqual("apriltag_map", engine.diagnostics[-1]["visual_evidence"]["kind"])

    def test_depth_conflict_rejects_map_pose_without_extending_accepted_map(self):
        engine = self.engine()
        engine.store_frame(*three_tags(), {"timestamp_s": 0})
        engine.process_frames()
        rgb, depth = three_tags()
        current_tags = engine._apriltag_detector.detect(rgb, depth, engine.settings.camera)
        mask = np.zeros(depth.shape, np.uint8)
        for tag in current_tags.tags.values():
            cv2.fillConvexPoly(mask, np.rint(tag.pixels).astype(np.int32), 1)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        depth[mask == 0] = 1500
        engine.store_frame(rgb, depth, {"timestamp_s": 3})
        rgbd = engine._make_rgbd(rgb, depth)
        self.assertIsNone(register(engine, engine._make_reg_pcd(rgbd, normals=False), rgbd))
        report = engine.frame_metadata[1]["apriltag_live_map"]
        self.assertEqual("independent_depth_rejected", report["reason"])
        self.assertGreater(report["depth_witness"]["free_space_fraction"], .08)
        self.assertEqual(1, engine._live_tag_map.pose_count)
        self.assertTrue(all(i == 0 for rows in engine._live_tag_map.samples.values() for i, _ in rows))
        self.assertEqual([0], [i for i, _ in engine.poses])

    def test_map_pools_tags_from_different_accepted_reference_views(self):
        engine = self.engine()
        for i in range(4):
            engine.store_frame(*three_tags(), {"timestamp_s": i})
        full = observation(engine, 0)
        keys = sorted(full.tags)
        frames = [TagFrame({k:v for k,v in full.tags.items() if k != missing}) for missing in keys]
        fake = SimpleNamespace(poses=[(i, np.eye(4)) for i in range(3)], _apriltag_repeated=set())
        bank = LiveTagMap()
        bank.sync(fake, lambda e, i: frames[i])
        pose, report = bank.fit(full, engine.settings.camera)
        self.assertIsNotNone(pose, report)
        self.assertEqual(3, report["inlier_tags"])
        self.assertTrue(all(len(frame.tags) == 2 for frame in frames))
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-6)

    def test_map_quarantines_late_duplicate_and_does_not_use_rejected_view(self):
        engine = self.engine()
        engine.store_frame(*three_tags(), {"timestamp_s": 0})
        engine.process_frames()
        engine.store_frame(*three_tags(4), {"timestamp_s": 1})
        rgbd = engine._make_rgbd(*engine._prepare_input(*engine.raw_frames[1], engine.settings))
        register(engine, engine._make_reg_pcd(rgbd, normals=False), rgbd)
        bank = engine._live_tag_map
        self.assertEqual(1, bank.pose_count)
        engine._apriltag_repeated.add((FAMILIES[0], 7))
        bank.sync(engine, observation)
        self.assertNotIn((FAMILIES[0], 7), bank.samples)
        self.assertTrue(all(index == 0 for rows in bank.samples.values() for index, _ in rows))
        pose, report = bank.fit(observation(engine, 1), engine.settings.camera)
        self.assertIsNone(pose)
        self.assertEqual("insufficient_mapped_tags", report["reason"])

    def test_missing_unsynchronized_sparse_and_moving_tags_have_no_map_authority(self):
        engine = self.engine()
        engine.store_frame(*three_tags(), {"timestamp_s": 0})
        engine.process_frames()
        bank = LiveTagMap()
        bank.sync(engine, observation)
        for current in (TagFrame(), engine._apriltag_detector.detect(*three_tags(), engine.settings.camera, synchronized=False)):
            self.assertIsNone(bank.fit(current, engine.settings.camera)[0])
        full = observation(engine, 0)
        self.assertIsNone(bank.fit(TagFrame(dict(list(full.tags.items())[:2])), engine.settings.camera)[0])
        moved = {}
        for i, (key, tag) in enumerate(full.tags.items()):
            from shared.apriltag import Tag
            delta = np.array([i * .1, 0, 0])
            moved[key] = Tag(tag.pixels + [i * 25, 0], tag.points + delta)
        self.assertIsNone(bank.fit(TagFrame(moved), engine.settings.camera)[0])

    def test_depth_witness_excludes_fitted_marker_surfaces(self):
        engine = self.engine()
        rgb, depth = three_tags()
        tags = engine._apriltag_detector.detect(rgb, depth, engine.settings.camera)
        view = depth_witness(depth.astype(float) / 1000, engine.settings.camera, tags)
        self.assertGreater(len(view.heldout.points), 100)
        points = view.heldout.points
        pixels = points[:, :2] / points[:, 2:3] * [engine.settings.camera.fx, engine.settings.camera.fy]
        pixels += [engine.settings.camera.cx, engine.settings.camera.cy]
        for tag in tags.tags.values():
            lo, hi = tag.pixels.min(axis=0), tag.pixels.max(axis=0)
            self.assertFalse(np.any(np.all((pixels >= lo) & (pixels <= hi), axis=1)))
            center = np.rint(tag.pixels.mean(axis=0)).astype(int)
            self.assertEqual(0, view.depth[center[1], center[0]])

    def test_lost_tracking_recovers_against_map_and_fusion_failure_cannot_extend_it(self):
        from scanner_server.cuda_fusion import FusionUpdateError
        engine = self.engine()
        engine.store_frame(*three_tags(), {"timestamp_s": 0})
        engine.process_frames()
        engine._tracking_lost_frames = 2
        engine.store_frame(*three_tags(100), {"timestamp_s": 3})
        with patch.object(engine, "_integrate_vbg", side_effect=FusionUpdateError("injected")):
            with self.assertRaises(FusionUpdateError):
                engine.process_frames()
        self.assertTrue(engine.frame_metadata[1]["apriltag_live_map"]["accepted"])
        self.assertEqual([0], [i for i, _ in engine.poses])
        self.assertEqual(1, engine._live_tag_map.pose_count)
        self.assertIsNotNone(engine.fusion_failure)

    def test_project_reload_rebuilds_world_corners_and_reset_clears_them(self):
        import tempfile
        from pathlib import Path
        from scanner_server.session import export_session, load_session
        source, loaded = self.engine(), self.engine()
        source.store_frame(*three_tags(), {"timestamp_s": 0})
        source.process_frames()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tags.zip"
            export_session(source, path)
            load_session(loaded, path)
        self.assertIsNone(loaded._live_tag_map)
        loaded.store_frame(*three_tags(100), {"timestamp_s": 3})
        loaded.process_frames()
        self.assertEqual(2, loaded.frame_count, loaded.diagnostics)
        self.assertEqual("apriltag_map", loaded.diagnostics[-1]["visual_evidence"]["kind"])
        loaded.reset(settings=source.settings)
        self.assertIsNone(loaded._live_tag_map)
        self.assertFalse(loaded.poses)

    def test_invalid_pair_fits_do_not_hide_an_older_verified_reference(self):
        from shared.apriltag import tag_motion
        engine = self.engine()
        for i in range(6):
            engine.store_frame(*tag_scene(i * 4), {"timestamp_s": i * .1})
            engine.process_frames()
        self.assertEqual(6, engine.frame_count)
        oldest = observation(engine, 0)
        key = sorted(oldest.tags)[0]
        def only_oldest(source, target, camera):
            if np.allclose(target.tags[key].pixels, oldest.tags[key].pixels):
                return tag_motion(source, target, camera)
            return None, {}
        engine.store_frame(*tag_scene(24), {"timestamp_s": .6})
        with patch("scanner_server.apriltag_tracking.tag_motion", side_effect=only_oldest):
            engine.process_frames()
        self.assertEqual(7, engine.frame_count, engine.diagnostics)
        self.assertEqual(0, engine.diagnostics[-1]["visual_evidence"]["target_index"])

    def test_proposal_only_pose_seeds_cannot_initialize_live_world_map(self):
        from scanner_server.live_tag_map import register as register_map
        engine = self.engine()
        engine.store_frame(*three_tags(), {"timestamp_s": 0})
        engine.poses = [(0, np.eye(4))]
        engine._pose_seeds_only = True
        engine.store_frame(*three_tags(4), {"timestamp_s": 1})
        engine._processed_count = 1
        current = observation(engine, 1)
        rgbd = engine._make_rgbd(*engine.raw_frames[1])
        self.assertIsNone(register_map(engine, current, rgbd, observation))
        self.assertIsNone(engine._live_tag_map)
        self.assertEqual("unverified_pose_seeds", engine.frame_metadata[1]["apriltag_live_map"]["reason"])


if __name__ == "__main__":
    unittest.main()
