"""Mixed-family detection and measured tag authority through camera/server paths."""

import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import cv2
import numpy as np

from shared.apriltag import AprilTagDetector, Tag, TagFrame, tag_agreement, tag_motion
from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker

FAMILIES = ("DICT_APRILTAG_16h5", "DICT_APRILTAG_36h11")


def tag_scene(shift=0, families=FAMILIES, ids=(7, 7)):
    rgb = np.full((480, 640, 3), 255, np.uint8)
    for name, identity, (x, y) in zip(families, ids, ((110, 100), (430, 280))):
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        marker = cv2.aruco.generateImageMarker(dictionary, identity, 100)
        rgb[y:y+100, x-shift:x-shift+100] = marker[..., None]
    return rgb, np.full((480, 640), 2000, np.uint16)


class AprilTagTests(unittest.TestCase):
    def setUp(self):
        self.settings = ScanSettings(apriltag_tracking=True, apriltag_dictionaries=FAMILIES)
        self.detector = AprilTagDetector(FAMILIES)

    def test_settings_roundtrip_defaults_and_validation(self):
        self.assertFalse(ScanSettings.from_dict({}).apriltag_tracking)
        loaded = ScanSettings.from_dict(json.loads(json.dumps(self.settings.to_dict())))
        self.assertEqual(self.settings, loaded)
        for changes in ({"apriltag_tracking": 1}, {"apriltag_dictionaries": "DICT_APRILTAG_16h5"},
                        {"apriltag_dictionaries": ["unknown"]}, {"apriltag_dictionaries": None},
                        {"apriltag_dictionaries": [FAMILIES[0]] * 2},
                        {"apriltag_tracking": True, "apriltag_dictionaries": []}):
            with self.assertRaises(ValueError):
                ScanSettings(**changes)

    def test_real_opencv_detects_both_families_with_same_id(self):
        frame = self.detector.detect(*tag_scene(), self.settings.camera)
        self.assertEqual(2, frame.detected)
        self.assertEqual({(name, 7) for name in FAMILIES}, set(frame.tags))
        self.assertEqual(0, frame.ambiguous)

    def test_real_opencv_detects_every_supported_family(self):
        from shared.settings import APRILTAG_DICTIONARIES
        for family in APRILTAG_DICTIONARIES:
            with self.subTest(family=family):
                frame = AprilTagDetector((family,)).detect(*tag_scene(families=(family,), ids=(7,)), self.settings.camera)
                self.assertEqual({(family, 7)}, set(frame.tags))

    def test_missing_module_has_actionable_error_only_when_enabled(self):
        with patch.object(cv2, "aruco", None):
            VisualTracker(ScanSettings())
            with self.assertRaisesRegex(ValueError, "OpenCV 4.8"):
                VisualTracker(self.settings)

    def test_duplicates_within_family_and_cross_family_decodes_are_rejected(self):
        duplicate = self.detector.detect(*tag_scene(families=(FAMILIES[0],) * 2), self.settings.camera)
        self.assertEqual(2, duplicate.ambiguous)
        self.assertFalse(duplicate.tags)
        pixels = np.array([[[110, 100], [210, 100], [210, 200], [110, 200]]], np.float32)
        detector = unittest.mock.Mock()
        detector.detectMarkers.return_value = ([pixels], np.array([[7]]), [])
        self.detector.detectors = [(name, detector) for name in FAMILIES]
        ambiguous = self.detector.detect(*tag_scene(), self.settings.camera)
        self.assertFalse(ambiguous.tags)
        self.assertEqual(2, ambiguous.ambiguous)

    def test_depth_and_timing_are_mandatory_and_rgb_only_still_detects(self):
        rgb, depth = tag_scene()
        missing = self.detector.detect(rgb, np.zeros_like(depth), self.settings.camera)
        self.assertEqual(2, missing.detected)
        self.assertFalse(missing.tags)
        late = self.detector.detect(rgb, depth, self.settings.camera, synchronized=False)
        self.assertEqual(2, late.detected)
        self.assertFalse(late.tags)

    def test_motion_uses_metric_depth_and_preserves_dictionary_identity(self):
        first = self.detector.detect(*tag_scene(), self.settings.camera)
        second = self.detector.detect(*tag_scene(shift=6), self.settings.camera)
        pose, stats = tag_motion(second, first, self.settings.camera)
        self.assertEqual(2, stats["inlier_tags"])
        self.assertAlmostEqual(6 * 2 / 525, pose[0, 3], delta=.001)
        self.assertTrue(tag_agreement(second, first, pose, self.settings.camera)[0])
        wrong = self.detector.detect(tag_scene()[0], tag_scene()[1] + 400, self.settings.camera)
        self.assertIsNone(tag_motion(wrong, first, self.settings.camera)[0])
        different = TagFrame({("DICT_APRILTAG_25h9", 7): next(iter(first.tags.values()))})
        self.assertIsNone(tag_motion(different, first, self.settings.camera)[0])
        self.assertFalse(tag_agreement(second, first, np.eye(4), self.settings.camera)[0])

    def test_moving_tag_groups_cannot_supply_consistent_scene_motion(self):
        first = self.detector.detect(*tag_scene(), self.settings.camera)
        second = self.detector.detect(*tag_scene(shift=6), self.settings.camera)
        key = (FAMILIES[1], 7)
        tag = second.tags[key]
        second.tags[key] = Tag(tag.pixels + [40, 0], tag.points + [40 * 2 / 525, 0, 0])
        self.assertIsNone(tag_motion(second, first, self.settings.camera)[0])

    def test_single_tag_recovers_six_axis_motion_from_measured_corners(self):
        camera = self.settings.camera
        points = np.array([[-.2, -.2, 2], [.2, -.2, 2], [.2, .2, 2], [-.2, .2, 2]])
        truth = np.eye(4)
        truth[:3, :3] = cv2.Rodrigues(np.array([.02, -.04, .015]))[0]
        truth[:3, 3] = [.03, -.015, .02]
        moved = points @ truth[:3, :3].T + truth[:3, 3]
        def pixels(p):
            return p[:, :2] / p[:, 2:3] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        key = (FAMILIES[0], 7)
        source, target = TagFrame({key: Tag(pixels(points), points)}), TagFrame({key: Tag(pixels(moved), moved)})
        pose, stats = tag_motion(source, target, camera)
        self.assertEqual(1, stats["inlier_tags"])
        np.testing.assert_allclose(pose, truth, atol=1e-6)

    def test_missing_tags_keep_ordinary_color_tracking_available(self):
        from tests.test_visual_tracking import textured_plane
        tracker = VisualTracker(ScanSettings(color_recovery=True, apriltag_tracking=True))
        first = tracker.update(*textured_plane(), {"timestamp_s": 0})
        second = tracker.update(*textured_plane(shift=4), {"timestamp_s": .1})
        self.assertTrue(first["valid"])
        self.assertTrue(second["valid"])
        self.assertEqual(0, second["apriltags"]["detected"])
        self.assertNotIn("apriltag_support", second)

    def test_disabled_tags_do_not_run_opencv_detection(self):
        with patch.object(AprilTagDetector, "detect", side_effect=AssertionError("unexpected detection")):
            tracker = VisualTracker(ScanSettings(color_recovery=True))
            tracker.update(*tag_scene(), {"timestamp_s": 0})

    def test_saved_session_keeps_tag_settings_and_recomputes_evidence(self):
        import tempfile
        from pathlib import Path
        from scanner_server.engine import ScanEngine
        from scanner_server.session import export_session, load_session
        from scanner_server.apriltag_tracking import observation
        source, loaded = ScanEngine(device="cpu"), ScanEngine(device="cpu")
        self.addCleanup(source.shutdown)
        self.addCleanup(loaded.shutdown)
        source.reset(settings=self.settings)
        source.store_frame(*tag_scene(), {"timestamp_s": 0})
        observation(source, 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tags.zip"
            export_session(source, path)
            load_session(loaded, path)
        self.assertEqual(self.settings, loaded.settings)
        self.assertFalse(loaded._apriltag_observations)
        self.assertEqual(2, len(observation(loaded, 0).tags))

    def test_camera_tracks_tags_when_ordinary_features_are_insufficient(self):
        tracker = VisualTracker(self.settings)
        with patch.object(cv2, "goodFeaturesToTrack", return_value=None):
            first = tracker.update(*tag_scene(), {"timestamp_s": 0})
            second = tracker.update(*tag_scene(shift=6), {"timestamp_s": .1})
            self.assertTrue(first["valid"])
            self.assertTrue(second["valid"], second)
            self.assertEqual(first["segment"], second["segment"])
            self.assertEqual(2, second["apriltag_support"]["inlier_tags"])
            self.assertAlmostEqual(6 * 2 / 525, tracker.pose[0, 3], delta=.001)
            late = tracker.update(*tag_scene(shift=7), {"timestamp_s": .2, "rgb_depth_delta_ms": 21})
            self.assertFalse(late["valid"])
            self.assertEqual(2, late["apriltags"]["detected"])
            recovered = tracker.update(*tag_scene(shift=8), {"timestamp_s": .3})
            self.assertTrue(recovered["valid"])
            gap = tracker.update(*tag_scene(shift=10), {"timestamp_s": 2})
            self.assertFalse(gap["valid"])
            self.assertNotEqual(first["segment"], gap["segment"])

    def test_server_recomputes_tags_and_uses_them_for_planar_tracking(self):
        from scanner_server.engine import ScanEngine
        engine = ScanEngine(device="cpu")
        engine.reset(settings=self.settings)
        for index, shift in enumerate((0, 6)):
            engine.store_frame(*tag_scene(shift), {"timestamp_s": index * .1,
                "apriltags_backend": {"detected": 999}, "visual_tracking": {"valid": False}})
        engine.process_frames()
        self.assertEqual(2, engine.frame_count, engine.diagnostics)
        self.assertEqual("apriltag", engine.diagnostics[-1]["visual_evidence"]["kind"])
        self.assertAlmostEqual(6 * 2 / 525, engine.cumulative_T[0, 3], delta=.002)
        self.assertEqual(2, engine.frame_metadata[0]["apriltags_backend"]["detected"])
        self.assertEqual(2, len(engine._apriltag_observations))

    def test_finish_fragment_and_depth_modes_retain_tag_constraint(self):
        from scanner_server.engine import ScanEngine
        from scanner_server.fragments import _view, _local_match, _visual_witness
        from scanner_server.depth_graph import propose_depth_poses
        engine = ScanEngine(device="cpu")
        engine.reset(settings=self.settings)
        for index, shift in enumerate((0, 6)):
            engine.store_frame(*tag_scene(shift), {"timestamp_s": index * .1})
        a, b = _view(engine, 1), _view(engine, 0)
        relative = _local_match(a, b, self.settings.camera, self.settings)
        self.assertIsNotNone(relative)
        self.assertTrue(_visual_witness(a, b, relative, self.settings.camera)[0])
        poses, report = propose_depth_poses(engine)
        self.assertEqual([0, 1], [i for i, _ in poses], report)
        self.assertAlmostEqual(6 * 2 / 525, poses[-1][1][0, 3], delta=.002)
        self.assertTrue(any(e["method"] == "apriltag" for e in report["pairs"]), report)
        self.assertTrue(any(e.get("apriltag_constraint") for e in report["verified_bridges"]), report)


if __name__ == "__main__":
    unittest.main()
