"""Actual joint camera/landmark recovery and conservative RGB-D proposal gates."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scanner_server.appearance import Features
from scanner_server import bundle_adjustment as bundle
from scanner_server.bundle_adjustment import (
    VALIDATION_CACHE_VIEWS, _DepthClouds, _heldout_points, _output_poses, _timely, build_tracks, make_problem,
    propose_bundle_poses, solve_bundle, validate_depth,
)
from shared.settings import CameraCalibration, ScanSettings


def bundle_fixture(count=6, points=100, *, corrupt=False):
    rng = np.random.default_rng(316)
    camera = CameraCalibration()
    world = rng.uniform([-0.55, -0.4, 1.05], [0.55, 0.4, 2.2], (points, 3))
    descriptors = rng.normal(size=(points, 128)).astype(np.float32)
    truth, initial, features = [], [], []
    for index in range(count):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.01 * index, 0.0]))[0]
        pose[:3, 3] = [0.02 * index, 0.003 * index, 0.0]
        camera_points = (world - pose[:3, 3]) @ pose[:3, :3]
        pixels = (camera_points[:, :2] / camera_points[:, 2, None] * [camera.fx, camera.fy]
                  + [camera.cx, camera.cy] + rng.normal(0, 0.25, (points, 2)))
        z = camera_points[:, 2] + rng.normal(0, 0.003, points)
        measured = np.column_stack(((pixels[:, 0] - camera.cx) * z / camera.fx,
                                    (pixels[:, 1] - camera.cy) * z / camera.fy, z))
        if corrupt and index:
            measured[:5, 2] += 0.15
        features.append(Features(pixels, measured, descriptors.copy()))
        drifted = pose.copy()
        if index:
            drifted[:3, :3] = cv2.Rodrigues(np.array([0.008 * index, 0, 0.006 * index]))[0] @ pose[:3, :3]
            drifted[:3, 3] += [0.008 * index, -0.002 * index, 0.004 * index]
        truth.append(pose)
        initial.append(drifted)
    tracks = [[(view, landmark) for view in range(count)] for landmark in range(points)]
    return camera, world, truth, initial, features, tracks


class BundleAdjustmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_quality import scene_frames
        cls.scene = scene_frames(6)

    def raycast_engine(self):
        poses = []
        for index, (_, _, truth) in enumerate(self.scene):
            drifted = truth.copy()
            drifted[0, 3] += index * 0.01
            drifted[2, 3] += index * 0.003
            poses.append((index, drifted))
        return SimpleNamespace(settings=ScanSettings(), poses=poses,
                               raw_frames=[(rgb, depth) for rgb, depth, _ in self.scene],
                               frame_metadata=[{} for _ in self.scene])

    def test_end_to_end_sift_tracks_and_independent_depth_recover_drift(self):
        engine = self.raycast_engine()
        originals = [(index, pose.copy()) for index, pose in engine.poses]
        proposals, report = propose_bundle_poses(engine)
        self.assertIsNotNone(proposals, report)
        self.assertGreater(report["landmarks"], 40)
        self.assertGreater(report["observations"], report["landmarks"] * 3)
        self.assertLess(report["validation_after_m2"], report["validation_before_m2"] * 0.98)
        for (index, pose), (_, old), frame in zip(proposals, originals, self.scene):
            self.assertLess(np.linalg.norm(pose[:3, 3] - frame[2][:3, 3]), 0.006, report)
            np.testing.assert_array_equal(engine.poses[index][1], old)
        np.testing.assert_array_equal(proposals[0][1], originals[0][1])

    def test_interpolated_omitted_views_are_also_validated_against_their_depth(self):
        engine = self.raycast_engine()
        with patch("scanner_server.bundle_adjustment.MAX_KEYFRAMES", 3):
            proposals, report = propose_bundle_poses(engine)
        self.assertIsNotNone(proposals, report)
        self.assertEqual(6, len(proposals))
        self.assertEqual(3, report["keyframes"])
        self.assertGreaterEqual(report["validation_pairs"], 5)
        # Positions 0, 2 and 5 are optimized. Missing depth at omitted view 1
        # must refuse the proposal, even though the optimization still succeeds.
        rgb, depth = engine.raw_frames[1]
        engine.raw_frames[1] = (rgb, np.zeros_like(depth))
        with patch("scanner_server.bundle_adjustment.MAX_KEYFRAMES", 3):
            proposals, report = propose_bundle_poses(engine)
        self.assertIsNone(proposals, report)
        self.assertIn("Independent depth", report["reason"])

    def test_output_interpolation_is_invariant_to_choice_of_world_origin(self):
        _, _, _, initial, _, _ = bundle_fixture(count=6)
        selected = np.array([0, 2, 5])
        refined = [initial[index].copy() for index in selected]
        refined[1][:3, 3] += [0.01, 0.015, -0.01]
        refined[2][:3, :3] = cv2.Rodrigues(np.array([0.01, 0.02, 0.01]))[0] @ refined[2][:3, :3]
        world = np.eye(4)
        world[:3, :3] = cv2.Rodrigues(np.array([0.2, 0.1, -0.3]))[0]
        world[:3, 3] = [20, -30, 7]
        output = _output_poses(list(enumerate(initial)), selected, refined)
        transformed = _output_poses([(i, world @ pose) for i, pose in enumerate(initial)], selected,
                                   [world @ pose for pose in refined])
        for (_, a), (_, b) in zip(output, transformed):
            np.testing.assert_allclose(world @ a, b, atol=1e-10)

    def test_interpolation_keeps_world_correction_direction_when_cameras_turn(self):
        poses = []
        for index in range(5):
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(np.array([0.0, index * 0.35, 0.0]))[0]
            pose[:3, 3] = [index * 0.1, 0, 0]
            poses.append((index, pose))
        selected = np.array([0, 4])
        refined = [poses[0][1].copy(), poses[4][1].copy()]
        refined[1][0, 3] += 0.04
        output = _output_poses(poses, selected, refined)
        for (index, old), (_, new) in zip(poses, output):
            np.testing.assert_allclose(new[:3, 3] - old[:3, 3], [index * 0.01, 0, 0], atol=1e-12)
            np.testing.assert_allclose(new[:3, :3], old[:3, :3], atol=1e-12)

    def test_joint_optimizer_recovers_cameras_and_landmarks_with_noisy_metric_data(self):
        camera, world, truth, initial, features, tracks = bundle_fixture()
        problem = make_problem(initial, features, tracks)
        old_landmarks = problem.landmarks.copy()
        refined, landmarks, report = solve_bundle(problem, camera)
        before = np.mean([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(initial, truth)])
        after = np.mean([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(refined, truth)])
        self.assertLess(after, before * 0.15, report)
        self.assertLess(np.mean(np.linalg.norm(landmarks - world, axis=1)), 0.003, report)
        self.assertLess(np.mean(np.linalg.norm(landmarks - world, axis=1)),
                        np.mean(np.linalg.norm(old_landmarks - world, axis=1)) * 0.2)
        self.assertLess(report["training_after"], report["training_before"] * 0.1)
        np.testing.assert_array_equal(refined[0], truth[0])
        np.testing.assert_array_equal(problem.landmarks, old_landmarks)

    def test_joint_optimizer_robust_loss_limits_corrupt_depth_influence(self):
        camera, _, truth, initial, features, tracks = bundle_fixture(corrupt=True)
        refined, _, report = solve_bundle(make_problem(initial, features, tracks), camera)
        self.assertLess(np.mean([np.linalg.norm(a[:3, 3] - b[:3, 3])
                                 for a, b in zip(refined, truth)]), 0.004, report)

    def test_nonidentity_world_anchor_is_exactly_preserved(self):
        camera, _, _, initial, features, tracks = bundle_fixture(count=3)
        world = np.eye(4)
        world[:3, :3] = cv2.Rodrigues(np.array([0.1, -0.2, 0.15]))[0]
        world[:3, 3] = [1.0, -2.0, 0.5]
        initial = [world @ pose for pose in initial]
        refined, _, _ = solve_bundle(make_problem(initial, features, tracks), camera)
        np.testing.assert_array_equal(refined[0], initial[0])

    def test_track_cycles_cannot_replace_feature_identity_in_one_camera(self):
        _, _, _, _, features, _ = bundle_fixture(count=3)
        pairs = [(0, 1, np.array([[0, 0], [1, 1]])),
                 (1, 2, np.array([[0, 0], [1, 1]])),
                 (0, 2, np.array([[2, 0]]))]
        tracks = build_tracks(features, pairs)
        self.assertEqual([[(0, 1), (1, 1), (2, 1)]], tracks)
        self.assertEqual([], build_tracks(features, pairs[:1]))

    def test_landmark_budget_keeps_support_for_shorter_anchored_tracks(self):
        _, _, _, _, features, _ = bundle_fixture(count=8, points=100)
        # 80 long tracks only cover views 2..7; shorter tracks connect 0 and 1
        # to 2. Taking longest tracks alone would erase the world anchor.
        pairs = [(2, view, np.column_stack((np.arange(80), np.arange(80)))) for view in range(3, 8)]
        pairs += [(0, 1, np.column_stack((np.arange(80, 100), np.arange(80, 100)))),
                  (1, 2, np.column_stack((np.arange(80, 100), np.arange(80, 100))))]
        with patch("scanner_server.bundle_adjustment.MAX_TRACKS", 40):
            tracks = build_tracks(features, pairs)
        self.assertEqual(40, len(tracks))
        self.assertGreaterEqual(sum(any(view == 0 for view, _ in track) for track in tracks), 15)
        self.assertGreaterEqual(sum(any(view == 7 for view, _ in track) for track in tracks), 15)

    def test_heldout_depth_never_uses_feature_patch_or_fills_unknown_center(self):
        camera = CameraCalibration()
        depth = np.full((480, 640), 1200, np.uint16)
        pixel = np.array([[3, 3], [10, 10]], float)
        features = Features(pixel, np.zeros((2, 3)), None)
        depth[17, 17] = 0
        points = _heldout_points(depth, camera, features)
        pixels = points[:, :2] / points[:, 2, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        self.assertFalse(np.any(np.linalg.norm(pixels - [3, 3], axis=1) <= 5))
        self.assertFalse(np.any(np.linalg.norm(pixels - [10, 10], axis=1) <= 5))
        self.assertFalse(np.any(np.all(np.isclose(pixels, [17, 17]), axis=1)))
        self.assertLessEqual(len(points), 2000)

    def test_independent_geometry_rejects_disappearing_overlap(self):
        rng = np.random.default_rng(317)
        cloud = rng.uniform([-0.3, -0.2, 1.0], [0.3, 0.2, 1.7], (1200, 3))
        old = [np.eye(4), np.eye(4)]
        wrong = [np.eye(4), np.eye(4)]
        wrong[1][0, 3] = 0.3
        valid, report = validate_depth([cloud, cloud.copy()], old, wrong, [(0, 1)])
        self.assertFalse(valid, report)
        self.assertGreater(report["validation_after_m2"], report["validation_before_m2"])

    def test_existing_partial_overlap_keeps_the_same_loss_tolerance(self):
        cloud = np.ones((120, 3))
        poses = [np.eye(4), np.eye(4)]
        # Refinement of an existing partial trajectory retains the ordinary
        # 5-percentage-point overlap tolerance. BA retains its absolute floor.
        for overlap, expected in ((0.325, True), (0.26, False)):
            measured = [(0.001, 0.33), (0.0008, overlap)] * 2
            with patch("scanner_server.bundle_adjustment._pair_distance", side_effect=measured):
                valid, report = validate_depth([cloud, cloud], poses, poses, [(0, 1)],
                                              retain_initial_overlap=True)
            self.assertEqual(expected, valid, report)
        with patch("scanner_server.bundle_adjustment._pair_distance",
                   side_effect=[(0.001, 0.33), (0.0008, 0.325)] * 2):
            valid, _ = validate_depth([cloud, cloud], poses, poses, [(0, 1)])
        self.assertFalse(valid)

    def test_timing_gates_nonfinite_or_malformed_lags(self):
        for lag in (30, -30, float("nan"), float("inf"), "bad"):
            self.assertFalse(_timely({"rgb_depth_delta_ms": lag}))
        self.assertTrue(_timely({}))
        self.assertTrue(_timely({"rgb_depth_delta_ms": 0}))

    def engine(self, count=3):
        image = np.full((480, 640, 3), 100, np.uint8)
        depth = np.full((480, 640), 1200, np.uint16)
        return SimpleNamespace(poses=[(i, np.eye(4)) for i in range(count)],
                               raw_frames=[(image.copy(), depth.copy()) for _ in range(count)],
                               frame_metadata=[{} for _ in range(count)], settings=ScanSettings())

    def test_blank_repeated_missing_and_unsynchronized_measurements_are_noops(self):
        cases = ["blank", "repeated", "missing", "unsynchronized"]
        for case in cases:
            with self.subTest(case=case):
                engine = self.engine()
                if case == "repeated":
                    y, x = np.indices((480, 640))
                    grid = (((x // 16 + y // 16) % 2) * 255).astype(np.uint8)
                    engine.raw_frames = [(np.repeat(grid[:, :, None], 3, axis=2), depth)
                                         for _, depth in engine.raw_frames]
                if case == "missing":
                    engine.raw_frames = [(rgb, np.zeros_like(depth)) for rgb, depth in engine.raw_frames]
                if case == "unsynchronized":
                    engine.frame_metadata[1]["rgb_depth_delta_ms"] = 30
                before = [(index, pose.copy()) for index, pose in engine.poses]
                proposals, report = propose_bundle_poses(engine)
                self.assertIsNone(proposals, report)
                self.assertFalse(report["applied"])
                for a, b in zip(before, engine.poses):
                    np.testing.assert_array_equal(a[1], b[1])

    def test_large_recording_attempts_bounded_keyframes_instead_of_size_refusal(self):
        engine = self.engine(143)
        with patch("scanner_server.bundle_adjustment.extract_features",
                   wraps=bundle.extract_features) as extract:
            proposals, report = propose_bundle_poses(engine)
        self.assertIsNone(proposals, report)  # Blank scene cannot support landmarks.
        self.assertEqual(24, extract.call_count)
        self.assertEqual(143, report["accepted_views"])
        self.assertEqual(24, report["keyframes"])
        self.assertIn("tracks", report["reason"])

    def test_runtime_budget_declines_before_unsafe_work(self):
        engine = self.engine(3)
        with patch("scanner_server.bundle_adjustment.MAX_SECONDS", 0):
            proposals, report = propose_bundle_poses(engine)
        self.assertIsNone(proposals)
        self.assertTrue(report["budget_exceeded"])

    def test_streaming_validation_covers_143_views_and_rejects_bad_late_view(self):
        import time
        engine = self.engine(143)
        report = {}
        old = [pose.copy() for _, pose in engine.poses]
        new = [pose.copy() for pose in old]
        # A bad late correction must be seen even after early cache eviction.
        new[-1][0, 3] = 0.3
        clouds = _DepthClouds(engine, {}, time.monotonic(), report)
        valid, validation = validate_depth(clouds, old, new,
            [(i, i + 1) for i in range(142)] + [(0, 142)])
        self.assertFalse(valid, validation)
        self.assertEqual(143, report["validation_views"])
        self.assertEqual(143, validation["validation_pairs"])
        self.assertLessEqual(report["validation_peak_cached_views"], VALIDATION_CACHE_VIEWS)
        self.assertLessEqual(report["validation_peak_cache_bytes"], VALIDATION_CACHE_VIEWS * 2000 * 3 * 8)
        self.assertTrue(any(row["source_position"] == 142 or row["target_position"] == 142
                            for row in validation["validation_rejected_pairs"]))
        # Missing independent depth in the last omitted camera must also fail.
        rgb, depth = engine.raw_frames[142]
        engine.raw_frames[142] = (rgb, np.zeros_like(depth))
        clouds = _DepthClouds(engine, {}, time.monotonic(), {})
        valid, validation = validate_depth(clouds, old, old, [(141, 142)])
        self.assertFalse(valid)
        self.assertEqual([141, 142], validation["validation_failed_pair"])

    def test_actual_bundle_proposal_validates_all_143_views_with_bounded_residency(self):
        # Six measured raycast views with denser captures between selected
        # views. The same static observation may recur, but distinct viewpoints
        # must still produce verified three-view feature tracks.
        raw, poses, truth = [], [], []
        for index in range(143):
            rgb, depth, actual = self.scene[min(5, index * 6 // 143)]
            drifted = actual.copy()
            drifted[:3, 3] += [index * 0.05 / 142, 0, index * 0.015 / 142]
            raw.append((rgb, depth))
            poses.append((index, drifted))
            truth.append(actual)
        engine = SimpleNamespace(settings=ScanSettings(), poses=poses, raw_frames=raw,
                                 frame_metadata=[{} for _ in raw])
        with patch("scanner_server.bundle_adjustment.MAX_KEYFRAMES", 6):
            proposals, report = propose_bundle_poses(engine)
        self.assertIsNotNone(proposals, report)
        self.assertEqual(143, len(proposals))
        self.assertEqual(143, report["validation_views"])
        self.assertLessEqual(report["validation_peak_cached_views"], VALIDATION_CACHE_VIEWS)
        self.assertEqual("complete", report["stage"])
        self.assertLess(max(np.linalg.norm(p[:3, 3] - t[:3, 3]) for (_, p), t in zip(proposals, truth)), 0.006)
        np.testing.assert_array_equal(proposals[0][1], poses[0][1])

    def test_runtime_budget_exhausted_in_depth_validation_returns_no_proposal(self):
        from scanner_server import bundle_adjustment as bundle
        engine = self.raycast_engine()
        pair_distance = bundle._pair_distance

        def exhaust_budget(*args):
            result = pair_distance(*args)
            bundle.MAX_SECONDS = 0
            return result

        with patch("scanner_server.bundle_adjustment.MAX_SECONDS", 45), \
             patch("scanner_server.bundle_adjustment._pair_distance", side_effect=exhaust_budget):
            proposals, report = propose_bundle_poses(engine)
        self.assertIsNone(proposals, report)
        self.assertTrue(report["budget_exceeded"])
