"""CUDA proposals retain measured identities, independent gates and CPU scope."""

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import numpy as np
import open3d as o3d

from scanner_server.appearance import Features, correspondences
from scanner_server.cuda_matching import Matcher, selection
from scanner_server.cuda_odometry import RGBDOdometry
from scanner_server.cuda_registration import registration_scope
from scanner_server.engine import ScanEngine
from scanner_server.refinement import _match, _trustworthy
from scanner_server.tracking_cache import TargetPyramid
from scanner_server.visual_refinement import measured_pose
from shared.settings import ScanSettings
from tests.test_icp_source_cache import CountedSource, cloud
from tests.test_quality import scene_frames


def features(descriptors):
    n = 0 if descriptors is None else len(descriptors)
    return Features(np.zeros((n, 2)), np.zeros((n, 3)), descriptors)


class CUDAMatchingTests(unittest.TestCase):
    def test_automatic_compiler_failure_falls_back_and_explicit_request_fails(self):
        with patch.dict(os.environ, KINECT_CUDA_MATCHING="auto"), \
                patch("scanner_server.cuda_matching._kernels", side_effect=ValueError("compile failure")):
            status = selection("CUDA:0")
            self.assertEqual("cpu", status["implementation"])
            self.assertIn("compile failure", status["fallback_reason"])
        with patch.dict(os.environ, KINECT_CUDA_MATCHING="cuda"), \
                patch("scanner_server.cuda_matching._kernels", side_effect=ValueError("compile failure")):
            with self.assertRaisesRegex(RuntimeError, "compile failure"):
                selection("CUDA:0")

    def test_cpu_selection_never_compiles(self):
        with patch.dict(os.environ, KINECT_CUDA_MATCHING="cpu"), \
                patch("scanner_server.cuda_matching._kernels") as compile_kernel:
            self.assertEqual("cpu", selection("CUDA:0")["implementation"])
        compile_kernel.assert_not_called()

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_batched_hamming_matches_cpu_with_ties_and_invalid_views(self):
        rng = np.random.default_rng(42)
        source_descriptors = rng.integers(0, 256, (127, 32), dtype=np.uint8)
        source_descriptors[1] = source_descriptors[0]
        a = features(source_descriptors)
        b = source_descriptors.copy()
        # Duplicate zero-distance targets must fail the strict ratio test.
        b[3] = b[4]
        b[50:70] ^= 1
        c = rng.integers(0, 256, (211, 32), dtype=np.uint8)
        c[:60] = source_descriptors[:60]
        targets = [features(None), features(b), features(c), features(b[:39]),
                   features(b.astype(np.float32))]
        matcher = Matcher()
        for repeat in range(2):
            for target, actual in zip(targets, matcher.match(a, targets)):
                np.testing.assert_array_equal(correspondences(a, target), actual)
        # The bounded bank releases GPU descriptors of evicted views.
        matcher.match(a, targets[1:2])
        self.assertEqual({id(targets[1])}, set(matcher.cache))

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_all_zero_distances_and_partial_warps(self):
        matcher = Matcher()
        for n in (40, 41, 127):
            source = features(np.zeros((n, 32), np.uint8))
            targets = [features(np.zeros((m, 32), np.uint8)) for m in (40, 43, 1199)]
            for result in matcher.match(source, targets):
                self.assertEqual((0, 2), result.shape)

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_quantized_sift_matches_cpu_including_close_neighbours(self):
        rng = np.random.default_rng(7)
        a = rng.integers(0, 180, (151, 128)).astype(np.float32)
        b = np.clip(a + rng.integers(-12, 13, a.shape), 0, 255).astype(np.float32)
        b[20:40] = b[:20]
        c = np.concatenate((a[:70], rng.integers(0, 180, (127, 128)).astype(np.float32)))
        source = features(a)
        targets = [features(b), features(c), features(np.zeros((41,128),np.float32))]
        for target, actual in zip(targets, Matcher().match(source, targets)):
            np.testing.assert_array_equal(correspondences(source, target), actual)
        # General floating descriptors do not inherit the quantized guarantee.
        fractional = features(a + .25)
        self.assertIsNone(Matcher().match(fractional, targets))

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_sift_feature_ties_preserve_every_feature_with_bounded_gpu_memory(self):
        rng = np.random.default_rng(69)
        matcher = Matcher()
        for n in (1201, 1280):
            descriptors = rng.integers(0, 256, (n, 128)).astype(np.float32)
            source, target = features(descriptors), features(descriptors.copy())
            actual = matcher.match(source, [target])
            self.assertIsNotNone(actual)
            np.testing.assert_array_equal(correspondences(source, target), actual[0])
            self.assertIn(n-1, actual[0][:, 0])
        too_large = features(rng.integers(0, 256, (1281, 128)).astype(np.float32))
        self.assertIsNone(matcher.match(too_large, [target]))
        self.assertIsNone(matcher.match(source, [too_large]))
        self.assertIsNone(matcher.match(source, [target]*41))


class CUDARegistrationTests(unittest.TestCase):
    def test_complete_cloud_cuda_verification_is_explicitly_opt_in(self):
        from scanner_server.cuda_registration import selection

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual("legacy", selection("CUDA:0")["implementation"])
        with patch.dict(os.environ, KINECT_CUDA_REGISTRATION="tensor"):
            self.assertEqual("tensor", selection("CUDA:0")["implementation"])

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_scoped_cuda_registration_preserves_pose_support_and_restores_cpu(self):
        target = cloud()
        truth = np.eye(4)
        truth[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.003, 0.012, -0.004))
        truth[:3, 3] = [0.013, -0.004, 0.002]
        source = copy.deepcopy(target).transform(np.linalg.inv(truth))
        expected = _match(source, target, np.eye(4))
        holder = SimpleNamespace(device=o3d.core.Device("CUDA:0"),
                                 backend={"geometric_verification": {"implementation": "tensor"}})
        @registration_scope
        def run(engine):
            return _match(source, target, np.eye(4))
        actual = run(holder)
        np.testing.assert_allclose(expected.transformation, actual.transformation, atol=2e-6, rtol=0)
        self.assertEqual(_trustworthy(expected, target), _trustworthy(actual, target))
        self.assertEqual(len(expected.correspondence_set), len(actual.correspondence_set))
        with patch("scanner_server.refinement.REG.registration_icp", wraps=o3d.pipelines.registration.registration_icp) as legacy:
            _match(source, target, np.eye(4))
        self.assertEqual(3, legacy.call_count)

    def test_scope_restores_cpu_after_exception(self):
        holder = SimpleNamespace(device="CUDA:0", backend={"geometric_verification": {"implementation": "tensor"}})
        @registration_scope
        def fail(engine):
            raise ValueError("injected failure")
        with self.assertRaises(ValueError):
            fail(holder)
        from scanner_server.cuda_registration import _device
        self.assertIsNone(_device.get())


class KeyframePreparationTests(unittest.TestCase):
    def test_deferred_raw_normals_replace_cached_levels_and_uploads(self):
        # This thin edge has ambiguous normal signs. Raw normal preparation
        # supplies an orientation inherited by voxel downsampling. Adding it
        # must update both cached derivative forms.
        x = np.arange(-.2, .201, .005)
        points = np.concatenate([np.column_stack((x, np.zeros_like(x), np.full_like(x, z)))
                                 for z in (1., 1.026, 1.031, 1.036)])
        raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        prepared = TargetPyramid(raw)
        voxel = .0075
        old = prepared.level(voxel)
        old_upload = prepared.tensor(voxel, lambda level: np.asarray(level.normals).copy())
        raw.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.03, max_nn=30))
        expected = raw.voxel_down_sample(voxel)
        expected.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=30))
        self.assertGreater(np.max(np.abs(np.asarray(old.normals)-np.asarray(expected.normals))), .1)
        actual = prepared.level(voxel)
        upload = prepared.tensor(voxel, lambda level: np.asarray(level.normals).copy())
        np.testing.assert_allclose(expected.normals, actual.normals, atol=1e-12)
        np.testing.assert_allclose(expected.normals, upload, atol=1e-12)
        self.assertIsNot(old, actual)
        self.assertIsNot(old_upload, upload)

    def test_target_levels_are_reused_and_generic_targets_are_not_cached(self):
        with patch.dict(os.environ, KINECT_MODEL_REFRESH="eager", KINECT_KEYFRAME_CACHE="on"):
            engine = ScanEngine(device="cpu")
        target = cloud()
        counted = CountedSource(target)
        engine._visual_cache[0] = features(None), counted
        source = cloud()
        engine._icp(source, counted, np.eye(4))
        engine._icp(source, counted, np.eye(4))
        self.assertEqual(3, counted.downsamples)
        generic = CountedSource(target)
        engine._icp(source, generic, np.eye(4))
        engine._icp(source, generic, np.eye(4))
        self.assertEqual(6, generic.downsamples)
        engine.reset()
        self.assertEqual({}, engine._visual_target_pyramids)

    def test_deferred_model_levels_use_the_existing_surface_snapshot(self):
        with patch.dict(os.environ, KINECT_MODEL_REFRESH="lazy"):
            engine = ScanEngine(device="cpu")
        rgb, depth, _ = scene_frames(1)[0]
        engine.store_frame(rgb, depth)
        engine.process_frames()
        engine._refresh_live_points()
        points = engine._live_points.copy()
        with patch.object(engine, "_refresh_live_points") as extract:
            engine._extract_model_pcd(from_snapshot=True)
        extract.assert_not_called()
        self.assertIsNone(engine._pending_model_cloud)
        np.testing.assert_array_equal(points, engine._live_points)

    def test_deferred_recovery_keeps_pose_and_never_runs_expensive_fallbacks(self):
        with patch.dict(os.environ, KINECT_LIVE_RECOVERY="deferred"):
            engine = ScanEngine(device="cpu")
            engine.reset(settings=ScanSettings(color_recovery=True))
        with patch.object(engine, "_visual_register", return_value=None), \
                patch.object(engine, "_recover_anchor") as anchor, \
                patch.object(engine, "_color_recovery") as color, \
                patch.object(engine, "_icp") as icp:
            pose, message = engine._register(None)
        self.assertIsNone(pose)
        self.assertIn("retained for Finish", message)
        anchor.assert_not_called()
        color.assert_not_called()
        icp.assert_not_called()


@unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
class CUDAOdometryTests(unittest.TestCase):
    def test_metric_depth_identity_and_bounded_image_cache(self):
        engine = ScanEngine(device="cuda")
        rgb, depth, _ = scene_frames(1)[0]
        frame = engine._make_rgbd(rgb, depth)
        odometry = RGBDOdometry(engine.device, capacity=2)
        pose = odometry.estimate(frame, frame, engine.intrinsic_tensor, np.eye(4), 4.0, "hybrid")
        np.testing.assert_allclose(pose, np.eye(4), atol=2e-6, rtol=0)
        odometry.image(engine._make_rgbd(rgb, depth))
        odometry.image(engine._make_rgbd(rgb, depth))
        self.assertEqual(2, len(odometry.frames))

    def test_invalid_odometry_never_returns_a_pose(self):
        engine = ScanEngine(device="cuda")
        rgb, depth, _ = scene_frames(1)[0]
        frame = engine._make_rgbd(rgb, depth)
        odometry = RGBDOdometry(engine.device)
        result = SimpleNamespace(transformation=o3d.core.Tensor(np.full((4, 4), np.nan)), fitness=1.0)
        with patch("scanner_server.cuda_odometry.o3d.t.pipelines.odometry.rgbd_odometry_multi_scale", return_value=result):
            self.assertIsNone(odometry.estimate(frame, frame, engine.intrinsic_tensor, np.eye(4), 4.0, "hybrid"))

    def test_visual_identity_gate_rejects_a_wrong_dense_pose(self):
        with patch.dict(os.environ, KINECT_CUDA_ODOMETRY="hybrid", KINECT_CUDA_MATCHING="cpu"):
            engine = ScanEngine(device="cuda")
        engine.reset(settings=ScanSettings(color_recovery=True))
        rgb, depth, _ = scene_frames(1)[0]
        engine.store_frame(rgb, depth, {"timestamp_s": 0})
        engine.process_frames()
        engine.store_frame(rgb, depth, {"timestamp_s": .1})
        wrong = np.eye(4)
        wrong[0, 3] = .18
        # A high-overlap wrong result must retain the observed appearance pose.
        with patch.object(engine, "_projective_pose", return_value=wrong):
            frame = engine._make_rgbd(rgb, depth)
            source = engine._make_reg_pcd(frame)
            result = engine._visual_register(source, frame)
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(result.transformation[:3, 3]), .005)


class MeasuredRefinementTests(unittest.TestCase):
    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_measured_fine_tracks_known_motion_with_both_feature_types(self):
        frames = scene_frames(20)
        for method in ("orb", "sift"):
            with self.subTest(method=method), patch.dict(os.environ,
                    KINECT_VISUAL_REFINEMENT="measured-fine", KINECT_VISUAL_FEATURES=method,
                    KINECT_CUDA_REGISTRATION="cpu", KINECT_CUDA_ODOMETRY="off", KINECT_LIVE_RECOVERY="full"):
                engine = ScanEngine(device="cuda")
                engine.reset(settings=ScanSettings(color_recovery=True, confidence_fusion=True))
                for i, (rgb, depth, _) in enumerate(frames):
                    engine.store_frame(rgb, depth, {"timestamp_s": i*.2, "rgb_depth_delta_ms": 0})
                engine.process_frames()
                self.assertEqual(20, engine.frame_count, engine.diagnostics)
                errors = [np.linalg.norm(pose[:3, 3] - frames[i][2][:3, 3]) for i, pose in engine.poses]
                self.assertLess(max(errors), .005, errors)
                if method == "sift":
                    self.assertGreater(engine.backend["visual_refinement_statistics"]["attempts"], 10)
                # The periodic synthetic texture does not consistently supply
                # ORB identities; that subcase also checks geometric fallback.

    def test_verified_volume_replaces_a_pending_lazy_model_snapshot(self):
        with patch.dict(os.environ, KINECT_MODEL_REFRESH="lazy"):
            engine = ScanEngine(device="cpu")
            engine.reset(settings=ScanSettings(reconnect_fragments=True, max_translation_m=.05, final_weight=.5))
        frames = scene_frames(14)
        for i, index in enumerate((0, 1, 2, 10, 11, 12)):
            rgb, depth, _ = frames[index]
            engine.store_frame(rgb, depth, {"timestamp_s": i if i < 3 else i+10, "rgb_depth_delta_ms": 30})
        engine.process_frames()
        self.assertEqual(3, engine.frame_count)
        engine._refresh_live_points()
        snapshot = engine._pending_model_cloud
        self.assertIsNotNone(snapshot)
        with patch.object(engine, "_create_vbg", side_effect=RuntimeError("simulated allocation failure")):
            engine._reconnect_volume()
        self.assertIs(snapshot, engine._pending_model_cloud)
        self.assertIsNone(engine._reconnection_count)
        self.assertTrue(engine.build_mesh()[0])
        self.assertEqual(6, engine.frame_count)
        self.assertIsNone(engine._pending_model_cloud)
        self.assertIsNone(engine._pending_model_frame_count)
        self.assertEqual(0, engine._integrations_since_preview)

    def test_insufficient_or_nonfinite_measured_motion_never_returns_a_pose(self):
        a = features(np.ones((41, 128), np.float32))
        camera = ScanSettings().camera
        self.assertIsNone(measured_pose(a, a, np.column_stack((np.arange(39), np.arange(39))), np.eye(4), camera))
        with patch("scanner_server.visual_refinement.measured_rigid_motion", return_value=np.full((4, 4), np.nan)), \
                patch("scanner_server.visual_refinement.refine_measured_motion", return_value=np.full((4, 4), np.nan)):
            self.assertIsNone(measured_pose(a, a, np.column_stack((np.arange(41), np.arange(41))), np.eye(4), camera))

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_known_moving_scene_keeps_metric_pose_without_truth_initialization(self):
        with patch.dict(os.environ, KINECT_VISUAL_REFINEMENT="measured", KINECT_VISUAL_FEATURES="sift",
                        KINECT_CUDA_REGISTRATION="cpu", KINECT_CUDA_ODOMETRY="off", KINECT_LIVE_RECOVERY="full"):
            engine = ScanEngine(device="cuda")
            engine.reset(settings=ScanSettings(color_recovery=True, confidence_fusion=True))
        frames = scene_frames(20)
        for i, (rgb, depth, _) in enumerate(frames):
            engine.store_frame(rgb, depth, {"timestamp_s": i*.2, "rgb_depth_delta_ms": 0})
        engine.process_frames()
        self.assertEqual(len(frames), engine.frame_count, engine.diagnostics)
        errors = [np.linalg.norm(pose[:3, 3] - frames[i][2][:3, 3]) for i, pose in engine.poses]
        self.assertLess(max(errors), .005, errors)
        self.assertGreater(engine.backend["visual_refinement_statistics"]["attempts"], 10)
        # The refinement consumes measured image/depth identities, never test poses.
        self.assertFalse(any("reference_pose" in item for item in engine.frame_metadata))

    def test_visual_first_bridge_skips_geometry_only_after_visual_and_heldout_verification(self):
        from scanner_server.fragments import _verify_bridge

        bridge = {"validation_scope": "visual and held-out camera pairs"}
        with patch("scanner_server.fragments._verify_visual_bridge", return_value=bridge) as visual, \
                patch("scanner_server.fragments._pair") as pair:
            self.assertIs(bridge, _verify_bridge(None, None, np.eye(4), ScanSettings().camera, visual_first=True))
        self.assertEqual(1, visual.call_count)
        pair.assert_not_called()
        with patch("scanner_server.fragments._verify_visual_bridge", return_value=None), \
                patch("scanner_server.fragments._pair", side_effect=ValueError("geometry fallback")):
            fragment = SimpleNamespace(train=None)
            with self.assertRaisesRegex(ValueError, "geometry fallback"):
                _verify_bridge(fragment, fragment, np.eye(4), ScanSettings().camera, visual_first=True)

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_measured_finish_connects_a_capture_gap_without_truth_seeds(self):
        from scanner_server.fragments import propose_fragment_poses

        with patch.dict(os.environ, KINECT_VISUAL_REFINEMENT="measured", KINECT_VISUAL_FEATURES="sift",
                        KINECT_CUDA_REGISTRATION="cpu", KINECT_CUDA_ODOMETRY="off", KINECT_LIVE_RECOVERY="full",
                        KINECT_CUDA_MATCHING="cuda",
                        KINECT_FINAL_VISUAL_FIRST="on", KINECT_FINAL_LOCAL_REFINEMENT="measured"):
            engine = ScanEngine(device="cuda")
            engine.reset(settings=ScanSettings(color_recovery=True, confidence_fusion=True))
        frames = scene_frames(20)
        for i, (rgb, depth, _) in enumerate(frames):
            engine.store_frame(rgb, depth, {"timestamp_s": i*.2 + (10 if i >= 10 else 0),
                                           "rgb_depth_delta_ms": 0})
        engine.process_frames()
        # Model a capture interruption: the later raw observations remain, but
        # have no retained world-pose estimates for Finish to trust or seed.
        engine.poses = [(i, pose) for i, pose in engine.poses if i < 10]
        engine.frame_count = len(engine.poses)
        for row in engine.diagnostics[10:]:
            row.update(success=False)
            row.pop("pose", None)
        poses, report = propose_fragment_poses(engine)
        self.assertIsNotNone(poses, report)
        self.assertEqual(20, len(poses), report)
        self.assertEqual(10, report["recovered_frames"])
        self.assertGreaterEqual(len(report["fragments"]), 2)
        self.assertTrue(report["verified_bridges"], report)
        errors = [np.linalg.norm(pose[:3, 3] - frames[i][2][:3, 3]) for i, pose in poses]
        self.assertLess(max(errors), .005, errors)
        self.assertGreater(engine.backend["descriptor_matching"]["final_cuda_batches"], 0)
        self.assertFalse(any("reference_pose" in item for item in engine.frame_metadata))

    @unittest.skipUnless(o3d.core.cuda.is_available(), "CUDA unavailable")
    def test_final_cuda_cache_preserves_exact_mutual_matches_and_new_feature_identity(self):
        from scanner_server.fragments import View, _matches, _prepare_matches

        with patch.dict(os.environ, KINECT_CUDA_MATCHING="cuda"):
            engine = ScanEngine(device="cuda")
        rng = np.random.default_rng(12)
        a = rng.integers(0, 160, (151, 128)).astype(np.float32)
        b = np.clip(a + rng.integers(-7, 8, a.shape), 0, 255).astype(np.float32)
        source = View(0, None, None, features(a))
        target = View(1, None, None, features(b))
        expected = correspondences(source.features, target.features)
        _prepare_matches(engine, [source], [target])
        with patch("scanner_server.fragments.correspondences", side_effect=AssertionError("cache miss")):
            actual = _matches(source, target)
            reverse = _matches(target, source)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(expected[:, ::-1][np.argsort(expected[:, 1], kind="stable")], reverse)
        self.assertFalse(actual.flags.writeable)
        target.features = features(rng.integers(0, 160, (151, 128)).astype(np.float32))
        with patch("scanner_server.fragments.correspondences", wraps=correspondences) as match:
            _matches(source, target)
        self.assertEqual(1, match.call_count)


if __name__ == "__main__":
    unittest.main()
