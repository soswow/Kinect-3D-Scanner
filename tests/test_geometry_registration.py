"""Pose observability, measured depth and registration recovery checks."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import open3d as o3d

from scanner_server.geometry_registration import observability, prepare_view, register_pair, evaluate, descriptor_pairs, DepthView, ppf_seeds, pose_distance
from shared.settings import ScanSettings
from tests.test_quality import scene_frames
from scanner_server.depth_graph import recover_depth_graph, solve_graph, bridge_visibility, scene_visibility, aggregate_bridge, geometric_revisits, reconnect_optimized_components, ALGORITHM_VERSION
from scanner_server.engine import ScanEngine


class ObservabilityTests(unittest.TestCase):
    def test_unconnected_singletons_do_not_request_matches_against_an_entire_map(self):
        rng = np.random.default_rng(52)
        descriptors = rng.normal(size=(100, 8))
        distances = np.sum((descriptors[:, None] - descriptors[None, :])**2, axis=2)
        pairs = geometric_revisits(distances, [list(range(98)), [98], [99]])
        self.assertLessEqual(len(pairs), 24)
        self.assertTrue(pairs)
        self.assertTrue(all(a >= 98 or b >= 98 for a,b in pairs))
        self.assertTrue(all(abs(a-b) > 3 for a,b in pairs))

    def test_batched_pair_votes_recover_motion_and_filter_gravity_before_ranking(self):
        rng = np.random.default_rng(341)
        points = rng.uniform(-.5, .5, (150, 3))
        normals = rng.normal(size=points.shape)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        source.normals = o3d.utility.Vector3dVector(normals)
        pose = np.eye(4)
        pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((.3, -.2, .4))
        pose[:3, 3] = [.3, -.15, .2]
        target = o3d.geometry.PointCloud(source).transform(pose)
        a, b = (DepthView(i, None, value, value, None) for i, value in enumerate((source, target)))
        up = np.array([0., 1., 0.])
        for gravity in (None, (up, pose[:3, :3] @ up, 5.)):
            seeds = ppf_seeds(a, b, limit=6, gravity=gravity)
            self.assertTrue(seeds)
            self.assertLess(min(pose_distance(pose, p)[0] + pose_distance(pose, p)[1] for _, p in seeds), 1e-5)
            if gravity is not None:
                for _, candidate in seeds:
                    self.assertGreaterEqual((candidate[:3, :3] @ up) @ gravity[1], np.cos(np.radians(5.)) - 1e-10)

    def test_plane_and_sphere_do_not_determine_six_direction_pose(self):
        rng = np.random.default_rng(45)
        points = rng.normal(size=(1000, 3))
        points[:, 2] = 2
        normals = np.tile([0., 0., 1.], (len(points), 1))
        self.assertLess(observability(points, normals)[1], 1e-10)
        sphere = rng.normal(size=(1000, 3))
        sphere /= np.linalg.norm(sphere, axis=1)[:, None]
        self.assertLess(observability(sphere, sphere)[1], 1e-10)

    def test_condition_is_invariant_to_rigid_coordinates_and_scale(self):
        rng = np.random.default_rng(4)
        points = rng.normal(size=(1000, 3))
        normals = rng.normal(size=points.shape)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((.7, -.5, .4))
        before = observability(points, normals)[0]
        after = observability((points @ rotation.T) * 3 + [120, -60, 35], normals @ rotation.T)[0]
        np.testing.assert_allclose(before, after, rtol=1e-10, atol=1e-10)


class DepthRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        o3d.utility.set_max_threads(4)
        cls.frames = scene_frames(28)
        cls.settings = ScanSettings(filter_depth=True)
        cls.first = prepare_view(0, cls.frames[0][1], cls.settings)
        cls.last = prepare_view(26, cls.frames[26][1], cls.settings)

    def test_two_depth_frames_recover_motion_above_old_thirty_cm_limit(self):
        pose, report = register_pair(self.last, self.first)
        self.assertIsNotNone(pose, report)
        truth = np.linalg.inv(self.frames[0][2]) @ self.frames[26][2]
        self.assertGreater(np.linalg.norm(truth[:3, 3]), .3)
        self.assertLess(np.linalg.norm(pose[:3, 3] - truth[:3, 3]), .025)
        angle = np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1)))
        self.assertLess(angle, 2)

    def test_incorrect_pose_and_blank_depth_are_rejected(self):
        wrong = np.eye(4)
        wrong[:3, 3] = [2, 0, -.5]
        self.assertFalse(evaluate(self.first, self.last, wrong)["accepted"])
        empty = prepare_view(99, np.zeros((480, 640), np.uint16), self.settings)
        self.assertEqual("insufficient_depth", evaluate(empty, self.first, np.eye(4))["reason"])

    def test_measured_plane_with_depth_noise_is_not_six_direction_evidence(self):
        rng = np.random.default_rng(81)
        for noise_mm in (2., 6.):
            raw = np.rint(2000 + rng.normal(0, noise_mm, (480,640))).astype(np.uint16)
            view = prepare_view(0, raw, self.settings)
            evidence = evaluate(view, view, np.eye(4))
            self.assertFalse(evidence["accepted"], evidence)
            self.assertEqual("unobservable_pose", evidence["reason"])

    def test_registration_uses_no_rgb_or_stored_trajectory(self):
        self.assertFalse(hasattr(self.first, "rgb"))
        self.assertFalse(hasattr(self.first, "pose"))
        before = self.first.depth.copy()
        pose, report = register_pair(self.first, self.first)
        self.assertIsNotNone(pose, report)
        np.testing.assert_array_equal(before, self.first.depth)

    def test_cpu_cuda_descriptor_search_on_separated_descriptors(self):
        if not o3d.core.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import importlib.util
        if importlib.util.find_spec("cupy") is None:
            self.skipTest("Optional CuPy unavailable")
        rng = np.random.default_rng(87)
        source = rng.normal(size=(33, 60))
        target = source[:, ::-1] + rng.normal(0, .001, source.shape)
        np.testing.assert_array_equal(descriptor_pairs(source, target), descriptor_pairs(source, target, "cuda"))

    def test_depth_finish_bypasses_live_tracking_and_does_not_need_multiple_camera_pairs(self):
        engine = ScanEngine(device="cpu")
        engine.reset(settings=replace(self.settings, offline_registration="depth", final_weight=.5,
                                      refine_poses=True, bundle_adjustment=True))
        for i, source in enumerate((0, 26)):
            rgb, depth, _ = self.frames[source]
            engine.store_frame(np.zeros_like(rgb), depth, {"timestamp_s": i * 100.})
        with patch.object(engine, "process_frames", side_effect=AssertionError("live tracking must not gate depth registration")), \
             patch.object(engine, "_refine_volume", side_effect=AssertionError("legacy refinement must not change depth graph")), \
             patch.object(engine, "_bundle_volume", side_effect=AssertionError("RGB refinement must not gate depth registration")):
            success, report = engine.build_mesh()
        self.assertTrue(success, report)
        self.assertEqual(2, engine.frame_count)
        self.assertEqual(2, engine._processed_count)
        self.assertEqual(ALGORITHM_VERSION, engine.fragment_reconnection["algorithm"])
        truth = np.linalg.inv(self.frames[0][2]) @ self.frames[26][2]
        self.assertLess(np.linalg.norm(engine.poses[-1][1][:3, 3] - truth[:3, 3]), .025)
        self.assertGreater(len(engine.mesh.triangles), 100)
        with patch("scanner_server.depth_graph.propose_depth_poses", side_effect=AssertionError("completed graph rerun")):
            self.assertTrue(engine.build_mesh()[0])
        previous_volume, previous_poses = engine.vbg, [(i,p.copy()) for i,p in engine.poses]
        rgb, depth, _ = self.frames[13]
        engine.store_frame(rgb,depth)
        with patch("scanner_server.depth_graph.propose_depth_poses",side_effect=ValueError("deliberate invalid graph")):
            self.assertFalse(engine.build_mesh()[0])
        self.assertIs(previous_volume,engine.vbg)
        self.assertEqual(2,engine.frame_count)
        self.assertEqual(3,engine.stored_count)
        self.assertEqual(2,engine._processed_count)
        for (_,before),(_,after) in zip(previous_poses,engine.poses):
            np.testing.assert_array_equal(before,after)
        engine.shutdown()

    def test_blank_component_is_retained_without_a_made_up_pose_connection(self):
        empty = prepare_view(2, np.zeros((480,640), np.uint16), self.settings)
        last = prepare_view(1, self.frames[26][1], self.settings)
        solved, report = recover_depth_graph([self.first, last, empty])
        self.assertEqual([2,1], report["component_sizes"])
        self.assertEqual([2], solved[1]["frame_indices"])

    def test_graph_removes_a_false_loop_before_offering_poses(self):
        views = [prepare_view(i, self.frames[s][1], self.settings) for i,s in enumerate((0, 13, 26))]
        true = [np.linalg.inv(self.frames[0][2]) @ self.frames[s][2] for s in (0,13,26)]
        edges = [{"source": a, "target": b, "transform": np.linalg.inv(true[b]) @ true[a],
                  "score": 50., "method": "test"} for a,b in ((1,0),(2,1))]
        wrong = (np.linalg.inv(true[0]) @ true[2]).copy()
        wrong[0,3] += .5
        edges.append({"source": 2, "target": 0, "transform": wrong, "score": 1., "method": "test"})
        solved, active, rejected = solve_graph(views, edges)
        self.assertTrue(any(e["source"] == 2 and e["target"] == 0 for e in rejected))
        self.assertEqual(3, len(solved[0]["frame_indices"]))
        self.assertEqual(2, len(active))
        self.assertLess(np.linalg.norm(solved[0]["poses"][2][:3,3]-true[2][:3,3]), .025)

    def test_measured_loop_corrects_accumulated_drift_instead_of_being_pruned(self):
        views = [replace(self.first, index=i) for i in range(20)]
        step = np.eye(4)
        step[0, 3] = .005
        edges = [{"source":i, "target":i-1, "transform":step.copy(), "score":80., "method":"local"}
                 for i in range(1,20)]
        edges.append({"source":19, "target":0, "transform":np.eye(4), "score":40.,
                      "method":"appearance_revisit", "measured_appearance":True})
        solved, active, rejected = solve_graph(views, edges)
        self.assertFalse(rejected)
        self.assertEqual(20,len(active))
        self.assertLess(np.linalg.norm(solved[0]["poses"][19][:3,3]), .01)

    def test_one_pair_is_allowed_but_additional_empty_space_cannot_be_ignored(self):
        a = prepare_view(0,self.frames[0][1],self.settings)
        b = prepare_view(1,self.frames[0][1],self.settings)
        wall = prepare_view(2,np.full((480,640),3000,np.uint16),self.settings)
        candidate = {"source": 1,"target": 0,"transform": np.eye(4),"score": 100.,"method": "test"}
        self.assertTrue(bridge_visibility([a,b],[],candidate)["accepted"])
        existing = [{"source": 2,"target": 1,"transform": np.eye(4),"score": 10.,"method": "test"}]
        report = bridge_visibility([a,b,wall],existing,candidate)
        self.assertFalse(report["accepted"],report)
        self.assertGreater(report["conflicting_view_pairs"],0)
        self.assertFalse(scene_visibility([a,b,wall],{0:np.eye(4),1:np.eye(4),2:np.eye(4)})["accepted"])

    def test_accumulated_geometry_proposal_retains_raw_depth_validation(self):
        a=prepare_view(0,self.frames[0][1],self.settings)
        b=prepare_view(1,self.frames[26][1],self.settings)
        truth=np.linalg.inv(self.frames[0][2]) @ self.frames[26][2]
        with patch("scanner_server.depth_graph.global_seeds",return_value=[("known-test-seed",truth)]), \
             patch("scanner_server.depth_graph.ppf_seeds",return_value=[]):
            edge,report=aggregate_bridge([a,b],[1],[0],[],"cpu")
        self.assertIsNotNone(edge,report)
        self.assertTrue(report["component_validation"]["accepted"])
        self.assertTrue(edge["aggregate_evidence"])
        self.assertLess(np.linalg.norm(edge["transform"][:3,3]-truth[:3,3]),.025)

    def test_local_object_change_cannot_veto_an_otherwise_supported_room(self):
        stable = np.full((480, 640), 3000, np.uint16)
        partial = np.zeros_like(stable)
        partial[180:270, 280:370] = 2900
        views = [prepare_view(i, stable, self.settings) for i in range(5)]
        views.append(prepare_view(5, partial, self.settings))
        poses = {i:np.eye(4) for i in range(6)}
        evidence = scene_visibility(views, poses)
        self.assertTrue(evidence["accepted"], evidence)
        self.assertTrue(any(p["free_space_fraction"] > .9 for p in evidence["worst_pairs"]))
        # A whole camera displaced into measured empty space remains a failure.
        poses[0][2, 3] -= .3
        self.assertFalse(scene_visibility(views, poses)["accepted"])

    def test_accumulated_map_gauges_do_not_change_camera_motion_evidence(self):
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        views = [prepare_view(0, self.frames[0][1], self.settings),
                 prepare_view(1, self.frames[26][1], self.settings)]
        truth = np.linalg.inv(self.frames[0][2]) @ self.frames[26][2]
        source_gauge, target_gauge = np.eye(4), np.eye(4)
        source_gauge[:3,:3] = Rotation.from_euler('z', 7, degrees=True).as_matrix()
        source_gauge[0,3] = .18
        target_gauge[:3,:3] = Rotation.from_euler('y', 8, degrees=True).as_matrix()
        target_gauge[0,3] = -.12
        poses = {0:target_gauge, 1:source_gauge @ truth}
        up = np.array([0., -1., 0.])
        source_up = truth[:3,:3].T @ up
        checked = []

        def check(a, b, pose):
            checked.append(pose)
            distance, angle = pose_distance(truth, pose)
            return {"accepted":distance < .025 and angle < 3}

        motion = SimpleNamespace(seeds=lambda a,b:[("measured_rgbd_features", truth)], check=check,
                                 gravity_pair=lambda a,b:(source_up, up, 60.))
        map_pose = target_gauge @ np.linalg.inv(source_gauge)
        # Supply the known rigid proposal to isolate coordinate conversion.
        # Actual selected raw depth still validates it in both directions.
        with patch("scanner_server.depth_graph.global_seeds",return_value=[]), \
                patch("scanner_server.depth_graph.ppf_seeds",return_value=[]) as ppf, \
                patch("scanner_server.depth_graph.refine",side_effect=lambda a,b,p:
                      map_pose if a.index == 1 else np.linalg.inv(map_pose)):
            edge, report = aggregate_bridge(views, [1], [0], [], "cpu", motion, poses=poses)
        self.assertIsNotNone(edge, report)
        self.assertTrue(any(pose_distance(truth, p)[0] < .025 for p in checked))
        self.assertLess(pose_distance(truth, edge["transform"])[0], .025)
        gravity = ppf.call_args.kwargs["gravity"]
        np.testing.assert_allclose(gravity[0], source_gauge[:3,:3] @ up)
        np.testing.assert_allclose(gravity[1], target_gauge[:3,:3] @ up)

    def test_corrected_component_connection_uses_optimized_poses_and_checks_the_whole_map(self):
        global_poses = {i:np.linalg.inv(self.frames[0][2]) @ self.frames[i*8][2] for i in range(4)}
        views = [prepare_view(i, self.frames[i*8][1], self.settings) for i in range(4)]
        source = {i:np.linalg.inv(global_poses[2]) @ global_poses[i] for i in (2, 3)}
        target = {i:global_poses[i].copy() for i in (0, 1)}
        wrong = np.eye(4); wrong[2, 3] = .3
        active = [{"source":a, "target":b, "transform":np.linalg.inv(global_poses[b]) @ global_poses[a],
                   "initial_transform":wrong, "score":20., "method":"local"} for a,b in ((1,0),(3,2))]
        solved = [{"frame_indices":[0,1], "poses":target}, {"frame_indices":[2,3], "poses":source}]
        measured = {"source":2,"target":0,"transform":global_poses[2],"score":30.,"method":"aggregate","aggregate_evidence":True}
        with patch("scanner_server.depth_graph.aggregate_bridge", return_value=(measured,{"accepted":True})) as bridge:
            connected, _, _ = reconnect_optimized_components(views, solved, active, None)
        self.assertEqual(1, len(connected))
        np.testing.assert_allclose(bridge.call_args.kwargs["poses"][3], source[3])
        for i in range(4):
            np.testing.assert_allclose(connected[0]["poses"][i], global_poses[i], atol=1e-10)
        # Even a falsely positive proposal still needs the separate raw audit.
        wrong_bridge = {**measured,"transform":global_poses[2].copy()}
        wrong_bridge["transform"][2,3] -= .3
        solved = [{"frame_indices":[0,1],"poses":{i:global_poses[i] for i in (0,1)}},
                  {"frame_indices":[2,3],"poses":source}]
        with patch("scanner_server.depth_graph.aggregate_bridge",return_value=(wrong_bridge,{"accepted":True})):
            connected, _, reports = reconnect_optimized_components(views, solved, active, None)
        self.assertEqual(2, len(connected))
        self.assertFalse(reports[0]["accepted"])

    def test_a_single_depth_view_can_build_a_component_without_registration(self):
        engine=ScanEngine(device="cpu")
        engine.reset(settings=replace(self.settings,offline_registration="depth",final_weight=.5))
        rgb,depth,_=self.frames[0]
        engine.store_frame(rgb,depth)
        success,report=engine.build_mesh()
        self.assertTrue(success,report)
        self.assertEqual(1,engine.frame_count)
        np.testing.assert_array_equal(engine.poses[0][1],np.eye(4))
        engine.shutdown()

    def test_measured_texture_can_determine_a_flat_wall_pose(self):
        from tests.test_visual_tracking import textured_plane
        from scanner_server.appearance import extract_features
        from scanner_server.motion_evidence import MotionEvidence
        frames = [textured_plane(shift=s) for s in (0, 21)]
        views = [prepare_view(i, raw, self.settings) for i, (_, raw) in enumerate(frames)]
        appearance = [extract_features(rgb, raw, self.settings.camera, method="sift") for rgb, raw in frames]
        motion = MotionEvidence([{}, {}], self.settings.camera, appearance, gravity=False)
        seeds = motion.seeds(1, 0)
        self.assertTrue(seeds)
        self.assertEqual("unobservable_pose", evaluate(views[1], views[0], seeds[0][1])["reason"])
        pose, report = register_pair(views[1], views[0], extra_seeds=seeds, pose_check=lambda p: motion.check(1, 0, p),
                                     reciprocal_refine=lambda p: motion.reciprocal_pose(1, 0, p))
        self.assertIsNotNone(pose, report)
        self.assertTrue(report["evidence"]["motion_evidence"]["appearance"]["accepted"])
        self.assertEqual("measured_rgbd", report["reciprocal_method"])
        self.assertGreater(abs(pose[0, 3]), .06)
        wrong = pose.copy(); wrong[0, 3] += .05
        self.assertFalse(motion.check(1, 0, wrong)["accepted"])
        self.assertIsNone(register_pair(views[1], views[0], initial=pose)[0])
        with patch.object(motion, "reciprocal_pose", return_value=None):
            self.assertIsNone(register_pair(views[1], views[0], extra_seeds=seeds,
                              pose_check=lambda p: motion.check(1, 0, p),
                              reciprocal_refine=lambda p: motion.reciprocal_pose(1, 0, p))[0])

    def test_color_revisit_search_includes_loops_within_a_connected_map(self):
        from tests.test_visual_tracking import textured_plane
        from scanner_server.appearance import extract_features
        from scanner_server.motion_evidence import MotionEvidence
        rgb, raw = textured_plane()
        features = extract_features(rgb, raw, self.settings.camera, method="sift")
        metadata = [{"captured_monotonic_s": float(i)} for i in range(10)]
        motion = MotionEvidence(metadata, self.settings.camera, [features] * 10, gravity=False)
        pairs = motion.revisit_pairs([list(range(10))])
        self.assertIn((9, 0), pairs)
        self.assertTrue(all(abs(a-b) > 3 for a, b in pairs))
        self.assertTrue(all(motion.appearance_pair(a, b) is not None for a, b in pairs))


if __name__ == "__main__":
    unittest.main()
