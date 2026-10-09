"""Pose observability, measured depth and registration recovery checks."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import open3d as o3d

from scanner_server.geometry_registration import observability, prepare_view, register_pair, evaluate, descriptor_pairs
from shared.settings import ScanSettings
from tests.test_quality import scene_frames
from scanner_server.depth_graph import recover_depth_graph, solve_graph, bridge_visibility, scene_visibility, aggregate_bridge
from scanner_server.engine import ScanEngine


class ObservabilityTests(unittest.TestCase):
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
        self.assertEqual("offline_depth_graph_v1", engine.fragment_reconnection["algorithm"])
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


if __name__ == "__main__":
    unittest.main()
