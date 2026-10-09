"""Raw revalidation of retained fragment hints and diagnostic refusal gates."""

import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scanner_server.engine import ScanEngine
from scanner_server.pose_candidates import choose_keyframes, fragment_pairs, validate_fragment_boundaries
from scanner_server.refinement import _match_failure, propose_poses
from tests.test_quality import scene_frames


class PoseRefinementTests(unittest.TestCase):
    def test_keyframes_include_exact_retained_pairs_and_preserve_coverage(self):
        engine = SimpleNamespace(poses=[(i * 2, np.eye(4)) for i in range(143)],
            fragment_reconnection={"applied": True, "verified_bridges": [
                {"connected_to_scan": True, "support": [[6, 270], [8, 268]]},
                {"connected_to_scan": False, "support": [[10, 200]]},
                {"connected_to_scan": True, "support": [[7, 100]]}]})
        hints = fragment_pairs(engine)
        self.assertEqual([(3, 135), (4, 134)], hints)
        chosen = choose_keyframes(143, 24, hints)
        self.assertEqual(24, len(chosen))
        self.assertTrue({0, 142, 3, 135, 4, 134} <= set(chosen))
        self.assertLessEqual(max(np.diff(chosen)), 13)
        engine.fragment_reconnection["applied"] = False
        self.assertEqual([], fragment_pairs(engine))

    def test_fragment_support_finds_raw_loop_without_importing_saved_transform(self):
        engine = ScanEngine(device="cpu")
        base = scene_frames(10)
        frames = base + base[-2::-1]
        for i, (rgb, depth, truth) in enumerate(frames):
            engine.store_frame(rgb, depth)
            drifted = truth.copy()
            drifted[0, 3] += i * 0.025
            engine.poses.append((i, drifted))
        originals = [(i, p.copy()) for i, p in engine.poses]
        # Saved transform is deliberately invalid: only raw support identities
        # qualify as retrieval hints; reciprocal measured geometry supplies poses.
        engine.fragment_reconnection = {"applied": True, "verified_bridges": [
            {"connected_to_scan": True, "support": [[3, 15]],
             "transform": np.full((4, 4), np.nan).tolist()}]}
        with patch("scanner_server.refinement.retrieve_pairs", return_value=[]):
            proposals, report = propose_poses(engine, max_keyframes=16)
        self.assertIsNotNone(proposals, report)
        self.assertTrue({3, 15} <= set(report["keyframe_indices"]))
        self.assertGreater(report["fragment_loops"], 0)
        self.assertTrue(any(row["origin"] == "fragment_support" and row["accepted"]
                            for row in report["candidate_results"]))
        self.assertLess(np.linalg.norm(proposals[-1][1][:3, 3]), 0.04)
        for (_, old), (_, unchanged) in zip(originals, engine.poses):
            np.testing.assert_array_equal(old, unchanged)
        json.dumps(report, allow_nan=False)

    def test_flat_loop_records_actual_rejection_gate(self):
        engine = ScanEngine(device="cpu")
        rgb = np.full((480, 640, 3), 100, np.uint8)
        depth = np.full((480, 640), 1200, np.uint16)
        for i in range(10):
            engine.store_frame(rgb, depth)
            pose = np.eye(4)
            pose[0, 3] = 0.03 * (i if i < 5 else 9 - i)
            engine.poses.append((i, pose))
        proposals, report = propose_poses(engine)
        self.assertIsNone(proposals)
        self.assertEqual(0, report["loops"])
        self.assertGreater(report["rejection_counts"].get("insufficient_normal_diversity", 0), 0)
        self.assertTrue(all(not row["accepted"] for row in report["candidate_results"]))
        with patch("scanner_server.refinement.MAX_SECONDS", 0):
            proposals, report = propose_poses(engine)
        self.assertIsNone(proposals)
        self.assertTrue(report["budget_exceeded"])

    def test_match_diagnostics_distinguish_overlap_error_and_degeneracy(self):
        target = SimpleNamespace(normals=np.tile(np.eye(3), (40, 1)))
        result = SimpleNamespace(transformation=np.eye(4), fitness=0.9, inlier_rmse=0.005,
                                 correspondence_set=np.column_stack((np.arange(120), np.arange(120))))
        self.assertIsNone(_match_failure(result, target))
        result.fitness = float("nan")
        self.assertEqual("insufficient_overlap", _match_failure(result, target))
        result.fitness, result.inlier_rmse = 0.9, 0.02
        self.assertEqual("alignment_error", _match_failure(result, target))
        result.inlier_rmse = 0.005
        target.normals[:] = [0, 0, 1]
        self.assertEqual("insufficient_normal_diversity", _match_failure(result, target))

    def test_later_corrections_cannot_override_measured_temporal_boundary(self):
        engine = ScanEngine(device="cpu")
        frames = scene_frames(3)
        for i, (rgb, depth, truth) in enumerate(frames):
            engine.store_frame(rgb, depth)
            engine.poses.append((i, truth.copy()))
        engine.fragment_reconnection = {"applied": True, "verified_bridges": [
            {"connected_to_scan": True, "temporal_constraint": True,
             "validation_scope": "temporal camera pair", "visual_constraint": True,
             "support": [[0, 1]]}]}
        valid, report = validate_fragment_boundaries(engine, engine.poses)
        self.assertTrue(valid, report)
        self.assertEqual(1, report["validated_fragment_boundaries"])
        wrong = [(i, pose.copy()) for i, pose in engine.poses]
        wrong[1][1][0, 3] += 0.1
        valid, report = validate_fragment_boundaries(engine, wrong)
        self.assertFalse(valid, report)
        self.assertEqual("boundary_correction_bounds", report["fragment_boundary_checks"][0]["reason"])
        # A correction inside the 3 cm bound must still pass measured identity.
        wrong[1][1][0, 3] -= 0.08
        valid, report = validate_fragment_boundaries(engine, wrong)
        self.assertFalse(valid, report)
        self.assertEqual("boundary_measurement_disagreement", report["fragment_boundary_checks"][0]["reason"])
        for (_, pose), (_, _, truth) in zip(engine.poses, frames):
            np.testing.assert_array_equal(pose, truth)


if __name__ == "__main__":
    unittest.main()
