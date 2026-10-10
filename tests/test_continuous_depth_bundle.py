import unittest
import numpy as np

from scanner_server.continuous_depth_bundle import expand_visual_measurements, refine_continuous_graph, intermediate_features
from scanner_server.motion_evidence import MotionEvidence
from tests.test_joint_depth_bundle import fixture


def observations():
    truth, source, _, _ = fixture()
    tracked = [(np.arange(60), source.features_for_pair(i, 0)[0]) for i in range(3)]
    rows = []
    for i in range(3):
        feature = tracked[i][1]
        rows.append({"host_monotonic_s": i*.1, "capture_generation": "camera",
            "sensor_frame_sequences": {"rgb": i, "depth": i},
            "visual_tracking": {"valid": True, "segment": "same", "camera_to_local": truth[i].tolist(), "steps": i},
            "feature_observations": {"version": 1, "image_size": [640, 480], "ids": list(range(60)),
                                     "pixels": feature.pixels.tolist(), "depths_m": feature.points[:, 2].tolist()}})
    metadata = [{**rows[i], "captured_monotonic_s": i*.1} for i in (0, 2)]
    metadata[1]["motion_history"] = {"visual": rows}
    motion = MotionEvidence(metadata, source.camera, tracked=[tracked[0], tracked[2]], gravity=False)
    poses = {0: truth[0], 1: truth[2].copy()}; poses[1][1, 3] += .02
    edges = [{"source": 1, "target": 0, "transform": poses[1].copy()}]
    return poses, edges, motion, rows


class ContinuousDepthBundleTests(unittest.TestCase):
    def test_intermediate_measurements_refine_selected_poses_without_extra_images(self):
        poses, edges, motion, _ = observations()
        expanded, relations, extended = expand_visual_measurements(poses, edges, motion)
        self.assertEqual(3, len(expanded))
        self.assertEqual(60, len(extended.direct_groups))
        self.assertEqual(3, len(relations))
        result, report = refine_continuous_graph(poses, edges, motion, max_seconds=10)
        self.assertEqual({0, 1}, set(result))
        self.assertEqual(1, report["intermediate_cameras"])
        self.assertFalse(report["intermediate_images_uploaded"])
        expected = np.eye(4); expected[0, 3] = .1
        np.testing.assert_allclose(result[1], expected, atol=.001)
        self.assertAlmostEqual(.02, poses[1][1, 3])

    def test_observations_cannot_cross_generations_or_inherit_failed_frame_poses(self):
        poses, edges, motion, rows = observations()
        rows[1]["capture_generation"] = "another_camera"
        expanded, _, _ = expand_visual_measurements(poses, edges, motion)
        self.assertEqual(2, len(expanded))
        rows[1]["capture_generation"] = "camera"
        motion.metadata[0]["visual_tracking"]["valid"] = False
        motion.tracked[0] = None
        _, relations, _ = expand_visual_measurements(poses, edges, motion)
        self.assertFalse(any(e.get("sensor_visual_prior") and (e["source"] == 0 or e["target"] == 0) for e in relations))
        duplicate = dict(rows[1]["feature_observations"])
        duplicate["ids"] = [0]*60
        self.assertIsNone(intermediate_features(duplicate, motion.camera))


if __name__ == "__main__":
    unittest.main()
