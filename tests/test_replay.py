"""Dataset conversion must preserve units, timing and session calibration."""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_scan import load_dataset, nearest_pairs, score

from shared.recording import RecordingWriter
from shared.settings import ScanSettings


class ReplayTests(unittest.TestCase):
    def test_tum_depth_scale_and_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            rgb = np.zeros((480, 640, 3), np.uint8)
            cv2.imwrite(str(path / "rgb.png"), rgb)
            cv2.imwrite(str(path / "depth.png"), np.full((480, 640), 5000, np.uint16))
            (path / "rgb.txt").write_text("# comment\n1.0 rgb.png\n")
            (path / "depth.txt").write_text("1.005 depth.png\n")
            (path / "groundtruth.txt").write_text("1.0 0.1 0.2 0.3 0 0 0 1\n")
            settings, frames = load_dataset("tum", path)
            self.assertEqual(1000, frames[0][1][0, 0])
            self.assertEqual(525, settings.camera.fx)
            np.testing.assert_allclose([0.1, 0.2, 0.3], frames[0][3][:3, 3])

    def test_recording_preserves_settings_and_reference_is_scoring_only(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = ScanSettings(near_m=0.7, far_m=2, roi=(100, 100, 540, 400))
            writer = RecordingWriter(Path(folder) / "scan", settings.to_dict())
            writer.append(
                np.zeros((480, 640, 3), np.uint8), np.full((480, 640), 1234, np.uint16)
            )
            writer.manifest["frames"][0]["reference_pose"] = np.eye(4).tolist()
            writer._save_manifest()
            loaded, frames = load_dataset("recording", writer.path)
            self.assertEqual(settings, loaded)
            self.assertEqual(1234, frames[0][1][0, 0])
            np.testing.assert_array_equal(np.eye(4), frames[0][3])

    def test_association_is_one_to_one_and_bounded(self):
        pairs = nearest_pairs(
            [(1.0, ["a"]), (1.01, ["b"]), (2.0, ["c"])], [(1.009, ["d"])]
        )
        self.assertEqual(1, len(pairs))
        self.assertEqual(1.01, pairs[0][0][0])

    def test_pose_scoring_removes_initial_world_frame_only(self):
        gt = [np.eye(4), np.eye(4)]
        gt[1][0, 3] = 0.1
        world = np.eye(4)
        world[1, 3] = 5
        frames = [(None, None, i, pose) for i, pose in enumerate(gt)]
        report = score([(i, world @ pose) for i, pose in enumerate(gt)], frames)
        self.assertLess(report["anchored_translation_rmse_m"], 1e-9)
        wrong = world @ gt[1]
        wrong[0, 3] = 0.2
        report = score([(0, world), (1, wrong)], frames)
        self.assertGreater(report["anchored_translation_rmse_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
