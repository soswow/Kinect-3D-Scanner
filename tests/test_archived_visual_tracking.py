"""Keep sparse capture replay separate from controlled stationary evidence."""

import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

from scripts.evaluate_archived_visual_tracking import evaluate, input_summary, measured_motion
from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker


class ArchivedVisualTrackingTests(unittest.TestCase):
    def test_bootstrap_is_not_measured_motion(self):
        self.assertFalse(measured_motion({"valid": True, "steps": 0}))
        self.assertFalse(measured_motion({"valid": False, "reference_timestamp_s": 1}))
        self.assertTrue(measured_motion({"valid": True, "reference_timestamp_s": 0}))

    def test_input_summary_preserves_gap_and_timing_guards(self):
        manifest = {"frames": [
            {"timestamp_s": 0, "metadata": {"rgb_depth_delta_ms": -20}},
            {"timestamp_s": .1, "metadata": {"rgb_depth_delta_ms": 20.1}},
            {"timestamp_s": 2, "metadata": {}},
        ]}
        result = input_summary(manifest)
        self.assertEqual(result["adjacent_pairs_within_tracking_gap"], 1)
        self.assertEqual(result["timing_eligible_frames"], 2)
        self.assertEqual(result["unknown_rgb_depth_timing_frames"], 1)

    def test_sparse_replay_cannot_claim_stationary_motion_as_live_motion(self):
        rng = np.random.default_rng(8)
        rgb = cv2.resize(rng.integers(0, 256, (120, 160, 3), np.uint8), (640, 480))
        depth = np.full((480, 640), 2000, np.uint16)
        detector = cv2.goodFeaturesToTrack
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.zip"
            manifest = {"settings": ScanSettings(filter_depth=False).to_dict(), "frames": []}
            with zipfile.ZipFile(path, "w") as archive:
                for index in range(3):
                    entry = {"rgb": f"rgb/{index}.png", "depth": f"depth/{index}.png",
                             "timestamp_s": index * 2,
                             "metadata": {"rgb_depth_delta_ms": 0 if index < 2 else 25}}
                    manifest["frames"].append(entry)
                    archive.writestr(entry["rgb"], cv2.imencode(".png", rgb)[1].tobytes())
                    archive.writestr(entry["depth"], cv2.imencode(".png", depth)[1].tobytes())
                archive.writestr("manifest.json", json.dumps(manifest))
            details = io.StringIO()
            result = evaluate(path, {"after": VisualTracker}, 2, details)
            actual = result["recorded_time"]["after"]
            stationary = result["stationary_diagnostic"]["after"]
            self.assertEqual(actual["measured_motion_updates"], 0)
            self.assertEqual(actual["seeded_reference_updates"], 2)
            self.assertEqual(actual["reasons"]["RGB/depth timing exceeds 20 ms"], 1)
            self.assertEqual(stationary["measured_motion_updates"], 6)
            self.assertEqual(stationary["seeded_reference_updates"], 3)
            self.assertLess(stationary["zero_motion_translation_error_mm"]["max"], .001)
            self.assertEqual(stationary["stationary_seed_identity_survival_fraction"]["min"], 1)
            self.assertAlmostEqual(stationary["stationary_end_median_feature_age_s"]["min"], .2)
            self.assertEqual(len(details.getvalue().splitlines()), 12)
        self.assertIs(cv2.goodFeaturesToTrack, detector)


if __name__ == "__main__":
    unittest.main()
