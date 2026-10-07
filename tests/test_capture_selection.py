"""Freshness, bounded storage and actual sharp/blurred image selection."""

import unittest

import cv2
import numpy as np

from kinect_scanner.capture_selection import CaptureSelector, sharpness


class CaptureSelectionTests(unittest.TestCase):
    def test_sharp_pair_wins_and_keeps_its_own_metadata_and_depth(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(image, (15, 15), 4)
        self.assertGreater(sharpness(image), 10 * sharpness(blurred))
        selector = CaptureSelector()
        selector.offer(image, np.full((2, 2), 1000), {"frame_id": "sharp"}, 1)
        selector.offer(blurred, np.full((2, 2), 2000), {"frame_id": "blurred"}, 1.1)
        rgb, depth, metadata = selector.choose(1.1)
        self.assertEqual("sharp", metadata["frame_id"])
        np.testing.assert_array_equal(image, rgb)
        self.assertTrue((depth == 1000).all())
        self.assertEqual(2, metadata["capture_selection"]["candidates"])
        self.assertIsNone(selector.choose(1.401))
        self.assertIsNone(selector.choose(1.1, after=1.1))

    def test_newest_wins_ties_and_window_is_bounded(self):
        selector = CaptureSelector()
        image = np.full((240, 320, 3), 128, np.uint8)
        for i in range(10):
            selector.offer(image, np.ones((2, 2)), {"frame_id": i}, i * 0.01)
        self.assertEqual(5, len(selector.frames))
        self.assertEqual(9, selector.choose(0.1)[2]["frame_id"])
        selector.clear()
        self.assertIsNone(selector.choose(0.1))

    def test_verified_motion_candidate_wins_over_a_broken_chain(self):
        selector = CaptureSelector()
        image = np.full((240, 320, 3), 128, np.uint8)
        selector.offer(image, np.ones((2, 2)), {"frame_id": 1, "visual_tracking": {"valid": True}}, 1)
        selector.offer(image, np.ones((2, 2)), {"frame_id": 2, "visual_tracking": {"valid": False}}, 1.1)
        self.assertEqual(1, selector.choose(1.1)[2]["frame_id"])
