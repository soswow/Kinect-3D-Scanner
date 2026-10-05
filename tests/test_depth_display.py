"""Depth preview semantics, crop indication, and frame interval labels."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from kinect_scanner.gui.widgets import colorize_depth, depth_legend_text, FrameIntervalSpinBox


class DepthDisplayTests(unittest.TestCase):
    def test_range_is_inclusive_and_excluded_differs_from_missing(self):
        rgb = colorize_depth(np.array([[0, 499, 500, 1000, 1500, 1501]], dtype=np.uint16), 500, 1500)[0]
        np.testing.assert_array_equal(rgb[0], [0, 0, 0])
        np.testing.assert_array_equal(rgb[1], [42, 42, 42])
        np.testing.assert_array_equal(rgb[5], [42, 42, 42])
        np.testing.assert_array_equal(rgb[2], [0, 0, 128])
        np.testing.assert_array_equal(rgb[4], [128, 0, 0])

    def test_roi_outside_is_excluded_and_border_matches_exclusive_bounds(self):
        rgb = colorize_depth(np.full((8, 8), 1000, dtype=np.uint16), 500, 1500, (2, 2, 6, 6))
        np.testing.assert_array_equal(rgb[0, 0], [42, 42, 42])
        for y, x in [(2, 2), (5, 5), (2, 4), (4, 5)]:
            np.testing.assert_array_equal(rgb[y, x], [255, 255, 255])
        self.assertFalse(np.array_equal(rgb[3, 3], [255, 255, 255]))
        np.testing.assert_array_equal(rgb[6, 6], [42, 42, 42])
        legend = depth_legend_text(500, 1500)
        self.assertIn("500 mm", legend)
        self.assertIn("1500 mm", legend)
        self.assertIn("no depth", legend)
        self.assertIn("excluded", legend)

    def test_interval_label_is_short_and_cadence_stays_whole_frames(self):
        app = QApplication.instance() or QApplication([])
        widget = FrameIntervalSpinBox(fps=10)
        self.assertEqual(widget.text(), "0.5 s")
        widget.set_interval_seconds(0.34)
        self.assertEqual(widget.value(), 3)
        self.assertEqual(widget.interval_seconds, 0.3)
        self.assertEqual(widget.text(), "0.3 s")
        widget.set_interval_seconds(1)
        self.assertEqual(widget.text(), "1 s")
        widget.close()


if __name__ == "__main__":
    unittest.main()
