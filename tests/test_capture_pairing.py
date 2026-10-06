"""Pair asynchronous RGB/depth by device time, including delayed callbacks."""

import unittest

import numpy as np

from shared.capture import RGBDepthPairer, timestamp_delta_ms


def ticks(ms):
    return round(ms * 60_000) % (1 << 32)


class CapturePairingTests(unittest.TestCase):
    def setUp(self):
        self.pairer = RGBDepthPairer()

    def depth(self, ms):
        self.pairer.add_depth(np.array([ms]), ticks(ms))

    def rgb(self, ms):
        self.pairer.add_rgb(np.array([ms]), ticks(ms))

    def test_delayed_rgb_uses_retained_depth_instead_of_latest(self):
        # The saved scan commonly paired RGB ~40 ms behind latest depth.
        for ms in (966.667, 1000, 1033.333):
            self.depth(ms)
        self.rgb(993.333)
        rgb, depth, rgb_stamp, depth_stamp = self.pairer.pop_pair()
        self.assertEqual([1000], depth.tolist())
        self.assertAlmostEqual(-6.667, timestamp_delta_ms(rgb_stamp, depth_stamp))
        self.assertIsNone(self.pairer.pop_pair())

    def test_rgb_ahead_waits_for_surrounding_depths_and_chooses_closest(self):
        self.depth(1000)
        self.rgb(1025)
        self.assertIsNone(self.pairer.pop_pair())
        self.depth(1033)
        self.assertEqual([1033], self.pairer.pop_pair()[1].tolist())

    def test_earlier_closer_depth_wins_and_newer_depth_is_retained(self):
        self.depth(1000)
        self.rgb(1002)
        self.depth(1033)
        self.assertEqual([1000], self.pairer.pop_pair()[1].tolist())
        self.rgb(1033)
        self.assertEqual([1033], self.pairer.pop_pair()[1].tolist())
        self.rgb(1033)
        self.assertIsNone(self.pairer.pop_pair())  # No depth reuse.

    def test_pairing_across_uint32_clock_wrap(self):
        wrap_ms = (1 << 32) / 60_000
        self.depth(wrap_ms - 15)
        self.rgb(wrap_ms - 1)
        self.assertIsNone(self.pairer.pop_pair())
        self.depth(wrap_ms + 18)
        pair = self.pairer.pop_pair()
        self.assertAlmostEqual(14, timestamp_delta_ms(pair[2], pair[3]), places=4)

    def test_callbacks_copy_reused_driver_arrays(self):
        rgb, depth = np.array([42]), np.array([750])
        self.pairer.add_depth(depth, ticks(1000))
        self.pairer.add_rgb(rgb, ticks(1000))
        rgb[:] = 99
        depth[:] = 999
        pair = self.pairer.pop_pair()
        self.assertEqual([42], pair[0].tolist())
        self.assertEqual([750], pair[1].tolist())

    def test_backpressure_is_bounded_and_publishes_newest_rgb(self):
        for ms in range(0, 1000, 33):
            self.depth(ms)
            self.rgb(ms)
        self.assertLessEqual(len(self.pairer.depths), 8)
        self.assertEqual([990], self.pairer.pop_pair()[0].tolist())
        self.pairer.clear()
        self.assertIsNone(self.pairer.pop_pair())
        self.assertFalse(self.pairer.depths)

    def test_missing_close_depth_never_hides_bad_timing(self):
        self.depth(1000)
        self.rgb(1060)
        self.depth(1120)
        self.assertIsNone(self.pairer.pop_pair())  # Both depths are 60 ms away.
        self.rgb(1120)
        self.assertEqual([1120], self.pairer.pop_pair()[1].tolist())

    def test_gap_over_assistance_limit_still_publishes_for_capture(self):
        self.rgb(1000)
        self.depth(1034)  # Missing the closer depth must remain visible in metadata.
        pair = self.pairer.pop_pair()
        self.assertEqual(-34, timestamp_delta_ms(pair[2], pair[3]))

    def test_both_rgb_rates_delivered_40ms_late_pair_with_30fps_depth(self):
        for fps in (10, 30):
            with self.subTest(fps=fps):
                self.pairer.clear()
                events = [(i * 1000 / 30, "depth", i * 1000 / 30) for i in range(64)]
                events += [(i * 1000 / fps + 51, "rgb", i * 1000 / fps + 11) for i in range(20)]
                pairs = []
                for arrival, stream, stamp in sorted(events):
                    (self.depth if stream == "depth" else self.rgb)(stamp)
                    pair = self.pairer.pop_pair()
                    if pair is not None:
                        pairs.append(pair)
                self.assertEqual(20, len(pairs))
                self.assertLessEqual(max(abs(timestamp_delta_ms(p[2], p[3])) for p in pairs), 17)
                self.assertEqual(20, len({p[3] for p in pairs}))


if __name__ == "__main__":
    unittest.main()
