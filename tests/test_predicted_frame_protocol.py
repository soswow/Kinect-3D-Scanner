"""Prediction is reversible across pixel boundaries, depth codes and legacy packets."""

import unittest

import numpy as np

from shared.protocol import pack_frame, pack_frames, unpack_frame_with_metadata, unpack_frames


class PredictedFrameProtocolTests(unittest.TestCase):
    def test_random_extremes_and_noncontiguous_frames_round_trip_at_both_resolutions(self):
        rng = np.random.default_rng(7)
        for height, width in ((480, 640), (1024, 1280)):
            rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)[::-1]
            depth = rng.integers(0, 65536, (480, 640), dtype=np.uint16)[::-1]
            depth[0, :4] = [0, 65535, 2047, 0]
            metadata = {"frame_id": "capture", "depth_encoding": "raw_11bit"}
            for level in (0, 1):
                packet = pack_frame(rgb, depth, metadata, compression_level=level, spatial_prediction=True)
                color, decoded_depth, decoded_metadata = unpack_frame_with_metadata(packet)
                np.testing.assert_array_equal(rgb, color)
                np.testing.assert_array_equal(depth, decoded_depth)
                self.assertEqual({**metadata, "rgb_shape": [height, width, 3]}, decoded_metadata)
                self.assertEqual(metadata, {"frame_id": "capture", "depth_encoding": "raw_11bit"})
                with self.assertRaises(ValueError):
                    unpack_frame_with_metadata(packet[:-1])

    def test_prediction_reduces_smooth_image_payload_and_batch_is_lossless(self):
        x = np.arange(640, dtype=np.uint16)
        rgb = np.broadcast_to((x % 256).astype(np.uint8)[None, :, None], (480, 640, 3)).copy()
        depth = np.broadcast_to(x + 1000, (480, 640)).copy()
        plain = pack_frame(rgb, depth)
        predicted = pack_frame(rgb, depth, spatial_prediction=True)
        self.assertLess(len(predicted), len(plain) / 2)
        packet = pack_frames([(rgb, depth, {"frame_id": 1}), (rgb, depth, {"frame_id": 2})],
                             spatial_prediction=True)
        for index, (color, metric, metadata) in enumerate(unpack_frames(packet, with_metadata=True), 1):
            np.testing.assert_array_equal(rgb, color)
            np.testing.assert_array_equal(depth, metric)
            self.assertEqual(index, metadata["frame_id"])
        color, metric, _ = unpack_frame_with_metadata(plain)
        np.testing.assert_array_equal(rgb, color)
        np.testing.assert_array_equal(depth, metric)


if __name__ == "__main__":
    unittest.main()
