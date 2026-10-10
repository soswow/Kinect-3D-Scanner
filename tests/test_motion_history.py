import unittest
import numpy as np

from shared.capture import DeviceClockMapper
from shared.motion_history import MotionHistory, capture_interval
from shared.protocol import pack_frame, unpack_frame_with_metadata, MAX_METADATA_BYTES
from kinect_scanner.capture_selection import MotionCapturePolicy
from scanner_server.motion_evidence import MotionEvidence, visual_seed
from shared.settings import CameraCalibration


def observation(stamp, index, up=(0., -1., 0.)):
    return {"capture_generation": "camera", "sequence": index, "valid": True,
            "host_monotonic_s": stamp, "read_start_s": stamp - .001, "read_end_s": stamp + .001,
            "acceleration_m_s2": (np.array(up) * 9.80665).tolist(), "raw_counts": [0, -819, 0],
            "gravity": {"valid": True, "confidence": .9, "up_camera": list(up),
                        "calibration_id": "measured", "calibration_verified": True}}


def frame(stamp, pose=None, valid=True, segment="chain"):
    return {"captured_monotonic_s": stamp, "depth_host_monotonic_s": stamp,
            "capture_generation": "camera", "host_mapping_uncertainty_s": .01,
            "visual_tracking": {"valid": valid, "segment": segment,
                                "camera_to_local": (np.eye(4) if pose is None else pose).tolist()}}


class MotionHistoryTests(unittest.TestCase):
    def test_different_stream_delivery_delays_do_not_disable_depth_association(self):
        clock = DeviceClockMapper()
        for i in range(200):
            depth = clock.observe(round(i * .1 * 60_000_000) % 2**32, 10 + i * .1 + (i % 3) * .001, "depth")
            color = clock.observe(round((i * .1 - .01) * 60_000_000) % 2**32, 10 + i * .1 + .07, "rgb")
        self.assertLess(depth["host_mapping_uncertainty_s"], .003)
        self.assertAlmostEqual(.01, depth["estimated_host_monotonic_s"] - color["estimated_host_monotonic_s"], places=6)
        self.assertGreater(color["stream_delivery_delay_floor_s"], .07)
        delayed = clock.observe(round(20 * 60_000_000), 30.1, "depth")
        self.assertGreater(delayed["host_mapping_uncertainty_s"], .09)

    def test_all_between_capture_reads_including_failures_survive_wire_transport(self):
        history = MotionHistory()
        history.add_frame(frame(0.))
        for i in range(1, 21):
            read = observation(i / 20, i)
            if i == 10:
                read.update(valid=False, reason="USB read failed")
            history.add_acceleration(read)
            history.add_frame(frame(i / 20))
        metadata = {**frame(1.), "motion_history": capture_interval(history.snapshot(1.), 0.)}
        self.assertTrue(metadata["motion_history"]["complete_interval"])
        rgb, depth = np.zeros((480, 640, 3), np.uint8), np.full((480, 640), 2000, np.uint16)
        _, _, restored = unpack_frame_with_metadata(pack_frame(rgb, depth, metadata, spatial_prediction=True))
        self.assertEqual(list(range(1, 21)), [s["sequence"] for s in restored["motion_history"]["accelerometer"]])
        self.assertFalse(restored["motion_history"]["accelerometer"][9]["valid"])
        self.assertEqual(20, len(restored["motion_history"]["visual"]))

    def test_preview_coalescing_retains_history_and_long_gaps_are_explicit(self):
        history = MotionHistory()
        for i in range(300):
            stamp = i / 20
            history.add_acceleration(observation(stamp, i))
            history.add_frame(frame(stamp))
        recent = capture_interval(history.snapshot(14.95), 13.95)
        self.assertTrue(recent["complete_interval"])
        self.assertEqual(20, len(recent["accelerometer"]))
        self.assertFalse(capture_interval(history.snapshot(14.95), 0.)["complete_interval"])
        self.assertLessEqual(len(history.acceleration), history.MAX_ACCELERATION)

    def test_metadata_remains_bounded(self):
        rgb, depth = np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.uint16)
        with self.assertRaises(ValueError):
            pack_frame(rgb, depth, {"motion_history": "x" * MAX_METADATA_BYTES})

    def test_compact_features_preserve_spatial_coverage_and_longest_identity_in_each_cell(self):
        history = MotionHistory()
        pixels = [[col*80+20+n, row*80+20] for row in range(6) for col in range(8) for n in range(5)]
        count = len(pixels)
        measured = {"ids": list(range(count)), "pixels": pixels, "depths_m": [2.]*count,
                    "observations": list(range(count)), "version": 1, "image_size": [640, 480]}
        metadata = frame(0.)
        metadata["visual_tracking"]["measured_tracks"] = measured
        history.add_frame(metadata)
        kept = history.snapshot(0.)["visual"][0]["feature_observations"]
        self.assertEqual(192, len(kept["ids"]))
        self.assertEqual(48, len({int(x//80)+8*int(y//80) for x, y in kept["pixels"]}))
        self.assertEqual({cell*5+n for cell in range(48) for n in (1, 2, 3, 4)}, set(kept["ids"]))
        self.assertEqual([2.]*192, kept["depths_m"])

    def test_peak_feature_history_fits_high_resolution_wire_packet_and_reports_truncation(self):
        history = MotionHistory()
        rng = np.random.default_rng(8)
        measured = {"version": 1, "image_size": [640, 480], "ids": list(range(40_000, 40_500)),
                    "pixels": np.round(rng.uniform([1., 1.], [638., 478.], (500, 2)), 3).tolist(),
                    "depths_m": np.round(rng.uniform(.5, 4., 500), 6).tolist(),
                    "observations": list(range(500))}
        for i in range(256):
            stamp = i/32
            history.add_acceleration(observation(stamp, i))
            metadata = frame(stamp)
            metadata["visual_tracking"]["measured_tracks"] = measured
            history.add_frame(metadata)
        snapshot = history.snapshot(255/32)
        self.assertEqual(80, sum(bool(row.get("feature_observations")) for row in snapshot["visual"]))
        metadata = {**frame(255/32), "motion_history": capture_interval(snapshot, 0.)}
        self.assertFalse(metadata["motion_history"]["complete_feature_interval"])
        rgb = np.zeros((1024, 1280, 3), np.uint8)
        depth = np.full((480, 640), 2000, np.uint16)
        packet = pack_frame(rgb, depth, metadata, compression_level=0)
        _, _, restored = unpack_frame_with_metadata(packet)
        self.assertEqual(80, sum(bool(row.get("feature_observations")) for row in restored["motion_history"]["visual"]))

    def test_motion_retains_overlap_and_reports_chain_breaks(self):
        policy = MotionCapturePolicy()
        policy.captured(frame(0.))
        moved = np.eye(4); moved[0, 3] = .12
        self.assertFalse(policy.needed(frame(.1, moved)))
        self.assertTrue(policy.needed(frame(.3, moved)))
        self.assertTrue(policy.needed(frame(.3, valid=False)))
        self.assertTrue(policy.needed(frame(.3, segment="new_origin")))
        self.assertFalse(policy.needed(frame(.3)))

    def test_gravity_rejects_inversion_but_preserves_yaw_and_missing_data(self):
        rows = [frame(0.), frame(1.)]
        for i, row in enumerate(rows):
            row["motion_history"] = {"accelerometer": [observation(float(i), i)]}
        evidence = MotionEvidence(rows, CameraCalibration())
        inverted = np.diag([1., -1., -1., 1.])
        self.assertFalse(evidence.check(1, 0, inverted)["accepted"])
        yaw = np.diag([-1., 1., -1., 1.])
        self.assertTrue(evidence.check(1, 0, yaw)["accepted"])
        rows[1]["motion_history"]["accelerometer"][0]["valid"] = False
        self.assertTrue(MotionEvidence(rows, CameraCalibration()).check(1, 0, inverted)["accepted"])

    def test_visual_motion_is_a_relative_initializer_and_cannot_cross_origins(self):
        moved = np.eye(4); moved[0, 3] = .2
        a, b = frame(1., moved), frame(0.)
        np.testing.assert_allclose(visual_seed(a, b), moved)
        a["visual_tracking"]["segment"] = "new"
        self.assertIsNone(visual_seed(a, b))

    def test_observed_track_identities_survive_capture_and_use_server_depth(self):
        from shared.visual_tracking import VisualTracker
        from shared.settings import ScanSettings
        from scanner_server.motion_evidence import extract_tracks
        from tests.test_visual_tracking import textured_plane
        settings = ScanSettings(color_recovery=True)
        tracker = VisualTracker(settings)
        rows, tracks = [], []
        for i, shift in enumerate((0, 4)):
            rgb, depth = textured_plane(shift=shift)
            metadata = {**frame(i*.1), "timestamp_s": i*.1}
            metadata["visual_tracking"] = tracker.update(rgb, depth, metadata)
            _, _, metadata = unpack_frame_with_metadata(pack_frame(rgb, depth, metadata))
            rows.append(metadata)
            tracks.append(extract_tracks(metadata, depth, settings.camera))
        evidence = MotionEvidence(rows, settings.camera, tracked=tracks, gravity=False)
        pair = evidence.appearance_pair(1, 0)
        self.assertIsNotNone(pair)
        self.assertEqual("tracked_rgbd_identities", evidence.features_for_pair(1, 0)[2])
        self.assertAlmostEqual(4*2/525, pair[0][0, 3], delta=.003)
        self.assertTrue(evidence.check(1, 0, pair[0])["accepted"])
        rows[0]["visual_tracking"]["valid"] = False
        self.assertIsNotNone(extract_tracks(rows[0], depth, settings.camera))
        self.assertIsNone(visual_seed(rows[1], rows[0]))
        self.assertIsNotNone(MotionEvidence(rows, settings.camera, tracked=tracks).appearance_pair(1, 0))
        wrong = pair[0].copy(); wrong[0, 3] += .05
        self.assertFalse(evidence.check(1, 0, wrong)["accepted"])
        history = MotionHistory(); history.add_frame(rows[1])
        self.assertNotIn("measured_tracks", history.visual[0]["visual_tracking"])
        # Identities from another tracker origin/generation cannot be reused.
        rows[1]["capture_generation"] = "another_camera"
        self.assertIsNone(MotionEvidence(rows, settings.camera, tracked=tracks).appearance_pair(1, 0))
        bad = tracker.update(np.zeros_like(rgb), depth, {"timestamp_s": .2})
        self.assertFalse(bad["valid"])
        self.assertIsNone(bad["measured_tracks"])

    def test_malformed_or_unsupported_tracks_are_not_pose_evidence(self):
        from scanner_server.motion_evidence import extract_tracks
        camera = CameraCalibration()
        metadata = frame(0.)
        metadata["visual_tracking"]["measured_tracks"] = {
            "version": 1, "image_size": [640, 480], "ids": list(range(40)),
            "pixels": [[100 + i*5, 100] for i in range(40)]}
        depth = np.full((480, 640), 2000, np.uint16)
        self.assertIsNotNone(extract_tracks(metadata, depth, camera))
        self.assertIsNone(extract_tracks(metadata, np.zeros_like(depth), camera))
        metadata["visual_tracking"]["measured_tracks"]["ids"][0] = 1
        self.assertIsNone(extract_tracks(metadata, depth, camera))

    def test_track_identity_remains_measured_through_depth_noise_and_motion(self):
        from shared.visual_tracking import VisualTracker
        from shared.settings import ScanSettings
        from tests.test_visual_tracking import textured_plane
        tracker = VisualTracker(ScanSettings())
        rng = np.random.default_rng(19)
        first = None
        for i in range(12):
            rgb, depth = textured_plane(shift=i*2)
            depth = (depth.astype(np.int32) + rng.integers(-8, 9, depth.shape)).astype(np.uint16)
            report = tracker.update(rgb, depth, {"timestamp_s": i*.1})
            self.assertTrue(report["valid"], report["reason"])
            ids = set(report["measured_tracks"]["ids"])
            first = ids if first is None else first
        self.assertGreaterEqual(len(first & ids), 40)
        self.assertAlmostEqual(22*2/525, tracker.pose[0, 3], delta=.004)


if __name__ == "__main__":
    unittest.main()
