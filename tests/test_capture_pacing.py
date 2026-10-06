"""Pacing must bound latency without exceeding the user's capture frequency."""

import unittest

from kinect_scanner.capture_pacing import CapturePacer


class CapturePacingTests(unittest.TestCase):
    def test_minimum_delay_and_camera_frame_rounding(self):
        pacer = CapturePacer()
        pacer.captured("first", 10, live=False)
        self.assertFalse(pacer.ready(10.999, 1, 30))
        self.assertTrue(pacer.ready(11, 1, 30))
        pacer.observe({"processed_count": 1, "processing_interval_s": 1.61}, 12)
        self.assertEqual(1.9, pacer.interval_seconds(1, 10))
        self.assertGreaterEqual(pacer.interval_seconds(1, 30), 1.61 * 1.15)
        self.assertEqual(3, pacer.interval_seconds(3, 10))
        self.assertEqual(1, pacer.interval_seconds(1, 10, adaptive=False))

    def test_speed_recovers_gradually_and_duplicate_feedback_does_not_accelerate(self):
        pacer = CapturePacer()
        slow = {"processed_count": 1, "processing_interval_s": 2}
        pacer.observe(slow, 2)
        self.assertEqual(2.3, pacer.interval_seconds(1, 10))
        fast = {"processed_count": 2, "processing_interval_s": 0.1}
        pacer.observe(fast, 3)
        recovering = pacer.interval_seconds(1, 10)
        self.assertGreater(recovering, 1)
        self.assertLess(recovering, 2.3)
        for _ in range(100):
            pacer.observe(fast, 3)
        self.assertEqual(recovering, pacer.interval_seconds(1, 10))
        for count in range(3, 40):
            pacer.observe({**fast, "processed_count": count}, count)
        self.assertEqual(1, pacer.interval_seconds(1, 10))

    def test_upload_acknowledgement_does_not_release_processing_slot(self):
        pacer = CapturePacer()
        pacer.captured("first", 0, live=True)
        pacer.acknowledge([{"frame_id": "first", "success": True, "index": 0}], 0.2)
        self.assertEqual(1, pacer.pending_count)
        self.assertEqual(1, pacer.outstanding_count)
        pacer.observe({"processed_count": 1}, 2)
        self.assertEqual(0, pacer.pending_count)
        self.assertEqual(0, pacer.outstanding_count)
        # Full capture-to-feedback latency accounts for upload and publication.
        self.assertEqual(2.3, pacer.interval_seconds(1, 10))

    def test_live_result_before_http_ack_releases_all_prior_local_captures(self):
        pacer = CapturePacer()
        pacer.captured("first", 0, live=True)
        pacer.captured("second", 1, live=True)
        pacer.observe({"processed_count": 2, "result": {
            "metadata": {"frame_id": "second"}, "success": False,
        }}, 3)
        self.assertEqual(0, pacer.pending_count)
        interval = pacer.interval_seconds(1, 10)
        pacer.acknowledge([
            {"frame_id": "first", "success": True, "index": 0},
            {"frame_id": "second", "success": True, "index": 1},
        ], 10)
        self.assertEqual(0, pacer.pending_count)
        self.assertEqual(interval, pacer.interval_seconds(1, 10))

    def test_feedback_before_ack_from_another_client_uses_server_indices(self):
        pacer = CapturePacer()
        pacer.captured("local", 0, live=True)
        pacer.observe({"processed_count": 3, "result": {
            "metadata": {"frame_id": "remote"},
        }}, 2)
        self.assertEqual(1, pacer.pending_count)
        pacer.acknowledge([{"frame_id": "local", "success": True, "index": 1}], 2)
        self.assertEqual(0, pacer.pending_count)

    def test_rejected_uploads_and_new_sessions_release_slots(self):
        pacer = CapturePacer()
        pacer.captured("rejected", 0, live=True)
        pacer.acknowledge([{"frame_id": "rejected", "success": False}], 1)
        self.assertEqual(0, pacer.pending_count)
        pacer.captured("old", 2, live=True)
        pacer.observe({"processed_count": 1, "processing_interval_s": 5}, 2)
        pacer.reset()
        self.assertEqual(0, pacer.pending_count)
        self.assertEqual(1, pacer.interval_seconds(1, 10))
        self.assertTrue(pacer.ready(2, 1, 10))

    def test_older_server_timing_and_missing_timing(self):
        pacer = CapturePacer()
        pacer.observe({"result": {"index": 0, "elapsed_ms": 2000}}, 2)
        self.assertEqual(2.3, pacer.interval_seconds(1, 10))
        pacer.observe({"processed_count": 2}, 3)
        self.assertEqual(2.3, pacer.interval_seconds(1, 10))
        pacer.observe({"processed_count": 3, "processing_interval_s": float("nan")}, 4)
        self.assertEqual(2.3, pacer.interval_seconds(1, 10))

    def test_restored_server_backlog_and_new_upload_both_count(self):
        pacer = CapturePacer()
        pacer.observe({"stored_count": 111, "unprocessed_count": 1}, 10)
        self.assertEqual(1, pacer.outstanding_count)
        pacer.captured("local", 10, live=True)
        self.assertEqual(2, pacer.outstanding_count)
        pacer.acknowledge([{"frame_id": "local", "success": True, "index": 111}], 11)
        self.assertEqual(2, pacer.outstanding_count)
        pacer.observe({"processed_count": 112, "stored_count": 112}, 12)
        self.assertEqual(0, pacer.outstanding_count)


if __name__ == "__main__":
    unittest.main()
