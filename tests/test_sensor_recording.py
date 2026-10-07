"""Full stream retention, checkpoint boundaries, replay, and transactional saves."""

import json
import io
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from kinect_scanner.server_client import ServerClient
from kinect_scanner.server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker
from scanner_server.session import export_session
from shared.inertial import G, rotate_display
from shared.sensor_recording import SensorJournal, augment_session_archive, journal_snapshot
from shared.sensor_replay import load_sensor_observations
from shared.settings import ScanSettings


def frame_metadata(sequence, stamp, generation="connection"):
    return {"sequence": sequence, "capture_generation": generation, "timestamp_s": 100 + stamp,
            "device_timestamp_ticks": int(stamp * 60_000_000),
            "device_timestamp_unwrapped_s": stamp,
            "estimated_host_monotonic_s": stamp, "host_receipt_monotonic_s": stamp + .003}


def acceleration(sequence, stamp, valid=True):
    return {"sequence": sequence, "capture_generation": "connection", "valid": valid,
            "host_monotonic_s": stamp, "read_start_s": stamp - .001,
            "read_end_s": stamp + .001, "raw_counts": [0, 819, 0],
            "acceleration_m_s2": [0, G, 0] if valid else None,
            "reason": "test error" if not valid else "test"}


def checkpoint(journal):
    deadline = time.monotonic() + 3
    while not journal.request_flush("test"):
        if time.monotonic() >= deadline:
            raise RuntimeError("Writer did not accept checkpoint")
        time.sleep(.01)
    return journal.notifications.get(timeout=3)


class SensorRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rgb = np.full((480, 640, 3), [24, 54, 94], np.uint8)
        self.depth = (np.arange(480 * 640).reshape(480, 640) % 2048).astype(np.uint16)

    def journal(self, generation="segment", capacity=32):
        journal = SensorJournal(self.root / "raw", generation, ScanSettings().to_dict(), capacity=capacity,
                                capture_generation="connection")
        self.addCleanup(journal.close, 3)
        return journal

    def server_zip(self, destination=None):
        destination = destination or self.root / "session.zip"
        engine = SimpleNamespace(settings=ScanSettings(), raw_frames=[(self.rgb, self.depth)],
                                 frame_metadata=[{"timestamp_s": 1, "orientation": {"rotation_cw_degrees": 90}}],
                                 reconstruction_report=lambda: {"version": 1, "frames": []})
        export_session(engine, destination)
        return destination

    def test_retains_independent_streams_and_failed_reads_not_only_selected_images(self):
        journal = self.journal()
        for sequence in range(4):
            journal.submit("depth", frame_metadata(sequence, 1 + sequence / 30), self.depth)
        for sequence in range(2):
            journal.submit("rgb", frame_metadata(sequence, 1 + sequence / 10), self.rgb)
        for sequence in range(10):
            journal.submit("accelerometer", acceleration(sequence, .8 + sequence * .05, valid=sequence != 3))
        response = checkpoint(journal)
        snapshot = journal_snapshot(journal.path.parent, response)
        path = self.server_zip()
        augment_session_archive(path, snapshot)
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
            archive.extractall(self.root / "saved")
            manifest = json.loads(archive.read("manifest.json"))
            self.assertTrue(manifest["sensor_archive"]["complete"])
            self.assertEqual(len(manifest["frames"]), 1)
            rows = archive.read("sensors/segment/accelerometer.jsonl").splitlines()
            self.assertEqual(len(rows), 10)
            self.assertFalse(json.loads(rows[3])["valid"])
        saved = self.root / "saved"
        np.testing.assert_array_equal(np.asarray(Image.open(saved / "rgb/000000.png")), self.rgb)
        np.testing.assert_array_equal(np.asarray(Image.open(saved / "depth/000000.png")), self.depth)
        np.testing.assert_array_equal(np.asarray(Image.open(saved / "display/depth/000000.png")), rotate_display(self.depth, 90))
        observations = list(load_sensor_observations(saved, ScanSettings()))
        self.assertEqual(len(observations), 2)
        self.assertTrue(any(frame[2]["accelerometer"]["valid"] for frame in observations))
        self.assertNotIn("visual_tracking", observations[0][2])

    def test_checkpoint_is_a_stable_prefix_and_copies_borrowed_arrays(self):
        journal = self.journal()
        original = self.rgb.copy()
        journal.submit("rgb", frame_metadata(0, 1), self.rgb)
        self.rgb[:] = 0
        response = checkpoint(journal)
        snapshot = journal_snapshot(journal.path.parent, response)
        journal.submit("rgb", frame_metadata(1, 2), self.rgb)
        journal.close(3)
        path = self.server_zip()
        augment_session_archive(path, snapshot)
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(len(archive.read("sensors/segment/rgb.jsonl").splitlines()), 1)
            self.assertNotIn("sensors/segment/rgb/000000001.npy", archive.namelist())
            image = np.load(io.BytesIO(archive.read("sensors/segment/rgb/000000000.npy")), allow_pickle=False)
            np.testing.assert_array_equal(image, original)

    def test_raw_replay_recomputes_every_intermediate_motion_before_stride(self):
        from scripts.replay_scan import load_dataset
        journal = self.journal()
        journal.submit("events", {"host_monotonic_s": 1.05, "type": "orientation_mode", "value": "portrait_right"})
        for i in range(5):
            stamp = 1 + i / 10
            journal.submit("rgb", frame_metadata(i, stamp), self.rgb)
            journal.submit("depth", {**frame_metadata(i, stamp), "host_mapping_uncertainty_s": .1}, self.depth)
        for i in range(15):
            journal.submit("accelerometer", acceleration(i, .8 + i * .05))
        path = self.server_zip()
        augment_session_archive(path, journal_snapshot(journal.path.parent, checkpoint(journal)))
        with zipfile.ZipFile(path) as archive:
            archive.extractall(self.root / "replay")
        with patch("shared.visual_tracking.VisualTracker") as factory:
            factory.return_value.update.return_value = {"valid": False, "reason": "fixture"}
            settings, frames = load_dataset("recording", self.root / "replay", stride=2,
                                            sensor_streams=True, recompute_motion=True)
        self.assertEqual(factory.return_value.update.call_count, 5)
        self.assertEqual(len(frames), 3)
        self.assertFalse(frames[-1].metadata["accelerometer"]["valid"])
        self.assertEqual(frames[-1].metadata["orientation"]["rotation_cw_degrees"], 270)

    def test_overflow_and_disk_failure_are_explicit(self):
        journal = self.journal(capacity=1)
        entered, release = threading.Event(), threading.Event()
        native = np.save

        def blocked(*args, **kwargs):
            entered.set()
            release.wait(3)
            return native(*args, **kwargs)

        with patch("shared.sensor_recording.np.save", side_effect=blocked):
            self.assertTrue(journal.submit("rgb", frame_metadata(0, 1), self.rgb))
            self.assertTrue(entered.wait(2))
            self.assertTrue(journal.submit("depth", frame_metadata(0, 1), self.depth))
            self.assertFalse(journal.submit("accelerometer", acceleration(0, 1)))
            release.set()
            response = checkpoint(journal)
        self.assertEqual(response["status"]["dropped"]["accelerometer"], 1)
        self.assertFalse(response["status"]["complete"])

        failed = self.journal("failed")
        with patch("shared.sensor_recording.np.save", side_effect=OSError("disk full")):
            failed.submit("rgb", frame_metadata(0, 1), self.rgb)
            response = checkpoint(failed)
        self.assertFalse(response["status"]["complete"])
        self.assertIn("disk full", response["status"]["error"])

    def test_missing_sensor_file_does_not_replace_existing_archive(self):
        journal = self.journal()
        journal.submit("rgb", frame_metadata(0, 1), self.rgb)
        snapshot = journal_snapshot(journal.path.parent, checkpoint(journal))
        (journal.path / "rgb/000000000.npy").unlink()
        path = self.server_zip()
        previous = path.read_bytes()
        with self.assertRaises(FileNotFoundError):
            augment_session_archive(path, snapshot)
        self.assertEqual(path.read_bytes(), previous)
        self.assertFalse(path.with_name(path.name + ".sensors.tmp").exists())

    def test_unclean_segment_is_reported_and_segments_keep_capture_order(self):
        first = self.journal("z-first")
        checkpoint(first)
        second = self.journal("a-second")
        snapshot = journal_snapshot(first.path.parent, checkpoint(second))
        self.assertEqual([s["generation"] for s in snapshot["segments"]], ["z-first", "a-second"])
        self.assertFalse(snapshot["complete"])
        self.assertIn("stopped before", snapshot["segments"][0]["status"]["error"])

    def test_export_augmentation_failure_preserves_user_destination(self):
        destination = self.root / "important.zip"
        destination.write_bytes(b"existing complete user archive")
        client = ServerClient()
        worker = ServerTaskWorker(client)

        def download(fmt, path, options=None):
            self.server_zip(Path(path))
            return True

        client.request_export = download
        recorder = Mock()
        recorder.flush_sensor_recording.return_value = {"root": str(self.root / "missing"), "segments": [{"generation": "missing", "status": {"index_bytes": {}}}], "complete": False}
        with self.assertRaises(FileNotFoundError):
            worker._dispatch(ServerTask(ServerTaskType.EXPORT_SESSION, {"path": str(destination), "sensor_recorder": recorder}))
        self.assertEqual(destination.read_bytes(), b"existing complete user archive")
        self.assertEqual(list(self.root.glob(".sensor-session-*")), [])
