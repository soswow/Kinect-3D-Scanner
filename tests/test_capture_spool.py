"""Offline buffering must preserve images and keep Finish behind all uploads."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtCore import Qt

from kinect_scanner.capture_spool import CaptureSpool
from kinect_scanner.server_client import ServerClient
from kinect_scanner.server_task_worker import ServerTask, ServerTaskType as T, ServerTaskWorker


class CaptureSpoolTests(unittest.TestCase):
    def frame(self, index):
        return ServerTask(T.SEND_FRAME, {"rgb": np.full((8, 8, 3), index % 256, np.uint8),
                         "depth": np.full((8, 8), index, np.uint16),
                         "metadata": {"frame_id": index, "timestamp_s": index * .5}})

    def test_spool_round_trip_and_disk_failure_preserve_existing_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = CaptureSpool(directory, "scan", {"live_reconstruction": False})
            original = self.frame(7).kwargs
            path = spool.append(**original)
            rgb, depth, metadata = spool.load(path)
            np.testing.assert_array_equal(original["rgb"], rgb)
            np.testing.assert_array_equal(original["depth"], depth)
            self.assertEqual(original["metadata"], metadata)
            self.assertEqual("scan", json.loads((spool.path / "session.json").read_text())["session_id"])
            with patch("kinect_scanner.capture_spool.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, "insufficient disk space"):
                    spool.append(**self.frame(8).kwargs)
            self.assertEqual([path], list(spool.path.glob("*.npz")))
            self.assertEqual(1, spool.pending_count)
            spool.acknowledge(path)
            spool.cleanup()
            self.assertFalse(spool.path.exists())

    def test_slow_upload_does_not_limit_capture_and_finish_waits_for_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ServerClient()
            client.session_id = "scan"
            worker = ServerTaskWorker(client)
            worker._spool = CaptureSpool(directory, "scan", {})
            entered, release, built = threading.Event(), threading.Event(), threading.Event()
            uploaded = []

            def send(frames):
                entered.set()
                self.assertTrue(release.wait(10))
                ids = [metadata["frame_id"] for _, _, metadata in frames]
                uploaded.extend(ids)
                return {"success": True, "results": [{"success": True, "index": i} for i in ids]}

            client.send_frame = lambda *frame: send([frame])
            client.send_frames_batch = send
            def build():
                self.assertEqual(list(range(120)), uploaded)
                self.assertEqual(0, worker.pending_capture_count)
                return {"success": True}
            client.request_build = build
            worker._save_reconstruction = Mock()
            client.build_mesh_done.connect(lambda *_: built.set(), Qt.ConnectionType.DirectConnection)
            worker.start()
            try:
                self.assertTrue(worker.submit(self.frame(0)))
                self.assertTrue(entered.wait(2))
                for index in range(1, 120):
                    self.assertTrue(worker.submit(self.frame(index)))
                self.assertEqual(120, worker.pending_capture_count)
                self.assertTrue(all("rgb" not in task.kwargs for task in worker._queue.queue))
                worker.submit(ServerTask(T.BUILD_MESH))
                self.assertFalse(built.is_set())
                release.set()
                self.assertTrue(built.wait(10))
            finally:
                release.set()
                worker.stop()
                self.assertTrue(worker.wait(3000))

    def test_failed_upload_retains_file_and_blocks_incomplete_build_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ServerClient()
            client.send_frame = Mock(side_effect=OSError("network lost"))
            client.request_build = Mock()
            client.request_export = Mock()
            worker = ServerTaskWorker(client)
            worker._spool = CaptureSpool(directory, "scan", {})
            worker.submit(self.frame(1))
            with self.assertRaisesRegex(OSError, "network lost"):
                worker._dispatch(worker._queue.get_nowait())
            self.assertEqual(1, worker.pending_capture_count)
            for task in (ServerTask(T.BUILD_MESH), ServerTask(T.EXPORT_SESSION, {"path": "scan.zip"})):
                with self.assertRaisesRegex(RuntimeError, "retained"):
                    worker._dispatch(task)
            client.request_build.assert_not_called()
            client.request_export.assert_not_called()
            self.assertEqual(1, len(list(worker._spool.path.glob("*.npz"))))

    def test_partial_rejection_only_deletes_confirmed_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ServerClient()
            client.send_frames_batch = Mock(return_value={"success": True, "results": [
                {"success": True, "index": 0}, {"success": False}]})
            worker = ServerTaskWorker(client)
            worker._spool = CaptureSpool(directory, "scan", {})
            for index in range(2):
                worker.submit(self.frame(index))
            result = worker._drain_send_frames(worker._queue.get_nowait())
            self.assertFalse(result["success"])
            self.assertEqual(1, worker.pending_capture_count)
            retained = CaptureSpool.load(next(worker._spool.path.glob("*.npz")))
            self.assertEqual(1, retained[2]["frame_id"])


if __name__ == "__main__":
    unittest.main()
