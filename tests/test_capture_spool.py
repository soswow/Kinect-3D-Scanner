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
            self.assertTrue((spool.path / "session.json").exists())
            reopened = CaptureSpool.open(spool.path)
            self.assertEqual(1, reopened.next_sequence)

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
            client.session_id = "scan"
            client.send_frame = Mock(side_effect=OSError("network lost"))
            client.request_build = Mock()
            client.request_export = Mock()
            worker = ServerTaskWorker(client)
            worker._spool = CaptureSpool(directory, "scan", {})
            worker.submit(self.frame(1))
            with self.assertRaisesRegex(OSError, "network lost"):
                worker._dispatch(worker._queue.get_nowait())
            self.assertEqual(1, worker.pending_capture_count)
            for task in (ServerTask(T.BUILD_MESH), ServerTask(T.EXPORT_SESSION, {"path": "scan.zip"}),
                         ServerTask(T.EXPORT_BLENDER, {"path": "scan.blend"})):
                with self.assertRaisesRegex(RuntimeError, "retained"):
                    worker._dispatch(task)
            client.request_build.assert_not_called()
            client.request_export.assert_not_called()
            self.assertEqual(1, len(list(worker._spool.path.glob("*.npz"))))

    def test_partial_rejection_only_deletes_confirmed_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ServerClient()
            client.session_id = "scan"
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

    def test_receipt_stream_reopens_in_sequence_order_without_reusing_acknowledged_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = CaptureSpool(directory, "scan", {}, receipts=True, base_index=4)
            paths = [spool.append(**self.frame(index).kwargs) for index in range(12)]
            spool.acknowledge(paths[0])
            reopened = CaptureSpool.open(spool.path)
            self.assertTrue(reopened.retry_safe)
            self.assertEqual(paths[1:], list(reopened.pending_paths))
            metadata = reopened.load(paths[10])[2]
            self.assertEqual({"stream_id": spool.stream_id, "sequence": 10, "expected_index": 14}, metadata["upload_receipt"])
            self.assertEqual(12, reopened.next_sequence)
            for path in reopened.pending_paths:
                reopened.acknowledge(path)
            reopened.cleanup()
            again = CaptureSpool.open(spool.path)
            path = again.append(**self.frame(12).kwargs)
            self.assertEqual(12, again.load(path)[2]["upload_receipt"]["sequence"])

    def test_published_capture_survives_manifest_update_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = CaptureSpool(directory, "scan", {}, receipts=True)
            spool._save_manifest()
            with patch.object(spool, "_save_manifest", side_effect=OSError("manifest disk failure")):
                with self.assertRaisesRegex(OSError, "manifest disk failure"):
                    spool.append(**self.frame(0).kwargs)
            self.assertEqual(1, spool.pending_count)
            self.assertTrue(spool.ordering_error)
            with self.assertRaisesRegex(ValueError, "counter checkpoint failed"):
                spool.append(**self.frame(1).kwargs)
            reopened = CaptureSpool.open(spool.path)
            self.assertEqual(1, reopened.next_sequence)
            self.assertEqual(1, reopened.pending_count)
            second = reopened.append(**self.frame(1).kwargs)
            self.assertEqual(1, reopened.load(second)[2]["upload_receipt"]["sequence"])

    def test_failed_acknowledgement_deletion_keeps_file_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = CaptureSpool(directory, "scan", {})
            path = spool.append(**self.frame(0).kwargs)
            with patch.object(Path, "unlink", side_effect=OSError("file busy")):
                self.assertFalse(spool.acknowledge(path))
            self.assertEqual((path,), spool.pending_paths)
            self.assertEqual(1, spool.pending_count)
            self.assertTrue(path.exists())

    def test_partial_disk_write_remains_visible_and_preserves_completed_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = CaptureSpool(directory, "scan", {}, receipts=True)
            original = spool.append(**self.frame(0).kwargs)
            original_bytes = original.read_bytes()

            def failed_write(output, **kwargs):
                output.write(b"partial capture")
                raise OSError("disk full")

            with patch("kinect_scanner.capture_spool.np.savez", side_effect=failed_write):
                with self.assertRaisesRegex(OSError, "disk full"):
                    spool.append(**self.frame(1).kwargs)
            self.assertEqual(original_bytes, original.read_bytes())
            self.assertEqual(1, spool.pending_count)
            self.assertTrue(spool.ordering_error)
            discovered = CaptureSpool.discover(directory)
            self.assertEqual(1, len(discovered))
            self.assertTrue(discovered[0].ordering_error)
            self.assertFalse(discovered[0].retry_safe)
            self.assertEqual(b"partial capture", next(spool.path.glob("*.npz.tmp")).read_bytes())

    def test_legacy_order_and_corrupt_discovery_never_delete_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "session.json").write_text(json.dumps({"session_id": "scan", "settings": {}}))
            for name, identifier in (("a", "run:10"), ("b", "run:2")):
                with (legacy / (name + ".npz")).open("wb") as output:
                    np.savez(output, rgb=np.zeros((8, 8, 3), np.uint8), depth=np.zeros((8, 8), np.uint16),
                             metadata=json.dumps({"frame_id": identifier}))
            spool = CaptureSpool.open(legacy)
            self.assertFalse(spool.retry_safe)
            self.assertEqual(["run:2", "run:10"], [spool.load(p)[2]["frame_id"] for p in spool.pending_paths])
            with self.assertRaisesRegex(ValueError, "read-only"):
                spool.append(**self.frame(0).kwargs)
            broken = root / "broken"
            broken.mkdir()
            (broken / "session.json").write_bytes(b"{")
            (broken / "capture.npz").write_bytes(b"broken data")
            found = CaptureSpool.discover(root)
            self.assertEqual([broken, legacy], [s.path for s in found])
            self.assertTrue(found[0].ordering_error)
            self.assertEqual(1, found[0].pending_count)
            self.assertEqual(b"broken data", (broken / "capture.npz").read_bytes())


if __name__ == "__main__":
    unittest.main()
