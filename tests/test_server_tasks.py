"""Queue responsiveness and dispatch regressions with mocked transports."""

import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PyQt6.QtCore import Qt

from kinect_scanner.server_client import ServerClient
from kinect_scanner.server_task_worker import (
    ServerTask,
    ServerTaskType,
    ServerTaskWorker,
)


class ServerTaskTests(unittest.TestCase):
    def test_compact_project_review_dispatches_and_export_skips_local_index_reuse(self):
        client = ServerClient()
        client.request_compact_project_info = Mock(return_value={"retained_count": 2})
        client.request_export = Mock(return_value=True)
        updates = []
        client.compact_project_info.connect(updates.append)
        worker = ServerTaskWorker(client)
        worker._dispatch(ServerTask(ServerTaskType.COMPACT_PROJECT_INFO))
        self.assertEqual([{"retained_count": 2}], updates)
        options = {"compact": True, "compact_revision": "review"}
        with patch.object(worker, "_local_export_captures") as reuse:
            worker._dispatch(ServerTask(ServerTaskType.EXPORT_SESSION, {"path": "/unused/compact.zip", "options": options}))
            reuse.assert_not_called()
        client.request_export.assert_called_once_with("session", "/unused/compact.zip", options=options)

    def test_compact_sensor_merge_omits_display_copies_and_preserves_destination_on_failure(self):
        client = ServerClient()
        worker = ServerTaskWorker(client)
        snapshot = {"complete": True}
        recorder = Mock()
        recorder.flush_sensor_recording.return_value = snapshot
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.zip"
            path.write_bytes(b"existing project")

            def download(fmt, destination, **kwargs):
                Path(destination).write_bytes(b"compact project")
                return True

            client.request_export = Mock(side_effect=download)
            task = ServerTask(ServerTaskType.EXPORT_SESSION, {"path": str(path), "options": {"compact": True},
                                                            "sensor_recorder": recorder})
            with patch("kinect_scanner.server_task_worker.augment_session_archive", side_effect=OSError("merge failed")) as augment:
                with self.assertRaisesRegex(OSError, "merge failed"):
                    worker._dispatch(task)
                augment.assert_called_once_with(client.request_export.call_args.args[1], snapshot, portrait=False)
            self.assertEqual(b"existing project", path.read_bytes())
            with patch("kinect_scanner.server_task_worker.augment_session_archive") as augment:
                worker._dispatch(task)
                self.assertFalse(augment.call_args.kwargs["portrait"])
            self.assertEqual(b"compact project", path.read_bytes())

    def test_full_upload_queue_reports_failure_once_until_capacity_returns(self):
        worker = ServerTaskWorker(ServerClient())
        task = ServerTask(ServerTaskType.SEND_FRAME)
        for _ in range(100):
            self.assertTrue(worker.submit(task))
        with self.assertLogs("kinect_scanner.server_task_worker", level="INFO") as logs:
            self.assertFalse(worker.submit(task))
            self.assertFalse(worker.submit(task))
            worker._queue.get_nowait()
            self.assertTrue(worker.submit(task))
            self.assertFalse(worker.submit(task))
        self.assertEqual(3, len(logs.output))
        self.assertIn("space again", logs.output[1])

    def test_partial_batch_acknowledgements_identify_each_capture(self):
        client = ServerClient()
        client.send_frames_batch = Mock(return_value={"success": True, "results": [
            {"success": True, "index": 0}, {"success": False},
        ]})
        worker = ServerTaskWorker(client)
        worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {
            "rgb": 2, "depth": 2, "metadata": {"frame_id": "second"},
        }))
        result = worker._drain_send_frames(ServerTask(ServerTaskType.SEND_FRAME, {
            "rgb": 1, "depth": 1, "metadata": {"frame_id": "first"},
        }))
        self.assertEqual([
            {"frame_id": "first", "success": True, "index": 0},
            {"frame_id": "second", "success": False},
        ], result["capture_acknowledgements"])

    def test_failed_transport_releases_captures_in_that_batch(self):
        client = ServerClient()
        client.session_id = "scan"
        client.send_frame = Mock(side_effect=RuntimeError("upload failed"))
        updates = []
        client.frame_stored.connect(updates.append)
        worker = ServerTaskWorker(client)
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            worker._drain_send_frames(ServerTask(ServerTaskType.SEND_FRAME, {
                "rgb": 1, "depth": 1, "metadata": {"frame_id": "failed"},
            }))
        self.assertFalse(updates[0]["success"])
        self.assertEqual("scan", updates[0]["session_id"])
        self.assertEqual([{"frame_id": "failed", "success": False}],
                         updates[0]["capture_acknowledgements"])

    def test_connection_runs_off_submitting_thread(self):
        client = ServerClient()
        worker = ServerTaskWorker(client)
        entered = threading.Event()
        release = threading.Event()
        thread_ids = []

        def connect(host, port):
            thread_ids.append(threading.get_ident())
            entered.set()
            release.wait(2)
            return True

        client.connect_to_server = connect
        worker.start()
        try:
            self.assertTrue(worker.submit(ServerTask(ServerTaskType.CONNECT, {"host": "test", "port": 8000})))
            self.assertTrue(entered.wait(2))
            self.assertNotEqual(thread_ids, [threading.get_ident()])
            # Submission remains responsive while the HTTP connection is blocked.
            self.assertTrue(worker.submit(ServerTask(ServerTaskType.STATUS)))
        finally:
            worker.stop()
            release.set()
            self.assertTrue(worker.wait(3000))

    def test_final_preview_dispatches_distinct_signal(self):
        client = ServerClient()
        client.request_final_preview = Mock(return_value="/tmp/final.ply")
        client.request_preview = Mock()
        final = []
        preview = []
        client.final_preview_done.connect(final.append)
        client.preview_done.connect(preview.append)
        ServerTaskWorker(client)._dispatch(ServerTask(ServerTaskType.FINAL_PREVIEW))
        self.assertEqual(final, ["/tmp/final.ply"])
        self.assertEqual(preview, [])
        client.request_preview.assert_not_called()

    def test_status_and_disconnect_are_queue_commands(self):
        client = ServerClient()
        client.get_status = Mock(return_value={"session_id": "restored", "has_mesh": True})
        client.disconnect = Mock()
        updates = []
        client.status_updated.connect(updates.append)
        worker = ServerTaskWorker(client)
        worker._dispatch(ServerTask(ServerTaskType.STATUS))
        worker._dispatch(ServerTask(ServerTaskType.DISCONNECT))
        self.assertEqual(updates, [{"session_id": "restored", "has_mesh": True}])
        self.assertEqual(client.session_id, "restored")
        client.disconnect.assert_called_once_with()

    def test_recording_setup_failure_still_reports_committed_reset(self):
        client = ServerClient()
        result = {"success": True, "session_id": "new", "settings": {"live_reconstruction": True}}
        client.reset_scan = Mock(return_value=result)
        resets, warnings, failures = [], [], []
        client.reset_done.connect(resets.append)
        client.task_error.connect(warnings.append)
        client.task_failed.connect(lambda kind, detail: failures.append((kind, detail)))
        worker = ServerTaskWorker(client)
        with patch("kinect_scanner.server_task_worker.RecordingWriter", side_effect=OSError("disk full")):
            worker._dispatch(ServerTask(ServerTaskType.RESET, {"record": True}))
        self.assertEqual(resets, [result])
        self.assertEqual(warnings, ["Local recording unavailable: disk full"])
        self.assertEqual(failures, [])
        self.assertIsNone(worker._recording)
        self.assertIsNone(worker._recording_session_id)
        self.assertEqual(client.session_id, "new")
        self.assertTrue(worker._live)

    def test_reconnect_preserves_only_matching_session_recording(self):
        client = ServerClient()
        client.connect_to_server = Mock(return_value=True)
        client.last_status = {"session_id": "same", "settings": {"live_reconstruction": True}}
        worker = ServerTaskWorker(client)
        recording = worker._recording = Mock()
        worker._recording_session_id = "same"
        task = ServerTask(ServerTaskType.CONNECT, {"host": "server", "port": 8000})
        worker._dispatch(task)
        self.assertIs(worker._recording, recording)
        self.assertEqual(worker._recording_session_id, "same")
        self.assertTrue(worker._live)
        client.last_status = {"session_id": "different", "settings": {"live_reconstruction": False}}
        worker._dispatch(task)
        self.assertIsNone(worker._recording)
        self.assertIsNone(worker._recording_session_id)
        self.assertFalse(worker._live)

    def test_successful_recording_tracks_committed_session(self):
        client = ServerClient()
        client.reset_scan = Mock(return_value={"success": True, "session_id": "recorded", "settings": {}})
        worker = ServerTaskWorker(client)
        with patch("kinect_scanner.server_task_worker.RecordingWriter") as recording:
            worker._dispatch(ServerTask(ServerTaskType.RESET, {"record": True}))
        self.assertIs(worker._recording, recording.return_value)
        self.assertEqual(worker._recording_session_id, "recorded")

    def test_project_save_reuses_only_recordings_for_current_session(self):
        client = ServerClient()
        client.session_id = "recorded"
        client._project_export_supported = True
        client.request_export = Mock(return_value=True)
        worker = ServerTaskWorker(client)
        worker._recording = Mock()
        worker._recording_session_id = "recorded"
        task = ServerTask(ServerTaskType.EXPORT_SESSION, {"path": "/unused/project.zip", "options": {"rgb_format": "jpeg"}})
        with patch("shared.project_archive.recording_captures", return_value={0: {"rgb": "local"}}):
            worker._dispatch(task)
        client.request_export.assert_called_once_with("session", "/unused/project.zip", options={"rgb_format": "jpeg"}, local_captures={0: {"rgb": "local"}})
        worker._recording_session_id = "old"
        worker._dispatch(task)
        self.assertNotIn("local_captures", client.request_export.call_args.kwargs)

    def test_connection_recovers_matching_local_recording_after_restart(self):
        client = ServerClient()
        client._project_export_supported = True
        client.connect_to_server = Mock(return_value=True)
        client.last_status = {"session_id": "existing", "settings": {"live_reconstruction": True}}
        worker = ServerTaskWorker(client)
        with patch("kinect_scanner.server_task_worker.RecordingWriter.for_session", return_value=Mock()) as recover:
            worker._dispatch(ServerTask(ServerTaskType.CONNECT, {"host": "server", "port": 8000}))
        self.assertIs(worker._recording, recover.return_value)
        self.assertEqual("existing", worker._recording_session_id)

    def test_exception_identifies_failed_task_and_worker_continues(self):
        client = ServerClient()
        client.request_final_preview = Mock(side_effect=RuntimeError("lost transport"))
        client.get_status = Mock(return_value={"session_id": "same"})
        failures = []
        errors = []
        completed = threading.Event()
        client.task_failed.connect(lambda kind, msg: failures.append((kind, msg)), Qt.ConnectionType.DirectConnection)
        client.task_error.connect(errors.append, Qt.ConnectionType.DirectConnection)
        client.status_updated.connect(lambda _: completed.set(), Qt.ConnectionType.DirectConnection)
        worker = ServerTaskWorker(client)
        worker.submit(ServerTask(ServerTaskType.FINAL_PREVIEW))
        worker.submit(ServerTask(ServerTaskType.STATUS))
        worker.start()
        try:
            self.assertTrue(completed.wait(2))
            self.assertEqual(failures, [("FINAL_PREVIEW", "lost transport")])
            self.assertEqual(errors, ["FINAL_PREVIEW: lost transport"])
        finally:
            worker.stop()
            self.assertTrue(worker.wait(3000))
        self.assertFalse(worker.submit(ServerTask(ServerTaskType.STATUS)))


if __name__ == "__main__":
    unittest.main()
