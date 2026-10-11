"""Queue responsiveness and dispatch regressions with mocked transports."""

import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from PyQt6.QtCore import Qt

from kinect_scanner.server_client import ServerClient
from kinect_scanner.server_task_worker import (
    ServerTask,
    ServerTaskType,
    ServerTaskWorker,
)


class ServerTaskTests(unittest.TestCase):
    def test_retry_preserves_order_conflict_message_and_original_capture(self):
        client = ServerClient()
        client.session_id = "scan"
        client._connected = True
        client._capture_receipts_supported = True
        client.get_status = Mock(return_value={"session_id": "scan", "settings": {}, "stored_count": 0})
        client.send_frame = Mock(return_value={"success": False, "session_id": "scan",
                                              "message": "Capture expected index 3 but scan has 0 frames"})
        with tempfile.TemporaryDirectory() as directory, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(directory)):
            worker = ServerTaskWorker(client)
            worker._configure_spool({}, "scan", 3)
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": np.zeros((2, 2, 3), np.uint8),
                "depth": np.ones((2, 2), np.uint16), "metadata": {"frame_id": "run:3"}}))
            task = worker._queue.get_nowait()
            with self.assertRaisesRegex(RuntimeError, "expected index 3"):
                worker._dispatch(ServerTask(ServerTaskType.RETRY_CAPTURES, {"path": str(worker._spool.path)}))
            self.assertTrue(task.kwargs["spool_path"].exists())
    def test_proven_receipt_survives_optional_corrupt_recording_and_removes_only_confirmed_source(self):
        from shared.capture_receipts import receipt_for
        from shared.recording import RecordingWriter
        client = ServerClient()
        client.session_id = "scan"
        client._capture_receipts_supported = True
        warnings = []
        client.task_error.connect(warnings.append)
        with tempfile.TemporaryDirectory() as directory, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(directory)):
            worker = ServerTaskWorker(client)
            worker._configure_spool({}, "scan")
            recording = RecordingWriter(Path(directory) / "recording", {}, session_id="scan")
            recording.manifest["frames"] = [None]
            recording._save_manifest()
            previous_manifest = (recording.path / "manifest.json").read_bytes()
            worker._recording = recording
            worker._recording_session_id = "scan"
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": np.zeros((2, 2, 3), np.uint8),
                "depth": np.ones((2, 2), np.uint16), "metadata": {"frame_id": "run:0"}}))
            task = worker._queue.get_nowait()
            rgb, depth, metadata = task.kwargs["spool"].load(task.kwargs["spool_path"])
            client.send_frame = Mock(return_value={"success": True, "session_id": "scan", "index": 0,
                "stored_count": 1, "capture_receipt": receipt_for(rgb, depth, metadata)})
            worker._dispatch(task)
            self.assertIsNone(worker._recording)
            self.assertIn("Recording stopped", warnings[-1])
            self.assertFalse(task.kwargs["spool_path"].exists())
            self.assertEqual(previous_manifest, (recording.path / "manifest.json").read_bytes())
    def test_status_between_healthy_uploads_does_not_turn_queued_successor_into_recovery(self):
        from shared.capture_receipts import receipt_for
        client = ServerClient()
        client.session_id = "scan"
        client._capture_receipts_supported = True
        accepted = []
        def send(rgb, depth, metadata):
            proof = receipt_for(rgb, depth, metadata)
            accepted.append(proof["expected_index"])
            return {"success": True, "session_id": "scan", "index": proof["expected_index"],
                    "stored_count": len(accepted), "capture_receipt": proof}
        client.send_frame = Mock(side_effect=send)
        client.get_status = Mock(side_effect=lambda: {"session_id": "scan", "settings": {}, "stored_count": len(accepted)})
        with tempfile.TemporaryDirectory() as directory, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(directory)):
            worker = ServerTaskWorker(client)
            worker._configure_spool({}, "scan")
            frame = {"rgb": np.zeros((2, 2, 3), np.uint8), "depth": np.ones((2, 2), np.uint16)}
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {**frame, "metadata": {"frame_id": "run:0"}}))
            worker.submit(ServerTask(ServerTaskType.STATUS))
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {**frame, "metadata": {"frame_id": "run:1"}}))
            worker._dispatch(worker._queue.get_nowait())
            worker._dispatch(worker._pending)
            worker._pending = None
            self.assertFalse(worker._failed_uploads)
            worker._dispatch(worker._queue.get_nowait())
            self.assertEqual([0, 1], accepted)
            self.assertEqual(0, worker.pending_capture_count)

    def test_multiple_matching_spools_keep_barrier_and_new_stream_uses_recovered_tail(self):
        from shared.capture_receipts import receipt_for
        from kinect_scanner.capture_spool import CaptureSpool
        client = ServerClient()
        client.session_id = "scan"
        client._connected = True
        client._capture_receipts_supported = True
        accepted = []
        def send(rgb, depth, metadata):
            proof = receipt_for(rgb, depth, metadata)
            self.assertEqual(len(accepted), proof["expected_index"])
            accepted.append(proof)
            return {"success": True, "session_id": "scan", "index": proof["expected_index"],
                    "stored_count": len(accepted), "capture_receipt": proof}
        client.send_frame = Mock(side_effect=send)
        client.get_status = Mock(side_effect=lambda: {"session_id": "scan", "settings": {}, "stored_count": len(accepted)})
        with tempfile.TemporaryDirectory() as directory, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(directory)):
            frame = (np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.uint16))
            first = CaptureSpool(Path(directory) / "capture-queue", "scan", {}, receipts=True, base_index=0)
            second = CaptureSpool(Path(directory) / "capture-queue", "scan", {}, receipts=True, base_index=1)
            first.append(*frame, {"frame_id": "run:0"})
            second.append(*frame, {"frame_id": "run:1"})
            worker = ServerTaskWorker(client)
            worker._restore_session_resources(client.get_status())
            self.assertEqual(2, worker.pending_capture_count)
            worker._dispatch(ServerTask(ServerTaskType.RETRY_CAPTURES, {"path": str(first.path)}))
            self.assertTrue(worker.capture_recovery_required)
            self.assertEqual(1, worker.pending_capture_count)
            worker._dispatch(ServerTask(ServerTaskType.RETRY_CAPTURES, {"path": str(second.path)}))
            self.assertFalse(worker.capture_recovery_required)
            self.assertEqual(2, worker._spool.base_index)
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": frame[0], "depth": frame[1], "metadata": {"frame_id": "run:2"}}))
            worker._dispatch(worker._queue.get_nowait())
            self.assertEqual([0, 1, 2], [proof["expected_index"] for proof in accepted])

    def test_lost_receipt_blocks_later_uploads_then_restart_retry_continues_in_order(self):
        from shared.capture_receipts import receipt_for
        client = ServerClient()
        client.session_id = "scan"
        client._connected = True
        client._capture_receipts_supported = True
        rgb, depth = np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.uint16)
        accepted = {}
        lose_ack = True

        def send(rgb, depth, metadata):
            nonlocal lose_ack
            proof = receipt_for(rgb, depth, metadata)
            index = proof["expected_index"]
            if index not in accepted:
                self.assertEqual(len(accepted), index)
                accepted[index] = proof
            else:
                self.assertEqual(accepted[index], proof)
            if lose_ack:
                lose_ack = False
                raise OSError("Response lost after commit")
            return {"success": True, "session_id": "scan", "index": index,
                    "stored_count": len(accepted), "capture_receipt": proof}

        client.send_frame = Mock(side_effect=send)
        client.get_status = Mock(side_effect=lambda: {"session_id": "scan", "settings": {}, "stored_count": len(accepted)})
        with tempfile.TemporaryDirectory() as directory, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(directory)):
            worker = ServerTaskWorker(client)
            worker._configure_spool({}, "scan")
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": rgb, "depth": depth, "metadata": {"frame_id": "run:0"}}))
            first = worker._queue.get_nowait()
            with self.assertRaisesRegex(OSError, "Response lost"):
                worker._dispatch(first)
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": rgb, "depth": depth, "metadata": {"frame_id": "run:1"}}))
            with self.assertRaisesRegex(RuntimeError, "Earlier captures"):
                worker._dispatch(worker._queue.get_nowait())
            self.assertEqual(1, client.send_frame.call_count)
            restarted = ServerTaskWorker(client)
            restarted._restore_session_resources(client.get_status())
            self.assertEqual(2, restarted.pending_capture_count)
            self.assertTrue(restarted.capture_recovery_required)
            restarted._dispatch(ServerTask(ServerTaskType.RETRY_CAPTURES, {"path": str(first.kwargs["spool"].path)}))
            self.assertEqual(2, len(accepted))
            self.assertEqual(0, restarted.pending_capture_count)
            self.assertFalse(restarted.capture_recovery_required)
            self.assertEqual(2, restarted._spool.base_index)
            restarted.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": rgb, "depth": depth, "metadata": {"frame_id": "run:2"}}))
            restarted._dispatch(restarted._queue.get_nowait())
            self.assertEqual(3, len(accepted))

    def test_unverified_receipt_acknowledgements_never_delete_source(self):
        from shared.capture_receipts import receipt_for
        cases = ("wrong-session", "bool-index", "bool-proof", "truthy-success", "non-dict-ack", "extra-ack", "non-dict-result")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(directory)):
                client = ServerClient()
                client.session_id = "scan"
                client._capture_receipts_supported = True
                worker = ServerTaskWorker(client)
                worker._configure_spool({}, "scan")
                worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": np.zeros((2, 2, 3), np.uint8),
                    "depth": np.ones((2, 2), np.uint16), "metadata": {"frame_id": "one"}}))
                task = worker._queue.get_nowait()
                rgb, depth, metadata = task.kwargs["spool"].load(task.kwargs["spool_path"])
                proof = receipt_for(rgb, depth, metadata)
                ack = {"success": True, "session_id": "scan", "index": 0, "capture_receipt": proof}
                if case == "wrong-session": ack["session_id"] = "other"
                if case == "bool-index": ack["index"] = False
                if case == "bool-proof": proof["sequence"] = False
                if case == "truthy-success": ack["success"] = "false"
                result = {"success": True, "session_id": "scan", "results": [ack]}
                if case == "non-dict-ack": result["results"] = [None]
                if case == "extra-ack": result["results"] = [ack, ack]
                if case == "non-dict-result": result = None
                client.send_frame = Mock(return_value=result)
                try:
                    worker._dispatch(task)
                except RuntimeError:
                    pass
                self.assertTrue(task.kwargs["spool_path"].exists())
                self.assertTrue(worker._failed_uploads)

    def test_recovery_export_is_offline_and_does_not_clear_pending_sources(self):
        client = ServerClient()
        client.session_id = "scan"
        client.get_status = Mock()
        client.reset_scan = Mock()
        client.send_frame = Mock()
        with tempfile.TemporaryDirectory() as directory, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(directory)):
            worker = ServerTaskWorker(client)
            worker._configure_spool({}, "scan")
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": np.zeros((2, 2, 3), np.uint8),
                "depth": np.ones((2, 2), np.uint16), "metadata": {"frame_id": "one"}}))
            task = worker._queue.get_nowait()
            with patch("kinect_scanner.capture_recovery.export_recovery_project", return_value=Path(directory) / "recovery.zip") as save:
                worker._dispatch(ServerTask(ServerTaskType.EXPORT_RECOVERY, {"path": str(worker._spool.path), "destination": "recovery.zip"}))
            save.assert_called_once()
            self.assertTrue(task.kwargs["spool_path"].exists())
            client.get_status.assert_not_called()
            client.reset_scan.assert_not_called()
            client.send_frame.assert_not_called()

    def test_opened_project_starts_new_receipt_stream_after_saved_frames(self):
        client = ServerClient()
        client._capture_receipts_supported = True
        client.request_open_project = Mock(return_value={"settings": {}, "session_id": "opened", "stored_count": 12})
        worker = ServerTaskWorker(client)
        worker._dispatch(ServerTask(ServerTaskType.OPEN_PROJECT, {"path": "project.zip"}))
        self.assertEqual(12, worker._spool.base_index)

    def test_live_build_barrier_does_not_count_queued_status_as_capture(self):
        client = ServerClient()
        worker = ServerTaskWorker(client)
        worker.submit(ServerTask(ServerTaskType.STATUS))
        worker._require_uploaded_captures()

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

    def test_captures_keep_origin_across_reconnect_and_preserve_stale_spool(self):
        client = ServerClient()
        client.session_id = "old"
        client.send_frame = Mock(return_value={"success": True})
        client.connect_to_server = Mock(return_value=True)
        client.last_status = {"session_id": "new", "settings": {}}
        client._project_export_supported = True
        worker = ServerTaskWorker(client)
        failures = []
        client.frame_stored.connect(failures.append)
        rgb, depth = np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.uint16)
        with tempfile.TemporaryDirectory() as temporary, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(temporary)), patch("kinect_scanner.server_task_worker.RecordingWriter.for_session", return_value=None):
            worker._configure_spool({}, "old")
            old_spool = worker._spool
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": rgb, "depth": depth, "metadata": {"frame_id": "old-frame"}}))
            old_task = worker._queue.get_nowait()
            client.session_id = "new"  # connect_to_server commits this before it returns.
            worker._dispatch(ServerTask(ServerTaskType.CONNECT, {"host": "new-host", "port": 8000}))
            client.connect_to_server.assert_called_once_with("new-host", 8000, notify=False)
            recording = worker._recording = Mock()
            worker._recording_session_id = "new"
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": rgb, "depth": depth, "metadata": {"frame_id": "new-frame"}}))
            with self.assertRaisesRegex(RuntimeError, "different server session"):
                worker._drain_send_frames(old_task)
            self.assertTrue(old_task.kwargs["spool_path"].exists())
            self.assertEqual(1, old_spool.pending_count)
            self.assertFalse(worker._failed_uploads)
            self.assertEqual("old", failures[-1]["session_id"])
            self.assertEqual("old-frame", failures[-1]["capture_acknowledgements"][0]["frame_id"])
            recording.append.assert_not_called()
            client.send_frame.assert_not_called()
            self.assertEqual("new", worker._pending.kwargs["capture_session_id"])
            result = worker._drain_send_frames(worker._pending)
            self.assertTrue(result["success"])
            client.send_frame.assert_called_once()
            recording.append.assert_called_once()
            client.request_build = Mock(return_value={"success": True})
            with patch.object(worker, "_save_reconstruction"):
                worker._dispatch(ServerTask(ServerTaskType.BUILD_MESH))
            client.request_build.assert_called_once()

    def test_wrong_session_acknowledgement_keeps_spool_and_blocks_save(self):
        client = ServerClient()
        client.session_id = "expected"
        client.send_frame = Mock(return_value={"success": True, "session_id": "other", "index": 0})
        worker = ServerTaskWorker(client)
        with tempfile.TemporaryDirectory() as temporary, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(temporary)):
            worker._configure_spool({}, "expected")
            worker._recording = Mock()
            worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {
                "rgb": np.zeros((2, 2, 3), np.uint8), "depth": np.ones((2, 2), np.uint16),
            }))
            task = worker._queue.get_nowait()
            with self.assertRaisesRegex(RuntimeError, "acknowledged a different"):
                worker._drain_send_frames(task)
            self.assertTrue(task.kwargs["spool_path"].exists())
            worker._recording.append.assert_not_called()
            self.assertTrue(worker._failed_uploads)
            with self.assertRaisesRegex(RuntimeError, "not confirmed"):
                worker._require_uploaded_captures()

    def test_live_capture_origin_is_not_rebound_by_status(self):
        client = ServerClient()
        client.session_id = "old"
        client.send_frame = Mock()
        worker = ServerTaskWorker(client)
        worker._live = True
        worker._configure_spool({"live_reconstruction": True}, "old")
        worker.submit(ServerTask(ServerTaskType.SEND_FRAME, {"rgb": 1, "depth": 1, "metadata": {"frame_id": "old-frame"}}))
        task = worker._queue.get_nowait()
        client.get_status = Mock(return_value={"session_id": "new", "settings": {"live_reconstruction": True}})
        worker._dispatch(ServerTask(ServerTaskType.STATUS))
        self.assertEqual("old", task.kwargs["capture_session_id"])
        with self.assertRaisesRegex(RuntimeError, "different server session"):
            worker._drain_send_frames(task)
        client.send_frame.assert_not_called()

    def test_connect_notifies_only_after_resources_are_rebound(self):
        client = ServerClient()
        client.session_id = "new"
        client.connect_to_server = Mock(return_value=True)
        client.last_status = {"session_id": "new", "settings": {}}
        worker = ServerTaskWorker(client)
        notifications = []
        client.connected.connect(lambda: notifications.append(("connected", worker._spool.session_id)))
        client.status_updated.connect(lambda status: notifications.append(("status", worker._spool.session_id)))
        worker._dispatch(ServerTask(ServerTaskType.CONNECT, {"host": "test", "port": 8000}))
        self.assertEqual([("connected", "new"), ("status", "new")], notifications)

    def test_status_rebinds_resources_after_uncertain_session_replacement(self):
        client = ServerClient()
        client.session_id = "old"
        worker = ServerTaskWorker(client)
        with tempfile.TemporaryDirectory() as temporary, patch("kinect_scanner.server_task_worker.data_root", return_value=Path(temporary)):
            worker._configure_spool({}, "old")
            old_spool = worker._spool
            rgb = np.zeros((2, 2, 3), np.uint8)
            depth = np.ones((2, 2), np.uint16)
            retained = old_spool.append(rgb, depth, {"frame_id": "old-frame"})
            worker._failed_uploads = True
            worker._recording = Mock()
            worker._recording_session_id = "old"
            client.get_status = Mock(return_value={"session_id": "replacement", "settings": {}})
            worker._dispatch(ServerTask(ServerTaskType.STATUS))
            self.assertEqual("replacement", client.session_id)
            self.assertEqual("replacement", worker._spool.session_id)
            self.assertFalse(worker._live)
            self.assertFalse(worker._failed_uploads)
            self.assertIsNone(worker._recording)
            self.assertTrue(retained.exists())
            self.assertEqual(1, old_spool.pending_count)
            self.assertEqual(0, worker.pending_capture_count)
            worker._recording = Mock()
            worker._recording_session_id = "replacement"
            spool, recording = worker._spool, worker._recording
            worker._failed_uploads = True
            worker._dispatch(ServerTask(ServerTaskType.STATUS))
            self.assertIs(spool, worker._spool)
            self.assertIs(recording, worker._recording)
            self.assertTrue(worker._failed_uploads)
            client.get_status.return_value = {"session_id": "live", "settings": {"live_reconstruction": True}}
            worker._dispatch(ServerTask(ServerTaskType.STATUS))
            self.assertTrue(worker._live)
            self.assertIsNone(worker._spool)
            worker._failed_uploads = True
            worker._dispatch(ServerTask(ServerTaskType.STATUS))
            self.assertTrue(worker._failed_uploads)

    def test_status_recovers_only_matching_recording_and_tolerates_recovery_failure(self):
        client = ServerClient()
        client._project_export_supported = True
        client.get_status = Mock(return_value={"session_id": "new", "settings": {"live_reconstruction": True}})
        worker = ServerTaskWorker(client)
        with patch("kinect_scanner.server_task_worker.RecordingWriter.for_session", return_value=Mock()) as recover:
            worker._dispatch(ServerTask(ServerTaskType.STATUS))
        recover.assert_called_once()
        self.assertEqual("new", worker._recording_session_id)
        client.get_status.return_value["session_id"] = "another"
        updates = []
        client.status_updated.connect(updates.append)
        with patch("kinect_scanner.server_task_worker.RecordingWriter.for_session", side_effect=OSError("disk unavailable")):
            worker._dispatch(ServerTask(ServerTaskType.STATUS))
        self.assertEqual("another", updates[-1]["session_id"])
        self.assertIsNone(worker._recording)
        self.assertIsNone(worker._recording_session_id)

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

        def connect(host, port, *, notify=True):
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
