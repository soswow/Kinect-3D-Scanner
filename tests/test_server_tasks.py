"""Queue responsiveness and dispatch regressions with mocked transports."""

import threading
import unittest
from unittest.mock import Mock

from PyQt6.QtCore import Qt

from kinect_scanner.server_client import ServerClient
from kinect_scanner.server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker


class ServerTaskTests(unittest.TestCase):
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
