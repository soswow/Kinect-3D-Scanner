"""Transport and file-integrity regressions without a running server."""

import os
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from PyQt6.QtCore import Qt

from kinect_scanner.server_client import ServerClient


def response(data=None, content=b"ply\nfinal mesh", content_type="application/octet-stream"):
    result = Mock()
    result.json.return_value = data
    result.content = content
    result.headers = {"content-type": content_type}
    return result


class ServerClientTests(unittest.TestCase):
    def test_connection_emits_full_session_status_and_uses_bounded_checks(self):
        client = ServerClient()
        status = {"session_id": "existing", "settings": {"voxel_size": .004},
                  "stored_count": 24, "frame_count": 12, "has_mesh": True}
        http = Mock()
        http.get.side_effect = [response({"status": "ok"}), response(status)]
        updates = []
        client.status_updated.connect(updates.append)
        with patch("kinect_scanner.server_client.httpx.Client", return_value=http), \
                patch("kinect_scanner.server_client.threading.Thread"):
            self.assertTrue(client.connect_to_server("localhost", 8000))
        self.assertEqual(updates, [status])
        self.assertEqual(client.session_id, "existing")
        self.assertEqual(http.get.call_args_list[0].kwargs["timeout"], 5.0)
        self.assertEqual(http.get.call_args_list[1].kwargs["timeout"], 10.0)
        client.disconnect()

    def test_failed_status_closes_http_and_leaves_disconnected(self):
        client = ServerClient()
        http = Mock()
        http.get.side_effect = [response({"status": "ok"}), RuntimeError("status timeout")]
        failures = []
        client.disconnected.connect(failures.append)
        with patch("kinect_scanner.server_client.httpx.Client", return_value=http):
            self.assertFalse(client.connect_to_server("localhost", 8000))
        http.close.assert_called_once()
        self.assertFalse(client.is_connected)
        self.assertIsNone(client._http)
        self.assertEqual(failures, ["status timeout"])

    def test_disconnect_invalidates_old_listener_and_retains_session(self):
        client = ServerClient()
        old_stop = client._ws_stop
        old_generation = client._ws_generation
        client.session_id = "saved-session"
        sock = client._ws_socket = Mock()
        client.disconnect()
        self.assertTrue(old_stop.is_set())
        self.assertGreater(client._ws_generation, old_generation)
        sock.close.assert_called_once()
        self.assertEqual(client.session_id, "saved-session")
        # A reconnect must not clear the event captured by the old thread.
        client._ws_stop = threading.Event()
        self.assertTrue(old_stop.is_set())

    def test_old_listener_cannot_emit_after_reconnect(self):
        client = ServerClient()
        stop = client._ws_stop
        generation = client._ws_generation
        receiving = threading.Event()
        release = threading.Event()
        sock = Mock()

        def receive():
            receiving.set()
            release.wait(2)
            return '{"type":"live","session_id":"old"}'

        sock.recv.side_effect = receive
        module = SimpleNamespace(WebSocket=Mock(return_value=sock),
                                 WebSocketTimeoutException=TimeoutError)
        messages = []
        client.live_updated.connect(messages.append, Qt.ConnectionType.DirectConnection)
        with patch.dict(sys.modules, {"websocket": module}):
            listener = threading.Thread(target=client._ws_listener,
                                        args=(stop, "ws://old", generation))
            listener.start()
            try:
                self.assertTrue(receiving.wait(2))
                client.disconnect()
                client._ws_stop = threading.Event()
                release.set()
                listener.join(2)
                self.assertFalse(listener.is_alive())
                self.assertEqual(messages, [])
            finally:
                release.set()
                listener.join(2)

    def test_final_preview_downloads_final_export_without_preview_post(self):
        client = ServerClient()
        client._http = Mock()
        client._http.get.return_value = response()
        path = client.request_final_preview()
        try:
            self.assertEqual(Path(path).read_bytes(), b"ply\nfinal mesh")
            client._http.get.assert_called_once_with("/api/scan/export/ply", timeout=600.0)
            client._http.post.assert_not_called()
        finally:
            os.unlink(path)

    def test_atomic_export_keeps_previous_file_if_replace_fails(self):
        client = ServerClient()
        client._http = Mock()
        client._http.get.return_value = response()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing.ply"
            destination.write_bytes(b"previous final mesh")
            with patch("kinect_scanner.server_client.os.replace", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    client.request_export("ply", str(destination))
            self.assertEqual(destination.read_bytes(), b"previous final mesh")
            self.assertEqual(list(Path(directory).iterdir()), [destination])
            self.assertTrue(client.request_export("ply", str(destination)))
            self.assertEqual(destination.read_bytes(), b"ply\nfinal mesh")

    def test_json_export_error_does_not_modify_destination(self):
        client = ServerClient()
        client._http = Mock()
        client._http.get.return_value = response(content_type="application/json")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing.ply"
            destination.write_bytes(b"previous")
            self.assertFalse(client.request_export("ply", str(destination)))
            self.assertEqual(destination.read_bytes(), b"previous")


if __name__ == "__main__":
    unittest.main()
