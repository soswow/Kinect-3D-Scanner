"""Transport and file-integrity regressions without a running server."""

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PyQt6.QtCore import Qt

from kinect_scanner.server_client import ServerClient


def response(data=None, content=b"ply\nfinal mesh", content_type="application/octet-stream"):
    result = Mock()
    result.json.return_value = data
    result.content = content
    result.headers = {"content-type": content_type}
    result.iter_bytes.return_value = [content]
    return result


class ServerClientTests(unittest.TestCase):
    def test_loopback_uploads_use_stored_zlib_and_remote_uploads_use_compression(self):
        for host, level in (("localhost", 0), ("127.0.0.1", 0), ("127.4.5.6", 0),
                            ("192.168.1.10", 1), ("scanner.local", 1)):
            client = ServerClient()
            http = Mock()
            http.get.side_effect = [response({"status": "ok"}), response({})]
            http.post.return_value = response({"success": True})
            with patch("kinect_scanner.server_client.httpx.Client", return_value=http), \
                    patch("kinect_scanner.server_client.threading.Thread"), \
                    patch("kinect_scanner.server_client.pack_frame", return_value=b"frame") as frame, \
                    patch("kinect_scanner.server_client.pack_frames", return_value=b"batch") as batch:
                self.assertTrue(client.connect_to_server(host, 8000))
                client.send_frame(None, None)
                client.send_frames_batch([])
                self.assertEqual(level, frame.call_args.kwargs["compression_level"])
                self.assertEqual(level, batch.call_args.kwargs["compression_level"])
                self.assertIn("geometry=xyzrgb-f32le", client._ws_url)
            client.disconnect()

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
        reply = response()
        client._http.stream.return_value.__enter__ = Mock(return_value=reply)
        client._http.stream.return_value.__exit__ = Mock(return_value=False)
        path = client.request_final_preview()
        try:
            self.assertEqual(Path(path).read_bytes(), b"ply\nfinal mesh")
            client._http.stream.assert_called_once_with("GET", "/api/scan/export/ply", timeout=600.0)
            reply.iter_bytes.assert_called_once_with(chunk_size=1024 * 1024)
            client._http.get.assert_not_called()
            client._http.post.assert_not_called()
        finally:
            os.unlink(path)

    def test_preview_stream_failure_removes_partial_file(self):
        for failure in ("truncated", "disk_full", "transport", "json"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                client = ServerClient()
                client._http = Mock()
                reply = response(content_type="application/json" if failure == "json" else "application/octet-stream")
                if failure == "truncated":
                    reply.headers["content-length"] = "999"
                if failure == "transport":
                    def broken_stream(**kwargs):
                        yield b"partial"
                        raise OSError("Connection lost")
                    reply.iter_bytes.side_effect = broken_stream
                client._http.stream.return_value.__enter__ = Mock(return_value=reply)
                client._http.stream.return_value.__exit__ = Mock(return_value=False)
                with patch("kinect_scanner.server_client.tempfile.tempdir", directory), \
                        patch("kinect_scanner.server_client.shutil.disk_usage",
                              return_value=SimpleNamespace(free=0 if failure == "disk_full" else 1024**3)):
                    if failure == "json":
                        self.assertIsNone(client.request_preview())
                    else:
                        with self.assertRaises(OSError):
                            client.request_preview()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_atomic_export_keeps_previous_file_if_replace_fails(self):
        client = ServerClient()
        client._http = Mock()
        client._http.stream.return_value.__enter__ = Mock(return_value=response())
        client._http.stream.return_value.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing.ply"
            destination.write_bytes(b"previous final mesh")
            with (
                patch("kinect_scanner.server_client.os.replace", side_effect=OSError("disk error")),
                self.assertRaisesRegex(OSError, "disk error"),
            ):
                client.request_export("ply", str(destination))
            self.assertEqual(destination.read_bytes(), b"previous final mesh")
            self.assertEqual(list(Path(directory).iterdir()), [destination])
            self.assertTrue(client.request_export("ply", str(destination)))
            self.assertEqual(destination.read_bytes(), b"ply\nfinal mesh")

    def test_json_export_error_does_not_modify_destination(self):
        client = ServerClient()
        client._http = Mock()
        client._http.stream.return_value.__enter__ = Mock(return_value=response(content_type="application/json"))
        client._http.stream.return_value.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing.ply"
            destination.write_bytes(b"previous")
            self.assertFalse(client.request_export("ply", str(destination)))
            self.assertEqual(destination.read_bytes(), b"previous")

    def test_interrupted_and_truncated_downloads_preserve_existing_project(self):
        for interrupted in (False, True):
            client = ServerClient()
            client._http = Mock()
            resp = response(content=b"partial")
            resp.headers["content-length"] = "100"
            if interrupted:
                def chunks(**kwargs):
                    yield b"partial"
                    raise OSError("connection lost")
                resp.iter_bytes.side_effect = chunks
            client._http.stream.return_value.__enter__ = Mock(return_value=resp)
            client._http.stream.return_value.__exit__ = Mock(return_value=False)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "project.zip"
                path.write_bytes(b"original")
                with self.assertRaises(OSError):
                    client.request_export("session", str(path))
                self.assertEqual(b"original", path.read_bytes())
                self.assertEqual([path], list(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
