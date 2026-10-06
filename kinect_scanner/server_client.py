"""HTTP + WebSocket client for communicating with the scanner server."""

import ipaddress
import json
import logging
import os
import tempfile
import threading

import httpx
from PyQt6.QtCore import QObject, pyqtSignal

from shared.live import GEOMETRY_ENCODING, decode_live_geometry
from shared.protocol import pack_frame, pack_frames

logger = logging.getLogger(__name__)


class ServerClient(QObject):
    """Network client that talks to the scanner_server REST + WebSocket API.

    Signals mirror the old ScanTaskManager interface for easy GUI wiring.
    """

    connected = pyqtSignal()
    disconnected = pyqtSignal(str)

    reset_done = pyqtSignal(dict)
    frame_stored = pyqtSignal(dict)
    process_progress = pyqtSignal(int, int, dict)
    build_mesh_done = pyqtSignal(bool, str)
    preview_done = pyqtSignal(str)  # PLY temp file path
    final_preview_done = pyqtSignal(str)
    export_done = pyqtSignal(bool, str)
    save_mesh_done = pyqtSignal(bool, str)
    status_updated = pyqtSignal(dict)
    live_updated = pyqtSignal(dict)
    task_started = pyqtSignal(str)
    task_error = pyqtSignal(str)
    task_failed = pyqtSignal(str, str)
    websocket_status = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_url: str | None = None
        self._ws_url: str | None = None
        self._http: httpx.Client | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        self._connected = False
        self._ws_generation = 0
        self._ws_socket = None
        self.session_id = None
        self.last_status = {}
        self._compression_level = 1

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Connection ─────────────────────────────────────────────────────

    def connect_to_server(self, host: str, port: int) -> bool:
        """Connect synchronously; the GUI invokes this through ServerTaskWorker."""
        self.disconnect(notify=False)
        self._base_url = f"http://{host}:{port}"
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        # Level 0 retains the existing lossless zlib wire format. On loopback,
        # copying a few MB is much cheaper than compressing high-resolution RGB.
        self._compression_level = 0 if loopback else 1
        self._ws_url = f"ws://{host}:{port}/ws/progress?geometry={GEOMETRY_ENCODING}"
        try:
            self._http = httpx.Client(base_url=self._base_url, timeout=30.0)
            resp = self._http.get("/api/health", timeout=5.0)
            resp.raise_for_status()
            health = resp.json()
            if health.get("status") != "ok":
                raise RuntimeError(f"Server not ok: {health}")
            status = self.get_status()
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            self.disconnect(notify=False)
            self.disconnected.emit(str(exc))
            return False
        self._connected = True
        self.session_id = status.get("session_id")
        self.last_status = dict(status)
        self._ws_stop = threading.Event()
        self._ws_thread = threading.Thread(
            target=self._ws_listener,
            args=(self._ws_stop, self._ws_url, self._ws_generation),
            daemon=True, name="ws-listener",
        )
        self._ws_thread.start()
        self.connected.emit()
        self.status_updated.emit(status)
        return True

    def disconnect(self, notify=True):
        """Stop local transports without resetting the server's scan session."""
        self._ws_generation += 1
        self._ws_stop.set()
        sock, self._ws_socket = self._ws_socket, None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                logger.debug("Could not close WebSocket", exc_info=True)
        http, self._http = self._http, None
        if http is not None:
            try:
                http.close()
            except Exception:
                logger.debug("Could not close HTTP client", exc_info=True)
        self._connected = False
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=0.25)
        self._ws_thread = None
        if notify:
            self.disconnected.emit("Disconnected")

    # ── WebSocket listener ─────────────────────────────────────────────

    def _ws_listener(self, stop=None, url=None, generation=None):
        """Reconnect progress transport; retain the active server session."""
        stop = stop if stop is not None else self._ws_stop
        url = url if url is not None else self._ws_url
        generation = generation if generation is not None else self._ws_generation
        try:
            import websocket as ws_lib
        except ImportError as exc:
            if generation == self._ws_generation and not stop.is_set():
                self.websocket_status.emit("reconnecting", str(exc))
            return

        def active():
            return not stop.is_set() and generation == self._ws_generation

        while active():
            sock = None
            try:
                sock = ws_lib.WebSocket()
                sock.settimeout(5.0)
                sock.connect(url)
                if not active():
                    break
                self._ws_socket = sock
                self.websocket_status.emit("connected", "Progress connection established")
                while active():
                    try:
                        raw = sock.recv()
                    except ws_lib.WebSocketTimeoutException:
                        continue
                    if not raw:
                        raise RuntimeError("Progress connection closed")
                    try:
                        msg = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if active() and isinstance(msg, dict):
                        self._handle_ws_message(msg)
            except (ws_lib.WebSocketException, OSError, RuntimeError, ValueError) as exc:
                if active():
                    logger.warning("WebSocket error: %s, reconnecting...", exc)
                    self.websocket_status.emit("reconnecting", str(exc))
            finally:
                if self._ws_socket is sock:
                    self._ws_socket = None
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:  # Closing a failed transport is best effort.
                        logger.debug("Could not close failed WebSocket", exc_info=True)
            if active():
                stop.wait(1.0)

    def _handle_ws_message(self, msg: dict):
        """Route a parsed WebSocket message to the appropriate Qt signal."""
        msg_type = msg.get("type")

        if msg_type == "progress":
            self.process_progress.emit(
                msg.get("current", 0),
                msg.get("total", 0),
                msg.get("result", {"message": msg.get("message", "")}),
            )
        elif msg_type == "live":
            self.live_updated.emit(decode_live_geometry(msg))
        elif msg_type == "done":
            # Both preview and final build broadcast "done". Their HTTP
            # responses are handled by ServerTaskWorker with the correct signal.
            # Treating preview completion as a build enables exports too early.
            pass
        elif msg_type == "error":
            self.task_error.emit(msg.get("message", "Unknown server error"))

    # ── API methods (called from ServerTaskWorker thread) ──────────────

    def send_frame(self, rgb, depth, metadata=None) -> dict:
        """Pack and upload a frame. Returns the server response dict."""
        data = pack_frame(rgb, depth, metadata, compression_level=self._compression_level)
        resp = self._http.post(
            "/api/scan/frame",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        return resp.json()

    def send_frames_batch(self, frames: list[tuple]) -> dict:
        """Pack and upload multiple frames as a single batch."""
        data = pack_frames(frames, compression_level=self._compression_level)
        resp = self._http.post(
            "/api/scan/frames",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=120.0,
        )
        if resp.status_code == 404:
            # Server doesn't support batch — fall back to individual sends
            logger.warning("Server lacks batch endpoint, sending individually")
            results = [self.send_frame(*frame) for frame in frames]
            return {
                **results[-1],
                "results": results,
                "success": any(r.get("success") for r in results),
            }
        resp.raise_for_status()
        return resp.json()

    def reset_scan(self, settings=None) -> dict:
        resp = self._http.post("/api/scan/reset", json=settings or {})
        resp.raise_for_status()
        return resp.json()

    def request_build(self) -> dict:
        """Start a build. Progress comes via WebSocket."""
        resp = self._http.post("/api/scan/build", timeout=600.0)
        resp.raise_for_status()
        return resp.json()

    def request_preview(self) -> str | None:
        """Request preview, save returned PLY to a temp file, return path."""
        resp = self._http.post("/api/scan/preview", timeout=600.0)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "octet-stream" not in content_type:
            # Got JSON error response
            return None

        return self._save_temp_mesh(resp.content)

    @staticmethod
    def _save_temp_mesh(content: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".ply")
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(content)
        except BaseException:
            os.unlink(path)
            raise
        return path

    def request_final_preview(self) -> str | None:
        """Download the current final mesh without rebuilding a preview."""
        resp = self._http.get("/api/scan/export/ply", timeout=600.0)
        resp.raise_for_status()
        if "octet-stream" not in resp.headers.get("content-type", ""):
            return None
        return self._save_temp_mesh(resp.content)

    def request_export(self, fmt: str, save_path: str, options=None) -> bool:
        """Download and atomically replace the destination only after success."""
        resp = self._http.get(
            f"/api/scan/export/{fmt}", params=options or {}, timeout=600.0
        )
        resp.raise_for_status()
        if "octet-stream" not in resp.headers.get("content-type", ""):
            return False
        destination = os.path.abspath(save_path)
        fd, temporary = tempfile.mkstemp(
            prefix=".scanner-export-", dir=os.path.dirname(destination)
        )
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(resp.content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return True

    def get_status(self) -> dict:
        resp = self._http.get("/api/scan/status", timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def get_reconstruction(self) -> dict:
        resp = self._http.get("/api/scan/diagnostics", timeout=600.0)
        resp.raise_for_status()
        return resp.json()
