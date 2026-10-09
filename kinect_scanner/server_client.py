"""HTTP + WebSocket client for communicating with the scanner server."""

import ipaddress
import json
import logging
import os
import shutil
import tempfile
import threading
import time

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
    project_opened = pyqtSignal(dict, str)
    transfer_progress = pyqtSignal(str, float, float)
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
        self._batch_unavailable = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Connection ─────────────────────────────────────────────────────

    def connect_to_server(self, host: str, port: int) -> bool:
        """Connect synchronously; the GUI invokes this through ServerTaskWorker."""
        self.disconnect(notify=False)
        self._base_url = f"http://{host}:{port}"
        logger.info("Connecting server=%s", self._base_url)
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
        self._batch_unavailable = False
        self.session_id = status.get("session_id")
        self.last_status = dict(status)
        logger.info("Connected session=%s stored=%s has_mesh=%s", self.session_id,
                    status.get("stored_count"), status.get("has_mesh"))
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
        logger.info("Disconnect session=%s notify=%s", self.session_id, notify)
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

        last_error = None
        while active():
            sock = None
            try:
                sock = ws_lib.WebSocket()
                sock.settimeout(5.0)
                sock.connect(url)
                if not active():
                    break
                self._ws_socket = sock
                last_error = None
                logger.info("Progress connection established session=%s", self.session_id)
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
                    if str(exc) != last_error:
                        logger.warning("WebSocket error: %s, reconnecting...", exc, extra={"ui_log": False})
                        last_error = str(exc)
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
            logger.info("Server progress session=%s current=%s total=%s message=%s",
                        self.session_id, msg.get("current"), msg.get("total"),
                        msg.get("result", {}).get("message", msg.get("message", "")))
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
            logger.info("Server done session=%s success=%s detail=%s", self.session_id,
                        msg.get("success"), msg.get("detail"))
        elif msg_type == "error":
            logger.error("Server error session=%s %s", self.session_id, msg.get("message"), extra={"ui_log": False})
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
            if not self._batch_unavailable:
                logger.warning("Server lacks batch endpoint, sending individually", extra={"ui_state_key": "batch-upload"})
                self._batch_unavailable = True
            results = [self.send_frame(*frame) for frame in frames]
            return {
                **results[-1],
                "results": results,
                "success": any(r.get("success") for r in results),
            }
        resp.raise_for_status()
        if self._batch_unavailable:
            logger.info("Batch capture upload available again", extra={"ui_event": True, "ui_state_key": "batch-upload"})
            self._batch_unavailable = False
        return resp.json()

    def reset_scan(self, settings=None) -> dict:
        resp = self._http.post("/api/scan/reset", json=settings or {})
        resp.raise_for_status()
        return resp.json()

    def request_build(self, options=None) -> dict:
        """Start a build. Progress comes via WebSocket."""
        resp = self._http.post("/api/scan/build", timeout=600.0, **({"json": options} if options else {}))
        resp.raise_for_status()
        return resp.json()

    def request_preview(self) -> str | None:
        """Request preview, save returned PLY to a temp file, return path."""
        return self._download_preview("POST", "/api/scan/preview")

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
        return self._download_preview("GET", "/api/scan/export/ply")

    def _download_preview(self, method, endpoint):
        started = time.monotonic()
        fd, path = tempfile.mkstemp(suffix=".ply")
        complete = False
        logger.info("Preview download started session=%s endpoint=%s path=%s", self.session_id, endpoint, path)
        try:
            with os.fdopen(fd, "wb") as output:
                with self._http.stream(method, endpoint, timeout=600.0) as resp:
                    resp.raise_for_status()
                    if "octet-stream" not in resp.headers.get("content-type", ""):
                        return None
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    reserve = 256 * 1024**2
                    free = shutil.disk_usage(os.path.dirname(path)).free
                    if total + reserve > free:
                        raise OSError("Not enough free disk space for mesh preview")
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        # Also protect unknown-length responses and concurrent disk use.
                        if shutil.disk_usage(os.path.dirname(path)).free < len(chunk) + reserve:
                            raise OSError("Low disk space; preview download stopped")
                        output.write(chunk)
                        downloaded += len(chunk)
                    if total and downloaded != total:
                        raise OSError("Incomplete preview download")
                output.flush()
            complete = True
            logger.info("Preview download finished session=%s bytes=%s seconds=%.2f path=%s",
                        self.session_id, downloaded, time.monotonic() - started, path)
            return path
        finally:
            if not complete:
                os.unlink(path)

    def request_export(self, fmt: str, save_path: str, options=None) -> bool:
        """Download and atomically replace the destination only after success."""
        destination = os.path.abspath(save_path)
        fd, temporary = tempfile.mkstemp(
            prefix=".scanner-export-", dir=os.path.dirname(destination)
        )
        started = time.monotonic()
        try:
            with os.fdopen(fd, "wb") as output:
                with self._http.stream("GET", f"/api/scan/export/{fmt}",
                                       params=options or {}, timeout=600.0) as resp:
                    resp.raise_for_status()
                    if "octet-stream" not in resp.headers.get("content-type", ""):
                        return False
                    prepared = time.monotonic()
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    last_update = 0
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_update >= .2:
                            self.transfer_progress.emit("Downloading", downloaded, total)
                            last_update = now
                    if total and downloaded != total:
                        raise OSError("Incomplete export download; existing file retained")
                    self.transfer_progress.emit("Downloading", downloaded, total)
                    logger.info("Export %s: preparation %.2f s, download %.2f s, %.1f MiB",
                                fmt, prepared - started, time.monotonic() - prepared, downloaded / 1024**2)
                self.task_started.emit("Finishing save to disk…")
                finishing = time.monotonic()
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            logger.info("Export %s: disk completion %.2f s, total %.2f s: %s", fmt,
                        time.monotonic() - finishing, time.monotonic() - started, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return True

    def request_open_project(self, path: str) -> dict:
        total = os.path.getsize(path)

        def chunks(source):
            sent = 0
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                sent += len(chunk)
                self.transfer_progress.emit("Uploading project", sent, total)
                yield chunk
            self.task_started.emit("Restoring project on server…")

        with open(path, "rb") as source:
            response = self._http.post("/api/scan/project", content=chunks(source),
                                       headers={"Content-Type": "application/zip", "Content-Length": str(total)},
                                       timeout=600.0)
        response.raise_for_status()
        result = response.json()
        if not result.get("success"):
            raise RuntimeError(result.get("message", "Could not open project"))
        self.session_id = result.get("session_id")
        self.last_status = dict(result)
        return result

    def get_status(self) -> dict:
        resp = self._http.get("/api/scan/status", timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def get_reconstruction(self) -> dict:
        resp = self._http.get("/api/scan/diagnostics", timeout=600.0)
        resp.raise_for_status()
        return resp.json()
