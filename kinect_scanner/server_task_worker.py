"""Background task queue dispatching work to ServerClient.

Same pattern as the original ScanTaskManager, but sends requests to the
remote server instead of calling ScanEngine directly.
"""

import logging
import os
import queue
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread

from shared.recording import RecordingWriter
from shared.sensor_recording import augment_session_archive

logger = logging.getLogger(__name__)


class ServerTaskType(Enum):
    CONNECT = auto()
    DISCONNECT = auto()
    STATUS = auto()
    FINAL_PREVIEW = auto()
    SEND_FRAME = auto()
    BUILD_MESH = auto()
    PREVIEW = auto()
    EXPORT_PLY = auto()
    EXPORT_OBJ = auto()
    EXPORT_TEXTURE = auto()
    EXPORT_SESSION = auto()
    OPEN_PROJECT = auto()
    SAVE_MESH = auto()
    RESET = auto()


@dataclass
class ServerTask:
    task_type: ServerTaskType
    kwargs: dict[str, Any] = field(default_factory=dict)


class ServerTaskWorker(QThread):
    """Sequential task runner backed by a thread-safe queue.

    Dispatches to ServerClient methods instead of local ScanEngine.
    """

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self._client = client
        self._queue: queue.Queue[ServerTask | None] = queue.Queue()
        self._stop_flag = False
        self._pending = None
        self._recording = None
        self._recording_session_id = None
        self._live = False
        self._queue_full = False

    def submit(self, task: ServerTask):
        if self._stop_flag:
            return False
        if task.task_type == ServerTaskType.SEND_FRAME and self._queue.qsize() >= 100:
            if not self._queue_full:
                logger.warning("Capture rejected: upload queue full session=%s", self._client.session_id,
                               extra={"ui_state_key": "upload-queue"})
                self._queue_full = True
            return False
        if task.task_type != ServerTaskType.SEND_FRAME:
            logger.info("Task queued task=%s session=%s queue=%s", task.task_type.name,
                        self._client.session_id, self._queue.qsize())
        if task.task_type == ServerTaskType.SEND_FRAME and self._queue_full:
            logger.info("Capture upload queue has space again", extra={"ui_event": True, "ui_state_key": "upload-queue"})
            self._queue_full = False
        self._queue.put(task)
        return True

    @property
    def queued_task_count(self):
        return self._queue.qsize()

    def stop(self):
        self._stop_flag = True
        self._queue.put(None)

    def run(self):
        while not self._stop_flag:
            task = self._pending if self._pending is not None else self._queue.get()
            self._pending = None
            if task is None:
                break
            started = time.monotonic()
            logger.info("Task started task=%s session=%s queue=%s", task.task_type.name,
                        self._client.session_id, self._queue.qsize())
            try:
                self._dispatch(task)
            except Exception as exc:  # Contain failures at the queued task boundary.
                logger.exception("Task exception task=%s session=%s", task.task_type.name, self._client.session_id, extra={"ui_log": False})
                self._client.task_error.emit(f"{task.task_type.name}: {exc}")
                self._client.task_failed.emit(task.task_type.name, str(exc))
            finally:
                logger.info("Task ended task=%s session=%s seconds=%.2f queue=%s", task.task_type.name,
                            self._client.session_id, time.monotonic() - started, self._queue.qsize())

    _MAX_BATCH = 100  # max frames per HTTP request

    def _drain_send_frames(self, first_task: ServerTask) -> dict:
        """Batch only consecutive frames, preserving command barriers exactly."""

        def frame(task):
            return (
                task.kwargs["rgb"],
                task.kwargs["depth"],
                task.kwargs.get("metadata", {}),
            )

        frames = [frame(first_task)]
        while len(frames) < (8 if self._live else self._MAX_BATCH):
            try:
                task = self._queue.get_nowait()
            except queue.Empty:
                break
            if task is None:
                self._stop_flag = True
                break
            if task.task_type != ServerTaskType.SEND_FRAME:
                self._pending = task
                break
            frames.append(frame(task))
        logger.info("Uploading captures session=%s count=%s frame_ids=%s", self._client.session_id,
                    len(frames), [metadata.get("frame_id") for _, _, metadata in frames])
        try:
            if len(frames) == 1:
                result = self._client.send_frame(*frames[0])
            else:
                result = self._client.send_frames_batch(frames)
        except Exception as exc:
            # Release precisely these captures on a failed transport, before
            # the task failure pauses scanning. A later retry must not wedge.
            self._client.frame_stored.emit({
                "success": False, "message": str(exc),
                "session_id": self._client.session_id,
                "capture_acknowledgements": [
                    {"frame_id": metadata.get("frame_id"), "success": False}
                    for _, _, metadata in frames
                ],
            })
            raise
        acknowledgements = result.get("results", [result] if len(frames) == 1 else [])
        logger.info("Capture upload session=%s count=%s success=%s stored=%s frame_ids=%s",
                    self._client.session_id, len(frames), result.get("success"), result.get("stored_count"),
                    [metadata.get("frame_id") for _, _, metadata in frames])
        capture_acks = acknowledgements or [{"success": result.get("success", False)}] * len(frames)
        result = {
            **result,
            "capture_acknowledgements": [
                {**ack, "frame_id": metadata.get("frame_id")}
                for (_, _, metadata), ack in zip(frames, capture_acks)
            ],
        }
        if result.get("success") and self._recording:
            try:
                if not acknowledgements:
                    # Older servers cannot identify partial-batch acceptance.
                    # Preserve captures, but do not claim an index mapping.
                    acknowledgements = [{"success": True}] * len(frames)
                for frame, ack in zip(frames, acknowledgements):
                    if ack.get("success"):
                        rgb, depth, metadata = frame
                        metadata = dict(metadata)
                        if "index" in ack:
                            metadata["server_index"] = ack["index"]
                        self._recording.append(rgb, depth, metadata)
            except OSError as exc:
                logger.exception("Local capture recording stopped path=%s", self._recording.path, extra={"ui_log": False})
                self._recording = None
                self._client.task_error.emit(f"Recording stopped: {exc}")
        return result

    def _dispatch(self, task: ServerTask):
        tt = task.task_type

        if tt == ServerTaskType.CONNECT:
            self._client.task_started.emit("Connecting to server...")
            if not self._client.connect_to_server(task.kwargs["host"], task.kwargs["port"]):
                self._client.task_failed.emit(tt.name, "Could not connect to server")
            else:
                status = self._client.last_status
                self._live = bool(status.get("settings", {}).get("live_reconstruction"))
                if status.get("session_id") != self._recording_session_id:
                    self._recording = None
                    self._recording_session_id = None

        elif tt == ServerTaskType.DISCONNECT:
            self._client.disconnect()

        elif tt == ServerTaskType.STATUS:
            result = self._client.get_status()
            self._client.session_id = result.get("session_id")
            self._client.last_status = dict(result)
            self._client.status_updated.emit(result)

        elif tt == ServerTaskType.FINAL_PREVIEW:
            self._client.task_started.emit("Downloading final mesh preview...")
            path = self._client.request_final_preview()
            if not path:
                raise RuntimeError("Final preview failed — no final mesh")
            self._client.final_preview_done.emit(path)

        elif tt == ServerTaskType.SEND_FRAME:
            result = self._drain_send_frames(task)
            self._client.frame_stored.emit(result)

        elif tt == ServerTaskType.RESET:
            self._client.task_started.emit("Resetting scan on server...")
            result = self._client.reset_scan(task.kwargs.get("settings"))
            if not result.get("success"):
                raise RuntimeError(result.get("message", "Reset failed"))
            self._recording = None
            self._recording_session_id = None
            self._client.session_id = result.get("session_id")
            self._live = result["settings"].get("live_reconstruction", False)
            if task.kwargs.get("record"):
                from .runtime import data_root

                root = data_root() / "recordings"
                path = root / datetime.now(timezone.utc).astimezone().strftime("scan-%Y%m%d-%H%M%S-%f")
                try:
                    self._recording = RecordingWriter(path, result["settings"])
                    self._recording_session_id = result.get("session_id")
                    logger.info("Local capture recording started session=%s path=%s", self._recording_session_id, path)
                except Exception as exc:  # noqa: BLE001 — optional recording must not undo a committed reset.
                    self._client.task_error.emit(f"Local recording unavailable: {exc}")
            # The server already committed the new session, even if local disk setup failed.
            logger.info("Scan reset session=%s settings=%s", self._client.session_id, result["settings"])
            self._client.reset_done.emit(result)

        elif tt == ServerTaskType.BUILD_MESH:
            self._client.task_started.emit("Building mesh on server...")
            options = task.kwargs.get("options")
            result = self._client.request_build(options) if options else self._client.request_build()
            self._save_reconstruction()
            success = result.get("success", False)
            detail = result.get("detail", "Build complete")
            # Only the final-build HTTP response enables exports.
            self._client.build_mesh_done.emit(success, detail)

        elif tt == ServerTaskType.OPEN_PROJECT:
            path = task.kwargs["path"]
            self._client.task_started.emit("Opening project…")
            result = self._client.request_open_project(path)
            self._recording = None
            self._recording_session_id = None
            self._live = result["settings"].get("live_reconstruction", False)
            self._client.project_opened.emit(result, path)

        elif tt == ServerTaskType.PREVIEW:
            self._client.task_started.emit("Generating preview on server...")
            path = self._client.request_preview()
            self._save_reconstruction()
            if path:
                self._client.preview_done.emit(path)
            else:
                raise RuntimeError("Preview failed — no geometry")

        elif tt == ServerTaskType.EXPORT_PLY:
            path = task.kwargs["path"]
            self._client.task_started.emit(f"Downloading PLY to {path}...")
            success = self._client.request_export("ply", path)
            self._client.export_done.emit(success, path)

        elif tt == ServerTaskType.EXPORT_OBJ:
            path = task.kwargs["path"]
            self._client.task_started.emit(f"Downloading OBJ to {path}...")
            success = self._client.request_export("obj", path)
            self._client.export_done.emit(success, path)

        elif tt in (ServerTaskType.EXPORT_TEXTURE, ServerTaskType.EXPORT_SESSION):
            path = task.kwargs["path"]
            fmt = task.kwargs.get("format", "session")
            self._client.task_started.emit(f"Exporting {fmt} to {path}...")
            snapshot = None
            recorder = task.kwargs.get("sensor_recorder")
            if tt == ServerTaskType.EXPORT_SESSION and recorder is not None:
                self._client.task_started.emit("Finishing sensor recording…")
                snapshot = recorder.flush_sensor_recording(task.kwargs.get("sensor_path"), stop=True)
            self._client.task_started.emit("Preparing project on server…" if tt == ServerTaskType.EXPORT_SESSION else "Preparing textured model…")
            if snapshot is None:
                success = self._client.request_export(fmt, path, options=task.kwargs.get("options"))
            else:
                # Keep an existing complete session intact if downloading or
                # merging the local observations fails.
                fd, temporary = tempfile.mkstemp(prefix=".sensor-session-", suffix=".zip", dir=Path(path).resolve().parent)
                os.close(fd)
                try:
                    success = self._client.request_export(fmt, temporary, options=task.kwargs.get("options"))
                    if success:
                        self._client.task_started.emit("Adding sensor recording…")
                        started = time.monotonic()
                        augment_session_archive(temporary, snapshot)
                        logging.getLogger(__name__).info("Sensor archive merge: %.2f s", time.monotonic() - started)
                        Path(temporary).replace(path)
                        if not snapshot["complete"]:
                            self._client.task_error.emit("Session saved with incomplete sensor recording; see sensor_archive and per-stream drop/error reports.")
                finally:
                    Path(temporary).unlink(missing_ok=True)
            self._client.export_done.emit(success, path)

        elif tt == ServerTaskType.SAVE_MESH:
            path = task.kwargs["path"]
            self._client.task_started.emit(f"Saving mesh to {path}...")
            success = self._client.request_export("ply", path)
            self._client.save_mesh_done.emit(success, path)

    def _save_reconstruction(self):
        if self._recording is not None:
            try:
                self._recording.save_reconstruction(self._client.get_reconstruction())
            except Exception as exc:  # noqa: BLE001 — a report failure must not hide a completed reconstruction.
                self._client.task_error.emit(
                    f"Could not save reconstruction report: {exc}"
                )
