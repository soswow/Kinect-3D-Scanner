"""Background task queue dispatching work to ServerClient.

Same pattern as the original ScanTaskManager, but sends requests to the
remote server instead of calling ScanEngine directly.
"""

import queue
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread

from shared.recording import RecordingWriter


class ServerTaskType(Enum):
    SEND_FRAME = auto()
    BUILD_MESH = auto()
    PREVIEW = auto()
    EXPORT_PLY = auto()
    EXPORT_OBJ = auto()
    EXPORT_TEXTURE = auto()
    EXPORT_SESSION = auto()
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
        self._live = False

    def submit(self, task: ServerTask):
        if task.task_type == ServerTaskType.SEND_FRAME and self._queue.qsize() >= 100:
            return False
        self._queue.put(task)
        return True

    def stop(self):
        self._stop_flag = True
        self._queue.put(None)

    def run(self):
        while not self._stop_flag:
            task = self._pending if self._pending is not None else self._queue.get()
            self._pending = None
            if task is None:
                break
            try:
                self._dispatch(task)
            except Exception as exc:
                traceback.print_exc()
                self._client.task_error.emit(f"{task.task_type.name}: {exc}")

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
        if len(frames) == 1:
            result = self._client.send_frame(*frames[0])
        else:
            result = self._client.send_frames_batch(frames)
        if result.get("success") and self._recording:
            try:
                acknowledgements = result.get(
                    "results", [result] if len(frames) == 1 else []
                )
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
                self._recording = None
                self._client.task_error.emit(f"Recording stopped: {exc}")
        return result

    def _dispatch(self, task: ServerTask):
        tt = task.task_type

        if tt == ServerTaskType.SEND_FRAME:
            result = self._drain_send_frames(task)
            self._client.frame_stored.emit(result)

        elif tt == ServerTaskType.RESET:
            self._client.task_started.emit("Resetting scan on server...")
            result = self._client.reset_scan(task.kwargs.get("settings"))
            if not result.get("success"):
                raise RuntimeError(result.get("message", "Reset failed"))
            self._recording = None
            self._live = result["settings"].get("live_reconstruction", False)
            if task.kwargs.get("record"):
                root = Path(__file__).resolve().parents[1] / "recordings"
                path = root / datetime.now().strftime("scan-%Y%m%d-%H%M%S-%f")
                self._recording = RecordingWriter(path, result["settings"])
            self._client.reset_done.emit(result)

        elif tt == ServerTaskType.BUILD_MESH:
            self._client.task_started.emit("Building mesh on server...")
            result = self._client.request_build()
            self._save_reconstruction()
            success = result.get("success", False)
            detail = result.get("detail", "Build complete")
            # Only the final-build HTTP response enables exports.
            self._client.build_mesh_done.emit(success, detail)

        elif tt == ServerTaskType.PREVIEW:
            self._client.task_started.emit("Generating preview on server...")
            path = self._client.request_preview()
            self._save_reconstruction()
            if path:
                self._client.preview_done.emit(path)
            else:
                self._client.task_error.emit("Preview failed — no geometry")

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
            success = self._client.request_export(fmt, path)
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
            except Exception as exc:
                self._client.task_error.emit(
                    f"Could not save reconstruction report: {exc}"
                )
