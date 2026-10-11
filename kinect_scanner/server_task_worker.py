"""Background task queue dispatching work to ServerClient.

Same pattern as the original ScanTaskManager, but sends requests to the
remote server instead of calling ScanEngine directly.
"""

import logging
import os
import queue
import threading
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

from .capture_spool import CaptureSpool
from .runtime import data_root

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
    COMPACT_PROJECT_INFO = auto()
    EXPORT_BLENDER = auto()
    OPEN_PROJECT = auto()
    SAVE_MESH = auto()
    RESET = auto()
    LIST_RECOVERY = auto()
    RETRY_CAPTURES = auto()
    EXPORT_RECOVERY = auto()


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
        self._spool = None
        self._resource_session_id = None
        self._inflight_captures = 0
        self._failed_uploads = False
        self._retained_spools = {}
        self._capture_submission_lock = threading.RLock()
        self.capture_error = ""

    def submit(self, task: ServerTask):
        # Snapshot checks must not mistake the append->enqueue window for an
        # orphan capture when STATUS lands while the GUI submits a fresh frame.
        with self._capture_submission_lock:
            return self._submit(task)

    def _submit(self, task: ServerTask):
        if self._stop_flag:
            return False
        if task.task_type == ServerTaskType.SEND_FRAME:
            origin = self._spool.session_id if self._spool is not None else self._resource_session_id or self._client.session_id
            task = ServerTask(task.task_type, {**task.kwargs, "capture_session_id": origin})
        if task.task_type == ServerTaskType.SEND_FRAME and self._spool is not None:
            try:
                path = self._spool.append(task.kwargs["rgb"], task.kwargs["depth"],
                                          task.kwargs.get("metadata", {}))
            except (OSError, ValueError) as exc:
                self.capture_error = f"Capture paused: {exc}"
                if self._spool.pending_count or self._spool.ordering_error:
                    self._failed_uploads = True
                logger.error("%s", self.capture_error)
                return False
            task = ServerTask(ServerTaskType.SEND_FRAME, {
                "spool_path": path, "spool": self._spool, "capture_session_id": origin,
            })
            self.capture_error = ""
        elif task.task_type == ServerTaskType.SEND_FRAME and self._queue.qsize() >= 100:
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

    @property
    def pending_capture_count(self):
        matching = self._matching_spools()
        if matching:
            return sum(spool.pending_count for spool in matching)
        if self._spool is not None:
            return self._spool.pending_count
        with self._queue.mutex:
            queued = sum(task is not None and task.task_type == ServerTaskType.SEND_FRAME
                         for task in self._queue.queue)
        queued += bool(self._pending is not None and self._pending.task_type == ServerTaskType.SEND_FRAME)
        return queued + self._inflight_captures

    @property
    def capture_recovery_required(self):
        return self._failed_uploads or any(spool.pending_count for spool in self._matching_spools()
                                           if spool is not self._spool)

    def _matching_spools(self):
        spools = dict(self._retained_spools)
        if self._spool is not None:
            spools[str(self._spool.path)] = self._spool
        return [spool for spool in spools.values() if spool.session_id == self._client.session_id]

    def _discover_recovery(self):
        found = {str(spool.path): spool for spool in CaptureSpool.discover(data_root() / "capture-queue")}
        if self._spool is not None:
            key = str(self._spool.path)
            if self._spool.ordering_error and key in found:
                self._spool = found[key]
            else:
                found[key] = self._spool
        self._retained_spools = found
        rows = [{"path": str(spool.path), "session_id": spool.session_id,
                 "count": spool.pending_count, "retry_safe": spool.retry_safe,
                 "error": spool.ordering_error}
                for spool in found.values() if spool.pending_count or spool.ordering_error]
        self._client.recovery_list.emit(rows)
        return rows

    def _configure_spool(self, settings, session_id, base_index=0):
        with self._capture_submission_lock:
            return self._bind_spool(settings, session_id, base_index)

    def _bind_spool(self, settings, session_id, base_index):
        if self._spool is not None and self._spool.session_id == session_id and not self._live:
            if self._spool.pending_count or self._spool.base_index + self._spool.next_sequence == base_index:
                return
        if self._spool is None and self._live and self._resource_session_id == session_id:
            return
        if self._spool is not None:
            self._spool.cleanup()
        candidates = [spool for spool in self._retained_spools.values()
                      if spool.session_id == session_id and spool.pending_count and spool.retry_safe]
        self._spool = ((candidates[0] if len(candidates) == 1 else CaptureSpool(
            data_root() / "capture-queue", session_id, settings,
            receipts=self._client._capture_receipts_supported, base_index=base_index))
                       if session_id and not self._live else None)
        self._resource_session_id = session_id
        self._failed_uploads = False

    def _restore_session_resources(self, status):
        """Rebind local resources after authoritative connection/status checks.

        Replacing a spool preserves its pending captures and durable stream
        manifest. Same-session checks preserve upload failures and recordings.
        """
        if "settings" not in status:
            return
        session_id = status.get("session_id")
        self._discover_recovery()
        self._live = bool(status["settings"].get("live_reconstruction"))
        self._configure_spool(status["settings"], session_id, status.get("stored_count", 0))
        with self._capture_submission_lock:
            with self._queue.mutex:
                scheduled = {task.kwargs.get("spool_path") for task in self._queue.queue
                             if task is not None and task.task_type == ServerTaskType.SEND_FRAME}
            if self._pending is not None and self._pending.task_type == ServerTaskType.SEND_FRAME:
                scheduled.add(self._pending.kwargs.get("spool_path"))
            if any(path not in scheduled for spool in self._matching_spools() for path in spool.pending_paths):
                self._failed_uploads = True
        if session_id != self._recording_session_id or self._recording is None:
            self._recording = None
            self._recording_session_id = None
            if session_id and self._client._project_export_supported:
                try:
                    self._recording = RecordingWriter.for_session(data_root() / "recordings", session_id)
                except Exception as exc:  # Optional local recovery must not hide authoritative status.
                    self._client.task_error.emit(f"Local recording unavailable: {exc}")
                if self._recording is not None:
                    self._recording_session_id = session_id
                    logger.info("Recovered local capture recording for session=%s path=%s",
                                session_id, self._recording.path)

    def _require_uploaded_captures(self):
        self._discover_recovery()
        if self._failed_uploads or self.pending_capture_count:
            retained = f" Pending captures are retained in {self._spool.path}." if self._spool else ""
            raise RuntimeError("Some captures were not confirmed by the server; build/save stopped." + retained)

    def _local_export_captures(self):
        """Offer acknowledged images only for the current, reuse-capable session."""
        if (self._recording is None
                or self._recording_session_id != self._client.session_id
                or not self._client._project_export_supported):
            return {}
        from shared.project_archive import recording_captures
        self._client.task_started.emit("Checking selected captures in local recording…")
        return recording_captures(self._recording)

    def stop(self):
        self._stop_flag = True
        self._queue.put(None)

    def run(self):
        try:
            self._discover_recovery()
        except (OSError, ValueError) as exc:
            self._client.task_error.emit(f"Could not check retained captures: {exc}")
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
        if self._spool is not None:
            if self._spool.pending_count:
                logger.info("Unconfirmed captures retained count=%s path=%s",
                            self._spool.pending_count, self._spool.path)
            else:
                self._spool.cleanup()

    _MAX_BATCH = 8  # Bound peak RAM and acknowledgement latency while a spool drains.

    def _drain_send_frames(self, first_task: ServerTask, *, recovering=False) -> dict:
        """Batch only consecutive frames, preserving command barriers exactly."""

        def frame(task):
            if "spool_path" in task.kwargs:
                return CaptureSpool.load(task.kwargs["spool_path"])
            return (
                task.kwargs["rgb"],
                task.kwargs["depth"],
                task.kwargs.get("metadata", {}),
            )

        def origin(task):
            return task.kwargs.get("capture_session_id", task.kwargs["spool"].session_id
                                   if "spool" in task.kwargs else self._client.session_id)

        capture_session_id = origin(first_task)
        tasks = [first_task]
        while not recovering and len(tasks) < self._MAX_BATCH:
            try:
                task = self._queue.get_nowait()
            except queue.Empty:
                break
            if task is None:
                self._stop_flag = True
                break
            if task.task_type != ServerTaskType.SEND_FRAME or origin(task) != capture_session_id:
                self._pending = task
                break
            tasks.append(task)
        self._inflight_captures = len(tasks)
        frames = []
        try:
            frames = [frame(task) for task in tasks]
            if capture_session_id != self._client.session_id:
                raise RuntimeError("Capture belongs to a different server session; upload stopped and local spool retained")
            if self._failed_uploads and not recovering:
                raise RuntimeError("Earlier captures need recovery; later captures remain on disk. Choose Recover Captures.")
            logger.info("Uploading captures session=%s count=%s frame_ids=%s", self._client.session_id,
                        len(frames), [metadata.get("frame_id") for _, _, metadata in frames])
            if len(frames) == 1:
                result = self._client.send_frame(*frames[0])
            else:
                result = self._client.send_frames_batch(frames)
            if not isinstance(result, dict):
                raise RuntimeError("Invalid upload confirmation; captures remain on disk")
            if result.get("session_id") and result["session_id"] != capture_session_id:
                raise RuntimeError("Upload acknowledged a different server session; reconnect before continuing. Local spool retained")
        except Exception as exc:
            if capture_session_id == self._client.session_id:
                self._failed_uploads = True
            # Release precisely these captures on a failed transport, before
            # the task failure pauses scanning. A later retry must not wedge.
            self._client.frame_stored.emit({
                "success": False, "message": str(exc),
                "session_id": capture_session_id,
                "capture_acknowledgements": [
                    {"frame_id": metadata.get("frame_id"), "success": False}
                    for _, _, metadata in frames
                ],
            })
            raise
        finally:
            self._inflight_captures = 0
        acknowledgements = result.get("results", [result] if len(frames) == 1 else [])
        if not isinstance(acknowledgements, list) or any(not isinstance(ack, dict) for ack in acknowledgements):
            self._failed_uploads = True
            raise RuntimeError("Invalid capture confirmations; captures remain on disk")
        logger.info("Capture upload session=%s count=%s success=%s stored=%s frame_ids=%s",
                    self._client.session_id, len(frames), result.get("success"), result.get("stored_count"),
                    [metadata.get("frame_id") for _, _, metadata in frames])
        receipt_frames = any(metadata.get("upload_receipt") for _, _, metadata in frames)
        capture_acks = acknowledgements or ([] if receipt_frames else [{"success": result.get("success", False)}] * len(frames))
        if receipt_frames:
            if len(capture_acks) != len(frames):
                self._failed_uploads = True
                raise RuntimeError("Upload confirmation count is incomplete; captures remain on disk")
            from shared.capture_receipts import receipt_for
            verified = []
            for (rgb, depth, metadata), ack in zip(frames, capture_acks):
                if ack.get("success") is False:
                    verified.append(ack)
                    continue
                try:
                    expected = receipt_for(rgb, depth, metadata)
                except (ValueError, TypeError) as exc:
                    self._failed_uploads = True
                    raise RuntimeError("Capture proof is unavailable; originals remain on disk") from exc
                proof = ack.get("capture_receipt")
                if (ack.get("success") is not True or not isinstance(proof, dict)
                        or type(proof.get("sequence")) is not int or type(proof.get("expected_index")) is not int
                        or proof != expected or type(ack.get("index")) is not int
                        or ack.get("index") != expected["expected_index"]
                        or ack.get("session_id") != capture_session_id
                        or result.get("session_id") != capture_session_id):
                    ack = {**ack, "success": False, "message": "Upload confirmation could not be verified; capture retained"}
                verified.append(ack)
            capture_acks = verified
            acknowledgements = verified
        complete = len(capture_acks) == len(tasks) and all(ack.get("success") for ack in capture_acks)
        if not complete:
            self._failed_uploads = True
            result["message"] = next((ack.get("message") for ack in capture_acks
                                      if not ack.get("success") and ack.get("message")),
                                     result.get("message", "Capture upload was not confirmed"))
        result = {
            **result,
            "success": complete,
            "capture_acknowledgements": [
                {**ack, "frame_id": metadata.get("frame_id")}
                for (_, _, metadata), ack in zip(frames, capture_acks)
            ],
        }
        if any(ack.get("success") for ack in capture_acks) and self._recording:
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
            except (OSError, ValueError) as exc:
                logger.exception("Local capture recording stopped path=%s", self._recording.path, extra={"ui_log": False})
                self._recording = None
                self._client.task_error.emit(f"Recording stopped: {exc}")
        for task, ack in zip(tasks, capture_acks):
            if ack.get("success") and "spool_path" in task.kwargs:
                if not task.kwargs["spool"].acknowledge(task.kwargs["spool_path"]):
                    self._failed_uploads = True
                    result.update(success=False, message="Confirmed capture could not be removed locally; choose Recover Captures.")
        return result

    def _dispatch(self, task: ServerTask):
        tt = task.task_type

        if tt == ServerTaskType.CONNECT:
            self._client.task_started.emit("Connecting to server...")
            if not self._client.connect_to_server(task.kwargs["host"], task.kwargs["port"], notify=False):
                self._client.task_failed.emit(tt.name, "Could not connect to server")
            else:
                status = self._client.last_status
                self._restore_session_resources(status)
                self._client.connected.emit()
                self._client.status_updated.emit(status)

        elif tt == ServerTaskType.DISCONNECT:
            self._client.disconnect()

        elif tt == ServerTaskType.STATUS:
            result = self._client.get_status()
            self._client.session_id = result.get("session_id")
            self._client.last_status = dict(result)
            self._restore_session_resources(result)
            self._client.status_updated.emit(result)

        elif tt == ServerTaskType.LIST_RECOVERY:
            self._discover_recovery()

        elif tt == ServerTaskType.RETRY_CAPTURES:
            spool = CaptureSpool.open(task.kwargs["path"])
            if not spool.retry_safe or not self._client._capture_receipts_supported or not self._client.is_connected:
                raise RuntimeError("This capture buffer cannot be retried safely; save a recovery project instead")
            status = self._client.get_status()
            if status.get("session_id") != spool.session_id or status.get("operation"):
                raise RuntimeError("Connect to the matching idle scan before retrying these captures")
            self._client.session_id = status["session_id"]
            self._client.last_status = dict(status)
            self._retained_spools[str(spool.path)] = spool
            if self._spool is not None and self._spool.path == spool.path:
                self._spool = spool
            paths = spool.pending_paths
            for index, path in enumerate(paths):
                self._client.task_started.emit(f"Recovering capture {index + 1} of {len(paths)}…")
                result = self._drain_send_frames(ServerTask(ServerTaskType.SEND_FRAME, {
                    "spool_path": path, "spool": spool, "capture_session_id": spool.session_id,
                }), recovering=True)
                self._client.frame_stored.emit(result)
                if not result.get("success") or path.exists():
                    reason = result.get("message", "Capture recovery was not confirmed")
                    raise RuntimeError(reason + "; original captures remain on disk. "
                                       "Save a recovery project if this scan cannot accept them")
            self._discover_recovery()
            self._failed_uploads = bool(self.pending_capture_count)
            status = self._client.get_status()
            if status.get("session_id") != spool.session_id:
                self._failed_uploads = True
                raise RuntimeError("Scan changed during recovery; reconnect before continuing")
            self._client.last_status = dict(status)
            if not self.pending_capture_count and not self._live:
                self._spool = CaptureSpool(data_root() / "capture-queue", spool.session_id, status["settings"],
                    receipts=True, base_index=status.get("stored_count", 0))
            self._client.status_updated.emit(status)
            self._client.recovery_done.emit({"session_id": spool.session_id, "recovered": len(paths),
                                             "remaining": self.pending_capture_count})

        elif tt == ServerTaskType.EXPORT_RECOVERY:
            from .capture_recovery import export_recovery_project
            spool = CaptureSpool.open(task.kwargs["path"])
            self._client.task_started.emit("Saving retained captures to a recovery project…")
            destination = export_recovery_project(spool, task.kwargs["destination"])
            self._client.recovery_export_done.emit(str(destination))

        elif tt == ServerTaskType.COMPACT_PROJECT_INFO:
            self._require_uploaded_captures()
            self._client.task_started.emit("Reviewing project contributions…")
            self._client.compact_project_info.emit(self._client.request_compact_project_info())

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
            self._configure_spool(result["settings"], result.get("session_id"))
            if task.kwargs.get("record"):
                root = data_root() / "recordings"
                path = root / datetime.now(timezone.utc).astimezone().strftime("scan-%Y%m%d-%H%M%S-%f")
                try:
                    self._recording = RecordingWriter(path, result["settings"], session_id=result.get("session_id"))
                    self._recording_session_id = result.get("session_id")
                    logger.info("Local capture recording started session=%s path=%s", self._recording_session_id, path)
                except Exception as exc:  # noqa: BLE001 — optional recording must not undo a committed reset.
                    self._client.task_error.emit(f"Local recording unavailable: {exc}")
            # The server already committed the new session, even if local disk setup failed.
            logger.info("Scan reset session=%s settings=%s", self._client.session_id, result["settings"])
            self._client.reset_done.emit(result)

        elif tt == ServerTaskType.BUILD_MESH:
            self._require_uploaded_captures()
            recorder = task.kwargs.get("sensor_recorder")
            if recorder is not None:
                from shared.motion_journal import acceleration_journal
                snapshot = recorder.completed_sensor_snapshot(task.kwargs.get("sensor_path"))
                if snapshot is not None:
                    self._client.task_started.emit("Uploading complete acceleration journal…")
                    self._client.send_motion_journal(acceleration_journal(snapshot))
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
            self._configure_spool(result["settings"], result.get("session_id"), result.get("stored_count", 0))
            self._client.project_opened.emit(result, path)

        elif tt == ServerTaskType.PREVIEW:
            self._require_uploaded_captures()
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

        elif tt == ServerTaskType.EXPORT_BLENDER:
            self._require_uploaded_captures()
            from .blender_export import convert_session, _require_disk_space

            path = task.kwargs["path"]
            _require_disk_space(Path(path).resolve().parent)
            with tempfile.TemporaryDirectory(prefix="scanner-blender-") as temporary:
                session = Path(temporary) / "session.zip"
                model = Path(temporary) / "model.zip"
                for fmt, destination, options in (("session", session, None),
                                                  ("obj.zip", model, task.kwargs.get("options"))):
                    self._client.task_started.emit("Downloading RGB-D session…" if fmt == "session" else "Preparing textured fused mesh…")
                    download_options = {"options": options}
                    if fmt == "session":
                        local = self._local_export_captures()
                        if local:
                            download_options["local_captures"] = local
                    if not self._client.request_export(fmt, str(destination), **download_options):
                        raise RuntimeError(f"Could not download {fmt} for the Blender export")
                result = convert_session(session, path, blender=task.kwargs["blender"],
                                         textured_model=model, stride=task.kwargs["stride"],
                                         surface=task.kwargs.get("surface", "smooth"),
                                         progress=self._client.task_started.emit)
                logger.info("Blender export: %s", result, extra={"ui_event": True})
            self._client.export_done.emit(True, path)

        elif tt in (ServerTaskType.EXPORT_TEXTURE, ServerTaskType.EXPORT_SESSION):
            self._require_uploaded_captures()
            path = task.kwargs["path"]
            fmt = task.kwargs.get("format", "session")
            download_options = {"options": task.kwargs.get("options")}
            compact = bool((task.kwargs.get("options") or {}).get("compact"))
            if tt == ServerTaskType.EXPORT_SESSION and not compact:
                local = self._local_export_captures()
                if local:
                    download_options["local_captures"] = local
            self._client.task_started.emit(f"Exporting {fmt} to {path}...")
            snapshot = None
            recorder = task.kwargs.get("sensor_recorder")
            if tt == ServerTaskType.EXPORT_SESSION and recorder is not None:
                self._client.task_started.emit("Finishing sensor recording…")
                snapshot = recorder.flush_sensor_recording(task.kwargs.get("sensor_path"), stop=True)
            self._client.task_started.emit("Preparing project on server…" if tt == ServerTaskType.EXPORT_SESSION else "Preparing textured model…")
            if snapshot is None:
                success = self._client.request_export(fmt, path, **download_options)
            else:
                # Keep an existing complete session intact if downloading or
                # merging the local observations fails.
                fd, temporary = tempfile.mkstemp(prefix=".sensor-session-", suffix=".zip", dir=Path(path).resolve().parent)
                os.close(fd)
                try:
                    success = self._client.request_export(fmt, temporary, **download_options)
                    if success:
                        self._client.task_started.emit("Adding sensor recording…")
                        started = time.monotonic()
                        if compact:
                            augment_session_archive(temporary, snapshot, portrait=False)
                        else:
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
