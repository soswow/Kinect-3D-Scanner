"""Camera supervisor with bounded waits and recoverable native-driver stalls."""

import logging
import multiprocessing
import queue
import threading
import time
import uuid

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from shared.capture import validate_rgb_exposure
from shared.sensor_recording import journal_snapshot
from shared.visual_tracking import VisualTracker

from .capture_process import DEPTH_SHAPE, RGB_SHAPE, capture_frames

logger = logging.getLogger(__name__)


class KinectWorker(QThread):
    """Keep libfreenect in a child process so USB failures cannot hang Qt."""

    frame_ready = pyqtSignal(np.ndarray, np.ndarray)
    frame_pair_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    frame_available = pyqtSignal()
    error_occurred = pyqtSignal(str)
    accelerometer_ready = pyqtSignal(dict)
    sensor_recording_status = pyqtSignal(dict)

    def __init__(
        self,
        parent=None,
        *,
        capture_target=capture_frames,
        startup_timeout=8.0,
        frame_timeout=3.0,
        retry_delay=2.0,
        rgb_mode="rgb_high_res",
        rgb_exposure_mode="auto",
        rgb_shutter_speed=125,
        rgb_gain=1,
        accelerometer_calibration=None,
    ):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self._finishing = threading.Event()
        self._capture_target = capture_target
        self._startup_timeout = startup_timeout
        self._frame_timeout = frame_timeout
        self._retry_delay = retry_delay
        if rgb_mode not in ("rgb_high_res", "rgb_low_res"):
            raise ValueError("RGB mode must be rgb_high_res or rgb_low_res")
        self._high_res = rgb_mode == "rgb_high_res"
        validate_rgb_exposure(rgb_exposure_mode, rgb_shutter_speed, rgb_mode, rgb_gain)
        self._rgb_exposure_mode = rgb_exposure_mode
        self._rgb_shutter_speed = rgb_shutter_speed
        self._rgb_gain = rgb_gain
        from shared.inertial import calibration_profile
        self._accelerometer_calibration = calibration_profile(accelerometer_calibration)
        self._rgb_shape = (1024, 1280, 3) if self._high_res else RGB_SHAPE
        self._tracking_lock = threading.Lock()
        self._tracking_request = (0, None)
        self._tracking_debug = False
        self._recording_request = (0, None)
        self._control_queue = None
        self._flush_condition = threading.Condition()
        self._flush_results = {}
        self._recording_control_errors = {}
        self._accelerometer_enabled = True
        self.coalesce_frames = False
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self.preview_frames_replaced = 0

    def take_latest_frame(self):
        with self._frame_lock:
            frame, self._latest_frame = self._latest_frame, None
        return frame

    def _deliver_frame(self, rgb, depth, metadata):
        if self._finishing.is_set():
            return
        if not self.coalesce_frames:
            self.frame_pair_ready.emit(rgb, depth, metadata)
            self.frame_ready.emit(rgb, depth)
            return
        # Queue one tiny notification, never an unbounded queue of image arrays.
        # Full sensor recordings remain independent in the acquisition process.
        with self._frame_lock:
            notify = self._latest_frame is None
            if not notify:
                self.preview_frames_replaced += 1
            self._latest_frame = (rgb, depth, metadata)
        if notify:
            self.frame_available.emit()

    def set_sensor_recording(self, path, settings=None):
        configuration = {"path": str(path), "settings": settings} if path else None
        with self._tracking_lock:
            if configuration != self._recording_request[1]:
                self._recording_request = (self._recording_request[0] + 1, configuration)

    def record_sensor_event(self, event):
        with self._tracking_lock:
            controls = self._control_queue
            configuration = self._recording_request[1]
        if controls is not None:
            try:
                controls.put(("event", event), timeout=0.2)
            except (queue.Full, OSError, ValueError):
                if configuration:
                    with self._tracking_lock:
                        self._recording_control_errors[configuration["path"]] = "Sensor control event could not be recorded"
                self.sensor_recording_status.emit({"complete": False, "error": "Sensor control event could not be recorded"})

    def flush_sensor_recording(self, path=None, timeout=8, stop=False):
        """Called by the task thread; the USB child checkpoints a FIFO barrier."""
        with self._tracking_lock:
            configuration = self._recording_request[1]
            controls = self._control_queue
        root = str(path) if path is not None else configuration["path"] if configuration else None
        if root is None:
            return None
        def snapshot(active=None):
            result = journal_snapshot(root, active)
            with self._tracking_lock:
                error = self._recording_control_errors.get(root)
            if error:
                result.update(complete=False, control_error=error)
            return result
        if controls is None:
            return snapshot()
        token = uuid.uuid4().hex
        if stop:
            controls.put(("stop_recording", root), timeout=0.2)
        controls.put(("flush", {"id": token, "root": root}), timeout=0.2)
        with self._flush_condition:
            ready = self._flush_condition.wait_for(
                lambda: token in self._flush_results or self._stop_event.is_set(), timeout)
            result = self._flush_results.pop(token, None)
        if not ready or result is None:
            raise RuntimeError("Sensor recording did not acknowledge its checkpoint")
        return snapshot(result)

    def set_tracking_settings(self, settings):
        """Hand immutable session settings to the camera thread; never run in Qt."""
        with self._tracking_lock:
            self._tracking_request = (self._tracking_request[0] + 1, settings)

    def set_tracking_debug(self, enabled):
        """Toggle preview diagnostics without restarting the motion chain."""
        with self._tracking_lock:
            self._tracking_debug = bool(enabled)

    def stop(self):
        self._stop_event.set()
        with self._flush_condition:
            self._flush_condition.notify_all()

    def finish_capture(self, path=None):
        """Stop delivery now, checkpoint recording, then shut down USB off the UI thread."""
        if self._finishing.is_set():
            return
        self._finishing.set()
        self.take_latest_frame()

        def finish():
            logger.info("Camera shutdown requested recording=%s", path)
            try:
                snapshot = self.flush_sensor_recording(path, stop=True)
                if snapshot is not None:
                    logger.info("Final sensor checkpoint complete=%s root=%s", snapshot["complete"], snapshot["root"])
                    self.sensor_recording_status.emit(snapshot)
            except Exception as exc:  # Shutdown must proceed even if the camera/disk failed.
                logger.exception("Final sensor checkpoint failed")
                self.sensor_recording_status.emit({"complete": False, "error": str(exc)})
            finally:
                self.stop()

        self._finish_thread = threading.Thread(target=finish, name="Finish camera recording", daemon=True)
        self._finish_thread.start()

    @staticmethod
    def _stop_capture(process, stop_event):
        stop_event.set()
        process.join(timeout=0.3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.5)
        if not process.is_alive():
            process.close()

    def run(self):
        # Never fork the already running Qt/Open3D runtime.
        context = multiprocessing.get_context("spawn")
        sequence = 0
        last_camera_error = last_visual_error = None
        while not self._stop_event.is_set():
            tracker, tracking_generation = None, -1
            recording_generation = -1
            acceleration_call = False
            parent, child = context.Pipe()
            stop_event = context.Event()
            controls = context.Queue(maxsize=16)
            with self._tracking_lock:
                self._control_queue = controls if self._capture_target is capture_frames else None
            rgb_buffer = context.RawArray("B", int(np.prod(self._rgb_shape)))
            depth_buffer = context.RawArray("H", int(np.prod(DEPTH_SHAPE)))
            process = context.Process(
                target=self._capture_target,
                args=(child, stop_event, rgb_buffer, depth_buffer)
                + ((self._high_res, self._rgb_exposure_mode, self._rgb_shutter_speed, self._rgb_gain,
                    controls, self._accelerometer_enabled, self._accelerometer_calibration)
                   if self._capture_target is capture_frames else ()),
                daemon=True,
                name="Kinect capture",
            )
            started = False
            try:
                process.start()
                started = True
                child.close()
                deadline = time.monotonic() + self._startup_timeout
                streaming = False
                while not self._stop_event.is_set():
                    with self._tracking_lock:
                        requested_generation, configuration = self._recording_request
                    if self._capture_target is capture_frames and requested_generation != recording_generation:
                        controls.put(("record", configuration), timeout=0.2)
                        recording_generation = requested_generation
                    # Acceleration/status traffic cannot keep a stalled image
                    # stream alive indefinitely.
                    if time.monotonic() > deadline:
                        raise RuntimeError("Kinect stopped delivering RGB/depth frames. Retrying camera; check USB connection and external power.")
                    if parent.poll(0.1):
                        kind, payload = parent.recv()
                        if kind == "phase":
                            continue
                        if kind == "error":
                            raise RuntimeError(payload)
                        if kind == "accelerometer_poll":
                            acceleration_call = True
                            continue
                        if kind == "accelerometer":
                            acceleration_call = False
                            latency = payload.get("read_end_s", 0) - payload.get("read_start_s", 0)
                            if not payload.get("valid") or latency > .05:
                                logger.warning("Accelerometer read sequence=%s valid=%s latency_ms=%.1f reason=%s",
                                               payload.get("sequence"), payload.get("valid"), latency * 1000,
                                               payload.get("reason", payload.get("gravity", {}).get("reason")))
                            self.accelerometer_ready.emit(payload)
                            continue
                        if kind == "sensor_status":
                            self.sensor_recording_status.emit(payload)
                            continue
                        if kind == "sensor_flush":
                            with self._flush_condition:
                                self._flush_results[payload["request_id"]] = payload
                                self._flush_condition.notify_all()
                            continue
                        if kind != "frame":
                            raise RuntimeError("Invalid camera process message")
                        if self._finishing.is_set():
                            # Keep the control/flush handshake moving without copying or tracking images.
                            parent.send("copied")
                            deadline = time.monotonic() + self._frame_timeout
                            continue
                        rgb = (
                            np.frombuffer(rgb_buffer, np.uint8)
                            .reshape(self._rgb_shape)
                            .copy()
                        )
                        depth = (
                            np.frombuffer(depth_buffer, np.uint16)
                            .reshape(DEPTH_SHAPE)
                            .copy()
                        )
                        parent.send("copied")
                        sequence += 1
                        metadata = dict(payload, frame_id=sequence)
                        with self._tracking_lock:
                            generation, settings = self._tracking_request
                            tracking_debug = self._tracking_debug
                        if generation != tracking_generation:
                            tracker = VisualTracker(settings) if settings is not None else None
                            tracking_generation = generation
                        if tracker is not None:
                            # The driver already owns the next shared buffer.
                            # Tracking uses our copies, independently of HTTP/fusion.
                            try:
                                tracker.debug_enabled = tracking_debug
                                metadata["visual_tracking"] = tracker.update(rgb, depth, metadata)
                                if last_visual_error is not None and metadata["visual_tracking"].get("valid"):
                                    logger.info("Visual motion estimate recovered", extra={"ui_event": True, "ui_state_key": "visual-estimate"})
                                    last_visual_error = None
                                if tracking_debug and tracker.debug_snapshot is not None:
                                    metadata["_tracking_debug"] = tracker.debug_snapshot
                            except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
                                if str(exc) != last_visual_error:
                                    logger.warning("Visual motion estimate failed: %s", exc,
                                                   extra={"ui_state_key": "visual-estimate"})
                                    last_visual_error = str(exc)
                                tracker.reset()
                                metadata["visual_tracking"] = {
                                    "valid": False, "reason": "Visual estimate unavailable; chain reset"}
                        if not streaming:
                            logger.info(
                                "Kinect %s RGB/raw-depth stream ready",
                                self._rgb_shape,
                            )
                            streaming = True
                            last_camera_error = None
                        self._deliver_frame(rgb, depth, metadata)
                        deadline = time.monotonic() + self._frame_timeout
                    elif not process.is_alive():
                        raise RuntimeError("Kinect camera process stopped unexpectedly")
                    elif time.monotonic() > deadline:
                        raise RuntimeError(
                            "Kinect stopped delivering RGB/depth frames. "
                            "Retrying camera; "
                            "check USB connection and external power."
                        )
            except Exception as exc:  # noqa: BLE001 -- acquisition boundary must recover
                if self._finishing.is_set():
                    self.stop()  # Never retry USB after Finish, including a failed checkpoint.
                if not self._stop_event.is_set():
                    message = str(exc) or "Kinect camera process disconnected"
                    if message != last_camera_error:
                        logger.warning("Camera acquisition failed: %s", message, extra={"ui_log": False})
                        last_camera_error = message
                    self.error_occurred.emit(message)
            finally:
                if acceleration_call:
                    self._accelerometer_enabled = False
                    logger.warning("Disabling acceleration after a stalled native sensor read")
                if started:
                    self._stop_capture(process, stop_event)
                    if self._finishing.is_set():
                        logger.info("Camera process stopped after Finish")
                parent.close()
                child.close()
                with self._tracking_lock:
                    self._control_queue = None
                controls.cancel_join_thread()
                controls.close()
            self._stop_event.wait(self._retry_delay)
