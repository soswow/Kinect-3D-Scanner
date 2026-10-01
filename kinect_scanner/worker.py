"""Camera supervisor with bounded waits and recoverable native-driver stalls."""

import logging
import multiprocessing
import threading
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .capture_process import DEPTH_SHAPE, RGB_SHAPE, capture_frames

logger = logging.getLogger(__name__)


class KinectWorker(QThread):
    """Keep libfreenect in a child process so USB failures cannot hang Qt."""

    frame_ready = pyqtSignal(np.ndarray, np.ndarray)
    frame_pair_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        *,
        capture_target=capture_frames,
        startup_timeout=8.0,
        frame_timeout=3.0,
        retry_delay=2.0,
    ):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self._capture_target = capture_target
        self._startup_timeout = startup_timeout
        self._frame_timeout = frame_timeout
        self._retry_delay = retry_delay

    def stop(self):
        self._stop_event.set()

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
        while not self._stop_event.is_set():
            parent, child = context.Pipe()
            stop_event = context.Event()
            rgb_buffer = context.RawArray("B", int(np.prod(RGB_SHAPE)))
            depth_buffer = context.RawArray("H", int(np.prod(DEPTH_SHAPE)))
            process = context.Process(
                target=self._capture_target,
                args=(child, stop_event, rgb_buffer, depth_buffer),
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
                    if parent.poll(0.1):
                        kind, payload = parent.recv()
                        if kind == "error":
                            raise RuntimeError(payload)
                        if kind != "frame":
                            raise RuntimeError("Invalid camera process message")
                        rgb = (
                            np.frombuffer(rgb_buffer, np.uint8)
                            .reshape(RGB_SHAPE)
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
                        if not streaming:
                            logger.info("Kinect RGB/registered-depth stream ready")
                            streaming = True
                        self.frame_pair_ready.emit(rgb, depth, metadata)
                        self.frame_ready.emit(rgb, depth)
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
                if not self._stop_event.is_set():
                    message = str(exc) or "Kinect camera process disconnected"
                    logger.warning("Camera acquisition failed: %s", message)
                    self.error_occurred.emit(message)
            finally:
                if started:
                    self._stop_capture(process, stop_event)
                parent.close()
                child.close()
            self._stop_event.wait(self._retry_delay)
