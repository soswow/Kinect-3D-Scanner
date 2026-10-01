"""KinectWorker — background thread for continuous frame grabbing."""

import time

import freenect
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from shared.capture import timestamp_delta_ms


class KinectWorker(QThread):
    """Grabs RGB + registered-depth frames via freenect sync interface."""

    frame_ready = pyqtSignal(np.ndarray, np.ndarray)  # (rgb, depth_mm)
    frame_pair_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False

    def stop(self):
        self._running = False

    def _wait_before_retry(self):
        # Back off on missing hardware, while allowing a prompt window close.
        for _ in range(20):
            if not self._running:
                break
            self.msleep(100)

    def run(self):
        self._running = True
        camera_ready = False
        sequence = 0
        dropped = 0
        while self._running:
            try:
                if not camera_ready:
                    ctx = freenect.init()
                    if ctx is None:
                        raise RuntimeError("Cannot initialise libfreenect")
                    try:
                        camera_ready = freenect.num_devices(ctx) > 0
                    finally:
                        freenect.shutdown(ctx)
                    if not camera_ready:
                        self.error_occurred.emit(
                            "No Kinect camera detected. Connect USB and external power."
                        )
                        self._wait_before_retry()
                        continue

                depth_result = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
                if depth_result is None:
                    self.error_occurred.emit("Failed to get depth frame")
                    freenect.sync_stop()
                    camera_ready = False
                    self._wait_before_retry()
                    continue
                depth, depth_stamp = depth_result
                depth = depth.copy()  # libfreenect owns and reuses this buffer

                vid_result = freenect.sync_get_video(format=freenect.VIDEO_RGB)
                if vid_result is None:
                    self.error_occurred.emit("Failed to get video frame")
                    freenect.sync_stop()
                    camera_ready = False
                    self._wait_before_retry()
                    continue
                video, rgb_stamp = vid_result
                video = video.copy()
                delta = timestamp_delta_ms(rgb_stamp, depth_stamp)
                if abs(delta) > 50:
                    dropped += 1
                    if dropped % 30 == 1:
                        self.error_occurred.emit(
                            "Dropped unsynchronised RGB/depth pair; check USB bandwidth"
                        )
                    continue
                sequence += 1
                metadata = {
                    "frame_id": sequence,
                    "captured_monotonic_s": time.monotonic(),
                    "timestamp_s": time.time(),
                    "depth_timestamp_ms": int(depth_stamp),
                    "rgb_timestamp_ms": int(rgb_stamp),
                    "rgb_depth_delta_ms": delta,
                }
                self.frame_pair_ready.emit(video, depth, metadata)
                self.frame_ready.emit(video, depth)
            except Exception as e:
                self.error_occurred.emit(str(e))
                camera_ready = False
                self._wait_before_retry()

        try:
            freenect.sync_stop()
        except Exception:
            pass
