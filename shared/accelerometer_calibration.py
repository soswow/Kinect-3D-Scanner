"""Stationary calibration capture without Qt, images, or unbounded USB waits."""

import math
import multiprocessing
import time

import numpy as np

from .inertial import AccelerometerPoller, G


# Directions refer to the unrotated depth image: x right, y down, z forward.
POSES = (
    ("upright", "Upright, lenses horizontal, camera's top facing the ceiling.", (0, -1, 0)),
    ("upside_down", "Upside down, lenses horizontal, camera's top facing the floor.", (0, 1, 0)),
    ("right_up", "Lenses horizontal; camera's RIGHT side facing the ceiling (viewed from behind the camera).", (1, 0, 0)),
    ("left_up", "Lenses horizontal; camera's LEFT side facing the ceiling (viewed from behind the camera).", (-1, 0, 0)),
    ("lenses_up", "Lenses pointing straight at the ceiling.", (0, 0, 1)),
    ("lenses_down", "Lenses pointing straight at the floor.", (0, 0, -1)),
)
# Fresh captures, collected after the fitting set; never reuse fitting samples.
VALIDATION_POSES = (POSES[0], POSES[2], POSES[4])


def capture_process(connection, stop_event, device_index):
    """Only this spawned child enters native freenect calls."""
    context = device = None
    try:
        import freenect

        context = freenect.init()
        if context is None:
            raise RuntimeError("Cannot initialise libfreenect")
        if freenect.num_devices(context) <= device_index:
            raise RuntimeError("No Kinect detected at this index. Check USB and external power.")
        device = freenect.open_device(context, device_index)
        if device is None:
            raise RuntimeError("Cannot open Kinect. Close the scanner and other camera apps.")
        poller = AccelerometerPoller(freenect, device)
        if not poller.enabled:
            raise RuntimeError("The installed freenect binding does not expose accelerometer reads.")
        connection.send(("ready", {"device_index": device_index, "capture_generation": poller.generation}))
        while not stop_event.is_set():
            if not connection.poll(0.1):
                continue
            seconds, settle = connection.recv()
            start = time.monotonic()
            while not stop_event.is_set() and time.monotonic() - start < seconds + settle:
                sample = poller.poll()
                if sample is not None:
                    phase = "settle" if sample["read_start_s"] - start < settle else "measure"
                    connection.send(("sample", {**sample, "phase": phase}))
                if not poller.enabled:
                    raise RuntimeError(poller.reason)
                stop_event.wait(0.01)
            connection.send(("done", None))
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        try:
            message = str(exc)
            if isinstance(exc, ImportError):
                message += ". Run with the scanner's Python environment (numpy and freenect required)."
            connection.send(("error", message))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if device is not None:
            freenect.close_device(device)
        if context is not None:
            freenect.shutdown(context)
        connection.close()


class CalibrationCapture:
    """Request stationary windows; kill a stalled native call on timeout/quit."""

    def __init__(self, device_index=0, *, target=capture_process, timeout=8.0):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.stop_event = context.Event()
        self.process = context.Process(target=target, args=(child, self.stop_event, device_index),
                                       daemon=True, name="Kinect calibration")
        self.timeout = timeout
        self._child = child
        self.metadata = None

    def __enter__(self):
        try:
            self.process.start()
            self._child.close()
            kind, payload = self._receive(self.timeout)
            if kind != "ready":
                raise RuntimeError("Unexpected accelerometer startup response")
            self.metadata = payload
            return self
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise RuntimeError("Kinect accelerometer timed out. Check USB/power and close other camera apps.")
        try:
            kind, payload = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("Kinect calibration process disconnected") from exc
        if kind == "error":
            raise RuntimeError(payload)
        return kind, payload

    def collect(self, seconds=3.0, settle=1.0, on_sample=None):
        self.connection.send((seconds, settle))
        samples = []
        deadline = time.monotonic() + seconds + settle + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Kinect stationary capture exceeded its time limit")
            kind, payload = self._receive(min(self.timeout, remaining))
            if kind == "done":
                return samples
            if kind != "sample":
                raise RuntimeError("Unexpected accelerometer capture response")
            samples.append(payload)
            if on_sample is not None:
                on_sample(payload)

    def close(self):
        self.stop_event.set()
        if self.process.pid is not None:
            self.process.join(timeout=0.3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=0.5)
            if not self.process.is_alive():
                self.process.close()
        self.connection.close()
        self._child.close()

    def __exit__(self, *_):
        self.close()


def stationary_observation(samples, up_camera, seconds):
    """Average original driver m/s² readings, never already-calibrated gravity."""
    measured = [s for s in samples if s.get("phase") == "measure"]
    good = []
    for sample in measured:
        if not sample.get("valid"):
            continue
        vector = np.asarray(sample.get("acceleration_m_s2"), dtype=float)
        latency = sample["read_end_s"] - sample["read_start_s"]
        if (vector.shape == (3,) and np.isfinite(vector).all()
                and 0 <= latency <= 0.03 and math.isfinite(sample["host_monotonic_s"])):
            good.append(sample)
    minimum = max(20, math.ceil(seconds * 8))
    if len(good) < minimum or len(good) < 0.9 * len(measured):
        raise ValueError(f"Too few reliable readings ({len(good)}); need at least {minimum} and 90% reliable reads.")
    stamps = np.asarray([s["host_monotonic_s"] for s in good])
    if np.any(np.diff(stamps) <= 0) or stamps[-1] - stamps[0] < 0.8 * seconds:
        raise ValueError("Readings do not cover the recording interval.")
    vectors = np.asarray([s["acceleration_m_s2"] for s in good])
    mean = vectors.mean(axis=0)
    deviations = np.linalg.norm(vectors - mean, axis=1)
    rms = float(np.sqrt(np.mean(deviations ** 2)))
    maximum = float(deviations.max())
    if rms > 0.12 or maximum > 0.35:
        raise ValueError(f"Camera moved or vibrated (scatter {rms:.3f} m/s², peak {maximum:.3f}). Support it and retry.")
    if not 0.5 * G < np.linalg.norm(mean) < 1.5 * G:
        raise ValueError("Stationary acceleration is implausible. Check the sensor and retry.")
    generations = {s.get("capture_generation") for s in good}
    if len(generations) != 1 or None in generations:
        raise ValueError("Readings cross a sensor reconnect; retry this pose.")
    return {"acceleration_m_s2": mean.tolist(), "up_camera": list(up_camera),
            "capture_summary": {"sample_count": len(good), "rejected_reads": len(measured) - len(good),
                                "duration_s": float(stamps[-1] - stamps[0]),
                                "scatter_rms_m_s2": rms, "peak_deviation_m_s2": maximum}}
