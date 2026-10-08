"""Independent lossless sensor streams and atomic augmentation of saved sessions.

Each connection has its own clock epoch. Indices contain raw observations, not
only the images selected for fusion. A bounded writer reports every overflow.
"""

import json
import queue
import shutil
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

STREAMS = ("rgb", "depth", "accelerometer", "events")


class SensorJournal:
    def __init__(self, root, generation, settings, capacity=32, capture_generation=None):
        self.path = Path(root) / generation
        self.path.mkdir(parents=True, exist_ok=False)
        self.settings = settings
        self.record_images = settings.get("record_full_camera_streams", False)
        self.generation = capture_generation or generation
        self.queue = queue.Queue(maxsize=capacity)
        self.notifications = queue.SimpleQueue()
        self.closing = threading.Event()
        self.lock = threading.Lock()
        self.counts = dict.fromkeys(STREAMS, 0)
        self.dropped = dict.fromkeys(STREAMS, 0)
        self.error = None
        self.closed = False
        self.last_checkpoint = {**self.status(), "index_bytes": {}}
        (self.path / "configuration.json").write_text(json.dumps({
            "version": 1, "capture_generation": self.generation, "recording_segment": generation, "settings": settings,
            "started_timestamp_s": time.time(), "started_monotonic_s": time.monotonic(),
            "clock": {"image_device_hz": 60_000_000, "image_reference": "packet_end",
                      "accelerometer_reference": "host_read_interval", "host_clock": "monotonic"},
            "streams": {"rgb": {"units": "uint8 RGB", "encoding": "NPY, no pickle", "recorded": self.record_images},
                        "depth": {"units": "raw_11bit_disparity", "encoding": "uint16 NPY, no pickle", "recorded": self.record_images},
                        "accelerometer": {"units": "m/s^2", "raw_units": "driver_counts", "encoding": "JSONL"},
                        "events": {"encoding": "JSONL", "scope": "orientation and recording controls"}},
        }, indent=2, allow_nan=False) + "\n")
        self.thread = threading.Thread(target=self._write, name="Sensor recording", daemon=True)
        self.thread.start()

    def submit(self, stream, metadata, array=None):
        if stream not in STREAMS:
            raise ValueError("Unknown sensor stream")
        if stream in ("rgb", "depth") and not self.record_images:
            return True  # Intentionally omitted; never copy or queue image arrays.
        with self.lock:
            failed = self.error is not None or self.closing.is_set()
        if failed:
            with self.lock:
                self.dropped[stream] += 1
            return False
        try:
            # Driver callback arrays are borrowed and can be overwritten immediately.
            self.queue.put_nowait((stream, dict(metadata), None if array is None else array.copy()))
            return True
        except queue.Full:
            with self.lock:
                self.dropped[stream] += 1
            return False

    def request_flush(self, token):
        if not self.thread.is_alive():
            failure = {**self.last_checkpoint, "complete": False, "error": self.error or "Recording writer stopped"}
            self.notifications.put({"request_id": token, "path": str(self.path), "status": failure})
            return True
        try:
            self.queue.put_nowait(("flush", token, None))
            return True
        except queue.Full:
            return False

    def status(self):
        with self.lock:
            return {"version": 1, "capture_generation": self.generation,
                    "recording_segment": self.path.name, "root": str(self.path.parent),
                    "record_full_camera_streams": self.record_images,
                    "counts": self.counts.copy(), "dropped": self.dropped.copy(),
                    "error": self.error, "closed": self.closed,
                    "complete": self.error is None and not any(self.dropped.values())}

    def _checkpoint(self, handles):
        for handle in handles.values():
            handle.flush()
        status = self.status()
        status.update(checkpoint_timestamp_s=time.time(), checkpoint_monotonic_s=time.monotonic())
        status["index_bytes"] = {stream: (self.path / (stream + ".jsonl")).stat().st_size for stream in STREAMS}
        temporary = self.path / "status.json.tmp"
        temporary.write_text(json.dumps(status, allow_nan=False) + "\n")
        temporary.replace(self.path / "status.json")
        self.last_checkpoint = status
        return status

    def _write(self):
        handles = {}
        try:
            handles = {stream: (self.path / (stream + ".jsonl")).open("w") for stream in STREAMS}
            self._checkpoint(handles)
            next_checkpoint = time.monotonic() + 1
            while not self.closing.is_set() or not self.queue.empty():
                if time.monotonic() >= next_checkpoint:
                    self._checkpoint(handles)
                    next_checkpoint = time.monotonic() + 1
                try:
                    stream, metadata, array = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if stream == "flush":
                        self.notifications.put({"request_id": metadata, "path": str(self.path),
                                                "status": self._checkpoint(handles)})
                        continue
                    if self.error:
                        with self.lock:
                            self.dropped[stream] += 1
                        continue
                    if array is not None:
                        relative = f"{stream}/{metadata['sequence']:09d}.npy"
                        image_path = self.path / relative
                        image_path.parent.mkdir(exist_ok=True)
                        # Avoid PNG encoding in the acquisition path: independent
                        # RGB + depth total ~58 MB/s in high-res mode. SSD writes
                        # of native arrays are inexpensive and exactly lossless.
                        np.save(image_path, array, allow_pickle=False)
                        metadata.update(image=relative, shape=list(array.shape), dtype=str(array.dtype))
                    handles[stream].write(json.dumps(metadata, allow_nan=False, separators=(",", ":")) + "\n")
                    with self.lock:
                        self.counts[stream] += 1
                except (OSError, ValueError, cv2.error) as exc:
                    with self.lock:
                        self.error = str(exc)
                        if stream in STREAMS:
                            self.dropped[stream] += 1
                finally:
                    self.queue.task_done()
            with self.lock:
                self.closed = True
            self._checkpoint(handles)
        except Exception as exc:  # Disk failures must be visible without killing USB capture.
            with self.lock:
                self.error = str(exc)
        finally:
            for handle in handles.values():
                handle.close()

    def close(self, timeout=0.2):
        self.closing.set()
        self.thread.join(timeout)


def journal_snapshot(root, active=None):
    """Inventory only checkpointed index prefixes; subsequent writes stay outside."""
    segments = []
    root = Path(root)
    for path in sorted(root.iterdir()) if root.exists() else []:
        if not path.is_dir() or not (path / "configuration.json").exists():
            continue
        try:
            status = json.loads((path / "status.json").read_text())
        except (OSError, ValueError):
            status = {"complete": False, "error": "No sensor checkpoint", "index_bytes": {}}
        if active and active.get("path") == str(path):
            status = active["status"]
        elif not status.get("closed"):
            status = {**status, "complete": False, "error": status.get("error") or "Capture stopped before recording checkpoint"}
        configuration = json.loads((path / "configuration.json").read_text())
        segments.append({"generation": path.name, "status": status,
                         "started_timestamp_s": configuration.get("started_timestamp_s", 0)})
    segments.sort(key=lambda s: s["started_timestamp_s"])
    return {"version": 1, "root": str(root), "segments": segments,
            "complete": bool(segments) and all(s["status"].get("complete") for s in segments)}


def augment_session_archive(path, snapshot, portrait=True):
    """Add full streams and portrait derivatives without rewriting native geometry."""
    path = Path(path)
    temporary = path.with_name(path.name + ".sensors.tmp")
    root = Path(snapshot["root"])
    try:
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as target:
            manifest = json.loads(source.read("manifest.json"))
            for item in source.infolist():
                if item.filename != "manifest.json":
                    with source.open(item) as data, target.open(item, "w", force_zip64=True) as output:
                        shutil.copyfileobj(data, output, length=1024 * 1024)
            for segment in snapshot["segments"]:
                generation = segment["generation"]
                directory = (root / generation).resolve()
                if not directory.is_relative_to(root.resolve()):
                    raise ValueError("Invalid sensor segment path")
                target.write(directory / "configuration.json", f"sensors/{generation}/configuration.json")
                for stream, length in segment["status"].get("index_bytes", {}).items():
                    if stream not in STREAMS or type(length) is not int or length < 0:
                        raise ValueError("Invalid sensor index")
                    with (directory / (stream + ".jsonl")).open("rb") as handle:
                        data = handle.read(length)
                    if len(data) != length or (data and not data.endswith(b"\n")):
                        raise ValueError("Incomplete sensor index checkpoint")
                    target.writestr(f"sensors/{generation}/{stream}.jsonl", data)
                    for line in data.splitlines():
                        row = json.loads(line)
                        if "image" in row:
                            image_path = (directory / row["image"]).resolve()
                            if not image_path.is_relative_to(directory):
                                raise ValueError("Invalid recorded image path")
                            target.write(image_path, f"sensors/{generation}/{row['image']}")
                target.writestr(f"sensors/{generation}/status.json", json.dumps(segment["status"], allow_nan=False))
            from shared.inertial import rotate_display
            if portrait:
                for frame in manifest["frames"]:
                    rotation = frame.get("metadata", {}).get("orientation", {}).get("rotation_cw_degrees", 0)
                    if rotation not in (90, 180, 270):
                        continue
                    for stream in ("rgb", "depth"):
                        if "display_" + stream in frame:
                            continue
                        raw = cv2.imdecode(np.frombuffer(source.read(frame[stream]), np.uint8), cv2.IMREAD_UNCHANGED)
                        success, encoded = cv2.imencode(".png", rotate_display(raw, rotation), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        if not success:
                            raise OSError("Could not encode portrait image")
                        relative = f"display/{frame[stream]}"
                        target.writestr(relative, encoded.tobytes())
                        frame["display_" + stream] = relative
            previous = manifest.get("sensor_archive", {})
            segments = {s["generation"]: s for s in previous.get("segments", [])}
            segments.update({s["generation"]: s for s in snapshot["segments"]})
            manifest.update(version=max(2, manifest.get("version", 1)), sensor_archive={
                "version": 1, "root": "sensors", "segments": list(segments.values()),
                "complete": snapshot["complete"] and previous.get("complete", True), "scope": "All accelerometer read attempts and orientation events; full RGB/depth streams only in explicitly enabled recording segments",
                "control_error": snapshot.get("control_error"),
            })
            target.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
