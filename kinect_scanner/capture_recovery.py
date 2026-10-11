"""Offline partial projects made from every retained capture, without server access."""

import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

from shared.project_archive import encode_image, write_member
from shared.settings import ScanSettings


def export_recovery_project(spool, destination):
    """Publish a lossless pending-only v3 project; retain all original buffer files.

    Requires a readable ordered spool and its original calibration. No reconstruction
    poses or mesh are invented. Failure leaves an existing destination untouched.
    """
    if spool.ordering_error:
        raise ValueError(spool.ordering_error)
    paths = spool.pending_paths
    if not paths:
        raise ValueError("No retained captures are available for recovery")
    settings = ScanSettings.from_dict(spool.settings)
    destination = Path(destination).resolve()
    if destination.is_relative_to(spool.path.resolve()):
        raise ValueError("Save the recovery project outside its original capture buffer")
    manifest = {"version": 3, "kind": "kinect-scanner-project", "settings": spool.settings,
                "depth_unit": "raw_11bit_disparity" if settings.sensor_calibration else "millimetres",
                "depth_encoding": settings.depth_encoding, "rgb_format": "png", "frames": [],
                "recovery": {"version": 1, "partial": True, "scope": "pending captures only",
                             "source_session_id": spool.session_id, "source_stream_id": getattr(spool, "stream_id", None),
                             "capture_count": len(paths), "ordering": "durable sequence" if spool.version == 2 else "validated legacy order"}}
    descriptor, temporary = tempfile.mkstemp(prefix=".capture-recovery-", suffix=".zip", dir=destination.parent)
    os.close(descriptor)
    try:
        identities = set()
        with zipfile.ZipFile(temporary, "w") as archive:
            for index, path in enumerate(paths):
                rgb, depth, metadata = spool.load(path)
                camera = settings.rgb_camera
                if rgb.shape != (camera.height, camera.width, 3) or depth.shape != (480, 640):
                    raise ValueError(f"Retained capture {path.name} does not match its calibration")
                if settings.sensor_calibration and (depth > 2047).any():
                    raise ValueError("Retained raw depth codes exceed 11-bit calibration")
                timestamp = metadata.get("timestamp_s", index)
                if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                    raise ValueError("Invalid retained capture timestamp")
                if metadata.get("depth_encoding", settings.depth_encoding) != settings.depth_encoding:
                    raise ValueError("Retained capture depth encoding differs from its calibration")
                identity = metadata.get("frame_id")
                if identity is not None:
                    if type(identity) not in (str, int) or (type(identity), identity) in identities:
                        raise ValueError("Retained captures have invalid or duplicate frame identities")
                    identities.add((type(identity), identity))
                color_path, depth_path = f"rgb/{index:06d}.png", f"depth/{index:06d}.png"
                write_member(archive, color_path, encode_image(rgb))
                write_member(archive, depth_path, encode_image(depth))
                manifest["frames"].append({"rgb": color_path, "depth": depth_path,
                                           "timestamp_s": timestamp, "metadata": metadata})
            write_member(archive, "manifest.json", json.dumps(manifest, allow_nan=False, separators=(",", ":")))
        with open(temporary, "r+b") as completed:
            os.fsync(completed.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination
