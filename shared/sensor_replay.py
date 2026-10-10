"""Rebuild RGB-D pairing and gravity from saved independent sensor streams."""

import json
from pathlib import Path

import cv2
import numpy as np

from .capture import RGB_DEPTH_CAPTURE_LIMIT_MS
from .inertial import GravityEstimator, OrientationTracker, MAX_HOST_MAPPING_UNCERTAINTY_S, orientation_observation


def _rows(directory, stream):
    path = directory / (stream + ".jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def sensor_pairs(rgb, depth):
    """Nearest unused depth for each RGB, using unwrapped native packet clocks."""
    consumed, result = -1, []
    times = np.array([d["device_timestamp_unwrapped_s"] for d in depth])
    for color in rgb:
        stamp = color["device_timestamp_unwrapped_s"]
        position = int(np.searchsorted(times, stamp))
        candidates = [i for i in (position - 1, position, position + 1) if consumed < i < len(times)]
        if not candidates:
            continue
        index = min(candidates, key=lambda i: abs(stamp - times[i]))
        if abs(stamp - times[index]) * 1000 > RGB_DEPTH_CAPTURE_LIMIT_MS:
            continue
        consumed = index
        result.append((color, depth[index]))
    return result


def load_sensor_observations(path, settings):
    """Yield every rebuilt pair; re-estimate gravity from raw read attempts."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    archive = manifest.get("sensor_archive")
    if archive is None:
        raise ValueError("This session has no full sensor streams; replay its selected frames instead")
    if not any(segment.get("status", {}).get("counts", {}).get("rgb", 0)
               and segment.get("status", {}).get("counts", {}).get("depth", 0) for segment in archive["segments"]):
        raise ValueError("This session contains selected images and an accelerometer log; full camera recording was not enabled. Replay its selected frames without --sensor-streams.")
    orientation = OrientationTracker()
    for segment in archive["segments"]:
        directory = (path / archive["root"] / segment["generation"]).resolve()
        if not directory.is_relative_to(path.resolve()):
            raise ValueError("Invalid sensor archive path")
        configuration = json.loads((directory / "configuration.json").read_text())
        estimator = GravityEstimator(configuration["settings"].get("accelerometer_calibration"))
        samples = _rows(directory, "accelerometer")
        samples.sort(key=lambda s: s["host_monotonic_s"])
        for sample in samples:
            sample["gravity"] = estimator.update(sample)
        sample_times = np.array([s["host_monotonic_s"] for s in samples])
        colors = sorted(_rows(directory, "rgb"), key=lambda r: r["device_timestamp_unwrapped_s"])
        depths = sorted(_rows(directory, "depth"), key=lambda r: r["device_timestamp_unwrapped_s"])
        events = sorted(_rows(directory, "events"), key=lambda r: r["host_monotonic_s"])
        event_index = 0
        mode = configuration["settings"].get("orientation_mode", "auto")
        for color, depth in sensor_pairs(colors, depths):
            stamp = depth["estimated_host_monotonic_s"]
            display_stamp = max(color.get("host_receipt_monotonic_s", stamp),
                                depth.get("host_receipt_monotonic_s", stamp))
            while event_index < len(events) and events[event_index]["host_monotonic_s"] <= display_stamp:
                event = events[event_index]
                if event.get("type") == "orientation_mode":
                    mode = event["value"]
                event_index += 1
            acceleration = {"valid": False, "reason": "No accelerometer samples"}
            uncertainty = depth.get("host_mapping_uncertainty_s", 0)
            if samples:
                center = int(np.searchsorted(sample_times, stamp))
                candidates = [i for i in (center - 1, center) if 0 <= i < len(samples)]
                selected = samples[min(candidates, key=lambda i: abs(stamp - sample_times[i]))]
                delta = stamp - selected["host_monotonic_s"]
                acceleration = {"version": 1,
                                "valid": bool(selected.get("valid") and selected["gravity"]["valid"] and abs(delta) <= .15
                                              and 0 <= uncertainty <= MAX_HOST_MAPPING_UNCERTAINTY_S),
                                "reason": "Recomputed from raw accelerometer reads",
                                "capture_generation": selected["capture_generation"],
                                "sequence": selected["sequence"], "sample_delta_ms": delta * 1000,
                                "read_start_s": selected["read_start_s"], "read_end_s": selected["read_end_s"],
                                "acceleration_m_s2": selected.get("acceleration_m_s2"), "gravity": selected["gravity"]}
            # Replay coarse display decisions using only reads already received
            # by the host, while retaining strict exposure association for tracking.
            latest_index = int(np.searchsorted(sample_times, display_stamp, side="right")) - 1
            display_acceleration = orientation_observation(samples[latest_index] if latest_index >= 0 else None,
                                                           display_stamp, depth["capture_generation"])
            metadata = {"timestamp_s": depth["timestamp_s"], "captured_monotonic_s": stamp,
                        "depth_host_monotonic_s": stamp,
                        "rgb_host_monotonic_s": color["estimated_host_monotonic_s"],
                        "host_mapping_uncertainty_s": uncertainty,
                        "capture_generation": depth["capture_generation"],
                        "sensor_recording_segment": segment["generation"],
                        "sensor_frame_sequences": {"rgb": color["sequence"], "depth": depth["sequence"]},
                        "rgb_depth_delta_ms": 1000 * (color["device_timestamp_unwrapped_s"] - depth["device_timestamp_unwrapped_s"]),
                        "rgb_timestamp_ticks": color["device_timestamp_ticks"], "depth_timestamp_ticks": depth["device_timestamp_ticks"],
                        "device_timestamp_hz": 60_000_000, "device_timestamp_reference": "packet_end",
                        "depth_encoding": settings.depth_encoding, "accelerometer": acceleration,
                        "orientation_accelerometer": display_acceleration, "orientation_host_monotonic_s": display_stamp,
                        "orientation": orientation.update(display_acceleration, display_stamp, mode)}
            for field in ("rgb_exposure_mode", "rgb_exposure_us", "rgb_shutter_speed", "rgb_gain", "rgb_mode", "rgb_fps",
                          "exposure_phase", "settling_frames_remaining"):
                if field in color:
                    metadata[field] = color[field]
            arrays = []
            for row in (color, depth):
                image_path = (directory / row["image"]).resolve()
                if not image_path.is_relative_to(directory):
                    raise ValueError("Invalid sensor image path")
                image = np.load(image_path, allow_pickle=False) if image_path.suffix == ".npy" else cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"Cannot read sensor image {image_path}")
                arrays.append(image)
            rgb = arrays[0] if (directory / color["image"]).suffix == ".npy" else cv2.cvtColor(arrays[0], cv2.COLOR_BGR2RGB)
            yield rgb, arrays[1], metadata
