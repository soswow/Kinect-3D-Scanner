"""Small, complete acceleration journals, independent of optional camera archives."""

import json
import math
from pathlib import Path

from .inertial import calibration_profile

MAX_JOURNAL_BYTES = 32 * 1024**2


def acceleration_journal(snapshot):
    root = Path(snapshot["root"]).resolve()
    segments, total = [], 0
    for segment in snapshot["segments"]:
        directory = (root / segment["generation"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Invalid sensor segment path")
        length = segment["status"].get("index_bytes", {}).get("accelerometer", 0)
        if type(length) is not int or length < 0 or total + length > MAX_JOURNAL_BYTES:
            raise ValueError("Acceleration journal exceeds 32 MiB")
        total += length
        data = b""
        if length:
            with (directory / "accelerometer.jsonl").open("rb") as handle:
                data = handle.read(length)
        if len(data) != length or data and not data.endswith(b"\n"):
            raise ValueError("Incomplete acceleration journal checkpoint")
        configuration = json.loads((directory / "configuration.json").read_text())
        status = segment["status"]
        segments.append({"recording_segment": segment["generation"],
            "capture_generation": configuration["capture_generation"],
            "calibration": configuration["settings"].get("accelerometer_calibration"),
            "status": {"closed": status.get("closed", False), "error": status.get("error") or snapshot.get("control_error"),
                       "dropped_reads": status.get("dropped", {}).get("accelerometer", 0),
                       "complete": bool(status.get("closed") and not status.get("error")
                                        and not status.get("dropped", {}).get("accelerometer", 0) and not snapshot.get("control_error"))},
            "samples": [json.loads(line) for line in data.splitlines()]})
    return {"version": 1, "segments": segments}


def validate_journal(payload, default_calibration=None):
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode()) > MAX_JOURNAL_BYTES:
        raise ValueError("Acceleration journal exceeds 32 MiB")
    result = json.loads(encoded)
    if not isinstance(result, dict) or result.get("version") != 1 or not isinstance(result.get("segments"), list) or len(result["segments"]) > 1000:
        raise ValueError("Invalid acceleration journal")
    identities = set()
    reads = {}
    for segment in result["segments"]:
        if not isinstance(segment, dict) or not all(isinstance(segment.get(k), str) and 0 < len(segment[k]) <= 128 for k in ("recording_segment", "capture_generation")):
            raise ValueError("Invalid acceleration recording identity")
        if segment["recording_segment"] in identities:
            raise ValueError("Duplicate acceleration recording segment")
        identities.add(segment["recording_segment"])
        segment["calibration"] = calibration_profile(segment.get("calibration") or default_calibration)
        if not isinstance(segment.get("samples"), list) or not isinstance(segment.get("status"), dict):
            raise ValueError("Invalid acceleration recording contents")
        status = segment["status"]
        if (type(status.get("closed")) is not bool or type(status.get("complete")) is not bool
                or type(status.get("dropped_reads")) is not int or status["dropped_reads"] < 0
                or status.get("error") is not None and not isinstance(status["error"], str)):
            raise ValueError("Invalid acceleration recording status")
        if status["complete"] and (not status["closed"] or status["dropped_reads"] or status.get("error")):
            raise ValueError("Acceleration recording incorrectly marked complete")
        previous = -1
        previous_time = -math.inf
        for sample in segment["samples"]:
            if not isinstance(sample, dict) or sample.get("capture_generation") != segment["capture_generation"]:
                raise ValueError("Acceleration capture generation changed")
            sequence = sample.get("sequence")
            if type(sequence) is not int or sequence <= previous or type(sample.get("valid")) is not bool:
                raise ValueError("Invalid acceleration read order")
            previous = sequence
            times = [sample.get(k) for k in ("host_monotonic_s", "read_start_s", "read_end_s")]
            if not all(type(t) in (int, float) and math.isfinite(t) for t in times) or not times[1] <= times[0] <= times[2]:
                raise ValueError("Invalid acceleration read interval")
            if times[0] < previous_time:
                raise ValueError("Acceleration read times are out of order")
            previous_time = times[0]
            if sample["valid"]:
                vector = sample.get("acceleration_m_s2")
                if not isinstance(vector, list) or len(vector) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in vector):
                    raise ValueError("Invalid acceleration vector")
            identity = (segment["capture_generation"], sequence)
            if identity in reads and reads[identity] != sample:
                raise ValueError("Acceleration read changed across recording segments")
            reads[identity] = sample
    return {"version": 1, "segments": result["segments"]}


def merge_journals(previous, incoming):
    segments = {s["recording_segment"]: s for s in previous["segments"]}
    for segment in incoming["segments"]:
        old = segments.get(segment["recording_segment"])
        if old:
            count = min(len(old["samples"]), len(segment["samples"]))
            if (old["capture_generation"] != segment["capture_generation"]
                    or old["samples"][:count] != segment["samples"][:count]
                    or old["calibration"] != segment["calibration"]):
                raise ValueError("Recorded acceleration prefix changed")
            if len(segment["samples"]) < len(old["samples"]):
                continue
        segments[segment["recording_segment"]] = segment
    return validate_journal({"version": 1, "segments": list(segments.values())})
