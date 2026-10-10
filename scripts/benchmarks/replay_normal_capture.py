"""Replay a camera archive as an ordinary client, then optionally build its payloads.

Intermediate images are available only to the virtual client. The resulting ZIP
contains selected captures, compact motion metadata and the Finish acceleration
journal. The reconstruction engine opens only that ZIP, without saved poses.
"""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import open3d as o3d
from PIL import Image

from kinect_scanner.capture_selection import CaptureSelector, MotionCapturePolicy
from shared.capture import DeviceClockMapper
from shared.inertial import GravityEstimator
from shared.motion_history import MotionHistory, capture_interval
from shared.motion_journal import validate_journal
from shared.protocol import pack_frame, unpack_frame_with_metadata
from shared.sensor_replay import sensor_pairs
from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker


def replay(source, destination, interval):
    started = time.monotonic()
    if source.resolve() == destination.resolve():
        raise ValueError("Output must differ from the source archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows, journals = [], []
    counts = dict(camera_pairs=0, valid_visual_pairs=0, overlap_captures=0,
                  wire_bytes=0, metadata_bytes=0)
    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(destination, "w") as output:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        first = min(f["metadata"]["captured_monotonic_s"] for f in manifest["frames"])
        last = max(f["metadata"]["captured_monotonic_s"] for f in manifest["frames"])
        sensor_archive = manifest["sensor_archive"]
        if not any(s["status"]["counts"].get("rgb") and s["status"]["counts"].get("depth") for s in sensor_archive["segments"]):
            raise ValueError("Replay needs the optional full camera archive")
        for segment in sensor_archive["segments"]:
            base = sensor_archive["root"].rstrip("/") + "/" + segment["generation"] + "/"
            configuration = json.loads(archive.read(base + "configuration.json"))
            streams = {s: [json.loads(line) for line in archive.read(base + s + ".jsonl").splitlines() if line]
                       for s in ("rgb", "depth", "accelerometer")}
            status = segment["status"]
            journals.append({"recording_segment": segment["generation"], "capture_generation": configuration["capture_generation"],
                "calibration": configuration["settings"].get("accelerometer_calibration"),
                "samples": streams["accelerometer"], "status": {"closed": status.get("closed", False),
                    "error": status.get("error"), "dropped_reads": status.get("dropped", {}).get("accelerometer", 0),
                    "complete": bool(status.get("closed") and not status.get("error") and not status.get("dropped", {}).get("accelerometer", 0))}})
            clock, timing = DeviceClockMapper(), {}
            for stream, row in sorted([(s,r) for s in ("rgb", "depth") for r in streams[s]], key=lambda v:v[1]["host_receipt_monotonic_s"]):
                timing[stream, row["sequence"]] = clock.observe(row["device_timestamp_ticks"], row["host_receipt_monotonic_s"], stream)
            tracker, selector, policy, history = VisualTracker(settings), CaptureSelector(), MotionCapturePolicy(), MotionHistory()
            estimator = GravityEstimator(configuration["settings"].get("accelerometer_calibration"))
            previous, read_index = None, 0
            for color, depth in sensor_pairs(streams["rgb"], streams["depth"]):
                now = max(color["host_receipt_monotonic_s"], depth["host_receipt_monotonic_s"])
                if now > last + .02:
                    break
                while read_index < len(streams["accelerometer"]) and streams["accelerometer"][read_index]["host_monotonic_s"] <= now:
                    read = streams["accelerometer"][read_index]
                    history.add_acceleration({**read, "gravity": estimator.update(read)})
                    read_index += 1
                rgb, raw = (np.load(io.BytesIO(archive.read(base + r["image"])), allow_pickle=False) for r in (color, depth))
                timing_row = timing["depth", depth["sequence"]]
                metadata = {"captured_monotonic_s": now, "timestamp_s": depth["timestamp_s"],
                    "depth_device_timestamp_unwrapped_s": timing_row["device_timestamp_unwrapped_s"],
                    "depth_host_monotonic_s": timing_row["estimated_host_monotonic_s"],
                    "host_mapping_uncertainty_s": timing_row["host_mapping_uncertainty_s"],
                    "capture_generation": depth["capture_generation"], "depth_encoding": settings.depth_encoding,
                    "sensor_frame_sequences": {"rgb": color["sequence"], "depth": depth["sequence"]},
                    "rgb_depth_delta_ms": 1000 * (color["device_timestamp_unwrapped_s"] - depth["device_timestamp_unwrapped_s"])}
                metadata["visual_tracking"] = tracker.update(rgb, raw, metadata)
                history.add_frame(metadata)
                metadata["motion_history"] = history.snapshot(now)
                counts["camera_pairs"] += 1
                counts["valid_visual_pairs"] += bool(metadata["visual_tracking"]["valid"])
                if now < first:
                    continue
                selector.offer(rgb, raw, metadata, now)
                bridge = policy.needed(metadata)
                if previous is not None and now - previous < interval and not bridge:
                    continue
                selected = (rgb, raw, metadata) if bridge else selector.choose(now, previous)
                if selected is None:
                    continue
                rgb, raw, metadata = selected
                metadata = {**metadata, "motion_history": capture_interval(metadata["motion_history"], previous),
                            "frame_id": f"normal-replay:{len(rows)}"}
                packet = pack_frame(rgb, raw, metadata, spatial_prediction=True)
                rgb, raw, metadata = unpack_frame_with_metadata(packet)
                counts["wire_bytes"] += len(packet)
                counts["metadata_bytes"] += len(json.dumps(metadata).encode())
                paths = {s: f"{s}/{len(rows):06d}.png" for s in ("rgb", "depth")}
                for key, array in (("rgb", rgb), ("depth", raw)):
                    buffer = io.BytesIO()
                    Image.fromarray(array).save(buffer, format="PNG")
                    output.writestr(paths[key], buffer.getvalue())
                rows.append({**paths, "metadata": metadata, "timestamp_s": metadata["timestamp_s"]})
                previous = metadata["captured_monotonic_s"]
                policy.captured(metadata)
                selector.clear()
                counts["overlap_captures"] += int(bridge)
                if len(rows) % 25 == 0:
                    print(f"Client replay: {len(rows)} captures from {counts['camera_pairs']} camera pairs", flush=True)
        journal = validate_journal({"version": 1, "segments": journals})
        output.writestr("motion.json", json.dumps(journal, separators=(",", ":")))
        output.writestr("manifest.json", json.dumps({"version": 3, "kind": "kinect-scanner-project",
            "settings": {**settings.to_dict(), "record_full_camera_streams": False}, "frames": rows, "motion_journal": "motion.json"}))
    return {**counts, "captures": len(rows), "complete_capture_intervals": sum(r["metadata"]["motion_history"]["complete_interval"] for r in rows),
            "journal_reads": sum(len(s["samples"]) for s in journals), "client_replay_seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not np.isfinite(args.interval) or args.interval <= 0:
        parser.error("Interval must be positive and finite")
    o3d.utility.set_max_threads(int(os.environ["OMP_NUM_THREADS"]))
    summary = replay(args.source, args.output, args.interval)
    if args.build:
        from scanner_server.engine import ScanEngine
        from scanner_server.session import load_session
        engine = ScanEngine(device=args.device)
        try:
            load_session(engine, args.output)
            started = time.monotonic()
            success, result = engine.build_mesh(progress_cb=lambda done,total,data:print(data.get("message"), flush=True))
            report = engine.reconstruction_report()
            args.output.with_suffix(".reconstruction.json").write_text(json.dumps(report, indent=2))
            if success:
                engine.export_ply(str(args.output.with_suffix(".ply")))
            summary.update(build_success=success, build_seconds=time.monotonic()-started, result=result)
        finally:
            engine.shutdown()
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
