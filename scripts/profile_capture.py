"""Profile recorded capture and live feedback without touching a running server.

Includes lossless packing/unpacking, storage, one-frame processing, and feedback
encoding/decoding. Excludes file loading, startup, final building, USB, and actual
network transport. Recording settings and timestamp gates are preserved.
"""

import argparse
import cProfile
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.process_metrics import peak_rss_bytes

import numpy as np

from scanner_server.engine import ScanEngine
from scripts.profile_backends import source_hash
from scripts.replay_scan import capture_metadata, load_dataset, score
from shared.live import decode_live_geometry, encode_live_message
from shared.protocol import pack_frame, unpack_frame_with_metadata


def summarize(values, unit="ms"):
    return {
        f"total_{unit}": float(sum(values)),
        f"p50_{unit}": float(np.percentile(values, 50)) if values else None,
        f"p95_{unit}": float(np.percentile(values, 95)) if values else None,
    }


def compare_reports(before, after):
    """Compare accepted observations and poses, without claiming ground truth."""
    old = {p["index"]: np.asarray(p["pose"]) for p in before["poses"]}
    new = {p["index"]: np.asarray(p["pose"]) for p in after["poses"]}
    translation, rotation = [], []
    for index in old.keys() & new.keys():
        delta = np.linalg.inv(old[index]) @ new[index]
        translation.append(float(np.linalg.norm(delta[:3, 3])))
        rotation.append(
            float(
                np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
            )
        )
    return {
        "same_accepted_indices": old.keys() == new.keys(),
        "compared_poses": len(translation),
        "max_translation_change_m": max(translation, default=None),
        "max_rotation_change_deg": max(rotation, default=None),
        "elapsed_reduction_percent": 100
        * (1 - after["elapsed_s"] / before["elapsed_s"]),
        "note": "Pose agreement with baseline is a regression check, not an accuracy measurement",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--feedback", choices=("packed", "json"), default="packed")
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=(0, 1),
        default=0,
        help="0 models loopback capture; 1 models compressed network capture",
    )
    parser.add_argument(
        "--compare", type=Path, help="Previous profile_capture JSON report"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Also save cProfile data beside the JSON"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-output/capture-profile.json"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Limit must be positive")
    settings, frames = load_dataset("recording", args.path, limit=args.limit)
    if not frames:
        parser.error("Recording has no frames")
    source_before = source_hash()
    engine = ScanEngine(device="cpu")
    engine.reset(settings=settings)
    profile = cProfile.Profile() if args.profile else None
    timings, feedback_sizes, upload_sizes = {}, [], []

    def timed(name, function, *positional, **keywords):
        started = time.monotonic()
        result = function(*positional, **keywords)
        timings.setdefault(name, []).append((time.monotonic() - started) * 1000)
        return result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if profile:
        profile.enable()
    started = time.monotonic()
    for index, frame in enumerate(frames):
        rgb, depth, _, _ = frame
        packet = timed(
            "pack",
            pack_frame,
            rgb,
            depth,
            capture_metadata(frame, index),
            compression_level=args.compression_level,
        )
        decoded = timed("unpack", unpack_frame_with_metadata, packet)
        stored = timed("store", engine.store_frame, *decoded)
        if not stored["success"]:
            raise RuntimeError(
                f"Capture {index + 1} rejected at upload: {stored['message']}"
            )
        timed("process", engine.process_frames, max_frames=1)
        snapshot = timed(
            "snapshot",
            engine.live_snapshot,
            **({"array_geometry": True} if args.feedback == "packed" else {}),
        )
        feedback = timed(
            "feedback_encode",
            encode_live_message,
            snapshot,
            packed_geometry=args.feedback == "packed",
        )
        timed(
            "feedback_decode",
            lambda data=feedback: decode_live_geometry(json.loads(data)),
        )
        upload_sizes.append(len(packet))
        feedback_sizes.append(len(feedback))
        if (index + 1) % 10 == 0 or index + 1 == len(frames):
            print(
                f"{index + 1}/{len(frames)} accepted={engine.frame_count} elapsed={time.monotonic() - started:.2f}s",
                flush=True,
            )
    elapsed = time.monotonic() - started
    if profile:
        profile.disable()
        profile.dump_stats(str(args.output.with_suffix(".prof")))
    report = {
        "method": "sequential isolated CPU recording replay; startup/loading/final build/USB/network excluded",
        "source_sha256": source_before,
        "source_changed_during_profile": source_before != source_hash(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "omp_threads": os.environ["OMP_NUM_THREADS"],
        "initial_blocks": os.environ["KINECT_BLOCK_COUNT"],
        "feedback_format": args.feedback,
        "compression_level": args.compression_level,
        "frames": len(frames),
        "accepted": engine.frame_count,
        "elapsed_s": elapsed,
        "pipeline_frames_per_second": len(frames) / elapsed,
        "pipeline_stages": {key: summarize(values) for key, values in timings.items()},
        "processing_stages": {
            key: summarize(
                [
                    d["timings_ms"][key]
                    for d in engine.diagnostics
                    if key in d["timings_ms"]
                ]
            )
            for key in engine.stage_totals_ms
        },
        "tracking_frame_p50_ms": float(
            np.median([d["elapsed_ms"] for d in engine.diagnostics if d["success"]])
        ),
        "feedback_bytes": summarize(feedback_sizes, "bytes"),
        "upload_bytes": summarize(upload_sizes, "bytes"),
        "peak_rss_bytes": peak_rss_bytes(),
        "backend": engine.backend,
        "poses": [
            {"index": index, "pose": pose.tolist()} for index, pose in engine.poses
        ],
        "tracking_score": score(engine.poses, frames),
        "diagnostics": engine.diagnostics,
    }
    if args.compare:
        report["comparison"] = compare_reports(
            json.loads(args.compare.read_text()), report
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("diagnostics", "poses")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
