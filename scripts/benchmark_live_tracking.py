"""Interleaved camera-side tracking comparison against an explicit Git revision.

No saved/reference pose initializes tracking. The textured plane has a known
metric translation for scoring; optional recordings compare accepted poses and
support but do not establish absolute accuracy or acquisition throughput.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker


def plane_sequence(count, holes=0):
    rng = np.random.default_rng(8)
    rgb = cv2.resize(rng.integers(0, 256, (120, 160, 3), np.uint8), (640, 480))
    depth = np.full((480, 640), 2000, np.uint16)
    depth[rng.random(depth.shape) < holes] = 0
    frames = []
    for i in range(count):
        transform = np.array([[1, 0, -i * 2], [0, 1, 0]], float)
        frame = cv2.warpAffine(rgb, transform, (640, 480))
        measured = cv2.warpAffine(depth, transform, (640, 480), flags=cv2.INTER_NEAREST)
        truth = np.eye(4)
        truth[0, 3] = i * 2 * 2 / 525
        frames.append((frame, measured, {"timestamp_s": i * 0.1}, truth))
    return ScanSettings(filter_depth=False), frames


def recording_sequence(path, count, stationary=False):
    manifest = json.loads((path / "manifest.json").read_text())
    settings = ScanSettings.from_dict(manifest["settings"])
    frames = []
    entries = manifest["frames"][:1 if stationary else count]
    for entry in entries:
        rgb = cv2.imread(str(path / entry["rgb"]))
        depth = cv2.imread(str(path / entry["depth"]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise ValueError(f"Missing recording images for {entry['rgb']}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        metadata = {**entry.get("metadata", {}), "timestamp_s": entry["timestamp_s"]}
        frames.append((rgb, depth, metadata, None))
    if stationary:
        rgb, depth, metadata, _ = frames[0]
        frames = [(rgb, depth, {**metadata, "timestamp_s": i * 0.1,
                               "benchmark_source_rgb_depth_delta_ms": metadata.get(
                                   "rgb_depth_delta_ms"),
                               "rgb_depth_delta_ms": 0}, np.eye(4))
                  for i in range(count)]
    return settings, frames


def run_tracker(tracker_type, settings, frames, seed):
    cv2.setRNGSeed(seed)
    tracker = tracker_type(settings)
    reports, times = [], []
    for rgb, depth, metadata, _ in frames:
        started = time.perf_counter()
        report = tracker.update(rgb, depth, metadata)
        elapsed = (time.perf_counter() - started) * 1000
        if reports and "elapsed_ms" in report:
            times.append(elapsed)
        reports.append(report)
    return reports, times


def compare(before, after, frames):
    support_keys = ("valid", "steps", "reference_timestamp_s", "inliers", "matches",
                    "support_fraction", "distributed", "reason")
    disagreements = [i for i, (a, b) in enumerate(zip(before, after))
                     if any(a.get(k) != b.get(k) for k in support_keys)]
    pose_error = max(float(np.max(np.abs(np.asarray(a["camera_to_local"]) -
                                          np.asarray(b["camera_to_local"]))))
                     for a, b in zip(before, after))
    truth_errors = []
    truth_angles = []
    # Only the synthetic/duplicated-frame sequences provide evaluation truth.
    for report, (_, _, _, truth) in zip(after, frames):
        if truth is None or not report["valid"]:
            continue
        relative = np.linalg.inv(truth) @ np.asarray(report["camera_to_local"])
        truth_errors.append(float(np.linalg.norm(relative[:3, 3])))
        truth_angles.append(float(np.degrees(np.arccos(np.clip(
            (np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))))
    return {
        "support_disagreement_indices": disagreements,
        "max_pose_matrix_difference": pose_error,
        "accepted_before": sum(r["valid"] for r in before),
        "accepted_after": sum(r["valid"] for r in after),
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(truth_errors))))
        if truth_errors else None,
        "max_rotation_error_deg": max(truth_angles) if truth_angles else None,
    }


def distribution(values):
    return {"min": float(np.min(values)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 2 or args.repeats < 1 or args.threads < 1:
        parser.error("Require frames >=2 and positive repeats/threads")
    cv2.setNumThreads(args.threads)
    baseline_revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline_revision], cwd=ROOT,
        text=True).strip()
    source = subprocess.check_output(
        ["git", "show", f"{baseline_revision}:shared/visual_tracking.py"], cwd=ROOT,
        text=True)
    module = types.ModuleType("shared._benchmark_baseline_tracking")
    module.__package__ = "shared"
    sys.modules[module.__name__] = module
    exec(compile(source, f"{baseline_revision}:shared/visual_tracking.py", "exec"),
         module.__dict__)
    variants = {"before": module.VisualTracker, "after": VisualTracker}
    workloads = {"textured_plane": plane_sequence(args.frames),
                 "textured_plane_20_percent_depth_holes": plane_sequence(args.frames, 0.2)}
    if args.recording:
        workloads["saved_captures"] = recording_sequence(args.recording, args.frames)
        workloads["stationary_native_compute"] = recording_sequence(
            args.recording, args.frames, stationary=True)
    report = {
        "schema_version": 1,
        "baseline_revision": baseline_revision,
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "current_source_sha256": hashlib.sha256(
            (ROOT / "shared/visual_tracking.py").read_bytes()).hexdigest(),
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
        "opencv_threads": cv2.getNumThreads(), "repeats": args.repeats,
        "workloads": {},
        "notes": ["Interleaved full runs after one warmup per variant; startup frame and timing-guard rejections excluded from timing",
                  "No camera acquisition, upload, server tracking, fusion or preview work included",
                  "Saved captures are sparse and have no independent pose truth",
                  "Stationary native compute duplicates one recorded image with artificial synchronized 10 fps timestamps",
                  "Synthetic/duplicated-frame evaluation truth never initializes tracking"],
    }
    for name, (settings, frames) in workloads.items():
        for tracker_type in variants.values():
            run_tracker(tracker_type, settings, frames, 0)
        timings = {name: [] for name in variants}
        comparisons = []
        digest = hashlib.sha256(json.dumps(settings.to_dict(), sort_keys=True).encode())
        for rgb, depth, metadata, _ in frames:
            digest.update(rgb.tobytes())
            digest.update(depth.tobytes())
            digest.update(json.dumps(metadata, sort_keys=True).encode())
        for repeat in range(args.repeats):
            results = {}
            order = ("before", "after") if repeat % 2 else ("after", "before")
            for variant in order:
                results[variant], elapsed = run_tracker(
                    variants[variant], settings, frames, repeat)
                timings[variant].extend(elapsed)
            comparisons.append(compare(results["before"], results["after"], frames))
        medians = {variant: float(np.median(values)) for variant, values in timings.items()}
        report["workloads"][name] = {
            "input_sha256": digest.hexdigest(), "frames": len(frames),
            "timing_ms": {variant: distribution(values) for variant, values in timings.items()},
            "median_speedup": medians["before"] / medians["after"],
            "comparisons": comparisons,
        }
        print(name, json.dumps(report["workloads"][name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
