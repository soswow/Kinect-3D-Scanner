"""Compare camera trackers on selected session captures without inventing motion.

The recorded-time replay preserves camera timestamps and RGB/depth skew. The
separate stationary diagnostic repeats each raw pair at 10 Hz with zero skew;
it measures compute, feature retention and zero-motion consistency, not real
moving-camera tracking. Bootstrap references are never counted as motion.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import types
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS
from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker


def distribution(values):
    if not values:
        return None
    return {name: float(value) for name, value in zip(
        ("min", "median", "p90", "max"), np.percentile(values, (0, 50, 90, 100)))}


def input_summary(manifest):
    frames = manifest["frames"]
    stamps = [f["timestamp_s"] for f in frames]
    gaps = np.diff(stamps)
    lags = [f.get("metadata", {}).get("rgb_depth_delta_ms") for f in frames]
    return {
        "frames": len(frames),
        "duration_s": stamps[-1] - stamps[0] if stamps else 0,
        "frame_gap_s": distribution(gaps.tolist()),
        "adjacent_pairs_within_tracking_gap": int(np.count_nonzero(
            (gaps > 0) & (gaps <= VisualTracker.MAX_GAP_S))),
        "timing_eligible_frames": sum(v is None or abs(v) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS for v in lags),
        "unknown_rgb_depth_timing_frames": sum(v is None for v in lags),
        "sensor_archive_present": manifest.get("sensor_archive") is not None,
    }


def measured_motion(report):
    return bool(report["valid"] and report.get("reference_timestamp_s") is not None)


def pose_error(pose):
    pose = np.asarray(pose)
    return (float(np.linalg.norm(pose[:3, 3]) * 1000),
            float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3]) - 1) / 2, -1, 1)))))


def recorded_live_summary(manifest):
    reports = [f["metadata"]["visual_tracking"] for f in manifest["frames"]
               if "visual_tracking" in f.get("metadata", {})]
    if not reports:
        return None
    supported = [r for r in reports if r.get("valid") and r.get("steps", 0) > 0 and r.get("inliers", 0) >= 35]
    return {
        "sampled_reports": len(reports), "supported_reports": len(supported),
        "reasons": dict(Counter(r.get("reason") for r in reports)),
        "observed_segments": len({r.get("segment") for r in reports}),
        "steps": distribution([r.get("steps", 0) for r in reports]),
        "elapsed_ms": distribution([r["elapsed_ms"] for r in reports if "elapsed_ms" in r]),
        "inliers": distribution([r["inliers"] for r in supported]),
        "support_fraction": distribution([r["support_fraction"] for r in supported]),
        "median_pixel_error": distribution([r["median_pixel_error"] for r in supported]),
        "median_depth_error_mm": distribution([r["median_depth_error_m"] * 1000 for r in supported]),
        "scope": "Previously recorded tracker; capture-selected samples, not every live frame, and not an evaluation of the new tracker",
    }


def load_pair(archive, entry):
    arrays = [cv2.imdecode(np.frombuffer(archive.read(entry[key]), np.uint8), flag)
              for key, flag in (("rgb", cv2.IMREAD_COLOR), ("depth", cv2.IMREAD_UNCHANGED))]
    if any(a is None for a in arrays):
        raise ValueError(f"Cannot decode raw recording pair {entry['rgb']}")
    return cv2.cvtColor(arrays[0], cv2.COLOR_BGR2RGB), arrays[1]


def baseline_tracker(revision):
    revision = subprocess.check_output(["git", "rev-parse", "--verify", revision], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{revision}:shared/visual_tracking.py"], cwd=ROOT, text=True)
    module = types.ModuleType("shared._archived_tracking_baseline")
    module.__package__ = "shared"
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:shared/visual_tracking.py", "exec"), module.__dict__)
    return revision, source, module.VisualTracker


class Measurements:
    def __init__(self):
        self.motion = 0
        self.seeds = 0
        self.reasons = Counter()
        self.detector_calls = 0
        self.times = []
        self.motion_times = []
        self.fields = []
        self.retained = []
        self.added = []
        self.coverage = []
        self.ages = []
        self.translation = []
        self.rotation = []
        self.seed_survival = []
        self.final_ages = []

    def add(self, tracker, report, elapsed, calls, stamp, stationary=False, seed_step=False):
        motion = measured_motion(report)
        self.motion += motion
        self.reasons[report["reason"]] += 1
        self.detector_calls += calls
        current = tracker.history[-1] if tracker.history else None
        seeded = current is not None and not motion and current[3] == stamp
        self.seeds += seeded
        if "elapsed_ms" in report:
            self.times.append(elapsed)
        if motion:
            self.motion_times.append(elapsed)
        if (stationary and seed_step) or (not stationary and seeded):
            self.fields.append(len(current[2]) if current is not None else 0)
        if motion and "tracks" in report:
            field = report["tracks"]
            self.retained.append(field["retained"])
            self.added.append(field["added"])
            self.coverage.append([field["occupied_cells"], field["eligible_cells"]])
            self.ages.append(field["median_age_s"])
        if stationary and motion:
            translation, rotation = pose_error(report["camera_to_local"])
            self.translation.append(translation)
            self.rotation.append(rotation)

    def summary(self):
        return {
            "measured_motion_updates": self.motion, "seeded_reference_updates": self.seeds,
            "reasons": dict(self.reasons), "corner_detector_calls": self.detector_calls,
            "processed_update_ms": distribution(self.times),
            "measured_motion_update_ms": distribution(self.motion_times),
            "initial_measured_features": distribution(self.fields),
            "retained_features_per_motion": distribution(self.retained),
            "added_features_per_motion": distribution(self.added),
            "median_feature_age_s_per_motion": distribution(self.ages),
            "occupied_cells": distribution([c[0] for c in self.coverage]),
            "eligible_cells": distribution([c[1] for c in self.coverage]),
            "zero_motion_translation_error_mm": distribution(self.translation),
            "zero_motion_rotation_error_deg": distribution(self.rotation),
            "stationary_seed_identity_survival_fraction": distribution(self.seed_survival),
            "stationary_end_median_feature_age_s": distribution(self.final_ages),
        }


def evaluate(path, variants, stationary_steps, details):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        actual = {name: kind(settings) for name, kind in variants.items()}
        measurements = {mode: {name: Measurements() for name in variants}
                        for mode in ("recorded_time", "stationary_diagnostic")}
        detector = cv2.goodFeaturesToTrack
        calls = 0

        def counted_detector(*args, **kwargs):
            nonlocal calls
            calls += 1
            return detector(*args, **kwargs)

        def observe(mode, name, tracker, rgb, depth, metadata, frame, step):
            nonlocal calls
            calls = 0
            cv2.setRNGSeed(frame * (stationary_steps + 1) + step)
            started = time.perf_counter()
            report = tracker.update(rgb, depth, metadata)
            elapsed = (time.perf_counter() - started) * 1000
            measurement = measurements[mode][name]
            measurement.add(tracker, report, elapsed, calls, metadata["timestamp_s"],
                            mode == "stationary_diagnostic", step == 0)
            if details is not None:
                details.write(json.dumps({"session": path.name, "mode": mode, "variant": name,
                                          "frame": frame, "step": step, "wall_ms": elapsed,
                                          "detector_calls": calls, "report": report}, allow_nan=False) + "\n")

        # Warm calibration/native libraries and both code paths, outside timing.
        if manifest["frames"]:
            rgb, depth = load_pair(archive, manifest["frames"][0])
            for kind in variants.values():
                warmup = kind(settings)
                for step in range(3):
                    warmup.update(rgb, depth, {"timestamp_s": step * .1, "rgb_depth_delta_ms": 0})
        cv2.goodFeaturesToTrack = counted_detector
        try:
            for index, entry in enumerate(manifest["frames"]):
                rgb, depth = load_pair(archive, entry)
                order = list(variants) if index % 2 else list(reversed(variants))
                metadata = {**entry.get("metadata", {}), "timestamp_s": entry["timestamp_s"]}
                for name in order:
                    observe("recorded_time", name, actual[name], rgb, depth, metadata, index, 0)
                stationary = {name: kind(settings) for name, kind in variants.items()}
                seed_ids = {}
                for step in range(stationary_steps + 1):
                    step_order = order if step % 2 else list(reversed(order))
                    for name in step_order:
                        observe("stationary_diagnostic", name, stationary[name], rgb, depth,
                                {"timestamp_s": step * .1, "rgb_depth_delta_ms": 0}, index, step)
                        if step == 0 and stationary[name].history:
                            reference = stationary[name].history[-1]
                            if hasattr(reference, "ids"):
                                seed_ids[name] = reference.ids.copy()
                for name, ids in seed_ids.items():
                    reference = stationary[name].history[-1]
                    measurement = measurements["stationary_diagnostic"][name]
                    measurement.seed_survival.append(len(np.intersect1d(ids, reference.ids)) / len(ids))
                    measurement.final_ages.append(float(np.median(reference.stamp - reference.born_s)))
                if (index + 1) % 25 == 0:
                    print(f"{path.name}: {index + 1}/{len(manifest['frames'])} raw pairs", flush=True)
        finally:
            cv2.goodFeaturesToTrack = detector
    summaries = {mode: {name: measurement.summary() for name, measurement in values.items()}
                 for mode, values in measurements.items()}
    return {"archive": str(path.resolve()), "archive_sha256": digest.hexdigest(),
            "input": input_summary(manifest), "recorded_live_diagnostics": recorded_live_summary(manifest),
            **summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path, nargs="+", help="Selected-capture session ZIPs")
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--stationary-steps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    args = parser.parse_args()
    if args.stationary_steps < 1 or args.threads < 1:
        parser.error("Require positive stationary steps and threads")
    cv2.setNumThreads(args.threads)
    revision, source, baseline = baseline_tracker(args.baseline_revision)
    report = {
        "schema_version": 1, "baseline_revision": revision,
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "current_source_sha256": hashlib.sha256((ROOT / "shared/visual_tracking.py").read_bytes()).hexdigest(),
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "host": {"platform": platform.platform(), "logical_cpus": os.cpu_count()},
        "opencv_threads": cv2.getNumThreads(), "stationary_steps": args.stationary_steps,
        "notes": [
            "Before is the immediate pre-adaptive tracker; shared calibration/filtering are held constant",
            "Recorded-time replay preserves timestamps, calibration and skew; no archived pose initializes tracking",
            "Seeded references are counted separately from measured motion",
            "Stationary diagnostic duplicates each raw pair at artificial 10 Hz and zero RGB/depth skew",
            "Stationary zero motion is diagnostic truth only; it does not measure physical moving-camera accuracy or feature lifetimes",
            "Alternating before/after order, OpenCV RNG seed shared for each observation, per-session warmup excluded",
            "Timing includes calibration, depth filtering, tracking and field maintenance; excludes ZIP/image I/O, capture, server fusion and preview",
            "Saved live diagnostics are capture-selected reports from the previously recorded tracker",
        ], "sessions": {},
    }
    details = None
    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        details = args.details_output.open("w")
    try:
        for path in args.sessions:
            result = evaluate(path, {"before": baseline, "after": VisualTracker}, args.stationary_steps, details)
            report["sessions"][path.name] = result
            print(path.name, json.dumps(result), flush=True)
    finally:
        if details is not None:
            details.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
