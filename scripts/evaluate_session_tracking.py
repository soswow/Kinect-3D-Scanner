"""Interleaved read-only tracking comparisons on exported session ZIPs.

Camera comparison loads an explicit historical visual tracker. Server comparison
only bypasses the per-registration source cache: all other current algorithms,
settings, recorded hints, and acceptance gates are identical. Sparse stored
captures cannot reproduce the camera's original full-rate stream.
"""

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from scanner_server.engine import ScanEngine
from shared.settings import ScanSettings
from shared.visual_tracking import VisualTracker


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(output, inputs):
    if output.suffix.lower() != ".json":
        raise ValueError("Output must be a JSON report")
    for source in inputs:
        if output.resolve() == source.resolve() or (
                output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("Report output must differ from every input session")


def frames(archive, entries):
    for index, entry in enumerate(entries):
        rgb = np.array(Image.open(io.BytesIO(archive.read(entry["rgb"])))).astype(np.uint8)
        depth = np.array(Image.open(io.BytesIO(archive.read(entry["depth"])))).astype(np.uint16)
        metadata = dict(entry.get("metadata", {}))
        metadata.pop("reference_pose", None)
        metadata.update(timestamp_s=entry["timestamp_s"], frame_id=index)
        yield rgb, depth, metadata


def distribution(values):
    if not values:
        return None
    return {"samples": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))}


def camera_run(tracker_type, settings, archive, entries, seed):
    cv2.setRNGSeed(seed)
    tracker, reports, elapsed = tracker_type(settings), [], []
    for index, (rgb, depth, metadata) in enumerate(frames(archive, entries)):
        started = time.perf_counter()
        report = tracker.update(rgb, depth, metadata)
        ms = (time.perf_counter() - started) * 1000
        reports.append(report)
        if index and "elapsed_ms" in report:
            elapsed.append(ms)
    return {"reports": reports, "compute_ms": elapsed}


def compare_camera(before, after):
    keys = ("valid", "steps", "reference_timestamp_s", "inliers", "matches",
            "support_fraction", "distributed", "reason")
    a, b = before["reports"], after["reports"]
    return {"support_disagreement_indices": [i for i, (x, y) in enumerate(zip(a, b))
            if any(x.get(k) != y.get(k) for k in keys)],
            "max_pose_matrix_difference": max(float(np.max(np.abs(
                np.asarray(x["camera_to_local"]) - np.asarray(y["camera_to_local"]))))
                for x, y in zip(a, b)),
            "valid_reports_before": sum(bool(r["valid"]) for r in a),
            "valid_reports_after": sum(bool(r["valid"]) for r in b),
            "measured_motion_reports_after": sum(bool(r["valid"]) and r["steps"] > 0 for r in b)}


def server_run(cached, settings, archive, entries, seed):
    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    engine = ScanEngine(device="cpu", tracking="legacy")
    engine.reset(settings=settings)
    if not cached:
        engine._register = types.MethodType(ScanEngine._register.__wrapped__, engine)
    frame_times = []
    for rgb, depth, metadata in frames(archive, entries):
        stored = engine.store_frame(rgb, depth, metadata)
        if not stored["success"]:
            raise ValueError(stored["message"])
        started = time.perf_counter()
        engine.process_frames()
        frame_times.append((time.perf_counter() - started) * 1000)
    result = {"poses": [(i, p.copy()) for i, p in engine.poses],
              "diagnostics": engine.diagnostics, "frame_ms": frame_times[1:],
              "tracking_ms": [d["timings_ms"]["tracking"] for d in engine.diagnostics
                              if "tracking" in d.get("timings_ms", {})],
              "backend": engine.backend}
    del engine
    gc.collect()
    return result


def compare_server(before, after):
    a, b = dict(before["poses"]), dict(after["poses"])
    common = sorted(a.keys() & b.keys())
    keys = ("success", "method", "inliers", "matches", "fitness", "inlier_rmse")
    # Float diagnostics can change at machine precision while support/poses agree.
    changed = []
    for i, (x, y) in enumerate(zip(before["diagnostics"], after["diagnostics"])):
        for key in keys:
            left, right = x.get(key), y.get(key)
            equal = np.isclose(left, right, atol=1e-10, rtol=1e-10) if (
                isinstance(left, (float, int)) and isinstance(right, (float, int))) else left == right
            if not equal:
                changed.append(i)
                break
    relative = [np.linalg.inv(a[i]) @ b[i] for i in common]
    return {"accepted_before": list(a), "accepted_after": list(b),
            "diagnostic_disagreement_indices": changed,
            "max_common_pose_translation_difference_m": max(
                (float(np.linalg.norm(p[:3, 3])) for p in relative), default=None),
            "max_common_pose_rotation_difference_deg": max(
                (float(np.degrees(np.arccos(np.clip((np.trace(p[:3, :3]) - 1) / 2, -1, 1))))
                 for p in relative), default=None),
            "max_common_pose_matrix_difference": max(
                (float(np.max(np.abs(a[i] - b[i]))) for i in common), default=None)}


def repeatability_controls(path, repeats, server_frames):
    checksum = file_hash(path)
    result = {"session": path.name, "input_sha256": checksum, "server_repeatability": {}}
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        selected = manifest["frames"][:server_frames]
        result["selected_indices"] = list(range(len(selected)))
        # Identical seeds, implementation and observations expose run variation.
        for cached, label in ((False, "uncached"), (True, "cached")):
            server_run(cached, settings, archive, selected[:min(8, len(selected))], 0)
            anchor, comparisons = None, []
            for repeat in range(repeats):
                current = server_run(cached, settings, archive, selected, 0)
                if anchor is None:
                    anchor = current
                else:
                    comparisons.append(compare_server(anchor, current))
                print(f"{path.name}: repeatability {label} {repeat + 1}/{repeats}", flush=True)
            result["server_repeatability"][label] = comparisons
    result["input_unchanged"] = file_hash(path) == checksum
    return result


def evaluate(path, baseline, repeats, server_frames):
    checksum = file_hash(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        entries = manifest["frames"]
        result = {"session": path.name, "input_sha256": checksum, "stored_views": len(entries),
                  "settings_sha256": hashlib.sha256(json.dumps(
                      settings.to_dict(), sort_keys=True).encode()).hexdigest(),
                  "camera": {}, "server": {"selected_indices": list(range(min(server_frames, len(entries))))}}
        for phase in ("camera", "server"):
            selected = entries if phase == "camera" else entries[:server_frames]
            variants = {"before": baseline, "after": VisualTracker} if phase == "camera" else {
                "before": False, "after": True}
            run = camera_run if phase == "camera" else server_run
            compare = compare_camera if phase == "camera" else compare_server
            # Warm both implementations on the same short real sequence.
            for implementation in variants.values():
                run(implementation, settings, archive, selected[:min(8, len(selected))], 0)
            timings, comparisons = {name: {} for name in variants}, []
            for repeat in range(repeats):
                runs = {}
                for name in (("after", "before") if repeat % 2 == 0 else ("before", "after")):
                    runs[name] = run(variants[name], settings, archive, selected, repeat)
                    for metric in ("compute_ms",) if phase == "camera" else ("frame_ms", "tracking_ms"):
                        timings[name].setdefault(metric, []).extend(runs[name][metric])
                    print(f"{path.name}: {phase} {name} repeat {repeat + 1}/{repeats}", flush=True)
                comparisons.append(compare(runs["before"], runs["after"]))
            result[phase].update(timing_ms={name: {key: distribution(values) for key, values in metrics.items()}
                                           for name, metrics in timings.items()}, comparisons=comparisons)
    result["input_unchanged"] = file_hash(path) == checksum
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--server-frames", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--controls-only", action="store_true",
                        help="Repeat each server variant with identical seeds to measure run variation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.server_frames < 2:
        parser.error("Require positive repeats and at least two server frames")
    if args.controls_only and args.repeats < 2:
        parser.error("Repeatability controls require at least two repeats")
    paths = sorted(args.sessions.glob("*.zip")) if args.sessions.is_dir() else [args.sessions]
    if not paths:
        parser.error("No session ZIPs found")
    try:
        validate_output(args.output, paths)
    except ValueError as error:
        parser.error(str(error))
    cv2.setNumThreads(4)
    revision = subprocess.check_output(["git", "rev-parse", args.baseline_revision], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{revision}:shared/visual_tracking.py"], cwd=ROOT, text=True)
    module = types.ModuleType("shared._recorded_tracking_baseline")
    module.__package__ = "shared"
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:shared/visual_tracking.py", "exec"), module.__dict__)
    report = {"schema_version": 1, "baseline_revision": revision,
              "current_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (
                  ROOT / "shared/visual_tracking.py", ROOT / "scanner_server/engine.py",
                  ROOT / "scanner_server/tracking_cache.py")},
              "baseline_camera_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "versions": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                           "numpy": np.__version__, "open3d": o3d.__version__},
              "threads": {"opencv": cv2.getNumThreads(), "omp": os.environ["OMP_NUM_THREADS"]},
              "repeats": args.repeats, "controls_only": args.controls_only, "sessions": [],
              "notes": ["Original timing, calibration and depth units retained; inputs never overwritten",
                        "Replay frame_id replaces upload identity; reference_pose is removed and never used to initialize tracking",
                        "Historical camera code shares current calibration/depth/native helpers; this is a tracker-code comparison",
                        "Sparse uploaded captures do not reproduce continuous camera-rate tracking; no absolute pose truth",
                        "Camera timing excludes decoding, origin frame and timing-guard rejections",
                        "Server uses current algorithms with only the ICP source cache toggled; archived poses are never seeds",
                        "Server timings exclude decoding and initial frame; selected prefix is explicitly listed",
                        "Interleaved repeats after short warmups; no acquisition/network/rendering or Finish included"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        result = (repeatability_controls(path, args.repeats, args.server_frames) if args.controls_only else
                  evaluate(path, module.VisualTracker, args.repeats, args.server_frames))
        report["sessions"].append(result)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
