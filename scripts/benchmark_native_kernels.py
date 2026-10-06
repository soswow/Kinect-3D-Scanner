"""Check native RGB-D parity and measure warmed, paired image preparation.

Reads session ZIPs without extracting or modifying the recorded observations.
Use --all-frames for full-session parity; timing always uses a bounded sample.
These helper timings exclude tracking, fusion, I/O, transport, and mesh work.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.profile_backends import source_hash
from shared.calibration import prepare_rgbd
from shared.native import kernels, native_status
from shared.settings import ScanSettings


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def native_mode(value):
    previous = os.environ.get("KINECT_NATIVE")
    os.environ["KINECT_NATIVE"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("KINECT_NATIVE", None)
        else:
            os.environ["KINECT_NATIVE"] = previous


def prepare_with_mode(mode, rgb, depth, settings):
    with native_mode(mode):
        return prepare_rgbd(rgb, depth, settings)


def read_frame(archive, frame):
    rgb = cv2.imdecode(
        np.frombuffer(archive.read(frame["rgb"]), np.uint8), cv2.IMREAD_COLOR
    )
    depth = cv2.imdecode(
        np.frombuffer(archive.read(frame["depth"]), np.uint8), cv2.IMREAD_UNCHANGED
    )
    if rgb is None or depth is None:
        raise ValueError("Cannot decode an archived RGB/depth observation")
    if rgb.dtype != np.uint8 or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("Require archived uint8 RGB and uint16 depth")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth


def differences(reference, native):
    result = {}
    for name, before, after in zip(("color", "depth"), reference, native):
        if before.shape != after.shape or before.dtype != after.dtype:
            result[name] = {"shape_or_dtype_changed": True}
        elif not np.array_equal(before, after):
            difference = np.abs(before.astype(np.int64) - after.astype(np.int64))
            result[name] = {
                "changed_values": int(np.count_nonzero(difference)),
                "max_absolute_difference": int(difference.max()),
            }
    return result


def check_session(path, samples, all_frames):
    before_hash = file_hash(path)
    sampled = {}
    mismatches = []
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        frames = manifest["frames"]
        if not frames:
            raise ValueError(f"Recording has no observations: {path}")
        indices = np.unique(
            np.linspace(0, len(frames) - 1, min(samples, len(frames)), dtype=int)
        ).tolist()
        parity_indices = range(len(frames)) if all_frames else indices
        for index in parity_indices:
            rgb, depth = read_frame(archive, frames[index])
            reference = prepare_with_mode("off", rgb, depth, settings)
            native = prepare_with_mode("on", rgb, depth, settings)
            changed = differences(reference, native)
            if changed:
                mismatches.append({"index": index, **changed})
            if index in indices:
                sampled[index] = (rgb, depth)
        checked = len(frames) if all_frames else len(indices)
    result = {
        "archive": path.name,
        "input_sha256": before_hash,
        "frames": len(frames),
        "parity_frames": checked,
        "all_frame_parity": all_frames,
        "bit_exact_colors_and_depth": not mismatches,
        "mismatches": mismatches,
        "timed_frame_indices": indices,
        "settings_sha256": hashlib.sha256(
            json.dumps(settings.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }
    return result, settings, sampled


def timed_prepare(mode, rgb, depth, settings):
    # Environment switching and GC state restoration are outside the timer.
    with native_mode(mode):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            started = time.perf_counter()
            output = prepare_rgbd(rgb, depth, settings)
            elapsed = (time.perf_counter() - started) * 1000
        finally:
            if was_enabled:
                gc.enable()
    del output
    return elapsed


def benchmark(settings, sampled, repeats):
    timings = {"off": [], "on": []}
    ratios = []
    for index, (rgb, depth) in sampled.items():
        for mode in ("off", "on"):
            prepare_with_mode(mode, rgb, depth, settings)
        for repeat in range(repeats):
            order = ("off", "on") if (index + repeat) % 2 == 0 else ("on", "off")
            pair = {}
            for mode in order:
                pair[mode] = timed_prepare(mode, rgb, depth, settings)
                timings[mode].append(pair[mode])
            ratios.append(pair["off"] / max(pair["on"], 1e-9))
    result = {}
    for mode, label in (("off", "numpy"), ("on", "native")):
        values = timings[mode]
        result[label] = {
            "measurements": len(values),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    result["median_paired_speedup"] = float(np.median(ratios))
    return result


def git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+", help="Saved session ZIPs")
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-output/native-kernels.json",
    )
    args = parser.parse_args()
    if not 1 <= args.samples <= 100 or not 1 <= args.repeats <= 100:
        parser.error("Samples and repeats must each be 1–100")
    with native_mode("on"):
        extension = kernels()  # Fail explicitly rather than timing a fallback.
        status = native_status()
    extension_path = Path(extension.__file__).resolve()
    extension_before = file_hash(extension_path)
    source_before = source_hash()
    sessions = []
    for path in args.archives:
        print(f"Checking {path.name}", flush=True)
        sessions.append(check_session(path, args.samples, args.all_frames))
    parity_passed = all(
        result["bit_exact_colors_and_depth"] for result, _, _ in sessions
    )
    if parity_passed:
        for result, settings, sampled in sessions:
            print(f"Timing {result['archive']}", flush=True)
            result["prepare_rgbd"] = benchmark(settings, sampled, args.repeats)
    for path, (result, _, _) in zip(args.archives, sessions):
        result["input_changed"] = file_hash(path) != result["input_sha256"]
    source_changed = source_hash() != source_before
    extension_changed = file_hash(extension_path) != extension_before
    valid = (
        parity_passed
        and not source_changed
        and not extension_changed
        and not any(result["input_changed"] for result, _, _ in sessions)
    )
    if not valid:
        for result, _, _ in sessions:
            if "prepare_rgbd" in result:
                result["prepare_rgbd"].pop("median_paired_speedup", None)
    try:
        package_version = metadata.version("kinect-scanner-native")
    except metadata.PackageNotFoundError:
        package_version = None
    report = {
        "valid": valid,
        "method": "Warmed alternating-order paired RGB-D helper calls; GC disabled only while timing",
        "scope": "No I/O, tracking, fusion, transport, final mesh, or ground-truth accuracy measurement",
        "parity_requirement": "Entire output color and depth arrays must be bit-exact before timing",
        "source_sha256": source_before,
        "source_changed_during_run": source_changed,
        "benchmark_script_sha256": file_hash(__file__),
        "git_commit": git_commit(),
        "extension_sha256": extension_before,
        "extension_changed_during_run": extension_changed,
        "extension_package_version": package_version,
        "native": status,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "omp_threads": os.environ["OMP_NUM_THREADS"],
        "samples_per_session": args.samples,
        "repeats_per_sample": args.repeats,
        "sessions": [result for result, _, _ in sessions],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
