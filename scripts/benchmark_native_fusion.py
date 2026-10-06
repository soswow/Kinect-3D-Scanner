"""Measure paired CPU confidence-weighted fusion on recorded session samples.

Example::

    python scripts/benchmark_native_fusion.py export/session.zip --samples 3 --repeats 5

Times full ScanEngine._integrate_vbg calls on fixed prepared RGB/depth and an
identity camera pose. Includes confidence, block discovery, voxel coordinates,
NumPy camera transforms, and attribute updates; excludes I/O, RGB-D preparation,
attribute clearing, tracking, extraction, and mesh construction. Existing active
blocks are reused, so this does not measure first-observation block allocation.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from importlib import metadata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.benchmark_native_kernels import (
    file_hash,
    git_commit,
    native_mode,
    read_frame,
)
from scripts.profile_backends import source_hash
from shared.calibration import prepare_rgbd
from shared.native import kernels, native_status
from shared.settings import ScanSettings


def reset_attributes(attributes, ids):
    for values in attributes.values():
        values[ids] = 0


def attribute_hashes(attributes, ids):
    """Hash exact active values in bounded chunks, without copying the volume."""
    result = {}
    for name, values in attributes.items():
        digest = hashlib.sha256()
        for start in range(0, len(ids), 32):
            chunk = values[ids[start : start + 32]]
            digest.update(memoryview(chunk).cast("B"))
        result[name] = digest.hexdigest()
    return result


def activate_frame(engine, depth, extrinsic, o3d):
    volume = engine.vbg
    hashmap = volume.hashmap()
    previous = hashmap.active_buf_indices()
    if len(previous):
        hashmap.erase(hashmap.key_tensor()[previous])
    image = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth)))
    blocks = volume.compute_unique_block_coordinates(
        image,
        engine.intrinsic_tensor,
        o3d.core.Tensor(extrinsic, dtype=o3d.core.float64),
        depth_scale=1000.0,
        depth_max=engine.max_depth_m,
        trunc_voxel_multiplier=engine.sdf_trunc / engine.voxel_size,
    )
    if not len(blocks):
        raise ValueError("Selected frame has no reconstructable voxel blocks")
    if len(blocks) > engine.BLOCK_COUNT:
        raise ValueError(
            "Sample exceeds configured block capacity; increase KINECT_BLOCK_COUNT"
        )
    hashmap.activate(blocks)
    ids = hashmap.active_buf_indices().numpy().astype(np.int64)
    attributes = {
        name: volume.attribute(name)
        .numpy()
        .reshape(-1, 4096, 3 if name == "color" else 1)
        for name in ("tsdf", "weight", "color")
    }
    return ids, attributes


def timed_integrate(engine, mode, rgb, depth, extrinsic, attributes, ids):
    # Each pair begins with identical weights, TSDF, color, and active blocks.
    reset_attributes(attributes, ids)
    with native_mode(mode):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            started = time.perf_counter()
            engine._integrate_vbg(rgb, depth, extrinsic)
            elapsed = (time.perf_counter() - started) * 1000
        finally:
            if was_enabled:
                gc.enable()
    if engine.vbg.hashmap().size() != len(ids):
        raise RuntimeError("Timed integration allocated unexpected blocks")
    return elapsed


def benchmark_sample(engine, rgb, depth, index, repeats, o3d):
    extrinsic = np.eye(4)
    ids, attributes = activate_frame(engine, depth, extrinsic, o3d)
    parity = {}
    statistics = {}
    # These observations also warm both modes. Repeat an observation to check
    # updates with nonzero prior weights, in addition to first-observation math.
    for mode in ("off", "on"):
        reset_attributes(attributes, ids)
        with native_mode(mode):
            for _ in range(2):
                engine._integrate_vbg(rgb, depth, extrinsic)
        if engine.vbg.hashmap().size() != len(ids):
            raise RuntimeError("Parity integration allocated unexpected blocks")
        parity[mode] = attribute_hashes(attributes, ids)
        statistics[mode] = dict(engine._last_fusion_stats)
    exact = parity["off"] == parity["on"] and statistics["off"] == statistics["on"]
    result = {
        "index": index,
        "active_blocks": len(ids),
        "allocated_blocks": int(engine.vbg.hashmap().capacity()),
        "bit_exact_attributes_and_statistics": exact,
        "attribute_sha256": parity,
        "observation_statistics": statistics,
    }
    if exact:
        pairs = []
        for repeat in range(repeats):
            order = ("off", "on") if (index + repeat) % 2 == 0 else ("on", "off")
            pairs.append(
                {
                    mode: timed_integrate(
                        engine, mode, rgb, depth, extrinsic, attributes, ids
                    )
                    for mode in order
                }
            )
        result["paired_timings_ms"] = pairs
    return result


def summarize(samples):
    pairs = [pair for sample in samples for pair in sample.get("paired_timings_ms", [])]
    if not pairs:
        return None
    result = {}
    for mode, label in (("off", "numpy"), ("on", "native")):
        values = [pair[mode] for pair in pairs]
        result[label] = {
            "measurements": len(values),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    result["median_paired_speedup"] = float(
        np.median([pair["off"] / max(pair["on"], 1e-9) for pair in pairs])
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("archives", type=Path, nargs="+", help="Saved session ZIPs")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-output/native-fusion.json"
    )
    args = parser.parse_args()
    if not 1 <= args.samples <= 100 or not 1 <= args.repeats <= 100:
        parser.error("Samples and repeats must each be 1–100")
    import open3d as o3d

    from scanner_server.engine import ScanEngine

    with native_mode("on"):
        extension = kernels()
        status = native_status()
    if not hasattr(extension, "integrate_weighted_cpu"):
        raise RuntimeError("Rebuild ./native to include weighted fusion")
    extension_path = Path(extension.__file__).resolve()
    extension_before = file_hash(extension_path)
    source_before = source_hash()
    script_before = file_hash(__file__)
    helper_before = file_hash(ROOT / "scripts/benchmark_native_kernels.py")
    engine = ScanEngine(device="cpu")
    sessions = []
    for path in args.archives:
        input_before = file_hash(path)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            settings = ScanSettings.from_dict(manifest["settings"])
            if not settings.confidence_fusion:
                raise ValueError(
                    f"Session does not enable confidence fusion: {path.name}"
                )
            frames = manifest["frames"]
            if not frames:
                raise ValueError(f"Session has no observations: {path.name}")
            indices = np.unique(
                np.linspace(
                    0, len(frames) - 1, min(args.samples, len(frames)), dtype=int
                )
            ).tolist()
            # Release the previous volume before reset allocates its replacement.
            engine.vbg = None
            engine.reset(settings=settings)
            sampled = []
            for index in indices:
                print(f"Fusion {path.name} frame {index}/{len(frames) - 1}", flush=True)
                rgb, depth = read_frame(archive, frames[index])
                with native_mode("off"):
                    rgb, depth = prepare_rgbd(rgb, depth, settings)
                sampled.append(
                    benchmark_sample(engine, rgb, depth, index, args.repeats, o3d)
                )
        sessions.append(
            {
                "archive": path.name,
                "input_sha256": input_before,
                "input_changed": file_hash(path) != input_before,
                "frames": len(frames),
                "timed_frame_indices": indices,
                "settings_sha256": hashlib.sha256(
                    json.dumps(
                        settings.to_dict(), sort_keys=True, allow_nan=False
                    ).encode()
                ).hexdigest(),
                "bit_exact_attributes_and_statistics": all(
                    sample["bit_exact_attributes_and_statistics"] for sample in sampled
                ),
                "samples": sampled,
                "integrate_vbg": summarize(sampled),
            }
        )
    source_changed = source_hash() != source_before
    extension_changed = file_hash(extension_path) != extension_before
    script_changed = file_hash(__file__) != script_before
    helper_changed = (
        file_hash(ROOT / "scripts/benchmark_native_kernels.py") != helper_before
    )
    # Recheck all archives after all timing, so later changes invalidate earlier inputs too.
    for path, session in zip(args.archives, sessions):
        session["input_changed"] |= file_hash(path) != session["input_sha256"]
    valid = not (
        source_changed or extension_changed or script_changed or helper_changed
    ) and all(
        session["bit_exact_attributes_and_statistics"] and not session["input_changed"]
        for session in sessions
    )
    if not valid:
        for session in sessions:
            if session["integrate_vbg"]:
                session["integrate_vbg"].pop("median_paired_speedup", None)
    try:
        package_version = metadata.version("kinect-scanner-native")
    except metadata.PackageNotFoundError:
        package_version = None
    report = {
        "valid": valid,
        "method": "Warmed alternating-order paired full confidence-weighted CPU fusion calls; identical zero attributes and preactivated blocks; GC disabled only while timing",
        "scope": "Includes confidence, block discovery, voxel coordinates, NumPy transform, and attribute update; excludes I/O, RGB-D preparation, attribute reset, first block allocation, tracking, extraction, and mesh work",
        "parity_requirement": "Exact SHA256 of all active TSDF, weight, and color bytes plus observation statistics after two identical observations",
        "source_sha256": source_before,
        "source_changed_during_run": source_changed,
        "benchmark_script_sha256": script_before,
        "benchmark_script_changed_during_run": script_changed,
        "benchmark_helper_sha256": helper_before,
        "benchmark_helper_changed_during_run": helper_changed,
        "git_commit": git_commit(),
        "extension_path": str(extension_path),
        "extension_sha256": extension_before,
        "extension_changed_during_run": extension_changed,
        "extension_package_version": package_version,
        "native": status,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "open3d": o3d.__version__,
        "omp_threads": os.environ["OMP_NUM_THREADS"],
        "initial_blocks": engine.BLOCK_COUNT,
        "samples_per_session": args.samples,
        "repeats_per_sample": args.repeats,
        "sessions": sessions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
