"""Replay an exported ZIP in an isolated process without touching a server.

Examples (timings exclude image decoding, imports, and report preparation)::

    KINECT_NATIVE=off python scripts/profile_session.py export/scan.zip --limit 20
    KINECT_NATIVE=off python scripts/profile_session.py export/scan.zip --finish --cprofile
    KINECT_NATIVE=on python scripts/profile_session.py export/scan.zip --finish \
        --output benchmark-output/native.json --compare benchmark-output/python.json

Use --use-pose-seeds --finish to profile Finish independently of live tracking.
Archived poses remain proposals: the normal reconnection validates them again.
Raw images and geometry artifacts must remain under ignored benchmark-output/.
"""

import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
import pstats
import random
import sys
import time
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.process_metrics import finish_cuda_worker, gpu_info, peak_rss_bytes


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for folder in (ROOT / "scanner_server", ROOT / "shared", ROOT / "native"):
        for path in sorted(folder.rglob("*")):
            if (
                path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".toml")
                and "build" not in path.parts
            ):
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def profile_summary(profile, destination):
    profile.dump_stats(str(destination))
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(
        profile
    ).stats.items():
        try:
            filename = str(Path(filename).relative_to(ROOT))
        except ValueError:
            pass
        rows.append(
            {
                "function": f"{filename}:{line}({name})",
                "primitive_calls": primitive,
                "calls": calls,
                "self_s": own,
                "cumulative_s": cumulative,
            }
        )
    return {
        "accounted_self_s": sum(row["self_s"] for row in rows),
        "warning": "Open3D/pybind profiler hooks may omit calls on some Python versions; use phase and stage wall times for speed claims. Self time includes native calls.",
        "self": sorted(rows, key=lambda row: row["self_s"], reverse=True)[:40],
        "cumulative": sorted(rows, key=lambda row: row["cumulative_s"], reverse=True)[
            :40
        ],
    }


def run_phase(action, enabled, destination):
    profile = cProfile.Profile() if enabled else None
    started = time.perf_counter()
    if profile:
        profile.enable()
    try:
        result = action()
    finally:
        if profile:
            profile.disable()
    elapsed = time.perf_counter() - started
    return result, elapsed, profile_summary(profile, destination) if profile else None


def seed_poses(engine, report, selected_indices, np):
    from scanner_server.fragments import _rigid

    remap = {original: stored for stored, original in enumerate(selected_indices)}
    seen, seeds = set(), []
    for item in report.get("poses", []):
        original = item["index"]
        pose = np.asarray(item["camera_to_world"], dtype=float)
        if type(original) is not int or original in seen or not _rigid(pose):
            raise ValueError("Invalid archived pose seed")
        seen.add(original)
        if original in remap:
            seeds.append((remap[original], pose))
    if not seeds:
        raise ValueError("No archived pose seeds in the selected frames")
    seeds.sort(key=lambda item: item[0])
    source_results = {
        remap[f["index"]]: f for f in report.get("frames", []) if f["index"] in remap
    }
    accepted = {index for index, _ in seeds}
    engine.poses = seeds
    engine.diagnostics = [
        {**source_results.get(index, {}), "index": index, "success": index in accepted}
        for index in range(engine.stored_count)
    ]
    engine.frame_count = len(seeds)
    engine._processed_count = engine.stored_count
    engine.cumulative_T = seeds[-1][1].copy()
    engine._pose_seeds_only = True


def stage_summary(diagnostics, totals, np):
    result = {}
    for name, total in totals.items():
        values = [
            item["timings_ms"][name]
            for item in diagnostics
            if name in item.get("timings_ms", {})
        ]
        result[name] = {
            "total_ms": total,
            "frames": len(values),
            "p50_ms": float(np.percentile(values, 50)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None,
        }
    return result


def geometry_summary(engine, destination, np):
    mesh = engine.mesh
    points = (
        np.asarray(mesh.vertices)
        if mesh is not None
        else np.asarray(engine.point_cloud.points)
        if engine.point_cloud is not None
        else np.asarray(engine._live_points)
    )
    faces = (
        np.asarray(mesh.triangles)
        if mesh is not None
        else np.empty((0, 3), dtype=np.int32)
    )
    colors = (
        np.asarray(mesh.vertex_colors)
        if mesh is not None
        else np.asarray(engine.point_cloud.colors)
        if engine.point_cloud is not None
        else np.asarray(engine._live_colors)
    )
    # Save the actual result to compare native/Python surfaces independently of vertex order.
    np.savez_compressed(destination, points=points, faces=faces, colors=colors)
    quantized = np.rint(points * 1e6).astype(np.int64)
    if len(points):
        quantized = quantized[np.lexsort(quantized.T[::-1])]
    return {
        "vertices": len(points),
        "triangles": len(faces),
        "bounds_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
        if len(points)
        else None,
        "surface_positions_1um_sha256": hashlib.sha256(quantized.tobytes()).hexdigest(),
        "artifact": str(destination.resolve()),
    }


def comparison_settings(report, *, preview_resolution=False):
    """Permit only an explicit preview tradeoff with unchanged effective Finish detail."""
    settings = dict(report["settings"])
    if not preview_resolution:
        return settings
    final_voxel = settings.get("final_voxel_m") or settings["voxel_m"]
    if (not report["finish_requested"] or not report.get("mesh_built", False)
            or report["final_reconstruction"].get("voxel_m") != final_voxel):
        raise ValueError("Preview comparison requires a completed mesh at the requested final voxel size")
    settings.pop("voxel_m")
    settings.pop("final_voxel_m", None)
    settings["effective_final_voxel_m"] = final_voxel
    return settings


def compare_quality(report, baseline_path, np, *, preview_resolution=False):
    import open3d as o3d

    baseline = json.loads(baseline_path.read_text())
    if (
        report["input_sha256"],
        report["selected_indices"],
        report["seed"],
        comparison_settings(report, preview_resolution=preview_resolution),
        report["finish_requested"],
        report["pose_seeds_used"],
    ) != (
        baseline["input_sha256"],
        baseline["selected_indices"],
        baseline["seed"],
        comparison_settings(baseline, preview_resolution=preview_resolution),
        baseline["finish_requested"],
        baseline["pose_seeds_used"],
    ):
        raise ValueError(
            "Comparison requires identical input, frame selection, settings, seed, and phases"
        )
    old_poses = {
        item["index"]: np.asarray(item["camera_to_world"]) for item in baseline["poses"]
    }
    new_poses = {
        item["index"]: np.asarray(item["camera_to_world"]) for item in report["poses"]
    }
    translations, rotations = [], []
    for index in old_poses.keys() & new_poses.keys():
        delta = np.linalg.inv(old_poses[index]) @ new_poses[index]
        translations.append(float(np.linalg.norm(delta[:3, 3])))
        rotations.append(
            float(
                np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
            )
        )
    result = {
        "baseline": str(baseline_path.resolve()),
        "preview_resolution_comparison": preview_resolution,
        "same_accepted_indices": report["accepted_indices"]
        == baseline["accepted_indices"],
        "same_mesh_success": report["mesh_built"] == baseline["mesh_built"],
        "same_vertex_count": report["geometry"]["vertices"]
        == baseline["geometry"]["vertices"],
        "same_triangle_count": report["geometry"]["triangles"]
        == baseline["geometry"]["triangles"],
        "common_poses": len(translations),
        "max_pose_translation_delta_m": max(translations, default=None),
        "max_pose_rotation_delta_deg": max(rotations, default=None),
        "same_surface_positions_1um": report["geometry"]["surface_positions_1um_sha256"]
        == baseline["geometry"]["surface_positions_1um_sha256"],
    }
    previous = np.load(baseline["geometry"]["artifact"])["points"]
    current = np.load(report["geometry"]["artifact"])["points"]
    if len(previous) and len(current):
        old_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(previous))
        new_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(current))
        distances = np.concatenate(
            (
                np.asarray(old_cloud.compute_point_cloud_distance(new_cloud)),
                np.asarray(new_cloud.compute_point_cloud_distance(old_cloud)),
            )
        )
        result["symmetric_vertex_distance_m"] = {
            "rms": float(np.sqrt(np.mean(distances**2))),
            "p95": float(np.percentile(distances, 95)),
            "max": float(distances.max()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-output/session-profile.json"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--bundle-adjustment", action="store_true",
                        help="Enable joint RGB-D refinement during Finish")
    parser.add_argument("--use-pose-seeds", action="store_true")
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Adds Python profiler overhead; use unprofiled runs for speed comparisons",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--tracking", choices=("auto", "legacy", "tensor"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--live-voxel", type=float,
                        help="Explicit preview tradeoff; retain the archive's effective final voxel size")
    parser.add_argument("--visual-fallback-sift", action="store_true",
                        help="Research only: try original verified SIFT proposals after ORB fails")
    parser.add_argument(
        "--final-block-count",
        type=int,
        help="Explicit common final-volume budget for matched comparisons (1–50000)",
    )
    args = parser.parse_args()
    if args.stride < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Require positive stride and limit")
    if args.use_pose_seeds and not args.finish:
        parser.error("--use-pose-seeds requires --finish")
    if args.bundle_adjustment and not args.finish:
        parser.error("--bundle-adjustment requires --finish")
    if args.final_block_count is not None and not 1 <= args.final_block_count <= 50000:
        parser.error("Final block count must be 1–50000")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
    sys.path.insert(0, str(ROOT))
    import cv2
    import numpy as np
    import open3d as o3d
    from PIL import Image

    from scanner_server.engine import ScanEngine
    policy_path = ROOT / "scripts/research/archive/adaptive_visual_experiment.py"
    policy_hash = file_hash(policy_path) if args.visual_fallback_sift else None
    if args.visual_fallback_sift:
        from scripts.research.archive.adaptive_visual_experiment import AdaptiveVisualEngine
        ScanEngine = AdaptiveVisualEngine
    from shared.settings import ScanSettings

    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    o3d.utility.random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_before = file_hash(args.session)
    source_before = source_hash()
    extension_spec = importlib.util.find_spec("_kinect_native")
    extension_path = (
        Path(extension_spec.origin)
        if extension_spec and extension_spec.origin
        else None
    )
    extension_hash = file_hash(extension_path) if extension_path else None
    with zipfile.ZipFile(args.session) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        original_settings_sha256 = hashlib.sha256(
            json.dumps(settings.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        overrides = {}
        if args.live_voxel is not None:
            overrides.update(voxel_m=args.live_voxel,
                             final_voxel_m=settings.final_voxel_m or settings.voxel_m)
            settings = replace(settings, **overrides)
        if args.bundle_adjustment:
            overrides["bundle_adjustment"] = True
            settings = replace(settings, **overrides)
        if args.final_block_count is not None:
            overrides["final_block_count"] = args.final_block_count
            settings = replace(settings, **overrides)
        if args.use_pose_seeds and not settings.reconnect_fragments:
            parser.error(
                "Archived-pose Finish requires fragment reconnection in the session settings; replay live to populate its volume"
            )
        indices = list(range(len(manifest["frames"])))[:: args.stride][: args.limit]
        if not indices or len(indices) > ScanEngine.MAX_FRAMES:
            raise ValueError(
                f"Session selection must contain 1–{ScanEngine.MAX_FRAMES} frames"
            )
        frames = []
        for original in indices:
            item = manifest["frames"][original]
            with archive.open(item["rgb"]) as stream, Image.open(stream) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            with archive.open(item["depth"]) as stream, Image.open(stream) as image:
                depth = np.array(image, dtype=np.uint16)
            frames.append((rgb, depth, item))
        archived = (
            json.loads(
                archive.read(manifest.get("reconstruction", "reconstruction.json"))
            )
            if args.use_pose_seeds
            else None
        )
    engine = ScanEngine(device=args.device, tracking=args.tracking)
    engine.reset(settings=settings)

    def live():
        for index, (rgb, depth, item) in enumerate(frames):
            metadata = dict(item.get("metadata", {}))
            metadata.pop("reference_pose", None)
            metadata.update(frame_id=index, timestamp_s=item["timestamp_s"])
            stored = engine.store_frame(rgb, depth, metadata)
            if not stored["success"]:
                raise ValueError(f"Cannot store frame {index}: {stored['message']}")
            if not args.use_pose_seeds:
                engine.process_frames()
            print(
                f"{index + 1}/{len(frames)} accepted={engine.frame_count} "
                f"{engine.diagnostics[-1].get('message', '') if engine.diagnostics else ''}", flush=True
            )

    _, live_s, live_profile = run_phase(
        live,
        args.cprofile and not args.use_pose_seeds,
        args.output.with_suffix(".live.pstats"),
    )
    live_stages = dict(engine.stage_totals_ms)
    live_diagnostics = list(engine.diagnostics)
    accepted_before_finish = engine.frame_count
    accepted_indices_before_finish = [index for index, _ in engine.poses]
    print(f"Live complete: {live_s:.3f}s, accepted={accepted_before_finish}/{len(frames)}", flush=True)
    if args.use_pose_seeds:
        seed_poses(engine, archived, indices, np)
        accepted_before_finish = engine.frame_count
        accepted_indices_before_finish = [index for index, _ in engine.poses]
    built, build_result, finish_s, finish_profile = None, None, None, None
    if args.finish:

        def progress(current, total, result):
            print(f"Finish {current}/{total}: {result.get('message', '')}", flush=True)

        (built, build_result), finish_s, finish_profile = run_phase(
            lambda: engine.build_mesh(progress_cb=progress),
            args.cprofile,
            args.output.with_suffix(".finish.pstats"),
        )
    reconstruction = engine.reconstruction_report()
    report = {
        "schema_version": 1,
        "session": args.session.name,
        "input_sha256": input_before,
        "input_changed_during_profile": file_hash(args.session) != input_before,
        "source_sha256": source_before,
        "source_changed_during_profile": source_hash() != source_before,
        "native_extension": {
            "path": str(extension_path) if extension_path else None,
            "sha256": extension_hash,
            "changed_during_profile": bool(
                extension_path and file_hash(extension_path) != extension_hash
            ),
        },
        "selected_indices": indices,
        "seed": args.seed,
        "frames": len(frames),
        "native_mode": os.environ.get("KINECT_NATIVE", "auto"),
        "omp_threads": os.environ["OMP_NUM_THREADS"],
        "thread_policy": {
            "opencv_threads": cv2.getNumThreads(),
            "open3d_threads": o3d.utility.get_max_threads()
            if hasattr(o3d.utility, "get_max_threads") else None,
            **{key: os.environ.get(key) for key in (
                "OMP_WAIT_POLICY", "KMP_BLOCKTIME", "OPENCV_FOR_THREADS_NUM",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        },
        "gpu_hardware": gpu_info(),
        "experimental_visual_fallback": args.visual_fallback_sift,
        "experimental_policy_sha256": policy_hash,
        "experimental_policy_changed": file_hash(policy_path) != policy_hash if policy_hash else False,
        "initial_blocks": os.environ["KINECT_BLOCK_COUNT"],
        "pipeline_options": {key: os.environ.get(key) for key in (
            "KINECT_CUDA_REGISTRATION", "KINECT_CUDA_ODOMETRY", "KINECT_CUDA_MATCHING",
            "KINECT_MODEL_REFRESH", "KINECT_KEYFRAME_CACHE", "KINECT_LIVE_RECOVERY", "KINECT_VISUAL_FEATURES",
            "KINECT_VISUAL_REFINEMENT", "KINECT_FINAL_VISUAL_FIRST", "KINECT_FINAL_LOCAL_REFINEMENT",
            "KINECT_ADAPTIVE_EXPERIMENTAL", "KINECT_CUDA_INPUT", "KINECT_CUDA_CONFIDENCE")},
        "backend": engine.backend,
        "settings": settings.to_dict(),
        "settings_overrides": overrides,
        "original_settings_sha256": original_settings_sha256,
        "versions": {
            "python": sys.version.split()[0],
            "open3d": o3d.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "cprofile_enabled": args.cprofile,
        "live_s": live_s,
        "finish_s": finish_s,
        "processing_s": live_s + (finish_s or 0),
        "accepted_before_finish": accepted_before_finish,
        "accepted_indices_before_finish": accepted_indices_before_finish,
        "accepted": engine.frame_count,
        "accepted_indices": [item["index"] for item in reconstruction["poses"]],
        "finish_requested": args.finish,
        "pose_seeds_used": args.use_pose_seeds,
        "mesh_built": built,
        "build_result": build_result,
        "live_stages": stage_summary(live_diagnostics, live_stages, np),
        "live_diagnostics": live_diagnostics,
        "all_stages": stage_summary(engine.diagnostics, engine.stage_totals_ms, np),
        "tracking_methods": dict(
            Counter(
                item.get("method", "reference")
                for item in engine.diagnostics
                if item["success"]
            )
        ),
        "poses": reconstruction["poses"],
        "diagnostics": engine.diagnostics,
        "fragment_reconnection": engine.fragment_reconnection,
        "refinement": engine.refinement,
        "bundle_adjustment": engine.bundle_adjustment,
        "final_reconstruction": engine.final_reconstruction,
        "peak_process_rss_bytes": peak_rss_bytes(),
        "profiles": {"live": live_profile, "finish": finish_profile},
        "measurement": "Sequential single-process replay. Processing includes storage/live tracking and requested Finish. Excludes imports, image decoding, hashing, geometry comparison and exports. RSS includes loaded images and native library startup. cProfile self time includes opaque native calls; it is not Python overhead. Captured poses have no independent ground truth.",
    }
    report["geometry"] = geometry_summary(
        engine, args.output.with_suffix(".geometry.npz"), np
    )
    if args.compare:
        report["comparison"] = compare_quality(report, args.compare, np)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "session",
                    "frames",
                    "native_mode",
                    "live_s",
                    "finish_s",
                    "accepted",
                    "mesh_built",
                    "geometry",
                    "comparison",
                )
                if key in report
            },
            indent=2,
        )
    )
    if sys.platform == "win32" and args.device != "cpu" and o3d.core.cuda.is_available():
        # The official Windows CUDA wheel may finalize CUDA DLL state before
        # its static Open3D objects, producing "driver shutting down" after a
        # completed run. This isolated CLI has already written/closed artifacts
        # and synchronized all measured work; avoid those broken finalizers.
        o3d.core.cuda.synchronize()
        finish_cuda_worker()
    del engine
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
