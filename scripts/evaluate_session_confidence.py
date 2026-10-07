"""Compare confidence estimators on held-out session depth at fixed saved poses.

The original ZIP is opened read-only. Every fifth archived accepted view is
held out; 2--4-view sessions hold out their last view. All other accepted raw
captures are fused, without tracking, pose fitting, reconnection or sampling.
Triangle distances and fixed-pose depth rays score withheld stable measurements.
These are sensor-consistency measurements, not independent absolute accuracy.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from scanner_server.engine import ScanEngine
from scanner_server.fragments import _rigid
from scripts.benchmark_confidence_fusion import legacy_confidence
from shared.calibration import camera_matrix, prepare_metric_depth, prepare_rgbd
from shared.confidence import depth_confidence
from shared.native import native_status
from shared.settings import ScanSettings


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_file(left, right):
    """Recognize resolved symlinks and existing hardlink aliases."""
    left, right = Path(left), Path(right)
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def require_distinct_paths(source, detailed=None, summary=None, mesh=None):
    """Reject output aliases before any write can damage the read-only input."""
    paths = [("source ZIP", Path(source))]
    for label, value in (("detailed report", detailed), ("summary report", summary),
                         ("mesh output", mesh)):
        if value is None:
            continue
        for previous_label, previous in paths:
            if same_file(value, previous):
                raise ValueError(f"{label.capitalize()} aliases {previous_label}")
        paths.append((label, Path(value)))


def split_accepted(poses):
    """Split chronologically ordered accepted positions, retaining the anchor."""
    if len(poses) < 2:
        raise ValueError("Need at least two accepted views for disjoint evaluation")
    withheld = {i for i in range(len(poses)) if (i + 1) % 5 == 0}
    if not withheld:
        withheld = {len(poses) - 1}
    return (
        [item for position, item in enumerate(poses) if position not in withheld],
        [item for position, item in enumerate(poses) if position in withheld],
    )


class SessionReader:
    def __init__(self, archive):
        self.archive = archive
        self.manifest_bytes = archive.read("manifest.json")
        self.manifest = json.loads(self.manifest_bytes)
        self.settings = ScanSettings.from_dict(self.manifest["settings"])
        self.report_bytes = archive.read(
            self.manifest.get("reconstruction", "reconstruction.json")
        )
        self.report = json.loads(self.report_bytes)
        if self.report.get("pose_convention", "camera_to_world") != "camera_to_world":
            raise ValueError("Only archived camera_to_world poses are supported")
        if self.report.get("length_unit", "metres") != "metres":
            raise ValueError("Archived pose translations must be in metres")
        seen, poses = set(), []
        for item in self.report.get("poses", []):
            index = item["index"]
            pose = np.asarray(item["camera_to_world"], dtype=np.float64)
            if (type(index) is not int or index in seen
                    or not 0 <= index < len(self.manifest["frames"])
                    or not _rigid(pose)):
                raise ValueError("Invalid or duplicate archived accepted pose")
            seen.add(index)
            poses.append((index, pose))
        self.poses = sorted(poses, key=lambda item: item[0])

    def depth(self, index):
        with self.archive.open(self.manifest["frames"][index]["depth"]) as source:
            with Image.open(source) as image:
                depth = np.asarray(image, dtype=np.uint16).copy()
        if depth.shape != (480, 640):
            raise ValueError(f"Frame {index} depth is not the expected 640x480 sensor image")
        return depth

    def rgbd(self, index):
        with self.archive.open(self.manifest["frames"][index]["rgb"]) as source:
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        c = self.settings.rgb_camera
        expected = (c.height, c.width, 3)
        if rgb.shape != expected:
            raise ValueError(f"Frame {index} RGB shape {rgb.shape} differs from {expected}")
        return rgb, self.depth(index)


def fusion_context(settings, budget):
    """Supply only the normal fusion methods' attributes; allocate no live map."""
    return SimpleNamespace(
        settings=settings, device=o3d.core.Device("CPU:0"),
        voxel_size=settings.voxel_m, sdf_trunc=settings.truncation_m,
        max_depth_m=settings.far_m,
        intrinsic_tensor=o3d.core.Tensor(camera_matrix(settings.camera), dtype=o3d.core.float64),
        _fusion_block_limit=budget,
    )


def plan_blocks(reader, training, context, budget):
    """Normal exact frustum planning, with a one-block scratch volume.

    Stop once the hard limit is exceeded. A refusal reports a lower bound and
    processed indices, rather than pretending the incomplete plan is exact.
    """
    scratch = ScanEngine._create_vbg(context, block_count=1)
    blocks, processed = set(), []
    started = time.perf_counter()
    for index, pose in training:
        depth = prepare_metric_depth(reader.depth(index), context.settings)
        coordinates = scratch.compute_unique_block_coordinates(
            o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth))),
            context.intrinsic_tensor,
            o3d.core.Tensor(np.linalg.inv(pose), dtype=o3d.core.float64),
            depth_scale=1000.0, depth_max=context.max_depth_m,
            trunc_voxel_multiplier=context.sdf_trunc / context.voxel_size,
        ).numpy()
        blocks.update(map(tuple, coordinates))
        processed.append(index)
        if len(blocks) > budget:
            break
    exact = len(processed) == len(training)
    return {
        "required_blocks" if exact else "required_at_least_blocks": len(blocks),
        "exact": exact, "processed_indices": processed,
        "block_budget": budget, "within_budget": len(blocks) <= budget,
        "elapsed_s": time.perf_counter() - started,
    }


def stable_depth_samples(depth, camera, near_m, far_m, stride=4, max_samples=6000):
    """Use measured depths with common complete 3x3 support, never smoothing."""
    z = depth.astype(np.float32) / 1000
    valid = np.isfinite(z) & (z > 0) & (z >= near_m) & (z <= far_m)
    kernel = np.ones((3, 3), np.uint8)
    complete = cv2.erode(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                         borderValue=0).astype(bool)
    minimum = cv2.erode(z, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    maximum = cv2.dilate(z, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    stable = complete & ((maximum - minimum) <= np.maximum(0.02, 0.02 * z))
    y, x = np.indices(z.shape)
    selected = np.flatnonzero(stable & (y % stride == 0) & (x % stride == 0))
    if len(selected) > max_samples:
        selected = selected[np.linspace(0, len(selected) - 1, max_samples, dtype=int)]
    y, x = y.reshape(-1)[selected], x.reshape(-1)[selected]
    observed = z[y, x]
    directions = np.stack(
        ((x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, np.ones_like(x)),
        axis=-1,
    ).astype(np.float32)
    return directions, observed, {
        "valid_pixels": int(valid.sum()), "stable_pixels": int(stable.sum()),
        "sampled_pixels": len(selected),
    }


def heldout_samples(reader, withheld, settings, stride, max_samples):
    """Held-out depth/calibration only; no confidence estimator or train access."""
    observed_settings = replace(settings, filter_depth=False)
    samples = []
    for index, pose in withheld:
        depth = prepare_metric_depth(reader.depth(index), observed_settings)
        directions, observed, counts = stable_depth_samples(
            depth, settings.camera, settings.near_m, settings.far_m, stride, max_samples,
        )
        rotation, origin = pose[:3, :3], pose[:3, 3]
        rays_directions = directions @ rotation.T
        points = directions * observed[:, None] @ rotation.T + origin
        rays = np.concatenate((np.broadcast_to(origin, rays_directions.shape), rays_directions), axis=-1)
        samples.append({
            "index": index, "timestamp_s": reader.manifest["frames"][index]["timestamp_s"],
            "counts": counts, "points": points.astype(np.float32),
            "rays": rays.astype(np.float32), "depth_m": observed,
        })
    if not sum(len(item["depth_m"]) for item in samples):
        raise ValueError("Held-out views contain no stable measured depth samples")
    return samples


def score_mesh(mesh, samples, threshold_m=0.01, cap_m=0.1):
    """Measured points to actual triangles plus visibility-aware depth rays."""
    if not 0 < threshold_m <= cap_m <= 1:
        raise ValueError("Require 0 < threshold <= missing/error cap <= 1 metre")
    scene = o3d.t.geometry.RaycastingScene()
    has_surface = mesh is not None and len(mesh.triangles) > 0
    if has_surface:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances, ray_errors, per_view = [], [], []
    total, hits, ray_good, surface_good = 0, 0, 0, 0
    for item in samples:
        count = len(item["depth_m"])
        if has_surface and count:
            distance = scene.compute_distance(o3d.core.Tensor(item["points"])).numpy()
            rendered = scene.cast_rays(o3d.core.Tensor(item["rays"]))["t_hit"].numpy()
        else:
            distance, rendered = np.full(count, np.inf), np.full(count, np.inf)
        hit = np.isfinite(rendered) & (rendered > 0)
        error = np.abs(rendered[hit] - item["depth_m"][hit])
        good = int(np.count_nonzero(error <= threshold_m))
        close = int(np.count_nonzero(distance <= threshold_m))
        total += count
        hits += int(hit.sum())
        ray_good += good
        surface_good += close
        distances.append(distance)
        ray_errors.append(error)
        per_view.append({
            "index": item["index"], "timestamp_s": item["timestamp_s"],
            **item["counts"], "ray_hits": int(hit.sum()), "ray_correct": good,
            "surface_close": close,
        })
    if not total:
        raise ValueError("No held-out measured points to score")
    distance = np.concatenate(distances).astype(np.float64)
    error = np.concatenate(ray_errors).astype(np.float64)
    finite = distance[np.isfinite(distance)]
    ray_loss = float(np.minimum(error, cap_m).dot(np.minimum(error, cap_m)))
    ray_loss += (total - hits) * cap_m**2
    return {
        "samples": total, "surface_present": bool(has_surface),
        "surface_distance_rmse_m": float(np.sqrt(np.mean(finite**2))) if len(finite) else None,
        "surface_distance_p95_m": float(np.percentile(finite, 95)) if len(finite) else None,
        "surface_completeness_within_threshold": surface_good / total,
        "surface_capped_rmse_including_missing_m": float(np.sqrt(np.mean(np.minimum(distance, cap_m)**2))),
        "ray_hit_fraction": hits / total,
        "ray_depth_rmse_over_hits_m": float(np.sqrt(np.mean(error**2))) if hits else None,
        "ray_depth_p95_over_hits_m": float(np.percentile(error, 95)) if hits else None,
        "ray_precision_over_hits": ray_good / hits if hits else 0,
        "ray_completeness_within_threshold": ray_good / total,
        "ray_capped_rmse_including_missing_m": float(np.sqrt(ray_loss / total)),
        "threshold_m": threshold_m, "missing_and_error_cap_m": cap_m,
        "per_view": per_view,
        "interpretation": "Held-out depth consistency at fixed archived estimated poses; observed surfaces only, no independent absolute-accuracy claim.",
    }


def fuse_variant(reader, training, context, required_blocks, estimator):
    volume = ScanEngine._create_vbg(context, block_count=max(1, required_blocks))
    digest = hashlib.sha256()
    started = time.perf_counter()
    with patch("scanner_server.weighted_fusion.depth_confidence", estimator):
        for index, pose in training:
            rgb, depth = prepare_rgbd(*reader.rgbd(index), context.settings)
            digest.update(str(index).encode())
            digest.update(depth.tobytes())
            ScanEngine._integrate_vbg(context, rgb, depth, np.linalg.inv(pose), volume=volume)
    fused_s = time.perf_counter() - started
    started = time.perf_counter()
    mesh = volume.extract_triangle_mesh(weight_threshold=context.settings.final_weight).to_legacy()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    if len(mesh.triangles) and context.settings.min_component_triangles:
        labels, sizes, _ = mesh.cluster_connected_triangles()
        mesh.remove_triangles_by_mask(
            np.asarray(sizes)[np.asarray(labels)] < context.settings.min_component_triangles
        )
        mesh.remove_unreferenced_vertices()
    return mesh, {
        "training_indices": [index for index, _ in training],
        "prepared_training_depth_sha256": digest.hexdigest(),
        "blocks": int(volume.hashmap().size()), "triangles": len(mesh.triangles),
        "surface_area_m2": float(mesh.get_surface_area()) if len(mesh.triangles) else 0,
        "preparation_and_fusion_s": fused_s,
        "extraction_and_component_filter_s": time.perf_counter() - started,
    }


def evaluate(path, *, voxel_m=None, final_weight=None, block_budget=None,
             pixel_stride=4, samples_per_view=6000, threshold_m=0.01,
             cap_m=0.1, mesh_dir=None):
    path = Path(path)
    source_digest = file_hash(path)
    report = {
        "schema_version": 1, "source": path.name, "input_zip_sha256": source_digest,
        "status": "refused", "variants": {},
        "pose_policy": "Archived accepted camera_to_world estimates fixed identically; no fitting, tracking, reconnection or Finish refinement.",
        "split_policy": "Chronological accepted positions: each fifth held out; for 2--4 views hold out last; retain all others.",
        "limitations": [
            "Archived poses may originally have used all recording observations; disjoint fusion does not make those poses independent truth.",
            "Held-out depth shares the sensor's noise and calibration; scores cover observed stable surfaces, not whole-object absolute accuracy.",
        ],
        "units": {"prepared_depth": "integer millimetres", "geometry_and_poses": "metres"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "open3d": o3d.__version__, "opencv": cv2.__version__},
        "native": native_status(), "platform": platform.platform(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "source_sha256": {
            name: file_hash(ROOT / name) for name in (
                "scripts/evaluate_session_confidence.py", "scripts/benchmark_confidence_fusion.py",
                "shared/confidence.py", "shared/calibration.py", "shared/settings.py",
                "scanner_server/weighted_fusion.py", "scanner_server/engine.py",
                "native/weighted_fusion.h",
            )
        },
        "validation": {
            "pixel_stride": pixel_stride, "max_samples_per_heldout_view": samples_per_view,
            "stable_support": "Complete observed 3x3 patch, depth range <= max(20 mm, 2% of depth); no smoothing or confidence selection.",
            "threshold_m": threshold_m, "missing_and_error_cap_m": cap_m,
        },
    }
    try:
        if type(pixel_stride) is not int or not 1 <= pixel_stride <= 16:
            raise ValueError("Pixel stride must be 1--16")
        if type(samples_per_view) is not int or not 1 <= samples_per_view <= 100000:
            raise ValueError("Samples per view must be 1--100000")
        if not 0 < threshold_m <= cap_m <= 1:
            raise ValueError("Require 0 < threshold <= missing/error cap <=1 metre")
        if mesh_dir is not None:
            resolved = Path(mesh_dir).resolve()
            if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / "benchmark-output"):
                raise ValueError("Mesh output must be outside repository or within ignored benchmark-output")
            for name in ("legacy", "local_planes"):
                require_distinct_paths(path, mesh=Path(mesh_dir) / f"{path.stem}-{name}.ply")
        with zipfile.ZipFile(path, "r") as archive:
            reader = SessionReader(archive)
            report.update(
                manifest_sha256=hashlib.sha256(reader.manifest_bytes).hexdigest(),
                reconstruction_sha256=hashlib.sha256(reader.report_bytes).hexdigest(),
                archived_settings=reader.settings.to_dict(),
                archived_session_id=reader.report.get("session_id"),
                raw_frame_count=len(reader.manifest["frames"]),
                archived_accepted_indices=[index for index, _ in reader.poses],
                archived_accepted_pose_sha256=hashlib.sha256(json.dumps(
                    [[index, pose.tolist()] for index, pose in reader.poses], sort_keys=True
                ).encode()).hexdigest(),
            )
            report["units"]["archived_raw_depth"] = reader.manifest.get("depth_unit", "unspecified")
            report["units"]["archived_depth_encoding"] = reader.settings.depth_encoding
            training, withheld = split_accepted(reader.poses)
            settings = reader.settings
            resolution = voxel_m if voxel_m is not None else settings.final_voxel_m or settings.voxel_m
            budget = block_budget if block_budget is not None else settings.final_block_count
            settings = replace(settings, voxel_m=resolution, final_voxel_m=None,
                               confidence_fusion=True, final_block_count=budget,
                               final_weight=final_weight if final_weight is not None else settings.final_weight)
            report.update(fusion_settings=settings.to_dict(), training_indices=[i for i, _ in training],
                          heldout_indices=[i for i, _ in withheld], block_budget=budget,
                          attribute_budget_mib=budget * 4096 * 20 / 2**20)
            context = fusion_context(settings, budget)
            report["planning"] = plan_blocks(reader, training, context, budget)
            if not report["planning"]["within_budget"]:
                raise ValueError("Training fusion exceeds hard block budget; no candidate map allocated")
            required = report["planning"]["required_blocks"]
            samples = heldout_samples(reader, withheld, settings, pixel_stride, samples_per_view)
            for name, estimator in (("legacy", legacy_confidence), ("local_planes", depth_confidence)):
                mesh, result = fuse_variant(reader, training, context, required, estimator)
                result["heldout"] = score_mesh(mesh, samples, threshold_m, cap_m)
                if mesh_dir is not None and len(mesh.triangles):
                    directory = Path(mesh_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    destination = directory / f"{path.stem}-{name}.ply"
                    require_distinct_paths(path, mesh=destination)
                    if not o3d.io.write_triangle_mesh(str(destination), mesh):
                        raise OSError(f"Cannot export mesh to {destination}")
                    result["mesh_path"] = str(destination.resolve())
                report["variants"][name] = result
            if (report["variants"]["legacy"]["prepared_training_depth_sha256"]
                    != report["variants"]["local_planes"]["prepared_training_depth_sha256"]):
                raise ValueError("Estimator runs did not receive identical prepared training depths")
            report["status"] = "evaluated"
    except Exception as error:
        report["reason"] = str(error)
    report["input_zip_unchanged"] = file_hash(path) == source_digest
    if not report["input_zip_unchanged"]:
        report.update(status="refused", reason="Source ZIP changed during evaluation")
    return report


def compact_report(report):
    """Keep source identity, scalar conditions and outcomes without profiles."""
    excluded = {"archived_settings", "fusion_settings", "variants"}
    compact = {key: value for key, value in report.items() if key not in excluded}
    for label in ("archived_settings", "fusion_settings"):
        settings = report.get(label)
        if settings is None:
            continue
        compact[f"{label}_sha256"] = hashlib.sha256(
            json.dumps(settings, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        compact[label] = {
            key: settings[key] for key in (
                "voxel_m", "final_voxel_m", "truncation_m", "final_weight",
                "final_block_count", "min_component_triangles", "near_m", "far_m",
                "filter_depth", "roi", "rgb_mode",
            )
        }
        compact[label]["camera"] = settings["camera"]
        if settings.get("sensor_calibration") is not None:
            compact[label]["sensor_calibration_sha256"] = hashlib.sha256(
                json.dumps(settings["sensor_calibration"], sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
    compact["variants"] = {}
    for name, value in report["variants"].items():
        compact["variants"][name] = {
            **{key: item for key, item in value.items() if key != "heldout"},
            "heldout": {key: item for key, item in value["heldout"].items() if key != "per_view"},
        }
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path,
                        help="Optional compact JSON suitable for docs/benchmarks; full profiles remain in --output")
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--voxel-mm", type=float)
    parser.add_argument("--final-weight", type=float)
    parser.add_argument("--block-budget", type=int)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--samples-per-view", type=int, default=6000)
    parser.add_argument("--threshold-mm", type=float, default=10)
    parser.add_argument("--cap-mm", type=float, default=100)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("Output must be a JSON report")
    resolved = args.output.resolve()
    if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / "benchmark-output"):
        parser.error("Detailed output must be outside repository or within ignored benchmark-output")
    if args.summary_output is not None and (
        args.summary_output.suffix.lower() != ".json"
    ):
        parser.error("Summary output must be a separate JSON report")
    try:
        require_distinct_paths(args.session, args.output, args.summary_output)
    except ValueError as error:
        parser.error(str(error))
    report = evaluate(
        args.session, voxel_m=args.voxel_mm / 1000 if args.voxel_mm is not None else None,
        final_weight=args.final_weight, block_budget=args.block_budget,
        pixel_stride=args.pixel_stride, samples_per_view=args.samples_per_view,
        threshold_m=args.threshold_mm / 1000, cap_m=args.cap_mm / 1000, mesh_dir=args.mesh_dir,
    )
    require_distinct_paths(args.session, args.output, args.summary_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(compact_report(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"source": report["source"], "status": report["status"],
                      "reason": report.get("reason"), "training": len(report.get("training_indices", [])),
                      "heldout": len(report.get("heldout_indices", [])),
                      "variants": {name: value["heldout"] for name, value in report["variants"].items()}},
                     indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
