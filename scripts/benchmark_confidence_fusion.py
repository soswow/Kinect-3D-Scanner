"""Controlled depth-confidence and repeated TSDF surface measurements.

Uses known planes/details, independent synthetic depth noise, and fixed exact
poses. This isolates sensor fusion; it is not a real-device or tracking result.
The frozen legacy estimator is retained here solely as a reproducible baseline.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from shared.confidence import depth_confidence


def legacy_confidence(depth, camera):
    """Estimator before the inverse-depth patch fit, frozen for comparison."""
    z = depth.astype(np.float32) / 1000
    y, x = np.indices(z.shape, dtype=np.float32)
    points = np.stack(
        ((x - camera.cx) * z / camera.fx, (y - camera.cy) * z / camera.fy, z), axis=-1
    )
    padded = np.pad(z, 1)
    neighbours = np.stack(
        (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:])
    )
    supported = np.all(neighbours > 0, axis=0)
    edge = np.any(
        (neighbours > 0) & (np.abs(neighbours - z) > np.maximum(0.02, 0.02 * z)), axis=0
    )
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    normal = np.cross(dx, dy)
    lengths = np.linalg.norm(normal, axis=-1)
    ray = points / np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), 1e-6)
    cosine = np.abs((normal * ray).sum(axis=-1)) / np.maximum(lengths, 1e-10)
    angle = np.where(supported & (lengths > 1e-10), cosine**2, 0.25)
    sigma = 0.001 + 0.002 * z**2
    weight = np.clip((0.003 / sigma) ** 2, 0.05, 1) * angle
    weight[(z <= 0) | edge | (supported & (cosine < 0.25))] = 0
    return np.ascontiguousarray(weight, np.float32)


def camera_for(width=160, height=120):
    return SimpleNamespace(
        width=width, height=height, fx=525.0, fy=525.0,
        cx=(width - 1) / 2, cy=(height - 1) / 2,
    )


def plane_depth(camera, distance=1.0, angle_degrees=0.0):
    _, x = np.indices((camera.height, camera.width), dtype=np.float64)
    rx = (x - camera.cx) / camera.fx
    return distance / (1 + np.tan(np.deg2rad(angle_degrees)) * rx)


def observations(truth, count, seed=17, sigma_m=None):
    rng = np.random.default_rng(seed)
    sigma = 0.001 + 0.002 * truth**2 if sigma_m is None else sigma_m
    for _ in range(count):
        measured = np.rint((truth + rng.normal(size=truth.shape) * sigma) * 1000)
        measured[truth == 0] = 0
        yield np.clip(measured, 0, 65535).astype(np.uint16)


def depth_average_report(truth, camera, count, estimator, sigma_m=None):
    """Noise reduction of the near-surface weighted mean, without voxelization."""
    total = np.zeros_like(truth)
    weighted = np.zeros_like(truth)
    for depth in observations(truth, count, sigma_m=sigma_m):
        weight = estimator(depth, camera)
        total += weight
        weighted += weight * depth / 1000
    reconstructed = np.divide(weighted, total, out=np.zeros_like(truth), where=total > 0)
    observed = truth > 0
    confident = observed & (total >= 2.0)
    error = reconstructed - truth
    return {
        "views": count,
        "mean_weight": float(total[observed].mean() / count),
        "rmse_m": float(np.sqrt(np.mean(error[confident] ** 2)))
        if np.any(confident) else None,
        "completeness_at_weight_2": float(confident[observed].mean()),
        # Missing pixels count as a 10 mm error, exposing coverage/error tradeoffs.
        "missing_aware_capped_rmse_m": float(np.sqrt(np.mean(
            np.where(confident[observed], np.minimum(error[observed] ** 2, 0.01**2), 0.01**2)
        ))),
    }


def fuse_surface(truth, camera, count, estimator, detail_period_pixels=None):
    """Exercise the actual CPU projective TSDF and final weight threshold."""
    import open3d as o3d

    from scanner_server.weighted_fusion import integrate_weighted
    from shared.native import native_status

    core = o3d.core
    engine = SimpleNamespace(
        settings=SimpleNamespace(camera=camera), device=core.Device("CPU:0"),
        max_depth_m=4.0, sdf_trunc=0.04,
    )
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(core.float32,) * 3,
        attr_channels=((1,), (1,), (3,)), voxel_size=0.005,
        block_resolution=8, block_count=2000, device=engine.device,
    )
    intrinsic = core.Tensor(
        [[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]],
        dtype=core.float64,
    )
    identity = core.Tensor(np.eye(4), dtype=core.float64)
    rgb = np.full((*truth.shape, 3), 120, np.uint8)
    elapsed = []
    with patch("scanner_server.weighted_fusion.depth_confidence", estimator):
        for index, depth in enumerate(observations(truth, count), 1):
            started = time.perf_counter()
            blocks = volume.compute_unique_block_coordinates(
                o3d.t.geometry.Image(core.Tensor(depth)), intrinsic, identity,
                depth_scale=1000.0, depth_max=4.0, trunc_voxel_multiplier=8.0,
            )
            integrate_weighted(engine, volume, blocks, rgb, depth, np.eye(4))
            elapsed.append((time.perf_counter() - started) * 1000)
    points = volume.extract_point_cloud(weight_threshold=2.0).point.positions.numpy()
    reference = volume.extract_point_cloud(weight_threshold=0.01).point.positions.numpy()
    # Sample visible measured pixels at 5 mm surface spacing. Their geometry is
    # known analytically; nearest-neighbour distance jointly measures error and
    # holes. Crop image boundaries, where the finite observed field ends.
    y, x = np.indices(truth.shape)
    keep = (truth > 0) & (x >= 6) & (x < camera.width - 6)
    keep &= (y >= 6) & (y < camera.height - 6) & (x % 3 == 0) & (y % 3 == 0)
    target = np.stack(
        ((x - camera.cx) * truth / camera.fx, (y - camera.cy) * truth / camera.fy, truth),
        axis=-1,
    )[keep]
    def cloud(array):
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(array))
    if len(points):
        coverage = np.asarray(cloud(target).compute_point_cloud_distance(cloud(points)))
        # Report analytic axial residual at projected measured coordinates too;
        # unlike nearest neighbours, this is insensitive to reference sampling.
        u = points[:, 0] * camera.fx / points[:, 2] + camera.cx
        v = points[:, 1] * camera.fy / points[:, 2] + camera.cy
        comparable = (u >= 6) & (u < camera.width - 6) & (v >= 6) & (v < camera.height - 6)
        error = np.asarray(cloud(points[comparable]).compute_point_cloud_distance(cloud(target)))
        if detail_period_pixels is not None:
            phase = 2 * np.pi * u[comparable] / detail_period_pixels
            design = np.stack((np.sin(phase), np.cos(phase), np.ones_like(phase)), axis=-1)
            coefficients = np.linalg.lstsq(design, points[comparable, 2], rcond=None)[0]
            amplitude = float(np.linalg.norm(coefficients[:2]))
        u = np.clip(np.rint(u).astype(int), 0, camera.width - 1)
        v = np.clip(np.rint(v).astype(int), 0, camera.height - 1)
        known = truth[v, u] > 0
        axial = points[known, 2] - truth[v[known], u[known]]
    else:
        coverage = np.full(len(target), np.inf)
        error = np.empty(0)
        axial = np.empty(0)
        known = np.empty(0, bool)
    result = {
        "views": count,
        "surface_points": len(points),
        "low_threshold_points": len(reference),
        "axial_rmse_m": float(np.sqrt(np.mean(axial**2))) if len(axial) else None,
        "precision_within_0_01m": float(np.mean(error < 0.01)) if len(error) else 0,
        "completeness_within_0_01m": float(np.mean(coverage < 0.01)),
        "missing_aware_capped_rmse_m": float(np.sqrt(np.mean(np.minimum(coverage, 0.01) ** 2))),
        "points_projecting_into_holes": int(np.count_nonzero(~known)),
        "fusion_median_ms": float(np.median(elapsed)),
        "fusion_first_ms": elapsed[0],
        "fusion_warm_median_ms": float(np.median(elapsed[1:])),
        "native": native_status(),
    }
    if detail_period_pixels is not None:
        result["detail_amplitude_m"] = amplitude if len(points) else None
    return result


def fusion_cost_report(camera, depth, repeats):
    """Alternate estimators on the same preallocated volume and exact frame.

    Includes block discovery, confidence, voxel coordinates/transforms, and
    native/NumPy updates. Excludes attribute reset, allocation, extraction,
    camera preparation, tracking and network work.
    """
    import open3d as o3d

    from scanner_server.weighted_fusion import integrate_weighted

    core = o3d.core
    engine = SimpleNamespace(
        settings=SimpleNamespace(camera=camera), device=core.Device("CPU:0"),
        max_depth_m=4.0, sdf_trunc=0.04,
    )
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(core.float32,) * 3,
        attr_channels=((1,), (1,), (3,)), voxel_size=0.005,
        block_resolution=8, block_count=8000, device=engine.device,
    )
    intrinsic = core.Tensor(
        [[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]],
        dtype=core.float64,
    )
    image = o3d.t.geometry.Image(core.Tensor(depth))
    identity = core.Tensor(np.eye(4), dtype=core.float64)
    rgb = np.full((*depth.shape, 3), 120, np.uint8)
    def integrate():
        blocks = volume.compute_unique_block_coordinates(
            image, intrinsic, identity, depth_scale=1000.0,
            depth_max=4.0, trunc_voxel_multiplier=8.0,
        )
        integrate_weighted(engine, volume, blocks, rgb, depth, np.eye(4))
    integrate()
    attributes = [volume.attribute(name).numpy() for name in ("tsdf", "weight", "color")]
    timings = {"legacy": [], "local_planes": []}
    for index in range(repeats + 2):
        methods = [("legacy", legacy_confidence), ("local_planes", depth_confidence)]
        for name, estimator in methods if index % 2 == 0 else methods[::-1]:
            for values in attributes:
                values.fill(0)
            with patch("scanner_server.weighted_fusion.depth_confidence", estimator):
                started = time.perf_counter()
                integrate()
                elapsed = (time.perf_counter() - started) * 1000
            if index >= 2:
                timings[name].append(elapsed)
    return {
        "image_size": [camera.width, camera.height],
        "scope": "Warm allocated volume: block discovery + confidence + voxel transforms/updates; excludes preparation, tracking, extraction, network.",
        "median_ms": {name: float(np.median(values)) for name, values in timings.items()},
    }


def benchmark(repeats=12, include_tsdf=True):
    report = {
        "scope": "Known synthetic geometry, fixed exact poses, independent per-pixel Gaussian noise; engineering noise profile, not device-calibrated uncertainty.",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "omp_threads": os.environ["OMP_NUM_THREADS"],
        "confidence_ms": {}, "depth_mean": {}, "tsdf": {},
        "source_sha256": {
            filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            for filename in ("shared/confidence.py", "scanner_server/weighted_fusion.py", "scripts/benchmark_confidence_fusion.py")
        },
    }
    large = camera_for(640, 480)
    sample = next(observations(plane_depth(large), 1))
    timings = {"legacy": [], "local_planes": []}
    for _ in range(2):
        legacy_confidence(sample, large)
        depth_confidence(sample, large)
    for i in range(repeats):
        methods = [("legacy", legacy_confidence), ("local_planes", depth_confidence)]
        for name, estimator in methods if i % 2 == 0 else methods[::-1]:
            started = time.perf_counter()
            estimator(sample, large)
            timings[name].append((time.perf_counter() - started) * 1000)
    report["confidence_ms"] = {name: float(np.median(values)) for name, values in timings.items()}
    if include_tsdf:
        report["fusion_cost"] = fusion_cost_report(large, sample, repeats)
    camera = camera_for()
    for angle in (0, 55):
        truth = plane_depth(camera, angle_degrees=angle)
        case = f"plane_{angle}deg"
        report["depth_mean"][case] = {}
        for name, estimator in (("legacy", legacy_confidence), ("local_planes", depth_confidence)):
            report["depth_mean"][case][name] = [
                depth_average_report(truth, camera, count, estimator) for count in (4, 12, 32)
            ]
            if include_tsdf:
                report["tsdf"].setdefault(case, {})[name] = [
                    fuse_surface(truth, camera, count, estimator)
                    for count in ((12, 32) if angle else (4, 12))
                ]
    truth = plane_depth(camera)
    truth[40:60, 50:70] = 0
    truth[:, 90:] = 1.06
    if include_tsdf:
        for name, estimator in (("legacy", legacy_confidence), ("local_planes", depth_confidence)):
            report["tsdf"].setdefault("step_and_hole", {})[name] = fuse_surface(truth, camera, 12, estimator)
        _, x = np.indices(truth.shape)
        detail = 1 + 0.01 * np.sin(2 * np.pi * x / 24)
        for name, estimator in (("legacy", legacy_confidence), ("local_planes", depth_confidence)):
            report["tsdf"].setdefault("10mm_ridges", {})[name] = fuse_surface(detail, camera, 16, estimator, detail_period_pixels=24)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--skip-tsdf", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.repeats, not args.skip_tsdf)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
