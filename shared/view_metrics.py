"""Mesh consistency with withheld measured depth at evaluation-only poses.

Pixel-weighted, fixed first-camera anchor: no pose fitting, hole filling or
reference data in tracking. These are sensor consistency metrics, not laser GT.
"""

from dataclasses import replace

import numpy as np
import open3d as o3d

from shared.calibration import camera_matrix, prepare_metric_depth


def heldout_view_metrics(
    mesh,
    frames,
    settings,
    world_from_reference,
    *,
    pixel_stride=4,
    threshold_m=0.01,
    missing_penalty_m=0.1,
):
    if type(pixel_stride) is not int or not 1 <= pixel_stride <= 16:
        raise ValueError("Pixel stride must be 1–16")
    if not 0 < threshold_m <= missing_penalty_m <= 1:
        raise ValueError("Require 0 < threshold <= missing penalty <= 1 m")
    if not np.all(np.isfinite(world_from_reference)) or np.shape(
        world_from_reference
    ) != (4, 4):
        raise ValueError("Evaluation anchor must be a finite 4x4 transform")
    scene = o3d.t.geometry.RaycastingScene()
    if mesh is not None and len(mesh.triangles):
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    errors, supported, correct, evaluated, per_view = [], 0, 0, 0, []
    c = settings.camera
    # Withheld measurements keep their holes/noise; do not apply reconstruction's filter.
    observed_settings = replace(settings, filter_depth=False)
    for rgb, depth, stamp, reference_pose in frames:
        if reference_pose is None:
            continue
        observed = prepare_metric_depth(depth, observed_settings)
        observed = observed[::pixel_stride, ::pixel_stride].astype(np.float32) / 1000
        rays = scene.create_rays_pinhole(
            camera_matrix(c),
            np.linalg.inv(world_from_reference @ reference_pose),
            c.width,
            c.height,
        )
        rendered = scene.cast_rays(rays[::pixel_stride, ::pixel_stride])[
            "t_hit"
        ].numpy()
        valid = (
            (observed > 0)
            & (observed >= settings.near_m)
            & (observed <= settings.far_m)
        )
        count = int(valid.sum())
        if not count:
            continue
        overlap = valid & np.isfinite(rendered) & (rendered > 0)
        error = np.abs(rendered[overlap] - observed[overlap])
        errors.append(error)
        supported += count
        good = int(np.count_nonzero(error <= threshold_m))
        correct += good
        evaluated += 1
        per_view.append(
            {
                "timestamp_s": float(stamp),
                "reference_pixels": count,
                "overlap_pixels": int(overlap.sum()),
                "correct_pixels": good,
            }
        )
    if not supported:
        raise ValueError("No withheld frames with reference poses and valid depth")
    error = np.concatenate(errors)
    hits = len(error)
    capped_sum = float(np.sum(np.minimum(error, missing_penalty_m).astype(float) ** 2))
    capped_sum += (supported - hits) * missing_penalty_m**2
    return {
        "frames_with_reference_pose": sum(f[3] is not None for f in frames),
        "evaluated_frames": evaluated,
        "reference_pixels": supported,
        "overlap_pixels": hits,
        "hit_fraction": hits / supported,
        "depth_rmse_over_hits_m": float(np.sqrt(np.mean(error**2))) if hits else None,
        "depth_p95_over_hits_m": float(np.percentile(error, 95)) if hits else None,
        "precision_within_threshold": correct / hits if hits else 0,
        "completeness_within_threshold": correct / supported,
        "capped_rmse_including_missing_m": float(np.sqrt(capped_sum / supported)),
        "threshold_m": threshold_m,
        "missing_penalty_m": missing_penalty_m,
        "pixel_stride": pixel_stride,
        "per_view": per_view,
        "interpretation": "Withheld sensor-depth consistency; pixel-weighted observed views, not whole-object accuracy",
        "alignment": "Fixed first accepted reference camera; no scale or trajectory fit",
    }
