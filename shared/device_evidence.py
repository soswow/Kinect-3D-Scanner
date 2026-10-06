"""Measured-target calibration and recording quality; never infer a known size."""

import cv2
import numpy as np

from .calibration import prepare_metric_depth
from .settings import CameraCalibration


def board_points(columns, rows, square_m):
    if columns < 3 or rows < 3 or not 0.001 <= square_m <= 0.2:
        raise ValueError("Require >=3x3 inner corners and a measured 1–200 mm square")
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_m
    return points


def calibrate_corners(corners, columns, rows, square_m, name="Measured Kinect RGB"):
    """Fit intrinsics/lens distortion on training views; score separate views."""
    objects = board_points(columns, rows, square_m)
    corners = [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in corners]
    if len(corners) < 16 or any(len(c) != len(objects) for c in corners):
        raise ValueError("Need at least 16 complete checkerboard views")
    if not all(np.all(np.isfinite(c)) for c in corners):
        raise ValueError("Non-finite checkerboard corners")
    heldout = set(range(3, len(corners), 4))
    train = [c for i, c in enumerate(corners) if i not in heldout]
    rms, matrix, distortion, rotations, _ = cv2.calibrateCamera(
        [objects] * len(train),
        train,
        (640, 480),
        None,
        None,
        flags=cv2.CALIB_FIX_K3,
    )
    camera = CameraCalibration(
        name=name,
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        distortion=tuple(distortion.reshape(-1)),
    )
    errors = []
    for i in sorted(heldout):
        ok, rotation, translation = cv2.solvePnP(
            objects, corners[i], matrix, distortion
        )
        if not ok:
            raise ValueError("Held-out checkerboard pose failed")
        projection, _ = cv2.projectPoints(
            objects, rotation, translation, matrix, distortion
        )
        errors.append(
            float(np.sqrt(np.mean(np.sum((projection - corners[i]) ** 2, axis=2))))
        )
    centers = np.array([c.mean(axis=(0, 1)) for c in corners])
    coverage = float(
        np.prod(np.ptp(np.concatenate(corners).reshape(-1, 2), axis=0)) / (640 * 480)
    )
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rotations])
    tilt_spread = float(np.linalg.norm(np.std(normals, axis=0)))
    reasons = []
    if rms > 0.8 or max(errors) > 1.0:
        reasons.append("Reprojection error too high; use sharp, flat measured targets")
    if coverage < 0.2 or np.ptp(centers[:, 0]) < 80 or np.ptp(centers[:, 1]) < 60:
        reasons.append("Insufficient image coverage; move the target across the frame")
    if tilt_spread < 0.08:
        reasons.append("Insufficient target tilt; include several different angles")
    report = {
        "accepted": not reasons,
        "reasons": reasons,
        "views": len(corners),
        "training_rms_px": float(rms),
        "heldout_rms_px": errors,
        "image_coverage_fraction": coverage,
        "normal_spread": tilt_spread,
        "square_m": square_m,
        "depth_scale_measured": False,
    }
    return camera, report


def plane_evidence(depth, camera, roi, expected_z_m=None):
    """Robust plane residuals; metric z check only for a front-facing known plane."""
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= camera.width and 0 <= y0 < y1 <= camera.height):
        raise ValueError("Plane ROI must lie within the image")
    z = depth[y0:y1, x0:x1].astype(float) / 1000
    y, x = np.mgrid[y0:y1, x0:x1]
    valid = z > 0
    if valid.sum() < 500:
        raise ValueError("Too few valid plane samples")
    points = np.column_stack(
        (
            (x[valid] - camera.cx) * z[valid] / camera.fx,
            (y[valid] - camera.cy) * z[valid] / camera.fy,
            z[valid],
        )
    )
    points = points[:: max(1, len(points) // 20000)]
    keep = np.ones(len(points), bool)
    for _ in range(4):
        center = np.median(points[keep], axis=0)
        _, _, vectors = np.linalg.svd(points[keep] - center, full_matrices=False)
        normal = vectors[-1]
        residual = (points - center) @ normal
        scale = max(0.0005, float(np.median(np.abs(residual[keep])) * 1.4826))
        keep = np.abs(residual) < max(0.003, 3 * scale)
    report = {
        "valid_fraction": float(valid.mean()),
        "inlier_fraction": float(keep.mean()),
        "median_z_m": float(np.median(z[valid])),
        "plane_rmse_m": float(np.sqrt(np.mean(residual[keep] ** 2))),
        "plane_p95_m": float(np.percentile(np.abs(residual[keep]), 95)),
        "normal": normal.tolist(),
    }
    if expected_z_m is not None:
        if not 0.4 <= expected_z_m <= 8 or abs(normal[2]) < 0.98:
            raise ValueError(
                "Scale target needs known 0.4–8 m z and a front-facing plane"
            )
        report.update(
            z_error_m=report["median_z_m"] - expected_z_m,
            suggested_depth_scale=expected_z_m / report["median_z_m"],
        )
    return report


def recording_evidence(frames, metadata, settings, roi=None, expected_z_m=None):
    lags = [
        float(m["rgb_depth_delta_ms"])
        for m in metadata
        if m.get("rgb_depth_delta_ms") is not None
    ]
    planes, valid = [], []
    for rgb, depth in frames:
        metric = prepare_metric_depth(depth, settings)
        valid.append(float(np.count_nonzero(metric) / metric.size))
        if roi is not None:
            planes.append(plane_evidence(metric, settings.camera, roi, expected_z_m))
    return {
        "frames": len(valid),
        "camera": settings.to_dict()["camera"],
        "valid_depth_fraction": valid,
        "pairing_samples": len(lags),
        "pairing_abs_p95_ms": float(np.percentile(np.abs(lags), 95)) if lags else None,
        "pairing_over_20ms": sum(abs(v) > 20 for v in lags),
        "planes": planes,
        "measured_reference": expected_z_m is not None,
        "note": "Plane residuals measure precision, not absolute geometry accuracy",
    }
