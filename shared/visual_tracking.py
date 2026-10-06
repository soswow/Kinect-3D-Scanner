"""Bounded camera-side RGB-D motion, independent of fusion and upload cadence.

Coordinates are camera-to-local, not world poses. A broken chain starts a new
segment; the server may use it only as an initializer and verifies raw evidence.
"""

import time
import uuid

import cv2
import numpy as np

from .calibration import camera_matrix, prepare_rgbd
from .capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS


def feature_agreement(source_points, target_points, target_pixels, pose, camera):
    """Keep measured color correspondences as a constraint after geometric ICP."""
    moved = source_points @ pose[:3, :3].T + pose[:3, 3]
    z = np.maximum(moved[:, 2], 1e-6)
    projected = moved[:, :2] / z[:, None] * [camera.fx, camera.fy]
    projected += [camera.cx, camera.cy]
    pixels = np.linalg.norm(projected - target_pixels, axis=1)
    distances = np.linalg.norm(moved - target_points, axis=1)
    good = (moved[:, 2] > 0) & (pixels <= 3) & (distances <= 0.03)
    count = int(good.sum())
    report = {"inliers": count, "matches": len(good),
              "support_fraction": float(good.mean()) if len(good) else 0.0,
              "median_pixel_error": float(np.median(pixels)) if len(good) else None,
              "median_depth_error_m": float(np.median(distances)) if len(good) else None}
    # Planar textured geometry can constrain motion; collinear/tiny patches cannot.
    covered = False
    if count >= 35:
        extent = np.ptp(target_pixels[good], axis=0) / [camera.width, camera.height]
        eigenvalues = np.linalg.eigvalsh(np.cov(source_points[good].T))
        covered = bool(np.prod(extent) >= 0.04 and
                       eigenvalues[1] / max(eigenvalues.sum(), 1e-9) > 0.002)
    report["distributed"] = covered
    return bool(count >= 35 and report["support_fraction"] >= 0.65 and covered), report


def sampled_points(depth, pixels, camera):
    """Require measured center and a stable 3×3 patch, without filling holes."""
    pixels = np.asarray(pixels).reshape(-1, 2)
    x, y = np.rint(pixels).astype(int).T
    inside = (x >= 1) & (y >= 1) & (x < depth.shape[1] - 1) & (y < depth.shape[0] - 1)
    x = np.clip(x, 1, depth.shape[1] - 2)
    y = np.clip(y, 1, depth.shape[0] - 2)
    patch = np.array([depth[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], float)
    present = patch > 0
    # Zero entries are ignored in the median, but centers remain mandatory.
    ordered = np.sort(np.where(present, patch, np.inf), axis=0)
    n = present.sum(axis=0)
    z = ordered[np.maximum(0, (n - 1) // 2), np.arange(len(x))] / 1000
    z[~np.isfinite(z)] = 0
    spread = np.max(patch, axis=0) - np.min(np.where(present, patch, np.inf), axis=0)
    valid = inside & (depth[y, x] > 0) & (n >= 7) & (spread <= np.maximum(30, 40 * z))
    points = np.column_stack(((pixels[:, 0] - camera.cx) * z / camera.fx,
                              (pixels[:, 1] - camera.cy) * z / camera.fy, z))
    return points, valid


def measured_rigid_motion(source, target, seed):
    """Fit paired measured 3D points, rejecting depth/flow outliers.

    PnP alone can trade translation for rotation on a small textured plane.
    Metric depth at both ends removes that ambiguity; no nearest-neighbour
    association is allowed to replace the tracked feature identities.
    """
    pose = seed.copy()
    for _ in range(4):
        residual = np.linalg.norm(source @ pose[:3, :3].T + pose[:3, 3] - target, axis=1)
        selected = residual < 0.025
        if selected.sum() < 35:
            return None
        a, b = source[selected], target[selected]
        center_a, center_b = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - center_a).T @ (b - center_b))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        pose[:3, :3] = rotation
        pose[:3, 3] = center_b - rotation @ center_a
    return pose


def refine_measured_motion(source, target, pixels, pose, camera):
    """Joint pixel/depth refinement, with fixed measured feature identities."""
    for _ in range(6):
        moved = source @ pose[:3, :3].T + pose[:3, 3]
        x, y, z = moved.T
        if np.any(z <= 0):
            return None
        projected = moved[:, :2] / z[:, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        supported = (np.linalg.norm(projected - pixels, axis=1) < 3) & (
            np.linalg.norm(moved - target, axis=1) < 0.025)
        if supported.sum() < 35:
            return None
        # Small left SE(3) update: d(Rp+t)/d(angle,translation).
        jacobian = np.zeros((len(source), 3, 6))
        jacobian[:, :, :3] = np.stack((
            np.column_stack((z * 0, z, -y)),
            np.column_stack((-z, z * 0, x)),
            np.column_stack((y, -x, z * 0))), axis=1)
        jacobian[:, :, 3:] = np.eye(3)
        projection = np.zeros((len(source), 2, 3))
        projection[:, 0, 0], projection[:, 0, 2] = camera.fx / z, -camera.fx * x / z**2
        projection[:, 1, 1], projection[:, 1, 2] = camera.fy / z, -camera.fy * y / z**2
        # Engineering residual scales (1 pixel, 3 mm); not device uncertainty.
        residual = np.column_stack((projected - pixels, (z - target[:, 2]) / 0.003))
        j = np.concatenate((projection @ jacobian, jacobian[:, 2:3, :] / 0.003), axis=1)
        r, j = residual[supported].reshape(-1), j[supported].reshape(-1, 6)
        weight = np.sqrt(np.minimum(1, 1.5 / np.maximum(np.abs(r), 1e-9)))
        step = np.linalg.lstsq(j * weight[:, None], -r * weight, rcond=None)[0]
        correction = np.eye(4)
        correction[:3, :3] = cv2.Rodrigues(step[:3])[0]
        correction[:3, 3] = step[3:]
        pose = correction @ pose
        if np.linalg.norm(step) < 1e-6:
            break
    return pose


class VisualTracker:
    MAX_GAP_S = 0.75

    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.segment = uuid.uuid4().hex[:16]
        self.pose = np.eye(4)
        self.previous = None
        self.steps = 0

    def update(self, rgb, raw_depth, metadata):
        started = time.monotonic()
        stamp = metadata.get("timestamp_s", started)
        lag = metadata.get("rgb_depth_delta_ms")
        if lag is not None and abs(lag) > RGB_DEPTH_ASSISTANCE_LIMIT_MS:
            self.reset()
            return {"valid": False, "reason": "RGB/depth timing exceeds 20 ms"}
        rgb, depth = prepare_rgbd(rgb, raw_depth, self.settings)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 500, 0.015, 9, mask=(depth > 0).astype(np.uint8))
        valid = corners is not None and len(corners) >= 60
        stats = {}
        if self.previous is not None:
            old_gray, old_depth, old_corners, old_stamp = self.previous
            gap = stamp - old_stamp
            stats["gap_s"] = gap
            valid = valid and 0 < gap <= self.MAX_GAP_S
            if valid:
                new, forward, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, gray, old_corners, None, winSize=(21, 21), maxLevel=3)
                back, reverse, _ = cv2.calcOpticalFlowPyrLK(
                    gray, old_gray, new, None, winSize=(21, 21), maxLevel=3)
                a, b = old_corners.reshape(-1, 2), new.reshape(-1, 2)
                supported = (forward.ravel() > 0) & (reverse.ravel() > 0)
                supported &= np.linalg.norm(a - back.reshape(-1, 2), axis=1) < 0.8
                points_a, measured_a = sampled_points(old_depth, a, self.settings.camera)
                points_b, measured_b = sampled_points(depth, b, self.settings.camera)
                supported &= measured_a & measured_b
                pa, pb, pixels = points_a[supported], points_b[supported], b[supported]
                valid = len(pa) >= 40
                if valid:
                    ok, rotation, translation, inliers = cv2.solvePnPRansac(
                        pa, pixels, camera_matrix(self.settings.camera), None,
                        iterationsCount=80, reprojectionError=2.0, confidence=0.999,
                        flags=cv2.SOLVEPNP_EPNP)
                    valid = bool(ok and inliers is not None and len(inliers) >= 35)
                    if valid:
                        delta = np.eye(4)
                        delta[:3, :3] = cv2.Rodrigues(rotation)[0]
                        delta[:3, 3] = translation.ravel()
                        delta = measured_rigid_motion(pa, pb, delta)
                        if delta is not None:
                            delta = refine_measured_motion(pa, pb, pixels, delta, self.settings.camera)
                        valid = delta is not None
                        if valid:
                            valid, stats = feature_agreement(pa, pb, pixels, delta, self.settings.camera)
                            angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
                            valid = valid and np.linalg.norm(delta[:3, 3]) <= 0.15 and angle <= 15
                        if valid:
                            self.pose = self.pose @ np.linalg.inv(delta)
                            self.steps += 1
            if not valid:
                self.reset()
        self.previous = (gray, depth, corners, stamp) if corners is not None and len(corners) >= 60 else None
        return {"valid": bool(valid), "segment": self.segment, "camera_to_local": self.pose.tolist(),
                "steps": self.steps, "elapsed_ms": (time.monotonic() - started) * 1000,
                "reason": "Measured RGB-D motion" if valid else "Visual chain lost; new local origin",
                **stats}
