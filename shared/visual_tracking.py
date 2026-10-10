"""Bounded camera-side RGB-D motion, independent of fusion and upload cadence.

Coordinates are camera-to-local, not world poses. A broken chain starts a new
segment; the server may use it only as an initializer and verifies raw evidence.
"""

import time
import uuid
from collections import deque
import heapq
from typing import NamedTuple

import cv2
import numpy as np

from .calibration import camera_matrix, prepare_rgbd
from .capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS


def paired_depth_scale(source, target):
    """Range-dependent engineering axial scale, not per-device calibration."""
    a, b = (.002 + .002*points[:, 2]**2 for points in (source, target))
    return np.sqrt(a*a+b*b)


def _feature_support(source_points, target_points, target_pixels, pose, camera):
    moved = source_points @ pose[:3, :3].T + pose[:3, 3]
    z = np.maximum(moved[:, 2], 1e-6)
    projected = moved[:, :2] / z[:, None] * [camera.fx, camera.fy]
    projected += [camera.cx, camera.cy]
    pixels = np.linalg.norm(projected - target_pixels, axis=1)
    distances = np.linalg.norm(moved - target_points, axis=1)
    good = (moved[:, 2] > 0) & (pixels <= 3) & (distances <= np.maximum(.03, 3*paired_depth_scale(source_points, target_points)))
    return good, pixels, distances


def feature_agreement(source_points, target_points, target_pixels, pose, camera, *, require_distributed=True):
    """Keep measured color correspondences as a constraint after geometric ICP."""
    good, pixels, distances = _feature_support(source_points, target_points, target_pixels, pose, camera)
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
    return bool(count >= 35 and report["support_fraction"] >= 0.65 and (covered or not require_distributed)), report


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
        selected = residual < np.maximum(.025, 3*paired_depth_scale(source, target))
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


def refine_measured_motion(source, target, pixels, pose, camera, *, minimum=35):
    """Joint pixel/depth refinement, with fixed measured feature identities."""
    for _ in range(6):
        moved = source @ pose[:3, :3].T + pose[:3, 3]
        x, y, z = moved.T
        if np.any(z <= 0):
            return None
        projected = moved[:, :2] / z[:, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        supported = (np.linalg.norm(projected - pixels, axis=1) < 3) & (
            np.linalg.norm(moved - target, axis=1) < np.maximum(.025, 3*paired_depth_scale(source, target)))
        if supported.sum() < minimum:
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
        # Pixels retain fixed identities. Farther structured-light depth must
        # not overpower them with the same millimetre weight as close depth.
        # These are engineering scales, not calibrated sensor uncertainty.
        depth_scale = np.maximum(.003, paired_depth_scale(source, target))
        residual = np.column_stack((projected - pixels, (z - target[:, 2]) / depth_scale))
        j = np.concatenate((projection @ jacobian, jacobian[:, 2:3, :] / depth_scale[:, None, None]), axis=1)
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


class _Reference(NamedTuple):
    gray: np.ndarray
    depth: np.ndarray
    corners: np.ndarray
    stamp: float
    pose: np.ndarray
    points: np.ndarray
    ids: np.ndarray
    born_s: np.ndarray
    observations: np.ndarray


class VisualTracker:
    MAX_GAP_S = 0.75
    MAX_CORNERS = 500
    WINDOW_SIZE = 21
    PYRAMID_LEVEL = 3
    MIN_SEED_CORNERS = 60
    CORNER_QUALITY = 0.015
    MIN_SPACING_PX = 9
    GRID_COLS, GRID_ROWS = 8, 6
    REPLENISH_FRACTION = 0.8
    MIN_CELL_TRACKS = 3
    MIN_REPLENISH_INTERVAL_S = 0.2
    # A track may support this short step yet be too uncertain to keep alive.
    # Native depth quantization and color/depth delivery lag can move an
    # otherwise verified feature by more than half a pixel. Keep a tighter
    # subset of the existing 3 px / 30 mm accepted measurements, rather than
    # destroying their identities between ordinary selected captures.
    MAX_RETENTION_PIXEL_ERROR = 1.5
    MAX_RETENTION_DEPTH_ERROR_M = 0.03
    DEBUG_TRAIL_FRAMES = 20

    # Debug classifications follow the existing rejection gates, in order.
    FLOW_LOST, ROUNDTRIP_REJECTED, DEPTH_REJECTED, GEOMETRY_REJECTED, VERIFIED = range(5)

    def __init__(self, settings):
        self.settings = settings
        if settings.apriltag_tracking:
            from .apriltag import AprilTagDetector
            self._tag_detector = AprilTagDetector(settings.apriltag_dictionaries)
        else:
            self._tag_detector = None
        self.debug_enabled = False
        self.reset()

    @property
    def debug_enabled(self):
        return self._debug_enabled

    @debug_enabled.setter
    def debug_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled != getattr(self, "_debug_enabled", False):
            if hasattr(self, "_trail_frames"):
                self._trail_frames.clear()
            self.debug_snapshot = None
        self._debug_enabled = enabled

    def reset(self):
        self.segment = uuid.uuid4().hex[:16]
        self.pose = np.eye(4)
        self.previous = None
        self.history = deque(maxlen=5)
        self._tag_history = deque(maxlen=5)
        self.steps = 0
        self.debug_snapshot = None
        self._match_debug = None
        self._matched_reference = None
        self._next_feature_id = 0
        self._last_detection_s = None
        # Pixel positions only; the pose estimator still keeps five references.
        self._trail_frames = deque(maxlen=self.DEBUG_TRAIL_FRAMES)

    def _debug_trails(self, stamp):
        """Link retained identities only across consecutive observed frames.

        Empty entries age the display during rejected observations and break
        paths. Snapshots own their arrays, so queued previews cannot be changed
        by the next capture or by an identity being retired/replenished.
        """
        if not self._trail_frames:
            self._trail_frames.append(None)
        if self.history and self.history[-1].stamp == stamp:
            reference = self.history[-1]
            self._trail_frames[-1] = (reference.ids.copy(), reference.corners.reshape(-1, 2).copy())
        frames = list(self._trail_frames)
        segments, ages, identities = [], [], []
        for index, (previous, current) in enumerate(zip(frames, frames[1:]), start=1):
            if previous is None or current is None:
                continue
            ids, a, b = np.intersect1d(previous[0], current[0], assume_unique=True, return_indices=True)
            if len(ids):
                segments.append(np.stack((previous[1][a], current[1][b]), axis=1))
                ages.append(np.full(len(ids), len(frames) - 1 - index, np.uint8))
                identities.append(ids)
        return {
            "trail_segments": np.concatenate(segments) if segments else np.empty((0, 2, 2), np.float32),
            "trail_age_frames": np.concatenate(ages) if ages else np.empty(0, np.uint8),
            "trail_ids": np.concatenate(identities) if identities else np.empty(0, np.int64),
            "trail_frame_count": len(frames), "trail_frame_limit": self.DEBUG_TRAIL_FRAMES,
        }

    def _cell_indices(self, pixels):
        camera = self.settings.camera
        pixels = np.asarray(pixels).reshape(-1, 2)
        cols = np.clip((pixels[:, 0] * self.GRID_COLS / camera.width).astype(int), 0, self.GRID_COLS - 1)
        rows = np.clip((pixels[:, 1] * self.GRID_ROWS / camera.height).astype(int), 0, self.GRID_ROWS - 1)
        return rows * self.GRID_COLS + cols

    def _field_counts(self, pixels, depth):
        counts = np.bincount(self._cell_indices(pixels), minlength=self.GRID_ROWS * self.GRID_COLS)
        # Cropped/excluded depth cells do not create perpetual coverage deficits.
        eligible = np.array([np.count_nonzero(tile) >= 64
                             for row in np.array_split(depth, self.GRID_ROWS)
                             for tile in np.array_split(row, self.GRID_COLS, axis=1)])
        return counts, eligible

    def _replenish(self, gray, depth, reference, stamp):
        """Keep verified survivors; detect only in gaps, with a bounded cadence."""
        corners = reference.corners if reference is not None else np.empty((0, 1, 2), np.float32)
        counts, eligible = self._field_counts(corners, depth)
        room = self.MAX_CORNERS - len(corners)
        reason = "initial" if reference is None else (
            "count" if len(corners) < self.MAX_CORNERS * self.REPLENISH_FRACTION else
            "coverage" if np.any(eligible & (counts < self.MIN_CELL_TRACKS)) else "healthy")
        empty = np.empty((0, 1, 2), np.float32)
        points = np.empty((0, 3), float)
        if room <= 0 or reason == "healthy":
            return reference, empty, 0, "healthy"
        if (reference is not None and len(corners) >= self.MIN_SEED_CORNERS
                and self._last_detection_s is not None
                and stamp - self._last_detection_s < self.MIN_REPLENISH_INTERVAL_S - 1e-9):
            return reference, empty, 0, "cooldown"
        mask = (depth > 0).astype(np.uint8)
        # Round outward so subpixel survivors also retain the minimum spacing.
        for x, y in corners.reshape(-1, 2):
            cv2.circle(mask, (int(round(x)), int(round(y))), self.MIN_SPACING_PX + 1, 0, -1)
        if reason == "coverage":
            # Spend the remaining slots on under-covered cells, rather than
            # adding still more features to an already populated texture patch.
            for row, tiles in enumerate(np.array_split(mask, self.GRID_ROWS)):
                for col, tile in enumerate(np.array_split(tiles, self.GRID_COLS, axis=1)):
                    if counts[row * self.GRID_COLS + col] >= self.MIN_CELL_TRACKS:
                        tile[:] = 0
        candidates = cv2.goodFeaturesToTrack(gray, self.MAX_CORNERS * 4,
                                             self.CORNER_QUALITY, self.MIN_SPACING_PX, mask=mask)
        self._last_detection_s = stamp
        detected = len(candidates) if candidates is not None else 0
        if candidates is not None:
            candidate_points, measured = sampled_points(depth, candidates, self.settings.camera)
            candidates, candidate_points = candidates[measured], candidate_points[measured]
            # The detector orders by corner strength. Pick its strongest
            # remaining candidate in the least populated cell, repeatedly.
            pools = {}
            for index, cell in enumerate(self._cell_indices(candidates)):
                pools.setdefault(int(cell), deque()).append(index)
            queue = [(int(counts[cell]), cell) for cell in pools]
            heapq.heapify(queue)
            selected = []
            while queue and len(selected) < room:
                count, cell = heapq.heappop(queue)
                selected.append(pools[cell].popleft())
                if pools[cell]:
                    heapq.heappush(queue, (count + 1, cell))
            added, points = candidates[selected], candidate_points[selected]
        else:
            added = empty
        n = len(added)
        if reference is None and n < self.MIN_SEED_CORNERS:
            # A seed cannot manufacture measured support out of RGB corners.
            return None, empty, detected, "insufficient measured corners"
        if reference is not None and len(corners) + n < 40:
            return None, empty, detected, "insufficient reference tracks"
        ids = np.arange(self._next_feature_id, self._next_feature_id + n, dtype=np.int64)
        self._next_feature_id += n
        born = np.full(n, stamp, float)
        observations = np.ones(n, np.int32)
        if reference is not None:
            corners = np.concatenate((corners, added))
            points = np.concatenate((reference.points, points))
            ids = np.concatenate((reference.ids, ids))
            born = np.concatenate((reference.born_s, born))
            observations = np.concatenate((reference.observations, observations))
        else:
            corners = added
        return _Reference(gray, depth, corners, stamp, self.pose.copy(), points, ids, born, observations), added, detected, reason

    def _track_summary(self, reference, *, retained=0, added=0, retired=0, reason="unverified"):
        if reference is None:
            return {"active": 0, "retained": 0, "added": 0, "retired": retired, "median_age_s": 0.0,
                    "max_age_s": 0.0, "occupied_cells": 0, "eligible_cells": 0,
                    "replenishment": reason}
        counts, eligible = self._field_counts(reference.corners, reference.depth)
        ages = np.maximum(0, reference.stamp - reference.born_s)
        return {"active": len(reference.ids), "retained": retained, "added": added, "retired": retired,
                "median_age_s": float(np.median(ages)) if len(ages) else 0.0,
                "max_age_s": float(np.max(ages)) if len(ages) else 0.0,
                "occupied_cells": int(np.count_nonzero((counts > 0) & eligible)),
                "eligible_cells": int(eligible.sum()), "replenishment": reason}

    def measured_tracks(self, stamp):
        """Export current observed identities; the server samples its own depth.

        Failed frames must never inherit the last reference's observations.
        IDs are meaningful only within this tracker segment and generation.
        """
        if not self.history or self.history[-1].stamp != stamp:
            return None
        reference = self.history[-1]
        return {"version": 1, "image_size": [self.settings.camera.width, self.settings.camera.height],
                "ids": reference.ids.tolist(),
                "pixels": np.round(reference.corners.reshape(-1, 2), 3).tolist(),
                "depths_m": np.round(reference.points[:, 2], 6).tolist(),
                "observations": reference.observations.tolist()}

    def _match_reference(self, reference, gray, depth, stamp):
        old_gray, _, old_corners, old_stamp, old_pose, points_a = reference[:6]
        if not 0 < stamp - old_stamp <= self.MAX_GAP_S:
            return None, {}
        self._match_debug = None
        self._matched_reference = None
        if len(points_a) < 40:
            return None, {}
        # Only features with stable measured source depth can constrain a pose.
        # Request the already computed eigenvalue instead of the unused patch
        # score; forward/backward and measured-depth checks still decide support.
        new, forward, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, gray, old_corners, None,
            winSize=(self.WINDOW_SIZE, self.WINDOW_SIZE), maxLevel=self.PYRAMID_LEVEL,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
        if new is None or forward is None:
            return None, {}
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(
            gray, old_gray, new, None,
            winSize=(self.WINDOW_SIZE, self.WINDOW_SIZE), maxLevel=self.PYRAMID_LEVEL,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
        if back is None or reverse is None:
            return None, {}
        a, b = old_corners.reshape(-1, 2), new.reshape(-1, 2)
        supported = (forward.ravel() > 0) & (reverse.ravel() > 0)
        if self.debug_enabled:
            status = np.full(len(a), self.FLOW_LOST, np.uint8)
            status[supported] = self.ROUNDTRIP_REJECTED
            self._match_debug = {
                "source": a, "target": b, "status": status,
                "source_ids": reference.ids,
                "reference_age_ms": (stamp - old_stamp) * 1000,
                "reason": "Insufficient depth-supported tracks",
            }
        supported &= np.linalg.norm(a - back.reshape(-1, 2), axis=1) < 0.8
        if self.debug_enabled:
            status[supported] = self.DEPTH_REJECTED
        points_b, measured_b = sampled_points(depth, b, self.settings.camera)
        supported &= measured_b
        if self.debug_enabled:
            status[supported] = self.GEOMETRY_REJECTED
        pa, pb, pixels = points_a[supported], points_b[supported], b[supported]
        if len(pa) < 40:
            return None, {}
        ok, rotation, translation, inliers = cv2.solvePnPRansac(
            pa, pixels, camera_matrix(self.settings.camera), None,
            iterationsCount=80, reprojectionError=2.0, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < 35:
            if self.debug_enabled:
                self._match_debug["reason"] = "Pose RANSAC rejected"
            return None, {}
        delta = np.eye(4)
        delta[:3, :3] = cv2.Rodrigues(rotation)[0]
        delta[:3, 3] = translation.ravel()
        delta = measured_rigid_motion(pa, pb, delta)
        if delta is not None:
            delta = refine_measured_motion(pa, pb, pixels, delta, self.settings.camera)
        if delta is None:
            if self.debug_enabled:
                self._match_debug["reason"] = "Metric motion fit rejected"
            return None, {}
        valid, stats = feature_agreement(pa, pb, pixels, delta, self.settings.camera)
        angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
        if not valid or np.linalg.norm(delta[:3, 3]) > 0.15 or angle > 15:
            if self.debug_enabled:
                self._match_debug["reason"] = (
                    "Measured motion lacks distributed support" if not valid else
                    "Motion exceeds step budget")
            return None, stats
        agreed, pixel_errors, depth_errors = _feature_support(pa, pb, pixels, delta, self.settings.camera)
        reliable = (agreed & (pixel_errors <= self.MAX_RETENTION_PIXEL_ERROR)
                    & (depth_errors <= np.maximum(self.MAX_RETENTION_DEPTH_ERROR_M, 2*paired_depth_scale(pa, pb))))
        indices = np.flatnonzero(supported)[reliable]
        stats["retention_rejected"] = int(agreed.sum() - reliable.sum())
        pose = old_pose @ np.linalg.inv(delta)
        ids = reference.ids[indices]
        observations = reference.observations[indices].copy()
        if self.history and reference is not self.history[-1]:
            # Recovery can reuse an older identity. Its observation count must
            # include successful appearances in the newer retained references.
            for recent in self.history:
                _, here, there = np.intersect1d(ids, recent.ids, assume_unique=True, return_indices=True)
                observations[here] = np.maximum(observations[here], recent.observations[there])
        self._matched_reference = _Reference(
            gray, depth, new[indices], stamp, pose.copy(), points_b[indices],
            ids, reference.born_s[indices], observations + 1)
        if self.debug_enabled:
            status[np.flatnonzero(supported)[agreed]] = self.VERIFIED
            self._match_debug["reason"] = "Verified camera motion"
        return pose, stats

    def update(self, rgb, raw_depth, metadata):
        # Snapshots are local preview data, never part of the pose report.
        self.debug_snapshot = None
        self._match_debug = None
        self._matched_reference = None
        started = time.monotonic()
        stamp = metadata.get("depth_device_timestamp_unwrapped_s", metadata.get("timestamp_s", started))
        if self.debug_enabled:
            # Count camera observations, including timing/flow failures.
            self._trail_frames.append(None)
        lag = metadata.get("rgb_depth_delta_ms")
        synchronized = lag is None or abs(lag) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS
        if not synchronized and self._tag_detector is None:
            return {"valid": False, "reason": "RGB/depth timing exceeds 20 ms",
                    "segment": self.segment, "camera_to_local": self.pose.tolist(), "steps": self.steps}
        rgb, depth = prepare_rgbd(rgb, raw_depth, self.settings)
        tags = (self._tag_detector.detect(rgb, depth, self.settings.camera, synchronized=synchronized)
                if self._tag_detector is not None else None)
        if not synchronized:
            return {"valid": False, "reason": "RGB/depth timing exceeds 20 ms",
                    "segment": self.segment, "camera_to_local": self.pose.tolist(), "steps": self.steps,
                    "apriltags": tags.report()}
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        had_history = bool(self.history or self._tag_history)
        last_stamp = max([r.stamp for r in self.history] + [r[0] for r in self._tag_history], default=None)
        if last_stamp is not None and not 0 < stamp - last_stamp <= self.MAX_GAP_S:
            self.reset()
        seed_reference = not (self.history or self._tag_history)
        stats, matched_stamp = {}, None
        valid = False
        if tags is not None:
            from .apriltag import tag_motion
            for old_stamp, old_tags, old_pose in reversed(self._tag_history):
                if not 0 < stamp - old_stamp <= self.MAX_GAP_S:
                    continue
                relative, evidence = tag_motion(tags, old_tags, self.settings.camera)
                if relative is None:
                    continue
                angle = np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))
                if (np.linalg.norm(relative[:3, 3]) > self.settings.max_translation_m
                        or angle > self.settings.max_rotation_deg):
                    continue
                self.pose = old_pose @ relative
                self.steps += 1
                valid, matched_stamp = True, old_stamp
                stats = {"apriltag_support": evidence}
                break
        for reference in reversed(self.history):
            pose, flow_stats = self._match_reference(reference, gray, depth, stamp)
            if not valid:
                stats = flow_stats
            if pose is not None:
                if valid:
                    difference = np.linalg.inv(self.pose) @ pose
                    angle = np.degrees(np.arccos(np.clip((np.trace(difference[:3, :3]) - 1) / 2, -1, 1)))
                    if np.linalg.norm(difference[:3, 3]) > .03 or angle > 3:
                        self._matched_reference = None
                        continue
                    # Keep the ordinary feature identities when their motion
                    # agrees, while the measured tags determine this pose.
                    self._matched_reference = self._matched_reference._replace(pose=self.pose.copy())
                    stats.update(flow_stats)
                    break
                self.pose = pose
                self.steps += 1
                matched_stamp = reference.stamp
                valid = True
                break
        # A failed image never replaces the last trustworthy reference. An
        # expired chain starts a fresh origin, which is explicitly unverified.
        retained = len(self._matched_reference.ids) if self._matched_reference is not None else 0
        added, detected, replenishment = np.empty((0, 1, 2), np.float32), 0, "unverified"
        if valid or seed_reference:
            current, added, detected, replenishment = self._replenish(gray, depth, self._matched_reference, stamp)
            if current is not None:
                self.history.append(current)
                valid = valid or not had_history
            else:
                retained = 0
        if tags is not None and tags.tags and (valid or seed_reference):
            self._tag_history.append((stamp, tags, self.pose.copy()))
            valid = valid or not had_history
        if self.history:
            self.previous = self.history[-1][:4]
        tracks = self._track_summary(self.history[-1] if self.history else None,
                                     retained=retained, added=len(added),
                                     retired=stats.get("retention_rejected", 0), reason=replenishment)
        if self.debug_enabled:
            trails = self._debug_trails(stamp)
            self.debug_snapshot = {
                "image": rgb, "corners": added,
                "detected": detected, "tracks": tracks,
                "track_ids": self.history[-1].ids if self.history else np.empty(0, np.int64),
                "track_ages_s": self.history[-1].stamp - self.history[-1].born_s if self.history else np.empty(0),
                "window_size": self.WINDOW_SIZE, "pyramid_level": self.PYRAMID_LEVEL,
                "elapsed_ms": (time.monotonic() - started) * 1000,
                "reason": "Measured AprilTag RGB-D motion" if "apriltag_support" in stats else
                    "Seeded reference; no motion measured yet" if len(added) and matched_stamp is None else
                    "Visual motion unverified; retaining recent references" if self.history else
                    "Fewer than 60 measured corners; no reference seeded",
                **(self._match_debug or {}),
                **trails,
            }
        return {"valid": bool(valid), "segment": self.segment, "camera_to_local": self.pose.tolist(),
                "measured_tracks": self.measured_tracks(stamp),
                "steps": self.steps, "elapsed_ms": (time.monotonic() - started) * 1000,
                "reason": "Measured AprilTag RGB-D motion" if valid and "apriltag_support" in stats else
                    "Measured RGB-D motion" if valid else
                    "Visual motion unverified; retaining recent references" if self.history else
                    "Visual chain lost; new local origin",
                "reference_timestamp_s": matched_stamp, "tracks": tracks,
                **({"apriltags": tags.report()} if tags is not None else {}), **stats}
