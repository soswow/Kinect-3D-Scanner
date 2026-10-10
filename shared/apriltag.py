"""Optional OpenCV AprilTag identities backed by measured RGB-D corners.

All coordinates use the prepared, rectified depth grid. Printed size is not
needed: both views must measure the same static tag surface with depth.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .settings import validate_apriltag_dictionaries
from .visual_tracking import sampled_points, refine_measured_motion


@dataclass
class Tag:
    pixels: np.ndarray
    points: np.ndarray


@dataclass
class TagFrame:
    tags: dict = field(default_factory=dict)
    detected: int = 0
    ambiguous: int = 0
    reason: str = ""
    detections: list = field(default_factory=list, repr=False)
    repeated: set = field(default_factory=set, repr=False)

    def report(self):
        return {"detected": self.detected, "usable": len(self.tags),
                "ambiguous": self.ambiguous, "reason": self.reason,
                "identities": [{"dictionary": name, "id": identity}
                               for name, identity in sorted(self.tags)],
                "repeated_identities": [{"dictionary": name, "id": identity}
                                        for name, identity in sorted(self.repeated)]}

    def preview(self):
        """Local display data, including detections rejected for tracking."""
        return [{"id": key[1], "corners": pixels.copy(), "usable": key in self.tags}
                for key, pixels in self.detections]


class AprilTagDetector:
    def __init__(self, dictionaries):
        names = validate_apriltag_dictionaries(dictionaries)
        aruco = getattr(cv2, "aruco", None)
        if aruco is None or not hasattr(aruco, "ArucoDetector") or any(
            not hasattr(aruco, name) for name in names
        ):
            raise ValueError("AprilTag tracking requires OpenCV 4.8 or newer with cv2.aruco")
        parameters = aruco.DetectorParameters()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        # Exact codes reduce cross-family false positives in mixed scenes.
        parameters.errorCorrectionRate = 0.0
        self.detectors = [(name, aruco.ArucoDetector(
            aruco.getPredefinedDictionary(getattr(aruco, name)), parameters)) for name in names]

    def decoded(self, rgb):
        gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
        observations = []
        for name, detector in self.detectors:
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                observations.extend(((name, int(identity)), np.asarray(pixels, float).reshape(4, 2))
                                    for identity, pixels in zip(ids.ravel(), corners))
        return observations

    def repeated_identities(self, rgb):
        counts = {}
        for key, _ in self.decoded(rgb):
            counts[key] = counts.get(key, 0) + 1
        return {key for key, count in counts.items() if count > 1}

    def detect(self, rgb, depth, camera, *, synchronized=True):
        observations = self.decoded(rgb)
        # A repeated printed ID cannot identify a unique physical landmark.
        # A quadrilateral decoded by multiple families is ambiguous as well.
        counts = {}
        for key, _ in observations:
            counts[key] = counts.get(key, 0) + 1
        ambiguous = {i for i, (key, _) in enumerate(observations) if counts[key] > 1}
        for i, (_, a) in enumerate(observations):
            for j in range(i):
                b = observations[j][1]
                if min(np.max(np.linalg.norm(a - np.roll(b, shift, axis=0), axis=1))
                       for shift in range(4)) < 5:
                    ambiguous.update((i, j))
        result = TagFrame(detected=len(observations), ambiguous=len(ambiguous), detections=observations,
                          repeated={key for key, count in counts.items() if count > 1})
        if not synchronized:
            result.reason = "RGB/depth timing exceeds 20 ms"
            return result
        for i, (key, pixels) in enumerate(observations):
            if i in ambiguous or np.min(np.linalg.norm(pixels - np.roll(pixels, 1, axis=0), axis=1)) < 20:
                continue
            points, measured = sampled_points(depth, pixels, camera)
            if measured.all():
                result.tags[key] = Tag(pixels, points)
        if not result.tags:
            result.reason = "No unambiguous tags with measured depth"
        return result


def without_repeated(frames):
    """Quarantine known duplicate family/IDs across views, including hidden copies."""
    repeated = set().union(*(frame.repeated for frame in frames))
    return [TagFrame({k: v for k, v in f.tags.items() if k not in repeated},
                     f.detected, f.ambiguous, f.reason, f.detections, set(f.repeated))
            for f in frames], repeated


def _pairs(source, target):
    keys = sorted(source.tags.keys() & target.tags.keys())
    if not keys:
        return keys, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2))
    return (keys, np.concatenate([source.tags[k].points for k in keys]),
            np.concatenate([target.tags[k].points for k in keys]),
            np.concatenate([source.tags[k].pixels for k in keys]),
            np.concatenate([target.tags[k].pixels for k in keys]))


def _support(a, b, pa, pb, pose, camera):
    def errors(points, pixels, transform):
        moved = points @ transform[:3, :3].T + transform[:3, 3]
        projected = moved[:, :2] / np.maximum(moved[:, 2:3], 1e-9)
        projected = projected * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        return (np.linalg.norm(projected - pixels, axis=1), moved[:, 2] > 0)
    forward, front = errors(a, pb, pose)
    reverse, back = errors(b, pa, np.linalg.inv(pose))
    distance = np.linalg.norm(a @ pose[:3, :3].T + pose[:3, 3] - b, axis=1)
    good = front & back & (forward <= 3) & (reverse <= 3) & (distance <= .025)
    # Keep or reject complete markers, never one convenient corner.
    supported = good.reshape(-1, 4).all(axis=1)
    return supported, np.maximum(forward, reverse), distance


def tag_agreement(source, target, pose, camera):
    keys, a, b, pa, pb = _pairs(source, target)
    if not keys or pose is None or not np.isfinite(pose).all():
        return False, {}
    supported, pixels, distances = _support(a, b, pa, pb, pose, camera)
    count = int(supported.sum())
    report = {"matched_tags": len(keys), "inlier_tags": count, "inliers": count * 4,
              "support_fraction": count / len(keys),
              "median_pixel_error": float(np.median(pixels)),
              "median_depth_error_m": float(np.median(distances)),
              "identities": [{"dictionary": name, "id": identity}
                             for (name, identity), good in zip(keys, supported) if good]}
    return count >= 1 and report["support_fraction"] >= .65, report


def _fit(a, b):
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    if np.linalg.eigvalsh(np.cov(a.T))[1] < 1e-6:
        return None
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, cb - rotation @ ca
    return pose


def tag_motion(source, target, camera):
    """Return source-to-target motion; reject inconsistent/moving tag groups."""
    keys, a, b, pa, pb = _pairs(source, target)
    if not keys:
        return None, {}
    # Each complete tag is a deterministic RANSAC hypothesis. Bound the work
    # using the largest measured tags, while checking every matched identity.
    largest = sorted(range(len(keys)), key=lambda i: -np.ptp(pa[4*i:4*i+4], axis=0).prod())[:64]
    candidates = [_fit(a, b)] + [_fit(a[4*i:4*i+4], b[4*i:4*i+4]) for i in largest]
    best, selected, score = None, None, (0, -float("inf"))
    for pose in candidates:
        if pose is None:
            continue
        supported, pixels, _ = _support(a, b, pa, pb, pose, camera)
        rank = (int(supported.sum()), -float(np.median(pixels)))
        if rank > score:
            best, selected, score = pose, np.repeat(supported, 4), rank
    if best is None or score[0] / len(keys) < .65:
        return None, {}
    try:
        best = _fit(a[selected], b[selected])
        if best is not None:
            best = refine_measured_motion(a[selected], b[selected], pb[selected], best, camera, minimum=4)
        good, report = tag_agreement(source, target, best, camera)
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return None, {}
    return (best if good else None), report
