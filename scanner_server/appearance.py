"""Bounded measured RGB-D appearance proposals, never final pose authority."""

from dataclasses import dataclass

import cv2
import numpy as np

from shared.calibration import camera_matrix
from shared.visual_tracking import paired_depth_scale


@dataclass
class Features:
    pixels: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray | None


def extract_features(rgb, depth, camera, *, depth_support=True, method="orb"):
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    detector = (cv2.SIFT_create(nfeatures=1200) if method == "sift"
                else cv2.ORB_create(nfeatures=1200, fastThreshold=12))
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        return Features(np.empty((0, 2)), np.empty((0, 3)), None)
    pixels = np.array([k.pt for k in keypoints], np.float32)
    x, y = np.rint(pixels).astype(int).T
    inside = (x >= 1) & (y >= 1) & (x < depth.shape[1] - 1) & (y < depth.shape[0] - 1)
    x = np.clip(x, 0, depth.shape[1] - 1)
    y = np.clip(y, 0, depth.shape[0] - 1)
    z = np.asarray(depth)[y, x].astype(float) / 1000
    valid = (z > 0) & inside
    if depth_support:
        # A measured center is mandatory; neighbours never fill unknown pixels.
        # Both detector types require a fully supported 3x3 image patch.
        pixels, descriptors, x, y = (
            pixels[valid],
            descriptors[valid],
            x[valid],
            y[valid],
        )
        if not len(pixels):
            return Features(np.empty((0, 2)), np.empty((0, 3)), None)
        patch = (
            np.array(
                [depth[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], float
            )
            / 1000
        )
        present = patch > 0
        patch[~present] = np.nan
        z = np.nanmedian(patch, axis=0)
        spread = np.nanmax(patch, axis=0) - np.nanmin(patch, axis=0)
        # Engineering discontinuity bound, not measured per-device uncertainty.
        valid = (present.sum(axis=0) >= 7) & (spread <= np.maximum(0.03, 0.04 * z))
    points = np.column_stack(
        (
            (pixels[:, 0] - camera.cx) * z / camera.fx,
            (pixels[:, 1] - camera.cy) * z / camera.fy,
            z,
        )
    )
    return Features(pixels[valid], points[valid], descriptors[valid])


def correspondences(source, target):
    if (
        source.descriptors is None
        or target.descriptors is None
        or min(len(source.points), len(target.points)) < 40
    ):
        return np.empty((0, 2), int)
    if source.descriptors.dtype != target.descriptors.dtype:
        return np.empty((0, 2), int)
    norm = cv2.NORM_HAMMING if source.descriptors.dtype == np.uint8 else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    forward = matcher.knnMatch(source.descriptors, target.descriptors, k=2)
    backward = matcher.knnMatch(target.descriptors, source.descriptors, k=2)
    reverse = {
        p[0].queryIdx: p[0].trainIdx
        for p in backward
        if len(p) == 2 and p[0].distance < 0.75 * p[1].distance
    }
    return np.array(
        [
            (p[0].queryIdx, p[0].trainIdx)
            for p in forward
            if len(p) == 2
            and p[0].distance < 0.75 * p[1].distance
            and reverse.get(p[0].trainIdx) == p[0].queryIdx
        ],
        int,
    ).reshape(-1, 2)


def propose_transform(source, target, camera, matches=None):
    matches = correspondences(source, target) if matches is None else matches
    if len(matches) < 40:
        return None
    a, b = matches.T
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        source.points[a],
        target.pixels[b],
        camera_matrix(camera),
        None,
        iterationsCount=200,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if (
        not ok
        or inliers is None
        or len(inliers) < 35
        or len(inliers) / len(matches) < 0.65
    ):
        return None
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation.reshape(3)
    if not np.all(np.isfinite(transform)):
        return None
    ids = inliers.reshape(-1)
    moved = source.points[a[ids]] @ transform[:3, :3].T + transform[:3, 3]
    distance = np.linalg.norm(moved - target.points[b[ids]], axis=1)
    # RGB matches cannot authorize a loop on an occluder or wrong measured depth.
    scale = paired_depth_scale(source.points[a[ids]], target.points[b[ids]])
    if np.mean(distance < np.maximum(.03, 3*scale)) < .8 or np.median(distance/np.maximum(.015, 1.5*scale)) > 1:
        return None
    return transform


def retrieve_pairs(features, minimum_separation=4, budget=20):
    """Rank non-adjacent views by mutual descriptors, independent of drifted poses."""
    candidates = []
    for i in range(len(features)):
        for j in range(i + minimum_separation, len(features)):
            matches = correspondences(features[i], features[j])
            if len(matches) >= 40:
                candidates.append((len(matches), i, j, matches))
    return sorted(candidates, key=lambda row: (-row[0], row[1], row[2]))[:budget]
