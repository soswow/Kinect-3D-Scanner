"""Experimental relative depth precision; coefficients need per-device evidence.

Only the confidence image is filtered. Measured depths, including holes and
small details, remain untouched. Accumulated TSDF weights are evidence scores,
not calibrated posterior variances or counts of independent observations.
"""

import cv2
import numpy as np


def _plane_angle(z, camera, valid, edge):
    """Estimate incidence from local planes in inverse depth.

    A perspective-viewed plane has affine inverse depth, so a least-squares
    slope over a 7x7 patch is exact on an ideal plane and less noisy than a
    one-pixel 3D cross product. A 3x3 fit supplies normals on narrower surfaces.
    Fits must have complete observed support and cannot cross a discontinuity.
    Where neither fit is supported, retain a conservative contribution rather
    than inventing a surface orientation or borrowing from the other side.
    """
    inverse = np.zeros_like(z)
    np.divide(1, z, out=inverse, where=valid)
    y, x = np.indices(z.shape, dtype=np.float32)
    rx = (x - camera.cx) / camera.fx
    ry = (y - camera.cy) / camera.fy
    ray_length = np.sqrt(1 + rx**2 + ry**2)
    cosine = np.full(z.shape, 0.5, np.float32)
    fitted = np.zeros(z.shape, bool)
    # Larger supports take precedence; the small fit only fills gaps.
    for size in (7, 3):
        support = np.ones((size, size), np.uint8)
        complete = cv2.erode(
            valid.astype(np.uint8), support, borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        crossing = cv2.dilate(
            edge.astype(np.uint8), support, borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        use = complete & ~crossing & ~fitted
        if not np.any(use):
            continue
        offsets = np.arange(size, dtype=np.float32) - size // 2
        derivative = offsets / np.dot(offsets, offsets)
        average = np.full(size, 1 / size, np.float32)
        mean = cv2.sepFilter2D(inverse, -1, average, average)
        a = cv2.sepFilter2D(inverse, -1, derivative, average) * camera.fx
        b = cv2.sepFilter2D(inverse, -1, average, derivative) * camera.fy
        c = mean - a * rx - b * ry
        length = np.sqrt(a**2 + b**2 + c**2)
        # n dot the unnormalised ray is the fitted inverse depth at its center.
        estimate = mean / np.maximum(length * ray_length, 1e-10)
        cosine[use] = np.clip(estimate[use], 0, 1)
        fitted |= use
    return cosine, fitted


def depth_confidence(depth, camera):
    z = depth.astype(np.float32) / 1000
    valid = np.isfinite(z) & (z > 0)
    z[~valid] = 0
    padded = np.pad(z, 1)
    neighbours = np.stack(
        (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:])
    )
    edge = np.any(
        (neighbours > 0) & (np.abs(neighbours - z) > np.maximum(0.02, 0.02 * z)), axis=0
    )
    cosine, fitted = _plane_angle(z, camera, valid, edge)
    angle = cosine**2
    # Relative inverse variance with a 1 m reference; lower confidence far away.
    sigma = 0.001 + 0.002 * z**2
    weight = np.clip((0.003 / sigma) ** 2, 0.05, 1) * angle
    weight[~valid | edge | (fitted & (cosine < 0.25))] = 0
    return np.ascontiguousarray(weight, np.float32)
