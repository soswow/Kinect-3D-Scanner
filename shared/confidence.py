"""Experimental engineering confidence; coefficients need per-device evidence."""

import numpy as np


def depth_confidence(depth, camera):
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
    # Relative inverse variance with a 1 m reference; lower confidence far away.
    sigma = 0.001 + 0.002 * z**2
    weight = np.clip((0.003 / sigma) ** 2, 0.05, 1) * angle
    weight[(z <= 0) | edge | (supported & (cosine < 0.25))] = 0
    return np.ascontiguousarray(weight, np.float32)
