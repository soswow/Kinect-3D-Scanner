"""Partial projective surface measurements for a jointly determined camera map."""

import numpy as np
from scipy.spatial import cKDTree


def depth_constraints(poses, views, *, neighbours=4, points_per_pair=240):
    """Associate raw selected surfaces at their projected target pixels.

    Intermediate feature-only cameras have no depth image and add no surfaces.
    A surface contributes only its normal component; it supplies no tangential
    point identity or standalone six-dimensional pose. Callers check different
    depth samples after optimization, including measurements omitted here.
    """
    nodes = sorted(i for i in poses if i < len(views) and views[i] is not None)
    if not nodes:
        return []
    centers = np.array([poses[i][:3, 3] for i in nodes])
    distances = np.linalg.norm(centers[:, None]-centers[None, :], axis=2)
    pairs = set()
    for row, a in enumerate(nodes):
        if row and a-nodes[row-1] <= 3:
            pairs.add((a, nodes[row-1]))
        bins = {a//24}
        selected = 0
        for b in np.argsort(distances[row]):
            other = nodes[b]
            if distances[row, b] >= 1.5:
                break
            if abs(a-other) <= 3 or other//24 in bins:
                continue
            bins.add(other//24)
            pairs.add((max(a, other), min(a, other)))
            selected += 1
            if selected == neighbours:
                break
    constraints = []
    for a, b in sorted(pairs):
        source, target = views[a], views[b]
        if min(len(source.cloud.points), len(target.cloud.points)) < 100:
            continue
        if "joint_training_tree" not in target.features:
            target.features["joint_training_tree"] = cKDTree(np.asarray(target.cloud.points))
        ids = np.linspace(0, len(source.cloud.points)-1, min(points_per_pair, len(source.cloud.points)), dtype=int)
        points = np.asarray(source.cloud.points)[ids]
        normals = np.asarray(source.cloud.normals)[ids]
        transform = np.linalg.inv(poses[b]) @ poses[a]
        moved = points @ transform[:3, :3].T+transform[:3, 3]
        c = target.camera
        pixels = np.rint(moved[:, :2]/np.maximum(moved[:, 2, None], .05)*[c.fx, c.fy]+[c.cx, c.cy]).astype(int)
        inside = ((moved[:, 2] > .1) & (pixels[:, 0] >= 1) & (pixels[:, 0] < c.width-1)
                  & (pixels[:, 1] >= 1) & (pixels[:, 1] < c.height-1))
        points, normals, moved, pixels = (values[inside] for values in (points, normals, moved, pixels))
        x, y = pixels.T
        samples = np.array([target.depth[y+dy, x+dx] for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))])
        depth = samples[0]
        measured = ((samples.min(axis=0) > .1) & (np.ptp(samples, axis=0) < .03+.004*depth**2)
                    & (abs(moved[:, 2]-depth) < .08+.003*depth**2))
        points, normals, pixels, depth = (values[measured] for values in (points, normals, pixels, depth))
        if not len(depth):
            continue
        target_points = np.column_stack(((pixels-[c.cx, c.cy])/[c.fx, c.fy]*depth[:, None], depth))
        distance, indices = target.features["joint_training_tree"].query(target_points, workers=1)
        target_normals = np.asarray(target.cloud.normals)[indices]
        keep = ((distance < .05) & (np.abs(np.sum((normals @ transform[:3, :3].T)*target_normals, axis=1)) > .95))
        if keep.sum() >= 12:
            constraints.append((a, b, points[keep], target_points[keep], target_normals[keep]))
    return constraints
