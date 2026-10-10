"""Shared measured planar surfaces, supplying partial global pose information."""

import numpy as np
import open3d as o3d


def measured_planes(view):
    if "joint_planes" in view.features:
        return view.features["joint_planes"]
    points = np.asarray(view.cloud.points)
    remaining, found = points.copy(), []
    minimum = max(200, int(len(points)*.1))
    for _ in range(4):
        if len(remaining) < minimum:
            break
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining))
        tolerance = .012+.002*np.median(remaining[:, 2])**2
        _, indices = cloud.segment_plane(tolerance, 3, 150)
        if len(indices) < minimum:
            break
        patch = remaining[indices]
        center = patch.mean(axis=0)
        values, basis = np.linalg.eigh(np.cov(patch.T))
        normal = basis[:, 0]
        spread = np.ptp((patch-center) @ basis[:, 1:], axis=0)
        if np.prod(spread) >= .25 and values[0]/max(values[1], 1e-9) < .03:
            samples = patch[np.linspace(0, len(patch)-1, min(80, len(patch)), dtype=int)]
            found.append((normal, center, samples))
        remaining = np.delete(remaining, indices, axis=0)
    view.features["joint_planes"] = found
    return found


def plane_observations(poses, views):
    """Group overlapping plane hypotheses; their observations remain raw points.

    No Manhattan assumption is made. Angle/distance bounds initialize surface
    identities only; global raw-depth consistency remains with the caller.
    """
    if views is None:
        return []
    groups = []
    for node in sorted(i for i in poses if i < len(views) and views[i] is not None):
        pose = poses[node]
        for normal, center, samples in measured_planes(views[node]):
            n = pose[:3, :3] @ normal
            c = pose[:3, :3] @ center+pose[:3, 3]
            distance = float(n @ c)
            if distance < 0:
                n, distance = -n, -distance
            if distance < .05:
                continue
            candidates = [g for g in groups if all(abs(distance-d) < .15 and n @ other > np.cos(np.radians(12))
                          for other, d in g["measurements"])]
            group = min(candidates, key=lambda g: abs(distance-np.mean([d for _, d in g["measurements"]]))) if candidates else None
            if group is None:
                group = {"measurements": [], "observations": []}
                groups.append(group)
            group["measurements"].append((n, distance))
            group["observations"].append((node, samples))
    result = []
    for group in groups:
        nodes = {node for node, _ in group["observations"]}
        if len(nodes) < 3 or max(nodes)-min(nodes) <= 3:
            continue
        normal = np.mean([n for n, _ in group["measurements"]], axis=0)
        normal /= np.linalg.norm(normal)
        distance = np.mean([d for _, d in group["measurements"]])
        result.append((normal*distance, group["observations"]))
    return result
