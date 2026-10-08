"""Fast pose refinement with fixed measured RGB-D feature identities.

Callers must independently verify feature support, reciprocal geometry and
motion before authorizing integration or a graph constraint.
"""

import numpy as np

from shared.visual_tracking import measured_rigid_motion, refine_measured_motion


def measured_pose(source, target, matches, initial, camera):
    if len(matches) < 40:
        return None
    a, b = matches.T
    try:
        pose = measured_rigid_motion(source.points[a], target.points[b], initial.copy())
        if pose is not None:
            pose = refine_measured_motion(source.points[a], target.points[b], target.pixels[b], pose, camera)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return pose if pose is not None and np.isfinite(pose).all() else None
