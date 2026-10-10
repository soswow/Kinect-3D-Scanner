"""Use bounded camera-side RGB-D observations between selected raw captures.

Intermediate pixel/depth observations are measurements from the client, not
recorded pose authority. Only independently checked selected views reach fusion.
"""

import numpy as np

from .appearance import Features
from .geometry_registration import rigid
from .motion_evidence import MotionEvidence


def frame_identity(metadata):
    generation = metadata.get("capture_generation")
    if not isinstance(generation, str) or not generation:
        return None
    sequence = metadata.get("sensor_frame_sequences") or {}
    if all(type(sequence.get(k)) is int and sequence[k] >= 0 for k in ("rgb", "depth")):
        return generation, sequence["rgb"], sequence["depth"]
    stamp = metadata.get("captured_monotonic_s")
    return (generation, stamp) if isinstance(stamp, (float, int)) and np.isfinite(stamp) else None


def intermediate_features(observation, camera):
    """Validate measured pixels and axial depths without manufacturing points."""
    if not isinstance(observation, dict):
        return None
    try:
        if observation.get("version") != 1 or observation.get("image_size") != [camera.width, camera.height]:
            return None
        identities = observation["ids"]
        if not 1 <= len(identities) <= 192 or any(type(i) is not int or not 0 <= i <= 2**53 for i in identities):
            return None
        ids = np.asarray(identities, np.int64)
        pixels = np.asarray(observation["pixels"], float)
        depths = np.asarray(observation["depths_m"], float)
        if (pixels.shape != (len(ids), 2) or depths.shape != (len(ids),) or len(np.unique(ids)) != len(ids)
                or not np.isfinite(pixels).all() or not np.isfinite(depths).all()
                or np.any(pixels < 0) or np.any(pixels >= [camera.width, camera.height])
                or np.any(depths < .1) or np.any(depths > 10.)):
            return None
        points = np.column_stack(((pixels[:, 0]-camera.cx)*depths/camera.fx,
                                  (pixels[:, 1]-camera.cy)*depths/camera.fy, depths))
        return ids, Features(pixels, points, None)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def expand_visual_measurements(poses, edges, motion):
    metadata = list(motion.metadata)
    appearance = list(motion.appearance) or [None]*len(metadata)
    tracked = list(motion.tracked) or [None]*len(metadata)
    expanded, relations = dict(poses), list(edges)
    identities = {frame_identity(metadata[i]): i for i in poses if frame_identity(metadata[i]) is not None}
    anchors = {}
    for i in poses:
        tracking = metadata[i].get("visual_tracking", {})
        segment = tracking.get("segment")
        generation = metadata[i].get("capture_generation")
        if segment and generation and (tracking.get("valid") or tracked[i] is not None):
            try:
                local = np.asarray(tracking["camera_to_local"], float)
                if rigid(local):
                    anchors.setdefault((generation, segment), []).append((i, local))
            except (KeyError, TypeError, ValueError):
                pass
    observations = {}
    for m in motion.metadata:
        for row in m.get("motion_history", {}).get("visual", ()):
            if row.get("feature_observations"):
                record = {**row, "captured_monotonic_s": row.get("host_monotonic_s")}
                key = frame_identity(record)
                if key is not None:
                    observations[key] = record
    groups = {}
    for row in sorted(observations.values(), key=lambda row: row["captured_monotonic_s"]):
        tracking = row.get("visual_tracking", {})
        key = (row.get("capture_generation"), tracking.get("segment"))
        if key not in anchors:
            continue
        identity = frame_identity(row)
        node = identities.get(identity)
        if node is not None and tracked[node] is None:
            # A selected image's own measured depth takes precedence over the
            # compact client observation, including missing/cropped samples.
            continue
        if node is None:
            features = intermediate_features(row["feature_observations"], motion.camera)
            if features is None:
                continue
            try:
                local = np.asarray(tracking["camera_to_local"], float)
                if not rigid(local):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            anchor, anchor_local = min(anchors[key], key=lambda p:
                abs(metadata[p[0]].get("captured_monotonic_s", 0)-row["captured_monotonic_s"]))
            node = len(metadata)
            metadata.append(row); appearance.append(None); tracked.append(features)
            expanded[node] = poses[anchor] @ np.linalg.inv(anchor_local) @ local
            identities[identity] = node
        groups.setdefault(key, []).append(node)
    for key, nodes in groups.items():
        nodes = sorted(set(nodes) | {i for i, _ in anchors[key]},
                       key=lambda i: metadata[i].get("captured_monotonic_s", 0))
        for a, b in zip(nodes[1:], nodes[:-1]):
            ta, tb = (metadata[i]["visual_tracking"] for i in (a, b))
            pa, pb = np.asarray(ta["camera_to_local"]), np.asarray(tb["camera_to_local"])
            relations.append({"source": a, "target": b, "transform": np.linalg.inv(pb) @ pa,
                              "sensor_visual_prior": True})
    extended = MotionEvidence(metadata, motion.camera, appearance, tracked=tracked,
                              gravity=any(g is not None for g in motion.gravity),
                              visual=motion.visual, journal=motion.journal)
    extended.tag_frames = getattr(motion, "tag_frames", ())
    direct = {}
    for node in expanded:
        feature = tracked[node]
        if feature is None:
            continue
        tracking = metadata[node].get("visual_tracking", {})
        generation, segment = metadata[node].get("capture_generation"), tracking.get("segment")
        if not generation or not segment:
            continue
        for index, identity in enumerate(feature[0]):
            direct.setdefault((generation, segment, int(identity)), []).append((node, index))
    extended.direct_groups = [group for group in direct.values() if len(group) >= 2]
    return expanded, relations, extended


def refine_continuous_graph(poses, edges, motion, **kwargs):
    from .joint_depth_bundle import refine_motion_graph
    expanded, relations, extended = expand_visual_measurements(poses, edges, motion)
    # Only the selected views supply surfaces; intermediate cameras contribute
    # pixel/depth observations. Separate raw validation remains with caller.
    result, report = refine_motion_graph(expanded, relations, extended, **kwargs)
    report["globally_observed_cameras"] = [i for i in report.get("globally_observed_cameras", ()) if i in poses]
    report.update(intermediate_cameras=len(expanded)-len(poses),
                  intermediate_images_uploaded=False)
    return {i: result[i] for i in poses}, report
