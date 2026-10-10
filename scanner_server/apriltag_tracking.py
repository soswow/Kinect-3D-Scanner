"""Recompute optional tag evidence from raw server frames, never client poses."""

import copy
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from shared.apriltag import tag_motion
from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS

REG = o3d.pipelines.registration


def observation(engine, index, rgb=None, depth=None):
    if not engine.settings.apriltag_tracking:
        return None
    cached = engine._apriltag_observations.get(index)
    if cached is not None:
        return cached
    if rgb is None:
        rgb, depth = engine._prepare_input(*engine.raw_frames[index], engine.settings)
    lag = engine.frame_metadata[index].get("rgb_depth_delta_ms")
    result = engine._apriltag_detector.detect(rgb, depth, engine.settings.camera,
        synchronized=lag is None or abs(lag) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS)
    engine._apriltag_observations[index] = result
    engine.frame_metadata[index]["apriltags_backend"] = result.report()
    return result


def register(engine, source, rgbd):
    if not engine.settings.apriltag_tracking or rgbd is None or not engine.poses:
        return None
    from .refinement import motion

    current = observation(engine, engine._processed_count,
        np.asarray(rgbd.color), np.rint(np.asarray(rgbd.depth) * 1000).astype(np.uint16))
    if not current.tags:
        return None
    candidates = dict(engine.poses[:5] + engine.poses[::4][-27:] + engine.poses[-8:])
    engine._apriltag_clouds = {i: c for i, c in engine._apriltag_clouds.items() if i in candidates}
    ranked = sorted(candidates, key=lambda i: (-len(current.tags.keys() & observation(engine, i).tags.keys()), -i))
    for index in ranked[:5]:
        relative, support = tag_motion(current, observation(engine, index), engine.settings.camera)
        if relative is None:
            continue
        pose = candidates[index] @ relative
        translation, angle = motion(np.linalg.inv(engine.cumulative_T) @ pose)
        if not engine._tracking_lost_frames and (
            translation > engine.settings.max_translation_m or angle > engine.settings.max_rotation_deg
        ):
            continue
        target = engine._apriltag_clouds.get(index)
        if target is None:
            rgb, depth = engine._prepare_input(*engine.raw_frames[index], engine.settings)
            target = engine._apriltag_clouds[index] = engine._make_reg_pcd(engine._make_rgbd(rgb, depth), normals=False)
        forward = REG.evaluate_registration(source, target, .0225, relative)
        moved = copy.deepcopy(source).transform(relative)
        reverse = REG.evaluate_registration(target, moved, .0225, np.eye(4))
        if (forward.fitness < .6 or reverse.fitness < .45
                or max(forward.inlier_rmse, reverse.inlier_rmse) > .015):
            continue
        engine._visual_evidence = {"target_index": index, "kind": "apriltag",
            "apriltag_support": support, "forward_overlap": float(forward.fitness),
            "reverse_overlap": float(reverse.fitness), "relative_pose": relative.tolist()}
        return SimpleNamespace(transformation=pose, fitness=forward.fitness,
            inlier_rmse=forward.inlier_rmse, correspondence_set=forward.correspondence_set)
    return None
