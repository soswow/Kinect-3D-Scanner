"""Bounded world tag corners from accepted views, with separate depth witnesses.

Only committed camera poses extend the map. Current/rejected observations never
seed it. Pixel and measured-corner checks establish pose; sampled non-tag depth
checks it before fusion without requiring most of one old image to overlap.
"""

from types import SimpleNamespace

import cv2
import numpy as np

from shared.apriltag import _fit
from shared.calibration import camera_matrix


def _support(world, measured, pixels, world_to_camera, camera):
    moved = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = np.maximum(moved[:, 2], 1e-9)
    projected = moved[:, :2] / z[:, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
    pixel_error = np.linalg.norm(projected - pixels, axis=1)
    depth_error = np.linalg.norm(moved - measured, axis=1)
    # Range-dependent engineering tolerance, not a sensor uncertainty estimate.
    good = (moved[:, 2] > 0) & (pixel_error <= 3) & (
        depth_error <= .03 + .006 * measured[:, 2] ** 2)
    return good.reshape(-1, 4).all(axis=1), pixel_error, depth_error


def _refine(world, measured, pixels, pose, camera):
    """Small fixed-identity pixel/axial-depth solve in current camera coordinates."""
    pose = pose.copy()
    scale = .003 + .003 * measured[:, 2] ** 2
    for _ in range(10):
        moved = world @ pose[:3, :3].T + pose[:3, 3]
        x, y, z = moved.T
        if np.any(z <= .05):
            return None
        projected = moved[:, :2] / z[:, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        jacobian = np.zeros((len(world), 3, 6))
        jacobian[:, :, :3] = np.stack((np.column_stack((z * 0, z, -y)),
            np.column_stack((-z, z * 0, x)), np.column_stack((y, -x, z * 0))), axis=1)
        jacobian[:, :, 3:] = np.eye(3)
        projection = np.zeros((len(world), 2, 3))
        projection[:, 0, 0], projection[:, 0, 2] = camera.fx / z, -camera.fx * x / z ** 2
        projection[:, 1, 1], projection[:, 1, 2] = camera.fy / z, -camera.fy * y / z ** 2
        residual = np.column_stack((projected - pixels, (z - measured[:, 2]) / scale)).reshape(-1)
        matrix = np.concatenate((projection @ jacobian, jacobian[:, 2:3, :] / scale[:, None, None]), axis=1).reshape(-1, 6)
        weight = np.sqrt(np.minimum(1, 1.5 / np.maximum(np.abs(residual), 1e-9)))
        step = np.linalg.lstsq(matrix * weight[:, None], -residual * weight, rcond=None)[0]
        if not np.isfinite(step).all() or np.linalg.norm(step) > .5:
            return None
        correction = np.eye(4)
        correction[:3, :3], correction[:3, 3] = cv2.Rodrigues(step[:3])[0], step[3:]
        pose = correction @ pose
        if np.linalg.norm(step) < 1e-6:
            break
    return pose


def depth_witness(depth, camera, tags):
    """Disjoint, spread depth samples excluding the tags used for pose fitting."""
    excluded = np.zeros(depth.shape, np.uint8)
    for tag in tags.tags.values():
        cv2.fillConvexPoly(excluded, np.rint(tag.pixels).astype(np.int32), 1)
    excluded = cv2.dilate(excluded, np.ones((9, 9), np.uint8))
    witness_depth = depth.copy()
    witness_depth[excluded != 0] = 0
    y, x = np.mgrid[4:camera.height:8, 4:camera.width:8]
    z = depth[y, x]
    good = (z > 0) & (excluded[y, x] == 0)
    x, y, z = x[good], y[good], z[good]
    points = np.column_stack(((x - camera.cx) * z / camera.fx,
                              (y - camera.cy) * z / camera.fy, z))
    return SimpleNamespace(depth=witness_depth, camera=camera,
                           heldout=SimpleNamespace(points=points))


class LiveTagMap:
    """Up to 256 marker identities and eight accepted measurements per marker."""

    def __init__(self):
        self.samples = {}
        self.conflicts = {}
        self.excluded = set()
        self.pose_count = 0
        self.depth_cache = {}

    def sync(self, engine, observe):
        self.excluded.update(engine._apriltag_repeated)
        for index, pose in engine.poses[self.pose_count:]:
            frame = observe(engine, index)
            self.excluded.update(engine._apriltag_repeated)
            for key, tag in frame.tags.items():
                if key in self.excluded:
                    continue
                points = tag.points @ pose[:3, :3].T + pose[:3, 3]
                samples = self.samples.get(key, [])
                if samples:
                    center = np.median([row[1] for row in samples], axis=0)
                    if np.max(np.linalg.norm(points - center, axis=1)) > .04 + .006 * np.max(tag.points[:, 2]) ** 2:
                        self.conflicts[key] = self.conflicts.get(key, 0) + 1
                        if self.conflicts[key] >= 2:
                            self.excluded.add(key)
                        continue
                elif len(self.samples) >= 256:
                    continue
                self.conflicts[key] = 0
                samples.append((index, points))
                self.samples[key] = samples[:2] + samples[-6:] if len(samples) > 8 else samples
        for key in self.excluded:
            self.samples.pop(key, None)
        self.pose_count = len(engine.poses)

    def fit(self, current, camera):
        keys = sorted(current.tags.keys() & self.samples.keys())
        report = {"accepted": False, "mapped_tags": len(self.samples), "matched_tags": len(keys)}
        if len(keys) < 3:
            return None, {**report, "reason": "insufficient_mapped_tags"}
        world = np.concatenate([np.median([row[1] for row in self.samples[key]], axis=0) for key in keys])
        measured = np.concatenate([current.tags[key].points for key in keys])
        pixels = np.concatenate([current.tags[key].pixels for key in keys])
        candidates = [_fit(world, measured)]
        for i in sorted(range(len(keys)), key=lambda j: -np.ptp(pixels[4*j:4*j+4], axis=0).prod())[:16]:
            candidates.append(_fit(world[4*i:4*i+4], measured[4*i:4*i+4]))
        try:
            ok, rotation, translation, _ = cv2.solvePnPRansac(world, pixels, camera_matrix(camera), None,
                iterationsCount=100, reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
            if ok:
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rotation)[0], translation.ravel()
                candidates.append(pose)
        except cv2.error:
            pass
        best, selection, score = None, None, (0, -float("inf"))
        for pose in candidates:
            if pose is None:
                continue
            supported, errors, _ = _support(world, measured, pixels, pose, camera)
            selected = np.repeat(supported, 4)
            rank = (int(supported.sum()), -float(np.median(errors[selected])) if selected.any() else -float("inf"))
            if rank > score:
                best, selection, score = pose, selected, rank
        if best is None or score[0] < 3 or score[0] / len(keys) < .65:
            return None, {**report, "inlier_tags": score[0], "reason": "inconsistent_mapped_corners"}
        best = _refine(world[selection], measured[selection], pixels[selection], best, camera)
        if best is None:
            return None, {**report, "reason": "map_refinement_failed"}
        supported, errors, distances = _support(world, measured, pixels, best, camera)
        selected = np.repeat(supported, 4)
        inlier_keys = [key for key, good in zip(keys, supported) if good]
        if len(inlier_keys) < 3 or len(inlier_keys) / len(keys) < .65:
            return None, {**report, "reason": "map_refinement_lost_support"}
        extent = np.ptp(pixels[selected], axis=0) / [camera.width, camera.height]
        values = np.linalg.eigvalsh(np.cov(world[selected].T))
        distributed = bool(np.prod(extent) >= .04 and values[1] / max(values.sum(), 1e-9) > .002)
        report.update(inlier_tags=len(inlier_keys), support_fraction=len(inlier_keys)/len(keys),
            median_pixel_error=float(np.median(errors[selected])), median_depth_error_m=float(np.median(distances[selected])),
            depth_rmse_m=float(np.sqrt(np.mean(distances[selected] ** 2))),
            distributed=distributed, identities=[{"dictionary": k[0], "id": k[1]} for k in inlier_keys])
        if not distributed or report["median_pixel_error"] > 1.5:
            return None, {**report, "reason": "weak_map_pose_support"}
        observers = {}
        for key in inlier_keys:
            for index, _ in self.samples[key]:
                observers[index] = observers.get(index, 0) + 1
        references = sorted(observers, key=lambda i: (-observers[i], -i))[:5]
        report.update(accepted=True, reason="measured_world_tag_corners", reference_indices=references)
        return np.linalg.inv(best), report

    def verify_depth(self, engine, rgbd, current, pose, references, observe):
        from .geometry_registration import _visibility

        depth = np.asarray(rgbd.depth)
        source = depth_witness(depth, engine.settings.camera, current)
        poses = dict(engine.poses)
        rows = []
        self.depth_cache = {i:v for i,v in self.depth_cache.items() if i in references}
        for index in references:
            target = self.depth_cache.get(index)
            if target is None:
                _, depth = engine._prepare_input(*engine.raw_frames[index], engine.settings)
                target = depth_witness(depth.astype(float) / 1000, engine.settings.camera, observe(engine, index))
                self.depth_cache[index] = target
            relative = np.linalg.inv(poses[index]) @ pose
            for a, b, transform in ((source, target, relative), (target, source, np.linalg.inv(relative))):
                row = _visibility(a, b, transform)
                rows.append({"reference_index": index, **row})
        tested = sum(r["tested"] for r in rows)
        free = sum(r["tested"] * r["free_space_fraction"] for r in rows) / max(1, tested)
        agreement = sum(r["tested"] * r["depth_agreement"] for r in rows) / max(1, tested)
        accepted = (tested >= 200 and free <= .08 and agreement >= .65
                    and not any(r["tested"] >= 500 and r["free_space_fraction"] > .15 for r in rows))
        return {"accepted": accepted, "tested": tested, "free_space_fraction": free,
                "depth_agreement": agreement, "rows": rows,
                "reason": "independent_depth_supported" if accepted else "independent_depth_rejected"}


def register(engine, current, rgbd, observe):
    if getattr(engine, "_pose_seeds_only", False):
        engine.frame_metadata[engine._processed_count]["apriltag_live_map"] = {
            "accepted": False, "reason": "unverified_pose_seeds"}
        return None
    if engine._live_tag_map is None:
        engine._live_tag_map = LiveTagMap()
    bank = engine._live_tag_map
    bank.sync(engine, observe)
    pose, report = bank.fit(current, engine.settings.camera)
    engine.frame_metadata[engine._processed_count]["apriltag_live_map"] = report
    if pose is None:
        return None
    depth = bank.verify_depth(engine, rgbd, current, pose, report["reference_indices"], observe)
    report.update(accepted=depth["accepted"], depth_witness=depth)
    if not depth["accepted"]:
        report["reason"] = depth["reason"]
        return None
    index = report["reference_indices"][0]
    relative = np.linalg.inv(dict(engine.poses)[index]) @ pose
    engine._visual_evidence = {"target_index": index, "kind": "apriltag_map",
        "apriltag_support": report, "relative_pose": relative.tolist()}
    return SimpleNamespace(transformation=pose, fitness=depth["depth_agreement"],
        inlier_rmse=report["depth_rmse_m"], correspondence_set=np.empty((0, 2), int))
