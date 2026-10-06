"""Conservative bounded pose-graph refinement, followed by fresh TSDF fusion.

Only measured RGB-D geometry is used. No reference trajectory is read. A loop
must pass reciprocal ICP and normal-diversity checks; independent point samples
must improve after optimization before the proposed poses can be committed.
"""

import copy

import numpy as np
import open3d as o3d

from shared.calibration import prepare_rgbd
from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS

from .appearance import extract_features, propose_transform, retrieve_pairs

REG = o3d.pipelines.registration


def motion(transform):
    return float(np.linalg.norm(transform[:3, 3])), float(
        np.degrees(np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1)))
    )


def interpolate(a, b, fraction):
    result = np.eye(4)
    relative = a[:3, :3].T @ b[:3, :3]
    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    if angle < 1e-8:
        result[:3, :3] = a[:3, :3]
    else:
        axis = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        ) / (2 * np.sin(angle))
        result[:3, :3] = a[:3, :3] @ o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis * angle * fraction
        )
    result[:3, 3] = a[:3, 3] * (1 - fraction) + b[:3, 3] * fraction
    return result


def _match(source, target, initial):
    pose = initial
    for distance, iterations in ((0.12, 40), (0.06, 30), (0.03, 20)):
        result = REG.registration_icp(
            source,
            target,
            distance,
            pose,
            REG.TransformationEstimationPointToPlane(REG.HuberLoss(0.01)),
            REG.ICPConvergenceCriteria(max_iteration=iterations),
        )
        pose = result.transformation
    return result


def _trustworthy(result, target):
    if (
        not np.all(np.isfinite(result.transformation))
        or result.fitness < 0.6
        or result.inlier_rmse > 0.015
    ):
        return False
    matches = np.asarray(result.correspondence_set)
    if len(matches) < 100:
        return False
    normals = np.asarray(target.normals)[matches[:, 1]]
    return np.linalg.eigvalsh(normals.T @ normals / len(normals))[0] >= 0.002


def _distance(source, target, transform):
    moved = copy.deepcopy(source).transform(transform)
    distances = np.asarray(moved.compute_point_cloud_distance(target))
    # A saturated loss counts disappearing correspondences instead of letting
    # a smaller overlapping subset create an apparently better result.
    return float(np.mean(np.minimum(distances, 0.05) ** 2))


def propose_poses(engine, max_keyframes=32, max_loops=40):
    report = {
        "applied": False,
        "reason": "No trustworthy loop constraints",
        "loops": 0,
        "keyframes": 0,
        "validation": "independent point samples; saturated squared distance",
    }
    poses = engine.poses
    if len(poses) < 7:
        report["reason"] = "Too few accepted frames for loop refinement"
        return None, report
    chosen = np.unique(
        np.linspace(0, len(poses) - 1, min(max_keyframes, len(poses)), dtype=int)
    )
    originals = [poses[i][1] for i in chosen]
    training, validation, features = [], [], []
    for i in chosen:
        index, _ = poses[i]
        rgb, raw = engine.raw_frames[index]
        rgb, depth = prepare_rgbd(rgb, raw, engine.settings)
        lag = engine.frame_metadata[index].get("rgb_depth_delta_ms")
        features.append(
            extract_features(
                rgb,
                depth if lag is None or abs(lag) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS else np.zeros_like(depth),
                engine.settings.camera,
            )
        )
        rgbd = engine._make_rgbd(rgb, depth)
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, engine.intrinsic
        ).voxel_down_sample(0.02)
        if len(cloud.points) > 8000:
            cloud = cloud.uniform_down_sample(int(np.ceil(len(cloud.points) / 8000)))
        train = cloud.select_by_index(list(range(0, len(cloud.points), 2)))
        heldout = cloud.select_by_index(list(range(1, len(cloud.points), 2)))
        train.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30)
        )
        training.append(train)
        validation.append(heldout)
    report["keyframes"] = len(chosen)
    graph = REG.PoseGraph()
    for pose in originals:
        graph.nodes.append(REG.PoseGraphNode(pose.copy()))
    constraints = []

    def add_edge(i, j, uncertain, proposal=None):
        initial = np.linalg.inv(originals[j]) @ originals[i]
        result = _match(
            training[i], training[j], initial if proposal is None else proposal
        )
        reliable = _trustworthy(result, training[j])
        transform = result.transformation if reliable else initial
        if not uncertain and not reliable and proposal is not None:
            # A verified RGB-D seed can connect poorly conditioned ICP pairs
            # only with non-planar measured feature support and independent
            # training geometry agreement. Held-out points remain unused here.
            eigenvalues = np.linalg.eigvalsh(np.cov(features[i].points.T))
            if (
                eigenvalues[0] / max(eigenvalues.sum(), 1e-9) > 0.002
                and _distance(training[i], training[j], proposal) < 0.015**2
            ):
                transform = proposal
                reliable = True
        if reliable and uncertain:
            reverse = _match(
                training[j], training[i], np.linalg.inv(result.transformation)
            )
            trans, angle = motion(reverse.transformation @ result.transformation)
            correction, correction_angle = motion(
                np.linalg.inv(initial) @ result.transformation
            )
            reliable = (
                _trustworthy(reverse, training[i])
                and trans < 0.01
                and angle < 2
                and correction < (0.75 if proposal is not None else 0.2)
                and correction_angle < (30 if proposal is not None else 15)
            )
        if uncertain and not reliable:
            return False
        if uncertain:
            transform = result.transformation if reliable else initial
        information = REG.get_information_matrix_from_point_clouds(
            training[i], training[j], 0.03, transform
        )
        if not np.all(np.isfinite(information)):
            return False
        if not reliable:
            information *= 0.01  # Unverified odometry is a weak prior.
        graph.edges.append(REG.PoseGraphEdge(i, j, transform, information, uncertain))
        constraints.append((i, j))
        return True

    for i in range(len(chosen) - 1):
        adjacent = propose_transform(
            features[i], features[i + 1], engine.settings.camera
        )
        if not add_edge(i, i + 1, False, adjacent):
            report["reason"] = "Could not construct a connected trajectory graph"
            return None, report
        # Start the robust graph from its measured sequential constraints.
        # Large accumulated drift can otherwise cause valid loops to be pruned
        # before the optimizer has a chance to correct the trajectory.
        graph.nodes[i + 1].pose = graph.nodes[i].pose @ np.linalg.inv(
            graph.edges[-1].transformation
        )
    candidates = []
    travel = np.r_[
        0,
        np.cumsum(
            [
                np.linalg.norm(b[:3, 3] - a[:3, 3])
                for a, b in zip(originals, originals[1:])
            ]
        ),
    ]
    for i in range(len(chosen)):
        for j in range(i + 4, len(chosen)):
            separation, angle = motion(np.linalg.inv(originals[j]) @ originals[i])
            if travel[j] - travel[i] >= 0.15 and separation < 0.35 and angle < 45:
                candidates.append((separation, i, j))
    retrieved = retrieve_pairs(features, budget=max_loops // 2)
    report["appearance_candidates"] = len(retrieved)
    report["appearance_loops"] = 0
    seen = set()
    for _, i, j, matches in retrieved:
        seen.add((i, j))
        proposal = propose_transform(
            features[i], features[j], engine.settings.camera, matches
        )
        if proposal is not None:
            accepted = bool(add_edge(i, j, True, proposal))
            report["loops"] += accepted
            report["appearance_loops"] += accepted
    for _, i, j in sorted(candidates):
        if len(seen) >= max_loops:
            break
        if (i, j) in seen:
            continue
        seen.add((i, j))
        report["loops"] += bool(add_edge(i, j, True))
    if not report["loops"]:
        return None, report
    REG.global_optimization(
        graph,
        REG.GlobalOptimizationLevenbergMarquardt(),
        REG.GlobalOptimizationConvergenceCriteria(),
        REG.GlobalOptimizationOption(
            max_correspondence_distance=0.03,
            edge_prune_threshold=0.25,
            reference_node=0,
        ),
    )
    refined = [node.pose.copy() for node in graph.nodes]
    for old, new in zip(originals, refined):
        translation, angle = motion(new @ np.linalg.inv(old))
        if not np.all(np.isfinite(new)) or translation > 0.75 or angle > 30:
            report["reason"] = "Optimization exceeded correction bounds"
            return None, report
    before, after = [], []
    for i, j in constraints:
        before.append(
            _distance(
                validation[i], validation[j], np.linalg.inv(originals[j]) @ originals[i]
            )
        )
        after.append(
            _distance(
                validation[i], validation[j], np.linalg.inv(refined[j]) @ refined[i]
            )
        )
    report.update(
        validation_before_m2=float(np.mean(before)),
        validation_after_m2=float(np.mean(after)),
    )
    if np.mean(after) >= np.mean(before) * 0.98 or any(
        a > b * 1.1 + 1e-6 for a, b in zip(after, before)
    ):
        report["reason"] = "Independent geometry did not improve consistently"
        return None, report
    corrections = [new @ np.linalg.inv(old) for old, new in zip(originals, refined)]
    proposals = []
    for i, (index, pose) in enumerate(poses):
        right = min(int(np.searchsorted(chosen, i, side="right")), len(chosen) - 1)
        left = max(0, right - 1)
        fraction = (i - chosen[left]) / max(1, chosen[right] - chosen[left])
        correction = interpolate(corrections[left], corrections[right], fraction)
        proposals.append((index, correction @ pose))
    report["reason"] = "Validated pose proposal; awaiting fresh fusion"
    return proposals, report
