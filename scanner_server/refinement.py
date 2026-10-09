"""Conservative bounded pose-graph refinement, followed by fresh TSDF fusion.

Only measured RGB-D geometry is used. No reference trajectory is read. A loop
must pass reciprocal ICP and normal-diversity checks; independent point samples
must improve after optimization before the proposed poses can be committed.
"""

import copy
import time

import numpy as np
import open3d as o3d

from shared.calibration import prepare_rgbd

from .appearance import extract_features, propose_transform, retrieve_pairs
from .bundle_adjustment import _BudgetExceeded, _DepthClouds, _output_poses, _timely, validate_depth
from .pose_candidates import choose_keyframes, fragment_pairs, validate_fragment_boundaries

REG = o3d.pipelines.registration
MAX_SECONDS = 90.0
MAX_CLOUD_POINTS = 48000
VOXEL_SIZE_M = 0.0075


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
    from .cuda_registration import match

    result = match(source, target, initial)
    if result is not None:
        return result
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


def _match_failure(result, target):
    if not np.all(np.isfinite(result.transformation)):
        return "nonfinite_transform"
    if not np.isfinite(result.fitness) or result.fitness < 0.6:
        return "insufficient_overlap"
    if not np.isfinite(result.inlier_rmse) or result.inlier_rmse > 0.015:
        return "alignment_error"
    matches = np.asarray(result.correspondence_set)
    if len(matches) < 100:
        return "too_few_correspondences"
    normals = np.asarray(target.normals)[matches[:, 1]]
    if not np.isfinite(normals).all() or np.linalg.eigvalsh(normals.T @ normals / len(normals))[0] < 0.002:
        return "insufficient_normal_diversity"
    return None


def _trustworthy(result, target):
    return _match_failure(result, target) is None


def _distance(source, target, transform):
    moved = copy.deepcopy(source).transform(transform)
    distances = np.asarray(moved.compute_point_cloud_distance(target))
    # A saturated loss counts disappearing correspondences instead of letting
    # a smaller overlapping subset create an apparently better result.
    return float(np.mean(np.minimum(distances, 0.05) ** 2))


def propose_poses(engine, max_keyframes=64, max_loops=40):
    started = time.monotonic()
    report = {
        "applied": False,
        "reason": "Loop refinement has not completed",
        "loops": 0,
        "keyframes": 0,
        "validation": "separate training/witness/validation samples; all-output raw depth; saturated squared distance",
        "candidate_results": [], "rejection_counts": {},
        "sequential_results": [],
        "max_keyframes": max_keyframes, "max_loops": max_loops,
        "time_budget_s": MAX_SECONDS,
        "sampling_voxel_m": VOXEL_SIZE_M, "max_cloud_points": MAX_CLOUD_POINTS,
    }
    poses = engine.poses
    if len(poses) < 7:
        report["reason"] = "Too few accepted frames for loop refinement"
        return None, report
    hints = fragment_pairs(engine)
    protected = fragment_pairs(engine, temporal_only=True)
    chosen = choose_keyframes(len(poses), max_keyframes, protected + hints[:max_loops])
    report["keyframe_indices"] = [int(poses[i][0]) for i in chosen]
    report["fragment_candidates"] = len(hints)

    def within_budget():
        if time.monotonic() - started <= MAX_SECONDS:
            return True
        report.update(reason="Loop refinement exceeded its time budget", budget_exceeded=True)
        return False

    def record(i, j, origin, accepted, reason, **metrics):
        metrics = {key: value if np.isfinite(value) else None for key, value in metrics.items()}
        report["candidate_results"].append({"source_index": int(poses[chosen[i]][0]),
            "target_index": int(poses[chosen[j]][0]), "origin": origin,
            "accepted": bool(accepted), "reason": reason, **metrics})
        if not accepted:
            counts = report["rejection_counts"]
            counts[reason] = counts.get(reason, 0) + 1
    originals = [poses[i][1] for i in chosen]
    training, witnesses, validation, features = [], [], [], []
    for i in chosen:
        if not within_budget():
            return None, report
        index, _ = poses[i]
        rgb, raw = engine.raw_frames[index]
        rgb, depth = prepare_rgbd(rgb, raw, engine.settings)
        features.append(
            extract_features(
                rgb,
                depth if _timely(engine.frame_metadata[index]) else np.zeros_like(depth),
                engine.settings.camera, method="sift",
            )
        )
        rgbd = engine._make_rgbd(rgb, depth)
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, engine.intrinsic
        ).voxel_down_sample(VOXEL_SIZE_M)
        if len(cloud.points) > MAX_CLOUD_POINTS:
            cloud = cloud.uniform_down_sample(int(np.ceil(len(cloud.points) / MAX_CLOUD_POINTS)))
        train = cloud.select_by_index(list(range(0, len(cloud.points), 2)))
        witness = cloud.select_by_index(list(range(1, len(cloud.points), 4)))
        heldout = cloud.select_by_index(list(range(3, len(cloud.points), 4)))
        if min(len(train.points), len(witness.points), len(heldout.points)) < 100:
            report.update(reason="Insufficient measured depth at selected loop view", invalid_depth_index=int(index))
            return None, report
        train.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
        )
        training.append(train)
        witnesses.append(witness)
        validation.append(heldout)
    report["keyframes"] = len(chosen)
    graph = REG.PoseGraph()
    for pose in originals:
        graph.nodes.append(REG.PoseGraphNode(pose.copy()))
    constraints = []
    report["measured_sequential_edges"] = 0
    selected = {int(position): i for i, position in enumerate(chosen)}
    protected_transforms = {}
    # Raw visual boundary measurements constrain the graph, rather than merely
    # vetoing its output after geometric ICP has displaced those identities.
    from .fragments import View, _local_match, _visual_witness
    views = [View(int(poses[position][0]), train, heldout, feature)
             for position, train, heldout, feature in zip(chosen, training, witnesses, features)]
    report["protected_constraints"] = []
    for a, b in protected:
        if a not in selected or b not in selected:
            report["reason"] = "Keyframe budget does not cover retained measured boundaries"
            return None, report
        if not within_budget():
            return None, report
        i, j = selected[a], selected[b]
        # _local_match bounds per-capture motion in the later-to-earlier direction.
        measured = _local_match(views[j], views[i], engine.settings.camera, engine.settings,
            np.linalg.inv(originals[i]) @ originals[j], measured_first=True)
        if measured is None or not _visual_witness(views[j], views[i], measured, engine.settings.camera)[0]:
            report.update(reason="Could not remeasure a retained temporal boundary",
                          failed_boundary_indices=[views[i].index, views[j].index])
            return None, report
        protected_transforms[i, j] = np.linalg.inv(measured)
        report["protected_constraints"].append([views[i].index, views[j].index])

    def add_edge(i, j, uncertain, proposal=None, origin="geometry"):
        initial = np.linalg.inv(originals[j]) @ originals[i]
        result = _match(
            training[i], training[j], initial if proposal is None else proposal
        )
        reliable = _trustworthy(result, training[j])
        failure = _match_failure(result, training[j])
        metrics = {"forward_overlap": float(result.fitness),
                   "forward_rmse_m": float(result.inlier_rmse),
                   "correspondences": len(result.correspondence_set)}
        transform = result.transformation if reliable else initial
        measured_visual = False
        if not uncertain and proposal is not None and not (
                reliable and _visual_witness(views[i], views[j], result.transformation,
                                            engine.settings.camera)[0]):
            measured = _local_match(views[j], views[i], engine.settings.camera, engine.settings,
                                   np.linalg.inv(initial), measured_first=True)
            if measured is not None and _visual_witness(views[j], views[i], measured, engine.settings.camera)[0]:
                transform = np.linalg.inv(measured)
                reliable = True
                report["measured_sequential_edges"] += 1
                measured_visual = True
        if not uncertain and (i, j) in protected_transforms:
            transform = protected_transforms[i, j]
            reliable = True
            measured_visual = True
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
            metrics.update(reverse_overlap=float(reverse.fitness), reverse_rmse_m=float(reverse.inlier_rmse),
                           reciprocal_translation_m=trans, reciprocal_rotation_deg=angle,
                           correction_translation_m=correction, correction_rotation_deg=correction_angle)
            failure = _match_failure(reverse, training[i])
            if failure:
                failure = "reverse_" + failure
            elif trans >= 0.01 or angle >= 2:
                failure = "reciprocal_disagreement"
            elif correction >= (0.75 if proposal is not None else 0.2) or correction_angle >= (30 if proposal is not None else 15):
                failure = "correction_bounds"
        if uncertain and not reliable:
            record(i, j, origin, False, failure or "unreliable_alignment", **metrics)
            return False
        if uncertain:
            transform = result.transformation if reliable else initial
        information = REG.get_information_matrix_from_point_clouds(
            training[i], training[j], 0.03, transform
        )
        if not np.all(np.isfinite(information)):
            if uncertain:
                record(i, j, origin, False, "nonfinite_information", **metrics)
            return False
        if not reliable:
            information *= 0.01  # Unverified odometry is a weak prior.
        if not uncertain and (i, j) in protected_transforms:
            information *= 20  # Distributed measured identities outweigh ICP sliding.
        graph.edges.append(REG.PoseGraphEdge(i, j, transform, information, uncertain))
        constraints.append((i, j))
        if uncertain:
            record(i, j, origin, True, "accepted_raw_geometry", **metrics)
        else:
            report["sequential_results"].append({"source_index": views[i].index,
                "target_index": views[j].index, "reliable": bool(reliable),
                "method": "measured_rgbd" if measured_visual else "measured_geometry" if reliable else "weak_pose_prior",
                "icp_failure": failure})
        return True

    for i in range(len(chosen) - 1):
        if not within_budget():
            return None, report
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
    for (i, j), transform in protected_transforms.items():
        if j == i + 1:
            continue  # Already constrained by the sequential edge.
        information = REG.get_information_matrix_from_point_clouds(training[i], training[j], 0.03, transform)
        if not np.isfinite(information).all():
            report["reason"] = "Invalid measured boundary information"
            return None, report
        graph.edges.append(REG.PoseGraphEdge(i, j, transform, information * 20, False))
        constraints.append((i, j))
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
    report["fragment_loops"] = 0
    report["fragment_candidates_not_selected"] = sum(a not in selected or b not in selected for a, b in hints)
    for a, b in hints:
        if a not in selected or b not in selected or len(seen) >= max_loops:
            continue
        i, j = selected[a], selected[b]
        if j - i <= 1 or (i, j) in protected_transforms:
            continue  # Already measured as a sequential graph edge, not a cycle.
        if not within_budget():
            return None, report
        seen.add((i, j))
        proposal = propose_transform(features[i], features[j], engine.settings.camera)
        # A geometric fragment bridge may have no visual seed. Its saved pose
        # only initializes fresh reciprocal ICP; it cannot bypass any gate.
        accepted = bool(add_edge(i, j, True, proposal, "fragment_support"))
        report["loops"] += accepted
        report["fragment_loops"] += accepted
    for _, i, j, matches in retrieved:
        if (i, j) in seen or len(seen) >= max_loops:
            continue
        if not within_budget():
            return None, report
        seen.add((i, j))
        proposal = propose_transform(
            features[i], features[j], engine.settings.camera, matches
        )
        if proposal is not None:
            accepted = bool(add_edge(i, j, True, proposal, "appearance"))
            report["loops"] += accepted
            report["appearance_loops"] += accepted
        else:
            record(i, j, "appearance", False, "no_verified_rgbd_seed")
    report["geometric_candidates"] = len(candidates)
    for _, i, j in sorted(candidates):
        if len(seen) >= max_loops:
            break
        if (i, j) in seen:
            continue
        seen.add((i, j))
        if not within_budget():
            return None, report
        report["loops"] += bool(add_edge(i, j, True, origin="geometry"))
    if not report["loops"]:
        report["reason"] = "No loop candidates passed raw-measurement checks"
        return None, report
    if not within_budget():
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
    report["surviving_loops"] = sum(bool(edge.uncertain) for edge in graph.edges)
    if not report["surviving_loops"]:
        report["reason"] = "Optimizer pruned all measured loop constraints"
        return None, report
    if not within_budget():
        return None, report
    for old, new in zip(originals, refined):
        translation, angle = motion(np.linalg.inv(old) @ new)
        if not np.all(np.isfinite(new)) or translation > 0.75 or angle > 30:
            report["reason"] = "Optimization exceeded correction bounds"
            return None, report
    before, after = [], []
    for i, j in constraints:
        if not within_budget():
            return None, report
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
        validation_improvement_fraction=float(1 - np.mean(after) / max(np.mean(before), 1e-12)),
        minimum_validation_improvement_fraction=0.02,
        validation_rejected_pairs=[{"source_index": int(poses[chosen[i]][0]),
            "target_index": int(poses[chosen[j]][0]), "before_m2": b, "after_m2": a}
            for (i, j), b, a in zip(constraints, before, after) if a > b * 1.1 + 1e-6],
    )
    if np.mean(after) >= np.mean(before) * 0.98:
        report["reason"] = "Independent geometry improvement below the required 2%"
        return None, report
    if report["validation_rejected_pairs"]:
        report["reason"] = "Optimization worsened an independently measured camera pair"
        return None, report
    # Anchored corrections preserve the first camera and are world-origin invariant.
    proposals = _output_poses(poses, chosen, refined)
    for (_, old), (_, new) in zip(poses, proposals):
        translation, angle = motion(np.linalg.inv(old) @ new)
        if not np.all(np.isfinite(new)) or translation > 0.75 or angle > 30:
            report["reason"] = "Interpolated camera correction exceeded bounds"
            return None, report
    valid, boundary_report = validate_fragment_boundaries(engine, proposals, within_budget)
    report.update(boundary_report)
    if not valid:
        if not report.get("budget_exceeded"):
            report["reason"] = "Refinement conflicts with a retained measured fragment boundary"
        return None, report
    # Interpolation can worsen cameras absent from the graph. Check all of them,
    # excluding measured feature patches, with the same bounded cloud cache as BA.
    output_report = {}
    report["output_depth_validation"] = output_report
    try:
        clouds = _DepthClouds(engine, dict(zip(chosen, features)), started, output_report,
                              budget_seconds=MAX_SECONDS)
        pairs = set((i, i + 1) for i in range(len(poses) - 1))
        pairs.update((int(chosen[i]), int(chosen[j])) for i, j in constraints)
        valid, depth_report = validate_depth(clouds, [p for _, p in poses], [p for _, p in proposals],
            sorted(pairs), started=started, budget_seconds=MAX_SECONDS, minimum_improvement=0.0,
            retain_initial_overlap=True)
        output_report.update(depth_report)
    except _BudgetExceeded:
        report.update(reason="Loop refinement exceeded its time budget during output validation", budget_exceeded=True)
        return None, report
    if not valid or output_report["validation_views"] != len(poses):
        report["reason"] = "Independent depth did not improve consistently across interpolated loop views"
        return None, report
    report["reason"] = "Validated pose proposal; awaiting fresh fusion"
    return proposals, report
