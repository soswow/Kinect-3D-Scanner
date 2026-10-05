"""Bounded offline fragment registration with independent geometric verification.

Unconnected fragments retain local coordinates in the report. They are never
integrated into the world volume. RGB proposals require synchronized measured
RGB-D; geometric proposals use FPFH/RANSAC, reciprocal ICP, held-out points, and
agreement across two different views on each side of a bridge.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from shared.calibration import prepare_rgbd

from .appearance import Features, correspondences, extract_features, propose_transform
from .refinement import _match, motion

REG = o3d.pipelines.registration
MAX_FRAGMENTS = 32
MAX_PAIRS = 96
MAX_CLOUD_POINTS = 6000


@dataclass
class View:
    index: int
    train: object
    heldout: object
    features: Features
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class Fragment:
    index: int
    views: list = field(default_factory=list)
    seed: np.ndarray | None = None
    keys: list = field(default_factory=list)
    train: object = None
    heldout: object = None
    coarse: object = None
    fpfh: object = None


def _rigid(pose):
    return (
        np.asarray(pose).shape == (4, 4)
        and np.isfinite(pose).all()
        and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
        and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
        and abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-4
    )


def _disagrees(a, b, translation_limit=0.05, angle_limit=5):
    translation, angle = motion(np.linalg.inv(a) @ b)
    return translation > translation_limit or angle > angle_limit


def _strong(result, target, fitness=0.5):
    # Training clouds contain alternating 15 mm voxel samples: nearest-neighbour
    # RMSE includes their spacing. The denser held-out check stays at 15 mm.
    if not _rigid(result.transformation) or result.fitness < fitness or result.inlier_rmse > 0.02:
        return False
    pairs = np.asarray(result.correspondence_set)
    if len(pairs) < 100 or not target.has_normals():
        return False
    normals = np.asarray(target.normals)[pairs[:, 1]]
    return np.linalg.eigvalsh(normals.T @ normals / len(normals))[0] >= 0.002


def _heldout(source, target, pose, minimum=0.5):
    if not len(source.points) or not len(target.points) or not _rigid(pose):
        return False, {}
    forward = REG.evaluate_registration(source, target, 0.025, pose)
    reverse = REG.evaluate_registration(target, source, 0.025, np.linalg.inv(pose))
    stats = {"forward_overlap": forward.fitness, "reverse_overlap": reverse.fitness,
             "forward_rmse_m": forward.inlier_rmse, "reverse_rmse_m": reverse.inlier_rmse}
    return (min(forward.fitness, reverse.fitness) >= minimum
            and max(forward.inlier_rmse, reverse.inlier_rmse) <= 0.015), stats


def _pair(source, target, initial, minimum=0.5):
    forward = _match(source, target, initial)
    if not _strong(forward, target, minimum):
        return None
    reverse = _match(target, source, np.linalg.inv(forward.transformation))
    translation, angle = motion(reverse.transformation @ forward.transformation)
    if not _strong(reverse, source, minimum) or translation > 0.01 or angle > 2:
        return None
    return forward


def _notify(progress, current, total, message):
    if progress:
        progress(current, total, {"stage": "fragment_reconnection", "message": message})


def _view(engine, index):
    rgb, depth = prepare_rgbd(*engine.raw_frames[index], engine.settings)
    if np.count_nonzero(depth) < 1000:
        return None
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        engine._make_rgbd(rgb, depth), engine.intrinsic
    ).voxel_down_sample(0.015)
    if len(cloud.points) > 10000:
        cloud = cloud.uniform_down_sample(int(np.ceil(len(cloud.points) / 10000)))
    if len(cloud.points) < 200:
        return None
    train = cloud.select_by_index(list(range(0, len(cloud.points), 2)))
    heldout = cloud.select_by_index(list(range(1, len(cloud.points), 2)))
    train.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    lag = engine.frame_metadata[index].get("rgb_depth_delta_ms")
    features = (extract_features(rgb, depth, engine.settings.camera)
                if lag is None or abs(lag) <= 20 else Features(np.empty((0, 2)), np.empty((0, 3)), None))
    return View(index, train, heldout, features)


def _local_match(source, target, camera, settings):
    proposal = propose_transform(source.features, target.features, camera)
    result = _pair(source.train, target.train, np.eye(4) if proposal is None else proposal, 0.45)
    if result is None:
        return None
    translation, angle = motion(result.transformation)
    if translation > settings.max_translation_m or angle > settings.max_rotation_deg:
        return None
    valid, _ = _heldout(source.heldout, target.heldout, result.transformation, 0.4)
    return result.transformation if valid else None


def _aggregate(fragment, name):
    result = o3d.geometry.PointCloud()
    for view in fragment.keys:
        result += copy.deepcopy(getattr(view, name)).transform(view.pose)
    result = result.voxel_down_sample(0.02)
    if len(result.points) > MAX_CLOUD_POINTS:
        result = result.uniform_down_sample(int(np.ceil(len(result.points) / MAX_CLOUD_POINTS)))
    return result


def _prepare_fragment(fragment):
    chosen = np.unique(np.linspace(0, len(fragment.views) - 1, min(3, len(fragment.views)), dtype=int))
    fragment.keys = [fragment.views[i] for i in chosen]
    fragment.train = _aggregate(fragment, "train")
    fragment.heldout = _aggregate(fragment, "heldout")
    fragment.train.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    fragment.coarse = fragment.train.voxel_down_sample(0.04)
    fragment.coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    fragment.fpfh = REG.compute_fpfh_feature(
        fragment.coarse, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100)
    )


def _global_seed(source, target, seed):
    # Finite iteration budget, deterministic proposals; Finish runs exclusively.
    o3d.utility.random.seed(seed)
    result = REG.registration_ransac_based_on_feature_matching(
        source.coarse, target.coarse, source.fpfh, target.fpfh, True, 0.08,
        REG.TransformationEstimationPointToPoint(False), 3,
        [REG.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         REG.CorrespondenceCheckerBasedOnDistance(0.08)],
        REG.RANSACConvergenceCriteria(12000, 0.999),
    )
    return result.transformation if result.fitness >= 0.25 and _rigid(result.transformation) else None


def _verify_bridge(source, target, initial):
    result = _pair(source.train, target.train, initial)
    if result is None:
        return None
    pose = result.transformation
    valid, stats = _heldout(source.heldout, target.heldout, pose)
    if not valid:
        return None
    # Independent cameras must support the same fragment transform. A single
    # cube/floor coincidence cannot authorize an entire disconnected segment.
    support = []
    for a in source.keys:
        for b in target.keys:
            seed = np.linalg.inv(b.pose) @ pose @ a.pose
            matched = _pair(a.train, b.train, seed, 0.45)
            if matched is None:
                continue
            translation, angle = motion(np.linalg.inv(seed) @ matched.transformation)
            valid, _ = _heldout(a.heldout, b.heldout, matched.transformation, 0.45)
            if valid and translation <= 0.03 and angle <= 3:
                support.append((a.index, b.index))
    source_poses = {view.index: view.pose for view in source.keys}
    target_poses = {view.index: view.pose for view in target.keys}
    independent = [(a, b) for a, b in support if any(
        a != c and b != d and _disagrees(source_poses[a], source_poses[c], 0.02, 2)
        and _disagrees(target_poses[b], target_poses[d], 0.02, 2)
        for c, d in support)]
    if len(independent) < 2:
        return None
    information = REG.get_information_matrix_from_point_clouds(source.train, target.train, 0.03, pose)
    if not np.isfinite(information).all():
        return None
    return {"source": source.index, "target": target.index, "transform": pose,
            "information": information, "support": independent, "validation": stats}


def _reachable(count, edges, roots):
    connected = set(roots)
    for _ in range(count):
        before = len(connected)
        for edge in edges:
            a, b = edge["source"], edge["target"]
            if a in connected or b in connected:
                connected.update((a, b))
        if len(connected) == before:
            break
    return connected


def propose_fragment_poses(engine, progress_cb=None):
    """Pure proposal: preserve live poses and expose all unconnected local maps."""
    report = {"applied": False, "reason": "No verified fragment bridges", "fragments": [],
              "verified_bridges": [], "ambiguous_pairs": [], "pair_budget": MAX_PAIRS,
              "fragment_limit": MAX_FRAGMENTS, "budget_limited": False,
              "invalid_indices": [], "unassigned_indices": [], "recovered_frames": 0,
              "validation": "reciprocal ICP; independent held-out points; multiple camera pairs"}
    baseline = {i: p.copy() for i, p in engine.poses}
    if not baseline or not any(not d.get("success") for d in engine.diagnostics):
        report["reason"] = "No skipped views to reconnect"
        return None, report
    fragments = []
    current, previous = None, None
    total = len(engine.raw_frames)
    for index in range(total):
        _notify(progress_cb, index + 1, total, f"Recovering fragments: preparing view {index + 1}/{total}")
        view = _view(engine, index)
        if view is None:
            report["invalid_indices"].append(index)
            current, previous = None, None
            continue
        stamp = engine.frame_metadata[index].get("timestamp_s")
        last_stamp = engine.frame_metadata[previous.index].get("timestamp_s") if previous else None
        gap = previous is None or index != previous.index + 1 or (
            stamp is not None and last_stamp is not None and (stamp - last_stamp > 2 or stamp <= last_stamp)
        )
        relative = None
        if not gap:
            if index in baseline and previous.index in baseline:
                relative = np.linalg.inv(baseline[previous.index]) @ baseline[index]
            else:
                relative = _local_match(view, previous, engine.settings.camera, engine.settings)
            if relative is not None and index in baseline and current.seed is not None:
                predicted = current.seed @ previous.pose @ relative
                if _disagrees(predicted, baseline[index], 0.03, 3):
                    relative = None
        if gap or relative is None:
            if len(fragments) == MAX_FRAGMENTS:
                report["budget_limited"] = True
                report["unassigned_indices"].append(index)
                current, previous = None, None
                continue
            current = Fragment(len(fragments))
            fragments.append(current)
        else:
            view.pose = previous.pose @ relative
        current.views.append(view)
        if index in baseline and current.seed is None:
            current.seed = baseline[index] @ np.linalg.inv(view.pose)
        previous = view
    if not fragments:
        return None, report
    for fragment in fragments:
        _prepare_fragment(fragment)
    roots = [f.index for f in fragments if f.seed is not None]
    if not roots:
        report["reason"] = "No retained live anchor in the bounded fragment search"
        return None, report
    edges = []
    candidates = []
    for i, source in enumerate(fragments):
        for target in fragments[i + 1:]:
            if source.index in roots and target.index in roots:
                continue
            proposals = []
            for a in source.keys:
                for b in target.keys:
                    matches = correspondences(a.features, b.features)
                    proposal = propose_transform(a.features, b.features, engine.settings.camera, matches)
                    if proposal is not None:
                        proposals.append((len(matches), b.pose @ proposal @ np.linalg.inv(a.pose)))
            distance = np.linalg.norm(np.mean(source.fpfh.data, axis=1) - np.mean(target.fpfh.data, axis=1))
            candidates.append((-max([n for n, _ in proposals], default=0),
                               int(source.index not in roots and target.index not in roots),
                               distance, source, target, proposals))
    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3].index, row[4].index))
    report["candidate_pairs"] = len(candidates)
    report["tested_pairs"] = min(len(candidates), MAX_PAIRS)
    report["budget_limited"] |= len(candidates) > MAX_PAIRS
    for number, (_, _, _, source, target, appearances) in enumerate(candidates[:MAX_PAIRS], 1):
        _notify(progress_cb, number, min(len(candidates), MAX_PAIRS),
                f"Verifying fragment links {number}/{min(len(candidates), MAX_PAIRS)}")
        proposals = [p for _, p in sorted(appearances, key=lambda row: -row[0])[:2]]
        for seed in (source.index * 100 + target.index, source.index * 100 + target.index + 10000):
            proposal = _global_seed(source, target, seed)
            if proposal is not None:
                proposals.append(proposal)
        verified = []
        for proposal in proposals:
            bridge = _verify_bridge(source, target, proposal)
            if bridge is not None:
                verified.append(bridge)
        if not verified:
            continue
        best = verified[0]
        if any(_disagrees(best["transform"], bridge["transform"]) for bridge in verified[1:]):
            report["ambiguous_pairs"].append([source.index, target.index])
            continue
        edges.append(best)
    connected = _reachable(len(fragments), edges, roots)
    # All known live fragments have verified world coordinates already. Keep
    # those priors strong and optimize only the component anchored to the scan.
    primary = roots[0]
    world = {i: fragments[i].seed.copy() for i in roots}
    for _ in fragments:
        for edge in edges:
            a, b, pose = edge["source"], edge["target"], edge["transform"]
            if a in world and b not in world:
                world[b] = world[a] @ np.linalg.inv(pose)
            elif b in world and a not in world:
                world[a] = world[b] @ pose
    node_ids = sorted(connected)
    nodes = {i: j for j, i in enumerate(node_ids)}
    graph = REG.PoseGraph()
    for i in node_ids:
        graph.nodes.append(REG.PoseGraphNode(world[i].copy()))
    for i in roots:
        if i != primary:
            graph.edges.append(REG.PoseGraphEdge(nodes[primary], nodes[i],
                               np.linalg.inv(world[i]) @ world[primary], np.eye(6) * 1e6, False))
    for edge in edges:
        if edge["source"] in connected and edge["target"] in connected:
            graph.edges.append(REG.PoseGraphEdge(nodes[edge["source"]], nodes[edge["target"]],
                               edge["transform"], edge["information"], True))
    if edges and len(graph.nodes) > 1:
        _notify(progress_cb, 0, 1, "Optimizing verified fragment pose graph")
        REG.global_optimization(graph, REG.GlobalOptimizationLevenbergMarquardt(),
            REG.GlobalOptimizationConvergenceCriteria(), REG.GlobalOptimizationOption(
                max_correspondence_distance=0.03, edge_prune_threshold=0.25, reference_node=nodes[primary]))
    surviving = [{"source": node_ids[e.source_node_id], "target": node_ids[e.target_node_id]}
                 for e in graph.edges if not e.uncertain or e.confidence >= 0.25]
    connected = _reachable(len(fragments), surviving, roots)
    surviving_pairs = {(e["source"], e["target"]) for e in surviving}
    optimized = {i: graph.nodes[nodes[i]].pose.copy() for i in connected}
    valid_graph = all(_rigid(p) for p in optimized.values())
    for edge in edges:
        a, b = edge["source"], edge["target"]
        if a not in connected or b not in connected or (a, b) not in surviving_pairs:
            continue
        transform = np.linalg.inv(optimized[b]) @ optimized[a]
        translation, angle = motion(np.linalg.inv(edge["transform"]) @ transform)
        valid, _ = _heldout(fragments[a].heldout, fragments[b].heldout, transform)
        valid_graph &= valid and translation <= 0.03 and angle <= 3
    if not valid_graph:
        report["reason"] = "Optimized bridges failed independent validation"
        connected, optimized = set(roots), {i: fragments[i].seed for i in roots}
    proposals = dict(baseline)
    for fragment in fragments:
        transform = optimized.get(fragment.index)
        if transform is not None and valid_graph:
            for view in fragment.views:
                if view.index not in proposals:
                    proposals[view.index] = transform @ view.pose
        report["fragments"].append({
            "id": fragment.index, "connected": fragment.index in connected,
            "frame_indices": [v.index for v in fragment.views],
            "camera_to_fragment": [{"index": v.index, "pose": v.pose.tolist()} for v in fragment.views],
            "fragment_to_world": transform.tolist() if transform is not None else None,
        })
    report["connected_fragments"] = sorted(connected)
    report["unconnected_fragments"] = [f.index for f in fragments if f.index not in connected]
    report["verified_bridges"] = [
        {**{k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in e.items() if k != "information"},
         "connected_to_scan": valid_graph and e["source"] in connected and e["target"] in connected
                              and (e["source"], e["target"]) in surviving_pairs}
        for e in edges]
    report["recovered_frames"] = len(proposals) - len(baseline)
    if not report["recovered_frames"]:
        if valid_graph:
            report["reason"] = "No skipped views could be connected with verified geometry"
        return None, report
    report["reason"] = "Verified connected fragments; awaiting fresh fusion"
    return sorted(proposals.items()), report
