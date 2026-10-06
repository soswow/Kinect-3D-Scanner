"""Bounded offline fragment registration with independent geometric verification.

Unconnected fragments retain local coordinates in the report. They are never
integrated into the world volume. RGB proposals require synchronized measured
RGB-D; geometric proposals use FPFH/RANSAC, reciprocal ICP, held-out points, and
agreement across two different views on each side of a bridge.
"""

import copy
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import open3d as o3d

from shared.calibration import prepare_rgbd
from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS

from .appearance import Features, correspondences, extract_features, propose_transform
from .refinement import _match, motion

REG = o3d.pipelines.registration
MAX_FRAGMENTS = 32
MAX_PAIRS = 256
MAX_CLOUD_POINTS = 12000
MAX_VALIDATION_POINTS = 30000
MAX_FRAGMENT_VIEWS = 16


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
    if len(cloud.points) > MAX_VALIDATION_POINTS:
        cloud = cloud.uniform_down_sample(int(np.ceil(len(cloud.points) / MAX_VALIDATION_POINTS)))
    if len(cloud.points) < 200:
        return None
    train = cloud.select_by_index(list(range(0, len(cloud.points), 2)))
    heldout = cloud.select_by_index(list(range(1, len(cloud.points), 2)))
    train.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    lag = engine.frame_metadata[index].get("rgb_depth_delta_ms")
    features = (extract_features(rgb, depth, engine.settings.camera)
                if lag is None or abs(lag) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS
                else Features(np.empty((0, 2)), np.empty((0, 3)), None))
    return View(index, train, heldout, features)


def _local_match(source, target, camera, settings, initial=None):
    proposal = propose_transform(source.features, target.features, camera)
    seed = proposal if proposal is not None else np.eye(4) if initial is None else initial
    result = _pair(source.train, target.train, seed, 0.45)
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
    limit = MAX_VALIDATION_POINTS if name == "heldout" else MAX_CLOUD_POINTS
    result = result.voxel_down_sample(0.015 if name == "heldout" else 0.02)
    if len(result.points) > limit:
        result = result.uniform_down_sample(int(np.ceil(len(result.points) / limit)))
    return result


def _prepare_fragment(fragment):
    last = len(fragment.views) - 1
    # Keep witnesses near both ends as well as the middle. A sparse midpoint
    # alone can miss the small shared arc between consecutive camera runs.
    chosen = np.unique(np.r_[np.linspace(0, last, min(3, len(fragment.views)), dtype=int),
                             min(2, last), max(0, last - 2)])
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
    pose = initial if result is None else result.transformation
    valid, stats = _heldout(source.heldout, target.heldout, pose)
    if result is None or not valid:
        # Two fragment unions need not overlap by half: each can contain large
        # surfaces unseen by the other. Verify their shared camera observations
        # instead, without weakening reciprocal or held-out thresholds.
        return _verify_partial_bridge(source, target, initial)
    # Independent cameras must support the same fragment transform. A single
    # cube/floor coincidence cannot authorize an entire disconnected segment.
    support = []
    source_poses = {view.index: view.pose for view in source.keys}
    target_poses = {view.index: view.pose for view in target.keys}
    enough = False
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
                enough = any(
                    c != a.index and d != b.index
                    and _disagrees(source_poses[a.index], source_poses[c], 0.02, 2)
                    and _disagrees(target_poses[b.index], target_poses[d], 0.02, 2)
                    for c, d in support)
                if enough:
                    break
        if enough:
            break
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
            "information": information, "support": independent, "validation": stats,
            "validation_scope": "fragment unions and independent camera pairs"}


def _independent_pairs(support, source, target):
    a_poses = {v.index: v.pose for v in source.keys}
    b_poses = {v.index: v.pose for v in target.keys}
    return [(a, b) for a, b in support if any(
        a != c and b != d and _disagrees(a_poses[a], a_poses[c], 0.02, 2)
        and _disagrees(b_poses[b], b_poses[d], 0.02, 2) for c, d in support)]


def _has_independent_views(fragment):
    return any(_disagrees(a.pose, b.pose, 0.02, 2)
               for i, a in enumerate(fragment.keys) for b in fragment.keys[i + 1:])


def _verify_partial_bridge(source, target, initial):
    coarse = REG.evaluate_registration(source.train, target.train, 0.08, initial)
    if coarse.fitness < 0.25:
        return None
    evidence = []
    clusters = []
    for a in source.keys:
        for b in target.keys:
            seed = np.linalg.inv(b.pose) @ initial @ a.pose
            if REG.evaluate_registration(a.train, b.train, 0.08, seed).fitness < 0.35:
                continue
            matched = _pair(a.train, b.train, seed)
            if matched is None:
                continue
            valid, stats = _heldout(a.heldout, b.heldout, matched.transformation)
            if valid:
                evidence.append((a, b, b.pose @ matched.transformation @ np.linalg.inv(a.pose), stats))
                for _, _, pose, _ in evidence:
                    group = [(c, d, p, st) for c, d, p, st in evidence
                             if not _disagrees(pose, p, 0.03, 3)]
                    support = _independent_pairs([(c.index, d.index) for c, d, _, _ in group], source, target)
                    if len(support) >= 2:
                        clusters.append((pose, support))
                if clusters:
                    break
        if clusters:
            break
    if not clusters or any(_disagrees(clusters[0][0], p) for p, _ in clusters[1:]):
        return None
    pose, support = clusters[0]
    # Validate the single chosen transform on every supporting held-out view,
    # not merely each pair's separately optimized transform.
    a_views = {v.index: v for v in source.keys}
    b_views = {v.index: v for v in target.keys}
    validation = []
    for a, b in support:
        transform = np.linalg.inv(b_views[b].pose) @ pose @ a_views[a].pose
        valid, stats = _heldout(a_views[a].heldout, b_views[b].heldout, transform)
        if not valid:
            return None
        validation.append(stats)
    information = REG.get_information_matrix_from_point_clouds(source.train, target.train, 0.03, pose)
    if not np.isfinite(information).all():
        return None
    return {"source": source.index, "target": target.index, "transform": pose,
            "information": information, "support": support,
            "validation": {"camera_pairs": validation}, "validation_scope": "independent camera pairs"}


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


def capture_gap_limit(metadata):
    """Distinguish a pause from deliberately slow or backpressured capture."""
    intervals = [b["timestamp_s"] - a["timestamp_s"] for a, b in pairwise(metadata)
                 if a.get("timestamp_s") is not None and b.get("timestamp_s") is not None
                 and b["timestamp_s"] > a["timestamp_s"]]
    return max(2.0, 3 * float(np.median(intervals))) if len(intervals) >= 4 else 2.0


def propose_fragment_poses(engine, progress_cb=None):
    """Re-estimate local maps; only measured bridges connect them to the first map.

    Live poses seed nearby registration, but never authorize a graph edge or fix
    a later fragment in world coordinates. In particular, accepted views after
    a capture gap can contain drift just as rejected observations can.
    """
    report = {"applied": False, "reason": "No verified fragment bridges", "fragments": [],
              "verified_bridges": [], "ambiguous_pairs": [], "pair_budget": MAX_PAIRS,
              "fragment_limit": MAX_FRAGMENTS, "budget_limited": False,
              "invalid_indices": [], "unassigned_indices": [], "recovered_frames": 0,
              "corrected_frames": 0, "excluded_frames": [],
              "validation": "reciprocal ICP; independent held-out points; multiple camera pairs"}
    baseline = {i: p.copy() for i, p in engine.poses}
    if not baseline or (len(engine.raw_frames) < 2 and not getattr(engine, "_pose_seeds_only", False)):
        report["reason"] = "No skipped views or trajectory to revalidate"
        return None, report
    fragments = []
    sequential = []
    gap_limit = capture_gap_limit(engine.frame_metadata)
    report["capture_gap_limit_s"] = gap_limit
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
            stamp is not None and last_stamp is not None and (stamp - last_stamp > gap_limit or stamp <= last_stamp)
        )
        relative = None
        if not gap:
            initial = (np.linalg.inv(baseline[previous.index]) @ baseline[index]
                       if index in baseline and previous.index in baseline else None)
            relative = _local_match(view, previous, engine.settings.camera, engine.settings, initial)
        full = current is not None and len(current.views) >= MAX_FRAGMENT_VIEWS
        if gap or relative is None or full:
            if len(fragments) == MAX_FRAGMENTS:
                report["budget_limited"] = True
                report["unassigned_indices"].append(index)
                current, previous = None, None
                continue
            if not gap and relative is not None:
                # A size limit is not a tracking loss. Preserve the same raw
                # frame-to-frame measurement that would join these observations
                # inside a fragment. Live/world pose guesses never create edges.
                sequential.append((current.index, len(fragments), previous, view,
                                   previous.pose @ relative, relative))
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
    # A single gauge anchor. Other accepted fragments are estimates to verify,
    # not additional roots joined by fabricated certain constraints.
    roots = [next((f.index for f in fragments if f.seed is not None), -1)]
    roots = [i for i in roots if i >= 0]
    if not roots:
        report["reason"] = "No retained live anchor in the bounded fragment search"
        return None, report
    edges = []
    for source, target, a, b, local_pose, relative in sequential:
        transform = np.linalg.inv(local_pose)
        valid, stats = _heldout(b.heldout, a.heldout, relative, 0.4)
        information = REG.get_information_matrix_from_point_clouds(
            fragments[source].train, fragments[target].train, 0.03, transform)
        if valid and np.isfinite(information).all():
            edges.append({"source": source, "target": target, "transform": transform,
                          "information": information, "support": [(a.index, b.index)],
                          "validation": stats, "validation_scope": "sequential camera pair"})
    report["sequential_bridges"] = len(edges)
    sequential_pairs = {(e["source"], e["target"]) for e in edges}
    candidates = []
    eligible = {f.index for f in fragments if _has_independent_views(f)}
    for i, source in enumerate(fragments):
        for target in fragments[i + 1:]:
            if (source.index, target.index) in sequential_pairs:
                continue  # The measured boundary already supplies this edge.
            if source.index not in eligible or target.index not in eligible:
                continue  # These pairs cannot satisfy the existing witness rule.
            proposals = []
            if source.seed is not None and target.seed is not None:
                proposals.append((0, np.linalg.inv(target.seed) @ source.seed))
            # Returning camera coordinates can overlap despite a badly drifted
            # world trajectory. This seed must pass exactly the same verification.
            proposals.append((0, np.eye(4)))
            for a in source.keys:
                for b in target.keys:
                    matches = correspondences(a.features, b.features)
                    proposal = propose_transform(a.features, b.features, engine.settings.camera, matches)
                    if proposal is not None:
                        proposals.append((len(matches), b.pose @ proposal @ np.linalg.inv(a.pose)))
            distance = np.linalg.norm(np.mean(source.fpfh.data, axis=1) - np.mean(target.fpfh.data, axis=1))
            candidates.append((-max([n for n, _ in proposals], default=0),
                               int(source.index not in roots and target.index not in roots
                                   and target.index != source.index + 1),
                               distance, source, target, proposals))
    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3].index, row[4].index))
    report["candidate_pairs"] = len(candidates)
    report["tested_pairs"] = 0
    pending = list(candidates)
    while pending and report["tested_pairs"] < MAX_PAIRS:
        connected_now = _reachable(len(fragments), edges, roots)
        # Expand the anchored map first. A detached-to-detached match cannot
        # connect to it after every crossing pair has failed. Avoid spending the
        # bounded search on components that will never enter the reconstruction.
        frontier = next((i for i, row in enumerate(pending)
                         if (row[3].index in connected_now) != (row[4].index in connected_now)), None)
        if frontier is None:
            frontier = next((i for i, row in enumerate(pending)
                             if row[3].index in connected_now and row[4].index in connected_now), None)
        if frontier is None:
            report["unreachable_candidate_pairs"] = len(pending)
            break
        _, _, _, source, target, appearances = pending.pop(frontier)
        number = report["tested_pairs"] + 1
        report["tested_pairs"] = number
        _notify(progress_cb, number, min(len(candidates), MAX_PAIRS),
                f"Verifying fragment links {number}/{min(len(candidates), MAX_PAIRS)}")
        proposals = [p for _, p in sorted(appearances, key=lambda row: -row[0])[:4]]
        for seed in (source.index * 100 + target.index, source.index * 100 + target.index + 10000):
            proposal = _global_seed(source, target, seed)
            if proposal is not None:
                proposals.append(proposal)
        verified = []
        unique = []
        for proposal in proposals:
            if any(not _disagrees(proposal, previous, 0.01, 1) for previous in unique):
                continue
            unique.append(proposal)
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
        # Once every eligible map is joined and a redundant cycle has been
        # measured, finish with full optimized-geometry validation. Each tested
        # pair has already exhausted its competing independent proposals.
        if eligible.issubset(_reachable(len(fragments), edges, roots)) and len(edges) >= len(eligible):
            report["search_stopped_connected"] = True
            break
    connected_now = _reachable(len(fragments), edges, roots)
    report["budget_limited"] |= report["tested_pairs"] == MAX_PAIRS and any(
        row[3].index in connected_now or row[4].index in connected_now for row in pending)
    connected = _reachable(len(fragments), edges, roots)
    # Initialize from measured bridges, independently of the drifted live poses.
    primary = roots[0]
    world = {i: fragments[i].seed.copy() for i in roots}
    tree_pairs = set()
    for _ in fragments:
        for edge in edges:
            a, b, pose = edge["source"], edge["target"], edge["transform"]
            if a in world and b not in world:
                world[b] = world[a] @ np.linalg.inv(pose)
                tree_pairs.add((a, b))
            elif b in world and a not in world:
                world[a] = world[b] @ pose
                tree_pairs.add((a, b))
    node_ids = sorted(connected)
    nodes = {i: j for j, i in enumerate(node_ids)}
    graph = REG.PoseGraph()
    for i in node_ids:
        graph.nodes.append(REG.PoseGraphNode(world[i].copy()))
    for edge in edges:
        if edge["source"] in connected and edge["target"] in connected:
            graph.edges.append(REG.PoseGraphEdge(nodes[edge["source"]], nodes[edge["target"]],
                               edge["transform"], edge["information"],
                               (edge["source"], edge["target"]) not in tree_pairs))
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
        if edge.get("validation_scope") in ("independent camera pairs", "sequential camera pair"):
            a_views = {v.index: v for v in fragments[a].views}
            b_views = {v.index: v for v in fragments[b].views}
            minimum = 0.4 if edge["validation_scope"] == "sequential camera pair" else 0.5
            valid = all(_heldout(a_views[i].heldout, b_views[j].heldout,
                np.linalg.inv(b_views[j].pose) @ transform @ a_views[i].pose, minimum)[0]
                for i, j in edge["support"])
        else:
            valid, _ = _heldout(fragments[a].heldout, fragments[b].heldout, transform)
        valid_graph &= valid and translation <= 0.03 and angle <= 3
    if not valid_graph:
        report["reason"] = "Optimized bridges failed independent validation"
        connected, optimized = set(roots), {i: fragments[i].seed for i in roots}
    proposals = {}
    for fragment in fragments:
        transform = optimized.get(fragment.index)
        if transform is not None and valid_graph:
            for view in fragment.views:
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
    report["recovered_frames"] = len(set(proposals) - set(baseline))
    report["excluded_frames"] = sorted(set(baseline) - set(proposals))
    report["corrected_frames"] = sum(
        _disagrees(baseline[i], pose, 0.003, 0.3)
        for i, pose in proposals.items() if i in baseline)
    if not valid_graph:
        report["failed"] = True
        return None, report
    if (len(connected) == 1 and not report["recovered_frames"] and not report["excluded_frames"]
            and not getattr(engine, "_pose_seeds_only", False)):
        report["reason"] = "No independent fragment constraints require a trajectory change"
        return None, report
    if (not (report["recovered_frames"] or report["corrected_frames"] or report["excluded_frames"])
            and not getattr(engine, "_pose_seeds_only", False)):
        report["reason"] = "No changes required by verified fragment geometry"
        return None, report
    report["reason"] = "Verified connected fragments; awaiting fresh fusion"
    return sorted(proposals.items()), report
