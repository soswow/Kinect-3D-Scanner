"""Bounded offline fragment registration with independent geometric verification.

Unconnected fragments retain local coordinates in the report. They are never
integrated into the world volume. RGB proposals require synchronized measured
RGB-D; geometric proposals use FPFH/RANSAC, reciprocal ICP, held-out points, and
agreement across two different views on each side of a bridge.
"""

import copy
from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np
import open3d as o3d

from shared.calibration import prepare_rgbd
from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS
from shared.visual_tracking import feature_agreement

from .appearance import Features, correspondences, extract_features, propose_transform
from .refinement import _match, motion

REG = o3d.pipelines.registration
MAX_FRAGMENTS = 32
MAX_PAIRS = 256
MAX_CLOUD_POINTS = 12000
MAX_VALIDATION_POINTS = 30000
MAX_FRAGMENT_VIEWS = 16
MAX_MATCH_CACHE = 16


@dataclass
class View:
    index: int
    train: object
    heldout: object
    features: Features
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    match_cache: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass
class Fragment:
    index: int
    views: list = field(default_factory=list)
    context: list = field(default_factory=list)
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
    prepare = getattr(engine, "_prepare_input", None)
    rgb, depth = (prepare_rgbd(*engine.raw_frames[index], engine.settings)
                  if prepare is None else prepare(*engine.raw_frames[index],
                      engine.settings, cpu_prepare=prepare_rgbd))
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
    features = (extract_features(rgb, depth, engine.settings.camera, method="sift")
                if lag is None or abs(lag) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS
                else Features(np.empty((0, 2)), np.empty((0, 3)), None))
    return View(index, train, heldout, features)


def _matches(source, target):
    """Reuse exact descriptor matches across proposals and overlapping maps.

    Features are immutable for a prepared view. Keep the actual feature object
    in each entry so reusing a frame index with new features cannot hit a stale
    entry. Context views share this cache, but never cache pose-dependent checks.
    """
    cached = source.match_cache.get(target.index)
    if cached is not None and cached[0] is source.features and cached[1] is target.features:
        source.match_cache.pop(target.index)
        source.match_cache[target.index] = cached
        return cached[2]
    matches = correspondences(source.features, target.features)
    return _cache_matches(source, target, matches)


def _cache_matches(source, target, matches):
    """Install both exact directions using the existing bounded cache policy."""
    reverse = matches[:, ::-1].copy()
    reverse = reverse[np.argsort(reverse[:, 0], kind="stable")]
    matches.flags.writeable = reverse.flags.writeable = False
    for view, other, value in ((source, target, matches), (target, source, reverse)):
        view.match_cache.pop(other.index, None)
        view.match_cache[other.index] = (view.features, other.features, value)
        if len(view.match_cache) > MAX_MATCH_CACHE:
            view.match_cache.pop(next(iter(view.match_cache)))
    return matches


def _prepare_matches(engine, sources, targets):
    """Batch immutable descriptor retrieval; leave all pose checks uncached."""
    if not targets or getattr(engine, "backend", {}).get("descriptor_matching", {}).get("implementation") != "cuda":
        return
    from .cuda_matching import Matcher

    if engine._cuda_final_descriptor_matcher is None:
        engine._cuda_final_descriptor_matcher = Matcher(int(str(engine.device).split(":")[1]))
    for source in sources:
        matched = engine._cuda_final_descriptor_matcher.match(source.features, [v.features for v in targets])
        if matched is not None:
            engine.backend["descriptor_matching"]["final_cuda_batches"] += 1
            for target, matches in zip(targets, matched):
                _cache_matches(source, target, matches)


def _local_match(source, target, camera, settings, initial=None, *, measured_first=False):
    matches = _matches(source, target)
    proposal = propose_transform(source.features, target.features, camera, matches)
    if measured_first and proposal is not None:
        from .visual_refinement import measured_pose

        relative = measured_pose(source.features, target.features, matches, proposal, camera)
        if relative is not None and _visual_witness(source, target, relative, camera, matches)[0]:
            translation, angle = motion(relative)
            steps = max(1, min(3, source.index - target.index))
            if (translation <= settings.max_translation_m * steps and angle <= settings.max_rotation_deg * steps
                    and _heldout(source.heldout, target.heldout, relative, .4)[0]):
                return relative
    seed = proposal if proposal is not None else np.eye(4) if initial is None else initial
    result = _pair(source.train, target.train, seed, 0.45)
    relative = result.transformation if result is not None else None
    if proposal is not None:
        matches = _matches(source, target)
        if relative is None or not _visual_witness(source, target, relative, camera, matches)[0]:
            # A textured plane can constrain motion even when its normals do
            # not. Keep feature identities and validate on held-out depth.
            relative = proposal if _visual_witness(source, target, proposal, camera, matches)[0] else None
    if relative is None:
        return None
    translation, angle = motion(relative)
    # A direct link over a rejected capture spans several capture steps. Keep
    # the same per-step motion budget, bounded to three steps, while requiring
    # the identical reciprocal, color and held-out geometric evidence.
    steps = max(1, min(3, source.index - target.index))
    if translation > settings.max_translation_m * steps or angle > settings.max_rotation_deg * steps:
        return None
    valid, _ = _heldout(source.heldout, target.heldout, relative, 0.4)
    return relative if valid else None


def _visual_witness(source, target, pose, camera, matches=None):
    matches = _matches(source, target) if matches is None else matches
    if len(matches) < 40 or not _rigid(pose):
        return False, {}
    a, b = matches.T
    valid, features = feature_agreement(source.features.points[a], target.features.points[b],
                                       target.features.pixels[b], pose, camera)
    if not valid:
        return False, {}
    valid, geometry = _heldout(source.heldout, target.heldout, pose, 0.45)
    return valid, {"features": features, **geometry}


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
    # Keep the first and last five witnesses, plus the midpoint. Narrow shared
    # arcs and a single blurred boundary view must not vanish in a sparse
    # three-view sample. Aggregated point counts remain independently bounded.
    chosen = np.unique(np.r_[np.arange(min(5, last + 1)),
                             np.arange(max(0, last - 4), last + 1), last // 2])
    fragment.keys = fragment.context + [fragment.views[i] for i in chosen]
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


def _verify_bridge(source, target, initial, camera=None, *, visual_first=False):
    if visual_first and camera is not None:
        visual = _verify_visual_bridge(source, target, initial, camera)
        if visual is not None:
            return visual
    result = _pair(source.train, target.train, initial)
    pose = initial if result is None else result.transformation
    valid, stats = _heldout(source.heldout, target.heldout, pose)
    if result is None or not valid:
        # Two fragment unions need not overlap by half: each can contain large
        # surfaces unseen by the other. Verify their shared camera observations
        # instead, without weakening reciprocal or held-out thresholds.
        geometric = _verify_partial_bridge(source, target, initial)
        return geometric if geometric is not None or camera is None or visual_first else _verify_visual_bridge(
            source, target, initial, camera)
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
        return _verify_visual_bridge(source, target, initial, camera) if camera is not None and not visual_first else None
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


def _verify_visual_bridge(source, target, initial, camera):
    """Two independent RGB-D witnesses can constrain otherwise planar overlap."""
    if not _rigid(initial):
        return None
    support = []
    witnesses = {}
    for a in source.keys:
        for b in target.keys:
            pose = np.linalg.inv(b.pose) @ initial @ a.pose
            valid, stats = _visual_witness(a, b, pose, camera)
            if valid:
                support.append((a.index, b.index))
                witnesses[(a.index, b.index)] = stats
    support = _independent_pairs(support, source, target)
    if len(support) < 2:
        return None
    information = REG.get_information_matrix_from_point_clouds(source.train, target.train, 0.03, initial)
    if not np.isfinite(information).all():
        return None
    return {"source": source.index, "target": target.index, "transform": initial,
            "information": information, "support": support,
            "validation": {"camera_pairs": [witnesses[p] for p in support]},
            "validation_scope": "visual and held-out camera pairs"}


def _validate_bridge_pose(edge, poses, fragments, camera):
    """Recheck a bridge in a proposed world map, including its visual identity."""
    a, b = edge["source"], edge["target"]
    if a not in poses or b not in poses or not (_rigid(poses[a]) and _rigid(poses[b])):
        return False
    transform = np.linalg.inv(poses[b]) @ poses[a]
    translation, angle = motion(np.linalg.inv(edge["transform"]) @ transform)
    if translation > 0.03 or angle > 3:
        return False
    scope = edge.get("validation_scope")
    if scope in ("independent camera pairs", "sequential camera pair", "temporal camera pair", "visual and held-out camera pairs"):
        a_views = {v.index: v for v in fragments[a].context + fragments[a].views}
        b_views = {v.index: v for v in fragments[b].context + fragments[b].views}
        minimum = {"sequential camera pair": 0.4, "temporal camera pair": 0.45}.get(scope, 0.5)
        for i, j in edge["support"]:
            relative = np.linalg.inv(b_views[j].pose) @ transform @ a_views[i].pose
            if not _heldout(a_views[i].heldout, b_views[j].heldout, relative, minimum)[0]:
                return False
            if (scope == "visual and held-out camera pairs" or edge.get("visual_constraint")) and not (
                    _visual_witness(a_views[i], b_views[j], relative, camera)[0]):
                return False
        return True
    return _heldout(fragments[a].heldout, fragments[b].heldout, transform)[0]


def _temporal_bridges(engine, fragments, edges, baseline, gap_limit, progress):
    """Retain raw visual ties hidden by first-success fragment ownership.

    Only three capture steps are searched, within the ordinary capture gap.
    World estimates are seeds, never constraints. Every tie needs distributed
    RGB-D feature identities and independent held-out depth at the same pose.
    """
    owned = {v.index: (fragment.index, v) for fragment in fragments for v in fragment.views}
    measured = {}
    for index in sorted(owned):
        owner, view = owned[index]
        _notify(progress, index + 1, len(engine.raw_frames), "Verifying nearby fragment boundaries")
        for before in range(max(0, index - 3), index):
            if before not in owned or owned[before][0] == owner:
                continue
            other, reference = owned[before]
            stamp = engine.frame_metadata[index].get("timestamp_s")
            last_stamp = engine.frame_metadata[before].get("timestamp_s")
            if stamp is not None and last_stamp is not None and not 0 < stamp - last_stamp <= gap_limit:
                continue
            if len(_matches(view, reference)) < 40:
                continue
            initial = (np.linalg.inv(baseline[before]) @ baseline[index]
                       if before in baseline and index in baseline else None)
            relative = _local_match(view, reference, engine.settings.camera, engine.settings,
                                    initial, measured_first=True)
            if relative is None:
                continue
            valid, stats = _visual_witness(view, reference, relative, engine.settings.camera)
            if not valid:
                continue
            # relative maps the later camera into the earlier camera.
            transform = view.pose @ np.linalg.inv(relative) @ np.linalg.inv(reference.pose)
            a, b, support = other, owner, (before, index)
            if a > b:
                a, b, support, transform = b, a, support[::-1], np.linalg.inv(transform)
            measured.setdefault((a, b), []).append((transform, support, {
                "source_index": index, "target_index": before, **stats}))
    ambiguous = []
    added = 0
    for (a, b), witnesses in sorted(measured.items()):
        witnesses.sort(key=lambda row: -row[2]["features"]["inliers"])
        transform = witnesses[0][0]
        # A contradictory short-range measurement is ambiguity, not a vote.
        if any(_disagrees(transform, pose, 0.03, 3) for pose, _, _ in witnesses[1:]):
            ambiguous.append([a, b])
            continue
        edge = {"source": a, "target": b, "transform": transform,
                "information": REG.get_information_matrix_from_point_clouds(
                    fragments[a].train, fragments[b].train, 0.03, transform),
                "support": [witnesses[0][1]], "validation_scope": "temporal camera pair",
                "validation": {"camera_pairs": [witnesses[0][2]]},
                "visual_constraint": True, "temporal_constraint": True}
        existing = next((e for e in edges if (e["source"], e["target"]) == (a, b)), None)
        if existing is not None:
            if _disagrees(existing["transform"], transform, 0.03, 3):
                ambiguous.append([a, b])
                continue
            # A storage-boundary measurement already supplies this constraint.
            continue
        if np.isfinite(edge["information"]).all() and _validate_bridge_pose(
                edge, {a: np.eye(4), b: np.linalg.inv(transform)}, fragments, engine.settings.camera):
            edges.append(edge)
            added += 1
    return added, ambiguous


def _bridge_world(fragments, edges, roots):
    """Build a maximum-evidence forest before choosing world coordinates.

    Connecting from the root greedily can reach a component through a false
    geometric loop before its visual/temporal ties are visited. Kruskal's
    forest preserves those ties even when that component is initially detached.
    """
    def rank(edge):
        if edge.get("temporal_constraint") or edge.get("validation_scope") == "sequential camera pair":
            return 0
        if edge.get("visual_constraint") or edge.get("validation_scope") == "visual and held-out camera pairs":
            return 1
        return 2

    groups = list(range(len(fragments)))
    def root(i):
        while groups[i] != i:
            i = groups[i]
        return i

    tree = []
    for edge in sorted(edges, key=rank):
        a, b = root(edge["source"]), root(edge["target"])
        if a != b:
            groups[a] = b
            tree.append(edge)
    world = {i: fragments[i].seed.copy() for i in roots}
    for _ in fragments:
        for edge in tree:
            a, b, pose = edge["source"], edge["target"], edge["transform"]
            if a in world and b not in world:
                world[b] = world[a] @ np.linalg.inv(pose)
            elif b in world and a not in world:
                world[a] = world[b] @ pose
    return world, {(e["source"], e["target"]) for e in tree}


def _boundary_conflicts(edges, poses, fragments, camera):
    """Pruning a measured boundary cannot make contradictory output safe."""
    return [[e["source"], e["target"]] for e in edges
            if (e.get("temporal_constraint") or e.get("validation_scope") == "sequential camera pair")
            and e["source"] in poses and e["target"] in poses
            and not _validate_bridge_pose(e, poses, fragments, camera)]


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
    recent = []
    report["recent_reference_links"] = []
    total = len(engine.raw_frames)
    for index in range(total):
        _notify(progress_cb, index + 1, total, f"Recovering fragments: preparing view {index + 1}/{total}")
        view = _view(engine, index)
        if view is None:
            report["invalid_indices"].append(index)
            current, previous = None, None
            continue
        stamp = engine.frame_metadata[index].get("timestamp_s")
        relative = None
        gap = True
        _prepare_matches(engine, [view], [reference for reference, _ in reversed(recent[-5:])])
        for reference, owner in reversed(recent[-5:]):
            last_stamp = engine.frame_metadata[reference.index].get("timestamp_s")
            if stamp is not None and last_stamp is not None and not 0 < stamp - last_stamp <= gap_limit:
                continue
            initial = (np.linalg.inv(baseline[reference.index]) @ baseline[index]
                       if index in baseline and reference.index in baseline else None)
            relative = (_local_match(view, reference, engine.settings.camera, engine.settings, initial, measured_first=True)
                        if engine.backend.get("final_local_refinement") == "measured"
                        else _local_match(view, reference, engine.settings.camera, engine.settings, initial))
            if relative is not None:
                gap = False
                current, previous = owner, reference
                if index != reference.index + 1:
                    report["recent_reference_links"].append({"source_index": index,
                        "target_index": reference.index, "fragment_id": owner.index})
                break
        full = current is not None and len(current.views) >= MAX_FRAGMENT_VIEWS
        if gap or relative is None or full:
            if len(fragments) == MAX_FRAGMENTS:
                report["budget_limited"] = True
                report["unassigned_indices"].append(index)
                current, previous = None, None
                continue
            context = []
            if not gap and relative is not None:
                # A size limit is not a tracking loss. Preserve the same raw
                # frame-to-frame measurement that would join these observations
                # inside a fragment. Live/world pose guesses never create edges.
                local_pose = previous.pose @ relative
                sequential.append((current.index, len(fragments), previous, view,
                                   local_pose, relative))
                # Retain nearby measured witnesses across a storage boundary.
                # They provide overlap for later bridge verification, but do
                # not become owned views or get fused a second time.
                to_next = np.linalg.inv(local_pose)
                context = [replace(v, pose=to_next @ v.pose)
                           for v in current.views[-2:]]
            current = Fragment(len(fragments), context=context)
            fragments.append(current)
        else:
            view.pose = previous.pose @ relative
        current.views.append(view)
        if index in baseline and current.seed is None:
            current.seed = baseline[index] @ np.linalg.inv(view.pose)
        previous = view
        recent.append((view, current))
        recent = recent[-5:]
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
                          "validation": stats, "validation_scope": "sequential camera pair",
                          "visual_constraint": _visual_witness(b, a, relative, engine.settings.camera)[0]})
    report["sequential_bridges"] = len(edges)
    report["temporal_bridges"], report["ambiguous_temporal_pairs"] = _temporal_bridges(
        engine, fragments, edges, baseline, gap_limit, progress_cb)
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
            _prepare_matches(engine, source.keys, target.keys)
            for a in source.keys:
                for b in target.keys:
                    matches = _matches(a, b)
                    proposal = propose_transform(a.features, b.features, engine.settings.camera, matches)
                    if proposal is not None:
                        proposals.append((len(matches), b.pose @ proposal @ np.linalg.inv(a.pose)))
            distance = np.linalg.norm(np.mean(source.fpfh.data, axis=1) - np.mean(target.fpfh.data, axis=1))
            candidates.append((-max([n for n, _ in proposals], default=0),
                               int(source.index not in roots and target.index not in roots
                                   and target.index != source.index + 1),
                               distance, source, target, proposals))
    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3].index, row[4].index))
    priority = {(row[3].index, row[4].index): i for i, row in enumerate(candidates)}
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
            bridge = (_verify_bridge(source, target, proposal, engine.settings.camera, visual_first=True)
                      if engine.backend.get("final_visual_first") == "on"
                      else _verify_bridge(source, target, proposal, engine.settings.camera))
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
        last_rank = max((priority.get((e["source"], e["target"]), -1) for e in edges), default=-1)
        if (eligible.issubset(_reachable(len(fragments), edges, roots)) and len(edges) >= len(eligible)
                and not any(priority[(row[3].index, row[4].index)] < last_rank for row in pending)):
            report["search_stopped_connected"] = True
            break
    connected_now = _reachable(len(fragments), edges, roots)
    report["budget_limited"] |= report["tested_pairs"] == MAX_PAIRS and any(
        row[3].index in connected_now or row[4].index in connected_now for row in pending)
    # Search order is a compute policy, not graph authority. Initialize from
    # the original evidence ranking, with measured sequential edges first.
    edges.sort(key=lambda e: (priority.get((e["source"], e["target"]), -1), e["source"], e["target"]))
    connected = _reachable(len(fragments), edges, roots)
    # Initialize from measured bridges, independently of the drifted live poses.
    primary = roots[0]
    world, tree_pairs = _bridge_world(fragments, edges, roots)
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
    rejected = []
    for edge in edges:
        a, b = edge["source"], edge["target"]
        if a not in connected or b not in connected or (a, b) not in surviving_pairs:
            continue
        valid = _validate_bridge_pose(edge, optimized, fragments, engine.settings.camera)
        if not valid:
            rejected.append([a, b])
        valid_graph &= valid
    boundary_conflicts = _boundary_conflicts(edges, optimized, fragments, engine.settings.camera)
    valid_graph &= not boundary_conflicts
    if boundary_conflicts:
        report["rejected_optimized_boundaries"] = boundary_conflicts
    if not valid_graph:
        # Optimization is optional. Restore measured spanning-tree poses and
        # revalidate every surviving bridge in those coordinates. Pruned edges
        # stay pruned; inconsistent loops cannot authorize disconnected views.
        report["optimization_fallback"] = "Revalidated measured bridge poses"
        report["rejected_optimized_bridges"] = rejected
        retained = [e for e in edges if (e["source"], e["target"]) in surviving_pairs
                    and _validate_bridge_pose(e, world, fragments, engine.settings.camera)]
        surviving_pairs = {(e["source"], e["target"]) for e in retained}
        connected = _reachable(len(fragments), retained, roots)
        optimized = {i: world[i].copy() for i in connected}
        valid_graph = all(_rigid(p) for p in optimized.values())
        conflicts = _boundary_conflicts(edges, optimized, fragments, engine.settings.camera)
        if conflicts:
            report["reason"] = "Conflicting measured fragment boundaries; previous reconstruction retained"
            report["rejected_fallback_boundaries"] = conflicts
            valid_graph = False
        if not valid_graph:
            if not conflicts:
                report["reason"] = "Measured bridge poses failed independent validation"
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
            "context_frame_indices": [v.index for v in fragment.context],
            "context_camera_to_fragment": [{"index": v.index, "pose": v.pose.tolist()} for v in fragment.context],
            "camera_to_fragment": [{"index": v.index, "pose": v.pose.tolist()} for v in fragment.views],
            "fragment_to_world": transform.tolist() if transform is not None else None,
        })
    # Count cycles in the surviving graph, rather than the original tree:
    # optimizer pruning or fallback can turn an old redundant edge into the
    # only remaining connection. Such an edge is not a closed loop.
    groups = {i: i for i in connected}
    def root(i):
        while groups[i] != i:
            i = groups[i]
        return i
    report["loop_closures"] = []
    for edge in edges:
        a, b = edge["source"], edge["target"]
        if a not in connected or b not in connected or (a, b) not in surviving_pairs:
            continue
        ra, rb = root(a), root(b)
        if ra == rb:
            report["loop_closures"].append({"source": a, "target": b, "support": edge["support"]})
        else:
            groups[ra] = rb
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
