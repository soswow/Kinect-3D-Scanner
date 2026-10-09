"""Bounded, calibrated multi-view RGB-D bundle adjustment for Finish.

Camera poses and common 3D landmarks are optimized together, rather than
optimizing pairwise pose-graph edges. Pixels constrain reprojection; *measured*
axial depth supplies metric scale. Calibration and the first camera are fixed.
Proposals require improvement on other measured depth pixels from every output
view. This module never changes the engine or fills missing depth; the caller
must perform bounded fresh fusion before committing a proposal.
"""

from dataclasses import dataclass
from collections import OrderedDict
import time

import cv2
import numpy as np

from shared.calibration import prepare_rgbd
from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS

from .appearance import Features, correspondences, extract_features, propose_transform
from .pose_candidates import choose_keyframes, fragment_pairs, validate_fragment_boundaries

MAX_KEYFRAMES = 24
MAX_TRACKS = 800
MAX_PAIRS = 96
# Validation covers every output view, with bounded resident sampled clouds.
VALIDATION_CACHE_VIEWS = 8
MAX_VALIDATION_POINTS = 2000
MAX_EVALUATIONS = 60
MAX_SECONDS = 45.0
MIN_TRACKS_PER_VIEW = 24
MAX_TRANSLATION_M = 0.25
MAX_ROTATION_DEG = 20.0


@dataclass
class BundleProblem:
    """Track observations use camera indices, landmark indices, pixels and z."""

    poses: list
    landmarks: np.ndarray
    cameras: np.ndarray
    tracks: np.ndarray
    pixels: np.ndarray
    depths: np.ndarray


class _BudgetExceeded(RuntimeError):
    pass


def _check_budget(started, seconds=None):
    if time.monotonic() - started > (MAX_SECONDS if seconds is None else seconds):
        raise _BudgetExceeded("Bundle adjustment exceeded its time budget")


def _notify(callback, current, total, message):
    if callback:
        callback(current, total, {"stage": "bundle_adjustment", "message": message})


def _rigid(pose):
    p = np.asarray(pose)
    return (p.shape == (4, 4) and np.isfinite(p).all()
            and np.allclose(p[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-5)
            and abs(np.linalg.det(p[:3, :3]) - 1) < 1e-5)


def _motion(pose):
    angle = np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3]) - 1) / 2, -1, 1)))
    return np.linalg.norm(pose[:3, 3]), angle


def _timely(metadata):
    lag = metadata.get("rgb_depth_delta_ms")
    # Old recordings have no timing metadata. Nonfinite/malformed timing is
    # never allowed to authorize RGB-D correspondence on new recordings.
    if lag is None:
        return True
    try:
        return np.isfinite(float(lag)) and abs(float(lag)) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS
    except (TypeError, ValueError):
        return False


def _distinct_features(features):
    """Prevent SIFT's several orientations at one pixel becoming many votes."""
    if features.descriptors is None:
        return features
    selected, seen = [], set()
    for index, pixel in enumerate(features.pixels):
        cell = tuple(np.floor(pixel / 4).astype(int))
        if cell not in seen:
            selected.append(index)
            seen.add(cell)
    selected = np.asarray(selected, int)
    return Features(features.pixels[selected], features.points[selected], features.descriptors[selected])


def _verified_matches(source, target, camera):
    matches = correspondences(source, target)
    transform = propose_transform(source, target, camera, matches)
    if transform is None or not _rigid(transform):
        return np.empty((0, 2), int)
    a, b = matches.T
    moved = source.points[a] @ transform[:3, :3].T + transform[:3, 3]
    z = moved[:, 2]
    pixels = moved[:, :2] / np.maximum(z[:, None], 1e-6)
    pixels = pixels * [camera.fx, camera.fy] + [camera.cx, camera.cy]
    good = ((z > 0) & (np.linalg.norm(pixels - target.pixels[b], axis=1) <= 2.5)
            & (np.linalg.norm(moved - target.points[b], axis=1) <= 0.025))
    return matches[good] if np.count_nonzero(good) >= 35 else np.empty((0, 2), int)


def build_tracks(features, verified_pairs, minimum_views=3):
    """Merge feature identities, rejecting a union containing two from one view.

    An ambiguous identity union is discarded in full. In particular, a cycle
    cannot silently replace one feature with another feature from the same
    image. Each output landmark has one observation per image and >=3 views.
    """
    parents, members, bad = {}, {}, set()

    def find(node):
        parents.setdefault(node, node)
        members.setdefault(node, {node[0]: node[1]})
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for i, j, matches in verified_pairs:
        for a, b in matches:
            left, right = find((i, int(a))), find((j, int(b)))
            if left == right:
                continue
            if len(members[left]) < len(members[right]):
                left, right = right, left
            conflict = bool(set(members[left]) & set(members[right]))
            invalid = conflict or left in bad or right in bad
            parents[right] = left
            members[left].update(members.pop(right))
            bad.discard(right)
            if invalid:
                bad.add(left)
    tracks = [sorted((view, feature) for view, feature in observations.items())
              for root, observations in members.items()
              if root not in bad and len(observations) >= minimum_views]
    # Reserve spatially distributed support for each camera before filling the
    # remaining budget with long tracks. Otherwise a textured late section can
    # consume the entire landmark cap and silently starve the anchored section.
    tracks.sort(key=lambda track: (-len(track), track))
    if len(tracks) <= MAX_TRACKS:
        return tracks
    selected, counts, cells = set(), np.zeros(len(features), int), {}
    target = max(1, min(MIN_TRACKS_PER_VIEW * 2, MAX_TRACKS * 3 // len(features)))

    def cell(view, feature):
        return (view, *np.floor(features[view].pixels[feature] / 80).astype(int))

    def add(index):
        selected.add(index)
        for view, feature in tracks[index]:
            counts[view] += 1
            key = cell(view, feature)
            cells[key] = cells.get(key, 0) + 1

    for view in range(len(features)):
        candidates = [(index, feature) for index, track in enumerate(tracks)
                      for camera, feature in track if camera == view]
        while counts[view] < target and len(selected) < MAX_TRACKS:
            available = [(index, feature) for index, feature in candidates if index not in selected]
            if not available:
                break
            index, _ = min(available, key=lambda item: (cells.get(cell(view, item[1]), 0),
                                                        -len(tracks[item[0]]), item[0]))
            add(index)
    for index in range(len(tracks)):
        if len(selected) >= MAX_TRACKS:
            break
        if index not in selected:
            add(index)
    return [tracks[index] for index in sorted(selected)]


def make_problem(poses, features, tracks):
    cameras, ids, pixels, depths, landmarks = [], [], [], [], []
    for landmark, track in enumerate(tracks):
        world = []
        for view, feature in track:
            point = features[view].points[feature]
            world.append(poses[view][:3, :3] @ point + poses[view][:3, 3])
            cameras.append(view)
            ids.append(landmark)
            pixels.append(features[view].pixels[feature])
            depths.append(point[2])
        landmarks.append(np.median(world, axis=0))
    return BundleProblem([p.copy() for p in poses], np.asarray(landmarks, float).reshape(-1, 3),
                         np.asarray(cameras, int), np.asarray(ids, int),
                         np.asarray(pixels, float).reshape(-1, 2), np.asarray(depths, float))


def _depth_sigma(depth):
    # Conservative engineering weighting for structured-light Kinect noise,
    # whose axial random error grows approximately quadratically with range.
    # This is NOT a calibrated per-device standard deviation or confidence.
    return 0.002 + 0.002 * np.asarray(depth) ** 2


def _decode(problem, vector):
    count = len(problem.poses)
    increments = np.vstack((np.zeros((1, 6)), vector[:6 * (count - 1)].reshape(-1, 6)))
    world_to_camera = []
    for pose, increment in zip(problem.poses, increments):
        delta = np.eye(4)
        delta[:3, :3] = cv2.Rodrigues(increment[:3])[0]
        delta[:3, 3] = increment[3:]
        world_to_camera.append(delta @ np.linalg.inv(pose))
    return world_to_camera, vector[6 * (count - 1):].reshape(-1, 3)


def solve_bundle(problem, camera, *, started=None, progress_cb=None):
    """Sparse robust simultaneous camera/landmark least squares, anchored at 0.

    Public numerical core also supports deterministic synthetic benchmarks.
    The high-level proposal API adds correspondence and held-out depth gates.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    started = time.monotonic() if started is None else started
    n_camera = 6 * (len(problem.poses) - 1)
    vector = np.r_[np.zeros(n_camera), problem.landmarks.ravel()]
    sparsity = lil_matrix((3 * len(problem.cameras), len(vector)), dtype=np.int8)
    for observation, (view, landmark) in enumerate(zip(problem.cameras, problem.tracks)):
        rows = slice(3 * observation, 3 * observation + 3)
        if view:
            sparsity[rows, 6 * (view - 1):6 * view] = 1
        sparsity[rows, n_camera + 3 * landmark:n_camera + 3 * landmark + 3] = 1
    evaluations, last_notification = 0, started

    def residual(parameters):
        nonlocal evaluations, last_notification
        _check_budget(started)
        evaluations += 1
        if time.monotonic() - last_notification > 1:
            _notify(progress_cb, min(evaluations, MAX_EVALUATIONS), MAX_EVALUATIONS,
                    "Jointly refining camera positions and common 3D features")
            last_notification = time.monotonic()
        extrinsics, landmarks = _decode(problem, parameters)
        matrices = np.asarray(extrinsics)
        points = (np.einsum("nij,nj->ni", matrices[problem.cameras, :3, :3],
                            landmarks[problem.tracks]) + matrices[problem.cameras, :3, 3])
        z = np.maximum(points[:, 2], 0.05)
        projection = points[:, :2] / z[:, None] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
        result = np.empty((len(points), 3))
        result[:, :2] = (projection - problem.pixels) / 1.5
        result[:, 2] = (points[:, 2] - problem.depths) / _depth_sigma(problem.depths)
        return result.ravel()

    initial = residual(vector)
    camera_limits = np.tile([np.radians(MAX_ROTATION_DEG)] * 3 + [MAX_TRANSLATION_M] * 3,
                            len(problem.poses) - 1)
    lower = np.r_[-camera_limits, (problem.landmarks - 0.35).ravel()]
    upper = np.r_[camera_limits, (problem.landmarks + 0.35).ravel()]
    result = least_squares(residual, vector, jac_sparsity=sparsity.tocsr(), bounds=(lower, upper),
                           method="trf", tr_solver="lsmr", loss="soft_l1", f_scale=2.0,
                           x_scale="jac", max_nfev=MAX_EVALUATIONS,
                           ftol=1e-5, xtol=1e-5, gtol=1e-5)
    extrinsics, landmarks = _decode(problem, result.x)
    poses = [np.linalg.inv(pose) for pose in extrinsics]
    # Preserve the supplied gauge bit-for-bit, including nonidentity first pose.
    poses[0] = problem.poses[0].copy()
    final = residual(result.x).reshape(-1, 3)
    report = {"solver_evaluations": int(result.nfev), "residual_evaluations": evaluations,
              "solver_converged": bool(result.success), "solver_message": str(result.message),
              "training_before": float(np.mean(np.minimum(initial ** 2, 25))),
              "training_after": float(np.mean(np.minimum(final.ravel() ** 2, 25))),
              "training_inlier_fraction": float(np.mean((np.linalg.norm(final[:, :2], axis=1) <= 2)
                                                         & (np.abs(final[:, 2]) <= 3)))}
    return poses, landmarks, report


def _supported(problem, camera, report=None):
    counts = [int(np.count_nonzero(problem.cameras == view)) for view in range(len(problem.poses))]
    if report is not None:
        report["observations_per_keyframe"] = counts
        report["unsupported_keyframes"] = [view for view, count in enumerate(counts)
                                           if count < MIN_TRACKS_PER_VIEW]
        report["support_failure"] = None
    for view in range(len(problem.poses)):
        selected = problem.cameras == view
        if counts[view] < MIN_TRACKS_PER_VIEW:
            if report is not None:
                report["support_failure"] = "too_few_three_view_tracks"
            return False
        pixels = problem.pixels[selected] / [camera.width, camera.height]
        if np.linalg.eigvalsh(np.cov(pixels.T))[0] < 0.001:
            if report is not None:
                report["support_failure"] = "insufficient_spatial_distribution"
                report["spatially_unsupported_keyframe"] = view
            return False
    # Every camera must be connected through common multi-view landmarks to 0.
    reached = {0}
    while True:
        new = set(reached)
        for landmark in np.unique(problem.tracks):
            views = set(problem.cameras[problem.tracks == landmark])
            if reached & views:
                new.update(views)
        if new == reached:
            if report is not None:
                report["disconnected_keyframes"] = sorted(set(range(len(problem.poses))) - reached)
                if len(reached) != len(problem.poses):
                    report["support_failure"] = "disconnected_landmark_graph"
            return len(reached) == len(problem.poses)
        reached = new


def _heldout_points(depth, camera, features=None):
    mask = np.asarray(depth) > 0
    if features is not None:
        mask = mask.astype(np.uint8)
        for pixel in features.pixels:
            cv2.circle(mask, tuple(np.rint(pixel).astype(int)), 5, 0, -1)
    # No sampled validation pixel lies in a feature's measured 3x3 depth patch.
    y, x = np.indices(depth.shape)
    selected = (mask != 0) & (x % 7 == 3) & (y % 7 == 3)
    y, x = np.nonzero(selected)
    if len(x) > MAX_VALIDATION_POINTS:
        take = np.linspace(0, len(x) - 1, MAX_VALIDATION_POINTS, dtype=int)
        x, y = x[take], y[take]
    z = depth[y, x].astype(float) / 1000
    return np.column_stack(((x - camera.cx) * z / camera.fx, (y - camera.cy) * z / camera.fy, z))


def _pair_distance(source, target, transform):
    from scipy.spatial import cKDTree

    moved = source @ transform[:3, :3].T + transform[:3, 3]
    distance = cKDTree(target).query(moved, workers=1)[0]
    return float(np.mean(np.minimum(distance, 0.04) ** 2)), float(np.mean(distance < 0.04))


def validate_depth(clouds, originals, proposals, pairs, *, started=None,
                   budget_seconds=None, minimum_improvement=0.02,
                   retain_initial_overlap=False):
    """Symmetric saturated loss; clouds may be a bounded on-demand provider."""
    before_sum = after_sum = 0.0
    count = pair_count = 0
    min_overlap = 1.0
    consistent = True
    rejected = []
    for i, j in pairs:
        if started is not None:
            _check_budget(started, budget_seconds)
        source_cloud, target_cloud = clouds[i], clouds[j]
        if min(len(source_cloud), len(target_cloud)) < 100:
            return False, {"validation_failure": "Insufficient independent measured depth",
                           "validation_failed_pair": [i, j], "validation_pairs": pair_count}
        for source, target in ((i, j), (j, i)):
            if started is not None:
                _check_budget(started, budget_seconds)
            old = np.linalg.inv(originals[target]) @ originals[source]
            new = np.linalg.inv(proposals[target]) @ proposals[source]
            a_cloud, b_cloud = ((source_cloud, target_cloud) if source == i
                                else (target_cloud, source_cloud))
            b, overlap_b = _pair_distance(a_cloud, b_cloud, old)
            a, overlap_a = _pair_distance(a_cloud, b_cloud, new)
            if started is not None:
                _check_budget(started, budget_seconds)
            before_sum += b
            after_sum += a
            count += 1
            min_overlap = min(min_overlap, overlap_a)
            good = (np.isfinite([a, b, overlap_a, overlap_b]).all()
                    and a <= b * 1.1 + 1e-6
                    and overlap_a >= (min(0.35, max(0, overlap_b - 0.05))
                                      if retain_initial_overlap else 0.35)
                    and overlap_a >= overlap_b - 0.05)
            consistent &= good
            if not good and len(rejected) < 20:
                rejected.append({"source_position": source, "target_position": target,
                                 "before_m2": b, "after_m2": a,
                                 "overlap_before": overlap_b, "overlap_after": overlap_a})
        pair_count += 1
    if not count:
        return False, {"validation_failure": "No independent depth validation pairs"}
    report = {"validation_before_m2": before_sum / count,
              "validation_after_m2": after_sum / count,
              "validation_pairs": pair_count, "validation_min_overlap": float(min_overlap),
              "validation_rejected_pairs": rejected}
    valid = consistent and after_sum < before_sum * (1 - minimum_improvement)
    return bool(valid), report


class _DepthClouds:
    """Load calibrated held-out samples on demand; never retain all depth images."""

    def __init__(self, engine, features, started, report, progress_cb=None, *, budget_seconds=None):
        self.engine, self.features, self.started = engine, features, started
        self.report, self.progress_cb = report, progress_cb
        self.budget_seconds = budget_seconds
        self.cache = OrderedDict()
        self.visited = set()
        report.update(validation_views=0, validation_cloud_loads=0,
                      validation_peak_cached_views=0, validation_peak_cache_bytes=0)

    def __getitem__(self, position):
        _check_budget(self.started, self.budget_seconds)
        if position not in self.cache:
            # Evict before allocating the next cloud, including cross-batch pairs.
            if len(self.cache) >= VALIDATION_CACHE_VIEWS:
                self.cache.popitem(last=False)
            index, _ = self.engine.poses[position]
            prepare = getattr(self.engine, "_prepare_input", None)
            _, depth = (prepare_rgbd(*self.engine.raw_frames[index], self.engine.settings)
                        if prepare is None else prepare(*self.engine.raw_frames[index],
                            self.engine.settings, cpu_prepare=prepare_rgbd))
            _check_budget(self.started, self.budget_seconds)
            self.cache[position] = _heldout_points(depth, self.engine.settings.camera,
                                                   self.features.get(position))
            self.visited.add(position)
            self.report["validation_views"] = len(self.visited)
            self.report["validation_cloud_loads"] += 1
            self.report["validation_peak_cached_views"] = max(
                self.report["validation_peak_cached_views"], len(self.cache))
            self.report["validation_peak_cache_bytes"] = max(
                self.report["validation_peak_cache_bytes"], sum(c.nbytes for c in self.cache.values()))
            _notify(self.progress_cb, len(self.visited), len(self.engine.poses),
                    "Checking refined positions against independent depth")
        self.cache.move_to_end(position)
        return self.cache[position]


def _interpolate(a, b, fraction):
    result = np.eye(4)
    rotation = cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0]
    result[:3, :3] = a[:3, :3] @ cv2.Rodrigues(rotation * fraction)[0]
    result[:3, 3] = a[:3, 3] * (1 - fraction) + b[:3, 3] * fraction
    return result


def _output_poses(poses, chosen, refined):
    # Interpolate correction fields in the anchored first-camera world frame.
    # This is world-origin invariant and avoids rotating a correction into a
    # different camera-local axis system at each omitted view.
    anchor = poses[0][1]
    inverse_anchor = np.linalg.inv(anchor)
    corrections = [inverse_anchor @ new @ np.linalg.inv(poses[index][1]) @ anchor
                   for index, new in zip(chosen, refined)]
    proposals = []
    for position, (index, pose) in enumerate(poses):
        right = min(int(np.searchsorted(chosen, position, side="right")), len(chosen) - 1)
        left = max(0, right - 1)
        fraction = (position - chosen[left]) / max(1, chosen[right] - chosen[left])
        correction = _interpolate(corrections[left], corrections[right], fraction)
        proposals.append((index, anchor @ correction @ inverse_anchor @ pose))
    proposals[0] = (poses[0][0], poses[0][1].copy())
    return proposals


def propose_bundle_poses(engine, progress_cb=None):
    """Return immutable-engine pose proposal/report; fresh fusion is caller-owned."""
    started = time.monotonic()
    report = {"applied": False, "reason": "Insufficient connected multi-view RGB-D tracks",
              "method": "joint_camera_landmark_rgbd_bundle_adjustment",
              "fixed": "first accepted camera and calibrated intrinsics",
              "validation": "independent measured depth outside all feature patches, all output views",
              "depth_weighting": "engineering range-dependent model, not calibrated confidence",
              "max_keyframes": MAX_KEYFRAMES, "max_landmarks": MAX_TRACKS,
              "max_pairs": MAX_PAIRS, "validation_cache_views": VALIDATION_CACHE_VIEWS,
              "max_solver_evaluations": MAX_EVALUATIONS, "time_budget_s": MAX_SECONDS}

    def finish(proposals=None):
        report["elapsed_ms"] = (time.monotonic() - started) * 1000
        return proposals, report

    try:
        # Missing optional server dependency is a conservative no-op.
        from scipy.optimize import least_squares  # noqa: F401
        poses = engine.poses
        report.update(accepted_views=len(poses), stage="input")
        if len(poses) < 3:
            report["reason"] = "Bundle adjustment requires at least 3 accepted views"
            return finish()
        if any(not _rigid(pose) for _, pose in poses):
            report["reason"] = "Invalid input camera pose"
            return finish()
        hints = fragment_pairs(engine)
        chosen = choose_keyframes(len(poses), MAX_KEYFRAMES, hints[:MAX_PAIRS])
        report["keyframe_indices"] = [int(poses[position][0]) for position in chosen]
        selected = {int(position): i for i, position in enumerate(chosen)}
        hint_pairs = {tuple(sorted((selected[a], selected[b]))) for a, b in hints
                      if a in selected and b in selected}
        report["fragment_candidate_pairs"] = len(hint_pairs)
        report["stage"] = "features"
        features = []
        for completed, position in enumerate(chosen, 1):
            _check_budget(started)
            index, _ = poses[position]
            if not _timely(engine.frame_metadata[index]):
                report["reason"] = "Selected RGB and depth measurements are not synchronized"
                return finish()
            prepare = getattr(engine, "_prepare_input", None)
            rgb, depth = (prepare_rgbd(*engine.raw_frames[index], engine.settings)
                          if prepare is None else prepare(*engine.raw_frames[index],
                              engine.settings, cpu_prepare=prepare_rgbd))
            feature = _distinct_features(extract_features(rgb, depth, engine.settings.camera, method="sift"))
            features.append(feature)
            _notify(progress_cb, completed, len(chosen), "Finding features for joint RGB-D refinement")
        report["keyframes"] = len(chosen)
        originals = [poses[position][1] for position in chosen]
        pairs = [(i, j) for i in range(len(chosen)) for j in range(i + 1, len(chosen))]
        pairs.sort(key=lambda pair: (-1 if pair in hint_pairs else 0 if pair[1] - pair[0] <= 2 else 1,
                                    pair[1] - pair[0] if pair[1] - pair[0] <= 2 else
                                    np.linalg.norm(originals[pair[0]][:3, 3] - originals[pair[1]][:3, 3]), pair))
        verified = []
        report["stage"] = "matching"
        for completed, (i, j) in enumerate(pairs[:MAX_PAIRS], 1):
            _check_budget(started)
            matches = _verified_matches(features[i], features[j], engine.settings.camera)
            if len(matches):
                verified.append((i, j, matches))
            if completed % 8 == 0:
                _notify(progress_cb, completed, min(len(pairs), MAX_PAIRS), "Verifying multi-view feature identities")
        tracks = build_tracks(features, verified)
        report.update(verified_pairs=len(verified), landmarks=len(tracks),
                      observations=sum(len(track) for track in tracks))
        problem = make_problem(originals, features, tracks)
        if not _supported(problem, engine.settings.camera, report):
            report["unsupported_keyframe_indices"] = [int(poses[chosen[i]][0])
                for i in report.get("unsupported_keyframes", [])]
            return finish()
        report["stage"] = "solve"
        refined, _, solver_report = solve_bundle(problem, engine.settings.camera,
                                                 started=started, progress_cb=progress_cb)
        report.update(solver_report)
        if report["training_inlier_fraction"] < 0.9 or report["training_after"] >= report["training_before"] * 0.98:
            report["reason"] = "Multi-view optimization did not retain consistent measured observations"
            return finish()
        proposals = _output_poses(poses, chosen, refined)
        for (_, old), (_, new) in zip(poses, proposals):
            translation, angle = _motion(np.linalg.inv(old) @ new)
            if not _rigid(new) or translation > MAX_TRANSLATION_M or angle > MAX_ROTATION_DEG:
                report["reason"] = "Bundle optimization exceeded camera correction bounds"
                return finish()
        report["stage"] = "validation"
        by_position = dict(zip(chosen, features))
        clouds = _DepthClouds(engine, by_position, started, report, progress_cb)
        validation_pairs = set((i, i + 1) for i in range(len(poses) - 1))
        validation_pairs.update((int(chosen[i]), int(chosen[j])) for i, j, _ in verified)
        valid, validation = validate_depth(clouds, [p for _, p in poses], [p for _, p in proposals],
                                           sorted(validation_pairs), started=started)
        report.update(validation)
        _check_budget(started)
        if not valid or report["validation_views"] != len(poses):
            report["reason"] = "Independent depth did not improve consistently across output views"
            return finish()
        def budget_check():
            _check_budget(started)
            return True
        valid, boundary_report = validate_fragment_boundaries(engine, proposals, budget_check)
        report.update(boundary_report)
        _check_budget(started)
        if not valid:
            report["reason"] = "Bundle adjustment conflicts with a retained measured fragment boundary"
            return finish()
        report["reason"] = "Validated joint camera/landmark proposal; awaiting bounded fresh fusion"
        report["stage"] = "complete"
        return finish(proposals)
    except ImportError:
        report["reason"] = "SciPy is required for optional joint RGB-D bundle adjustment"
        return finish()
    except _BudgetExceeded as exc:
        report["reason"] = str(exc)
        report["budget_exceeded"] = True
        return finish()
