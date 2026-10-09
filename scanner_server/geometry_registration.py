"""Offline depth registration candidates and separate geometric samples.

No RGB, recorded poses, timestamp motion limits, or minimum camera count is
required. A candidate must explain depth, constrain all six pose directions,
and survive competing-pose and graph checks. Thresholds are engineering policy,
not calibrated sensor probabilities. This module never mutates an engine.
"""

from dataclasses import dataclass, field
from itertools import permutations, product
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from shared.calibration import prepare_metric_depth, pinhole_rays

REG = o3d.pipelines.registration


@dataclass
class DepthView:
    index: int
    depth: np.ndarray
    cloud: object
    heldout: object
    camera: object
    features: dict = field(default_factory=dict)
    levels: dict = field(default_factory=dict)
    tree: object = None
    proposal_voxel: float = .025


def cloud(points, voxel=.02):
    value = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if not len(points):
        return value
    value = value.voxel_down_sample(voxel)
    value.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(.06, voxel * 3), max_nn=40))
    value.orient_normals_towards_camera_location(np.zeros(3))
    return value


def prepare_view(index, raw, settings):
    depth = prepare_metric_depth(raw, settings).astype(np.float32) / 1000
    points = pinhole_rays(settings.camera) * depth[..., None]
    y, x = np.indices(depth.shape)
    # Disjoint pixel samples before geometric downsampling. Depth filtering
    # mixes neighbours, so these are not statistically independent noise draws.
    valid = depth > 0
    train = valid & ((x + y) % 2 == 0) & (x % 2 == 0)
    witness = valid & ((x + y) % 2 == 1) & (x % 2 == 1)
    a, b = cloud(points[train]), cloud(points[witness], .015)
    if len(b.points):
        # Suppress normal variation caused only by millimetre range noise.
        # Otherwise a flat wall falsely gains tangential pose information.
        b.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.12, max_nn=200))
        b.orient_normals_towards_camera_location(np.zeros(3))
    return DepthView(index, depth, a, b, settings.camera,
                     tree=cKDTree(np.asarray(b.points)))


def rigid(pose):
    return (np.shape(pose) == (4, 4) and np.isfinite(pose).all()
            and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7)
            and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            and abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-5)


def pose_distance(a, b):
    delta = np.linalg.inv(a) @ b
    return (float(np.linalg.norm(delta[:3, 3])),
            float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))))


def observability(points, normals):
    """Dimensionless six-direction point-to-plane Jacobian conditioning.

    Centering removes arbitrary world-origin dependence. RMS radius scales the
    rotational columns to the same units as translation. A diverse normal
    distribution alone misses rotational ambiguity on a sphere.
    """
    if len(points) < 100:
        return np.zeros(6), 0.
    centered = points - np.mean(points, axis=0)
    radius = max(.05, float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))))
    jacobian = np.column_stack((np.cross(centered, normals) / radius, normals))
    values = np.linalg.eigvalsh(jacobian.T @ jacobian / len(points))
    return values, float(max(0, values[0]) / max(values[-1], 1e-12))


def _visibility(source, target, pose):
    points = np.asarray(source.heldout.points)
    points = points[::max(1, len(points) // 5000)]
    moved = points @ pose[:3, :3].T + pose[:3, 3]
    positive = moved[:, 2] > .05
    moved = moved[positive]
    if not len(moved):
        return {"tested": 0, "free_space_fraction": 1., "depth_agreement": 0.}
    camera = target.camera
    pixels = np.rint(moved[:, :2] / moved[:, 2, None] * [camera.fx, camera.fy]
                     + [camera.cx, camera.cy]).astype(int)
    inside = ((pixels[:, 0] >= 1) & (pixels[:, 0] < camera.width - 1)
              & (pixels[:, 1] >= 1) & (pixels[:, 1] < camera.height - 1))
    moved, pixels = moved[inside], pixels[inside]
    x, y = pixels.T
    samples = np.stack([target.depth[y + dy, x + dx] for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))])
    supported = (samples.min(axis=0) > 0) & (np.ptp(samples, axis=0) < .04)
    measured = samples[0, supported]
    errors = moved[supported, 2] - measured
    # Larger axial tolerance at range; this is not a sensor calibration.
    tolerance = .02 + .006 * measured ** 2
    return {"tested": int(len(errors)),
            "free_space_fraction": float(np.mean(errors < -tolerance)) if len(errors) else 1.,
            "depth_agreement": float(np.mean(np.abs(errors) <= tolerance)) if len(errors) else 0.}


def evaluate(source, target, pose, *, minimum_overlap=.2, minimum_condition=1e-4, check_visibility=True):
    """Assess a separate depth sample without trusting algorithm fitness."""
    report = {"accepted": False}
    if not rigid(pose):
        return {**report, "reason": "nonrigid"}
    if min(len(source.heldout.points), len(target.heldout.points)) < 100:
        return {**report, "reason": "insufficient_depth"}
    rows = []
    for a, b, transform in ((source, target, pose), (target, source, np.linalg.inv(pose))):
        points = np.asarray(a.heldout.points)
        moved = points @ transform[:3, :3].T + transform[:3, 3]
        distance, ids = b.tree.query(moved, workers=1)
        normals = np.asarray(b.heldout.normals)[ids]
        source_normals = np.asarray(a.heldout.normals) @ transform[:3, :3].T
        matched = (distance < .03) & (np.abs(np.sum(source_normals * normals, axis=1)) > .85)
        eigenvalues, condition = observability(moved[matched], normals[matched])
        rows.append({"overlap": float(np.mean(matched)) if len(matched) else 0.,
                     "inliers": int(matched.sum()),
                     "rmse_m": float(np.sqrt(np.mean(distance[matched] ** 2))) if matched.any() else 1.,
                     "condition": condition, "eigenvalues": eigenvalues.tolist(),
                     **(_visibility(a, b, transform) if check_visibility else
                        {"tested": int(matched.sum()), "free_space_fraction": 0., "depth_agreement": float(np.mean(matched))})})
    report.update(forward=rows[0], reverse=rows[1])
    if min(r["inliers"] for r in rows) < 100:
        reason = "too_few_depth_matches"
    elif min(r["overlap"] for r in rows) < minimum_overlap:
        reason = "insufficient_overlap"
    elif max(r["rmse_m"] for r in rows) > .02:
        reason = "depth_residual"
    elif min(r["condition"] for r in rows) < minimum_condition:
        reason = "unobservable_pose"
    elif min(r["tested"] for r in rows) < 100 or max(r["free_space_fraction"] for r in rows) > .15:
        reason = "free_space_conflict"
    else:
        reason = "accepted_depth_evidence"
    report.update(accepted=reason == "accepted_depth_evidence", reason=reason, visibility_checked=check_visibility,
                  score=min(r["overlap"] for r in rows) / max(.005, max(r["rmse_m"] for r in rows)))
    return report


def _level(view, voxel):
    if voxel not in view.levels:
        level = view.cloud.voxel_down_sample(voxel)
        if len(level.points):
            level.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=40))
        view.levels[voxel] = level
    return view.levels[voxel]


def refine(source, target, initial, method="icp"):
    pose = initial.copy()
    for voxel, radius, iterations in ((.06, .18, 35), (.03, .09, 25), (.02, .045, 20)):
        a, b = _level(source, voxel), _level(target, voxel)
        if min(len(a.points), len(b.points)) < 100:
            return None
        if method == "gicp":
            result = REG.registration_generalized_icp(a, b, radius, pose,
                       REG.TransformationEstimationForGeneralizedICP(), REG.ICPConvergenceCriteria(max_iteration=iterations))
        else:
            result = REG.registration_icp(a, b, radius, pose,
                       REG.TransformationEstimationPointToPlane(REG.HuberLoss(.015)), REG.ICPConvergenceCriteria(max_iteration=iterations))
        pose = result.transformation
    return pose if rigid(pose) else None


def projective_seed(source, target, device="CPU:0"):
    """Dense depth-only odometry; RGB inputs are identically zero."""
    core, odo = o3d.core, o3d.t.pipelines.odometry
    def image(view):
        return o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(core.Tensor(np.zeros((*view.depth.shape, 3), np.uint8)).to(device)),
            o3d.t.geometry.Image(core.Tensor(view.depth).to(device)))
    c = source.camera
    intrinsic = core.Tensor([[c.fx, 0., c.cx], [0., c.fy, c.cy], [0., 0., 1.]], core.float64)
    result = odo.rgbd_odometry_multi_scale(image(source), image(target), intrinsic,
        depth_scale=1., depth_max=8., criteria_list=[odo.OdometryConvergenceCriteria(n) for n in (30, 20, 10)],
        method=odo.Method.PointToPlane, params=odo.OdometryLossParams(depth_outlier_trunc=.12, depth_huber_delta=.025))
    pose = result.transformation.cpu().numpy()
    return pose if rigid(pose) else None


def features(view, radius):
    if radius not in view.features:
        # Fine enough sampling to describe the smallest physical neighborhood.
        value = _level(view, view.proposal_voxel)
        if len(value.points):
            descriptor = REG.compute_fpfh_feature(value, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=150))
        else:
            descriptor = REG.Feature()
            descriptor.resize(33, 0)
        view.features[radius] = value, descriptor
    return view.features[radius]


def descriptor_pairs(a, b, device="cpu"):
    """KD-tree or chunked CUDA descriptor search, with mutual matches.

    CUDA uses float32 and may break near-ties differently from the CPU tree.
    Correspondences only propose poses; measured depth decides acceptance.
    """
    if not a.shape[1] or not b.shape[1]:
        return np.empty((0, 2), dtype=int)
    if device == "cuda":
        import cupy as cp
        left, right = cp.asarray(a.T, dtype=cp.float32), cp.asarray(b.T, dtype=cp.float32)
        def nearest(x, y):
            ids = []
            yn = cp.sum(y * y, axis=1)
            for start in range(0, len(x), 256):
                chunk = x[start:start + 256]
                distances = cp.maximum(0, cp.sum(chunk * chunk, axis=1)[:, None] + yn[None, :] - 2 * chunk @ y.T)
                ids.append(cp.asnumpy(cp.argmin(distances, axis=1)))
            return np.concatenate(ids)
        forward, reverse = nearest(left, right), nearest(right, left)
    else:
        forward = cKDTree(b.T).query(a.T, workers=1)[1]
        reverse = cKDTree(a.T).query(b.T, workers=1)[1]
    ids = np.arange(len(forward))
    return np.column_stack((ids[reverse[forward] == ids], forward[reverse[forward] == ids]))


def global_seeds(source, target, method="multiscale", device="cpu"):
    matching_device = device
    if device == "cuda":
        try:
            import cupy
        except ImportError:
            matching_device = "cpu"
    radii = (.08, .16, .32) if method == "multiscale" else (.2,)
    output = []
    descriptors = []
    for radius in radii:
        a, fa = features(source, radius)
        b, fb = features(target, radius)
        descriptors.append((np.asarray(fa.data), np.asarray(fb.data)))
    if method == "multiscale":
        # Separate radii remain useful when the larger patch is occluded.
        descriptors.append(tuple(np.vstack([d[i] / np.maximum(np.linalg.norm(d[i], axis=0), 1e-8)
                                            for d in descriptors]) for i in (0, 1)))
    for number, (fa, fb) in enumerate(descriptors):
        matches = descriptor_pairs(fa, fb, matching_device)
        if len(matches) < 10:
            continue
        o3d.utility.random.seed(source.index * 1009 + target.index * 31 + number)
        result = REG.registration_ransac_based_on_correspondence(a, b,
            o3d.utility.Vector2iVector(matches), .075, REG.TransformationEstimationPointToPoint(False), 3,
            [REG.CorrespondenceCheckerBasedOnEdgeLength(.9), REG.CorrespondenceCheckerBasedOnDistance(.075)],
            REG.RANSACConvergenceCriteria(18000, .999))
        if rigid(result.transformation) and result.fitness >= .15:
            output.append((f"{method}:{number}", result.transformation))
    return output


def ppf_seeds(source, target, limit=12):
    """Bounded oriented-pair hash votes, without a learned model or CAD mesh.

    This deliberately samples a bounded number of pairs instead of an N² table.
    Candidate bins tolerate range/normal noise; only independent raw-depth
    evaluation decides whether any voted transformation is usable.
    """
    def pairs(view):
        points, normals = np.asarray(view.cloud.points), np.asarray(view.cloud.normals)
        ids = np.linspace(0, len(points) - 1, min(180, len(points)), dtype=int)
        points, normals = points[ids], normals[ids]
        output = []
        for i in range(len(points)):
            for j in range(i + 1, len(points), 7):
                d = points[j] - points[i]
                length = np.linalg.norm(d)
                if not .08 < length < 1.5:
                    continue
                direction = d / length
                angles = np.arccos(np.clip([normals[i] @ direction, normals[j] @ direction, normals[i] @ normals[j]], -1, 1))
                axis = normals[i] - direction * (normals[i] @ direction)
                if np.linalg.norm(axis) < .15:
                    continue
                axis /= np.linalg.norm(axis)
                frame = np.column_stack((direction, axis, np.cross(direction, axis)))
                feature = np.r_[length / .04, angles / np.radians(12)]
                output.append((tuple(np.rint(feature).astype(int)), points[i], frame))
        return output
    table = {}
    for key, point, frame in pairs(target):
        values = table.setdefault(key, [])
        if len(values) < 6:
            values.append((point, frame))
    votes = {}
    for key, point, frame in pairs(source):
        # Full neighbouring bins, including angular quantization boundaries.
        for offset in product((-1, 0, 1), repeat=4):
            candidates = table.get(tuple(k + d for k, d in zip(key, offset)), ())
            for target_point, target_frame in candidates:
                rotation = target_frame @ frame.T
                translation = target_point - rotation @ point
                # Axis-angle avoids Euler ordering discontinuities.
                import cv2
                vector = cv2.Rodrigues(rotation)[0].reshape(3)
                cell = tuple(np.rint(np.r_[translation / .08, vector / np.radians(12)]).astype(int))
                if cell not in votes:
                    pose = np.eye(4)
                    pose[:3, :3], pose[:3, 3] = rotation, translation
                    votes[cell] = [0, pose]
                votes[cell][0] += 1
    return [("ppf", value[1]) for value in sorted(votes.values(), key=lambda value: -value[0])[:limit]]


def pca_seeds(source, target):
    def axes(view):
        points = np.asarray(view.cloud.points)
        center = np.mean(points, axis=0)
        return center, np.linalg.eigh(np.cov(points.T))[1]
    ca, a = axes(source)
    cb, b = axes(target)
    output = []
    for ordering in permutations(range(3)):
        for signs in product((-1., 1.), repeat=3):
            rotation = b[:, ordering] @ np.diag(signs) @ a.T
            if np.linalg.det(rotation) < 0:
                continue
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rotation, cb - rotation @ ca
            output.append(("pca", pose))
    return output


def register_pair(source, target, *, method="local", device="cpu", initial=None):
    started = time.monotonic()
    if min(len(source.cloud.points), len(target.cloud.points)) < 100:
        return None, {"source": source.index, "target": target.index, "method": method,
                      "elapsed_s": 0., "accepted": False, "reason": "insufficient_depth", "candidates": []}
    seeds = [("identity", np.eye(4))]
    if initial is not None:
        seeds.append(("geometry_prediction", initial))
    if method == "projective":
        seed = projective_seed(source, target, "CUDA:0" if device == "cuda" else "CPU:0")
        if seed is not None:
            seeds.append(("depth_projective", seed))
    if method in ("fpfh", "multiscale"):
        seeds += global_seeds(source, target, method, device)
    if method == "pca":
        seeds += pca_seeds(source, target)
    if method == "ppf":
        seeds += ppf_seeds(source, target)
    candidates, unique = [], []
    for name, seed in seeds:
        if any(max(pose_distance(seed, old)[0] / .02, pose_distance(seed, old)[1] / 2) < 1 for old in unique):
            continue
        unique.append(seed)
        pose = refine(source, target, seed, "gicp" if method == "gicp" else "icp")
        if pose is None:
            continue
        candidates.append({"method": name, "pose": pose})
    return verify_candidates(source,target,candidates,method=method,elapsed_s=time.monotonic()-started)


def verify_candidates(source,target,candidates,*,method="local",elapsed_s=0.):
    """Reassess every saved hypothesis, including previously rejected ones."""
    started = time.monotonic()
    candidates = [{"method":c["method"],"pose":np.asarray(c["pose"]),
                   **evaluate(source,target,np.asarray(c["pose"]))} for c in candidates]
    accepted = sorted([c for c in candidates if c["accepted"]], key=lambda c: -c["score"])
    report = {"source": source.index, "target": target.index, "method": method,
              "elapsed_s": elapsed_s + time.monotonic() - started,
              "candidates": [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in c.items()} for c in candidates]}
    if not accepted:
        report.update(accepted=False, reason="no_supported_pose")
        return None, report
    best = accepted[0]
    competing = [c for c in accepted[1:] if c["score"] >= .85 * best["score"]
                 and (pose_distance(c["pose"], best["pose"])[0] > .05 or pose_distance(c["pose"], best["pose"])[1] > 5)]
    if competing:
        report.update(accepted=False, reason="competing_geometry")
        return None, report
    reverse = refine(target, source, np.linalg.inv(best["pose"]), "gicp" if method == "gicp" else "icp")
    cycle = pose_distance(np.eye(4), reverse @ best["pose"]) if reverse is not None else (1., 180.)
    good = cycle[0] <= .025 and cycle[1] <= 3
    report.update(accepted=good, reason="accepted_depth_evidence" if good else "reciprocal_disagreement",
                  cycle_m=cycle[0], cycle_deg=cycle[1], score=best["score"], evidence={k: v for k, v in best.items() if k != "pose"})
    if good:
        report["transform"] = best["pose"].tolist()
    return best["pose"] if good else None, report
