"""Sparse camera/landmark refinement with explicit engineering residual scales.

Acceleration contributes uncertain gravity, never integrated displacement. Every
proposal requires the caller's separate raw-depth and feature revalidation.
"""

import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from .motion_evidence import visual_seed
from .depth_surface_constraints import depth_constraints
from .depth_planes import plane_observations


def landmark_observations(poses, edges, motion, *, features_per_edge=80):
    """Merge measured identities; discard unions with two features in one image."""
    parents, members, invalid, features = {}, {}, set(), {}

    def root(node):
        parents.setdefault(node, node)
        members.setdefault(node, {node[0]: node})
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for edge in edges:
        a, b = edge["source"], edge["target"]
        if edge.get("sensor_visual_prior"):
            continue
        pair = motion.appearance_pair(a, b)
        if pair is None:
            continue
        fa, fb, basis = motion.features_for_pair(a, b)
        if basis == "tracked_rgbd_identities" and hasattr(motion, "direct_groups"):
            continue
        matches = pair[1]
        selected = matches[np.linspace(0, len(matches)-1, min(features_per_edge, len(matches)), dtype=int)]
        for x, y in selected:
            na, nb = (a, basis, int(x)), (b, basis, int(y))
            features[na], features[nb] = (fa.points[x], fa.pixels[x]), (fb.points[y], fb.pixels[y])
            left, right = root(na), root(nb)
            if left == right:
                continue
            if len(members[left]) < len(members[right]):
                left, right = right, left
            bad = bool(set(members[left]) & set(members[right])) or left in invalid or right in invalid
            parents[right] = left
            members[left].update(members.pop(right))
            invalid.discard(right)
            if bad:
                invalid.add(left)
    cameras, identities, pixels, depths, landmarks = [], [], [], [], []
    for group in getattr(motion, "direct_groups", ()):
        landmark = len(landmarks)
        world = []
        for view, feature in group:
            point = motion.tracked[view][1].points[feature]
            world.append(poses[view][:3, :3] @ point + poses[view][:3, 3])
            cameras.append(view); identities.append(landmark)
            pixels.append(motion.tracked[view][1].pixels[feature]); depths.append(point[2])
        landmarks.append(np.median(world, axis=0))
    for key, group in members.items():
        if key in invalid or len(group) < 2:
            continue
        landmark = len(landmarks)
        world = []
        for view, node in sorted(group.items()):
            point, pixel = features[node]
            world.append(poses[view][:3, :3] @ point + poses[view][:3, 3])
            cameras.append(view); identities.append(landmark)
            pixels.append(pixel); depths.append(point[2])
        landmarks.append(np.median(world, axis=0))
    tag_start = len(cameras)
    tag_keys = []
    frames = getattr(motion, "tag_frames", ())
    groups = {}
    for view in poses:
        if view >= len(frames):
            continue
        for key, tag in frames[view].tags.items():
            for corner in range(4):
                groups.setdefault((key, corner), []).append((view, tag.points[corner], tag.pixels[corner]))
    for (key, corner), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        landmark = len(landmarks)
        world = []
        for view, point, pixel in group:
            world.append(poses[view][:3, :3] @ point + poses[view][:3, 3])
            cameras.append(view); identities.append(landmark)
            pixels.append(pixel); depths.append(point[2]); tag_keys.append(key)
        landmarks.append(np.median(world, axis=0))
    motion.tag_observation_start = tag_start
    motion.tag_observation_keys = tag_keys
    motion.tag_landmark_count = sum(len(group) >= 2 for group in groups.values())
    return (np.asarray(cameras, int), np.asarray(identities, int),
            np.asarray(pixels).reshape(-1, 2), np.asarray(depths),
            np.asarray(landmarks).reshape(-1, 3), len(invalid))


def observed_cameras(owners, landmarks, pixels, depths, local, camera):
    """Find the gauge-connected cameras determined by shared measured points.

    A camera prior or gravity direction cannot establish six degrees of freedom.
    Only non-collinear, spatially distributed, mutually consistent RGB-D points
    contribute here. This checks determination, not accuracy or a probability.
    """
    projected = local[:, :2]/np.maximum(local[:, 2, None], .05)*[camera.fx, camera.fy]+[camera.cx, camera.cy]
    keep = ((local[:, 2] > .05) & (np.linalg.norm(projected-pixels, axis=1) <= 3)
            & (abs(local[:, 2]-depths) <= np.maximum(.03, 3*(.008+.002*depths**2))))
    groups = {}
    for index in np.flatnonzero(keep):
        groups.setdefault(int(landmarks[index]), []).append(index)
    pairs = {}
    for group in groups.values():
        group.sort(key=lambda i: owners[i])
        # A star connects all observers without counting every pair in a long
        # track. Duplicate observations in one camera do not add information.
        for other in group[1:]:
            first = group[0]
            if owners[first] != owners[other]:
                pairs.setdefault((int(owners[first]), int(owners[other])), []).append((first, other))
    adjacency = {}
    for (a, b), indices in pairs.items():
        if len(indices) < 20:
            continue
        determined = True
        for column in (0, 1):
            index = np.array(indices)[:, column]
            points = np.column_stack(((pixels[index]-[camera.cx, camera.cy])/[camera.fx, camera.fy]*depths[index, None], depths[index]))
            eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
            extent = np.ptp(pixels[index], axis=0)/[camera.width, camera.height]
            # Require a broad, non-collinear metric patch along each gauge
            # connection. This is a conservative determination certificate,
            # not the full graph's covariance or an accuracy guarantee.
            determined &= bool(eigenvalues[1]/max(eigenvalues.sum(), 1e-9) > .002
                               and np.prod(extent) >= .04)
        if determined:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
    found, todo = {0}, [0]
    while todo:
        for node in adjacency.get(todo.pop(), set())-found:
            found.add(node); todo.append(node)
    return sorted(found)


def refine_motion_graph(poses, edges, motion, *, views=None, max_seconds=60., max_evaluations=40):
    nodes = sorted(poses)
    if len(nodes) < 3 or motion is None:
        return poses, {"applied": False, "reason": "Insufficient connected motion evidence"}
    ids = {i: n for n, i in enumerate(nodes)}
    base = np.array([poses[i] for i in nodes])
    owners, landmarks, pixels, depths, initial_landmarks, invalid_unions = landmark_observations(poses, edges, motion)
    if not len(owners):
        return poses, {"applied": False, "reason": "No fixed measured RGB-D identities"}
    owners = np.array([ids[i] for i in owners])
    depth_scale = .008 + .002*depths**2
    pixel_scale = np.full(len(owners), 1.5)
    pixel_scale[motion.tag_observation_start:] = .75
    priors = []
    for edge in edges:
        a, b = edge["source"], edge["target"]
        if edge.get("sensor_visual_prior"):
            priors.append((ids[a], ids[b], edge["transform"], .04, np.radians(4.)))
            continue
        measured = motion.appearance_pair(a, b) is not None
        if not edge.get("measurement_only"):
            priors.append((ids[a], ids[b], edge["transform"], .06 if measured else .03,
                           np.radians(4. if measured else 2.)))
        prediction = visual_seed(motion.metadata[a], motion.metadata[b]) if motion.visual else None
        if prediction is not None:
            uncertainty = motion.visual_uncertainty(a, b)
            priors.append((ids[a], ids[b], prediction, uncertainty["translation_scale_m"],
                           np.radians(uncertainty["rotation_scale_deg"])))
    gravity_ids = [ids[i] for i in nodes if motion.gravity[i] is not None]
    gravity_up = np.array([motion.gravity[nodes[i]]["up"] for i in gravity_ids])
    gravity_weight = np.array([np.sqrt(motion.gravity[nodes[i]]["confidence"])
                               / (.05 if motion.gravity[nodes[i]]["verified"] else .12) for i in gravity_ids])
    if gravity_ids:
        world_up = np.mean(np.einsum("nij,nj->ni", base[gravity_ids, :3, :3], gravity_up), axis=0)
        length = np.linalg.norm(world_up)
        if length > .1:
            world_up /= length
        else:
            gravity_ids = []
    depth_pairs = depth_constraints(poses, views) if views is not None else []
    da, db, dp, dq, dn = [], [], [], [], []
    for a, b, p, q, n in depth_pairs:
        da.extend([ids[a]]*len(p)); db.extend([ids[b]]*len(p))
        dp.append(p); dq.append(q); dn.append(n)
    da, db = np.array(da, int), np.array(db, int)
    if dp:
        dp, dq, dn = (np.concatenate(x) for x in (dp, dq, dn))
        surface_scale = .012 + .002*np.maximum(dp[:, 2], dq[:, 2])**2
    planes = plane_observations(poses, views)
    plane_owners, plane_ids, plane_points = [], [], []
    for plane_id, (_, observations) in enumerate(planes):
        for owner, samples in observations:
            plane_owners.extend([ids[owner]]*len(samples))
            plane_ids.extend([plane_id]*len(samples))
            plane_points.append(samples)
    plane_owners, plane_ids = np.asarray(plane_owners, int), np.asarray(plane_ids, int)
    plane_points = np.concatenate(plane_points) if plane_points else np.empty((0, 3))
    plane_scale = .012+.002*plane_points[:, 2]**2
    initial_planes = np.asarray([p for p, _ in planes]).reshape(-1, 3)
    camera_columns = (len(nodes)-1)*6
    landmark_end = camera_columns+initial_landmarks.size
    variable_count = landmark_end+initial_planes.size
    residual_count = len(owners)*3 + len(priors)*12 + len(gravity_ids)*3 + len(da)+len(plane_owners)
    row_parts, column_parts, blocks = [], [], []

    def pattern(rows, columns, selection):
        if not rows.size:
            return
        row_parts.append(rows.reshape(-1)); column_parts.append(columns.reshape(-1))
        blocks.append(selection)

    feature_rows = np.arange(len(owners)*3).reshape(-1, 3)
    live = owners > 0
    pattern(np.broadcast_to(feature_rows[live, :, None], (live.sum(), 3, 6)),
            np.broadcast_to(((owners[live]-1)*6)[:, None, None] + np.arange(6), (live.sum(), 3, 6)),
            ("features_camera", live))
    pattern(np.broadcast_to(feature_rows[:, :, None], (len(owners), 3, 3)),
            np.broadcast_to((camera_columns+landmarks*3)[:, None, None]+np.arange(3), (len(owners), 3, 3)),
            ("features_landmark", None))
    row = len(owners)*3
    for index, (a, b, *_rest) in enumerate(priors):
        for side, node in enumerate((a, b)):
            if node:
                pattern(np.broadcast_to(np.arange(row, row+12)[:, None], (12, 6)),
                        np.broadcast_to(np.arange((node-1)*6, node*6), (12, 6)), ("prior", (index, side)))
        row += 12
    for index, node in enumerate(gravity_ids):
        if node:
            pattern(np.broadcast_to(np.arange(row, row+3)[:, None], (3, 3)),
                    np.broadcast_to(np.arange((node-1)*6, (node-1)*6+3), (3, 3)), ("gravity", index))
        row += 3
    for side, values in enumerate((da, db)):
        keep = values > 0
        pattern(np.broadcast_to(np.arange(row, row+len(da))[keep, None], (keep.sum(), 6)),
                np.broadcast_to(((values[keep]-1)*6)[:, None]+np.arange(6), (keep.sum(), 6)), ("surface", (side, keep)))
    row += len(da)
    keep = plane_owners > 0
    pattern(np.broadcast_to(np.arange(row, row+len(plane_owners))[keep, None], (keep.sum(), 6)),
            np.broadcast_to(((plane_owners[keep]-1)*6)[:, None]+np.arange(6), (keep.sum(), 6)), ("plane_camera", keep))
    pattern(np.broadcast_to(np.arange(row, row+len(plane_owners))[:, None], (len(plane_owners), 3)),
            np.broadcast_to((landmark_end+plane_ids*3)[:, None]+np.arange(3), (len(plane_owners), 3)), ("plane_landmark", None))
    rows, columns = np.concatenate(row_parts), np.concatenate(column_parts)
    initial = np.r_[np.zeros(camera_columns), initial_landmarks.ravel(), initial_planes.ravel()]
    started = time.monotonic()
    best = [initial.copy(), np.inf]
    cache = [None, None, None]

    class BudgetExceeded(Exception):
        pass

    def decode(parameters, derivatives=False):
        increments = np.zeros((len(nodes), 6))
        increments[1:] = parameters[:camera_columns].reshape(-1, 6)
        rotations, rotation_derivatives = [], []
        for n, increment in enumerate(increments):
            rotation, derivative = cv2.Rodrigues(increment[:3])
            rotations.append(rotation @ base[n, :3, :3])
            if derivatives:
                rotation_derivatives.append(derivative.reshape(3, 3, 3) @ base[n, :3, :3])
        return (np.array(rotations), base[:, :3, 3]+increments[:, 3:],
                parameters[camera_columns:landmark_end].reshape(-1, 3), np.array(rotation_derivatives),
                parameters[landmark_end:].reshape(-1, 3))

    def compute(parameters):
        if time.monotonic()-started > max_seconds:
            raise BudgetExceeded()
        if cache[0] is not None and np.array_equal(cache[0], parameters):
            return cache[1], cache[2]
        rotations, translations, points, dr, plane_values = decode(parameters, True)
        delta = points[landmarks]-translations[owners]
        local = np.einsum("nji,nj->ni", rotations[owners], delta)
        z = np.maximum(local[:, 2], .05)
        projected = local[:, :2]/z[:, None]*[motion.camera.fx, motion.camera.fy]+[motion.camera.cx, motion.camera.cy]
        feature_errors = np.column_stack(((projected-pixels)/pixel_scale[:, None], (local[:, 2]-depths)/depth_scale))
        image_jac = np.zeros((len(owners), 3, 3))
        image_jac[:, 0, 0] = motion.camera.fx/z/pixel_scale
        image_jac[:, 1, 1] = motion.camera.fy/z/pixel_scale
        image_jac[:, 0, 2] = -motion.camera.fx*local[:, 0]/z**2/pixel_scale
        image_jac[:, 1, 2] = -motion.camera.fy*local[:, 1]/z**2/pixel_scale
        image_jac[local[:, 2] <= .05, :2, 2] = 0.
        image_jac[:, 2, 2] = 1/depth_scale
        local_camera_jac = np.concatenate((np.einsum("njba,nb->naj", dr[owners], delta),
                                           -rotations[owners].transpose(0, 2, 1)), axis=2)
        camera_jac = image_jac @ local_camera_jac
        landmark_jac = image_jac @ rotations[owners].transpose(0, 2, 1)
        chunks, prior_jacs, gravity_jacs = [feature_errors.ravel()], [], []
        for a, b, prediction, ts, rs in priors:
            scale = rs*np.sqrt(2.)
            orientation = (rotations[a]-rotations[b] @ prediction[:3, :3])/scale
            position = (translations[a]-translations[b]-rotations[b] @ prediction[:3, 3])/ts
            chunks.append(np.r_[orientation.ravel(), position])
            ja, jb = np.zeros((12, 6)), np.zeros((12, 6))
            ja[:9, :3] = dr[a].reshape(3, 9).T/scale
            jb[:9, :3] = -(dr[b] @ prediction[:3, :3]).reshape(3, 9).T/scale
            ja[9:, 3:] = np.eye(3)/ts; jb[9:, 3:] = -np.eye(3)/ts
            jb[9:, :3] = -(dr[b] @ prediction[:3, 3]).T/ts
            prior_jacs.append((ja, jb))
        for index, node in enumerate(gravity_ids):
            up = rotations[node] @ gravity_up[index]
            chunks.append(np.cross(up, world_up)*gravity_weight[index])
            derivative = dr[node] @ gravity_up[index]
            gravity_jacs.append(np.cross(derivative, world_up).T*gravity_weight[index])
        surface_jacs = None
        if len(da):
            wp = np.einsum("nij,nj->ni", rotations[da], dp)+translations[da]
            wq = np.einsum("nij,nj->ni", rotations[db], dq)+translations[db]
            wn = np.einsum("nij,nj->ni", rotations[db], dn)
            difference = wp-wq
            chunks.append(np.sum(difference*wn, axis=1)/surface_scale)
            dpa = np.einsum("njab,nb->nja", dr[da], dp)
            dqb = np.einsum("njab,nb->nja", dr[db], dq)
            dnb = np.einsum("njab,nb->nja", dr[db], dn)
            ja = np.column_stack((np.einsum("nja,na->nj", dpa, wn), wn))/surface_scale[:, None]
            jb = np.column_stack((-np.einsum("nja,na->nj", dqb, wn)
                                   +np.einsum("na,nja->nj", difference, dnb), -wn))/surface_scale[:, None]
            surface_jacs = (ja, jb)
        plane_camera_jac, plane_landmark_jac = None, None
        if len(plane_owners):
            values = plane_values[plane_ids]
            lengths = np.maximum(np.linalg.norm(values, axis=1), 1e-6)
            normals = values/lengths[:, None]
            world = np.einsum("nij,nj->ni", rotations[plane_owners], plane_points)+translations[plane_owners]
            chunks.append((np.sum(world*normals, axis=1)-lengths)/plane_scale)
            derivative = np.einsum("njab,nb->nja", dr[plane_owners], plane_points)
            plane_camera_jac = np.column_stack((np.einsum("nja,na->nj", derivative, normals), normals))/plane_scale[:, None]
            plane_landmark_jac = (world/lengths[:, None]-(np.sum(world*values, axis=1)/lengths**3)[:, None]*values-normals)/plane_scale[:, None]
        values = []
        for name, selection in blocks:
            if name == "features_camera": values.append(camera_jac[selection].ravel())
            elif name == "features_landmark": values.append(landmark_jac.ravel())
            elif name == "prior": values.append(prior_jacs[selection[0]][selection[1]].ravel())
            elif name == "gravity": values.append(gravity_jacs[selection].ravel())
            elif name == "surface": values.append(surface_jacs[selection[0]][selection[1]].ravel())
            elif name == "plane_camera": values.append(plane_camera_jac[selection].ravel())
            elif name == "plane_landmark": values.append(plane_landmark_jac.ravel())
        result = np.concatenate(chunks)
        jacobian = coo_matrix((np.concatenate(values), (rows, columns)), shape=(residual_count, variable_count)).tocsr()
        cost = float(np.sum(2*(np.sqrt(1+result**2)-1)))
        if cost < best[1]:
            best[:] = [parameters.copy(), cost]
        cache[:] = [parameters.copy(), result, jacobian]
        return result, jacobian

    compute(initial)
    initial_cost = best[1]
    budget_limited = False
    try:
        result = least_squares(lambda p: compute(p)[0], initial, jac=lambda p: compute(p)[1],
                               loss="soft_l1", x_scale="jac", max_nfev=max_evaluations,
                               ftol=1e-4, xtol=1e-4, gtol=1e-4)
        budget_limited = result.nfev >= max_evaluations
    except BudgetExceeded:
        budget_limited = True
    rotations, translations, points, _, _ = decode(best[0])
    local = np.einsum("nji,nj->ni", rotations[owners], points[landmarks]-translations[owners])
    determined = observed_cameras(owners, landmarks, pixels, depths, local, motion.camera)
    refined = {}
    for i, node in enumerate(nodes):
        pose = np.eye(4); pose[:3, :3] = rotations[i]; pose[:3, 3] = translations[i]
        refined[node] = pose
    refined[nodes[0]] = poses[nodes[0]].copy()
    projected = local[:, :2]/np.maximum(local[:, 2, None], .05)*[motion.camera.fx, motion.camera.fy]+[motion.camera.cx, motion.camera.cy]
    measured = np.column_stack(((pixels-[motion.camera.cx, motion.camera.cy])
                                /[motion.camera.fx, motion.camera.fy]*depths[:, None], depths))
    pixel_error = np.linalg.norm(projected-pixels, axis=1)
    good = (local[:, 2] > .05) & (pixel_error <= 3) & (np.linalg.norm(local-measured, axis=1) <= .025)
    tag_groups = {}
    for row, key in enumerate(motion.tag_observation_keys, motion.tag_observation_start):
        tag_groups.setdefault((nodes[owners[row]], key), []).append(row)
    tag_support = {}
    for (view, key), rows in tag_groups.items():
        support = tag_support.setdefault(view, {"observed_tags": 0, "inlier_tags": 0, "inlier_identities": []})
        support["observed_tags"] += 1
        if len(rows) == 4 and good[rows].all():
            support["inlier_tags"] += 1
            support["inlier_identities"].append({"dictionary": key[0], "id": key[1]})
    for support in tag_support.values():
        support["support_fraction"] = support["inlier_tags"] / support["observed_tags"]
        support["accepted"] = support["inlier_tags"] >= 1 and support["support_fraction"] >= .65
    return refined, {"applied": best[1] < initial_cost, "landmarks": len(initial_landmarks),
                     "feature_observations": len(owners), "invalid_identity_unions": invalid_unions,
                     "gravity_views": len(gravity_ids), "nonlocal_depth_pairs": len(depth_pairs),
                     "depth_constraints": len(da), "initial_cost": initial_cost, "final_cost": best[1],
                     "shared_depth_planes": len(planes), "plane_observations": len(plane_owners),
                     "budget_limited": budget_limited, "elapsed_s": time.monotonic()-started,
                     "apriltag_landmarks": motion.tag_landmark_count,
                     "apriltag_observations": len(motion.tag_observation_keys),
                     "apriltag_camera_support": tag_support,
                     "globally_observed_cameras": [nodes[i] for i in determined],
                     "weights_are_calibrated_probabilities": False}
