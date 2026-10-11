"""Portable UV textures with depth-tested RGB projection, including ARM CPUs.

Geometry is finalized before UV generation. Unobserved texels retain fused
vertex color; original RGB is used only where measured depth agrees with mesh.
"""

import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt

from shared.calibration import prepare_metric_depth, prepare_rgbd, project_rgb
from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS


def _winding_cuts(faces):
    """Find face cuts that make shared-edge winding constraints consistent."""
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    adjacency = [[] for _ in faces]
    forward = directed[:, 0] < directed[:, 1]
    for edge in np.flatnonzero(counts == 2):
        a, b = order[starts[edge] : starts[edge] + 2]
        flip = bool(forward[a] == forward[b])
        adjacency[a // 3].append((b // 3, flip))
        adjacency[b // 3].append((a // 3, flip))
    winding = np.full(len(faces), -1, dtype=np.int8)
    cuts = set()
    for root in range(len(faces)):
        if winding[root] >= 0 or root in cuts:
            continue
        winding[root] = 0
        pending = [root]
        while pending:
            face = pending.pop()
            if face in cuts:
                continue
            for neighbor, flip in adjacency[face]:
                if neighbor in cuts:
                    continue
                wanted = int(winding[face]) ^ flip
                if winding[neighbor] < 0:
                    winding[neighbor] = wanted
                    pending.append(neighbor)
                elif winding[neighbor] != wanted:
                    cuts.add(int(neighbor))
    return cuts


def _make_uv_manifold(mesh):
    """Cut ambiguous connections on an export copy without deleting faces."""
    bad_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    bad_vertices = mesh.get_non_manifold_vertices()
    report = {
        "non_manifold_edges": len(bad_edges),
        "non_manifold_vertices": len(bad_vertices),
        "isolated_triangles": 0,
        "winding_conflict_triangles": 0,
        "added_vertices": 0,
        "removed_triangles": 0,
    }
    if not len(bad_edges) and not len(bad_vertices) and mesh.is_orientable():
        mesh.orient_triangles()
        return report

    vertices = np.asarray(mesh.vertices).copy()
    colors = np.asarray(mesh.vertex_colors).copy()
    faces = np.asarray(mesh.triangles).copy()
    source_ids = list(range(len(vertices)))
    if len(bad_edges):
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, inverse, counts = np.unique(
            edges, axis=0, return_inverse=True, return_counts=True
        )
        order = np.argsort(inverse, kind="stable")
        starts = np.cumsum(counts) - counts
        # Keep two faces per shared edge; give any extra sheets their own
        # corners. Their positions and colors remain exactly the same.
        isolate = set()
        for edge_id in np.flatnonzero(counts > 2):
            incident = order[starts[edge_id] : starts[edge_id] + counts[edge_id]] // 3
            isolate.update(incident[2:].tolist())
        for face_id in sorted(isolate):
            original = faces[face_id].copy()
            faces[face_id] = np.arange(len(source_ids), len(source_ids) + 3)
            source_ids.extend(original.tolist())
        report["isolated_triangles"] = len(isolate)
        mesh.vertices = o3d.utility.Vector3dVector(vertices[source_ids])
        mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Manifold scans can still contain a twisted sheet with contradictory
    # winding. Cut that connection for UVs instead of rejecting the scan.
    if not mesh.is_orientable():
        cuts = _winding_cuts(faces)
        for face_id in sorted(cuts):
            original = faces[face_id].copy()
            faces[face_id] = np.arange(len(source_ids), len(source_ids) + 3)
            source_ids.extend(source_ids[int(vertex)] for vertex in original)
        report["winding_conflict_triangles"] = len(cuts)
        mesh.vertices = o3d.utility.Vector3dVector(vertices[source_ids])
        mesh.triangles = o3d.utility.Vector3iVector(faces)

    # A vertex can still join disjoint triangle fans (a bow-tie). Give each
    # fan its own coincident vertex so the UV atlas sees separate sheets.
    for vertex in mesh.get_non_manifold_vertices():
        incident, corners = np.where(faces == vertex)
        neighbors = {}
        for local, face_id in enumerate(incident):
            for other in faces[face_id]:
                if other != vertex:
                    neighbors.setdefault(int(other), []).append(local)
        adjacency = [set() for _ in incident]
        for joined in neighbors.values():
            for local in joined:
                adjacency[local].update(joined)
        remaining = set(range(len(incident)))
        first = True
        while remaining:
            pending = [remaining.pop()]
            component = []
            while pending:
                local = pending.pop()
                component.append(local)
                connected = adjacency[local] & remaining
                remaining.difference_update(connected)
                pending.extend(connected)
            if first:
                first = False
                continue
            new_vertex = len(source_ids)
            source_ids.append(source_ids[vertex])
            faces[incident[component], corners[component]] = new_vertex

    mesh.vertices = o3d.utility.Vector3dVector(vertices[source_ids])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if len(colors) == len(vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors[source_ids])
    report["added_vertices"] = len(source_ids) - len(vertices)
    if (
        not mesh.is_edge_manifold(allow_boundary_edges=True)
        or not mesh.is_vertex_manifold()
        or not mesh.orient_triangles()
    ):
        raise ValueError("Could not separate non-manifold connections for texture export")
    return report


def _views(engine, max_views):
    # Cover the whole trajectory; choose the sharpest synchronized image in
    # each interval rather than loading every image into a projection backend.
    candidates = [
        p
        for p in engine.poses
        if abs(engine.frame_metadata[p[0]].get("rgb_depth_delta_ms") or 0) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS
    ]
    selected = []
    for group in np.array_split(
        np.arange(len(candidates)), min(max_views, len(candidates)) or 1
    ):
        if not len(group):
            continue

        def sharpness(j):
            rgb = engine.raw_frames[candidates[j][0]][0].astype(np.float32).mean(axis=2)
            return float(
                np.mean(np.abs(np.diff(rgb, axis=0)))
                + np.mean(np.abs(np.diff(rgb, axis=1)))
            )

        selected.append(candidates[max(group, key=sharpness)])
    return selected


def _fallback_uv_atlas(mesh, size, max_faces=1000):
    """Atlas independent repaired charts; subdivide only charts rejected by UVAtlas."""
    tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    partitions = tensor.pca_partition(max_faces=max_faces)
    groups = tensor.triangle["partition_ids"].numpy()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    charts = []
    retries = 0

    def unwrap(ids):
        nonlocal retries
        if len(ids) == 1:
            charts.append((ids, np.array([[[0.05, 0.05], [0.95, 0.05], [0.5, 0.95]]])))
            return
        used, remap = np.unique(faces[ids], return_inverse=True)
        part = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices[used]),
            o3d.utility.Vector3iVector(remap.reshape(-1, 3)),
        )
        _make_uv_manifold(part)
        local = o3d.t.geometry.TriangleMesh.from_legacy(part)
        try:
            local.compute_uvatlas(size=256, parallel_partitions=1, nthreads=1)
        except RuntimeError as exc:
            if "Non-manifold mesh" not in str(exc):
                raise
            retries += 1
            middle = len(ids) // 2
            unwrap(ids[:middle])
            unwrap(ids[middle:])
            return
        # Local winding repair can reorder corners. Map UVs back to the
        # original triangles so geometry and attribute interpolation agree.
        original = vertices[faces[ids]]
        repaired = np.asarray(part.vertices)[np.asarray(part.triangles)]
        distance = ((original[:, :, None] - repaired[:, None, :]) ** 2).sum(axis=-1)
        corner = distance.argmin(axis=2)
        uv = local.triangle["texture_uvs"].numpy()
        charts.append((ids, uv[np.arange(len(ids))[:, None], corner]))

    for group in range(partitions):
        unwrap(np.flatnonzero(groups == group))
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    rectangles = _chart_rectangles([areas[ids].sum() for ids, _ in charts])
    uv = np.empty((len(faces), 3, 2), dtype=np.float32)
    for chart, (ids, local_uv) in enumerate(charts):
        origin, cell = rectangles[chart]
        padding = np.minimum(2 / size, cell * 0.1)
        uv[ids] = origin + padding + local_uv * (cell - 2 * padding)
    tensor.triangle["texture_uvs"] = o3d.core.Tensor(uv)
    return tensor, {"method": "repaired_chart_tiles", "packing": "surface_area",
                    "charts": len(charts), "subdivisions": retries}


def _chart_rectangles(areas):
    """Disjoint, area-proportional rectangles, avoiding equal tiles for tiny cuts."""
    areas = np.asarray(areas, float)
    areas = np.maximum(areas, max(areas.sum() * 1e-6, 1e-12))
    result = [None] * len(areas)

    def divide(ids, origin, extent):
        if len(ids) == 1:
            result[ids[0]] = (origin, extent)
            return
        order = ids[np.argsort(-areas[ids], kind="stable")]
        cumulative = np.cumsum(areas[order])
        cut = int(np.argmin(np.abs(cumulative[:-1] - cumulative[-1] / 2))) + 1
        fraction = cumulative[cut - 1] / cumulative[-1]
        axis = int(extent.argmax())
        first, second = extent.copy(), extent.copy()
        first[axis] *= fraction
        second[axis] *= 1 - fraction
        shifted = origin.copy()
        shifted[axis] += first[axis]
        divide(order[:cut], origin, first)
        divide(order[cut:], shifted, second)

    divide(np.arange(len(areas)), np.zeros(2), np.ones(2))
    return result


def _prepare_texture_views(engine, views):
    prepared = []
    for index, pose in views:
        source_rgb, source_depth = engine.raw_frames[index]
        if engine.settings.sensor_calibration:
            # Projection uses the native RGB lens directly. Registering a dense
            # RGB-D image here allocated work which was immediately discarded.
            rgb = source_rgb
            depth = prepare_metric_depth(source_depth, engine.settings)
        else:
            rgb, depth = prepare_rgbd(source_rgb, source_depth, engine.settings)
        prepared.append((pose, rgb, depth.astype(np.float32) / 1000))
    return prepared


def _project(
    world, normals, rgb, depth, pose, camera_model, near, far,
    sensor_calibration=None, rgb_camera=None,
):
    extrinsic = np.linalg.inv(pose)
    camera = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = camera[:, 2]
    u = camera[:, 0] * camera_model.fx / np.maximum(z, 1e-6) + camera_model.cx
    v = camera[:, 1] * camera_model.fy / np.maximum(z, 1e-6) + camera_model.cy
    x = np.clip(np.rint(u), 0, camera_model.width - 1).astype(int)
    y = np.clip(np.rint(v), 0, camera_model.height - 1).astype(int)
    observed = depth[y, x]
    if sensor_calibration is not None:
        pixels, rgb_z = project_rgb(camera * 1000, sensor_calibration, rgb_camera)
        color_u, color_v = pixels.T
        center_ir = (
            -np.asarray(sensor_calibration.rotation).T
            @ np.asarray(sensor_calibration.translation_mm) / 1000
        )
        center_world = pose[:3, :3] @ center_ir + pose[:3, 3]
    else:
        color_u, color_v, rgb_z = u, v, z
        center_world = pose[:3, 3]
    direction = center_world - world
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    cosine = np.abs(np.sum(normals * direction, axis=1))
    mask = (
        (z > near)
        & (z < far)
        & (u >= 0)
        & (v >= 0)
        & (u < camera_model.width - 1)
        & (v < camera_model.height - 1)
        & (observed > 0)
        & (np.abs(observed - z) <= np.maximum(0.015, 0.01 * z))
        & (cosine > 0.25)
        & (rgb_z > 0)
        & (color_u >= 0)
        & (color_u < rgb.shape[1] - 1)
        & (color_v >= 0)
        & (color_v < rgb.shape[0] - 1)
    )
    ids = np.flatnonzero(mask)
    colors = np.zeros((len(world), 3), np.float32)
    weights = np.zeros(len(world), np.float32)
    x0, y0 = np.floor(color_u[ids]).astype(int), np.floor(color_v[ids]).astype(int)
    dx, dy = (color_u[ids] - x0)[:, None], (color_v[ids] - y0)[:, None]
    colors[ids] = (
        rgb[y0, x0] * (1 - dx) * (1 - dy)
        + rgb[y0, x0 + 1] * dx * (1 - dy)
        + rgb[y0 + 1, x0] * (1 - dx) * dy
        + rgb[y0 + 1, x0 + 1] * dx * dy
    ) / 255
    weights[ids] = cosine[ids] ** 2 / np.maximum(z[ids] ** 2, 0.1)
    return colors, weights


def _face_neighbors(mesh):
    """Surface adjacency, independent of UV chart cuts; do not cross sharp folds."""
    faces = np.asarray(mesh.triangles)
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    shared = np.flatnonzero(counts == 2)
    a, b = order[starts[shared]], order[starts[shared] + 1]
    triangles = np.asarray(mesh.vertices)[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    keep = np.sum(normals[a // 3] * normals[b // 3], axis=1) > 0.5
    a, b = a[keep], b[keep]
    neighbors = np.full((len(faces), 3), -1, np.int32)
    neighbors[a // 3, a % 3] = b // 3
    neighbors[b // 3, b % 3] = a // 3
    return neighbors


def _select_face_sources(scores, neighbors):
    """Favor connected photo regions without assigning an unobserved source.

    Scores summarize all atlas samples of each face, including missing depth.
    A short diffusion supplies a coherent initial labeling, then graph-colored
    coordinate descent reduces a visibility-constrained Potts energy. Updating
    independent faces avoids the oscillation of simultaneous neighbor voting.
    """
    observed = scores > 0
    best = scores.max(axis=0)
    relative = scores / np.maximum(best, 1e-12)
    safe = np.maximum(neighbors, 0)
    joined = (neighbors >= 0) & (best[safe] > 0)
    degree = np.maximum(joined.sum(axis=1), 1)
    smooth = relative.copy()
    for _ in range(5):
        average = (smooth[:, safe] * joined[None]).sum(axis=2) / degree
        smooth = np.where(observed, 0.5 * relative + 0.5 * average, 0)
    initial = scores.argmax(axis=0)
    labels = smooth.argmax(axis=0).astype(np.int32)
    labels[best == 0] = -1
    costs = -np.log(np.maximum(relative, 1e-12))
    costs[~observed] = np.inf
    # A triangle has at most three manifold neighbors, so four colors suffice.
    groups = np.full(len(best), -1, np.int8)
    for face, adjacent in enumerate(neighbors):
        occupied = set(groups[adjacent[adjacent >= 0]])
        groups[face] = next(color for color in range(4) if color not in occupied)
    views = np.arange(len(scores))[:, None, None]
    iterations = 0
    for iterations in range(1, 13):
        changed = 0
        for color in range(4):
            ids = np.flatnonzero((groups == color) & (best > 0))
            adjacent_labels = labels[safe[ids]]
            penalty = 0.35 * (
                (views != adjacent_labels[None]) * joined[ids][None]
            ).sum(axis=2)
            energy = costs[:, ids] + penalty
            chosen = energy.argmin(axis=0)
            # Retain the current source on ties, including identical photos.
            improve = energy[chosen, np.arange(len(ids))] < (
                energy[labels[ids], np.arange(len(ids))] - 1e-6
            )
            changed += int(improve.sum())
            labels[ids[improve]] = chosen[improve]
        if not changed:
            break
    edges = joined & (best[:, None] > 0)
    return labels, {
        "method": "connected_surface_regions",
        "optimization_passes": iterations,
        "independent_face_boundaries": int(((initial[:, None] != initial[safe]) & edges).sum() // 2),
        "region_boundaries": int(((labels[:, None] != labels[safe]) & edges).sum() // 2),
    }


def _seam_partners(positions, sources, valid, size, width=3, *, face_ids=None, neighbors=None):
    """Find one neighboring photo in a narrow band, rejecting unrelated UV islands."""
    source_map = sources.reshape(size, size)
    world = positions.reshape(size, size, 3)
    observed = valid.reshape(size, size) & (source_map >= 0)
    boundary = np.zeros((size, size), bool)
    partner = np.full((size, size), -1, np.int32)
    for axis in (0, 1):
        first = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        second = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        separation = np.linalg.norm(world[first] - world[second], axis=2)
        adjacent = observed[first] & observed[second]
        if face_ids is not None:
            face_map = face_ids.reshape(size, size)
            a, b = face_map[first], face_map[second]
            linked = (a == b) | np.any(neighbors[np.maximum(a, 0)] == b[..., None], axis=2)
            adjacent &= linked
        # Estimate local texel spacing, rather than allowing neighboring atlas
        # tiles to blend different surfaces at their packed chart boundaries.
        spacing = np.median(separation[adjacent]) if adjacent.any() else 0
        seam = adjacent & (source_map[first] != source_map[second]) & (
            separation <= max(1e-6, spacing * 3)
        )
        boundary[first] |= seam
        boundary[second] |= seam
        partner[first][seam] = source_map[second][seam]
        partner[second][seam] = source_map[first][seam]
    if not boundary.any():
        return partner.reshape(-1), np.zeros(size * size, np.float32)
    distance, nearest = distance_transform_edt(~boundary, return_indices=True)
    origin = tuple(nearest)
    separation = np.linalg.norm(world - world[origin], axis=2)
    local_spacing = np.maximum(
        np.linalg.norm(world[origin] - world[np.clip(nearest[0] + 1, 0, size - 1), nearest[1]], axis=2),
        np.linalg.norm(world[origin] - world[nearest[0], np.clip(nearest[1] + 1, 0, size - 1)], axis=2),
    )
    band = observed & (distance < width) & (source_map == source_map[origin]) & (
        separation <= np.maximum(local_spacing * (width + 1), 1e-6)
    )
    result = np.where(band, partner[origin], -1).reshape(-1)
    amount = np.where(band, 0.5 * (1 - distance / width), 0).astype(np.float32).reshape(-1)
    return result, amount


def _interpolate_face_corrections(positions, face_ids, triangles, corrections):
    """Interpolate original face corners, independent of UV winding/vertex splits."""
    points = triangles[face_ids]
    ab, ac = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    relative = positions - points[:, 0]
    aa = np.einsum("ij,ij->i", ab, ab)
    cc = np.einsum("ij,ij->i", ac, ac)
    cross = np.einsum("ij,ij->i", ab, ac)
    denominator = np.maximum(aa * cc - cross * cross, 1e-30)
    ra = np.einsum("ij,ij->i", relative, ab)
    rc = np.einsum("ij,ij->i", relative, ac)
    second = (cc * ra - cross * rc) / denominator
    third = (aa * rc - cross * ra) / denominator
    barycentric = np.column_stack((1 - second - third, second, third))
    # Atlas padding can extrapolate just outside a face. Keep color adjustment
    # inside its corner range rather than amplifying it in the gutter.
    barycentric = np.clip(barycentric, 0, 1)
    barycentric /= np.maximum(barycentric.sum(axis=1, keepdims=True), 1e-30)
    return np.einsum("ij,ijk->ik", barycentric, corrections[face_ids])


def make_textured_mesh(
    engine,
    size=1024,
    max_triangles=200000,
    max_views=24,
    use_images=True,
    exposure_correction=False,
    blend_mode="blend",
    *,
    _prepared=None,
    _selected_views=None,
    _exposure=None,
    _face_sources=None,
    _color_field=None,
    _seam_report=None,
):
    if engine.mesh is None or not len(engine.mesh.triangles):
        raise ValueError("Build a mesh before texturing")
    if size not in (256, 512, 1024, 2048):
        raise ValueError("Texture size must be 256, 512, 1024, or 2048")
    if not 100 <= max_triangles <= 200000 or not 1 <= max_views <= 64:
        raise ValueError("Invalid texture triangle or view budget")
    if blend_mode not in ("blend", "best"):
        raise ValueError("Texture blend_mode must be blend or best")
    mesh = copy.deepcopy(engine.mesh)
    original_triangles = len(mesh.triangles)
    if original_triangles > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if _color_field is not None and (len(_color_field.face_indices) != len(mesh.triangles) or
                                     len(_color_field.view_indices) != len(_prepared)):
        raise ValueError("Color field must match the page faces and prepared photos")
    topology_report = _make_uv_manifold(mesh)
    if _face_sources is not None and len(_face_sources) != len(mesh.triangles):
        raise ValueError("Whole-export labels must preserve page face order")
    mesh.compute_vertex_normals()
    tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    partitions = max(1, min(256, int(np.ceil(len(mesh.triangles) / 1000))))
    atlas_report = {"method": "native_partitioned"}
    try:
        tensor.compute_uvatlas(size=size, parallel_partitions=partitions, nthreads=4)
    except RuntimeError as exc:
        if "Non-manifold mesh" not in str(exc):
            raise
        tensor, atlas_report = _fallback_uv_atlas(mesh, size)
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) != len(mesh.vertices):
        colors = np.full((len(mesh.vertices), 3), 0.6, np.float32)
    tensor.vertex["albedo"] = o3d.core.Tensor(np.clip(colors, 0, 1).astype(np.float32))
    baked = tensor.bake_vertex_attr_textures(
        size,
        {"albedo", "positions", "normals"},
        fill=float("nan"),
        update_material=False,
    )
    positions = baked["positions"].numpy().reshape(-1, 3)
    normals = baked["normals"].numpy().reshape(-1, 3)
    valid = np.all(np.isfinite(positions), axis=1)
    albedo = np.nan_to_num(baked["albedo"].numpy(), nan=0).reshape(-1, 3)
    tensor.triangle["source_face"] = o3d.core.Tensor(np.arange(len(mesh.triangles), dtype=np.int64))
    face_ids = tensor.bake_triangle_attr_textures(
        size, {"source_face"}, fill=-1, update_material=False,
    )["source_face"].numpy().reshape(-1)
    valid &= face_ids >= 0
    views = _selected_views
    if views is None:
        views = _views(engine, max_views) if use_images else []
    c = engine.settings.camera
    valid_indices = np.flatnonzero(valid)
    prepared = _prepare_texture_views(engine, views) if _prepared is None else _prepared
    exposure_report = {
        "applied": False,
        "reason": "Fewer than two source photos or no observed surface" if exposure_correction else "Not requested",
    }
    gains = np.ones((len(views), 3), np.float32)
    if _exposure is not None:
        gains, exposure_report = _exposure
        exposure_report = dict(exposure_report)
    elif exposure_correction and len(views) > 1 and len(valid_indices):
        from .photometric import estimate_gains

        sample_ids = valid_indices[:: max(1, int(np.ceil(len(valid_indices) / 8192)))]
        samples, visible = [], []
        for pose, rgb, depth in prepared:
            values, weight = _project(
                positions[sample_ids],
                normals[sample_ids],
                rgb,
                depth,
                pose,
                c,
                engine.settings.near_m,
                engine.settings.far_m,
                sensor_calibration=engine.settings.sensor_calibration,
                rgb_camera=engine.settings.rgb_camera,
            )
            samples.append(values)
            visible.append(weight > 0)
        gains, exposure_report = estimate_gains(
            np.asarray(samples), np.asarray(visible)
        )
    exposure_report["requested"] = bool(exposure_correction)
    best_weight = np.zeros(len(positions), np.float32)
    sources = np.full(len(positions), -1, np.int32)
    face_scores = np.zeros((len(views), len(mesh.triangles)), np.float32)
    for view_index, (pose, rgb, depth) in enumerate(prepared):
        for start in range(0, len(valid_indices), 65536):
            ids = valid_indices[start : start + 65536]
            values, weight = _project(
                positions[ids],
                normals[ids],
                rgb,
                depth,
                pose,
                c,
                engine.settings.near_m,
                engine.settings.far_m,
                sensor_calibration=engine.settings.sensor_calibration,
                rgb_camera=engine.settings.rgb_camera,
            )
            values = np.clip(values * gains[view_index], 0, 1)
            better = weight > best_weight[ids]
            chosen = ids[better]
            albedo[chosen] = values[better]
            best_weight[chosen] = weight[better]
            sources[chosen] = view_index
            if _face_sources is None:
                face_scores[view_index] += np.bincount(
                    face_ids[ids], weights=weight, minlength=len(mesh.triangles),
                )
    projected = best_weight > 0
    selection_report = {"method": "no_source_images"}
    softened = np.zeros(len(positions), bool)
    seam_report = dict(_seam_report) if _seam_report is not None else {
        "applied": False, "reason": "Fewer than two source photos" if blend_mode == "blend" else "Sharp boundaries requested",
    }
    corrected_texels, clipped_texels, corrected_fallback_texels = 0, 0, 0
    if views and valid.any():
        counts = np.bincount(face_ids[valid], minlength=len(mesh.triangles))
        face_scores /= np.maximum(counts, 1)
        neighbors = _face_neighbors(mesh)
        if _face_sources is None:
            face_sources, selection_report = _select_face_sources(face_scores, neighbors)
        else:
            face_sources = _face_sources
            selection_report = {"method": "whole_export_face_regions"}
        if _color_field is None and _seam_report is None and blend_mode == "blend" and len(prepared) > 1:
            from .surface_color import estimate_color_field

            _color_field, seam_report = estimate_color_field(
                mesh, prepared, engine.settings, gains=gains,
            )
        preferred = np.full(len(positions), -1, np.int32)
        preferred[valid] = face_sources[face_ids[valid]]
        # If a region's photo has missing depth at an individual texel, keep
        # the depth-tested best photo from the first pass there.
        for view_index, (pose, rgb, depth) in enumerate(prepared):
            selected = np.flatnonzero(valid & (preferred == view_index))
            for start in range(0, len(selected), 65536):
                ids = selected[start : start + 65536]
                values, weight = _project(
                    positions[ids], normals[ids], rgb, depth, pose, c,
                    engine.settings.near_m, engine.settings.far_m,
                    sensor_calibration=engine.settings.sensor_calibration,
                    rgb_camera=engine.settings.rgb_camera,
                )
                visible = weight > 0
                chosen = ids[visible]
                values = values[visible] * gains[view_index]
                albedo[chosen] = np.clip(values, 0, 1)
                sources[ids[visible]] = view_index
        if seam_report.get("applied", False):
            # Correct the actual depth-visible photo, including best-view
            # fallback where the face's preferred photo has missing depth.
            for view_index in range(len(prepared)):
                selected = np.flatnonzero(valid & (sources == view_index))
                for start in range(0, len(selected), 65536):
                    ids = selected[start : start + 65536]
                    adjustments = _color_field.evaluate(positions[ids], face_ids[ids], view_index)
                    values = albedo[ids] * np.exp(adjustments)
                    changed = np.any(np.abs(adjustments) > 1e-6, axis=1)
                    corrected_texels += int(changed.sum())
                    corrected_fallback_texels += int(np.sum(changed & (sources[ids] != preferred[ids])))
                    clipped_texels += int(np.any(values > 1, axis=1).sum())
                    albedo[ids] = np.clip(values, 0, 1)
        if blend_mode == "blend" and not seam_report.get("applied", False):
            partners, amounts = _seam_partners(
                positions, sources, valid, size, face_ids=face_ids, neighbors=neighbors,
            )
            for view_index, (pose, rgb, depth) in enumerate(prepared):
                selected = np.flatnonzero(partners == view_index)
                for start in range(0, len(selected), 65536):
                    ids = selected[start : start + 65536]
                    values, weight = _project(
                        positions[ids], normals[ids], rgb, depth, pose, c,
                        engine.settings.near_m, engine.settings.far_m,
                        sensor_calibration=engine.settings.sensor_calibration,
                        rgb_camera=engine.settings.rgb_camera,
                    )
                    values = np.clip(values * gains[view_index], 0, 1)
                    # Grossly disagreeing samples usually contain shifted
                    # detail/occlusion; averaging them would create ghosting.
                    compatible = (weight > 0) & (np.max(np.abs(values - albedo[ids]), axis=1) < 0.12)
                    chosen = ids[compatible]
                    mix = amounts[chosen, None]
                    albedo[chosen] = albedo[chosen] * (1 - mix) + values[compatible] * mix
                    softened[chosen] = True
        selection_report.update(
            source_faces=[int(np.sum(face_sources == i)) for i in range(len(views))],
            fallback_texels=int(np.sum(projected & (sources != preferred))),
        )
    image = Image.fromarray(
        np.clip(albedo.reshape(size, size, 3) * 255, 0, 255).astype(np.uint8)
    )
    # OBJ supports corner UVs. Trimesh/GLB use vertex UVs, so duplicate corners
    # deliberately rather than merging vertices across UV seams.
    faces = np.asarray(mesh.triangles)
    uv = tensor.triangle["texture_uvs"].numpy().reshape(-1, 2)
    material = trimesh.visual.material.SimpleMaterial(
        image=image, name="scan_material", diffuse=[255, 255, 255, 255]
    )
    result = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[faces].reshape(-1, 3),
        faces=np.arange(faces.size).reshape(-1, 3),
        vertex_normals=np.asarray(mesh.vertex_normals)[faces].reshape(-1, 3),
        visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    report = {
        "method": "depth_tested_rgb" if views else "vertex_color_bake",
        "texture_size": size,
        "blend_mode": blend_mode,
        "source_selection": selection_report,
        "seam_blending": {"width_texels": 3 if blend_mode == "blend" and not seam_report.get("applied", False) else 0,
                          "softened_texels": int(softened.sum()),
                          "projected_texels": int(projected.sum())},
        "exposure_correction": exposure_report,
        "seam_leveling": dict(seam_report, corrected_texels=corrected_texels, clipped_texels=clipped_texels,
                              corrected_fallback_texels=corrected_fallback_texels),
        "atlas_partitions": partitions,
        "atlas": atlas_report,
        "original_triangles": original_triangles,
        "export_triangles": len(faces),
        "topology_repair": topology_report,
        "views": [i for i, _ in views],
        "projected_fraction": float(projected.sum() / max(1, valid.sum())),
        "mapped_texels": int(valid.sum()),
        "projected_texels": int(projected.sum()),
        "atlas_occupancy": float(valid.sum() / size**2),
        "unobserved_fallback": "fused vertex colors",
        "depth_unit": "metres",
    }
    result.metadata["texture_report"] = report
    return result, report


def _single_object_gltf(tree):
    """Keep page material primitives in one scan mesh/node.

    Baked page vertices already use scan coordinates. Only identity transforms
    may be collapsed; silently dropping a transform would corrupt geometry.
    """
    identity = {"matrix": np.eye(4).reshape(-1), "translation": [0, 0, 0],
                "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]}
    for node in tree.get("nodes", []):
        for key, value in identity.items():
            if not np.allclose(node.get(key, value), value, rtol=0, atol=1e-12):
                raise ValueError("Texture pages must share scan coordinates before export")
    primitives = [primitive for mesh in tree["meshes"] for primitive in mesh["primitives"]]
    scene = dict(tree["scenes"][tree.get("scene", 0)])
    report = scene.get("extras", {}).get("texture_report")
    if report is None:
        report = tree["meshes"][0].get("extras", {}).get("texture_report")
    scan = {"name": "scan", "primitives": primitives}
    if report is not None:
        scan["extras"] = {"texture_report": report}
    tree["meshes"] = [scan]
    tree["nodes"] = [{"name": "scan", "mesh": 0}]
    scene["nodes"] = [0]
    tree["scenes"] = [scene]
    tree["scene"] = 0


def _single_object_obj(obj):
    """One OBJ object with per-face usemtl assignments, preserving all indices."""
    lines = [line for line in obj.splitlines() if not line.startswith(("o ", "g "))]
    return "o scan\n" + "\n".join(lines) + "\n"


def export_texture(engine, filepath, fmt="glb", **options):
    if fmt not in ("glb", "obj.zip"):
        raise ValueError("Texture format must be glb or obj.zip")
    from .texture_pages import make_textured_asset

    mesh, report = make_textured_asset(engine, **options)
    report["export_objects"] = 1
    mesh.metadata["texture_report"] = report
    if fmt == "glb":
        Path(filepath).write_bytes(trimesh.exchange.gltf.export_glb(
            mesh, tree_postprocessor=_single_object_gltf,
        ))
    else:
        obj, assets = trimesh.exchange.obj.export_obj(
            mesh,
            include_normals=True,
            return_texture=True,
            write_texture=False,
            mtl_name="scan.mtl",
        )
        obj = _single_object_obj(obj)
        with zipfile.ZipFile(
            filepath, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("scan.obj", obj)
            for name, data in assets.items():
                archive.writestr(name, data)
            archive.writestr(
                "texture-report.json", json.dumps(report, indent=2, allow_nan=False)
            )
    return report
