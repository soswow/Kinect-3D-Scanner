"""Portable UV textures with depth-tested RGB projection, including ARM CPUs.

Geometry is finalized before UV generation. Unobserved texels retain fused
vertex color; original RGB is used only where measured depth agrees with mesh.
"""

import copy
import json
import zipfile

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from shared.calibration import prepare_rgbd, project_rgb
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
    columns = int(np.ceil(np.sqrt(len(charts))))
    rows = int(np.ceil(len(charts) / columns))
    cell = np.array([1 / columns, 1 / rows])
    padding = np.minimum(1 / size, cell * 0.1)
    uv = np.empty((len(faces), 3, 2), dtype=np.float32)
    for chart, (ids, local_uv) in enumerate(charts):
        origin = np.array([chart % columns, chart // columns]) * cell
        uv[ids] = origin + padding + local_uv * (cell - 2 * padding)
    tensor.triangle["texture_uvs"] = o3d.core.Tensor(uv)
    return tensor, {"method": "repaired_chart_tiles", "charts": len(charts), "subdivisions": retries}


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


def make_textured_mesh(
    engine,
    size=1024,
    max_triangles=50000,
    max_views=24,
    use_images=True,
    exposure_correction=False,
    blend_mode="blend",
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
    topology_report = _make_uv_manifold(mesh)
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
    totals = np.zeros(len(positions), np.float32)
    sums = np.zeros((len(positions), 3), np.float32)
    views = _views(engine, max_views) if use_images else []
    c = engine.settings.camera
    valid_indices = np.flatnonzero(valid)
    prepared = []
    for index, pose in views:
        source_rgb, source_depth = engine.raw_frames[index]
        rgb, depth = prepare_rgbd(source_rgb, source_depth, engine.settings)
        if engine.settings.sensor_calibration:
            rgb = source_rgb  # Native lens pixels, sampled with K,D and R,T.
        prepared.append((pose, rgb, depth.astype(np.float32) / 1000))
    exposure_report = {"applied": False, "reason": "Not requested"}
    gains = np.ones((len(views), 3), np.float32)
    if exposure_correction and len(views) > 1 and len(valid_indices):
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
    best_weight = np.zeros(len(positions), np.float32)
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
            if blend_mode == "best":
                better = weight > best_weight[ids]
                chosen = ids[better]
                albedo[chosen] = values[better]
                best_weight[chosen] = weight[better]
            else:
                sums[ids] += values * weight[:, None]
            totals[ids] += weight
    projected = totals > 0
    if blend_mode == "blend":
        albedo[projected] = sums[projected] / totals[projected, None]
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
        "exposure_correction": exposure_report,
        "atlas_partitions": partitions,
        "atlas": atlas_report,
        "original_triangles": original_triangles,
        "export_triangles": len(faces),
        "topology_repair": topology_report,
        "views": [i for i, _ in views],
        "projected_fraction": float(projected.sum() / max(1, valid.sum())),
        "unobserved_fallback": "fused vertex colors",
        "depth_unit": "metres",
    }
    return result, report


def export_texture(engine, filepath, fmt="glb", **options):
    if fmt not in ("glb", "obj.zip"):
        raise ValueError("Texture format must be glb or obj.zip")
    mesh, report = make_textured_mesh(engine, **options)
    if fmt == "glb":
        mesh.export(filepath, file_type="glb")
    else:
        obj, assets = trimesh.exchange.obj.export_obj(
            mesh,
            include_normals=True,
            return_texture=True,
            write_texture=False,
            mtl_name="scan.mtl",
        )
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
