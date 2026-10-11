"""Bounded texture pages sized by surface area; compatible OBJ/GLB materials.

Pages are baked sequentially and reuse prepared observations. Source RGB detail,
visibility and the explicit page cap bound attainable resolution.
"""

import copy
import math
from itertools import product
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import trimesh

from .texturing import (
    _face_neighbors, _make_uv_manifold, _select_face_sources, _prepare_texture_views, _project, _views,
    make_textured_mesh,
)


def _surface(mesh):
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return triangles.mean(axis=1), np.linalg.norm(cross, axis=1) / 2


def _page_groups(mesh, area_budget, max_pages):
    """Bisect spatially along area medians; preserve every face exactly once."""
    centers, areas = _surface(mesh)
    groups = [np.arange(len(areas))]
    while len(groups) < max_pages:
        weights = [areas[ids].sum() if len(ids) > 1 else 0 for ids in groups]
        page = int(np.argmax(weights))
        if weights[page] <= area_budget:
            break
        ids = groups.pop(page)
        axis = np.ptp(centers[ids], axis=0).argmax()
        ids = ids[np.argsort(centers[ids, axis], kind="stable")]
        cumulative = np.cumsum(areas[ids])
        cut = int(np.argmin(np.abs(cumulative[:-1] - cumulative[-1] / 2))) + 1
        groups[page:page] = [ids[:cut], ids[cut:]]
    return groups, areas


def _submesh(mesh, ids):
    faces = np.asarray(mesh.triangles)[ids]
    used, inverse = np.unique(faces, return_inverse=True)
    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[used]),
        o3d.utility.Vector3iVector(inverse.reshape(-1, 3)),
    )
    if mesh.has_vertex_colors():
        result.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[used])
    return result


def _page_view_indices(engine, mesh, views):
    """Conservatively reject boxes wholly outside a depth-camera frustum.

    Depth visibility requires this pinhole frustum even for distorted native
    RGB. Linear half-space tests on all eight box corners also handle pages
    crossing the camera plane; sampled visibility would miss small surfaces.
    """
    vertices = np.asarray(mesh.vertices)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    corners = np.array(list(product(*zip(lower, upper))))
    c = engine.settings.camera
    selected = []
    for index, (_, pose) in enumerate(views):
        inverse = np.linalg.inv(pose)
        x, y, z = (corners @ inverse[:3, :3].T + inverse[:3, 3]).T
        inside = (
            z - engine.settings.near_m, engine.settings.far_m - z,
            x * c.fx + z * c.cx, z * (c.width - 1 - c.cx) - x * c.fx,
            y * c.fy + z * c.cy, z * (c.height - 1 - c.cy) - y * c.fy,
        )
        if all(np.any(plane >= -1e-9) for plane in inside):
            selected.append(index)
    return selected


def _shared_exposure(engine, mesh, prepared, requested):
    """One held-out gain solution for all pages, avoiding page brightness shifts."""
    gains = np.ones((len(prepared), 3), np.float32)
    report = {"applied": False, "requested": requested, "reason": "Not requested"}
    if not requested or len(prepared) < 2:
        if requested:
            report["reason"] = "Fewer than two source photos"
        return gains, report
    from .photometric import estimate_gains

    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    ids = np.arange(0, len(vertices), max(1, math.ceil(len(vertices) / 8192)))
    samples, visible = [], []
    for pose, rgb, depth in prepared:
        values, weight = _project(
            vertices[ids], np.asarray(mesh.vertex_normals)[ids], rgb, depth, pose,
            engine.settings.camera, engine.settings.near_m, engine.settings.far_m,
            engine.settings.sensor_calibration, engine.settings.rgb_camera,
        )
        samples.append(values)
        visible.append(weight > 0)
    gains, report = estimate_gains(np.asarray(samples), np.asarray(visible))
    report["requested"] = True
    return gains, report


def _whole_mesh_sources(engine, mesh, prepared):
    """Label the physical surface before UV/page cuts, using four face samples."""
    # Signed adjacency needs consistent winding. Repair only the scoring copy;
    # face IDs and positions survive, while ambiguous sheets remain separated.
    mesh = copy.deepcopy(mesh)
    topology = _make_uv_manifold(mesh)
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    barycentric = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
    scores = np.zeros((len(prepared), len(triangles)), np.float32)
    for start in range(0, len(triangles), 16384):
        stop = min(start + 16384, len(triangles))
        world = np.einsum('sk,fkj->fsj', barycentric, triangles[start:stop]).reshape(-1, 3)
        normal = np.repeat(normals[start:stop], 4, axis=0)
        for view, (pose, rgb, depth) in enumerate(prepared):
            _, weight = _project(
                world, normal, rgb, depth, pose, engine.settings.camera,
                engine.settings.near_m, engine.settings.far_m,
                engine.settings.sensor_calibration, engine.settings.rgb_camera,
            )
            scores[view, start:stop] = weight.reshape(-1, 4).mean(axis=1)
    labels, report = _select_face_sources(scores, _face_neighbors(mesh))
    report.update(scope="whole_export", samples_per_face=4,
                  source_faces=[int(np.sum(labels == i)) for i in range(len(prepared))],
                  topology_repair=topology)
    return labels, report


def make_textured_asset(
    engine, size=1024, max_triangles=200000, max_views=64, use_images=True,
    exposure_correction=False, blend_mode="blend", max_pages=16, texels_per_m=512,
):
    """Return a mesh or multi-material scene and its aggregate texture report.

    `size` is the edge of each page, `max_pages=1` requests a single atlas.
    The 55% packing allowance is a planning estimate, not a density guarantee;
    report occupancy and cap pressure rather than silently claiming full detail.
    """
    if size not in (256, 512, 1024, 2048):
        raise ValueError("Texture size must be 256, 512, 1024, or 2048")
    if not 100 <= max_triangles <= 200000 or not 1 <= max_views <= 64:
        raise ValueError("Invalid texture triangle or view budget")
    if not 1 <= max_pages <= 32 or not math.isfinite(texels_per_m) or not 1 <= texels_per_m <= 1024:
        raise ValueError("Texture page budget must be 1..32 and texels_per_m 1..1024")
    if blend_mode not in ("blend", "best"):
        raise ValueError("Texture blend_mode must be blend or best")
    if engine.mesh is None or not len(engine.mesh.triangles):
        raise ValueError("Build a mesh before texturing")
    original_triangles = len(engine.mesh.triangles)
    mesh = copy.deepcopy(engine.mesh)
    if original_triangles > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise ValueError("No non-degenerate faces remain for texture export")
    area_budget = size**2 * 0.55 / texels_per_m**2
    groups, areas = _page_groups(mesh, area_budget, max_pages)
    views = _views(engine, max_views) if use_images else []
    prepared = _prepare_texture_views(engine, views)
    exposure = _shared_exposure(engine, mesh, prepared, exposure_correction)
    exposure[1].update(scope="whole_export", views=[i for i, _ in views])
    labels, selection = (None, {"method": "no_source_images"})
    if prepared:
        labels, selection = _whole_mesh_sources(engine, mesh, prepared)
    pages, reports = [], []
    for page, ids in enumerate(groups):
        local = SimpleNamespace(mesh=_submesh(mesh, ids), settings=engine.settings)
        active = _page_view_indices(engine, local.mesh, views)
        local_labels = None
        if labels is not None:
            mapping = np.full(len(views), -1, np.int32)
            mapping[active] = np.arange(len(active))
            local_labels = np.full(len(ids), -1, np.int32)
            observed = labels[ids] >= 0
            local_labels[observed] = mapping[labels[ids][observed]]
        textured, report = make_textured_mesh(
            local, size=size, max_triangles=max_triangles, max_views=max_views,
            use_images=use_images, exposure_correction=exposure_correction, blend_mode=blend_mode,
            _prepared=[prepared[i] for i in active],
            _selected_views=[views[i] for i in active],
            _exposure=(exposure[0][active], exposure[1]),
            _face_sources=local_labels,
        )
        name = "scan_material" if len(groups) == 1 else f"scan_material_{page:03d}"
        textured.visual.material.name = name
        textured.metadata["name"] = f"scan_page_{page:03d}"
        report["surface_area_m2"] = float(areas[ids].sum())
        report["culled_views"] = len(views) - len(active)
        reports.append(report)
        pages.append(textured)
    if len(pages) == 1:
        result, report = pages[0], reports[0].copy()
    else:
        result = trimesh.Scene()
        for page in pages:
            result.add_geometry(page, geom_name=page.metadata["name"])
        mapped = sum(r["mapped_texels"] for r in reports)
        projected = sum(r["projected_texels"] for r in reports)
        report = {
            "method": "depth_tested_rgb" if views else "vertex_color_bake",
            "texture_size": size, "blend_mode": blend_mode,
            "views": [i for i, _ in views], "exposure_correction": exposure[1],
            "mapped_texels": mapped, "projected_texels": projected,
            "projected_fraction": projected / max(1, mapped),
            "atlas_occupancy": mapped / (len(pages) * size**2),
            "atlas": {"method": "surface_area_pages"},
            "seam_blending": {
                "width_texels": max(r["seam_blending"]["width_texels"] for r in reports),
                "softened_texels": sum(r["seam_blending"]["softened_texels"] for r in reports),
                "projected_texels": projected,
            },
            "unobserved_fallback": "fused vertex colors", "depth_unit": "metres",
        }
    report.update(
        original_triangles=original_triangles,
        export_triangles=sum(len(p.faces) for p in pages),
        texture_pages=len(pages), max_pages=max_pages,
        target_texels_per_m=texels_per_m, surface_area_m2=float(areas.sum()),
        total_texture_pixels=len(pages) * size**2,
        page_budget_limited=bool(max(r["surface_area_m2"] for r in reports) > area_budget),
        pages=reports,
        source_selection=selection,
        views=[i for i, _ in views],
    )
    result.metadata["texture_report"] = report
    return result, report
