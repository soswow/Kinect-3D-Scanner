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

from shared.depth import prepare_depth


def _views(engine, max_views):
    # Cover the whole trajectory; choose the sharpest synchronized image in
    # each interval rather than loading every image into a projection backend.
    candidates = [
        p
        for p in engine.poses
        if abs(engine.frame_metadata[p[0]].get("rgb_depth_delta_ms") or 0) <= 20
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


def make_textured_mesh(
    engine, size=1024, max_triangles=50000, max_views=24, use_images=True
):
    if engine.mesh is None or not len(engine.mesh.triangles):
        raise ValueError("Build a mesh before texturing")
    if size not in (256, 512, 1024, 2048):
        raise ValueError("Texture size must be 256, 512, 1024, or 2048")
    if not 100 <= max_triangles <= 200000 or not 1 <= max_views <= 64:
        raise ValueError("Invalid texture triangle or view budget")
    mesh = copy.deepcopy(engine.mesh)
    original_triangles = len(mesh.triangles)
    if original_triangles > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if (
        not mesh.is_edge_manifold(allow_boundary_edges=True)
        or not mesh.is_vertex_manifold()
    ):
        raise ValueError(
            "Texture UV generation needs a manifold mesh; inspect or clean the final surface"
        )
    mesh.compute_vertex_normals()
    tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # Iso-chart spectral solves can take minutes on one large connected chart.
    # Partition before atlas generation to bound each solve on scan meshes.
    partitions = max(1, min(256, int(np.ceil(len(mesh.triangles) / 1000))))
    tensor.compute_uvatlas(size=size, parallel_partitions=partitions, nthreads=4)
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
    for index, pose in views:
        rgb, raw_depth = engine.raw_frames[index]
        depth = prepare_depth(raw_depth, engine.settings).astype(np.float32) / 1000
        extrinsic = np.linalg.inv(pose)
        for start in range(0, len(valid_indices), 65536):
            ids = valid_indices[start : start + 65536]
            world = positions[ids]
            camera = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
            z = camera[:, 2]
            safe_z = np.maximum(z, 1e-6)
            u = camera[:, 0] * c.fx / safe_z + c.cx
            v = camera[:, 1] * c.fy / safe_z + c.cy
            inside = (
                (z > engine.settings.near_m)
                & (z < engine.settings.far_m)
                & (u >= 0)
                & (v >= 0)
                & (u < c.width - 1)
                & (v < c.height - 1)
            )
            x = np.clip(np.rint(u), 0, c.width - 1).astype(int)
            y = np.clip(np.rint(v), 0, c.height - 1).astype(int)
            observed = depth[y, x]
            tolerance = np.maximum(0.015, 0.01 * z)
            direction = pose[:3, 3] - world
            direction /= np.maximum(
                np.linalg.norm(direction, axis=1, keepdims=True), 1e-6
            )
            cosine = np.abs(np.sum(normals[ids] * direction, axis=1))
            mask = (
                inside
                & (observed > 0)
                & (np.abs(observed - z) <= tolerance)
                & (cosine > 0.25)
            )
            chosen = ids[mask]
            if not len(chosen):
                continue
            uf, vf = u[mask], v[mask]
            x0 = np.floor(uf).astype(int)
            y0 = np.floor(vf).astype(int)
            dx, dy = (uf - x0)[:, None], (vf - y0)[:, None]
            # Bilinear RGB sampling; depth remains nearest to avoid filling holes.
            values = (
                rgb[y0, x0] * (1 - dx) * (1 - dy)
                + rgb[y0, x0 + 1] * dx * (1 - dy)
                + rgb[y0 + 1, x0] * (1 - dx) * dy
                + rgb[y0 + 1, x0 + 1] * dx * dy
            ) / 255
            weight = (cosine[mask] ** 2 / np.maximum(z[mask] ** 2, 0.1)).astype(
                np.float32
            )
            sums[chosen] += values * weight[:, None]
            totals[chosen] += weight
    projected = totals > 0
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
        "atlas_partitions": partitions,
        "original_triangles": original_triangles,
        "export_triangles": len(faces),
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
