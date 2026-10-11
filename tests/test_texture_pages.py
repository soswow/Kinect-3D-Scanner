"""Adaptive page coverage, material round trips and source preparation reuse."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import tempfile
import json
import struct
import shutil
import subprocess
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from scanner_server.texture_pages import _page_groups, _page_view_indices, _whole_mesh_sources, make_textured_asset
from scanner_server.texturing import (
    _chart_rectangles, _prepare_texture_views, _single_object_gltf, export_texture,
    make_textured_mesh, _face_neighbors, _interpolate_face_corrections,
)
from shared.calibration import prepare_rgbd
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings
from scanner_server.surface_color import PhotoColorField
from tests.test_features import texture_scene


class TexturePageTests(unittest.TestCase):
    def test_corner_corrections_preserve_spatial_field_and_padding_bounds(self):
        triangles = np.array([[[0., 0., 1.], [2., 0., 1.], [0., 2., 1.]]])
        corrections = np.array([[[0., 0., 0.], [.08, .02, 0.], [.02, .08, 0.]]])
        positions = np.array([[.5, .5, 1.], [2., 0., 1.], [2.2, -.1, 1.]])
        result = _interpolate_face_corrections(positions, np.zeros(3, int), triangles, corrections)
        np.testing.assert_allclose(result[:2], [[.025, .025, 0.], [.08, .02, 0.]])
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= .08))
        # Corner order can change during winding repair without changing the field.
        reversed_result = _interpolate_face_corrections(
            positions, np.zeros(3, int), triangles[:, ::-1], corrections[:, ::-1],
        )
        np.testing.assert_allclose(result, reversed_result)

    def test_leveling_corrects_actual_visible_photo_including_depth_fallback(self):
        engine = texture_scene()
        depth = engine.raw_frames[0][1]
        red = np.full((480, 640, 3), (180, 70, 50), np.uint8)
        blue = np.full((480, 640, 3), (50, 70, 180), np.uint8)
        engine.poses = [(0, np.eye(4)), (1, np.eye(4))]
        labels = np.ones(len(engine.mesh.triangles), np.int32)
        field = PhotoColorField(np.asarray(engine.mesh.vertices), np.asarray(engine.mesh.triangles),
            np.broadcast_to(np.log([[[.8], [1.1]]]), (len(engine.mesh.vertices), 2, 3)),
            np.arange(len(labels)), np.arange(2))
        for selected_depth, expected, fallback in (
            (depth, (55, 77, 198), False), (np.zeros_like(depth), (144, 56, 40), True),
        ):
            engine.raw_frames = [(red, depth), (blue, selected_depth)]
            mesh, report = make_textured_mesh(
                engine, size=256, _prepared=_prepare_texture_views(engine, engine.poses),
                _selected_views=engine.poses, _face_sources=labels,
                _color_field=field, _seam_report={"applied": True},
            )
            front = np.isclose(mesh.vertices[mesh.faces].mean(axis=1)[:, 2], 1.2)
            colors = trimesh.visual.color.uv_to_color(
                mesh.visual.uv[mesh.faces].mean(axis=1), mesh.visual.material.image,
            )
            np.testing.assert_allclose(colors[front, :3], np.tile(expected, (front.sum(), 1)), atol=2)
            self.assertGreater(report["seam_leveling"]["corrected_texels"], 0)
            self.assertEqual(fallback, report["seam_leveling"]["corrected_fallback_texels"] > 0)
            self.assertEqual(0, report["seam_blending"]["softened_texels"])

    def test_global_leveling_is_shared_across_pages_and_culled_views(self):
        engine = texture_scene()
        rgb = np.full((480, 640, 3), 100, np.uint8)
        depth = engine.raw_frames[0][1]
        disjoint = np.eye(4)
        disjoint[0, 3] = 20
        engine.poses = [(0, disjoint), (1, np.eye(4)), (2, np.eye(4))]
        engine.raw_frames = [(rgb, depth)] * 3
        engine.frame_metadata = [{}] * 3
        triangles = np.asarray(engine.mesh.vertices)[np.asarray(engine.mesh.triangles)]
        vertices = np.asarray(engine.mesh.vertices)
        logs = np.zeros((len(vertices), 3, 3), np.float32)
        logs[:, :, 0] = .03 + vertices[:, None, 0] * .02
        field = PhotoColorField(vertices, np.asarray(engine.mesh.triangles), logs,
                               np.arange(len(triangles)), np.arange(3))
        for budget in (1, 4):
            with patch("scanner_server.surface_color.estimate_color_field",
                       return_value=(field, {"applied": True})) as solve:
                asset, report = make_textured_asset(engine, size=256, max_pages=budget)
            self.assertEqual(1, solve.call_count)
            self.assertEqual(3, len(solve.call_args.args[1]))
            self.assertEqual("whole_export", report["seam_leveling"]["scope"])
            pages = list(asset.geometry.values()) if isinstance(asset, trimesh.Scene) else [asset]
            for page in pages:
                centers = page.vertices[page.faces].mean(axis=1)
                front = np.isclose(centers[:, 2], 1.2)
                colors = trimesh.visual.color.uv_to_color(
                    page.visual.uv[page.faces].mean(axis=1), page.visual.material.image,
                )
                expected = 100 * np.exp(.03 + centers[front, 0] * .02)
                np.testing.assert_allclose(colors[front, 0], expected, atol=2)
                np.testing.assert_allclose(colors[front, 1:3], 100, atol=2)
        with patch("scanner_server.surface_color.estimate_color_field") as solve:
            make_textured_asset(engine, size=256, max_pages=1, blend_mode="best")
        solve.assert_not_called()

    @unittest.skipUnless(os.environ.get("BLENDER_EXECUTABLE") or shutil.which("blender"),
                         "Set BLENDER_EXECUTABLE to verify Blender's default imports")
    def test_blender_imports_one_object_with_all_page_materials(self):
        blender = os.environ.get("BLENDER_EXECUTABLE") or shutil.which("blender")
        engine = texture_scene()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = export_texture(engine, root / "scan.glb", size=256, max_pages=4)
            export_texture(engine, root / "scan.zip", "obj.zip", size=256, max_pages=4)
            with zipfile.ZipFile(root / "scan.zip") as archive:
                archive.extractall(root / "obj")
            check = root / "check.py"
            check.write_text(
                "import bpy\n"
                f"paths = {repr([str(root / 'scan.glb'), str(root / 'obj/scan.obj')])}\n"
                "for path in paths:\n"
                "    bpy.ops.object.select_all(action='SELECT')\n"
                "    bpy.ops.object.delete(use_global=False)\n"
                "    if path.endswith('.glb'): bpy.ops.import_scene.gltf(filepath=path)\n"
                "    else: bpy.ops.wm.obj_import(filepath=path)\n"
                "    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']\n"
                "    assert len(meshes) == 1, meshes\n"
                f"    assert len(meshes[0].data.polygons) == {len(engine.mesh.triangles)}\n"
                f"    assert len(meshes[0].data.materials) == {report['texture_pages']}\n"
                f"    assert {{p.material_index for p in meshes[0].data.polygons}} == set(range({report['texture_pages']}))\n"
                "    for material in meshes[0].data.materials:\n"
                "        assert any(n.type == 'TEX_IMAGE' and n.image for n in material.node_tree.nodes)\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [blender, "--background", "--factory-startup", "--threads", "2",
                 "--python-exit-code", "1", "--python", str(check)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                timeout=90,
            )
            self.assertEqual(0, result.returncode, result.stdout)

    def test_frustum_culling_keeps_crossing_boxes_and_rejects_disjoint_views(self):
        engine = texture_scene()
        self.assertEqual([0], _page_view_indices(engine, engine.mesh, engine.poses))
        engine.mesh.translate((20, 0, 0))
        self.assertEqual([], _page_view_indices(engine, engine.mesh, engine.poses))
        engine.mesh.translate((-20, 0, -1.35))
        engine.mesh.scale(10, center=(0, 0, 0))
        # Most corners are outside, but a surface crosses the near plane.
        self.assertEqual([0], _page_view_indices(engine, engine.mesh, engine.poses))

    def test_whole_surface_sources_are_independent_of_page_budget_and_culling(self):
        engine = texture_scene()
        depth = engine.raw_frames[0][1]
        red = np.full((480, 640, 3), (220, 30, 10), np.uint8)
        blue = np.full((480, 640, 3), (10, 30, 220), np.uint8)
        disjoint = np.eye(4)
        disjoint[0, 3] = 20
        engine.poses = [(0, disjoint), (1, np.eye(4)), (2, np.eye(4))]
        engine.raw_frames = [(blue, depth), (red, depth), (blue, depth)]
        engine.frame_metadata = [{}, {}, {}]
        reports = []
        for budget in (1, 4):
            with patch("scanner_server.texture_pages.make_textured_mesh", wraps=make_textured_mesh) as baker:
                asset, report = make_textured_asset(engine, size=256, max_pages=budget, blend_mode="best")
            reports.append(report)
            self.assertEqual(budget, report["texture_pages"])
            self.assertEqual([0, 1, 2], report["views"])
            self.assertEqual(0, report["source_selection"]["source_faces"][0])
            self.assertGreater(report["source_selection"]["source_faces"][1], 0)
            for call in baker.call_args_list:
                # Global source 1 must become local 0 after the disjoint source 0 is culled.
                self.assertEqual([1, 2], [i for i, _ in call.kwargs["_selected_views"]])
                self.assertTrue(set(call.kwargs["_face_sources"]).issubset({-1, 0}))
            pages = list(asset.geometry.values()) if isinstance(asset, trimesh.Scene) else [asset]
            for page in pages:
                centers = page.vertices[page.faces].mean(axis=1)
                front = np.isclose(centers[:, 2], 1.2)
                if front.any():
                    uv = page.visual.uv[page.faces].mean(axis=1)
                    colors = trimesh.visual.color.uv_to_color(uv, page.visual.material.image)
                    np.testing.assert_allclose(colors[front, :3], np.tile((220, 30, 10), (front.sum(), 1)), atol=2)
        self.assertEqual(reports[0]["source_selection"], reports[1]["source_selection"])
        self.assertEqual(reports[0]["views"], reports[1]["views"])

    def test_whole_source_scoring_repairs_mixed_winding_without_mutating_mesh(self):
        import open3d as o3d

        engine = texture_scene()
        engine.mesh = o3d.geometry.TriangleMesh()
        engine.mesh.vertices = o3d.utility.Vector3dVector([
            [-.1, -.1, 1.2], [.1, -.1, 1.2], [.1, .1, 1.2], [-.1, .1, 1.2],
        ])
        consistent = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
        reversed_faces = consistent.copy()
        reversed_faces[1] = reversed_faces[1, ::-1]
        prepared = _prepare_texture_views(engine, engine.poses) * 2
        outcomes = []
        for faces in (consistent, reversed_faces):
            engine.mesh.triangles = o3d.utility.Vector3iVector(faces)
            if np.array_equal(faces, reversed_faces):
                self.assertTrue(np.all(_face_neighbors(engine.mesh) == -1))
            # Nearly tied observations prefer opposite photos per face. Surface
            # continuity should join these coplanar faces despite input winding.
            weights = [np.repeat([1., .9], 4), np.repeat([.95, 1.], 4)]
            with patch("scanner_server.texture_pages._project", side_effect=[
                (np.zeros((8, 3), np.float32), weight) for weight in weights
            ]):
                labels, report = _whole_mesh_sources(engine, engine.mesh, prepared)
            np.testing.assert_array_equal(faces, np.asarray(engine.mesh.triangles))
            self.assertEqual(labels[0], labels[1])
            self.assertEqual(0, report["topology_repair"]["removed_triangles"])
            self.assertEqual("whole_export", report["scope"])
            outcomes.append((labels, report["source_faces"]))
        np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
        self.assertEqual(outcomes[0][1], outcomes[1][1])

    def test_precomputed_face_sources_override_scores_with_depth_tested_fallback(self):
        engine = texture_scene()
        depth = engine.raw_frames[0][1]
        red = np.full((480, 640, 3), (220, 30, 10), np.uint8)
        blue = np.full((480, 640, 3), (10, 30, 220), np.uint8)
        engine.poses = [(0, np.eye(4)), (1, np.eye(4))]
        engine.frame_metadata = [{}, {}]
        labels = np.ones(len(engine.mesh.triangles), np.int32)
        cases = ((depth, depth, (10, 30, 220)),
                 (depth, np.zeros_like(depth), (220, 30, 10)),
                 (np.zeros_like(depth), np.zeros_like(depth), (204, 51, 25)))
        for first_depth, second_depth, expected in cases:
            engine.raw_frames = [(red, first_depth), (blue, second_depth)]
            prepared = _prepare_texture_views(engine, engine.poses)
            mesh, report = make_textured_mesh(
                engine, size=256, blend_mode="best", _prepared=prepared,
                _selected_views=engine.poses, _face_sources=labels,
            )
            centers = mesh.vertices[mesh.faces].mean(axis=1)
            front = np.isclose(centers[:, 2], 1.2)
            self.assertTrue(front.any())
            uv = mesh.visual.uv[mesh.faces].mean(axis=1)
            colors = trimesh.visual.color.uv_to_color(uv, mesh.visual.material.image)
            np.testing.assert_allclose(colors[front, :3], np.tile(expected, (front.sum(), 1)), atol=2)
            self.assertEqual("whole_export_face_regions", report["source_selection"]["method"])
            if first_depth.any():
                self.assertGreater(report["projected_texels"], 0)
            else:
                self.assertEqual(0, report["projected_texels"])

    def test_native_preparation_preserves_depth_without_registering_unused_rgb(self):
        engine = texture_scene()
        engine.settings = ScanSettings(sensor_calibration=load_calibration())
        rgb = np.full((1024, 1280, 3), 100, np.uint8)
        raw = np.full((480, 640), 800, np.uint16)
        engine.raw_frames = [(rgb, raw)]
        _, expected = prepare_rgbd(rgb, raw, engine.settings)
        with patch("scanner_server.texturing.prepare_rgbd", side_effect=AssertionError("unused registration")):
            prepared = _prepare_texture_views(engine, engine.poses)
        self.assertIs(rgb, prepared[0][1])
        np.testing.assert_array_equal(prepared[0][2], expected.astype(np.float32) / 1000)

    def test_area_partition_preserves_all_faces_and_expands_with_scene(self):
        engine = texture_scene()
        small, _ = _page_groups(engine.mesh, 1, 16)
        self.assertEqual(1, len(small))
        engine.mesh.scale(8, center=(0, 0, 0))
        groups, areas = _page_groups(engine.mesh, 1, 16)
        self.assertGreater(len(groups), len(small))
        np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(len(areas)))
        capped, _ = _page_groups(engine.mesh, 0.001, 2)
        self.assertEqual(2, len(capped))

    def test_fallback_packing_allocates_by_area_without_overlaps(self):
        areas = np.array([10., 2., 1., 0.1])
        rectangles = _chart_rectangles(areas)
        for i, (origin, extent) in enumerate(rectangles):
            self.assertTrue(np.all(origin >= 0))
            self.assertTrue(np.all(origin + extent <= 1 + 1e-12))
            self.assertAlmostEqual(np.prod(extent), areas[i] / areas.sum())
            for other, width in rectangles[i + 1:]:
                self.assertTrue(np.any((origin + extent <= other + 1e-12) |
                                       (other + width <= origin + 1e-12)))

    def test_multimaterial_exports_preserve_geometry_colors_and_uv_orientation(self):
        engine = texture_scene()
        before = np.asarray(engine.mesh.vertices).copy()
        with patch("scanner_server.texture_pages._prepare_texture_views", wraps=_prepare_texture_views) as prepare:
            asset, report = make_textured_asset(engine, size=256, max_pages=4)
        self.assertGreater(report["texture_pages"], 1)
        self.assertEqual(1, prepare.call_count)
        self.assertEqual(len(engine.mesh.triangles), sum(len(g.faces) for g in asset.geometry.values()))
        np.testing.assert_array_equal(before, np.asarray(engine.mesh.vertices))
        for page in asset.geometry.values():
            centers = page.vertices[page.faces].mean(axis=1)
            front = np.isclose(centers[:, 2], 1.2)
            uv = page.visual.uv[page.faces].mean(axis=1)
            colors = trimesh.visual.color.uv_to_color(uv, page.visual.material.image)
            expected = np.column_stack(((centers[front, 0] * 525 / 1.2 + 319.5) / 640 * 255,
                                        (centers[front, 1] * 525 / 1.2 + 239.5) / 480 * 255,
                                        np.zeros(front.sum())))
            np.testing.assert_allclose(colors[front, :3], expected, atol=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_texture(engine, root / "scan.glb", size=256, max_pages=4)
            data = (root / "scan.glb").read_bytes()
            json_size = struct.unpack_from("<I", data, 12)[0]
            tree = json.loads(data[20:20 + json_size])
            self.assertEqual([{"name": "scan", "mesh": 0}], tree["nodes"])
            self.assertEqual(1, len(tree["meshes"]))
            self.assertEqual(1, tree["meshes"][0]["extras"]["texture_report"]["export_objects"])
            self.assertEqual(report["texture_pages"], len(tree["meshes"][0]["primitives"]))
            self.assertEqual(set(range(report["texture_pages"])),
                             {p["material"] for p in tree["meshes"][0]["primitives"]})
            glb = trimesh.load(root / "scan.glb", force="scene", process=False)
            self.assertEqual(report["texture_pages"], len(glb.geometry))
            for geometry in glb.geometry.values():
                self.assertIsNotNone(geometry.visual.material.baseColorTexture)
            self.assertEqual(len(engine.mesh.triangles), sum(len(g.faces) for g in glb.geometry.values()))
            export_texture(engine, root / "scan.zip", "obj.zip", size=256, max_pages=4)
            with zipfile.ZipFile(root / "scan.zip") as archive:
                obj_text = archive.read("scan.obj").decode()
                self.assertEqual(["o scan"], [l for l in obj_text.splitlines() if l.startswith("o ")])
                self.assertFalse(any(l.startswith("g ") for l in obj_text.splitlines()))
                self.assertEqual(report["texture_pages"], obj_text.count("usemtl "))
                self.assertEqual(report["texture_pages"], len([n for n in archive.namelist() if n.endswith('.png')]))
                archive.extractall(root / "obj")
            obj = trimesh.load(root / "obj/scan.obj", force="scene", process=False)
            self.assertEqual(report["texture_pages"], len(obj.geometry))
            self.assertEqual(len(engine.mesh.triangles), sum(len(g.faces) for g in obj.geometry.values()))

    def test_single_object_export_refuses_transformed_page_nodes(self):
        for transform in ({"translation": [1, 0, 0]}, {"scale": [1.000001, 1, 1]},
                          {"rotation": [0, 0, 1, 0]}):
            tree = {"nodes": [{"mesh": 0, **transform}],
                    "meshes": [{"primitives": [{"material": 0}]}]}
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                _single_object_gltf(tree)

    def test_page_cap_is_reported_and_single_atlas_remains_available(self):
        engine = texture_scene()
        asset, report = make_textured_asset(engine, size=256, max_pages=1)
        self.assertIsInstance(asset, trimesh.Trimesh)
        self.assertEqual(1, report["texture_pages"])
        self.assertTrue(report["page_budget_limited"])
        for options in ({"max_pages": 0}, {"max_pages": 33}, {"texels_per_m": float('nan')},
                        {"size": 4096}, {"max_views": 65}):
            with self.assertRaises(ValueError):
                make_textured_asset(engine, **options)


if __name__ == "__main__":
    unittest.main()
