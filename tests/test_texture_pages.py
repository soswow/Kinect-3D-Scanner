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

from scanner_server.texture_pages import _page_groups, _page_view_indices, make_textured_asset
from scanner_server.texturing import (
    _chart_rectangles, _prepare_texture_views, _single_object_gltf, export_texture,
)
from shared.calibration import prepare_rgbd
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings
from tests.test_features import texture_scene


class TexturePageTests(unittest.TestCase):
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
