"""Adaptive page coverage, material round trips and source preparation reuse."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from scanner_server.texture_pages import _page_groups, _page_view_indices, make_textured_asset
from scanner_server.texturing import _chart_rectangles, _prepare_texture_views, export_texture
from shared.calibration import prepare_rgbd
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings
from tests.test_features import texture_scene


class TexturePageTests(unittest.TestCase):
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
            glb = trimesh.load(root / "scan.glb", force="scene", process=False)
            self.assertEqual(report["texture_pages"], len(glb.geometry))
            for geometry in glb.geometry.values():
                self.assertIsNotNone(geometry.visual.material.baseColorTexture)
            export_texture(engine, root / "scan.zip", "obj.zip", size=256, max_pages=4)
            with zipfile.ZipFile(root / "scan.zip") as archive:
                self.assertEqual(report["texture_pages"], len([n for n in archive.namelist() if n.endswith('.png')]))
                archive.extractall(root / "obj")
            obj = trimesh.load(root / "obj/scan.obj", force="scene", process=False)
            self.assertEqual(report["texture_pages"], len(obj.geometry))
            self.assertEqual(len(engine.mesh.triangles), sum(len(g.faces) for g in obj.geometry.values()))

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
