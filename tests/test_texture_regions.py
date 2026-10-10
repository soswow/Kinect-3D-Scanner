"""Texture detail, coherent source regions and bounded seam support."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import unittest

import numpy as np
import trimesh

from scanner_server.texturing import _select_face_sources, _seam_partners, make_textured_mesh
from tests.test_features import texture_scene


class TextureRegionTests(unittest.TestCase):
    def test_similar_cameras_do_not_create_alternating_tiny_regions(self):
        count = 60
        neighbors = np.full((count, 3), -1, np.int32)
        neighbors[:-1, 0] = np.arange(1, count)
        neighbors[1:, 1] = np.arange(count - 1)
        scores = np.ones((2, count), np.float32)
        scores[0, ::2] += 0.05
        scores[1, 1::2] += 0.05
        labels, _ = _select_face_sources(scores, neighbors)
        self.assertLessEqual(np.count_nonzero(np.diff(labels)), 1)
        # Coherence must never force an invisible photo onto a surface.
        scores[0, count // 2:] = 0
        scores[1, :count // 2] = 0
        labels, _ = _select_face_sources(scores, neighbors)
        np.testing.assert_array_equal(labels[:count // 2], 0)
        np.testing.assert_array_equal(labels[count // 2:], 1)
        scores[:, 10] = 0
        labels, _ = _select_face_sources(scores, neighbors)
        self.assertEqual(-1, labels[10])

    def test_seam_band_does_not_touch_region_interiors_or_disjoint_uv_charts(self):
        size = 32
        y, x = np.indices((size, size))
        positions = np.stack((x * 0.005, y * 0.005, np.ones_like(x)), axis=-1).astype(np.float32)
        sources = (x >= size // 2).astype(np.int32)
        valid = np.ones(size * size, bool)
        partners, amount = _seam_partners(positions.reshape(-1, 3), sources.reshape(-1), valid, size)
        band = amount.reshape(size, size) > 0
        self.assertTrue(band.any())
        self.assertFalse(band[:, :size // 2 - 3].any())
        self.assertFalse(band[:, size // 2 + 3:].any())
        self.assertLessEqual(amount.max(), 0.5)
        self.assertTrue(np.all(partners[amount > 0] != sources.reshape(-1)[amount > 0]))
        # Adjacent pixels in packed UV islands may be metres apart in 3D.
        positions[:, size // 2:, 2] += 2
        partners, amount = _seam_partners(positions.reshape(-1, 3), sources.reshape(-1), valid, size)
        self.assertFalse(amount.any())
        self.assertTrue(np.all(partners == -1))

    def test_spatially_close_but_disconnected_surfaces_do_not_blend(self):
        size = 16
        y, x = np.indices((size, size))
        positions = np.stack((x * 0.005, y * 0.005, np.ones_like(x)), axis=-1).astype(np.float32)
        sources = (x >= size // 2).astype(np.int32)
        face_ids = sources.reshape(-1)
        neighbors = np.full((2, 3), -1, np.int32)
        _, amount = _seam_partners(
            positions.reshape(-1, 3), sources.reshape(-1), np.ones(size * size, bool),
            size, face_ids=face_ids, neighbors=neighbors,
        )
        self.assertFalse(amount.any())

    def test_default_keeps_detail_when_photos_disagree_in_alignment(self):
        engine = texture_scene()
        rgb, depth = engine.raw_frames[0]
        y, x = np.indices(depth.shape)
        stripes = np.where((x // 12) % 2, 200, 55).astype(np.uint8)
        rgb[:] = stripes[..., None]
        engine.raw_frames.append((255 - rgb, depth.copy()))
        engine.poses.append((1, np.eye(4)))
        engine.frame_metadata.append({})
        for mode in ("blend", "best"):
            textured, report = make_textured_mesh(engine, size=256, blend_mode=mode)
            centers = textured.vertices[textured.faces].mean(axis=1)
            front = np.isclose(centers[:, 2], 1.2)
            uv = textured.visual.uv[textured.faces].mean(axis=1)
            colors = trimesh.visual.color.uv_to_color(uv, textured.visual.material.image)
            # Averaging inverse stripes produces a flat 127.5, destroying detail.
            self.assertTrue(np.all(np.abs(colors[front, :3].astype(float) - 127.5) > 40))
            self.assertEqual(0, report["seam_blending"]["softened_texels"])
            self.assertEqual("connected_surface_regions", report["source_selection"]["method"])


if __name__ == "__main__":
    unittest.main()
