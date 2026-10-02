import unittest

import numpy as np
import trimesh

from scanner_server.photometric import estimate_gains
from scanner_server.texturing import make_textured_mesh
from tests.test_features import texture_scene


class PhotometricTests(unittest.TestCase):
    def test_relative_gains_use_heldout_observed_samples(self):
        rng = np.random.default_rng(2)
        base = rng.uniform(0.1, 0.7, (1024, 3))
        samples = np.array([base, base * 0.8, base * 0.7])
        mask = np.ones((3, 1024), bool)
        gains, report = estimate_gains(samples, mask)
        self.assertTrue(report["applied"], report)
        np.testing.assert_allclose(gains[0], 1, atol=0.001)
        np.testing.assert_allclose(gains[1], 1.25, atol=0.001)
        np.testing.assert_allclose(gains[2], 1 / 0.7, atol=0.001)
        self.assertLess(report["validation_after"], report["validation_before"] * 0.01)

    def test_disconnected_or_variable_lighting_is_not_corrected(self):
        rng = np.random.default_rng(3)
        base = rng.uniform(0.1, 0.7, (1024, 3))
        samples = np.array([base, base * rng.uniform(0.7, 1.3, (1024, 1))])
        gains, report = estimate_gains(samples, np.ones((2, 1024), bool))
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(gains, 1)
        gains, report = estimate_gains(
            np.array([base, base * 0.8]), np.zeros((2, 1024), bool)
        )
        self.assertFalse(report["applied"])

    def test_texture_correction_and_best_view_preserve_occlusion(self):
        engine = texture_scene()
        rgb, depth = engine.raw_frames[0]
        rgb[:, :, 2] = 128
        engine.raw_frames.append(
            ((rgb.astype(float) * 0.8).astype(np.uint8), depth.copy())
        )
        engine.poses.append((1, np.eye(4)))
        engine.frame_metadata.append({})
        plain, _ = make_textured_mesh(engine, size=256)
        corrected, report = make_textured_mesh(
            engine, size=256, exposure_correction=True
        )
        best, best_report = make_textured_mesh(engine, size=256, blend_mode="best")
        self.assertTrue(report["exposure_correction"]["applied"], report)

        def front_colors(mesh):
            centers = mesh.vertices[mesh.faces].mean(axis=1)
            front = np.isclose(centers[:, 2], 1.2)
            uv = mesh.visual.uv[mesh.faces].mean(axis=1)
            colors = trimesh.visual.color.uv_to_color(uv, mesh.visual.material.image)
            expected = np.stack(
                (
                    (centers[front, 0] * 525 / 1.2 + 319.5) / 640 * 255,
                    (centers[front, 1] * 525 / 1.2 + 239.5) / 480 * 255,
                    np.full(front.sum(), 128),
                ),
                axis=-1,
            )
            return float(np.mean(np.abs(colors[front, :3] - expected)))

        self.assertLess(front_colors(corrected), front_colors(plain) * 0.4)
        self.assertLess(front_colors(best), 4)
        self.assertEqual("best", best_report["blend_mode"])
        engine.raw_frames = [(r, np.full_like(d, 900)) for r, d in engine.raw_frames]
        _, occluded = make_textured_mesh(
            engine, size=256, exposure_correction=True, blend_mode="best"
        )
        self.assertFalse(occluded["exposure_correction"]["applied"])
        self.assertEqual(0, occluded["projected_fraction"])
