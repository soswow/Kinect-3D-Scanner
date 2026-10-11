"""Physical overlap, bounded color gains and original-detail preservation."""
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d

from scanner_server.surface_color import PhotoColorField, _bounded_log_gains, estimate_color_field
from shared.settings import ScanSettings


def plane():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[-.3,-.2,1.2], [0,-.2,1.2], [.3,-.2,1.2],
                                               [-.3,.2,1.2], [0,.2,1.2], [.3,.2,1.2]])
    mesh.triangles = o3d.utility.Vector3iVector([[0,3,1], [1,3,4], [1,4,2], [2,4,5]])
    return mesh


def photos():
    depth = np.full((480,640), 1.2, np.float32)
    return [(np.eye(4), np.full((480,640,3), color, np.uint8), depth.copy()) for color in (100,150)]


class SurfaceColorTests(unittest.TestCase):
    def test_held_out_overlap_matches_without_changing_geometry_or_photos(self):
        mesh = plane(); prepared = photos()
        faces = np.asarray(mesh.triangles).copy(); vertices = np.asarray(mesh.vertices).copy()
        field, report = estimate_color_field(mesh, prepared, ScanSettings())
        self.assertTrue(report['applied'], report)
        self.assertLess(report['validation_after'], .001)
        self.assertGreater(report['validation_before'], .09)
        points = vertices[faces].mean(axis=1)
        left = 100 * np.exp(field.evaluate(points, np.arange(4), 0))
        right = 150 * np.exp(field.evaluate(points, np.arange(4), 1))
        np.testing.assert_allclose(left, right, atol=.1)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.triangles, faces)
        np.testing.assert_array_equal(prepared[0][1], 100)
        # Original photo contrast survives a locally constant gain. No image
        # averaging, warping or additive gray offset affects dark details.
        gain = np.exp(field.evaluate(points[:1], np.array([0]), 0))[0]
        detail = np.array([0.,70.,150.])[:,None] * gain
        np.testing.assert_array_equal(detail[0], 0)
        np.testing.assert_allclose(detail[2] / detail[1], 150 / 70)

    def test_occluded_and_single_observation_surfaces_have_no_correction(self):
        prepared = photos(); prepared[1][2][:] = .8
        field, report = estimate_color_field(plane(), prepared, ScanSettings())
        self.assertFalse(report['applied']); self.assertIsNone(field)
        field, report = estimate_color_field(plane(), photos()[:1], ScanSettings())
        self.assertFalse(report['applied']); self.assertIsNone(field)

    def test_dark_color_channels_are_not_excluded_or_amplified_unboundedly(self):
        source = np.array([[[.4,.15,.005], [.5,.2,.02]]], np.float32)
        logs = _bounded_log_gains(np.array([[.45,.18,.025]], np.float32), source)
        self.assertTrue(np.isfinite(logs).all())
        self.assertGreater(np.exp(logs[0,0,0]), 1)
        self.assertLess(np.exp(logs[0,0,2]), 1.5)
        self.assertLessEqual(np.exp(logs).max(), 2.000001)
        self.assertGreaterEqual(np.exp(logs).min(), .499999)

    def test_pages_culling_and_winding_use_the_same_physical_field(self):
        mesh = plane(); vertices = np.asarray(mesh.vertices); faces = np.asarray(mesh.triangles)
        logs = np.repeat(vertices[:,None,:], 3, axis=1) * .02
        field = PhotoColorField(vertices, faces, logs, np.arange(4), np.arange(3))
        selected = np.array([3,1]); points = vertices[faces[selected]].mean(axis=1)
        page = field.subset(selected, [2,0])
        np.testing.assert_allclose(page.evaluate(points, np.arange(2), 0),
                                   field.evaluate(points, selected, 2))
        reversed_field = PhotoColorField(vertices, faces[:,::-1], logs, np.arange(4), np.arange(3))
        np.testing.assert_allclose(reversed_field.evaluate(points, selected, 2),
                                   field.evaluate(points, selected, 2))
        self.assertIs(page.log_gains, field.log_gains)

    def test_already_exposure_matched_photos_need_no_extra_field(self):
        field, report = estimate_color_field(plane(), photos(), ScanSettings(),
                                            gains=np.array([[1.5]*3,[1.]*3]))
        self.assertFalse(report['applied'])
        np.testing.assert_allclose(field.log_gains, 0, atol=1e-6)

    def test_contradictory_face_winding_does_not_cancel_fitted_normals(self):
        mesh = plane()
        expected, _ = estimate_color_field(mesh, photos(), ScanSettings())
        faces = np.asarray(mesh.triangles).copy(); faces[1] = faces[1, ::-1]
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        actual, report = estimate_color_field(mesh, photos(), ScanSettings())
        self.assertTrue(report['applied'], report)
        np.testing.assert_allclose(actual.log_gains, expected.log_gains, atol=1e-6)
        np.testing.assert_array_equal(mesh.triangles, faces)

    def test_bad_disconnected_fit_keeps_original_colors_without_discarding_good_surface(self):
        mesh = plane() + plane().translate((.2,0,0))
        def candidate(target, source):
            logs = _bounded_log_gains(target, source)
            logs[6:] *= -1
            return logs
        with patch('scanner_server.surface_color._bounded_log_gains', side_effect=candidate):
            field, report = estimate_color_field(mesh, photos(), ScanSettings())
        self.assertTrue(report['applied'], report)
        self.assertEqual(1, report['rejected_components'])
        self.assertTrue(np.any(field.log_gains[:6] != 0))
        np.testing.assert_array_equal(field.log_gains[6:], 0)


if __name__ == '__main__': unittest.main()
