"""Scanner viewpoint projection and software-rendered visibility."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
from PyQt6.QtWidgets import QApplication

from kinect_scanner.gui.live_view import LiveView
from shared.settings import CameraCalibration


class LiveViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.view = LiveView()
        self.view.resize(640, 480)

    def tearDown(self):
        self.view.close()

    def snapshot(self, points, pose=None, camera=None, **extra):
        self.view.set_snapshot({
            "points": np.asarray(points).tolist(),
            "camera_to_world": (np.eye(4) if pose is None else pose).tolist(),
            "camera": asdict(camera or CameraCalibration()),
            **extra,
        })

    def test_translation_rotation_and_perspective_match_camera_pixels(self):
        # A 90-degree turn plus translation makes a wrong transform direction
        # unmistakable. Near/far points on the same ray must project together.
        pose = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1.]])
        camera_points = np.array([[0, 0, 1], [0.21, 0.11, 1], [0.42, 0.22, 2]])
        world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
        self.snapshot(world_points, pose)
        xy, depth, indices = self.view._project_points(640, 480)
        np.testing.assert_array_equal(xy, [[320, 240], [430, 297], [430, 297]])
        np.testing.assert_array_equal(depth, [1, 1, 2])
        np.testing.assert_array_equal(indices, [0, 1, 2])
        self.snapshot(world_points, np.eye(4))
        self.assertFalse(np.array_equal(xy, self.view._project_points(640, 480)[0]))

    def test_custom_calibration_and_aspect_fit(self):
        camera = CameraCalibration(fx=600, fy=550, cx=300, cy=220)
        self.snapshot([[0, 0, 2], [0.2, 0.2, 2]], camera=camera)
        # 1000x480 leaves 180 pixels of horizontal padding on each side.
        xy, _, _ = self.view._project_points(1000, 480)
        np.testing.assert_array_equal(xy, [[480, 220], [540, 275]])
        xy, _, _ = self.view._project_points(320, 240)
        np.testing.assert_array_equal(xy, [[150, 110], [180, 138]])

    def test_clips_points_behind_camera_and_outside_sensor_field_of_view(self):
        self.snapshot([[0, 0, -1], [0, 0, 0], [2, 0, 1], [0, 2, 1], [0, 0, 1]])
        xy, depth, indices = self.view._project_points(1000, 480)
        np.testing.assert_array_equal(indices, [4])
        np.testing.assert_array_equal(xy, [[500, 240]])
        np.testing.assert_array_equal(depth, [1])

    def test_render_nearest_splat_and_empty_camera_view(self):
        self.snapshot([[0, 0, 2], [0, 0, 1]], colors=[[1, 0, 0], [0, 1, 0]])
        image = self.view.grab().toImage()
        viewport = self.view.drawing_rect
        xy, _, _ = self.view._project_points(viewport.width(), viewport.height())
        x, y = xy[0] + [viewport.x(), viewport.y()]
        for x, y in [(x, y), (x - 1, y - 1), (x + 1, y + 1)]:
            self.assertEqual((0, 255, 0), image.pixelColor(x, y).getRgb()[:3])
        self.snapshot([[0, 0, -1]])
        self.view.colored = False
        self.assertEqual((21, 32, 43), self.view.grab().toImage().pixelColor(int(x), int(y)).getRgb()[:3])

    def test_orbit_toggle_returns_to_latest_pose_and_reset_follows(self):
        self.snapshot([[0, 0, 1], [0.2, 0, 1]])
        self.assertTrue(self.view.follow_cb.isChecked())
        self.view.follow_cb.setChecked(False)
        orbit_xy = self.view._project_points(640, 480)[0]
        pose = np.eye(4)
        pose[0, 3] = 0.1
        self.snapshot([[0, 0, 1], [0.2, 0, 1]], pose, result={"success": False})
        np.testing.assert_array_equal(orbit_xy, self.view._project_points(640, 480)[0])
        self.view.follow_cb.setChecked(True)
        np.testing.assert_array_equal(self.view._project_points(640, 480)[0], [[267, 240], [372, 240]])
        self.view.follow_cb.setChecked(False)
        self.view.reset()
        self.assertTrue(self.view.follow_cb.isChecked())
        np.testing.assert_array_equal(self.view.camera_to_world, np.eye(4))

    def test_visible_controls_fit_narrow_view_and_switch_modes(self):
        self.view.resize(300, 240)
        self.view.show()
        self.app.processEvents()
        controls = [self.view.follow_button, self.view.orbit_button,
                    self.view.color_button, self.view.shape_button, self.view.fit_button]
        for button in controls:
            self.assertTrue(button.isVisible())
            self.assertTrue(button.accessibleName())
            position = button.mapTo(self.view, button.rect().topLeft())
            self.assertGreaterEqual(position.x(), 0)
            self.assertLessEqual(position.x() + button.width(), 300)
            self.assertGreaterEqual(button.width(), button.sizeHint().width())
        self.view.orbit_button.click()
        self.assertFalse(self.view.follow_cb.isChecked())
        self.view.follow_button.click()
        self.assertTrue(self.view.follow_cb.isChecked())
        self.view.shape_button.click()
        self.assertFalse(self.view.colored)
        self.view.color_button.click()
        self.assertTrue(self.view.colored)
        self.assertFalse(self.view.details_label.isVisible())
        self.view.details_button.click()
        self.assertTrue(self.view.details_label.isVisible())

    def test_fit_recalculates_bounds_after_cloud_grows_and_resets_orbit(self):
        self.snapshot([[0, 0, 1], [0.2, 0, 1]])
        self.snapshot([[0, 0, 1], [10, 0, 1]])
        self.view.yaw, self.view.pitch, self.view.zoom = 1, 1, 3
        self.view.fit_button.click()
        np.testing.assert_array_equal(self.view.center, [5, 0, 1])
        self.assertEqual(self.view.radius, 5)
        self.assertEqual((self.view.yaw, self.view.pitch, self.view.zoom), (0, 0, 1))
        self.assertFalse(self.view.follow_cb.isChecked())
        self.assertEqual(len(self.view._project_points(300, 240)[0]), 2)

    def test_fit_includes_points_omitted_from_bounded_display(self):
        with patch("kinect_scanner.gui.live_view.LIVE_MAX_POINTS", 2):
            self.snapshot([[0, 0, 1], [20, 0, 1], [1, 0, 1]])
        self.assertEqual(len(self.view.points), 2)
        self.view.fit_view()
        np.testing.assert_array_equal(self.view.center, [10, 0, 1])
        self.assertEqual(self.view.radius, 10)

    def test_expanded_details_reserve_uncovered_drawing_viewport(self):
        self.view.resize(300, 300)
        self.snapshot([[0, 0, 2], [0, 0, 1]], colors=[[1, 0, 0], [0, 1, 0]],
                      guidance="Move slowly and keep overlap", result={"success": True})
        self.view.show()
        self.app.processEvents()
        collapsed = self.view.drawing_rect
        self.view.details_button.click()
        self.app.processEvents()
        viewport = self.view.drawing_rect
        self.assertGreater(viewport.height(), 20)
        self.assertLess(viewport.height(), collapsed.height())
        self.assertEqual(viewport.top(), self.view.panel.geometry().bottom() + 1)
        xy, _, _ = self.view._project_points(viewport.width(), viewport.height())
        x, y = xy[0] + [viewport.x(), viewport.y()]
        image = self.view.grab().toImage()
        self.assertEqual((0, 255, 0), image.pixelColor(int(x), int(y)).getRgb()[:3])
        self.assertGreaterEqual(y, viewport.top())
        with patch.object(self.view, "_layout_panel", wraps=self.view._layout_panel) as layout:
            self.view.grab()
            layout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
