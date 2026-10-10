"""Scanner viewpoint projection and software-rendered visibility."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QPushButton

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

    def test_follow_backs_up_along_rotated_camera_axis_with_perspective(self):
        # A 90-degree turn plus translation makes a wrong transform direction
        # unmistakable. The backward offset widens the view without rotating it;
        # perspective changes with distance from the new viewpoint.
        pose = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1.]])
        camera_points = np.array([[0, 0, 1], [0.21, 0.11, 1], [0.42, 0.22, 2]])
        world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
        self.snapshot(world_points, pose)
        xy, depth, indices = self.view._project_points(640, 480)
        np.testing.assert_array_equal(xy, [[320, 240], [393, 278], [408, 286]])
        np.testing.assert_array_equal(depth, [1.5, 1.5, 2.5])
        np.testing.assert_array_equal(indices, [0, 1, 2])
        self.snapshot(world_points, np.eye(4))
        self.assertFalse(np.array_equal(xy, self.view._project_points(640, 480)[0]))

    def test_custom_calibration_and_aspect_fit(self):
        camera = CameraCalibration(fx=600, fy=550, cx=300, cy=220)
        self.snapshot([[0, 0, 2], [0.2, 0.2, 2]], camera=camera)
        # 1000x480 leaves 180 pixels of horizontal padding on each side.
        xy, _, _ = self.view._project_points(1000, 480)
        np.testing.assert_array_equal(xy, [[480, 220], [528, 264]])
        xy, _, _ = self.view._project_points(320, 240)
        np.testing.assert_array_equal(xy, [[150, 110], [174, 132]])

    def test_clips_at_backed_up_viewpoint_and_retains_newly_visible_surface(self):
        self.snapshot([[0, 0, -1], [0, 0, -0.5], [2, 0, 1], [0, 2, 1],
                       [0, 0, 1], [0.7, 0, 1], [0, 0, -0.25]])
        xy, depth, indices = self.view._project_points(1000, 480)
        np.testing.assert_array_equal(indices, [4, 5, 6])
        np.testing.assert_array_equal(xy, [[500, 240], [744, 240], [500, 240]])
        np.testing.assert_array_equal(depth, [1.5, 1.5, 0.25])

    def test_disconnected_guidance_survives_updates_until_feedback_returns(self):
        self.snapshot([[0, 0, 1]], guidance="Move around the subject")
        self.view.set_feedback_connected(False)
        self.view._tick()
        self.assertIn("Pause movement", self.view.guidance_label.text())
        self.snapshot([[0, 0, 1]], guidance="Move slowly with overlap")
        self.assertIn("Pause movement", self.view.guidance_label.text())
        self.view.set_feedback_connected(True)
        self.assertEqual("Move slowly with overlap", self.view.guidance_label.text())

    def test_surface_coverage_explanation_is_available_on_hover(self):
        text = "Live preview includes tentative surface. Finish may remove weak or unconnected areas."
        self.snapshot([[0, 0, 1]], surface_description=text)
        self.assertTrue(self.view.surface_label.isHidden())
        self.assertEqual(text, self.view.title_label.toolTip())

    def test_render_nearest_splat_and_empty_camera_view(self):
        self.snapshot([[0, 0, 2], [0, 0, 1]], colors=[[1, 0, 0], [0, 1, 0]])
        image = self.view.grab().toImage()
        viewport = self.view.drawing_rect
        xy, _, _ = self.view._project_points(viewport.width(), viewport.height())
        x, y = xy[0] + [viewport.x(), viewport.y()]
        for sample_x, sample_y in [(x, y), (x - 1, y - 1), (x + 1, y + 1)]:
            self.assertEqual((0, 255, 0), image.pixelColor(sample_x, sample_y).getRgb()[:3])
        self.snapshot([[0, 0, -1]])
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
        np.testing.assert_array_equal(self.view._project_points(640, 480)[0], [[284, 240], [354, 240]])
        self.view.follow_cb.setChecked(False)
        self.view.reset()
        self.assertTrue(self.view.follow_cb.isChecked())
        np.testing.assert_array_equal(self.view.camera_to_world, np.eye(4))

    def test_visible_controls_fit_narrow_view_and_switch_modes(self):
        self.view.resize(300, 240)
        self.view.show()
        self.app.processEvents()
        controls = [self.view.follow_button, self.view.orbit_button, self.view.fit_button]
        self.assertEqual(self.view.panel.findChildren(QPushButton), controls)
        self.assertEqual(len({button.y() for button in controls}), 1)
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

    def test_fit_recalculates_bounds_after_cloud_grows_and_resets_orbit(self):
        self.snapshot([[0, 0, 1], [0.2, 0, 1]])
        self.snapshot([[0, 0, 1], [10, 0, 1]])
        self.view.yaw, self.view.pitch, self.view.zoom = 1, 1, 3
        self.view.pan[:] = [2, -3]
        self.view.fit_button.click()
        np.testing.assert_array_equal(self.view.center, [5, 0, 1])
        self.assertEqual(self.view.radius, 5)
        self.assertEqual((self.view.yaw, self.view.pitch, self.view.zoom), (0, 0, 1))
        np.testing.assert_array_equal(self.view.pan, [0, 0])
        self.assertFalse(self.view.follow_cb.isChecked())
        self.assertEqual(len(self.view._project_points(300, 240)[0]), 2)

    def drag(self, button, modifiers=Qt.KeyboardModifier.NoModifier, delta=(20, 15)):
        start = QPointF(self.view.drawing_rect.center())
        end = start + QPointF(*delta)
        for kind, position, changed_button, held_buttons in (
            (QEvent.Type.MouseButtonPress, start, button, button),
            (QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, button),
            (QEvent.Type.MouseButtonRelease, end, button, Qt.MouseButton.NoButton),
        ):
            self.app.sendEvent(self.view, QMouseEvent(
                kind, position, position, changed_button, held_buttons, modifiers,
            ))

    def test_left_drag_orbits_and_other_drags_pan_in_screen_plane(self):
        self.snapshot([[-0.1, -0.1, 1], [0.1, 0.1, 1.1]])
        self.view.fit_view()
        # Leave room to pan at both zoom levels without clipping the samples.
        self.view.radius *= 2
        self.drag(Qt.MouseButton.LeftButton)
        self.assertAlmostEqual(self.view.yaw, 0.16)
        self.assertAlmostEqual(self.view.pitch, 0.12)
        np.testing.assert_array_equal(self.view.pan, [0, 0])
        for button, modifiers in (
            (Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier),
            (Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier),
            (Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier),
        ):
            for zoom in (0.5, 2):
                with self.subTest(button=button, zoom=zoom):
                    self.view.pan[:] = 0
                    self.view.zoom = zoom
                    # Pan must stay horizontal/vertical even after rotation.
                    self.view.yaw, self.view.pitch = 0.8, -0.3
                    viewport = self.view.drawing_rect
                    before, depth, indices = self.view._project_points(viewport.width(), viewport.height())
                    self.drag(button, modifiers)
                    after, new_depth, new_indices = self.view._project_points(viewport.width(), viewport.height())
                    np.testing.assert_array_equal(after - before, [[20, 15], [20, 15]])
                    np.testing.assert_array_equal(new_depth, depth)
                    np.testing.assert_array_equal(new_indices, indices)
                    self.assertEqual((self.view.yaw, self.view.pitch), (0.8, -0.3))
                    self.assertIsNone(self.view._drag)

    def test_follow_ignores_navigation_and_snapshot_preserves_pan(self):
        self.snapshot([[0, 0, 1], [0.2, 0, 1]])
        for button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.drag(button)
        np.testing.assert_array_equal(self.view.pan, [0, 0])
        self.assertEqual((self.view.yaw, self.view.pitch), (0, 0))
        self.view.fit_view()
        self.drag(Qt.MouseButton.RightButton)
        pan = self.view.pan.copy()
        before = self.view._project_points(640, 480)[0]
        self.snapshot([[0, 0, 1], [0.2, 0, 1], [0.1, 0, 1]])
        np.testing.assert_array_equal(self.view.pan, pan)
        np.testing.assert_array_equal(self.view._project_points(640, 480)[0][:2], before)
        self.view.reset()
        np.testing.assert_array_equal(self.view.pan, [0, 0])

    def test_drag_outside_cloud_viewport_does_not_navigate(self):
        self.snapshot([[0, 0, 1]])
        self.view.fit_view()
        position = QPointF(5, 5)
        self.app.sendEvent(self.view, QMouseEvent(
            QEvent.Type.MouseButtonPress, position, position,
            Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        ))
        self.assertIsNone(self.view._drag)

    def test_fit_includes_points_omitted_from_bounded_display(self):
        with patch("kinect_scanner.gui.live_view.LIVE_MAX_POINTS", 2):
            self.snapshot([[0, 0, 1], [20, 0, 1], [1, 0, 1]])
        self.assertEqual(len(self.view.points), 2)
        self.view.fit_view()
        np.testing.assert_array_equal(self.view.center, [10, 0, 1])
        self.assertEqual(self.view.radius, 10)

    def test_diagnostics_on_hover_reserve_uncovered_drawing_viewport(self):
        self.view.resize(300, 300)
        self.snapshot([[0, 0, 2], [0, 0, 1]], colors=[[1, 0, 0], [0, 1, 0]],
                      guidance="Move slowly and keep overlap", result={"success": True})
        self.view.show()
        self.app.processEvents()
        viewport = self.view.drawing_rect
        self.assertGreater(viewport.height(), 20)
        self.assertIn("tracking accepted", self.view.status_label.toolTip())
        self.assertIn("displayed: 2 points", self.view.status_label.toolTip())
        self.assertEqual(viewport.top(), self.view.panel.geometry().bottom() + 1)
        xy, _, _ = self.view._project_points(viewport.width(), viewport.height())
        x, y = xy[0] + [viewport.x(), viewport.y()]
        image = self.view.grab().toImage()
        self.assertEqual((0, 255, 0), image.pixelColor(int(x), int(y)).getRgb()[:3])
        self.assertGreaterEqual(y, viewport.top())
        self.app.sendEvent(self.view, QMouseEvent(
            QEvent.Type.MouseButtonDblClick, QPointF(x, y), QPointF(x, y),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ))
        self.assertEqual((0, 255, 0), self.view.grab().toImage().pixelColor(int(x), int(y)).getRgb()[:3])
        with patch.object(self.view, "_layout_panel", wraps=self.view._layout_panel) as layout:
            self.view.grab()
            layout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
