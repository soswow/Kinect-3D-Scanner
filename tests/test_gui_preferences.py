"""User defaults persist immediately and remain separate from restored sessions."""

import json
import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QSpinBox

from kinect_scanner.gui import dialogs, main_window, preferences
from shared.sensor_calibration import SensorCalibration, load_calibration
from shared.settings import ScanSettings
from tests.test_auto_capture import NoCamera, NoTasks


class GuiPreferencesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = str(Path(self.folder.name) / "scanner.ini")
        self.store = self.new_store()
        self.windows = []
        self.patches = [
            patch.object(main_window, "KinectWorker", NoCamera),
            patch.object(main_window, "ServerTaskWorker", NoTasks),
            patch.dict(os.environ, {}, clear=False),
        ]
        for item in self.patches:
            item.start()
        for key in ("KINECT_AUTOCONNECT", "KINECT_SERVER_HOST", "KINECT_SERVER_PORT"):
            os.environ.pop(key, None)

    def tearDown(self):
        for window in self.windows:
            window._close_approved = True
            window.close()
            window.worker.wait(2500)
            window.task_worker.wait(2500)
        self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.folder.cleanup()

    def new_store(self):
        return preferences.ScannerPreferences(QSettings(self.path, QSettings.Format.IniFormat))

    def window(self, store=None):
        window = main_window.MainWindow(preferences=store or self.new_store())
        self.windows.append(window)
        return window

    def custom_calibration(self):
        document = load_calibration().to_dict()
        document["camera_serial"] = "preferences-test-device"
        document["raw_depth_to_mm"]["scale"] = 1.01
        return SensorCalibration.from_dict(document)

    def test_portrait_changes_preview_and_metadata_without_rotating_measurements(self):
        window = self.window()
        window.orientation_combo.setCurrentIndex(window.orientation_combo.findData("portrait_left"))
        rgb = np.full((480, 640, 3), 42, np.uint8)
        depth = np.full((480, 640), 512, np.uint16)
        with patch.object(window, "_set_pixmap") as render:
            window._on_frame(rgb, depth, {"timestamp_s": 1})
        image = render.call_args.args[0]
        self.assertEqual((image.width(), image.height()), (480, 640))
        self.assertEqual(window._last_rgb.shape, (480, 640, 3))
        self.assertEqual(window._last_depth.shape, (480, 640))
        self.assertEqual(window._last_frame_metadata["orientation"]["rotation_cw_degrees"], 90)
        self.assertEqual(self.new_store().read("camera/orientation", "auto"), "portrait_left")

    def test_bind_saves_immediately_and_suspend_preserves_user_value(self):
        self.store.write("test/count", 8)
        widget = QSpinBox()
        widget.setRange(0, 100)
        self.store.bind(widget, "test/count")
        self.assertEqual(8, widget.value())
        widget.setValue(13)
        self.assertEqual(13, self.new_store().read("test/count", 0))
        with self.store.suspend():
            widget.setValue(21)
        self.assertEqual(13, self.new_store().read("test/count", 0))
        self.store.restore(widget, "test/count")
        self.assertEqual(13, widget.value())

    def test_user_controls_survive_new_window_before_first_closes(self):
        first = self.window()
        values = {
            "depth_near_spin": 700, "depth_far_spin": 2800,
            "crop_spin": 65, "voxel_spin": 8, "final_voxel_spin": 6,
            "final_blocks_spin": 6200, "weight_spin": 3.5,
            "server_port_spin": 8123,
        }
        for name, value in values.items():
            getattr(first, name).setValue(value)
        checks = {
            "crop_cb": True, "record_cb": True, "live_cb": False,
            "color_tracking_cb": True, "refine_cb": True,
            "relocalize_cb": True, "confidence_cb": True,
            "reconnect_fragments_cb": False,
        }
        for name, value in checks.items():
            getattr(first, name).setChecked(value)
        first.capture_mode_combo.setCurrentIndex(first.capture_mode_combo.findData("manual"))
        first.rgb_mode_combo.setCurrentIndex(first.rgb_mode_combo.findData("rgb_low_res"))
        first.auto_capture_spin.set_interval_seconds(0.4)
        first.server_ip_edit.setText("scanner.local")
        second = self.window()
        for name, expected in values.items():
            self.assertEqual(expected, getattr(second, name).value(), name)
        for name, expected in checks.items():
            self.assertEqual(expected, getattr(second, name).isChecked(), name)
        self.assertEqual("manual", second.capture_mode_combo.currentData())
        self.assertEqual("rgb_low_res", second.rgb_mode_combo.currentData())
        self.assertAlmostEqual(0.4, second.auto_capture_spin.interval_seconds)
        self.assertEqual(12, second.auto_capture_spin.value())
        self.assertIn("12 fresh RGB/depth frames", second.auto_capture_spin.toolTip())
        self.assertIn("30 fps", second.auto_capture_spin.toolTip())
        self.assertEqual("scanner.local", second.server_ip_edit.text())
        self.assertFalse(second._scanning)
        self.assertFalse(second.auto_capture_cb.isChecked())

    def test_restored_automatic_mode_does_not_start_capture(self):
        self.store.write("capture/mode", "automatic")
        window = self.window()
        self.assertEqual("automatic", window.capture_mode_combo.currentData())
        self.assertFalse(window._scanning)
        self.assertFalse(window.auto_capture_cb.isChecked())
        self.assertEqual([], window.task_worker.tasks)

    def test_rgb_exposure_persists_and_restores_before_camera_start(self):
        with patch.object(main_window, "KinectWorker", wraps=NoCamera) as camera:
            first = self.window()
            self.assertFalse(first.rgb_shutter_spin.isEnabled())
            self.assertFalse(first.rgb_gain_combo.isEnabled())
            first.rgb_exposure_combo.setCurrentIndex(first.rgb_exposure_combo.findData("manual"))
            first.rgb_shutter_spin.setValue(250)
            first.rgb_gain_combo.setCurrentIndex(first.rgb_gain_combo.findData(4))
            self.assertTrue(first.rgb_shutter_spin.isEnabled())
            self.assertTrue(first.rgb_gain_combo.isEnabled())
            self.assertEqual(4, camera.call_args.kwargs["rgb_gain"])
            self.assertEqual("manual", camera.call_args.kwargs["rgb_exposure_mode"])
            self.assertEqual(250, camera.call_args.kwargs["rgb_shutter_speed"])
            self.assertIsNone(first._last_rgb)
            second = self.window()
            self.assertEqual("manual", second.rgb_exposure_combo.currentData())
            self.assertEqual(250, second.rgb_shutter_spin.value())
            self.assertEqual(4, second.rgb_gain_combo.currentData())
            self.assertEqual(4, camera.call_args.kwargs["rgb_gain"])
            self.assertEqual(250, camera.call_args.kwargs["rgb_shutter_speed"])

    def test_resolution_change_clamps_shutter_and_restarts_camera_once(self):
        window = self.window()
        window.rgb_exposure_combo.setCurrentIndex(window.rgb_exposure_combo.findData("manual"))
        window.rgb_shutter_spin.setValue(10)
        with patch.object(window, "_restart_camera", wraps=window._restart_camera) as restart:
            window.rgb_mode_combo.setCurrentIndex(window.rgb_mode_combo.findData("rgb_low_res"))
        restart.assert_called_once()
        self.assertEqual(30, window.rgb_shutter_spin.value())
        self.assertEqual(30, self.new_store().read("camera/shutter_speed", 0))

    def test_session_exposure_is_restored_without_replacing_user_defaults(self):
        self.store.write("camera/exposure_mode", "manual")
        self.store.write("camera/shutter_speed", 250)
        self.store.write("camera/gain", 2)
        window = self.window()
        settings = ScanSettings(sensor_calibration=load_calibration(),
                                rgb_exposure_mode="manual", rgb_shutter_speed=500, rgb_gain=8)
        with patch.object(main_window, "KinectWorker", wraps=NoCamera) as camera:
            window._apply_session_settings(settings.to_dict())
        self.assertEqual(500, window.rgb_shutter_spin.value())
        self.assertEqual(500, camera.call_args.kwargs["rgb_shutter_speed"])
        self.assertEqual(8, window.rgb_gain_combo.currentData())
        self.assertEqual(8, camera.call_args.kwargs["rgb_gain"])
        self.assertEqual(2, self.new_store().read("camera/gain", 0))
        self.assertEqual(250, self.new_store().read("camera/shutter_speed", 0))
        reopened = self.window()
        self.assertEqual(250, reopened.rgb_shutter_spin.value())
        self.assertEqual(2, reopened.rgb_gain_combo.currentData())

    def test_invalid_gain_preferences_keep_default(self):
        for value in (True, 2.0, "2", 3):
            with self.subTest(value=value):
                self.store.write("camera/gain", value)
                self.assertEqual(1, self.window().rgb_gain_combo.currentData())

    def test_server_session_settings_do_not_replace_saved_defaults(self):
        custom = self.custom_calibration()
        self.store.save_calibration(custom)
        self.store.write("scan/rgb_mode", "rgb_low_res")
        self.store.write("capture/interval_s", 0.4)
        self.store.write("scan/voxel_mm", 8.0)
        self.store.write("scan/final_voxel_mm", 6.0)
        first = self.window()
        before = {key: self.new_store().settings.value(key) for key in self.new_store().settings.allKeys()}
        session = ScanSettings(sensor_calibration=load_calibration(), rgb_mode="rgb_high_res",
                               near_m=0.8, far_m=2, voxel_m=0.004,
                               final_voxel_m=0.002, final_weight=4,
                               live_reconstruction=True, refine_poses=True,
                               reconnect_fragments=False)
        first._apply_session_settings(session.to_dict())
        self.assertFalse(first.reconnect_fragments_cb.isChecked())
        after = {key: self.new_store().settings.value(key) for key in self.new_store().settings.allKeys()}
        self.assertEqual(before, after)
        reopened = self.window()
        self.assertEqual("rgb_low_res", reopened.rgb_mode_combo.currentData())
        self.assertAlmostEqual(0.4, reopened.auto_capture_spin.interval_seconds)
        self.assertEqual(8, reopened.voxel_spin.value())
        self.assertEqual(6, reopened.final_voxel_spin.value())
        self.assertEqual(custom.to_dict(), reopened._sensor_calibration.to_dict())
        self.assertTrue(reopened.reconnect_fragments_cb.isChecked())

    def test_invalid_stored_values_fall_back_safely(self):
        invalid = {
            "capture/mode": "unknown", "capture/interval_s": "not-a-number",
            "scan/rgb_mode": "unsupported", "scan/near_mm": "wrong",
            "scan/far_mm": "wrong", "scan/voxel_mm": float("nan"),
            "scan/final_voxel_mm": "wrong", "connection/port": "wrong",
            "scan/crop_enabled": "not-a-boolean", "calibration": "broken JSON",
        }
        for key, value in invalid.items():
            self.store.write(key, value)
        # Corruption can bypass write(), including nonfinite JSON values.
        self.store.settings.setValue("preferences/scan/voxel_mm", "NaN")
        self.store.settings.setValue("preferences/scan/final_blocks", "999999")
        self.store.settings.setValue("preferences/export/format", "{broken")
        self.store.settings.sync()
        window = self.window()
        self.assertEqual("automatic", window.capture_mode_combo.currentData())
        self.assertEqual("rgb_high_res", window.rgb_mode_combo.currentData())
        self.assertEqual(500, window.depth_near_spin.value())
        self.assertEqual(4000, window.depth_far_spin.value())
        self.assertEqual(5, window.voxel_spin.value())
        self.assertEqual(0, window.final_voxel_spin.value())
        self.assertAlmostEqual(0.5, window.auto_capture_spin.interval_seconds)
        self.assertEqual(8000, window.server_port_spin.value())
        self.assertEqual(5000, window.final_blocks_spin.value())
        self.assertFalse(window.crop_cb.isChecked())
        self.assertEqual(load_calibration().to_dict(), window._sensor_calibration.to_dict())

    def test_nested_suspend_unwinds_after_exception(self):
        self.store.write("test/value", "original")
        with self.assertRaises(RuntimeError), self.store.suspend():
            with self.store.suspend():
                self.store.write("test/value", "inner")
            self.store.write("test/value", "outer")
            raise RuntimeError("session restoration interrupted")
        self.assertEqual("original", self.new_store().read("test/value", None))
        self.store.write("test/value", "later edit")
        self.assertEqual("later edit", self.new_store().read("test/value", None))

    def test_huge_json_integer_falls_back_without_overflow(self):
        huge_integer = "1" + "0" * 1000
        for key in ("scan/voxel_mm", "capture/interval_s", "connection/port"):
            self.store.settings.setValue(f"preferences/{key}", huge_integer)
        self.store.settings.sync()
        window = self.window()
        self.assertEqual(5, window.voxel_spin.value())
        self.assertAlmostEqual(0.5, window.auto_capture_spin.interval_seconds)
        self.assertEqual(8000, window.server_port_spin.value())

    def test_sound_toggle_is_remembered_in_injected_backing_store(self):
        first = self.window()
        self.assertTrue(first.sound_action.isChecked())
        first.sound_action.setChecked(False)
        self.assertFalse(first.capture_sound.enabled)
        second = self.window()
        self.assertFalse(second.sound_action.isChecked())
        self.assertFalse(second.capture_sound.enabled)
        second.sound_action.setChecked(True)
        third = self.window()
        self.assertTrue(third.sound_action.isChecked())
        self.assertTrue(third.capture_sound.enabled)

    def test_custom_calibration_snapshot_restores_without_source_file(self):
        custom = self.custom_calibration()
        self.store.save_calibration(custom)
        restored = self.new_store().load_calibration(load_calibration())
        self.assertEqual(custom.to_dict(), restored.to_dict())
        window = self.window()
        self.assertEqual(custom.to_dict(), window._sensor_calibration.to_dict())
        self.assertEqual(custom.depth, window._camera)

    def test_loading_calibration_saves_snapshot_immediately(self):
        custom = self.custom_calibration()
        source = Path(self.folder.name) / "custom-calibration.json"
        source.write_text(json.dumps(custom.to_dict()), encoding="utf-8")
        first = self.window()
        with patch.object(main_window.QFileDialog, "getOpenFileName", return_value=(str(source), "JSON (*.json)")):
            first._load_calibration()
        source.unlink()
        second = self.window()
        self.assertEqual(custom.to_dict(), second._sensor_calibration.to_dict())

    def test_connection_environment_overrides_do_not_replace_saved_target(self):
        self.store.write("connection/host", "saved.local")
        self.store.write("connection/port", 8123)
        with patch.dict(os.environ, {"KINECT_SERVER_HOST": "override.local", "KINECT_SERVER_PORT": "9001"}):
            overridden = self.window()
        self.assertEqual("override.local", overridden.server_ip_edit.text())
        self.assertEqual(9001, overridden.server_port_spin.value())
        reopened = self.window()
        self.assertEqual("saved.local", reopened.server_ip_edit.text())
        self.assertEqual(8123, reopened.server_port_spin.value())

    def test_export_dialog_restores_choices_and_cancel_does_not_save_changes(self):
        self.store.write("export/format", "ply")
        self.store.write("export/exposure_correction", True)
        self.store.write("export/best_source", True)
        dialog = dialogs.ExportDialog(preferences=self.store)
        self.assertEqual("ply", dialog.selected_format)
        self.assertEqual({"exposure_correction": True, "blend_mode": "best"}, dialog.texture_options)
        dialog.format_combo.setCurrentIndex(dialog.format_combo.findData("obj.zip"))
        dialog.reject()
        self.assertEqual("ply", self.new_store().read("export/format", "glb"))

    def test_accepted_export_choices_are_remembered(self):
        window = self.window(self.store)
        window.server_client._connected = True
        window._has_mesh = True
        window._refresh_controls()
        dialog = dialogs.ExportDialog(preferences=self.store)
        dialog.format_combo.setCurrentIndex(dialog.format_combo.findData("obj.zip"))
        dialog.texture_exposure_cb.setChecked(True)
        dialog.texture_best_cb.setChecked(True)
        with patch.object(main_window, "ExportDialog", return_value=dialog), \
             patch.object(dialog, "exec", return_value=1), \
             patch.object(window, "_export_texture") as export:
            window._choose_export()
        export.assert_called_once_with("obj.zip")
        restored = dialogs.ExportDialog(preferences=self.new_store())
        self.assertEqual("obj.zip", restored.selected_format)
        self.assertEqual({"exposure_correction": True, "blend_mode": "best"}, restored.texture_options)
        restored.close()
        dialog.close()


if __name__ == "__main__":
    unittest.main()
