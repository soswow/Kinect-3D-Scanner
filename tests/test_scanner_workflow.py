"""Scan recovery and protection must retain frames until saving succeeds."""

import os
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QPoint, QSettings
from PyQt6.QtWidgets import QApplication

from kinect_scanner.gui import dialogs, main_window
from kinect_scanner.gui.preferences import ScannerPreferences
from kinect_scanner.server_task_worker import ServerTaskType
from tests.test_auto_capture import NoCamera, NoTasks


class ScannerWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.patches = [
            patch.object(main_window, "KinectWorker", NoCamera),
            patch.object(main_window, "ServerTaskWorker", NoTasks),
            patch.object(main_window.QMessageBox, "information"),
            patch.object(main_window.QMessageBox, "warning"),
        ]
        for item in self.patches:
            item.start()
        self.preferences_dir = tempfile.TemporaryDirectory()
        store = QSettings(os.path.join(self.preferences_dir.name, "scanner.ini"), QSettings.Format.IniFormat)
        self.window = main_window.MainWindow(preferences=ScannerPreferences(store))
        self.window.server_client._connected = True
        self.rgb = np.full((2, 2, 3), 42, np.uint8)
        self.depth = np.full((2, 2), 750, np.uint16)
        self.fresh_frame()

    def tearDown(self):
        self.window._close_approved = True
        self.window.close()
        self.window.worker.wait(2500)
        self.window.task_worker.wait(2500)
        self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.preferences_dir.cleanup()

    def fresh_frame(self):
        self.window._on_frame(self.rgb, self.depth, {"captured_monotonic_s": time.monotonic()})

    def retain_scan(self):
        self.window._session_id = "retained"
        self.window._server_stored = 3
        self.window._server_integrated = 2
        self.window._session_dirty = True
        self.window._scanning = True
        self.window._paused = False

    def task_types(self):
        return [task.task_type for task in self.window.task_worker.tasks]

    def protection(self, choice, path="/tmp/workflow-session.zip"):
        dialog_patch = patch.object(dialogs, "SessionProtectionDialog")
        mocked = dialog_patch.start()
        mocked.return_value.choice = choice
        mocked.return_value.exec.return_value = choice != "cancel"
        self.addCleanup(dialog_patch.stop)
        # Support a module-level import as well as a lazy dialog import.
        if hasattr(main_window, "SessionProtectionDialog"):
            alias_patch = patch.object(main_window, "SessionProtectionDialog", mocked)
            alias_patch.start()
            self.addCleanup(alias_patch.stop)
        file_patch = patch.object(main_window.QFileDialog, "getSaveFileName", return_value=(path, "ZIP (*.zip)"))
        file_patch.start()
        self.addCleanup(file_patch.stop)

    def test_automatic_begins_only_after_reset_acknowledgement(self):
        self.assertEqual("automatic", self.window.capture_mode_combo.currentData())
        self.assertFalse(self.window.auto_capture_cb.isChecked())
        self.window._start_scan()
        self.assertEqual([ServerTaskType.RESET], self.task_types())
        self.assertFalse(self.window.auto_capture_cb.isChecked())
        self.window._on_reset_done({"session_id": "new", "settings": {"live_reconstruction": True}})
        self.assertTrue(self.window._scanning)
        self.assertTrue(self.window.auto_capture_cb.isChecked())

    def test_invalid_range_stays_disabled_after_camera_update(self):
        self.window.depth_near_spin.setValue(2000)
        self.window.depth_far_spin.setValue(1000)
        self.fresh_frame()
        self.assertFalse(self.window.btn_start_scan.isEnabled())
        self.window._start_scan()
        self.assertNotIn(ServerTaskType.RESET, self.task_types())

    def test_stale_camera_disables_capture_without_losing_session(self):
        self.retain_scan()
        self.window._last_frame_time = time.monotonic() - 2
        self.window._refresh_controls()
        self.assertFalse(self.window._camera_ready())
        self.assertFalse(self.window.btn_capture.isEnabled())
        self.assertFalse(self.window.btn_start_scan.isEnabled())
        self.assertTrue(self.window.btn_stop_build.isEnabled())
        self.assertEqual(3, self.window._server_stored)

    def test_failed_build_can_retry_or_resume_without_reset(self):
        self.retain_scan()
        self.window._stop_and_build()
        self.window._on_build_mesh_done(False, "Capture more overlapping views")
        self.assertEqual("Retry Build", self.window.btn_stop_build.text())
        self.assertTrue(self.window.btn_stop_build.isEnabled())
        self.window._stop_and_build()
        self.window._on_build_mesh_done(False, "Still no surface")
        self.window._pause_or_resume()
        self.assertTrue(self.window._scanning)
        self.assertFalse(self.window._paused)
        self.assertEqual(3, self.window._server_stored)
        self.assertEqual([ServerTaskType.BUILD_MESH, ServerTaskType.BUILD_MESH], self.task_types())

    def test_cancel_protection_restores_capture_and_retains_frames(self):
        self.retain_scan()
        self.protection("cancel")
        self.assertFalse(self.window._protect_session("new_scan"))
        self.assertFalse(self.window._paused)
        self.assertTrue(self.window._session_dirty)
        self.assertEqual(3, self.window._server_stored)
        self.assertEqual([], self.task_types())

    def test_cancelled_save_path_does_not_continue(self):
        self.retain_scan()
        self.protection("save", path="")
        self.assertFalse(self.window._protect_session("close"))
        self.assertFalse(self.window._paused)
        self.assertEqual([], self.task_types())
        self.assertTrue(self.window._session_dirty)

    def test_protection_save_waits_for_success_before_new_scan(self):
        self.retain_scan()
        self.protection("save")
        self.assertFalse(self.window._protect_session("new_scan"))
        self.assertEqual("new_scan", self.window._pending_action)
        self.assertEqual([ServerTaskType.EXPORT_SESSION], self.task_types())
        self.assertTrue(self.window._paused)
        self.window._on_export_done(True, "/tmp/workflow-session.zip")
        self.assertEqual([ServerTaskType.EXPORT_SESSION, ServerTaskType.RESET], self.task_types())

    def test_failed_protection_save_does_not_reset_or_close(self):
        self.retain_scan()
        self.protection("save")
        self.window._protect_session("close")
        with patch.object(self.window, "close") as close:
            self.window._on_export_done(False, "/tmp/workflow-session.zip")
        close.assert_not_called()
        self.assertEqual([ServerTaskType.EXPORT_SESSION], self.task_types())
        self.assertEqual(3, self.window._server_stored)
        self.assertTrue(self.window._session_dirty)
        self.assertFalse(self.window._paused)
        self.assertFalse(self.window._pending_action)

    def test_reconnect_adopts_retained_session_without_capture_or_reset(self):
        self.window._restore_server_session({
            "session_id": "restored", "stored_count": 7, "frame_count": 5,
            "has_mesh": True, "settings": {"live_reconstruction": True},
        })
        self.assertEqual("restored", self.window._session_id)
        self.assertEqual(7, self.window._server_stored)
        self.assertEqual(5, self.window._server_integrated)
        self.assertTrue(self.window._has_mesh)
        self.assertFalse(self.window.auto_capture_cb.isChecked())
        self.assertNotIn(ServerTaskType.RESET, self.task_types())

    def test_reconnect_protects_new_frames_in_previously_saved_session(self):
        self.retain_scan()
        self.window._session_dirty = False
        self.window._restore_server_session({
            "session_id": "retained", "stored_count": 5, "frame_count": 3,
            "has_mesh": False, "settings": {},
        })
        self.assertTrue(self.window._session_dirty)
        self.assertTrue(self.window._paused)
        self.assertFalse(self.window.auto_capture_cb.isChecked())

    def test_transport_timeout_reconciles_completed_server_build(self):
        self.retain_scan()
        self.window._build_pending = True
        self.window._on_task_failed("BUILD_MESH", "HTTP timeout")
        self.assertEqual([ServerTaskType.STATUS], self.task_types())
        self.window._on_server_status({
            "session_id": "retained", "stored_count": 3, "frame_count": 3,
            "has_mesh": False, "settings": {}, "operation": "build",
        })
        self.assertFalse(self.window.btn_stop_build.isEnabled())
        self.window._on_server_status({
            "session_id": "retained", "stored_count": 3, "frame_count": 3,
            "has_mesh": True, "settings": {}, "operation": None,
        })
        self.assertTrue(self.window._has_mesh)
        self.assertFalse(self.window._build_failed)
        self.assertEqual(ServerTaskType.FINAL_PREVIEW, self.task_types()[-1])

    def test_session_save_pauses_capture_and_mesh_export_does_not_clear_dirty(self):
        self.retain_scan()
        self.window._has_mesh = True
        self.protection("save")
        self.window._export_mesh("ply")
        self.window._capture_frame()
        self.assertNotIn(ServerTaskType.SEND_FRAME, self.task_types())
        self.window._on_export_done(True, "/tmp/workflow-session.zip")
        self.assertTrue(self.window._session_dirty)
        self.assertFalse(self.window._paused)

    def test_stale_final_preview_cannot_replace_new_session(self):
        self.window._session_id = "new-session"
        self.window._final_preview_session = "old-session"
        self.window._has_mesh = True
        self.window._on_final_preview_done("/tmp/old-final.ply")
        self.assertIsNone(self.window._last_preview_path)

    def test_shortcuts_cannot_resume_capture_during_server_build(self):
        self.retain_scan()
        self.window._paused = True
        self.window._server_operation = "build"
        self.window._pause_or_resume()
        self.assertTrue(self.window._paused)
        self.window._paused = False
        self.window._capture_frame()
        self.window._auto_capture_tick()
        self.assertEqual([], self.task_types())

    def test_paused_scan_can_be_replaced_with_protection(self):
        self.retain_scan()
        self.window._paused = True
        self.window._refresh_controls()
        self.assertTrue(self.window.btn_start_scan.isEnabled())
        self.protection("cancel")
        self.window._start_scan()
        self.assertEqual([], self.task_types())
        self.assertTrue(self.window._session_dirty)

    def test_discard_cancel_returns_to_setup_without_a_build_or_camera(self):
        self.retain_scan()
        self.window.auto_capture_cb.setChecked(True)
        self.window._last_frame_time = time.monotonic() - 2
        self.window._refresh_controls()
        self.assertTrue(self.window.btn_cancel_scan.isEnabled())
        self.protection("discard")
        self.window.btn_cancel_scan.click()
        self.assertEqual([ServerTaskType.RESET], self.task_types())
        self.assertTrue(self.window._cancel_pending)
        self.assertFalse(self.window.auto_capture_cb.isChecked())
        self.assertEqual(3, self.window._server_stored)  # Clear only on server acknowledgement.
        self.window._on_reset_done({"session_id": "empty", "settings": {"live_reconstruction": True}})
        self.assertFalse(self.window._scanning)
        self.assertIsNone(self.window._session_id)
        self.assertEqual(0, self.window._server_stored)
        self.assertFalse(self.window._session_dirty)
        self.assertTrue(self.window.settings_group.isEnabled())
        self.assertFalse(self.window.btn_stop_build.isEnabled())
        self.assertFalse(self.window.btn_cancel_scan.isEnabled())
        self.fresh_frame()
        self.assertTrue(self.window.btn_start_scan.isEnabled())
        self.assertEqual("Start Scan", self.window.btn_start_scan.text())
        self.window._start_scan()
        self.assertEqual([ServerTaskType.RESET, ServerTaskType.RESET], self.task_types())

    def test_declining_cancel_keeps_the_running_scan(self):
        self.retain_scan()
        self.window.auto_capture_cb.setChecked(True)
        self.protection("cancel")
        self.window._cancel_scan()
        self.assertEqual([], self.task_types())
        self.assertTrue(self.window._scanning)
        self.assertFalse(self.window._paused)
        self.assertTrue(self.window.auto_capture_cb.isChecked())
        self.assertTrue(self.window._session_dirty)

    def test_cancel_after_save_waits_for_success_and_stays_idle(self):
        self.retain_scan()
        self.protection("save")
        self.window._cancel_scan()
        self.assertEqual([ServerTaskType.EXPORT_SESSION], self.task_types())
        self.assertTrue(self.window._paused)
        self.window._on_export_done(True, "/tmp/workflow-session.zip")
        self.assertEqual([ServerTaskType.EXPORT_SESSION, ServerTaskType.RESET], self.task_types())
        self.window._on_reset_done({"session_id": "empty", "settings": {}})
        self.assertFalse(self.window._scanning)
        self.assertFalse(self.window.auto_capture_cb.isChecked())

    def test_failed_save_before_cancel_keeps_captures(self):
        self.retain_scan()
        self.protection("save")
        self.window._cancel_scan()
        self.window._on_export_done(False, "/tmp/workflow-session.zip")
        self.assertEqual([ServerTaskType.EXPORT_SESSION], self.task_types())
        self.assertTrue(self.window._scanning)
        self.assertFalse(self.window._paused)
        self.assertEqual(3, self.window._server_stored)
        self.assertTrue(self.window._session_dirty)

    def test_empty_scan_can_be_cancelled_without_a_prompt(self):
        self.window._scanning = True
        self.window._session_id = "empty-running"
        with patch.object(main_window, "SessionProtectionDialog") as dialog:
            self.window._cancel_scan()
        dialog.assert_not_called()
        self.assertEqual([ServerTaskType.RESET], self.task_types())

    def test_failed_cancel_checks_server_before_clearing_captures(self):
        self.retain_scan()
        self.protection("discard")
        self.window._cancel_scan()
        self.window._on_task_failed("RESET", "HTTP timeout")
        self.assertEqual([ServerTaskType.RESET, ServerTaskType.STATUS], self.task_types())
        self.assertEqual(3, self.window._server_stored)
        self.assertTrue(self.window._paused)
        self.assertFalse(self.window._cancel_pending)
        self.window._on_server_status({"session_id": "retained", "stored_count": 3,
                                      "frame_count": 2, "settings": {}})
        self.assertTrue(self.window._scanning)
        self.assertTrue(self.window._paused)

        self.assertFalse(self.window._status_pending)
        self.window._on_task_failed("BUILD_MESH", "Another timeout")
        self.assertEqual([ServerTaskType.RESET, ServerTaskType.STATUS, ServerTaskType.STATUS], self.task_types())

    def test_cancel_timeout_restores_idle_when_server_reset_completed(self):
        self.retain_scan()
        self.protection("discard")
        self.window._cancel_scan()
        self.window._on_task_failed("RESET", "HTTP timeout")
        self.window._on_server_status({"session_id": "empty", "stored_count": 0,
                                      "frame_count": 0, "settings": {}})
        self.assertFalse(self.window._status_pending)
        self.assertFalse(self.window._restore_on_status)
        self.assertFalse(self.window._scanning)
        self.assertIsNone(self.window._session_id)
        self.assertEqual(0, self.window._server_stored)

    def test_failed_cancel_status_check_retries_without_resuming_capture(self):
        self.retain_scan()
        self.protection("discard")
        self.window._cancel_scan()
        self.window._on_task_failed("RESET", "HTTP timeout")
        with patch.object(main_window.QTimer, "singleShot") as retry:
            self.window._on_task_failed("STATUS", "Server unreachable")
        retry.assert_called_once_with(3000, self.window._poll_server_status)
        self.assertFalse(self.window._status_pending)
        self.assertTrue(self.window._paused)
        self.window._poll_server_status()
        self.assertEqual([ServerTaskType.RESET, ServerTaskType.STATUS, ServerTaskType.STATUS], self.task_types())

    def test_cancel_confirmation_blocks_capture_and_scan_commands(self):
        self.retain_scan()
        self.protection("discard")
        self.window._cancel_scan()
        self.window._on_task_failed("RESET", "HTTP timeout")
        self.window._pause_or_resume()
        self.window._capture_frame()
        self.window._start_scan()
        self.window._cancel_scan(protected=True)
        self.assertTrue(self.window._paused)
        self.assertFalse(self.window.btn_cancel_scan.isEnabled())
        self.assertEqual([ServerTaskType.RESET, ServerTaskType.STATUS], self.task_types())

    def test_cancel_does_not_interrupt_build_inspection_or_export(self):
        self.retain_scan()
        for flag, busy in (("_build_pending", True), ("_preview_pending", True),
                           ("_export_pending", {"kind": "session"}), ("_server_operation", "build")):
            with self.subTest(flag=flag):
                setattr(self.window, flag, busy)
                self.window._cancel_scan(protected=True)
                self.assertEqual([], self.task_types())
                setattr(self.window, flag, False)

    def test_shortcuts_cannot_capture_or_resume_during_disconnect(self):
        self.retain_scan()
        self.window._connect_pending = True
        self.window._capture_frame()
        self.window._auto_capture_tick()
        self.window._paused = True
        self.window._pause_or_resume()
        self.assertTrue(self.window._paused)
        self.assertEqual([], self.task_types())

    def test_retained_session_restores_into_scan_view(self):
        self.window.show()
        self.window._restore_server_session({"session_id": "existing", "stored_count": 3,
                                             "settings": {"live_reconstruction": True}})
        self.app.processEvents()
        self.assertEqual(main_window.MODE_SCANNER, self.window._mode)
        self.assertTrue(self.window.live_view.isVisible())
        self.assertTrue(self.window._mode_actions[0].isChecked())

    def test_depth_preview_uses_exact_restored_crop_during_capture(self):
        self.retain_scan()
        self.window._session_settings = {"roi": [20, 30, 60, 90], "near_m": 0.5, "far_m": 1.5}
        with patch.object(main_window, "raw_depth_to_mm", return_value=np.full((480, 640), 1000.0)):
            preview = self.window._depth_display(self.depth)
        np.testing.assert_array_equal(preview[30, 20], [255, 255, 255])
        np.testing.assert_array_equal(preview[200, 300], [42, 42, 42])
        self.assertFalse(np.array_equal(preview[50, 40], [42, 42, 42]))

    def test_loss_sound_follows_live_tracking_and_ignores_stale_sessions(self):
        self.retain_scan()
        with patch.object(self.window.capture_sound, "set_tracking_lost") as cue:
            self.window._on_live_updated({"session_id": "previous", "fusion_paused": True})
            cue.assert_not_called()
            self.window._on_live_updated({"session_id": "retained", "fusion_paused": True})
            cue.assert_called_once_with(True)
            self.window._on_live_updated({"session_id": "retained", "fusion_paused": False})
            cue.assert_called_with(False)
            cue.reset_mock()
            self.window._on_live_updated({"session_id": "retained"})
            cue.assert_not_called()
            self.window._scanning = False
            with patch.object(self.window.capture_sound, "reset_tracking") as reset:
                self.window._on_live_updated({"session_id": "retained", "fusion_paused": True})
                reset.assert_called_once()
            cue.assert_not_called()

    def test_new_scan_rearms_tracking_loss_sound(self):
        self.retain_scan()
        with patch.object(self.window.capture_sound, "reset_tracking") as reset:
            self.window._on_reset_done({"session_id": "new", "settings": {}})
            reset.assert_called_once()

    def test_finishing_lost_scan_clears_sound_state_without_recovery_cue(self):
        self.retain_scan()
        self.window.capture_sound._tracking_lost = True
        with patch.object(self.window.capture_sound, "_play_recovery") as recovery:
            self.window._stop_and_build()
        recovery.assert_not_called()
        self.assertFalse(self.window.capture_sound._tracking_lost)

    def test_capture_sound_confirms_successful_single_and_batch_uploads(self):
        self.retain_scan()
        with patch.object(self.window.capture_sound, "play") as cue:
            self.window._on_frame_stored({"session_id": "retained", "success": True,
                                          "stored_count": 4, "index": 3})
            cue.assert_called_once()
            self.window._on_frame_stored({"session_id": "retained", "success": True,
                                          "stored_count": 6, "batch_size": 2,
                                          "results": [{"success": True}, {"success": True}]})
            self.assertEqual(2, cue.call_count)
        self.assertEqual(6, self.window._server_stored)

    def test_rejected_stale_or_abandoned_captures_do_not_sound(self):
        self.retain_scan()
        with patch.object(self.window.capture_sound, "play") as cue:
            self.window._on_frame_stored({"session_id": "old-session", "success": True})
            self.window._on_frame_stored({"session_id": "retained", "success": False})
            self.window._reset_pending = True
            self.window._on_frame_stored({"session_id": "retained", "success": True})
            self.window._reset_pending = False
            with patch.object(self.window, "_closing", True):
                self.window._on_frame_stored({"session_id": "retained", "success": True})
            cue.assert_not_called()

    def test_final_preview_replaces_snapshot_without_resuming_capture(self):
        self.window._last_preview_path = "/tmp/old-snapshot.ply"
        self.window._has_mesh = True
        self.window._scanning = False
        with patch.object(main_window, "launch_viewer_subprocess"):
            self.window._on_final_preview_done("/tmp/final.ply")
        self.assertEqual("/tmp/final.ply", self.window._last_preview_path)
        self.assertFalse(self.window._scanning)

    def test_primary_actions_remain_visible_at_minimum_window_size(self):
        self.window.resize(960, 600)
        self.window.show()
        self.app.processEvents()
        for control in (self.window.btn_start_scan, self.window.btn_pause,
                        self.window.btn_stop_build, self.window.btn_preview_scan,
                        self.window.btn_cancel_scan,
                        self.window.btn_export, self.window.btn_export_session):
            self.assertTrue(control.isVisible())
            position = control.mapTo(self.window, QPoint(0, 0))
            self.assertGreaterEqual(position.y(), 0)
            self.assertGreaterEqual(position.x(), 0)
            self.assertLessEqual(position.x() + control.width(), self.window.width())
            self.assertLessEqual(position.y() + control.height(), self.window.height())
            self.assertFalse(self.window.settings_scroll.isAncestorOf(control))


if __name__ == "__main__":
    unittest.main()
