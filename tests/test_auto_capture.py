"""Auto-capture must select fresh camera frames at a whole-frame cadence."""

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from kinect_scanner.gui import main_window
from kinect_scanner.gui.preferences import ScannerPreferences
from kinect_scanner.server_task_worker import ServerTaskType
from shared.settings import ScanSettings


class NoCamera(QThread):
    frame_pair_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__()
        self.tracking_settings = []

    def run(self):
        pass

    def stop(self):
        pass

    def set_tracking_settings(self, settings):
        self.tracking_settings.append(settings)


class NoTasks(QThread):
    queued_task_count = 0

    def __init__(self, client):
        super().__init__()
        self.frames = []
        self.tasks = []
        self.accept = True

    def run(self):
        pass

    def stop(self):
        pass

    def submit(self, task):
        self.tasks.append(task)
        if self.accept and task.task_type == ServerTaskType.SEND_FRAME:
            self.frames.append(task.kwargs)
        return self.accept


class AutoCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.now = 100.0
        self.clock_patch = patch.object(main_window.time, "monotonic", side_effect=lambda: self.now)
        self.clock_patch.start()
        self.camera_patch = patch.object(main_window, "KinectWorker", NoCamera)
        self.tasks_patch = patch.object(main_window, "ServerTaskWorker", NoTasks)
        self.camera_patch.start()
        self.tasks_patch.start()
        self.preferences_dir = tempfile.TemporaryDirectory()
        store = QSettings(os.path.join(self.preferences_dir.name, "scanner.ini"), QSettings.Format.IniFormat)
        self.window = main_window.MainWindow(preferences=ScannerPreferences(store))
        self.window.server_client._connected = True
        self.window._scanning = True
        self.window.auto_capture_cb.setEnabled(True)
        self.window.auto_capture_spin.setEnabled(True)
        self.window.live_cb.setChecked(False)
        self.rgb = np.full((2, 2, 3), 42, np.uint8)
        self.depth = np.full((2, 2), 750, np.uint16)

    def tearDown(self):
        self.window._close_approved = True
        self.window.close()
        self.window.worker.wait(2500)
        self.window.task_worker.wait(2500)
        self.app.processEvents()
        self.camera_patch.stop()
        self.tasks_patch.stop()
        self.clock_patch.stop()
        self.preferences_dir.cleanup()

    def receive(self, count=1, **metadata):
        for _ in range(count):
            self.now += 1 / main_window.RGB_MODE_FPS[self.window.rgb_mode_combo.currentData()]
            self.window._on_frame(self.rgb, self.depth, metadata)

    def ids(self):
        return [int(f["metadata"]["frame_id"].rsplit(":", 1)[-1]) for f in self.window.task_worker.frames]

    def test_session_tracking_uses_frozen_settings_and_survives_status_updates(self):
        self.window.rgb_mode_combo.setCurrentIndex(1)
        profile = ScanSettings(color_recovery=True, live_reconstruction=True)
        self.window._on_reset_done({"session_id": "motion-trial", "settings": profile.to_dict()})
        self.assertEqual(profile, self.window.worker.tracking_settings[-1])
        requests = len(self.window.worker.tracking_settings)
        self.window._apply_session_settings(profile.to_dict())
        self.assertEqual(requests, len(self.window.worker.tracking_settings),
                         "A periodic status update must not reset the motion chain")
        self.window._on_build_mesh_done(False, "Synthetic failed build")
        self.assertIsNone(self.window.worker.tracking_settings[-1])
        self.window._restore_server_session({"session_id": "motion-trial", "settings": profile.to_dict(),
                                            "stored_count": 2, "frame_count": 1, "has_mesh": False})
        self.assertEqual(profile, self.window.worker.tracking_settings[-1])
        self.window._cancel_pending = True
        self.window._on_reset_done({"session_id": "empty", "settings": profile.to_dict()})
        self.assertIsNone(self.window.worker.tracking_settings[-1])

    def test_color_tracking_is_retained_when_live_reconstruction_is_off(self):
        self.window.rgb_mode_combo.setCurrentIndex(1)
        profile = ScanSettings(color_recovery=True, live_reconstruction=False)
        self.window._on_reset_done({"session_id": "offline-motion", "settings": profile.to_dict()})
        self.assertEqual(profile, self.window.worker.tracking_settings[-1])
        requests = len(self.window.worker.tracking_settings)
        self.window._apply_session_settings(profile.to_dict())
        self.assertEqual(requests, len(self.window.worker.tracking_settings),
                         "Status updates must preserve offline motion tracking")
        self.receive()
        self.window._refresh_status()
        self.assertIn("reconstruction checked at Finish", self.window.scan_status_label.text())
        self.assertTrue(self.window.live_view.isHidden())
        self.window._on_build_mesh_done(False, "Synthetic failed build")
        self.assertIsNone(self.window.worker.tracking_settings[-1])

    def test_disabled_color_tracking_remains_off_without_live_reconstruction(self):
        profile = ScanSettings(color_recovery=False, live_reconstruction=False)
        self.window._on_reset_done({"session_id": "offline-no-motion", "settings": profile.to_dict()})
        self.assertIsNone(self.window.worker.tracking_settings[-1])

    def test_offline_motion_warning_is_logged_without_moving_camera_views(self):
        profile = ScanSettings(color_recovery=True, live_reconstruction=False)
        self.window._on_reset_done({"session_id": "offline-warning", "settings": profile.to_dict()})
        self.window._progress_link_ok = True
        self.window.resize(1200, 800)
        self.window.show()
        self.receive(visual_tracking={"valid": True})
        self.app.processEvents()
        views = (self.window.view_stack, self.window.view_label, self.window.scan_depth_view)
        geometry = [view.geometry() for view in views]
        state = self.window.scan_status_label.text()
        for valid in (False, False, True, False, True):
            self.receive(visual_tracking={"valid": valid, "reason": "Visual motion unverified"})
            self.app.processEvents()
            self.assertTrue(self.window.guidance_label.isHidden())
            self.assertEqual(geometry, [view.geometry() for view in views])
            self.assertEqual(state, self.window.scan_status_label.text())
            self.assertFalse(self.window._paused)
            self.assertTrue(self.window.auto_capture_cb.isChecked())
        history = self.window.logs_panel.text.toPlainText()
        self.assertEqual(2, history.count("Camera motion could not be verified"))
        self.assertEqual(2, history.count("Camera motion warning cleared"))
        self.receive(visual_tracking={"valid": True})
        self.assertEqual(history, self.window.logs_panel.text.toPlainText())

    def test_interval_editor_caps_frequency_and_steps_whole_frames(self):
        spin = self.window.auto_capture_spin
        self.assertEqual(5, spin.value())
        self.assertEqual(0.5, spin.interval_seconds)
        spin.set_interval_seconds(0.03)
        self.assertEqual(0.1, spin.interval_seconds)
        spin.stepUp()
        self.assertEqual(0.2, spin.interval_seconds)
        spin.lineEdit().setText("0.260 s")
        spin.interpretText()
        self.assertEqual(3, spin.value())
        self.assertEqual("0.3 s", spin.text())
        self.window.rgb_mode_combo.setCurrentIndex(1)
        self.assertEqual(9, spin.value())  # Preserve the nominal .3 s interval.
        spin.setValue(1)
        self.assertEqual(1 / 30, spin.interval_seconds)
        self.assertEqual(1, spin.valueFromText(spin.text()))
        spin.stepUp()
        self.assertEqual(2 / 30, spin.interval_seconds)
        self.window.rgb_mode_combo.setCurrentIndex(0)
        self.assertEqual(0.1, spin.interval_seconds)

    def test_auto_capture_waits_for_new_frames_and_never_repeats_cached_pair(self):
        self.window.auto_capture_spin.setValue(1)
        self.receive()  # A cached frame before auto-capture starts.
        self.window.auto_capture_cb.setChecked(True)
        QTest.qWait(150)  # Longer than the nominal RGB period, with no new input.
        self.assertEqual([], self.ids())
        self.receive()
        self.assertEqual([2], self.ids())
        self.window._capture_frame()
        self.assertEqual([2], self.ids())
        QTest.qWait(150)
        self.assertEqual([2], self.ids())

    def test_capture_every_five_arrivals_and_apply_interval_edit_immediately(self):
        self.window.auto_capture_cb.setChecked(True)
        self.receive(4)
        self.assertEqual([], self.ids())
        self.receive(6)
        self.assertEqual([5, 10], self.ids())
        self.receive(3)
        self.window.auto_capture_spin.set_interval_seconds(0.2)
        self.receive()
        self.assertEqual([5, 10], self.ids())
        self.receive()
        self.assertEqual([5, 10, 15], self.ids())

    def test_manual_capture_restarts_auto_cadence(self):
        self.window.auto_capture_cb.setChecked(True)
        self.receive(2)
        self.window._capture_frame()
        self.receive(4)
        self.assertEqual([2], self.ids())
        self.receive()
        self.assertEqual([2, 7], self.ids())

    def test_offline_motion_adds_an_overlap_capture_before_the_nominal_cadence(self):
        profile = ScanSettings(color_recovery=True, live_reconstruction=False, offline_registration="depth")
        self.window._on_reset_done({"session_id": "overlap", "settings": profile.to_dict()})
        self.window.auto_capture_spin.set_interval_seconds(1.)
        self.window.auto_capture_cb.setChecked(True)
        visual = {"valid": True, "segment": "camera", "camera_to_local": np.eye(4).tolist()}
        self.receive(visual_tracking=visual)
        self.window._capture_frame()
        moved = np.eye(4); moved[0, 3] = .12
        self.receive(3, visual_tracking={**visual, "camera_to_local": moved.tolist()})
        self.assertEqual([1, 3], self.ids())

    def test_recovery_uses_next_arrival_without_waiting_for_normal_interval(self):
        self.window.live_cb.setChecked(True)
        self.window.auto_capture_spin.set_interval_seconds(2)
        self.window.auto_capture_cb.setChecked(True)
        self.window.live_view.snapshot = {"fusion_paused": True, "processed_count": 0}
        self.receive()
        self.assertEqual([1], self.ids())
        self.receive(10)
        self.assertEqual([1], self.ids(), "Only one recovery probe may be outstanding")
        self.window.live_view.snapshot = {"fusion_paused": True, "processed_count": 1,
            "result": {"metadata": self.window.task_worker.frames[0]["metadata"]}}
        self.receive()
        self.assertEqual([1, 12], self.ids())

    def test_backpressure_waits_and_resumes_once_without_catch_up_burst(self):
        self.window.live_cb.setChecked(True)
        self.window.auto_capture_spin.setValue(2)
        self.window.auto_capture_cb.setChecked(True)
        self.window.adaptive_capture_cb.setChecked(True)
        self.window.live_view.snapshot = {"pending_count": 5}
        self.receive(20)
        self.assertEqual([], self.ids())
        self.window.live_view.snapshot = {"pending_count": 0}
        self.receive()
        self.assertEqual([21], self.ids())
        self.receive()
        self.assertEqual([21], self.ids())
        self.receive()
        self.assertEqual([21, 23], self.ids())

    def test_offline_capture_keeps_half_second_cadence_with_upload_backlog(self):
        self.window._session_id = "offline"
        self.window.auto_capture_spin.set_interval_seconds(.5)
        self.window.adaptive_capture_cb.setChecked(True)
        self.window.auto_capture_cb.setChecked(True)
        self.window.task_worker.queued_task_count = 120
        with patch.object(self.window.capture_sound, "play") as cue:
            self.receive(20)
        self.assertEqual([5, 10, 15, 20], self.ids())
        self.assertEqual(4, cue.call_count)
        self.assertNotIn("waiting for uploads", self.window.scan_status_label.text())

    def test_finish_includes_first_capture_currently_in_upload(self):
        self.window._server_stored = 0
        self.window.task_worker.pending_capture_count = 1
        self.window.task_worker.queued_task_count = 0
        self.window._refresh_controls()
        self.assertTrue(self.window.btn_stop_build.isEnabled())
        self.window._stop_and_build()
        self.assertEqual(ServerTaskType.BUILD_MESH, self.window.task_worker.tasks[-1].task_type)
        self.assertIn("Uploading remaining captures", self.window.scan_status_label.text())

    def test_slow_processing_changes_pace_without_changing_minimum(self):
        self.window.live_cb.setChecked(True)
        self.window._session_id = "paced"
        self.window.auto_capture_spin.set_interval_seconds(1)
        self.window._on_live_updated({
            "session_id": "paced", "processed_count": 1,
            "processing_interval_s": 1.6, "pending_count": 0,
        })
        self.window.auto_capture_cb.setChecked(True)
        self.receive(10)
        self.assertEqual([10], self.ids())
        self.receive(18)
        self.assertEqual([10], self.ids())
        self.receive()
        self.assertEqual([10, 29], self.ids())
        self.assertEqual(1, self.window.auto_capture_spin.interval_seconds)
        self.assertIn("~1.9 s", self.window.interval_help.text())
        self.assertIn("adjusted for live reconstruction", self.window.interval_help.text())

    def test_delayed_feedback_and_uploads_allow_only_two_outstanding_captures(self):
        self.window.live_cb.setChecked(True)
        self.window._session_id = "paced"
        self.window.auto_capture_spin.set_interval_seconds(1)
        self.window.auto_capture_cb.setChecked(True)
        self.receive(100)
        self.assertEqual([10, 20], self.ids())
        # Neither queued_task_count nor the cached server snapshot sees these uploads.
        self.assertEqual(0, self.window.task_worker.queued_task_count)
        self.assertEqual(2, self.window._capture_pacer.pending_count)
        self.assertIn("paced by live reconstruction", self.window.scan_status_label.text())
        self.window._on_frame_stored({"session_id": "paced", "success": True,
            "stored_count": 2, "capture_acknowledgements": [
                {"frame_id": f["metadata"]["frame_id"], "success": True, "index": i}
                for i, f in enumerate(self.window.task_worker.frames)
            ]})
        self.receive(10)
        self.assertEqual([10, 20], self.ids())  # Stored still means awaiting processing.
        last_id = self.window.task_worker.frames[-1]["metadata"]["frame_id"]
        self.window._on_live_updated({"session_id": "paced", "processed_count": 2,
            "pending_count": 0, "result": {"metadata": {"frame_id": last_id}}})
        self.receive(100)
        self.assertEqual(3, len(self.ids()))  # Resume without a catch-up burst.

    def test_camera_burst_cannot_violate_wall_clock_minimum(self):
        self.window.auto_capture_spin.set_interval_seconds(1)
        self.window.auto_capture_cb.setChecked(True)
        self.receive(10)
        for _ in range(100):
            self.window._on_frame(self.rgb, self.depth, {})
        self.assertEqual([10], self.ids())
        self.receive(10)
        self.assertEqual([10, 120], self.ids())

    def test_new_session_resets_learned_pace_and_pending_uploads(self):
        self.window.live_cb.setChecked(True)
        self.window._capture_pacer.observe({"processed_count": 1, "processing_interval_s": 5}, self.now)
        self.receive()
        self.window._capture_frame()
        self.window._on_reset_done({"session_id": "new", "settings": {"live_reconstruction": True}})
        self.assertEqual(0, self.window._capture_pacer.pending_count)
        self.assertEqual(self.window.auto_capture_spin.interval_seconds, self.window._effective_capture_interval())

    def test_preview_progress_drains_slots_even_with_a_stale_live_snapshot(self):
        self.window.live_cb.setChecked(True)
        self.window._session_id = "paced"
        self.window.auto_capture_spin.set_interval_seconds(1)
        self.window.auto_capture_cb.setChecked(True)
        self.receive(20)
        self.window.live_view.snapshot = {
            "stored_count": 2, "processed_count": 0, "pending_count": 2,
        }
        self.window._preview_pending = True
        self.now += 30
        for index, frame in enumerate(self.window.task_worker.frames):
            self.window._on_process_progress(index + 1, 2, {
                "session_id": "paced", "index": index, "elapsed_ms": 100,
                "metadata": frame["metadata"],
            })
        self.assertEqual(0, self.window._capture_pacer.pending_count)
        self.assertEqual(1, self.window._effective_capture_interval())
        self.window._preview_pending = False
        self.receive(10)
        self.assertEqual([10, 20, 30], self.ids())

    def test_preview_pause_stale_frames_and_rejected_queue_never_get_uploaded(self):
        self.window.auto_capture_spin.setValue(1)
        self.window.auto_capture_cb.setChecked(True)
        self.window._preview_pending = True
        self.receive(3)
        self.assertEqual([], self.ids())
        self.window._resume_capture()
        self.receive(captured_monotonic_s=0)
        self.assertEqual([], self.ids())
        self.window.task_worker.accept = False
        self.receive()
        self.assertIsNone(self.window._last_capture_id)
        self.assertTrue(self.window._paused)
        self.window.task_worker.accept = True
        self.window._paused = False  # User resumes after resolving the buffer failure.
        self.receive()
        self.assertEqual([6], self.ids())


if __name__ == "__main__":
    unittest.main()
