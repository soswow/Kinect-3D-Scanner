"""Auto-capture must select fresh camera frames at a whole-frame cadence."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from kinect_scanner.gui import main_window
from kinect_scanner.server_task_worker import ServerTaskType


class NoCamera(QThread):
    frame_pair_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__()

    def run(self):
        pass

    def stop(self):
        pass


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
        self.camera_patch = patch.object(main_window, "KinectWorker", NoCamera)
        self.tasks_patch = patch.object(main_window, "ServerTaskWorker", NoTasks)
        self.camera_patch.start()
        self.tasks_patch.start()
        self.window = main_window.MainWindow()
        self.window.server_client._connected = True
        self.window._scanning = True
        self.window.auto_capture_cb.setEnabled(True)
        self.window.auto_capture_spin.setEnabled(True)
        self.rgb = np.full((2, 2, 3), 42, np.uint8)
        self.depth = np.full((2, 2), 750, np.uint16)

    def tearDown(self):
        self.window.close()
        self.window.worker.wait(2500)
        self.window.task_worker.wait(2500)
        self.app.processEvents()
        self.camera_patch.stop()
        self.tasks_patch.stop()

    def receive(self, count=1, **metadata):
        for _ in range(count):
            self.window._on_frame(self.rgb, self.depth, metadata)

    def ids(self):
        return [f["metadata"]["frame_id"] for f in self.window.task_worker.frames]

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

    def test_backpressure_waits_and_resumes_once_without_catch_up_burst(self):
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
        self.window.task_worker.accept = True
        self.receive()
        self.assertEqual([6], self.ids())


if __name__ == "__main__":
    unittest.main()
