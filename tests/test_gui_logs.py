"""Worker logging must stay bounded and must never touch Qt off the UI thread."""

import logging
import os
import unittest
from threading import Thread

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from kinect_scanner.gui.logs import BufferedLogHandler, LogsPanel


class GuiLogsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = LogsPanel()

    def tearDown(self):
        self.panel.stop()
        self.panel.close()

    def test_worker_message_is_buffered_then_flushed_and_handler_is_removed(self):
        thread = Thread(target=lambda: logging.getLogger("kinect_scanner.worker").warning("USB disconnected"))
        thread.start()
        thread.join()
        self.assertEqual("", self.panel.text.toPlainText())
        self.panel.flush()
        self.assertIn("WARNING · kinect_scanner.worker · USB disconnected", self.panel.text.toPlainText())
        self.panel.stop()
        self.assertNotIn(self.panel.handler, logging.getLogger().handlers)

    def test_pending_and_displayed_history_are_bounded(self):
        handler = BufferedLogHandler(capacity=2)
        for i in range(4):
            handler.handle(logging.LogRecord("camera", logging.WARNING, "", 0, f"frame {i}", (), None))
        messages = handler.drain()
        self.assertEqual(2, len(messages))
        self.assertIn("frame 2", messages[0])
        self.assertIn("frame 3", messages[1])
        self.panel.text.setMaximumBlockCount(2)
        self.panel._append_lines(["first", "second", "third"])
        self.assertEqual("second\nthird", self.panel.text.toPlainText())

    def test_repeated_messages_copy_and_clear(self):
        self.panel.append("Camera unavailable", "Camera")
        self.panel.append("Camera unavailable", "Camera")
        self.assertEqual(1, self.panel.text.document().blockCount())
        self.panel.copy_button.click()
        self.assertEqual(self.panel.text.toPlainText(), QApplication.clipboard().text())
        self.panel.clear_button.click()
        self.assertEqual("", self.panel.text.toPlainText())
        self.panel.append("Camera unavailable", "Camera")
        self.assertEqual("", self.panel.text.toPlainText())
        self.panel.append("Camera connected", "Camera")
        self.panel.append("Camera unavailable", "Camera")
        self.assertIn("Camera unavailable", self.panel.text.toPlainText())

    def test_state_changes_are_independent_of_unrelated_messages(self):
        self.panel.append("USB disconnected", "Camera")
        self.panel.append("Server connected", "Connection")
        self.panel.append("USB disconnected", "Camera")
        self.panel.append("Camera timed out", "Camera")
        self.panel.append("Camera connected", "Camera")
        self.panel.append("USB disconnected", "Camera")
        text = self.panel.text.toPlainText()
        self.assertEqual(2, text.count("USB disconnected"))
        self.assertEqual(1, text.count("Camera timed out"))

    def test_handler_suppresses_samples_mirrors_and_repeated_warnings(self):
        logger = logging.getLogger("kinect_scanner.worker")
        def record(message, level=logging.WARNING, **extra):
            item = logger.makeRecord(logger.name, level, "", 0, message, (), None, extra=extra)
            self.panel.handler.handle(item)
        record("Camera unavailable", ui_state_key="camera")
        record("Resources changed slightly", logging.INFO)
        record("Operation already shown by its GUI callback", ui_log=False)
        record("Sound unavailable")
        record("Camera unavailable", ui_state_key="camera")
        record("Camera connected", logging.INFO, ui_event=True, ui_state_key="camera")
        record("Camera unavailable", ui_state_key="camera")
        self.panel.flush()
        text = self.panel.text.toPlainText()
        self.assertEqual(2, text.count("Camera unavailable"))
        self.assertIn("Camera connected", text)
        self.assertNotIn("Resources", text)
        self.assertNotIn("GUI callback", text)

    def test_new_messages_do_not_scroll_away_from_older_entries(self):
        self.panel.resize(600, 250)
        self.panel.show()
        self.panel._append_lines([f"event {i}" for i in range(100)])
        self.app.processEvents()
        scroll = self.panel.text.verticalScrollBar()
        scroll.setValue(5)
        self.panel.append("New event")
        self.assertEqual(5, scroll.value())
