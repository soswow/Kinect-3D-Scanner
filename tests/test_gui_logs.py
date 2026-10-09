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
            handler.handle(logging.LogRecord("camera", logging.INFO, "", 0, f"frame {i}", (), None))
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
        self.assertIn("Camera unavailable", self.panel.text.toPlainText())

    def test_new_messages_do_not_scroll_away_from_older_entries(self):
        self.panel.resize(600, 250)
        self.panel.show()
        self.panel._append_lines([f"event {i}" for i in range(100)])
        self.app.processEvents()
        scroll = self.panel.text.verticalScrollBar()
        scroll.setValue(5)
        self.panel.append("New event")
        self.assertEqual(5, scroll.value())
