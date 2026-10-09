"""Bounded client event history, including messages from background workers."""

import logging
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class BufferedLogHandler(logging.Handler):
    def __init__(self, capacity=2000):
        super().__init__(logging.INFO)
        self.pending = deque(maxlen=capacity)
        self.pending_lock = Lock()
        self.setFormatter(logging.Formatter(
            "%(asctime)s · %(levelname)s · %(name)s · %(message)s", datefmt="%H:%M:%S"
        ))

    def emit(self, record):
        try:
            message = self.format(record)
            with self.pending_lock:
                self.pending.append(message)
        except Exception:  # noqa: BLE001 — logging must not interrupt acquisition.
            self.handleError(record)

    def drain(self):
        with self.pending_lock:
            messages = list(self.pending)
            self.pending.clear()
        return messages


class LogsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addStretch()
        self.copy_button = QPushButton("Copy Logs")
        self.clear_button = QPushButton("Clear Logs")
        row.addWidget(self.copy_button)
        row.addWidget(self.clear_button)
        layout.addLayout(row)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(2000)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.text.setAccessibleName("Ongoing client logs")
        self.text.setPlaceholderText("Connection, capture, reconstruction and save events appear here.")
        layout.addWidget(self.text)
        self.copy_button.clicked.connect(self.copy_logs)
        self.clear_button.clicked.connect(self.clear_logs)
        self.handler = BufferedLogHandler()
        logging.getLogger().addHandler(self.handler)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.flush)
        self.timer.start(200)
        self._last_message = None

    def append(self, message, source="Client"):
        if not message or (source, message) == self._last_message:
            return
        self._last_message = (source, message)
        self._append_lines([f"{datetime.now(timezone.utc).astimezone():%H:%M:%S} · {source} · {message}"])

    def _append_lines(self, lines):
        if not lines:
            return
        scroll = self.text.verticalScrollBar()
        follow = scroll.value() >= scroll.maximum()
        position = scroll.value()
        self.text.appendPlainText("\n".join(lines))
        scroll.setValue(scroll.maximum() if follow else position)

    def flush(self):
        self._append_lines(self.handler.drain())

    def copy_logs(self):
        from PyQt6.QtWidgets import QApplication
        self.flush()
        QApplication.clipboard().setText(self.text.toPlainText())

    def clear_logs(self):
        self.handler.drain()
        self.text.clear()
        self._last_message = None

    def stop(self):
        self.timer.stop()
        logging.getLogger().removeHandler(self.handler)
        self.flush()
        self.handler.close()
