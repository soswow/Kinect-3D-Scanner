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
        self._last_states = {}
        self.setFormatter(logging.Formatter(
            "%(asctime)s · %(levelname)s · %(name)s · %(message)s", datefmt="%H:%M:%S"
        ))

    def emit(self, record):
        # Routine samples and detailed timings belong in the diagnostic file.
        # GUI callbacks already supply readable versions of mirrored events.
        if getattr(record, "ui_log", True) is False:
            return
        if record.levelno < logging.WARNING and not getattr(record, "ui_event", False):
            return
        try:
            message = self.format(record)
            topic = getattr(record, "ui_state_key", (record.name, str(record.msg)))
            state = (record.levelno, record.getMessage(),
                     self.formatter.formatException(record.exc_info) if record.exc_info else None)
            with self.pending_lock:
                if self._last_states.get(topic) == state:
                    return
                self._last_states[topic] = state
                if len(self._last_states) > self.pending.maxlen:
                    del self._last_states[next(iter(self._last_states))]
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
        self._last_messages = {}

    def append(self, message, source="Client", *, key=None):
        topic = key if key is not None else source
        if not message or (source, message) == self._last_messages.get(topic):
            return
        self._last_messages[topic] = (source, message)
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
        # Clearing history does not change camera/connection state or re-arm retries.

    def stop(self):
        self.timer.stop()
        logging.getLogger().removeHandler(self.handler)
        self.flush()
        self.handler.close()
