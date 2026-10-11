"""Review retained local captures without changing the current scan."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)


class CaptureRecoveryDialog(QDialog):
    retry_requested = pyqtSignal(str)
    export_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recover Captures")
        self.resize(760, 390)
        self.rows = []
        self.session_id = None
        self.retry_supported = False
        self.busy = False
        layout = QVBoxLayout(self)
        explanation = QLabel("Captures kept on this computer can be retried with their matching scan, "
                             "or saved to a separate recovery project. Your current scan is preserved.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Scan", "Retained captures", "Recovery"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._refresh)
        layout.addWidget(self.table)
        self.detail = QLabel("Checking retained captures…")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.detail.setAccessibleName("Capture recovery details")
        layout.addWidget(self.detail)
        buttons = QHBoxLayout()
        self.retry_button = QPushButton("Retry Matching Scan")
        self.save_button = QPushButton("Save Recovery Project…")
        self.retry_button.clicked.connect(lambda: self._request(self.retry_requested))
        self.save_button.clicked.connect(lambda: self._request(self.export_requested))
        buttons.addWidget(self.retry_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.close)
        layout.addWidget(close)
        self._refresh()

    def selected(self):
        index = self.table.currentRow()
        return self.rows[index] if 0 <= index < len(self.rows) else None

    def set_rows(self, rows, session_id=None, retry_supported=False):
        selected_path = (self.selected() or {}).get("path")
        self.rows = list(rows)
        self.session_id = session_id
        self.retry_supported = retry_supported
        self.table.setRowCount(len(rows))
        selected_index = 0
        for index, row in enumerate(rows):
            values = [row.get("session_id") or "Unidentified scan", str(row["count"]),
                      "Check local files" if row.get("error") else
                      "Retry available" if row.get("retry_safe") and row.get("session_id") == session_id and retry_supported
                      else "Save a recovery copy"]
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(value))
            if row["path"] == selected_path:
                selected_index = index
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)
        if rows:
            self.table.selectRow(selected_index)
        self._refresh()

    def set_busy(self, busy, message=""):
        self.busy = busy
        self.table.setEnabled(not busy)
        self._refresh()
        if message:
            self.detail.setText(message)

    def _refresh(self):
        row = self.selected()
        can_retry = bool(row and row.get("retry_safe") and self.retry_supported
                         and row.get("session_id") == self.session_id)
        self.retry_button.setEnabled(can_retry and not self.busy)
        self.save_button.setEnabled(bool(row and row["count"] and not row.get("error")) and not self.busy)
        if self.busy:
            return
        if not row:
            self.detail.setText("No retained capture buffers were found.")
        elif row.get("error"):
            self.detail.setText("Some local files need attention: " + row["error"] + "\nLocal buffer: " + row["path"])
        elif can_retry:
            self.detail.setText("Retry will upload only unconfirmed captures to this matching scan. Capture stays paused.")
        else:
            self.detail.setText("Save an offline recovery copy, or connect to the matching scan with an updated server to retry. "
                                "Older capture buffers can be saved but cannot be retried safely.")

    def _request(self, signal):
        row = self.selected()
        if row and not self.busy:
            signal.emit(row["path"])
