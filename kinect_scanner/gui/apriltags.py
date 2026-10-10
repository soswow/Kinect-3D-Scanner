"""Dictionary selection for simultaneous AprilTag families."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from shared.settings import APRILTAG_DICTIONARIES, validate_apriltag_dictionaries


class AprilTagDictionaries(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for name in APRILTAG_DICTIONARIES:
            self.combo.addItem(name.removeprefix("DICT_APRILTAG_"), name)
        label = QLabel("AprilTag dictionary")
        label.setBuddy(self.combo)
        layout.addWidget(label)
        row = QHBoxLayout()
        row.addWidget(self.combo, 1)
        self.add_button = QPushButton("Add")
        row.addWidget(self.add_button)
        layout.addLayout(row)
        self.list = QListWidget()
        self.list.setAccessibleName("Active AprilTag dictionaries")
        self.list.setMaximumHeight(110)
        layout.addWidget(self.list)
        self.remove_button = QPushButton("Remove selected")
        layout.addWidget(self.remove_button)
        self.combo.currentIndexChanged.connect(self._refresh)
        self.list.currentRowChanged.connect(self._refresh)
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)
        self.set_dictionaries(("DICT_APRILTAG_36h11",))

    def dictionaries(self):
        return tuple(self.list.item(i).text() for i in range(self.list.count()))

    def set_dictionaries(self, names):
        names = validate_apriltag_dictionaries(names)
        self.list.clear()
        self.list.addItems(names)
        self._refresh()

    def _refresh(self, *_):
        self.add_button.setEnabled(self.combo.currentData() not in self.dictionaries())
        self.remove_button.setEnabled(self.list.currentRow() >= 0)

    def _add(self):
        name = self.combo.currentData()
        if name not in self.dictionaries():
            self.list.addItem(name)
            self.list.setCurrentRow(self.list.count() - 1)
            self._refresh()
            self.changed.emit()

    def _remove(self):
        if self.list.currentRow() >= 0:
            self.list.takeItem(self.list.currentRow())
            self._refresh()
            self.changed.emit()
