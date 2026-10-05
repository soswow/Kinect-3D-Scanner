"""Small workflow dialogs; file saving remains the caller's responsibility."""

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)


class ExportDialog(QDialog):
    """Choose a mesh format and relevant texture options."""

    _FORMATS = (
        ("GLB — textured", "glb", "A textured model in one self-contained file."),
        (
            "OBJ bundle — textured ZIP",
            "obj.zip",
            "A ZIP containing the OBJ model, MTL material file, and texture image. Keep the files together.",
        ),
        ("PLY — vertex colors", "ply", "Geometry with colors stored on its vertices."),
        (
            "OBJ — vertex colors",
            "obj",
            "Geometry using a vertex-color extension. Some 3D applications do not support these colors.",
        ),
    )

    def __init__(self, parent=None, *, preferences=None):
        super().__init__(parent)
        self.setWindowTitle("Export mesh")
        self.setMinimumWidth(390)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("File format"))
        self.format_combo = QComboBox()
        for label, fmt, _description in self._FORMATS:
            self.format_combo.addItem(label, fmt)
        layout.addWidget(self.format_combo)
        self.format_description = QLabel()
        self.format_description.setWordWrap(True)
        layout.addWidget(self.format_description)
        self.texture_exposure_cb = QCheckBox("Match texture exposures")
        self.texture_exposure_cb.setToolTip(
            "Reduce brightness differences between source photos when the correction improves the texture."
        )
        self.texture_best_cb = QCheckBox("Pick one source photo per surface area")
        self.texture_best_cb.setToolTip(
            "Can keep details sharper, but may make seams between photos more visible."
        )
        layout.addWidget(self.texture_exposure_cb)
        layout.addWidget(self.texture_best_cb)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.export_button = self.buttons.addButton(
            "Export", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.export_button.setDefault(True)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        if preferences is not None:
            preferences.restore(self.format_combo, "export/format")
            preferences.restore(self.texture_exposure_cb, "export/exposure_correction")
            preferences.restore(self.texture_best_cb, "export/best_source")
        self.format_combo.currentIndexChanged.connect(self._update_format)
        self._update_format()

    @property
    def selected_format(self):
        return self.format_combo.currentData()

    @property
    def texture_options(self):
        return {
            "exposure_correction": self.texture_exposure_cb.isChecked(),
            "blend_mode": "best" if self.texture_best_cb.isChecked() else "blend",
        }

    def _update_format(self, _index=None):
        self.format_description.setText(
            self._FORMATS[self.format_combo.currentIndex()][2]
        )
        textured = self.selected_format in ("glb", "obj.zip")
        self.texture_exposure_cb.setEnabled(textured)
        self.texture_best_cb.setEnabled(textured)


class SessionProtectionDialog(QDialog):
    """Ask how to protect captured frames before replacing or leaving a scan."""

    def __init__(self, reason, parent=None):
        super().__init__(parent)
        self.choice = "cancel"
        self.setWindowTitle("Save current scan?")
        self.setMinimumWidth(410)
        layout = QVBoxLayout(self)
        closing = reason in ("close", "closing", "quit", "exit")
        action = "closing the scanner" if closing else (
            "cancelling the scan" if reason == "cancel_scan" else "starting a new scan"
        )
        self.message_label = QLabel(
            f"Save the current scan before {action}?\n\n"
            "Save session preserves the captured RGB-D frames so you can rebuild later. "
            "Discard continues without saving a session."
        )
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        self.buttons = QDialogButtonBox()
        self.save_button = self.buttons.addButton(
            "Save session", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.discard_button = self.buttons.addButton(
            "Discard", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self.cancel_button = self.buttons.addButton(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.save_button.setDefault(True)
        self.discard_button.setAutoDefault(False)
        self.cancel_button.setAutoDefault(False)
        self.save_button.clicked.connect(lambda: self._choose("save"))
        self.discard_button.clicked.connect(lambda: self._choose("discard"))
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.buttons)

    def _choose(self, choice):
        self.choice = choice
        self.accept()

    def reject(self):
        self.choice = "cancel"
        super().reject()
