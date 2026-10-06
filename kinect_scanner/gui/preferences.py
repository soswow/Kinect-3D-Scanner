"""Remember user choices independently of settings restored from a server scan."""

import json
import logging
import math
from contextlib import contextmanager

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from shared.sensor_calibration import SensorCalibration

from .widgets import FrameIntervalSpinBox

logger = logging.getLogger(__name__)


class ScannerPreferences:
    def __init__(self, settings=None):
        self.settings = settings if settings is not None else QSettings("Kinect3DScanner", "Scanner")
        self._suspended = 0

    def read(self, key, default):
        value = self.settings.value(f"preferences/{key}")
        if not isinstance(value, str):
            return default
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default

    def write(self, key, value):
        if self._suspended:
            return
        try:
            encoded = json.dumps(value, allow_nan=False)
        except (ValueError, TypeError):
            logger.warning("Ignoring invalid preference: %s", key)
            return
        self.settings.setValue(f"preferences/{key}", encoded)
        # Save each edit, including when the app is interrupted before normal exit.
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            logger.warning("Could not save scanner preference: %s", key)

    @contextmanager
    def suspend(self):
        """Programmatic session restoration must not rewrite user defaults."""
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1

    @staticmethod
    def _binding(widget):
        if isinstance(widget, FrameIntervalSpinBox):
            return (lambda: widget.interval_seconds), widget.set_interval_seconds, widget.valueChanged
        if isinstance(widget, QComboBox):
            return widget.currentData, lambda value: widget.setCurrentIndex(widget.findData(value)), widget.currentIndexChanged
        if isinstance(widget, QCheckBox):
            return widget.isChecked, widget.setChecked, widget.toggled
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value, widget.setValue, widget.valueChanged
        if isinstance(widget, QLineEdit):
            return widget.text, widget.setText, widget.textChanged
        raise TypeError(f"Unsupported preference widget: {type(widget).__name__}")

    @staticmethod
    def _valid(widget, value):
        numeric = type(value) in (int, float)
        if numeric:
            try:
                numeric = math.isfinite(value)
            except OverflowError:
                numeric = False
        if isinstance(widget, FrameIntervalSpinBox):
            return numeric and 0 < value <= 30
        if isinstance(widget, QComboBox):
            return any(type(value) is type(widget.itemData(index))
                       and value == widget.itemData(index) for index in range(widget.count()))
        if isinstance(widget, QCheckBox):
            return type(value) is bool
        if isinstance(widget, QSpinBox):
            return type(value) is int and widget.minimum() <= value <= widget.maximum()
        if isinstance(widget, QDoubleSpinBox):
            return numeric and widget.minimum() <= value <= widget.maximum()
        if isinstance(widget, QLineEdit):
            return isinstance(value, str)
        return False

    def restore(self, widget, key):
        getter, setter, _signal = self._binding(widget)
        value = self.read(key, getter())
        if not self._valid(widget, value):
            return
        previous = widget.blockSignals(True)
        try:
            setter(value)
        finally:
            widget.blockSignals(previous)

    def bind(self, widget, key, *, restore=True):
        if restore:
            self.restore(widget, key)
        getter, _setter, signal = self._binding(widget)
        # Save only the changed control. Other controls may reflect a server scan.
        signal.connect(lambda *_args: self.write(key, getter()))

    def load_calibration(self, default):
        document = self.read("calibration", None)
        if document is None:
            return default
        try:
            return SensorCalibration.from_dict(document)
        except (ValueError, TypeError):
            logger.warning("Ignoring invalid saved sensor calibration")
            return default

    def save_calibration(self, profile):
        # Store the validated profile itself so moving its source file is harmless.
        self.write("calibration", profile.to_dict())
