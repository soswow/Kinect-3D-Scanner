"""Reusable widgets and utility functions for the scanner GUI."""

import math

import cv2
import numpy as np
from PyQt6.QtGui import QImage, QValidator
from PyQt6.QtWidgets import QSpinBox


def colorize_depth(depth: np.ndarray, near: float, far: float, roi=None) -> np.ndarray:
    """Color included distances; black means missing, gray means excluded.

    ROI coordinates are (left, top, right, bottom), with exclusive end pixels.
    """
    d = depth.astype(np.float32)
    present = np.isfinite(d) & (d > 0)
    valid = present & (d >= near) & (d <= far)
    if roi is not None:
        x0, y0, x1, y1 = roi
        inside = np.zeros_like(valid)
        inside[y0:y1, x0:x1] = True
        valid &= inside
    norm = np.zeros_like(d, dtype=np.uint8)
    if valid.any():
        norm[valid] = (
            (255.0 * (d[valid] - near) / max(far - near, 1))
            .clip(0, 255)
            .astype(np.uint8)
        )
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[~valid] = [42, 42, 42]
    colored[~present] = 0
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if roi is not None:
        cv2.rectangle(colored, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
    return colored


def depth_legend_text(near: float, far: float) -> str:
    """Accessible labels for the depth preview palette and exclusion states."""
    return (f"Blue: {near:g} mm · red: {far:g} mm\n"
            "Black: no depth · gray: excluded · white outline: crop")


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert an RGB numpy array to a QImage."""
    h, w, ch = arr.shape
    return QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGB888)


class FrameIntervalSpinBox(QSpinBox):
    """Edit seconds while storing an exact integer number of RGB frames."""

    def __init__(self, fps=10, parent=None):
        self._fps = fps
        super().__init__(parent)
        self.setRange(1, 30 * fps)
        self.setSuffix(" s")
        self.setKeyboardTracking(False)
        self.set_interval_seconds(0.5)
        self.valueChanged.connect(self._update_tooltip)
        self._update_tooltip()

    @property
    def interval_seconds(self):
        return self.value() / self._fps

    def set_interval_seconds(self, seconds):
        if not math.isfinite(seconds):
            raise ValueError("Capture interval must be finite")
        self.setValue(max(1, math.floor(seconds * self._fps + 0.5)))

    def set_fps(self, fps):
        seconds = self.interval_seconds
        self._fps = fps
        self.setRange(1, 30 * fps)
        self.set_interval_seconds(seconds)
        # Refresh text even if the frame count stayed the same.
        self.lineEdit().setText(self.textFromValue(self.value()) + self.suffix())
        self._update_tooltip()

    def textFromValue(self, frames):
        return self.locale().toString(frames / self._fps, "f", 3).rstrip("0").rstrip(
            self.locale().decimalPoint()
        )

    def valueFromText(self, text):
        seconds, valid = self.locale().toDouble(
            text.removesuffix(self.suffix()).strip()
        )
        if not valid or not math.isfinite(seconds):
            return self.value()
        return max(1, math.floor(seconds * self._fps + 0.5))

    def validate(self, text, position):
        clean = text.removesuffix(self.suffix()).strip()
        seconds, valid = self.locale().toDouble(clean)
        if not clean or clean == self.locale().decimalPoint():
            state = QValidator.State.Intermediate
        elif valid and 0 <= seconds <= 30:
            state = QValidator.State.Acceptable
        else:
            state = QValidator.State.Invalid
        return state, text, position

    def _update_tooltip(self):
        self.setToolTip(
            f"Capture every {self.value()} fresh RGB/depth frames "
            f"({self._fps} fps nominal). Intervals round to whole frames. "
            "Capture waits when frames or reconstruction are delayed."
        )
