"""Nonblocking capture feedback with a remembered sound preference."""

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, QUrl
from PyQt6.QtMultimedia import QSoundEffect

logger = logging.getLogger(__name__)


class CaptureSound(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = QSettings("Kinect3DScanner", "Scanner")
        self.enabled = self._settings.value("feedback/capture_sound", True, type=bool)
        self._effect = None
        self._pending = False

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self._settings.setValue("feedback/capture_sound", self.enabled)
        if not self.enabled:
            self.stop()

    def play(self):
        if not self.enabled:
            return
        if self._effect is None:
            self._effect = QSoundEffect(self)
            self._effect.setVolume(0.5)
            self._effect.setLoopCount(1)
            self._effect.statusChanged.connect(self._on_status_changed)
            source = Path(__file__).with_name("assets") / "capture.wav"
            self._effect.setSource(QUrl.fromLocalFile(str(source)))
        if self._effect.status() == QSoundEffect.Status.Ready:
            self._pending = False
            if not self._effect.isPlaying():
                self._effect.play()
        elif self._effect.status() == QSoundEffect.Status.Loading:
            self._pending = True

    def _on_status_changed(self):
        if self._effect.status() == QSoundEffect.Status.Ready and self._pending:
            self.play()
        elif self._effect.status() == QSoundEffect.Status.Error:
            self._pending = False
            logger.warning("Capture sound unavailable; check audio output and capture.wav")

    def stop(self):
        self._pending = False
        if self._effect is not None:
            self._effect.stop()
