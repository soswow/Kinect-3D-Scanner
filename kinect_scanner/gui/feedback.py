"""Nonblocking capture and tracking-loss feedback with a remembered mute."""

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, QUrl
from PyQt6.QtMultimedia import QSoundEffect

logger = logging.getLogger(__name__)


class CaptureSound(QObject):
    def __init__(self, parent=None, *, settings=None):
        super().__init__(parent)
        self._settings = settings if settings is not None else QSettings("Kinect3DScanner", "Scanner")
        self.enabled = self._settings.value("feedback/capture_sound", True, type=bool)
        self._effect = None
        self._pending = False
        self._loss_effect = None
        self._loss_pending = False
        self._tracking_lost = False

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self._settings.setValue("feedback/capture_sound", self.enabled)
        self._settings.sync()
        if not self.enabled:
            self.stop()

    def play(self):
        if not self.enabled or self._tracking_lost:
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

    def set_tracking_lost(self, lost):
        """Alert once per loss episode; recovery rearms the next alert."""
        lost = bool(lost)
        if lost == self._tracking_lost:
            return
        self._tracking_lost = lost
        if not lost:
            self._loss_pending = False
            if self._loss_effect is not None:
                self._loss_effect.stop()
            return
        # The warning takes priority over capture confirmations and probes.
        self._pending = False
        if self._effect is not None:
            self._effect.stop()
        if not self.enabled:
            return
        self._loss_pending = True
        if self._loss_effect is None:
            self._loss_effect = QSoundEffect(self)
            self._loss_effect.setVolume(0.7)
            self._loss_effect.setLoopCount(1)
            self._loss_effect.statusChanged.connect(self._on_loss_status_changed)
            source = Path(__file__).with_name("assets") / "tracking_lost.wav"
            self._loss_effect.setSource(QUrl.fromLocalFile(str(source)))
        self._on_loss_status_changed()

    def _on_loss_status_changed(self):
        status = self._loss_effect.status()
        if status == QSoundEffect.Status.Ready and self._loss_pending:
            self._loss_pending = False
            if self.enabled and self._tracking_lost:
                self._loss_effect.play()
        elif status == QSoundEffect.Status.Error:
            self._loss_pending = False
            logger.warning("Tracking-loss sound unavailable; check audio output and tracking_lost.wav")

    def stop(self):
        self._pending = False
        self._loss_pending = False
        if self._effect is not None:
            self._effect.stop()
        if self._loss_effect is not None:
            self._loss_effect.stop()
