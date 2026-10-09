"""Nonblocking capture and tracking feedback with a remembered mute."""

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
        self._recovery_effect = None
        self._recovery_pending = False
        self._tracking_lost = False
        self._sound_errors = set()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self._settings.setValue("feedback/capture_sound", self.enabled)
        self._settings.sync()
        if not self.enabled:
            self.stop()

    def play(self):
        if (
            not self.enabled or self._tracking_lost or self._recovery_pending
            or (self._recovery_effect is not None and self._recovery_effect.isPlaying())
        ):
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
        if self._effect.status() == QSoundEffect.Status.Ready:
            self._log_sound_recovery("Capture")
        if self._effect.status() == QSoundEffect.Status.Ready and self._pending:
            self.play()
        elif self._effect.status() == QSoundEffect.Status.Error:
            self._pending = False
            self._log_sound_error("Capture", "capture.wav")

    def set_tracking_lost(self, lost):
        """Alert once on loss and once when that lost track is reacquired."""
        lost = bool(lost)
        if lost == self._tracking_lost:
            return
        self._tracking_lost = lost
        if not lost:
            self._loss_pending = False
            if self._loss_effect is not None:
                self._loss_effect.stop()
            if self.enabled:
                self._play_recovery()
            return
        self._recovery_pending = False
        if self._recovery_effect is not None:
            self._recovery_effect.stop()
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
        if status == QSoundEffect.Status.Ready:
            self._log_sound_recovery("Tracking-loss")
        if status == QSoundEffect.Status.Ready and self._loss_pending:
            self._loss_pending = False
            if self.enabled and self._tracking_lost:
                self._loss_effect.play()
        elif status == QSoundEffect.Status.Error:
            self._loss_pending = False
            self._log_sound_error("Tracking-loss", "tracking_lost.wav")

    def _play_recovery(self):
        self._pending = False
        if self._effect is not None:
            self._effect.stop()
        self._recovery_pending = True
        if self._recovery_effect is None:
            self._recovery_effect = QSoundEffect(self)
            self._recovery_effect.setVolume(0.7)
            self._recovery_effect.setLoopCount(1)
            self._recovery_effect.statusChanged.connect(self._on_recovery_status_changed)
            source = Path(__file__).with_name("assets") / "tracking_reacquired.wav"
            self._recovery_effect.setSource(QUrl.fromLocalFile(str(source)))
        self._on_recovery_status_changed()

    def _on_recovery_status_changed(self):
        status = self._recovery_effect.status()
        if status == QSoundEffect.Status.Ready:
            self._log_sound_recovery("Tracking-recovery")
        if status == QSoundEffect.Status.Ready and self._recovery_pending:
            self._recovery_pending = False
            if self.enabled and not self._tracking_lost:
                self._recovery_effect.play()
        elif status == QSoundEffect.Status.Error:
            self._recovery_pending = False
            self._log_sound_error("Tracking-recovery", "tracking_reacquired.wav")

    def _log_sound_error(self, sound, filename):
        if sound not in self._sound_errors:
            self._sound_errors.add(sound)
            logger.warning("%s sound unavailable; check audio output and %s", sound, filename,
                           extra={"ui_state_key": ("sound", sound)})

    def _log_sound_recovery(self, sound):
        if sound in self._sound_errors:
            self._sound_errors.remove(sound)
            logger.info("%s sound available again", sound,
                        extra={"ui_event": True, "ui_state_key": ("sound", sound)})

    def reset_tracking(self):
        """Silently clear an abandoned or replaced session, without a recovery cue."""
        self.stop()
        self._tracking_lost = False

    def stop(self):
        self._pending = False
        self._loss_pending = False
        self._recovery_pending = False
        if self._effect is not None:
            self._effect.stop()
        if self._loss_effect is not None:
            self._loss_effect.stop()
        if self._recovery_effect is not None:
            self._recovery_effect.stop()
