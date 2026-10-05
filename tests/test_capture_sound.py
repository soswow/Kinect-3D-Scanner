"""Capture cues must respect mute, loading and already playing audio."""

import unittest
from unittest.mock import Mock, patch

from kinect_scanner.gui import feedback

SoundStatus = feedback.QSoundEffect.Status


class CaptureSoundTests(unittest.TestCase):
    def setUp(self):
        self.preferences = {}
        settings = Mock()
        settings.value.side_effect = lambda key, default, **kwargs: self.preferences.get(key, default)
        settings.setValue.side_effect = self.preferences.__setitem__
        self.settings_patch = patch.object(feedback, "QSettings", return_value=settings)
        self.settings_patch.start()
        self.effect_patch = patch.object(feedback, "QSoundEffect")
        self.effect_class = self.effect_patch.start()
        self.effect_class.Status = SoundStatus
        self.effect = self.effect_class.return_value
        self.effect.status.return_value = SoundStatus.Loading
        self.effect.isPlaying.return_value = False
        self.sound = feedback.CaptureSound()
        self.addCleanup(self.settings_patch.stop)
        self.addCleanup(self.effect_patch.stop)

    def test_mute_during_loading_cancels_deferred_cue_and_is_remembered(self):
        self.assertTrue(self.sound.enabled)
        self.sound.play()
        self.effect.play.assert_not_called()
        self.sound.set_enabled(False)
        self.effect.status.return_value = SoundStatus.Ready
        self.sound._on_status_changed()
        self.sound.play()
        self.effect.play.assert_not_called()
        self.assertFalse(feedback.CaptureSound().enabled)

    def test_rapid_confirmations_coalesce_and_never_overlap(self):
        for _ in range(5):
            self.sound.play()
        self.effect_class.assert_called_once()
        self.effect.play.assert_not_called()
        self.effect.status.return_value = SoundStatus.Ready
        self.sound._on_status_changed()
        self.effect.play.assert_called_once()
        self.effect.isPlaying.return_value = True
        for _ in range(5):
            self.sound.play()
        self.effect.play.assert_called_once()
        self.effect.isPlaying.return_value = False
        self.sound.play()
        self.assertEqual(2, self.effect.play.call_count)

    def test_stopped_capture_does_not_play_after_asset_loads(self):
        self.sound.play()
        self.sound.stop()
        self.effect.status.return_value = SoundStatus.Ready
        self.sound._on_status_changed()
        self.effect.play.assert_not_called()
        self.assertTrue(self.sound.enabled)


if __name__ == "__main__":
    unittest.main()
