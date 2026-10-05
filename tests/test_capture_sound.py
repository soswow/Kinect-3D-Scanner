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

    def test_loss_alert_plays_once_per_episode_and_recovery_rearms(self):
        recovery_effect = Mock()
        recovery_effect.status.return_value = SoundStatus.Ready
        self.effect_class.side_effect = [self.effect, recovery_effect]
        self.effect.status.return_value = SoundStatus.Ready
        for _ in range(5):
            self.sound.set_tracking_lost(True)
        self.effect.play.assert_called_once()
        self.sound.set_tracking_lost(False)
        self.sound.set_tracking_lost(True)
        self.assertEqual(2, self.effect.play.call_count)

    def test_recovery_cancels_warning_that_is_still_loading(self):
        self.sound.set_tracking_lost(True)
        self.sound.set_tracking_lost(False)
        self.effect.status.return_value = SoundStatus.Ready
        self.sound._on_loss_status_changed()
        self.effect.play.assert_not_called()

    def test_muting_cancels_warning_and_unmuting_does_not_repeat_episode(self):
        recovery_effect = Mock()
        recovery_effect.status.return_value = SoundStatus.Ready
        self.effect_class.side_effect = [self.effect, recovery_effect]
        self.sound.set_tracking_lost(True)
        self.sound.set_enabled(False)
        self.effect.status.return_value = SoundStatus.Ready
        self.sound._on_loss_status_changed()
        self.sound.set_enabled(True)
        self.sound.set_tracking_lost(True)
        self.effect.play.assert_not_called()
        self.sound.set_tracking_lost(False)
        self.sound.set_tracking_lost(True)
        self.effect.play.assert_called_once()

    def test_loss_warning_cancels_capture_click_and_silences_recovery_probes(self):
        capture_effect = self.effect
        loss_effect = Mock()
        loss_effect.status.return_value = SoundStatus.Ready
        recovery_effect = Mock()
        recovery_effect.status.return_value = SoundStatus.Ready
        recovery_effect.isPlaying.return_value = False
        self.effect_class.side_effect = [capture_effect, loss_effect, recovery_effect]
        self.sound.play()  # Still loading when tracking is lost.
        self.sound.set_tracking_lost(True)
        capture_effect.stop.assert_called_once()
        capture_effect.status.return_value = SoundStatus.Ready
        self.sound._on_status_changed()
        self.sound.play()
        capture_effect.play.assert_not_called()
        loss_effect.play.assert_called_once()
        self.sound.set_tracking_lost(False)
        self.sound.play()
        capture_effect.play.assert_called_once()

    def test_stopped_warning_does_not_play_after_asset_loads(self):
        self.sound.set_tracking_lost(True)
        self.sound.stop()
        self.effect.status.return_value = SoundStatus.Ready
        self.sound._on_loss_status_changed()
        self.effect.play.assert_not_called()

    def test_recovery_alert_requires_loss_and_plays_once(self):
        recovery_effect = Mock()
        recovery_effect.status.return_value = SoundStatus.Ready
        # Also cover a synchronous Ready signal while setSource runs.
        recovery_effect.setSource.side_effect = lambda _: self.sound._on_recovery_status_changed()
        self.effect_class.side_effect = [self.effect, recovery_effect]
        self.sound.set_tracking_lost(False)
        self.effect_class.assert_not_called()
        self.sound.set_tracking_lost(True)
        for _ in range(5):
            self.sound.set_tracking_lost(False)
        recovery_effect.play.assert_called_once()

    def test_muting_while_recovery_loads_cancels_cue(self):
        recovery_effect = Mock()
        recovery_effect.status.return_value = SoundStatus.Loading
        self.effect_class.side_effect = [self.effect, recovery_effect]
        self.sound.set_tracking_lost(True)
        self.sound.set_tracking_lost(False)
        self.sound.set_enabled(False)
        recovery_effect.status.return_value = SoundStatus.Ready
        self.sound._on_recovery_status_changed()
        self.sound.set_enabled(True)
        self.sound.set_tracking_lost(False)
        recovery_effect.play.assert_not_called()

    def test_new_loss_cancels_loading_recovery_cue(self):
        recovery_effect = Mock()
        recovery_effect.status.return_value = SoundStatus.Loading
        self.effect_class.side_effect = [self.effect, recovery_effect]
        self.sound.set_tracking_lost(True)
        self.sound.set_tracking_lost(False)
        self.sound.set_tracking_lost(True)
        recovery_effect.status.return_value = SoundStatus.Ready
        self.sound._on_recovery_status_changed()
        recovery_effect.play.assert_not_called()

    def test_reset_does_not_report_recovery_or_leave_a_deferred_cue(self):
        recovery_effect = Mock()
        recovery_effect.status.return_value = SoundStatus.Loading
        self.effect_class.side_effect = [self.effect, recovery_effect]
        self.sound.set_tracking_lost(True)
        self.sound.reset_tracking()
        self.assertEqual(1, self.effect_class.call_count)
        self.sound.set_tracking_lost(False)
        self.assertEqual(1, self.effect_class.call_count)
        self.sound.set_tracking_lost(True)
        self.sound.set_tracking_lost(False)
        self.sound.reset_tracking()
        recovery_effect.status.return_value = SoundStatus.Ready
        self.sound._on_recovery_status_changed()
        recovery_effect.play.assert_not_called()

    def test_recovery_cue_takes_priority_over_capture_until_finished(self):
        loss_effect, recovery_effect = Mock(), Mock()
        loss_effect.status.return_value = SoundStatus.Ready
        recovery_effect.status.return_value = SoundStatus.Loading
        recovery_effect.isPlaying.return_value = False
        self.effect.status.return_value = SoundStatus.Ready
        self.effect_class.side_effect = [loss_effect, recovery_effect, self.effect]
        self.sound.set_tracking_lost(True)
        self.sound.set_tracking_lost(False)
        self.sound.play()
        self.assertEqual(2, self.effect_class.call_count)
        recovery_effect.status.return_value = SoundStatus.Ready
        recovery_effect.isPlaying.return_value = True
        self.sound._on_recovery_status_changed()
        self.sound.play()
        self.assertEqual(2, self.effect_class.call_count)
        recovery_effect.isPlaying.return_value = False
        self.sound.play()
        self.effect.play.assert_called_once()


if __name__ == "__main__":
    unittest.main()
