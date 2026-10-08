"""Guided physical capture: fitting, rejection, aborts, and native-call bounds."""

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.calibrate_accelerometer import fitted_profile, guided_calibration, main
from shared.accelerometer_calibration import (CalibrationCapture, POSES, VALIDATION_POSES,
                                              capture_process, stationary_observation)
from shared.inertial import G, calibration_profile


ROTATION = np.asarray(calibration_profile()["sensor_to_camera"])
BIAS = np.array([0.1, -0.2, 0.08])
SCALE = np.array([1.02, 0.98, 1.01])


def samples_for(up, seconds=3.0):
    vector = (ROTATION.T @ (G * np.asarray(up))) / SCALE + BIAS
    result = []
    for index in range(int(seconds * 20)):
        stamp = 2 + index * 0.05
        result.append({"phase": "measure", "valid": True, "capture_generation": "test-device",
                       "sequence": index + 1, "raw_counts": [1, 2, 3],
                       "acceleration_m_s2": vector.tolist(), "host_monotonic_s": stamp,
                       "read_start_s": stamp - 0.001, "read_end_s": stamp + 0.001})
    return result


class FakeCapture:
    metadata = {"device_index": 0, "capture_generation": "test-device"}

    def __init__(self, index=0, *, noisy_first=False, abort=False, bad_validation=False):
        self.calls = 0
        self.closed = False
        self.noisy_first = noisy_first
        self.abort = abort
        self.bad_validation = bad_validation

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def collect(self, seconds, settle, on_sample):
        plan = POSES + VALIDATION_POSES
        pose_index = self.calls - (1 if self.noisy_first and self.calls > 0 else 0)
        samples = samples_for(plan[min(pose_index, len(plan) - 1)][2], seconds)
        if self.bad_validation and self.calls == 8:
            for sample in samples:
                sample["acceleration_m_s2"][1] += 0.6
        if self.noisy_first and self.calls == 0:
            for index, sample in enumerate(samples):
                sample["acceleration_m_s2"][0] += 0.5 if index % 2 else -0.5
        self.calls += 1
        for sample in samples:
            on_sample(sample)
            if self.abort:
                raise KeyboardInterrupt
        return samples


def stalled_capture(connection, stop_event, device_index):
    connection.send(("ready", {"device_index": device_index}))
    stop_event.wait(30)


class GuidedCalibrationTests(unittest.TestCase):
    def run_guide(self, root, *, checked="y", noisy=False, abort=False):
        output = root / "calibration.json"
        capture = FakeCapture(noisy_first=noisy, abort=abort)
        answers = iter([checked, "my-kinect"] + [""] * 10)
        profile = guided_calibration(output, capture_factory=lambda _: capture,
                                     input_fn=lambda _: next(answers), print_fn=lambda _: None)
        data = json.loads((root / "calibration.measurements.json").read_text())
        return profile, data, capture

    def test_guide_records_fresh_validation_and_recovers_known_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, data, capture = self.run_guide(root)
            self.assertTrue(profile["verified"])
            self.assertEqual(data["status"], "complete")
            self.assertEqual(len(data["observations"]), 6)
            self.assertEqual(len(data["validation"]), 3)
            self.assertEqual(capture.calls, 9)
            self.assertTrue(capture.closed)
            self.assertEqual(len(data["attempts"][0]["samples"]), 60)
            np.testing.assert_allclose(profile["bias_m_s2"], BIAS, atol=1e-10)
            np.testing.assert_allclose(profile["scale"], SCALE, atol=1e-10)
            np.testing.assert_allclose(profile["sensor_to_camera"], ROTATION, atol=1e-10)
            self.assertEqual(calibration_profile(json.loads((root / "calibration.json").read_text())), profile)
            self.assertEqual(fitted_profile(data), profile)

    def test_eyeballed_reference_stays_unverified_including_offline_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, data, _ = self.run_guide(root, checked="n")
            self.assertFalse(profile["verified"])
            self.assertFalse(fitted_profile(data)["verified"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root / "calibration.measurements.json"), str(root / "refitted.json")]), 0)
            self.assertFalse(json.loads((root / "refitted.json").read_text())["verified"])

    def test_motion_retries_and_retains_rejected_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            profile, data, capture = self.run_guide(Path(directory), noisy=True)
            self.assertTrue(profile["verified"])
            self.assertEqual(capture.calls, 10)
            self.assertFalse(data["attempts"][0]["accepted"])
            self.assertIn("unstable", data["attempts"][0]["error"])
            self.assertIn("sensor noise", data["attempts"][0]["error"])
            self.assertTrue(data["attempts"][1]["accepted"])

    def test_failed_validation_can_be_replaced_without_reusing_fit_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(bad_validation=True)
            answers = iter(["y", "my-kinect"] + [""] * 9 + ["9", ""])
            messages = []
            profile = guided_calibration(root / "calibration.json", capture_factory=lambda _: capture,
                                         input_fn=lambda _: next(answers), print_fn=messages.append)
            data = json.loads((root / "calibration.measurements.json").read_text())
            self.assertTrue(profile["verified"])
            self.assertEqual(capture.calls, 10)
            self.assertEqual(len(data["validation"]), 3)
            self.assertEqual(len(data["observations"]), 6)
            self.assertTrue(any("did not pass" in message for message in messages))
            self.assertEqual(len(data["attempts"]), 10)

    def test_abort_keeps_partial_raw_reads_and_releases_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(abort=True)
            answers = iter(["y", "my-kinect", ""])
            with self.assertRaises(KeyboardInterrupt):
                guided_calibration(root / "calibration.json", capture_factory=lambda _: capture,
                                   input_fn=lambda _: next(answers), print_fn=lambda _: None)
            data = json.loads((root / "calibration.measurements.json").read_text())
            self.assertEqual(data["status"], "interrupted")
            self.assertEqual(len(data["attempts"][0]["samples"]), 1)
            self.assertFalse((root / "calibration.json").exists())
            self.assertTrue(capture.closed)

    def test_existing_files_are_preserved_before_device_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("calibration.json", "calibration.measurements.json"):
                path = root / name
                path.write_text("existing")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    guided_calibration(root / "calibration.json", print_fn=lambda _: None)
                self.assertEqual(path.read_text(), "existing")
                path.unlink()

    def test_bad_quality_is_not_hidden_by_averaging(self):
        for kind in ("latency", "failed_reads", "short", "reconnect", "drift"):
            with self.subTest(kind=kind):
                samples = samples_for((0, -1, 0))
                if kind == "latency":
                    for sample in samples:
                        sample["read_end_s"] += 0.04
                elif kind == "failed_reads":
                    for sample in samples[:10]:
                        sample["valid"] = False
                elif kind == "short":
                    samples = samples[:30]
                elif kind == "reconnect":
                    samples[-1]["capture_generation"] = "new-device"
                else:
                    for index, sample in enumerate(samples):
                        sample["acceleration_m_s2"][0] += index * 0.01
                with self.assertRaises(ValueError):
                    stationary_observation(samples, (0, -1, 0), 3)

    def test_stationary_sensor_noise_is_averaged_instead_of_called_motion(self):
        for scatter in (0.161, 0.178):
            with self.subTest(scatter=scatter):
                samples = samples_for((0, -1, 0))
                original = np.asarray(samples[0]["acceleration_m_s2"])
                noise = np.random.default_rng(42).normal(size=(len(samples), 3))
                noise -= noise.mean(axis=0)
                noise *= scatter / np.sqrt(np.mean(np.sum(noise ** 2, axis=1)))
                for sample, variation in zip(samples, noise):
                    sample["acceleration_m_s2"] = (original + variation).tolist()
                observation = stationary_observation(samples, (0, -1, 0), 3)
                np.testing.assert_allclose(observation["acceleration_m_s2"], original)
                summary = observation["capture_summary"]
                self.assertAlmostEqual(summary["scatter_rms_m_s2"], scatter)
                self.assertLess(summary["linear_drift_m_s2"], 0.2)
                self.assertLess(summary["block_shift_m_s2"], 0.2)

    def test_slow_drift_and_a_step_are_rejected_despite_small_scatter(self):
        for kind in ("drift", "step"):
            with self.subTest(kind=kind):
                samples = samples_for((0, -1, 0))
                for index, sample in enumerate(samples):
                    change = (index / (len(samples) - 1) * 0.3 if kind == "drift"
                              else (0.3 if index >= len(samples) // 2 else 0))
                    sample["acceleration_m_s2"][0] += change
                with self.assertRaisesRegex(ValueError, "Readings are unstable"):
                    stationary_observation(samples, (0, -1, 0), 3)

    def test_large_isolated_disturbance_is_rejected(self):
        samples = samples_for((0, -1, 0))
        samples[15]["acceleration_m_s2"][0] += 1
        with self.assertRaisesRegex(ValueError, "Readings are unstable"):
            stationary_observation(samples, (0, -1, 0), 3)

    def test_settling_samples_are_excluded(self):
        samples = samples_for((0, -1, 0))
        settling = {**samples[0], "phase": "settle", "acceleration_m_s2": [0, 0, 0]}
        observation = stationary_observation([settling] + samples, (0, -1, 0), 3)
        self.assertEqual(observation["capture_summary"]["sample_count"], 60)

    def test_native_stall_is_bounded_and_child_is_reaped(self):
        capture = CalibrationCapture(target=stalled_capture, timeout=2)
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            with capture:
                capture.collect(seconds=2)
        self.assertLess(time.monotonic() - started, 6)
        self.assertTrue(capture.process._closed)

    def test_driver_acquisition_records_original_units_and_closes_device(self):
        stop = threading.Event()
        messages = []
        connection = Mock()
        connection.poll.return_value = True
        connection.recv.return_value = (0.2, 0.0)

        def send(message):
            messages.append(message)
            if message[0] == "done":
                stop.set()

        connection.send.side_effect = send
        driver = SimpleNamespace(init=Mock(return_value="context"), num_devices=Mock(return_value=1),
                                 open_device=Mock(return_value="device"), close_device=Mock(), shutdown=Mock(),
                                 update_tilt_state=Mock(return_value=10),
                                 get_tilt_state=Mock(return_value=SimpleNamespace(accelerometer_x=0, accelerometer_y=819, accelerometer_z=0)),
                                 get_mks_accel=Mock(return_value=(0, G, 0)))
        with patch.dict("sys.modules", {"freenect": driver}):
            capture_process(connection, stop, 0)
        reads = [payload for kind, payload in messages if kind == "sample"]
        self.assertGreaterEqual(len(reads), 3)
        self.assertEqual(reads[0]["raw_counts"], [0, 819, 0])
        self.assertEqual(reads[0]["acceleration_m_s2"], [0, G, 0])
        self.assertEqual([kind for kind, _ in messages][0], "ready")
        self.assertEqual([kind for kind, _ in messages][-1], "done")
        driver.close_device.assert_called_once_with("device")
        driver.shutdown.assert_called_once_with("context")

    def test_missing_hardware_is_reported_and_context_is_closed(self):
        connection = Mock()
        driver = SimpleNamespace(init=Mock(return_value="context"), num_devices=Mock(return_value=0), shutdown=Mock())
        with patch.dict("sys.modules", {"freenect": driver}):
            capture_process(connection, threading.Event(), 0)
        kind, message = connection.send.call_args.args[0]
        self.assertEqual(kind, "error")
        self.assertIn("No Kinect", message)
        driver.shutdown.assert_called_once_with("context")

    def test_cli_rejects_invalid_capture_arguments(self):
        for flags in (["--seconds", "nan"], ["--seconds", "0"], ["--device-index", "-1"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(["--interactive"] + flags)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
