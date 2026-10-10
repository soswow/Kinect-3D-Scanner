"""Sensor reliability, unobservable motion, native read errors, and orientation."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from shared.capture import DeviceClockMapper
from shared.inertial import (G, AccelerometerPoller, GravityEstimator, OrientationTracker,
                             calibration_profile, gravity_seed, rotate_display, orientation_observation)
from shared.inertial import fit_calibration
from shared.settings import ScanSettings


def sample(stamp=1, acceleration=(0, G, 0), valid=True):
    return {"valid": valid, "host_monotonic_s": stamp,
            "read_start_s": stamp - .001, "read_end_s": stamp + .001,
            "acceleration_m_s2": list(acceleration)}


def metadata(up=(0, -1, 0), confidence=1, generation="connection", verified=True):
    return {"accelerometer": {"valid": True, "capture_generation": generation,
                              "sample_delta_ms": 0,
                              "gravity": {"valid": True, "up_camera": list(up), "confidence": confidence,
                                          "calibration_id": "test", "calibration_verified": verified}}}


class InertialTests(unittest.TestCase):
    def test_display_orientation_does_not_need_exposure_clock_association(self):
        driver = SimpleNamespace(update_tilt_state=lambda _: 0, get_tilt_state=lambda _: None,
                                 get_mks_accel=lambda _: (0, G, 0))
        poller = AccelerometerPoller(driver, None, generation="connection")
        gravity = metadata((-1, 0, 0))["accelerometer"]["gravity"]
        reading = {**sample(1), "gravity": gravity, "sequence": 1, "capture_generation": "connection"}
        poller.samples.append(reading)
        self.assertFalse(poller.associate(1, 0.075)["valid"])
        self.assertTrue(poller.for_orientation(1.1)["valid"])
        self.assertFalse(poller.for_orientation(1.3)["valid"])
        self.assertFalse(orientation_observation(reading, 1.1, "reconnected")["valid"])
        self.assertFalse(orientation_observation(reading, .9, "connection")["valid"])
        poller.samples.append({**reading, "valid": False, "reason": "USB read failed"})
        self.assertFalse(poller.for_orientation(1.1)["valid"])
        self.assertFalse(poller.associate(1, 0.075)["valid"])

    def test_gravity_static_and_linear_acceleration_contamination(self):
        estimator = GravityEstimator()
        for i in range(8):
            gravity = estimator.update(sample(1 + i * .05))
        self.assertGreater(gravity["confidence"], .99)
        np.testing.assert_allclose(gravity["up_camera"], [0, -1, 0])
        contaminated = estimator.update(sample(1.4, (G * .8, G, 0)))
        self.assertLess(contaminated["confidence"], .4)
        self.assertFalse(estimator.update(sample(1.45, (0, 0, 0)))["valid"])
        self.assertFalse(estimator.update(sample(1.5, valid=False))["valid"])
        resumed = estimator.update(sample(2))
        self.assertLess(resumed["confidence"], .4)  # Rebuild evidence after a gap/error.

    def test_positive_driver_return_is_success_and_failed_reads_are_not_reused(self):
        state = SimpleNamespace(accelerometer_x=0, accelerometer_y=819, accelerometer_z=0)
        result = [10]
        driver = SimpleNamespace(update_tilt_state=lambda _: result[0],
                                 get_tilt_state=lambda _: state, get_mks_accel=lambda _: (0, G, 0))
        poller = AccelerometerPoller(driver, object(), "connection")
        for _ in range(6):
            poller.next_poll = 0
            observation = poller.poll()
        self.assertTrue(observation["valid"])
        self.assertEqual(observation["raw_counts"], [0, 819, 0])
        self.assertTrue(poller.associate(observation["host_monotonic_s"])["valid"])
        self.assertFalse(poller.associate(observation["host_monotonic_s"], .031)["valid"])
        self.assertFalse(poller.associate(observation["host_monotonic_s"] + 1)["valid"])
        result[0] = -1
        for _ in range(3):
            poller.next_poll = 0
            failed = poller.poll()
            self.assertFalse(failed["valid"])
            self.assertNotIn("acceleration_m_s2", failed)
        self.assertFalse(poller.enabled)
        self.assertFalse(poller.associate(failed["host_monotonic_s"])["valid"])
        self.assertIsNone(poller.poll())

    def test_unavailable_and_zero_cached_state_degrade_without_camera_errors(self):
        self.assertFalse(AccelerometerPoller(SimpleNamespace(), None).enabled)
        state = SimpleNamespace(accelerometer_x=0, accelerometer_y=0, accelerometer_z=0)
        driver = SimpleNamespace(update_tilt_state=lambda _: 0, get_tilt_state=lambda _: state,
                                 get_mks_accel=lambda _: (0, 0, 0))
        self.assertFalse(AccelerometerPoller(driver, None).poll()["valid"])

    def test_orientation_hysteresis_staleness_and_manual_lock(self):
        orientation = OrientationTracker()
        upright = metadata()["accelerometer"]
        self.assertEqual(orientation.update(upright, 1)["rotation_cw_degrees"], 0)
        left = metadata((-1, 0, 0))["accelerometer"]
        self.assertEqual(orientation.update(left, 1.1)["rotation_cw_degrees"], 0)
        self.assertEqual(orientation.update(left, 1.5)["rotation_cw_degrees"], 90)
        edge = math.radians(40)
        near_boundary = metadata((-math.sin(edge), -math.cos(edge), 0))["accelerometer"]
        self.assertEqual(orientation.update(near_boundary, 2)["rotation_cw_degrees"], 90)
        self.assertEqual(orientation.update({}, 3)["rotation_cw_degrees"], 90)
        vertical = orientation.update(metadata((0, 0, -1))["accelerometer"], 4)
        self.assertFalse(vertical["valid"])
        self.assertEqual(vertical["rotation_cw_degrees"], 90)
        self.assertEqual(orientation.update({}, 5, "portrait_right")["rotation_cw_degrees"], 270)

    def test_every_quarter_turn_keeps_native_depth_exact(self):
        raw = np.arange(480 * 640, dtype=np.uint16).reshape(480, 640)
        for rotation in (0, 90, 180, 270):
            shown = rotate_display(raw, rotation)
            self.assertTrue(shown.flags.c_contiguous)
            np.testing.assert_array_equal(rotate_display(shown, (-rotation) % 360), raw)
        self.assertEqual(rotate_display(raw, 90).shape, (640, 480))

    def test_provisional_roll_matches_upstream_y_up_viewer_conversion(self):
        for acceleration, clockwise in (((0, G, 0), 0), ((G, 0, 0), 90),
                                        ((0, -G, 0), 180), ((-G, 0, 0), 270)):
            estimator, orientation = GravityEstimator(), OrientationTracker()
            for i in range(20):
                stamp = 1 + i * .05
                gravity = estimator.update(sample(stamp, acceleration))
                decision = orientation.update({"valid": True, "gravity": gravity}, stamp)
            self.assertEqual(decision["rotation_cw_degrees"], clockwise)

    def test_fusion_reduces_tilt_without_changing_translation_or_correcting_yaw(self):
        angle = math.radians(8)
        seed = np.eye(4)
        seed[:3, :3] = [[math.cos(angle), -math.sin(angle), 0],
                       [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
        seed[:3, 3] = [.2, .3, .4]
        result, report = gravity_seed(seed, np.eye(4), metadata(), metadata())
        self.assertTrue(report["applied"])
        self.assertAlmostEqual(report["correction_deg"], 3)
        self.assertLess(np.linalg.norm(result[:3, :3] @ [0, -1, 0] - [0, -1, 0]),
                        np.linalg.norm(seed[:3, :3] @ [0, -1, 0] - [0, -1, 0]))
        np.testing.assert_array_equal(result[:3, 3], seed[:3, 3])
        yaw = np.eye(4)
        yaw[:3, :3] = [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                      [-math.sin(angle), 0, math.cos(angle)]]
        unchanged, report = gravity_seed(yaw, np.eye(4), metadata(), metadata())
        np.testing.assert_allclose(unchanged, yaw)
        self.assertFalse(report["applied"])
        weak, weak_report = gravity_seed(seed, np.eye(4), metadata(verified=False), metadata(verified=False))
        self.assertLess(weak_report["correction_deg"], 1.1)

    def test_fusion_rejects_inverted_contaminated_stale_and_reconnected_data(self):
        seed = np.eye(4)
        bad = [metadata((0, 1, 0)), metadata(confidence=.1), metadata(generation="other"), {}]
        stale = metadata()
        stale["accelerometer"]["sample_delta_ms"] = 300
        bad.append(stale)
        for observation in bad:
            result, report = gravity_seed(seed, np.eye(4), metadata(), observation)
            np.testing.assert_array_equal(result, seed)
            self.assertFalse(report["applied"])

    def test_clock_wrap_and_late_rgb_preserve_the_same_epoch(self):
        clock = DeviceClockMapper()
        first = clock.observe((1 << 32) - 3_000_000, 10)
        depth = clock.observe(3_000_000, 10.101)
        late_rgb = clock.observe(0, 10.11)
        self.assertAlmostEqual(depth["device_timestamp_unwrapped_s"], .1)
        self.assertAlmostEqual(late_rgb["device_timestamp_unwrapped_s"], .05)
        self.assertAlmostEqual(depth["estimated_host_monotonic_s"] - first["estimated_host_monotonic_s"], .1)

    def test_calibration_validation_and_settings_roundtrip(self):
        profile = calibration_profile()
        self.assertEqual(ScanSettings.from_dict(ScanSettings(accelerometer_calibration=profile).to_dict()).accelerometer_calibration, profile)
        for invalid in ({}, {**profile, "verified": 1}, {**profile, "scale": [1, 1, -1]},
                        {**profile, "sensor_to_camera": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]}):
            with self.assertRaises(ValueError):
                calibration_profile(invalid)

    def test_calibration_requires_independent_directions_and_heldout_verification(self):
        profile = calibration_profile()
        rotation = np.asarray(profile["sensor_to_camera"])
        bias = np.array([.1, -.2, .08])
        scale = np.array([1.02, .98, 1.01])
        rows = []
        for up in np.concatenate((np.eye(3), -np.eye(3))):
            measurement = (rotation.T @ (G * up)) / scale + bias
            rows.append({"acceleration_m_s2": measurement.tolist(), "up_camera": up.tolist()})
        fitted = fit_calibration(rows)
        self.assertFalse(fitted["verified"])
        np.testing.assert_allclose(fitted["bias_m_s2"], bias, atol=1e-9)
        np.testing.assert_allclose(fitted["scale"], scale, atol=1e-9)
        self.assertTrue(fit_calibration(rows, rows[:3])["verified"])
        wrong = [{**row, "acceleration_m_s2": (np.array(row["acceleration_m_s2"]) + [1, 0, 0]).tolist()} for row in rows[:3]]
        with self.assertRaisesRegex(ValueError, "Held-out"):
            fit_calibration(rows, wrong)
        with self.assertRaisesRegex(ValueError, "span"):
            fit_calibration([rows[0]] * 6)

    def test_approximate_calibration_checks_nominal_directions_without_verification(self):
        rotation = np.asarray(calibration_profile()["sensor_to_camera"])
        rows = [{"acceleration_m_s2": (rotation.T @ (G * up)).tolist(), "up_camera": up.tolist()}
                for up in np.concatenate((np.eye(3), -np.eye(3)))]
        angle = math.radians(4)
        validation = [dict(row) for row in rows[:3]]
        validation[0] = {"acceleration_m_s2": (rotation.T @ (G * np.array([math.cos(angle), math.sin(angle), 0]))).tolist(),
                         "up_camera": [1, 0, 0]}
        with self.assertRaisesRegex(ValueError, "Held-out"):
            fit_calibration(rows, validation)
        approximate = fit_calibration(rows, validation, reference_checked=False)
        self.assertFalse(approximate["verified"])
        report = approximate["evidence"]["report"]
        self.assertEqual(report["reference_mode"], "approximate")
        self.assertEqual(report["acceptance_limits"], {"max_m_s2": 1.0, "max_angle_deg": 6})
        self.assertAlmostEqual(report["validation"]["max_angle_deg"], 4)
        angle = math.radians(15)
        validation[0]["acceleration_m_s2"] = (rotation.T @ (G * np.array([math.cos(angle), math.sin(angle), 0]))).tolist()
        with self.assertRaisesRegex(ValueError, "Held-out"):
            fit_calibration(rows, validation, reference_checked=False)
        with self.assertRaisesRegex(ValueError, "boolean"):
            fit_calibration(rows, reference_checked="no")
