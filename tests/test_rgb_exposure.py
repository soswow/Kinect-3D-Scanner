"""Exposure requests reach the RGB driver after warmup, or fail visibly."""

import ctypes
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from kinect_scanner.capture_process import capture_frames
from kinect_scanner.rgb_exposure import (
    AUTO_EXPOSURE, AUTO_FLICKER, AUTO_WHITE_BALANCE,
    ExposureControlUnavailable, RGBExposureControl, apply_rgb_exposure,
)
from kinect_scanner.worker import KinectWorker
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings


def exposure_driver():
    return SimpleNamespace(set_flag=Mock(return_value=0),
                           set_exposure=Mock(return_value=0),
                           get_exposure=Mock(return_value=3957))


class RGBExposureTests(unittest.TestCase):
    def test_fixed_shutter_disables_ae_and_flicker_but_retains_white_balance(self):
        driver, device = exposure_driver(), object()
        metadata = apply_rgb_exposure(driver, device, "manual", 250)
        self.assertEqual([(device, AUTO_EXPOSURE, 0), (device, AUTO_FLICKER, 0),
                          (device, AUTO_WHITE_BALANCE, 1)],
                         [call.args for call in driver.set_flag.call_args_list])
        driver.set_exposure.assert_called_once_with(device, 4000)
        self.assertEqual(3957, metadata["rgb_exposure_us"])
        self.assertEqual(250, metadata["rgb_shutter_speed"])

    def test_auto_restores_automatic_flags_after_manual(self):
        driver, device = exposure_driver(), object()
        controls = RGBExposureControl(driver, device)
        controls.apply("manual", 250)
        driver.set_flag.reset_mock()
        driver.set_exposure.reset_mock()
        self.assertIsNone(controls.apply("auto", 250))
        self.assertEqual([(device, AUTO_EXPOSURE, 1), (device, AUTO_FLICKER, 1),
                          (device, AUTO_WHITE_BALANCE, 1)],
                         [call.args for call in driver.set_flag.call_args_list])
        driver.set_exposure.assert_not_called()

    def test_unavailable_bindings_allow_default_auto_but_reject_manual(self):
        driver = SimpleNamespace()
        self.assertFalse(apply_rgb_exposure(driver, object(), "auto", 125)["rgb_exposure_controls"])
        with self.assertRaises(ExposureControlUnavailable):
            apply_rgb_exposure(driver, object(), "manual", 125)

    def test_driver_errors_and_invalid_readback_do_not_claim_manual_success(self):
        for operation in ("set_flag", "set_exposure"):
            with self.subTest(operation=operation):
                driver = exposure_driver()
                getattr(driver, operation).return_value = -1
                with self.assertRaises(RuntimeError):
                    apply_rgb_exposure(driver, object(), "manual", 250)
        for readback in (-1, 0, 4001, None, True):
            with self.subTest(readback=readback):
                driver = exposure_driver()
                driver.get_exposure.return_value = readback
                with self.assertRaisesRegex(RuntimeError, "confirm"):
                    apply_rgb_exposure(driver, object(), "manual", 250)

    def test_native_bridge_uses_extension_library_and_validated_device_pointer(self):
        class DevPtr:
            def __repr__(self):
                return "<Dev Pointer 0x0000000000001234>"

        device = DevPtr()
        driver = SimpleNamespace(DevPtr=DevPtr, __file__="/driver/freenect.so")
        library = SimpleNamespace(freenect_set_flag=Mock(return_value=0),
                                  freenect_set_exposure=Mock(return_value=0))

        def read(pointer, output):
            self.assertEqual(0x1234, pointer.value)
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = 3957
            return 0

        library.freenect_get_exposure = Mock(side_effect=read)
        with patch("kinect_scanner.rgb_exposure.ctypes.CDLL", return_value=library) as loader:
            self.assertEqual(3957, RGBExposureControl(driver, device).apply("manual", 250))
            loader.assert_called_once_with(driver.__file__)
        pointer, time_us = library.freenect_set_exposure.call_args.args
        self.assertEqual((0x1234, 4000), (pointer.value, time_us))
        library.freenect_get_exposure.return_value = -1
        library.freenect_get_exposure.side_effect = None
        with patch("kinect_scanner.rgb_exposure.ctypes.CDLL", return_value=library):
            with self.assertRaisesRegex(RuntimeError, "read shutter"):
                RGBExposureControl(driver, device).apply("manual", 250)

    def test_native_bridge_rejects_unknown_handles_or_missing_symbols(self):
        class DevPtr:
            def __repr__(self):
                return "<Dev Pointer 0x0>"

        driver = SimpleNamespace(DevPtr=DevPtr, __file__="/driver/freenect.so")
        for device in (object(), DevPtr()):
            with self.assertRaises(ExposureControlUnavailable):
                RGBExposureControl(driver, device)
        with patch.object(DevPtr, "__repr__", return_value="<Dev Pointer 0x1234>"), \
             patch("kinect_scanner.rgb_exposure.ctypes.CDLL", return_value=SimpleNamespace()):
            with self.assertRaises(ExposureControlUnavailable):
                RGBExposureControl(driver, DevPtr())

    def test_settings_roundtrip_legacy_defaults_and_frame_period_validation(self):
        settings = ScanSettings(sensor_calibration=load_calibration(),
                                rgb_exposure_mode="manual", rgb_shutter_speed=250)
        self.assertEqual(settings, ScanSettings.from_dict(settings.to_dict()))
        legacy = settings.to_dict()
        del legacy["rgb_exposure_mode"], legacy["rgb_shutter_speed"]
        self.assertEqual("auto", ScanSettings.from_dict(legacy).rgb_exposure_mode)
        invalid = ({"rgb_exposure_mode": "shutter_priority"}, {"rgb_shutter_speed": True},
                   {"rgb_shutter_speed": 9}, {"rgb_shutter_speed": 10001},
                   {"rgb_shutter_speed": 125.0}, {"rgb_shutter_speed": "125"},
                   {"rgb_exposure_mode": "manual", "rgb_shutter_speed": 10, "rgb_mode": "rgb_low_res"})
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ScanSettings(sensor_calibration=load_calibration(), **values)
                with self.assertRaises(ValueError):
                    KinectWorker(**values)

    def run_capture(self, mode, *, high_res=True, fail=False, retry=False):
        driver = exposure_driver()
        driver.RESOLUTION_MEDIUM, driver.RESOLUTION_HIGH = 1, 2
        driver.DEPTH_11BIT, driver.VIDEO_RGB, driver.VIDEO_IR_10BIT = 0, 0, 2
        driver.init = lambda: object()
        driver.num_devices = lambda ctx: 1
        driver.open_device = lambda ctx, index: object()
        driver.set_depth_mode = lambda *args: 0
        driver.set_depth_callback = lambda dev, callback: setattr(driver, "depth_callback", callback)
        driver.set_video_callback = lambda dev, callback: setattr(driver, "video_callback", callback)
        switches = []

        def video_mode(dev, resolution, fmt):
            driver.video_mode = fmt
            switches.append(fmt)
            return 0

        driver.set_video_mode = video_mode
        for name in ("start_depth", "start_video", "stop_video", "stop_depth", "close_device", "shutdown"):
            setattr(driver, name, lambda *args: 0)
        if fail:
            driver.set_exposure.return_value = -1
        rgb_shape = (1024, 1280, 3) if high_res else (480, 640, 3)
        stamp = 0

        def events(ctx):
            nonlocal stamp
            stamp += 2_000_000
            driver.depth_callback(None, np.full((480, 640), 750, np.uint16), stamp)
            if retry and driver.video_mode == 0 and switches.count(0) == 1:
                return 0
            frame = np.zeros((488, 640), np.uint16) if driver.video_mode == 2 else np.full(rgb_shape, 42, np.uint8)
            driver.video_callback(None, frame, stamp)
            return 0

        driver.process_events = events
        stop, packets = threading.Event(), []

        def send(packet):
            packets.append(packet)
            if packet[0] in ("frame", "error"):
                stop.set()

        connection = SimpleNamespace(send=send, close=lambda: None)
        with patch.dict("sys.modules", freenect=driver), \
             patch("kinect_scanner.capture_process.time.monotonic", side_effect=lambda: stamp / 2_000_000):
            capture_frames(connection, stop, bytearray(int(np.prod(rgb_shape))),
                           bytearray(480 * 640 * 2), high_res, mode, 250)
        return driver, switches, packets

    def test_capture_reapplies_after_ir_retry_and_publishes_verified_exposure(self):
        for high_res in (True, False):
            with self.subTest(high_res=high_res):
                driver, switches, packets = self.run_capture("manual", high_res=high_res, retry=True)
                self.assertEqual([2, 0, 2, 0], switches)
                self.assertEqual(2, driver.set_exposure.call_count)
                frames = [payload for kind, payload in packets if kind == "frame"]
                self.assertEqual(1, len(frames), packets)
                self.assertEqual("manual", frames[0]["rgb_exposure_mode"])
                self.assertEqual(3957, frames[0]["rgb_exposure_us"])
                self.assertEqual(250, frames[0]["rgb_shutter_speed"])

    def test_failed_manual_capture_sends_error_without_frames(self):
        driver, switches, packets = self.run_capture("manual", fail=True)
        self.assertEqual("error", packets[-1][0])
        self.assertIn("shutter", packets[-1][1])
        self.assertFalse(any(kind == "frame" for kind, payload in packets))


if __name__ == "__main__":
    unittest.main()
