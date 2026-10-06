"""Kinect v1 shutter and MT9M112 gain controls in the camera subprocess.

Resolve native functions through the loaded freenect extension, keeping its USB
handle. Manual controls are applied after automatic white balance has settled;
AE, AWB and flicker must all be disabled before writing sensor shutter/gain.
"""

import ctypes
import math
import re

from shared.capture import validate_rgb_exposure

AUTO_EXPOSURE = 1 << 14
AUTO_FLICKER = 1 << 7
AUTO_WHITE_BALANCE = 1 << 1
AUTOMATIC_MASK = AUTO_EXPOSURE | AUTO_FLICKER | AUTO_WHITE_BALANCE
CHIP_VERSION = 0x000
MT9M112_VERSION = 0x148C
MODE_CONTROL = 0x106
CHANNEL_GAINS = (0x02B, 0x02C, 0x02D, 0x02E)
GAIN_MASK = 0x71FF
DIGITAL_GAINS = 0x262
UNITY_DIGITAL_GAINS = 0x1010
SHUTTER_DELAY = 0x00C


class ExposureControlUnavailable(RuntimeError):
    pass


def sensor_gain(value):
    """Decode MT9M112 Table 15: analog bits 7/8, digital bits 12/13/14."""
    multipliers = sum(bool(value & (1 << bit)) for bit in (7, 8, 12, 13, 14))
    return (value & 0x7F) / 32 * (2 ** multipliers)


def analog_gain(value):
    """Encode gain without digital amplification; retain colour balance ratios."""
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("RGB camera returned invalid colour gain")
    value = min(max(value, 1 / 32), 127 / 8)
    multiplier, flags = (1, 0) if value <= 127 / 32 else (2, 0x80)
    if value > 127 / 16:
        multiplier, flags = 4, 0x180
    return flags | min(127, max(1, round(value * 32 / multiplier)))


class RGBExposureControl:
    def __init__(self, driver, device):
        self.driver, self.device = driver, device  # Keep the USB context alive.
        self._library = self._pointer = None
        self._parameters = None
        self._expected_gains = None
        if all(callable(getattr(driver, name, None)) for name in
               ("set_flag", "set_exposure", "get_exposure")):
            self._set_flag = lambda flag, value: driver.set_flag(device, flag, value)
            self._set_exposure = lambda value: driver.set_exposure(device, value)
            self._get_exposure = lambda: driver.get_exposure(device)
        else:
            self._load_native()
            try:
                set_flag = self._library.freenect_set_flag
                set_exposure = self._library.freenect_set_exposure
                get_exposure = self._library.freenect_get_exposure
            except AttributeError as exc:
                raise ExposureControlUnavailable("Installed libfreenect lacks the RGB exposure API") from exc
            set_flag.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
            set_exposure.argtypes = (ctypes.c_void_p, ctypes.c_int)
            get_exposure.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
            for function in (set_flag, set_exposure, get_exposure):
                function.restype = ctypes.c_int
            self._set_flag = lambda flag, value: set_flag(self._pointer, flag, value)
            self._set_exposure = lambda value: set_exposure(self._pointer, value)

            def read():
                value = ctypes.c_int()
                self._check(get_exposure(self._pointer, ctypes.byref(value)), "read shutter time")
                return value.value

            self._get_exposure = read

    def _load_native(self):
        if self._library is not None:
            return
        device_type = getattr(self.driver, "DevPtr", None)
        path = getattr(self.driver, "__file__", None)
        if device_type is None or type(self.device) is not device_type or not path:
            raise ExposureControlUnavailable("These freenect bindings do not expose RGB controls")
        match = re.fullmatch(r"<Dev Pointer (0x[0-9a-fA-F]+)>", repr(self.device))
        if match is None or int(match[1], 16) == 0:
            raise ExposureControlUnavailable("Cannot access this freenect device's RGB controls")
        try:
            self._library = ctypes.CDLL(path)
        except OSError as exc:
            raise ExposureControlUnavailable("Cannot load the RGB control API") from exc
        self._pointer = ctypes.c_void_p(int(match[1], 16))

    def _load_registers(self):
        if all(callable(getattr(self.driver, name, None)) for name in
               ("read_cmos_register", "write_cmos_register")):
            self._read_register = lambda reg: self.driver.read_cmos_register(self.device, reg)
            self._write_register = lambda reg, value: self.driver.write_cmos_register(self.device, reg, value)
            return
        self._load_native()
        try:
            read = self._library.read_cmos_register
            write = self._library.write_cmos_register
        except AttributeError as exc:
            raise ExposureControlUnavailable(
                "Manual exposure needs freenect bindings with RGB CMOS register access"
            ) from exc
        read.argtypes, read.restype = (ctypes.c_void_p, ctypes.c_uint16), ctypes.c_uint16
        write.argtypes, write.restype = (ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16), ctypes.c_int
        self._read_register = lambda reg: read(self._pointer, reg)
        self._write_register = lambda reg, value: write(self._pointer, reg, value)

    @staticmethod
    def _check(result, action):
        if result is None or result < 0:
            raise RuntimeError(f"Cannot {action} on the RGB camera")

    def _read(self, reg):
        value = self._read_register(reg)
        if type(value) is not int or not 0 <= value < 0xFFFF:
            raise RuntimeError(f"Cannot read RGB camera register 0x{reg:03x}")
        return value

    def _write(self, reg, value):
        self._check(self._write_register(reg, value), f"write RGB register 0x{reg:03x}")

    def apply(self, mode, shutter_speed, gain=1):
        validate_rgb_exposure(mode, shutter_speed, "rgb_high_res", gain)
        if mode == "manual":
            self._load_registers()
            if self._read(CHIP_VERSION) != MT9M112_VERSION:
                raise ExposureControlUnavailable("Manual gain is unsupported on this RGB sensor")
        automatic = int(mode == "auto")
        for flag, action in ((AUTO_EXPOSURE, "set exposure mode"),
                             (AUTO_FLICKER, "set flicker compensation"),
                             (AUTO_WHITE_BALANCE, "set white balance mode")):
            self._check(self._set_flag(flag, automatic), action)
        self._parameters = (mode, shutter_speed, gain)
        if automatic:
            self._expected_gains = None
            return None
        # AWB supplies colour ratios, but must be frozen before changing gain.
        previous = [self._read(reg) for reg in CHANNEL_GAINS]
        reference = sensor_gain(previous[0])
        if reference <= 0:
            raise RuntimeError("RGB camera returned invalid white balance gain")
        self._expected_gains = [analog_gain(gain * sensor_gain(value) / reference)
                                for value in previous]
        for reg, previous_value, value in zip(CHANNEL_GAINS, previous, self._expected_gains):
            self._write(reg, (previous_value & ~GAIN_MASK) | value)
        # Automatic exposure may leave ISP digital amplification and fine timing.
        self._write(DIGITAL_GAINS, UNITY_DIGITAL_GAINS)
        self._write(SHUTTER_DELAY, 0)
        self._check(self._set_exposure(1_000_000 // shutter_speed), "set shutter time")
        return self.verify()["rgb_exposure_us"]

    def verify(self):
        """Read controls again after frames advance; pending readback alone is insufficient."""
        mode, shutter_speed, gain = self._parameters
        metadata = {"rgb_exposure_mode": mode, "rgb_exposure_controls": True,
                    "rgb_shutter_speed": None, "rgb_exposure_us": None, "rgb_gain": None}
        if mode == "auto":
            return metadata
        if self._read(MODE_CONTROL) & AUTOMATIC_MASK:
            raise RuntimeError("RGB camera restored automatic exposure controls")
        if (any(self._read(reg) & GAIN_MASK != expected
                for reg, expected in zip(CHANNEL_GAINS, self._expected_gains))
                or self._read(DIGITAL_GAINS) != UNITY_DIGITAL_GAINS
                or self._read(SHUTTER_DELAY) != 0):
            raise RuntimeError("RGB camera did not retain the requested sensitivity")
        actual_us = self._get_exposure()
        requested_us = 1_000_000 // shutter_speed
        # libfreenect rounds down to whole sensor rows (54.21 microseconds).
        if type(actual_us) is not int or not max(1, requested_us - 55) <= actual_us <= requested_us:
            raise RuntimeError("RGB camera did not confirm the requested shutter time")
        metadata.update(rgb_shutter_speed=shutter_speed, rgb_exposure_us=actual_us, rgb_gain=gain)
        return metadata


def apply_rgb_exposure(driver, device, mode, shutter_speed, gain=1):
    """One-shot API for diagnostics; capture additionally waits for new frames."""
    try:
        controls = RGBExposureControl(driver, device)
    except ExposureControlUnavailable:
        if mode == "manual":
            raise
        return {"rgb_exposure_mode": "auto", "rgb_exposure_controls": False}
    controls.apply(mode, shutter_speed, gain)
    return controls.verify()
