"""Kinect v1 exposure controls, used only inside the supervised camera process.

Upstream Python bindings omit the public exposure API. Resolve those functions
through the loaded extension (and thus its own libfreenect dependency). OpenKinect
DevPtr's repr exposes the native address; validate its exact type and format rather
than relying on Cython object memory layout or opening a second USB handle.
"""

import ctypes
import re

AUTO_EXPOSURE = 1 << 14
AUTO_FLICKER = 1 << 7
AUTO_WHITE_BALANCE = 1 << 1


class ExposureControlUnavailable(RuntimeError):
    pass


class RGBExposureControl:
    def __init__(self, driver, device):
        self.device = device  # Keep the Python handle and its context alive.
        if all(callable(getattr(driver, name, None)) for name in
               ("set_flag", "set_exposure", "get_exposure")):
            self._set_flag = lambda flag, value: driver.set_flag(device, flag, value)
            self._set_exposure = lambda value: driver.set_exposure(device, value)
            self._get_exposure = lambda: driver.get_exposure(device)
            return
        device_type = getattr(driver, "DevPtr", None)
        path = getattr(driver, "__file__", None)
        if device_type is None or type(device) is not device_type or not path:
            raise ExposureControlUnavailable("These freenect bindings do not expose RGB exposure controls")
        match = re.fullmatch(r"<Dev Pointer (0x[0-9a-fA-F]+)>", repr(device))
        if match is None or int(match[1], 16) == 0:
            raise ExposureControlUnavailable("Cannot access this freenect device's exposure controls")
        try:
            self._library = ctypes.CDLL(path)
            set_flag = self._library.freenect_set_flag
            set_exposure = self._library.freenect_set_exposure
            get_exposure = self._library.freenect_get_exposure
        except (OSError, AttributeError) as exc:
            raise ExposureControlUnavailable("Installed libfreenect lacks the RGB exposure API") from exc
        set_flag.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        set_exposure.argtypes = (ctypes.c_void_p, ctypes.c_int)
        get_exposure.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
        for function in (set_flag, set_exposure, get_exposure):
            function.restype = ctypes.c_int
        pointer = ctypes.c_void_p(int(match[1], 16))
        self._set_flag = lambda flag, value: set_flag(pointer, flag, value)
        self._set_exposure = lambda value: set_exposure(pointer, value)

        def read():
            value = ctypes.c_int()
            self._check(get_exposure(pointer, ctypes.byref(value)), "read shutter time")
            return value.value

        self._get_exposure = read

    @staticmethod
    def _check(result, action):
        if result is None or result < 0:
            raise RuntimeError(f"Cannot {action} on the RGB camera")

    def apply(self, mode, shutter_speed):
        automatic = int(mode == "auto")
        self._check(self._set_flag(AUTO_EXPOSURE, automatic), "set exposure mode")
        # Flicker avoidance can change shutter time; disable it for a fixed shutter.
        self._check(self._set_flag(AUTO_FLICKER, automatic), "set flicker compensation")
        self._check(self._set_flag(AUTO_WHITE_BALANCE, 1), "enable automatic white balance")
        if automatic:
            return None
        requested_us = 1_000_000 // shutter_speed
        self._check(self._set_exposure(requested_us), "set shutter time")
        actual_us = self._get_exposure()
        if type(actual_us) is not int or not 0 < actual_us <= requested_us:
            raise RuntimeError("RGB camera did not confirm the requested shutter time")
        return actual_us


def apply_rgb_exposure(driver, device, mode, shutter_speed):
    """Apply controls after RGB starts; older drivers retain default auto mode."""
    try:
        controls = RGBExposureControl(driver, device)
    except ExposureControlUnavailable:
        if mode == "manual":
            raise
        return {"rgb_exposure_mode": "auto", "rgb_exposure_controls": False}
    actual_us = controls.apply(mode, shutter_speed)
    return {
        "rgb_exposure_mode": mode,
        "rgb_exposure_controls": True,
        "rgb_shutter_speed": shutter_speed if mode == "manual" else None,
        "rgb_exposure_us": actual_us,
    }
