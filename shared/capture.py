"""Kinect v1 timestamps use a wrapping uint32 60 MHz device clock.

See libfreenect's OpenNI2-FreenectDriver/src/VideoStream.hpp. Wrap the
difference in ticks before converting; the device clock wraps every ~72 s.
"""

RGB_MODE_FPS = {"rgb_high_res": 10, "rgb_low_res": 30}
RGB_GAIN_CHOICES = (1, 2, 4, 8)


def validate_rgb_exposure(mode, shutter_speed, rgb_mode, gain=1):
    if mode not in ("auto", "manual"):
        raise ValueError("RGB exposure must be auto or manual")
    if type(shutter_speed) is not int or not 10 <= shutter_speed <= 10000:
        raise ValueError("RGB shutter speed must be 1/10–1/10000 s")
    if rgb_mode not in RGB_MODE_FPS:
        raise ValueError("RGB mode must be rgb_high_res or rgb_low_res")
    if mode == "manual" and shutter_speed < RGB_MODE_FPS[rgb_mode]:
        raise ValueError("RGB exposure time cannot exceed the camera frame period")
    if type(gain) is not int or gain not in RGB_GAIN_CHOICES:
        raise ValueError("RGB gain must be 1, 2, 4, or 8")


def timestamp_delta_ms(a, b):
    ticks = ((int(a) - int(b) + (1 << 31)) % (1 << 32)) - (1 << 31)
    return ticks / 60_000.0
