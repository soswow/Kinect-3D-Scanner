"""Kinect v1 timestamps use a wrapping uint32 60 MHz device clock.

See libfreenect's OpenNI2-FreenectDriver/src/VideoStream.hpp. Wrap the
difference in ticks before converting; the device clock wraps every ~72 s.
"""

RGB_MODE_FPS = {"rgb_high_res": 10, "rgb_low_res": 30}


def timestamp_delta_ms(a, b):
    ticks = ((int(a) - int(b) + (1 << 31)) % (1 << 32)) - (1 << 31)
    return ticks / 60_000.0
