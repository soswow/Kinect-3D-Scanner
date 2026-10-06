"""Kinect v1 timestamps use a wrapping uint32 60 MHz device clock.

See libfreenect's OpenNI2-FreenectDriver/src/VideoStream.hpp. Wrap the
difference in ticks before converting; the device clock wraps every ~72 s.
"""

from collections import deque

RGB_MODE_FPS = {"rgb_high_res": 10, "rgb_low_res": 30}
RGB_GAIN_CHOICES = (1, 2, 4, 8)
RGB_DEPTH_ASSISTANCE_LIMIT_MS = 20
RGB_DEPTH_CAPTURE_LIMIT_MS = 50


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


class RGBDepthPairer:
    """Match each RGB frame to the nearest unused depth device timestamp.

    RGB callbacks can arrive after several newer depth frames. Retain eight
    depth frames (~267 ms at 30 fps) instead of overwriting the closer frame.
    If RGB leads depth, wait for depth to reach its timestamp before choosing
    between the surrounding observations. Backpressure keeps only the newest
    pending RGB frame; neither memory nor output queues grow with a slow GUI.

    These are libfreenect packet-end timestamps, not measured exposure times.
    """

    def __init__(self):
        self.depths = deque(maxlen=8)
        self.rgb = None

    def clear(self):
        self.depths.clear()
        self.rgb = None

    def add_depth(self, array, stamp):
        self.depths.append((array.copy(), int(stamp)))

    def add_rgb(self, array, stamp):
        self.rgb = (array.copy(), int(stamp))

    def pop_pair(self):
        if self.rgb is None or not self.depths:
            return None
        rgb, rgb_stamp = self.rgb
        if timestamp_delta_ms(self.depths[-1][1], rgb_stamp) < 0:
            return None
        index = min(
            range(len(self.depths)),
            key=lambda i: abs(timestamp_delta_ms(rgb_stamp, self.depths[i][1])),
        )
        depth, depth_stamp = self.depths[index]
        self.rgb = None
        if abs(timestamp_delta_ms(rgb_stamp, depth_stamp)) > RGB_DEPTH_CAPTURE_LIMIT_MS:
            return None
        # Consume the selected observation and older depths, retaining newer
        # depths for the next RGB. Never publish the same depth frame twice.
        for _ in range(index + 1):
            self.depths.popleft()
        return rgb, depth, rgb_stamp, depth_stamp
