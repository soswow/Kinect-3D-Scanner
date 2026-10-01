"""Capture timestamps use the Kinect's wrapping uint32 millisecond clock."""


def timestamp_delta_ms(a, b):
    return ((int(a) - int(b) + (1 << 31)) % (1 << 32)) - (1 << 31)
