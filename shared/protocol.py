"""Bounded RGB-D serialization, compatible with original frame/batch packets.

V2 adds JSON metadata before the original compressed uint8 RGB / uint16 mm
payload. Legacy unpack_frame still returns two arrays. Dimensions are 640x480.
"""

import json
import struct
import zlib

import numpy as np

_HEADER = struct.Struct(">I")
_BATCH_MAGIC = 0x42415448
_V2_MAGIC = b"RGB2"
_MAX_BATCH = 100
_MAX_FRAME_BYTES = 3_000_000


def validate_arrays(rgb, depth):
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 480x640x3")
    if depth.shape != (480, 640) or depth.dtype != np.uint16:
        raise ValueError("Depth must be uint16 480x640 millimetres")


def pack_frame(rgb, depth, metadata=None):
    validate_arrays(rgb, depth)
    color = zlib.compress(np.ascontiguousarray(rgb).tobytes(), level=1)
    metric = zlib.compress(np.ascontiguousarray(depth, dtype="<u2").tobytes(), level=1)
    payload = _HEADER.pack(len(color)) + color + metric
    if metadata is not None:
        header = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
        if len(header) > 4096:
            raise ValueError("Frame metadata too large")
        payload = _V2_MAGIC + _HEADER.pack(len(header)) + header + payload
    return payload


def _decompress(data, size):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(data, size + 1)
    if len(raw) != size or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid compressed image size or trailing data")
    return raw


def unpack_frame_with_metadata(data):
    if not 8 <= len(data) <= _MAX_FRAME_BYTES:
        raise ValueError("Invalid frame length")
    metadata = {}
    if data[:4] == _V2_MAGIC:
        length = _HEADER.unpack_from(data, 4)[0]
        if length > 4096 or 8 + length + 4 > len(data):
            raise ValueError("Invalid metadata length")
        metadata = json.loads(data[8 : 8 + length])
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be an object")
        data = data[8 + length :]
    rgb_len = _HEADER.unpack_from(data)[0]
    if not 0 < rgb_len < len(data) - 4:
        raise ValueError("Invalid RGB payload length")
    rgb = np.frombuffer(
        _decompress(data[4 : 4 + rgb_len], 480 * 640 * 3), dtype=np.uint8
    ).reshape(480, 640, 3)
    depth = (
        np.frombuffer(_decompress(data[4 + rgb_len :], 480 * 640 * 2), dtype="<u2")
        .reshape(480, 640)
        .astype(np.uint16, copy=False)
    )
    return rgb, depth, metadata


def unpack_frame(data):
    rgb, depth, _ = unpack_frame_with_metadata(data)
    return rgb, depth


def pack_frames(frames):
    if not 1 <= len(frames) <= _MAX_BATCH:
        raise ValueError("Batch requires 1–100 frames")
    packed = [pack_frame(*frame) for frame in frames]
    return b"".join(
        [
            _HEADER.pack(_BATCH_MAGIC),
            _HEADER.pack(len(packed)),
            *[_HEADER.pack(len(p)) for p in packed],
            *packed,
        ]
    )


def unpack_frames(data, with_metadata=False):
    if len(data) < 8 or _HEADER.unpack_from(data)[0] != _BATCH_MAGIC:
        raise ValueError("Invalid batch header")
    n = _HEADER.unpack_from(data, 4)[0]
    if not 1 <= n <= _MAX_BATCH or len(data) < 8 + 4 * n:
        raise ValueError("Invalid batch count")
    offset = 8 + 4 * n
    lengths = [_HEADER.unpack_from(data, 8 + 4 * i)[0] for i in range(n)]
    if any(
        length > _MAX_FRAME_BYTES or length < 8 for length in lengths
    ) or offset + sum(lengths) != len(data):
        raise ValueError("Invalid batch lengths")
    frames = []
    for length in lengths:
        frame = unpack_frame_with_metadata(data[offset : offset + length])
        frames.append(frame if with_metadata else frame[:2])
        offset += length
    return frames
