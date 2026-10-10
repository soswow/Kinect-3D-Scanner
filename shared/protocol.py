"""Bounded RGB-D serialization, compatible with original frame/batch packets.

V2 adds metadata; V3 carries native RGB dimensions (VGA or 1280x1024).
V4 adds lossless horizontal prediction and depth byte-plane separation.
Depth stays uint16 640x480; session settings define raw disparity or metric units.
"""

import json
import struct
import zlib

import numpy as np

_HEADER = struct.Struct(">I")
_BATCH_MAGIC = 0x42415448
_V2_MAGIC = b"RGB2"
_V3_MAGIC = b"RGB3"
_V4_MAGIC = b"RGB4"
PREDICTED_FRAME_ENCODING = "rgbd-sub-zlib-v1"
_MAX_BATCH = 100
_MAX_FRAME_BYTES = 6_000_000


def validate_arrays(rgb, depth):
    if rgb.shape not in ((480, 640, 3), (1024, 1280, 3)) or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 VGA or 1280x1024")
    if depth.shape != (480, 640) or depth.dtype != np.uint16:
        raise ValueError("Depth must be uint16 480x640")


def pack_frame(rgb, depth, metadata=None, *, compression_level=1, spatial_prediction=False):
    validate_arrays(rgb, depth)
    if compression_level not in (0, 1):
        raise ValueError("Use zlib level 0 for loopback or level 1 for network capture")
    if spatial_prediction:
        color_array = rgb.copy()
        color_array[:, 1:] = rgb[:, 1:] - rgb[:, :-1]
        depth_array = np.array(depth, dtype="<u2", copy=True)
        depth_array[:, 1:] = depth[:, 1:] - depth[:, :-1]
        depth_array = depth_array.view(np.uint8).reshape(480, 640, 2).transpose(2, 0, 1)
    else:
        color_array = np.ascontiguousarray(rgb)
        depth_array = np.ascontiguousarray(depth, dtype="<u2")
    color = zlib.compress(color_array.tobytes(), level=compression_level)
    metric = zlib.compress(depth_array.tobytes(), level=compression_level)
    payload = _HEADER.pack(len(color)) + color + metric
    high_res = rgb.shape == (1024, 1280, 3)
    if metadata is not None or high_res or spatial_prediction:
        metadata = dict(metadata or {})
        if high_res or spatial_prediction:
            metadata["rgb_shape"] = list(rgb.shape)
        header = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
        if len(header) > 4096:
            raise ValueError("Frame metadata too large")
        payload = (
            (_V4_MAGIC if spatial_prediction else _V3_MAGIC if high_res else _V2_MAGIC)
            + _HEADER.pack(len(header))
            + header
            + payload
        )
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
    shape = (480, 640, 3)
    predicted = data[:4] == _V4_MAGIC
    version3 = data[:4] in (_V3_MAGIC, _V4_MAGIC)
    if data[:4] in (_V2_MAGIC, _V3_MAGIC, _V4_MAGIC):
        length = _HEADER.unpack_from(data, 4)[0]
        if length > 4096 or 8 + length + 4 > len(data):
            raise ValueError("Invalid metadata length")
        metadata = json.loads(data[8 : 8 + length])
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be an object")
        if version3:
            shape = tuple(metadata.get("rgb_shape", ()))
            if shape not in ((480, 640, 3), (1024, 1280, 3)):
                raise ValueError("Unsupported RGB dimensions")
        data = data[8 + length :]
    rgb_len = _HEADER.unpack_from(data)[0]
    if not 0 < rgb_len < len(data) - 4:
        raise ValueError("Invalid RGB payload length")
    rgb = np.frombuffer(
        _decompress(data[4 : 4 + rgb_len], int(np.prod(shape))), dtype=np.uint8
    ).reshape(shape)
    raw_depth = _decompress(data[4 + rgb_len :], 480 * 640 * 2)
    if predicted:
        rgb = np.cumsum(rgb, axis=1, dtype=np.uint8)
        depth = (np.frombuffer(raw_depth, dtype=np.uint8).reshape(2, 480, 640)
                 .transpose(1, 2, 0).copy().view("<u2").reshape(480, 640))
        depth = np.cumsum(depth, axis=1, dtype=np.uint16)
    else:
        depth = np.frombuffer(raw_depth, dtype="<u2").reshape(480, 640).astype(np.uint16, copy=False)
    return rgb, depth, metadata


def unpack_frame(data):
    rgb, depth, _ = unpack_frame_with_metadata(data)
    return rgb, depth


def pack_frames(frames, *, compression_level=1, spatial_prediction=False):
    if not 1 <= len(frames) <= _MAX_BATCH:
        raise ValueError("Batch requires 1–100 frames")
    packed = [pack_frame(*frame, compression_level=compression_level,
                         spatial_prediction=spatial_prediction) for frame in frames]
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
