"""Portable proofs for opt-in, session-scoped capture upload retries."""

import hashlib
import json
import struct

import numpy as np


MAX_STREAM_ID_BYTES = 128


def validate_upload_receipt(metadata):
    """Validate upload ordering metadata without granting retry authority."""
    receipt = metadata.get("upload_receipt")
    if not isinstance(receipt, dict) or set(receipt) != {"stream_id", "sequence", "expected_index"}:
        raise ValueError("upload_receipt requires stream_id, sequence and expected_index")
    stream = receipt["stream_id"]
    if not isinstance(stream, str) or not stream or len(stream.encode("utf-8")) > MAX_STREAM_ID_BYTES:
        raise ValueError("upload_receipt stream_id must be a nonempty bounded string")
    for field in ("sequence", "expected_index"):
        if type(receipt[field]) is not int or receipt[field] < 0:
            raise ValueError(f"upload_receipt {field} must be a nonnegative integer")
    return dict(receipt)


def capture_digest(rgb, depth, metadata):
    """Hash native pixels and all original JSON metadata, independent of layout."""
    metadata = dict(metadata)
    if "rgb_shape" in metadata:
        shape = metadata.pop("rgb_shape")
        if (not isinstance(shape, (list, tuple)) or any(type(size) is not int for size in shape)
                or tuple(shape) != tuple(np.shape(rgb))):
            raise ValueError("rgb_shape must match the native RGB pixels")
        # RGB3/RGB4 add this transport-derived field during packing. Shape is
        # already hashed with the pixels; its presence cannot change a receipt.
    digest = hashlib.sha256(b"kinect-capture-receipt-v1\0")

    def part(value):
        digest.update(struct.pack(">Q", len(value)))
        digest.update(value)

    part(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for array in (rgb, depth):
        array = np.asarray(array)
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        part(json.dumps([list(array.shape), array.dtype.str], separators=(",", ":")).encode("ascii"))
        part(memoryview(array).cast("B"))
    return digest.hexdigest()


def receipt_for(rgb, depth, metadata):
    """Return the wire acknowledgement proof for one original capture."""
    return {**validate_upload_receipt(metadata), "sha256": capture_digest(rgb, depth, metadata)}
