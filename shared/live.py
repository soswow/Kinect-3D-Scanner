"""Bounded, lossless float32 geometry for negotiated live WebSocket feedback."""

import base64
import json

import numpy as np

from .config import LIVE_MAX_POINTS

GEOMETRY_ENCODING = "xyzrgb-f32le"


def encode_live_message(message, *, packed_geometry=False):
    """Keep ordinary JSON lists for subscribers that did not opt in."""
    message = dict(message)
    if message.get("type") == "live":
        points = np.asarray(message.pop("points", []), dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(message.pop("colors", []), dtype=np.float32).reshape(-1, 3)
        if len(points) > LIVE_MAX_POINTS or len(colors) != len(points):
            raise ValueError("Invalid live geometry size")
        if packed_geometry:
            values = np.concatenate((points, colors), axis=1).astype("<f4", copy=False)
            message["geometry"] = {
                "encoding": GEOMETRY_ENCODING,
                "count": len(points),
                "data": base64.b64encode(values.tobytes()).decode("ascii"),
            }
        else:
            message.update(points=points.tolist(), colors=colors.tolist())
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


def decode_live_geometry(message):
    """Decode on the client's listener thread before delivering to the GUI."""
    if "geometry" not in message:
        return message  # Legacy server.
    geometry = message["geometry"]
    if not isinstance(geometry, dict) or geometry.get("encoding") != GEOMETRY_ENCODING:
        raise ValueError("Unsupported live geometry encoding")
    count = geometry.get("count")
    if type(count) is not int or not 0 <= count <= LIVE_MAX_POINTS:
        raise ValueError("Invalid live point count")
    data = geometry.get("data")
    if not isinstance(data, str) or len(data) != count * 32:
        raise ValueError("Invalid live geometry length")
    raw = base64.b64decode(data, validate=True)
    if len(raw) != count * 24:
        raise ValueError("Invalid decoded live geometry length")
    values = np.frombuffer(raw, dtype="<f4").reshape(count, 6)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite live geometry")
    result = dict(message)
    result.pop("geometry")
    result.update(points=values[:, :3], colors=values[:, 3:])
    return result
