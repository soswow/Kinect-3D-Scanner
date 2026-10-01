"""Conservative depth filtering shared by replay and reconstruction."""

import numpy as np


def prepare_depth(depth, settings):
    """Clip both tracking and fusion; remove isolated samples without filling holes.

    A pixel needs two neighbours within a depth-dependent tolerance. This drops
    flying pixels while retaining sharp boundaries and thin observed surfaces.
    """
    result = np.array(depth, dtype=np.uint16, copy=True)
    valid = (
        (result > 0)
        & (result >= settings.near_m * 1000)
        & (result <= settings.far_m * 1000)
    )
    if settings.roi is not None:
        x0, y0, x1, y1 = settings.roi
        region = np.zeros(result.shape, dtype=bool)
        region[y0:y1, x0:x1] = True
        valid &= region
    result[~valid] = 0
    if settings.filter_depth:
        signed = result.astype(np.int32)
        tolerance = np.maximum(15, signed * 0.01)
        padded = np.pad(signed, 1)
        count = np.zeros(result.shape, dtype=np.uint8)
        h, w = result.shape
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            neighbour = padded[dy : dy + h, dx : dx + w]
            count += (neighbour > 0) & (np.abs(neighbour - signed) <= tolerance)
        result[count < 2] = 0
    return np.ascontiguousarray(result)
