"""Apply measured calibration equally to RGB and registered millimetre depth."""

from functools import lru_cache

import cv2
import numpy as np

from .depth import prepare_depth


def camera_matrix(camera):
    return np.array(
        [[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]],
        dtype=np.float64,
    )


@lru_cache(maxsize=8)
def rectification_maps(camera):
    matrix = camera_matrix(camera)
    return cv2.initUndistortRectifyMap(
        matrix,
        np.array(camera.distortion),
        None,
        matrix,
        (camera.width, camera.height),
        cv2.CV_32FC1,
    )


def prepare_rgbd(rgb, depth, settings):
    """Return pinhole RGB/depth without modifying recorded sensor observations.

    Both images use the same remap into K's pixel grid. Depth is nearest-neighbour
    sampled so invalid regions and discontinuities never become interpolated
    measurements. RGB uses bilinear sampling. K remains unchanged.
    """
    camera = settings.camera
    if any(camera.distortion):
        x, y = rectification_maps(camera)
        rgb = cv2.remap(rgb, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        depth = cv2.remap(
            depth, x, y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
    if camera.depth_scale != 1:
        depth = np.clip(
            np.rint(depth.astype(np.float64) * camera.depth_scale), 0, 65535
        ).astype(np.uint16)
    return np.ascontiguousarray(rgb), prepare_depth(depth, settings)
