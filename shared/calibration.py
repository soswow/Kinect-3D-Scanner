"""Prepare pinhole RGB-D from calibrated native streams or metric datasets."""

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

    Native streams reconstruct on the rectified depth grid and project into
    RGB through the measured pose and lens. Registered metric datasets share a
    lens remap. Depth always uses nearest-neighbour sampling; K stays unchanged.
    """
    if settings.sensor_calibration is not None:
        return prepare_native_rgbd(rgb, depth, settings)
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


@lru_cache(maxsize=8)
def raw_depth_lut(calibration):
    """Continuous axial millimetres; 2047 is invalid, but raw zero is valid."""
    raw = np.arange(2048, dtype=np.float64)
    denominator = calibration.a_per_code * raw + calibration.b
    with np.errstate(divide="ignore", invalid="ignore"):
        metric = calibration.scale * 1000 / denominator
    valid = (raw != 2047) & (denominator > 0) & (metric > 0) & (metric < 10000)
    metric[~valid] = 0
    metric.flags.writeable = False
    return metric


def raw_depth_to_mm(raw, calibration):
    raw = np.asarray(raw)
    if raw.dtype != np.uint16:
        raise ValueError("Raw depth must be uint16 disparity codes")
    result = raw_depth_lut(calibration)[np.minimum(raw, 2047)].copy()
    result[raw > 2047] = 0
    return result


@lru_cache(maxsize=8)
def pinhole_rays(camera):
    y, x = np.indices((camera.height, camera.width), dtype=np.float64)
    rays = np.stack(
        ((x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, np.ones_like(x)),
        axis=-1,
    )
    rays.flags.writeable = False
    return rays


def project_rgb(points_ir_mm, calibration, rgb_camera):
    """Transform IR-frame points then project through the native RGB lens."""
    shape = points_ir_mm.shape[:-1]
    points = (
        points_ir_mm.reshape(-1, 3) @ np.asarray(calibration.rotation).T
        + calibration.translation_mm
    )
    pixels, _ = cv2.projectPoints(
        points,
        np.zeros(3),
        np.zeros(3),
        camera_matrix(rgb_camera),
        np.array(rgb_camera.distortion),
    )
    return pixels.reshape(*shape, 2), points[:, 2].reshape(shape)


def prepare_native_rgbd(rgb, raw, settings):
    """Keep geometry in the rectified depth grid; sample full native RGB by pose.

    The depth principal point already includes the IR/depth translation. No
    board-boundary nuisance term is an optical correction. Nearest remapping
    preserves discontinuities; conversion/rounding and correction happen once.
    """
    calibration = settings.sensor_calibration
    metric = raw_depth_to_mm(raw, calibration)
    x, y = rectification_maps(settings.camera)
    metric = cv2.remap(metric, x, y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    depth = prepare_depth(np.rint(metric).astype(np.uint16), settings)
    points = pinhole_rays(settings.camera) * metric[..., None]
    pixels, z = project_rgb(points, calibration, settings.rgb_camera)
    visible = (
        (depth > 0)
        & (z > 0)
        & (pixels[..., 0] >= 0)
        & (pixels[..., 0] < rgb.shape[1] - 1)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < rgb.shape[0] - 1)
    )
    # RGB occlusion check: a farther depth point projecting onto the same RGB
    # pixel must not borrow the foreground's colour across a parallax boundary.
    px = np.clip(np.rint(pixels[..., 0]), 0, rgb.shape[1] - 1).astype(int)
    py = np.clip(np.rint(pixels[..., 1]), 0, rgb.shape[0] - 1).astype(int)
    nearest = np.full(rgb.shape[:2], np.inf)
    np.minimum.at(nearest, (py[visible], px[visible]), z[visible])
    visible &= z <= nearest[py, px] + np.maximum(15, z * 0.01)
    colors = cv2.remap(
        rgb,
        pixels[..., 0].astype(np.float32),
        pixels[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    colors[~visible] = 0
    return np.ascontiguousarray(colors), depth
