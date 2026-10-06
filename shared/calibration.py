"""Prepare pinhole RGB-D from calibrated native streams or metric datasets."""

from functools import lru_cache

import cv2
import numpy as np

from .depth import prepare_depth
from .native import kernels


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
    return np.ascontiguousarray(rgb), prepare_metric_depth(depth, settings)


def _rectified_depth(depth, settings):
    camera = settings.camera
    native = settings.sensor_calibration is not None
    metric = raw_depth_to_mm(depth, settings.sensor_calibration) if native else depth
    if native or any(camera.distortion):
        x, y = rectification_maps(camera)
        metric = cv2.remap(
            metric, x, y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
    if not native and camera.depth_scale != 1:
        metric = np.clip(
            np.rint(metric.astype(np.float64) * camera.depth_scale), 0, 65535
        )
    return metric


def prepare_metric_depth(depth, settings):
    """Identical reconstruction depth without projecting an unused RGB image.

    Allocation planning and depth-only evaluation need no color registration,
    RGB occlusion buffer, or bilinear color sampling. Keep the conversion and
    nearest-neighbour rectification shared with the full RGB-D path.
    """
    return prepare_depth(
        np.rint(_rectified_depth(depth, settings)).astype(np.uint16), settings
    )


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
    # projectPoints also allocates a 2N x 15 Jacobian that dense RGB-D sampling
    # never uses. Evaluate the same five-coefficient lens model directly.
    z = points[:, 2]
    inverse_z = np.ones_like(z)
    np.divide(1.0, z, out=inverse_z, where=z != 0)
    x, y = (points[:, :2] * inverse_z[:, None]).T
    k1, k2, p1, p2, k3 = rgb_camera.distortion
    xx, yy, xy = x * x, y * y, x * y
    r2 = xx + yy
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    pixels = np.column_stack(
        (
            (x * radial + 2 * p1 * xy + p2 * (r2 + 2 * xx)) * rgb_camera.fx
            + rgb_camera.cx,
            (y * radial + p1 * (r2 + 2 * yy) + 2 * p2 * xy) * rgb_camera.fy
            + rgb_camera.cy,
        )
    )
    return pixels.reshape(*shape, 2), points[:, 2].reshape(shape)


def prepare_native_rgbd(rgb, raw, settings):
    """Keep geometry in the rectified depth grid; sample full native RGB by pose.

    The depth principal point already includes the IR/depth translation. No
    board-boundary nuisance term is an optical correction. Nearest remapping
    preserves discontinuities; conversion/rounding and correction happen once.
    """
    calibration = settings.sensor_calibration
    metric = _rectified_depth(raw, settings)
    depth = prepare_depth(np.rint(metric).astype(np.uint16), settings)
    native = kernels()
    if native is None:
        map_x, map_y, visible = _color_maps_numpy(metric, depth, settings, rgb.shape)
    else:
        c = settings.rgb_camera
        # Keep BLAS's rotation accumulation order: a last-bit difference can
        # cross an OpenCV interpolation bin at a half-pixel boundary.
        points = pinhole_rays(settings.camera) * metric[..., None]
        points_rgb = (
            points.reshape(-1, 3) @ np.asarray(calibration.rotation).T
            + calibration.translation_mm
        ).reshape(*metric.shape, 3)
        map_x, map_y, visible = native.project_native(
            points_rgb,
            depth,
            np.array([c.fx, c.fy, c.cx, c.cy, *c.distortion], dtype=np.float64),
            rgb.shape[0],
            rgb.shape[1],
        )
    colors = cv2.remap(
        rgb,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    colors[~visible] = 0
    return np.ascontiguousarray(colors), depth


def _color_maps_numpy(metric, depth, settings, rgb_shape):
    """Reference projection/occlusion, sharing OpenCV color sampling with C++."""
    points = pinhole_rays(settings.camera) * metric[..., None]
    pixels, z = project_rgb(points, settings.sensor_calibration, settings.rgb_camera)
    visible = (
        (depth > 0)
        & (z > 0)
        & (pixels[..., 0] >= 0)
        & (pixels[..., 0] < rgb_shape[1] - 1)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < rgb_shape[0] - 1)
    )
    # RGB occlusion check: a farther depth point projecting onto the same RGB
    # pixel must not borrow the foreground's colour across a parallax boundary.
    px = np.clip(np.rint(pixels[..., 0]), 0, rgb_shape[1] - 1).astype(int)
    py = np.clip(np.rint(pixels[..., 1]), 0, rgb_shape[0] - 1).astype(int)
    nearest = np.full(rgb_shape[:2], np.inf)
    np.minimum.at(nearest, (py[visible], px[visible]), z[visible])
    visible &= z <= nearest[py, px] + np.maximum(15, z * 0.01)
    return pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32), visible
