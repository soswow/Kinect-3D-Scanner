"""Optional fused confidence TSDF update on shared Open3D CUDA storage.

CuPy is optional. DLPack borrows the volume buffers without copying them.
Compilation happens before any voxel update; execution errors never retry an
already partly updated frame. Explicit ``fused`` requests fail on setup errors.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import open3d as o3d

from .fusion_allocation import activate_fusion_blocks

logger = logging.getLogger("scanner_server")


class FusionUpdateError(RuntimeError):
    """A CUDA volume update failed after partial mutation became possible."""


@lru_cache(maxsize=8)
def _kernel(device_id):
    try:
        import cupy as cp

        with cp.cuda.Device(device_id):
            kernel = cp.RawKernel(
                Path(__file__).with_name("weighted_fusion.cu").read_text(),
                "integrate_weighted", options=("--fmad=false",),
            )
            kernel.compile()
        return cp, kernel, None
    except Exception as exc:  # Optional setup only; update/launch errors propagate.
        logger.info("Fused CUDA fusion unavailable: %s", exc)
        return None, None, str(exc)


def selection(device):
    mode = os.environ.get("KINECT_CUDA_FUSION", "auto").lower()
    if mode not in ("auto", "tensor", "fused"):
        raise ValueError("KINECT_CUDA_FUSION must be auto, tensor, or fused")
    if mode == "tensor":
        return None, None, {"requested": mode, "implementation": "tensor", "fallback_reason": None}
    cp, kernel, reason = _kernel(device.get_id())
    if cp is None and mode == "fused":
        raise RuntimeError(f"Fused CUDA fusion requested but unavailable: {reason}")
    return cp, kernel, {
        "requested": mode, "implementation": "fused" if cp else "tensor",
        "fallback_reason": reason,
    }


def integrate(engine, volume, blocks, rgb, depth, extrinsic, confidence, cp, kernel):
    """One fused update per voxel, using the reference gates and float32 math."""
    device = engine.device
    # Raw kernels require the documented attribute layout. Check before writes.
    attributes = [volume.attribute(name) for name in ("tsdf", "weight", "color")]
    for attribute, channels in zip(attributes, (1, 1, 3)):
        if (attribute.dtype != o3d.core.float32 or attribute.device != device
                or attribute.shape[-1] != channels or not attribute.is_contiguous()):
            raise ValueError("Fused CUDA fusion requires contiguous float32 TSDF/weight/color")
    try:
        _integrate_validated(engine, volume, blocks, rgb, depth, extrinsic, confidence, cp, kernel)
    except Exception as exc:
        raise FusionUpdateError(f"CUDA voxel update failed: {type(exc).__name__}: {exc}") from exc


def _integrate_validated(engine, volume, blocks, rgb, depth, extrinsic, confidence, cp, kernel):
    camera = engine.settings.camera
    device = engine.device
    hashmap = volume.hashmap()
    activate_fusion_blocks(engine, hashmap, blocks)
    buffers, found = hashmap.find(blocks)
    buffers = buffers[found]
    if not len(buffers):
        return
    # Activation may grow/reallocate the hash map and attribute storage.
    # Borrow only the current buffers after activation has completed.
    attributes = [volume.attribute(name) for name in ("tsdf", "weight", "color")]
    with cp.cuda.Device(device.get_id()), cp.cuda.Stream.null:
        # Open3D capsules have no stream negotiation. Complete their producers
        # before borrowing, and finish our consumer before returning the buffers.
        o3d.core.cuda.synchronize()
        tsdf, weight, color = [cp.from_dlpack(a.to_dlpack()) for a in attributes]
        depth_image = cp.asarray(depth.astype(np.float32) / 1000)
        confidence_image = cp.asarray(np.ascontiguousarray(confidence, dtype=np.float32))
        color_image = cp.asarray(np.ascontiguousarray(rgb, dtype=np.uint8))
        transform = cp.asarray(np.ascontiguousarray(extrinsic, dtype=np.float32))
        for start in range(0, len(buffers), 1024):
            points, flat = volume.voxel_coordinates_and_flattened_indices(buffers[start:start + 1024])
            o3d.core.cuda.synchronize()
            xyz = cp.from_dlpack(points.to_dlpack())
            ids = cp.from_dlpack(flat.to_dlpack())
            kernel(((len(points) + 255) // 256,), (256,), (
                xyz, ids, np.int64(len(points)), depth_image, confidence_image,
                color_image, transform, tsdf, weight, color,
                np.int32(camera.width), np.int32(camera.height),
                np.float32(camera.fx), np.float32(camera.fy),
                np.float32(camera.cx), np.float32(camera.cy),
                np.float32(engine.max_depth_m), np.float32(engine.sdf_trunc),
            ))
            cp.cuda.Stream.null.synchronize()
