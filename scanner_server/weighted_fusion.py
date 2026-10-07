"""Experimental confidence-weighted projective TSDF using Open3D tensors.

CPU voxel math uses shared NumPy buffers, with optional fused C++ updates;
CUDA math stays on device tensors.
Confidence is computed on CPU.
The ordinary optimized integration remains the default.

For a static scalar with independent Gaussian measurements, this weighted
average has the same mean update as a Kalman filter with zero process noise:
gain = incoming / (old + incoming). Here weights are relative engineering
scores, not calibrated precisions; repeated, correlated frames must not be
interpreted as independent samples or as a physical posterior variance.
"""

import numpy as np
from open3d import core

from shared.confidence import depth_confidence
from shared.native import kernels


def integrate_weighted(engine, volume, blocks, rgb, depth, extrinsic):
    confidence = depth_confidence(depth, engine.settings.camera)
    if str(engine.device) == "CPU:0":
        _integrate_cpu(engine, volume, blocks, rgb, depth, extrinsic, confidence)
    else:
        _integrate_tensor(engine, volume, blocks, rgb, depth, extrinsic, confidence)
    valid = depth > 0
    return {
        "mean_observation_weight": float(confidence[valid].mean())
        if np.any(valid)
        else 0,
        "rejected_confidence_fraction": float(np.mean(confidence[valid] == 0))
        if np.any(valid)
        else 0,
    }


def _integrate_cpu(engine, volume, blocks, rgb, depth, extrinsic, confidence):
    """Update shared CPU attribute buffers in bounded NumPy batches.

    Open3D's general tensor indexing launches many small parallel operations on
    CPU. NumPy views avoid those launches and copies without changing the
    projective TSDF equations or the confidence and truncation gates.
    """
    hashmap = volume.hashmap()
    hashmap.activate(blocks)
    buffers, found = hashmap.find(blocks)
    buffers = buffers[found]
    tsdf = volume.attribute("tsdf").numpy().reshape(-1, 1)
    weight = volume.attribute("weight").numpy().reshape(-1, 1)
    color = volume.attribute("color").numpy().reshape(-1, 3)
    transform = np.asarray(extrinsic, dtype=np.float32)
    depth_m = depth.astype(np.float32) / 1000
    camera = engine.settings.camera
    native = kernels()
    for start in range(0, len(buffers), 128):
        points, flat = volume.voxel_coordinates_and_flattened_indices(
            buffers[start : start + 128]
        )
        xyz = points.numpy() @ transform[:3, :3].T + transform[:3, 3]
        if native is not None:
            native.integrate_weighted_cpu(
                xyz,
                flat.numpy().reshape(-1),
                depth_m,
                confidence,
                rgb,
                tsdf,
                weight,
                color,
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                engine.max_depth_m,
                engine.sdf_trunc,
            )
            continue
        z = xyz[:, 2]
        safe = np.maximum(z, np.float32(1e-6))
        # Open3D rounds half away from zero; np.rint uses ties to even.
        projected = xyz[:, :2] * [np.float32(camera.fx), np.float32(camera.fy)]
        projected /= safe[:, None]
        projected += [np.float32(camera.cx), np.float32(camera.cy)]
        pixels = np.copysign(np.floor(np.abs(projected) + 0.5), projected).astype(
            np.int64
        )
        u, v = pixels.T
        inside = (
            (z > 0) & (u >= 0) & (v >= 0) & (u < camera.width) & (v < camera.height)
        )
        u, v, z = u[inside], v[inside], z[inside]
        ids = flat.numpy().reshape(-1)[inside]
        observed, incoming = depth_m[v, u], confidence[v, u]
        sdf = observed - z
        valid = (
            (observed > 0)
            & (observed <= engine.max_depth_m)
            & (incoming > 0)
            & (sdf >= -engine.sdf_trunc)
        )
        ids = ids[valid]
        if not len(ids):
            continue
        distances = np.minimum(sdf[valid, None] / engine.sdf_trunc, 1)
        contribution = incoming[valid, None]
        old = weight[ids]
        total = old + contribution
        tsdf[ids] = (tsdf[ids] * old + distances * contribution) / total
        color[ids] = (color[ids] * old + rgb[v[valid], u[valid]] * contribution) / total
        weight[ids] = total


def _integrate_tensor(engine, volume, blocks, rgb, depth, extrinsic, confidence):
    device = engine.device
    weights_image = core.Tensor(confidence, device=device)
    depth_image = core.Tensor(depth.astype(np.float32) / 1000, device=device)
    color_image = core.Tensor(rgb.astype(np.float32), device=device)
    transform = core.Tensor(extrinsic, dtype=core.float32, device=device)
    camera = engine.settings.camera
    hashmap = volume.hashmap()
    hashmap.activate(blocks)
    buffers, found = hashmap.find(blocks)
    buffers = buffers[found]
    tsdf = volume.attribute("tsdf").reshape((-1, 1))
    weight = volume.attribute("weight").reshape((-1, 1))
    color = volume.attribute("color").reshape((-1, 3))
    for start in range(0, len(buffers), 128):
        points, flat = volume.voxel_coordinates_and_flattened_indices(
            buffers[start : start + 128]
        )
        flat = flat.reshape((-1,))
        xyz = points @ transform[:3, :3].T() + transform[:3, 3]
        z = xyz[:, 2]
        safe = z.clone()
        safe[safe < 1e-6] = 1e-6
        u = (xyz[:, 0] * camera.fx / safe + camera.cx).round().to(core.int64)
        v = (xyz[:, 1] * camera.fy / safe + camera.cy).round().to(core.int64)
        inside = (
            (z > 0) & (u >= 0) & (v >= 0) & (u < camera.width) & (v < camera.height)
        )
        u, v, z, flat = u[inside], v[inside], z[inside], flat[inside]
        observed = depth_image[v, u]
        incoming = weights_image[v, u]
        sdf = observed - z
        valid = (
            (observed > 0)
            & (observed <= engine.max_depth_m)
            & (incoming > 0)
            & (sdf >= -engine.sdf_trunc)
        )
        ids = flat[valid]
        if len(ids) == 0:
            continue
        distances = sdf[valid].reshape((-1, 1)) / engine.sdf_trunc
        distances[distances > 1] = 1
        contribution = incoming[valid].reshape((-1, 1))
        old = weight[ids]
        total = old + contribution
        tsdf[ids] = (tsdf[ids] * old + distances * contribution) / total
        color[ids] = (
            color[ids] * old + color_image[v[valid], u[valid]] * contribution
        ) / total
        weight[ids] = total
