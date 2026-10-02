"""Experimental confidence-weighted projective TSDF using Open3D tensors.

CPU/CUDA voxel math stays on the selected device. Confidence is computed on CPU.
The ordinary optimized integration remains the default.
"""

import numpy as np
from open3d import core

from shared.confidence import depth_confidence


def integrate_weighted(engine, volume, blocks, rgb, depth, extrinsic):
    device = engine.device
    confidence = depth_confidence(depth, engine.settings.camera)
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
    return {
        "mean_observation_weight": float(confidence[depth > 0].mean())
        if np.any(depth)
        else 0,
        "rejected_confidence_fraction": float(np.mean(confidence[depth > 0] == 0))
        if np.any(depth)
        else 0,
    }
