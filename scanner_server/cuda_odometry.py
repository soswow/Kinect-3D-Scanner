"""Experimental projective RGB-D refinement of measured appearance proposals.

The engine still verifies feature identities and reciprocal raw-cloud overlap.
These estimates never authorize fusion without those independent checks.
"""

import os
from collections import OrderedDict

import numpy as np
import open3d as o3d


def selection(device):
    mode = os.environ.get("KINECT_CUDA_ODOMETRY", "off").lower()
    if mode not in ("off", "hybrid", "point-to-plane"):
        raise ValueError("KINECT_CUDA_ODOMETRY must be off, hybrid, or point-to-plane")
    if mode != "off" and not str(device).startswith("CUDA"):
        raise RuntimeError("Projective CUDA odometry requested for a CPU engine")
    return mode


class RGBDOdometry:
    def __init__(self, device, capacity=40):
        self.device = device
        self.capacity = capacity
        self.frames = OrderedDict()

    def image(self, frame):
        key = id(frame)
        cached = self.frames.get(key)
        if cached is None or cached[0] is not frame:
            cached = (frame, o3d.t.geometry.RGBDImage(
                o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(frame.color)).to(self.device)),
                o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(frame.depth)).to(self.device))))
            self.frames[key] = cached
            while len(self.frames) > self.capacity:
                self.frames.popitem(last=False)
        self.frames.move_to_end(key)
        return cached[1]

    def estimate(self, source, target, intrinsic, initial, depth_max, mode, iterations=(20, 10, 5)):
        odo = o3d.t.pipelines.odometry
        result = odo.rgbd_odometry_multi_scale(
            self.image(source), self.image(target), intrinsic,
            o3d.core.Tensor(np.asarray(initial), dtype=o3d.core.float64),
            depth_scale=1.0, depth_max=depth_max,
            criteria_list=[odo.OdometryConvergenceCriteria(n) for n in iterations],
            method=odo.Method.Hybrid if mode == "hybrid" else odo.Method.PointToPlane,
            params=odo.OdometryLossParams(depth_outlier_trunc=0.05, depth_huber_delta=0.025))
        pose = result.transformation.cpu().numpy()
        if not np.isfinite(pose).all() or not np.isfinite(result.fitness) or result.fitness <= 0:
            return None
        return pose
