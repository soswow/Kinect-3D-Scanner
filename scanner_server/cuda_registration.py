"""CUDA execution for measured-cloud registration, scoped to one engine call.

CPU callers retain legacy registration. Geometry, iteration budgets, robust
losses and verification thresholds are shared with the legacy implementation.
"""

import os
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace

import numpy as np
import open3d as o3d

_device = ContextVar("registration_device", default=None)


def selection(device):
    # Full-cloud CUDA verification was slower and changed recovered geometry in
    # the recorded-session study. Keep the established CPU path by default.
    mode = os.environ.get("KINECT_CUDA_REGISTRATION", "cpu").lower()
    if mode not in ("auto", "cpu", "tensor"):
        raise ValueError("KINECT_CUDA_REGISTRATION must be auto, cpu, or tensor")
    cuda = str(device).startswith("CUDA")
    if mode == "tensor" and not cuda:
        raise RuntimeError("CUDA registration requested for a CPU engine")
    return {"requested": mode, "implementation": "tensor" if cuda and mode != "cpu" else "legacy"}


def registration_scope(action):
    @wraps(action)
    def wrapped(engine, *args, **kwargs):
        status = engine.backend.get("geometric_verification", {})
        token = _device.set(engine.device if status.get("implementation") == "tensor" else None)
        try:
            return action(engine, *args, **kwargs)
        finally:
            _device.reset(token)
    return wrapped


def as_legacy_result(result):
    matches = result.correspondence_set.cpu().numpy().reshape(-1)
    ids = np.flatnonzero(matches >= 0)
    return SimpleNamespace(transformation=result.transformation.cpu().numpy(),
                           fitness=float(result.fitness), inlier_rmse=float(result.inlier_rmse),
                           correspondence_set=np.column_stack((ids, matches[ids])))


def match(source, target, initial):
    """Return None for legacy selection, otherwise a synchronized tensor result."""
    device = _device.get()
    if device is None:
        return None
    core, reg = o3d.core, o3d.t.pipelines.registration
    a = o3d.t.geometry.PointCloud.from_legacy(source, dtype=core.float32, device=device)
    b = o3d.t.geometry.PointCloud.from_legacy(target, dtype=core.float32, device=device)
    estimator = reg.TransformationEstimationPointToPlane(
        reg.robust_kernel.RobustKernel(reg.robust_kernel.RobustKernelMethod.HuberLoss, 0.01))
    pose = core.Tensor(np.asarray(initial), dtype=core.float64)
    for distance, iterations in ((0.12, 40), (0.06, 30), (0.03, 20)):
        result = reg.icp(a, b, distance, pose, estimator,
                         reg.ICPConvergenceCriteria(max_iteration=iterations))
        pose = result.transformation
    return as_legacy_result(result)
