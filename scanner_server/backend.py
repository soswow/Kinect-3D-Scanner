"""Explicit compute selection; an unavailable requested CUDA device is an error."""

import os

import open3d as o3d

from shared.native import native_status


def select_backend(requested=None, tracking=None):
    requested = (requested or os.environ.get("KINECT_DEVICE", "auto")).lower()
    tracking = (tracking or os.environ.get("KINECT_TRACKING", "auto")).lower()
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("KINECT_DEVICE must be auto, cpu, or cuda")
    if tracking not in ("auto", "legacy", "tensor"):
        raise ValueError("KINECT_TRACKING must be auto, legacy, or tensor")
    available = bool(o3d.core.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError(
            "CUDA requested but unavailable; use a CUDA-enabled Open3D build and NVIDIA driver"
        )
    device = o3d.core.Device("CUDA:0" if available and requested != "cpu" else "CPU:0")
    mode = (
        ("tensor" if device.get_type() == o3d.core.Device.DeviceType.CUDA else "legacy")
        if tracking == "auto"
        else tracking
    )
    return device, {
        "requested": requested,
        "device": str(device),
        "cuda_available": available,
        "tracking": mode,
        "tracking_device": str(device) if mode == "tensor" else "CPU:0",
        "stage_devices": {
            "depth_filter": "CPU:0",
            "registration_cloud": "CPU:0",
            "tracking": str(device) if mode == "tensor" else "CPU:0",
            "fusion": str(device),
            "model_refresh": "hybrid"
            if available and str(device).startswith("CUDA")
            else "CPU:0",
            "final_refinement": "CPU:0",
            "texturing": "CPU:0",
        },
        "open3d_version": o3d.__version__,
        "native_kernels": native_status(),
        "fallback_reason": "CUDA unavailable; using CPU"
        if requested == "auto" and not available
        else None,
    }
