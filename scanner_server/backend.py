"""Explicit compute selection; an unavailable requested CUDA device is an error."""

import os

import open3d as o3d

from shared.native import native_status
from .cuda_registration import selection as registration_selection
from .cuda_odometry import selection as odometry_selection
from .cuda_matching import selection as matching_selection
from .cuda_input import selection as input_selection
from .cuda_confidence import selection as confidence_selection


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
    verification = registration_selection(device)
    matching = matching_selection(device)
    odometry = odometry_selection(device)
    input_preparation = input_selection(device)
    confidence = confidence_selection(device)
    return device, {
        "requested": requested,
        "device": str(device),
        "cuda_available": available,
        "tracking": mode,
        "tracking_device": str(device) if mode == "tensor" else "CPU:0",
        "stage_devices": {
            "depth_filter": "CPU:0",
            "depth_confidence": "CPU:0",
            "registration_cloud": "CPU:0",
            "tracking": str(device) if mode == "tensor" else "CPU:0",
            "fusion": str(device),
            "model_refresh": "hybrid"
            if available and str(device).startswith("CUDA")
            else "CPU:0",
            "final_refinement": "hybrid" if verification["implementation"] == "tensor" or matching["implementation"] == "cuda" else "CPU:0",
            "geometric_verification": str(device) if verification["implementation"] == "tensor" else "CPU:0",
            "visual_retrieval": str(device) if matching["implementation"] == "cuda" else "CPU:0",
            "rgbd_odometry": str(device) if odometry != "off" else "CPU:0",
            "texturing": "CPU:0",
        },
        "open3d_version": o3d.__version__,
        "native_kernels": native_status(),
        "geometric_verification": verification,
        "projective_odometry": odometry,
        "cuda_input": input_preparation,
        "depth_confidence": confidence,
        "descriptor_matching": matching,
        "fallback_reason": "CUDA unavailable; using CPU"
        if requested == "auto" and not available
        else None,
    }
