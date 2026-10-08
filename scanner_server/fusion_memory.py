"""Automatic fusion allocation from measured extent and available RAM/VRAM.

Voxel attributes occupy 16**3 * (1 + 1 + 3) float32 values per block.
Planning reserves extra workspace and leaves 20% of currently available memory
unused. These are conservative estimates, not a guarantee against native OOM.
"""

import os
import math
import subprocess

import psutil

MIB = 1024**2
BYTES_PER_BLOCK = 16**3 * 5 * 4


def available_gpu_bytes(device):
    """Use the selected CUDA context; support installations without CuPy."""
    device_id = device.get_id()
    try:
        import cupy as cp

        with cp.cuda.Device(device_id):
            # Open3D cannot borrow allocations held in CuPy's unused pool.
            cp.get_default_memory_pool().free_all_blocks()
            free, _ = cp.cuda.runtime.memGetInfo()
            return int(free)
    except Exception:
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            physical = visible.split(",")[device_id].strip() if visible is not None else str(device_id)
            result = subprocess.run(
                ["nvidia-smi", "-i", physical, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=3,
            )
            return int(result.stdout.strip()) * MIB
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Cannot read available GPU memory; check the NVIDIA driver or CuPy installation") from exc


def available_memory(device):
    pools = {"RAM": int(psutil.virtual_memory().available)}
    if str(device).startswith("CUDA"):
        pools["GPU memory"] = available_gpu_bytes(device)
    return pools


def plan_fusion(device, blocks, voxel_m):
    """Reject a memory shortage before allocating the candidate voxel grid."""
    pools = available_memory(device)
    # Allow small coordinate-boundary differences between validated CPU/CUDA
    # preparation paths without making scan coverage a fixed user budget.
    allocated = max(1, math.ceil(blocks * 1.02))
    attributes = allocated * BYTES_PER_BLOCK
    workspace = max(256 * MIB, attributes // 2)
    cuda = str(device).startswith("CUDA")
    requirements = {"RAM": workspace if cuda else attributes + workspace}
    if cuda:
        requirements["GPU memory"] = attributes + workspace
    report = {
        "allocation": "automatic",
        "required_blocks": blocks,
        "allocated_blocks": allocated,
        "attribute_budget_mib": attributes / MIB,
        "estimated_workspace_mib": workspace / MIB,
        "available_memory_mib": {name: size / MIB for name, size in pools.items()},
    }
    for name, required in requirements.items():
        available = pools[name]
        if required > available * .8:
            raise ValueError(
                f"Not enough {name} to reconstruct at {voxel_m * 1000:g} mm: "
                f"about {required / MIB:.0f} MiB needed including working memory, "
                f"{available / MIB:.0f} MiB available (20% reserved). "
                "Choose a coarser Final voxel size or free memory on the processing machine. "
                "The captured scan and previous model are retained."
            )
    return report
