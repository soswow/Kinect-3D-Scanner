"""Separate, unproved CUDA FP64 6x6 LDLT/Euler micro-prototype.

This is not imported by ResidentICP or the scanner. It ports pinned Eigen
algorithm semantics while scalar summation/libdevice trig can change roundoff.
Only fresh original-Eigen CPU shadows can establish the declared tolerance.
No nearest-neighbor, objective, convergence or full-Finish authority is implied.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

MAX_BATCH = 4096
BYTES_PER_SYSTEM = 844  # Inputs 288+48+128, outputs 4+48+128+128+24+48.
OPTIONS = ("--fmad=false",)
POLICY = "separate-pinned-eigen-lower-ldlt-quaternion-rzyx-device-v1"
SHADER = Path(__file__).with_suffix(".cu")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DeviceLDLT:
    """One scalar GPU thread per system, explicit selected device/null stream.

    launch() operates entirely on caller-owned resident input/output buffers;
    no host transfers, result decisions, synchronization or implicit CPU retry.
    Outputs with status!=0 are rejected, never applied to an ICP trajectory.
    Native execution damage propagates. This narrow prototype has no authority
    to bypass original Eigen or any per-query CPU ambiguity resolution.
    """

    def __init__(self, *, device=0):
        if type(device) is not int or device < 0:
            raise ValueError("Require explicit nonnegative CUDA device ID")
        import cupy as cp
        import numpy as np
        self.cp, self.np, self.device = cp, np, device
        source = SHADER.read_text(encoding="utf-8")
        started = time.perf_counter()
        with cp.cuda.Device(device), cp.cuda.Stream.null:
            self.kernel = cp.RawKernel(source, "device_ldlt_solve", options=OPTIONS)
            self.kernel.compile()
            cp.cuda.Stream.null.synchronize()
        properties = cp.cuda.runtime.getDeviceProperties(device)
        name = properties["name"]
        self.provenance = {"policy": POLICY, "status": "source-prepared; fresh numerical proof unexecuted",
            "adapter_sha256": file_hash(__file__), "shader_sha256": file_hash(SHADER),
            "kernel_sha256": hashlib.sha256(source.encode()).hexdigest(), "options": list(OPTIONS),
            "device": f"CUDA:{device}", "device_name": name.decode() if isinstance(name, bytes) else str(name),
            "cupy": cp.__version__, "numpy": np.__version__,
            "driver_version": cp.cuda.runtime.driverGetVersion(), "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "setup_and_compile_s": time.perf_counter()-started,
            "maximum_batch": MAX_BATCH, "input_output_bytes_per_system": BYTES_PER_SYSTEM,
            "arithmetic": "Explicit binary64 RN operations/no FMA. Pinned Eigen Lower LDLT pivot/permutation/positivity/1e-12 guard, diagonal pseudo-inverse DBL_MIN, quaternion Rz*Ry*Rx. Scalar evaluator/libdevice trig and pose product order are new and must be CPU-shadowed.",
            "status_codes": {"0": "accepted micro-solve", "2": "nonfinite A or gradient",
                "3": "factorization/nonpositive rejection", "4": "original pivot conditioning guard",
                "5": "nonfinite solution", "6": "nonfinite incremental Euler matrix",
                "8": "nonfinite previous-pose prototype input", "9": "nonfinite composed pose"},
            "authority": "No original Open3D binary, ResidentICP, graph/convergence/NN/pose acceptance, full-scan speed or production authority"}

    def _array(self, value, tail, count=None):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64
                or value.ndim != len(tail)+1 or tuple(value.shape[1:]) != tuple(tail)
                or not 0 < value.shape[0] <= MAX_BATCH or not value.flags.c_contiguous
                or value.device.id != self.device or count is not None and value.shape[0] != count):
            raise ValueError("Require selected-device contiguous bounded resident FP64 batch")
        return value.shape[0]

    def allocate(self, count):
        if type(count) is not int or not 0 < count <= MAX_BATCH:
            raise ValueError("Batch is outside the explicit micro-prototype cap")
        cp = self.cp
        with cp.cuda.Device(self.device), cp.cuda.Stream.null:
            return {"status": cp.empty(count, cp.int32), "step": cp.empty((count, 6), cp.float64),
                "update": cp.empty((count, 4, 4), cp.float64), "composed": cp.empty((count, 4, 4), cp.float64),
                "pivots": cp.empty((count, 6), cp.int32), "diagonal": cp.empty((count, 6), cp.float64)}

    def launch(self, matrices, gradients, previous, output):
        count = self._array(matrices, (6, 6))
        self._array(gradients, (6,), count)
        self._array(previous, (4, 4), count)
        cp, np = self.cp, self.np
        if set(output) != {"status", "step", "update", "composed", "pivots", "diagonal"}:
            raise ValueError("Require exact owned micro-prototype output fields")
        for name, shape, dtype in (("status", (count,), cp.int32), ("step", (count, 6), cp.float64),
                ("update", (count, 4, 4), cp.float64), ("composed", (count, 4, 4), cp.float64),
                ("pivots", (count, 6), cp.int32), ("diagonal", (count, 6), cp.float64)):
            value = output[name]
            if (not isinstance(value, cp.ndarray) or value.dtype != dtype or value.shape != shape
                    or not value.flags.c_contiguous or value.device.id != self.device):
                raise ValueError("Malformed resident micro-prototype output "+name)
        # Disallow any pointer overlap: a failing update must never overwrite
        # original input buffers, nor may diagnostics alias each other.
        values = [matrices, gradients, previous, *output.values()]
        ranges = sorted((value.data.ptr, value.data.ptr+value.nbytes) for value in values)
        if any(end > next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:])):
            raise ValueError("Micro-prototype inputs/outputs must have disjoint storage")
        with cp.cuda.Device(self.device), cp.cuda.Stream.null:
            self.kernel(((count+63)//64,), (64,), (matrices, gradients, previous, np.int32(count),
                output["status"], output["step"], output["update"], output["composed"],
                output["pivots"], output["diagonal"]))
        return output

    def synchronize(self):
        with self.cp.cuda.Device(self.device), self.cp.cuda.Stream.null:
            self.cp.cuda.Stream.null.synchronize()

    def close(self):
        self.synchronize()
