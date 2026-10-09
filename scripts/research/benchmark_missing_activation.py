"""Allocated CPU/tensor/fused CUDA causal proof for weighted missing activation.

Small synthetic inputs only: compare the original caller on a native VBG with
the same caller on the private proxy. Every original frustum is still fused.
Key-mapped TSDF/weight/color bytes must match within each selected backend.
No whole-session, cross-backend bit equivalence or scan speed authority.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.profile_session import file_hash, source_hash
from scripts.process_metrics import finish_cuda_worker, gpu_info, peak_rss_bytes
from scripts.research.final_missing_activation import (
    MissingOnlyGrid, WeightedActivationSourceContract, MissingActivationContractError)

KIND = "weighted-final-missing-activation-causal-synthetic-v2"
FROZEN_CORE = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
ARTIFACTS = (
    "scripts/research/final_missing_activation.py", "scripts/research/benchmark_missing_activation.py",
    "tests/test_final_missing_activation.py", "scripts/research/final_allocation_scope.py",
    "scripts/profile_session.py", "scripts/process_metrics.py", "scanner_server/engine.py",
    "scanner_server/weighted_fusion.py", "scanner_server/cuda_fusion.py",
    "scanner_server/weighted_fusion.cu", "shared/native.py", "native/kernels.cpp", "native/weighted_fusion.h")
THREAD_ENV = {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def artifacts():
    return {name: file_hash(ROOT/name) for name in ARTIFACTS}


def native_binding(np, o3d, cp):
    binaries = {}
    families = ("numpy.", "open3d.", "cupy.", "cupy_backends.", "_kinect_native")
    for name, module in list(sys.modules.items()):
        path_text = getattr(module, "__file__", None)
        if name.startswith(families) and path_text:
            path = Path(path_text).resolve()
            if path.suffix.lower() in (".pyd", ".so", ".dll"):
                binaries[str(path)] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    if not any("open3d" in path.lower() for path in binaries):
        raise RuntimeError("Cannot bind the actual loaded Open3D native binary")
    with cp.cuda.Device(0), cp.cuda.Stream.null:
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        cuda = {"device": 0, "name": device_name,
                "driver_version": cp.cuda.runtime.driverGetVersion(),
                "runtime_version": cp.cuda.runtime.runtimeGetVersion()}
    return {"python": sys.version, "executable": sys.executable,
            "versions": {"numpy": np.__version__, "open3d": o3d.__version__, "cupy": cp.__version__},
            "binaries": binaries, "cuda": cuda, "gpu": gpu_info(),
            "open3d_threads": o3d.utility.get_max_threads(),
            "environment": {key: os.environ.get(key) for key in THREAD_ENV}}


def array_record(value):
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(), "bytes": value.nbytes}


def key_mapped_attributes(volume, keys, np, core):
    requested = core.Tensor(keys, dtype=core.int32, device=volume.hashmap().device)
    indices, found = volume.hashmap().find(requested)
    if not bool(found.cpu().numpy().all()):
        raise RuntimeError("Final key-mapped attribute snapshot has missing original keys")
    slots = indices.cpu().numpy().astype(np.int64)
    capacity = int(volume.hashmap().capacity())
    return {name: np.ascontiguousarray(volume.attribute(name).cpu().numpy().reshape(
        (capacity, 16**3, channels))[slots]) for name, channels in (("tsdf", 1), ("weight", 1), ("color", 3))}


def input_arrays(np):
    rows, columns = np.indices((24, 32))
    depth = np.full((24, 32), 1000, np.uint16)
    depth[:2, :2] = 0
    confidence = np.ones((24, 32), np.float32)
    confidence[4:7, 4:7] = 0
    confidence[10:13, 10:13] = .5
    rgb = np.ascontiguousarray(np.stack(((columns*7)%256, (rows*11)%256,
                                         (columns+rows*3)%256), axis=-1), dtype=np.uint8)
    extrinsic = np.eye(4, dtype=np.float64)
    return rgb, depth, confidence, extrinsic


def compare_backend(name, *, np, o3d, cp, weighted, cuda, kernel, contract, pair):
    device = o3d.core.Device("CPU:0" if name == "cpu-original" else "CUDA:0")
    core = o3d.core
    camera = SimpleNamespace(width=32, height=24, fx=40., fy=40., cx=15.5, cy=11.5)
    engine = SimpleNamespace(device=device, settings=SimpleNamespace(camera=camera, confidence_fusion=True),
                             max_depth_m=3., sdf_trunc=.08, voxel_size=.005)
    keys = ((-1, -1, 12), (0, -1, 12), (-1, 0, 12), (0, 0, 12))
    batches = (keys[:3], (keys[1], keys[3], keys[0]), keys)
    rgb, depth, confidence, extrinsic = input_arrays(np)
    pair.update({"backend": name, "device": str(device), "complete": False,
            "inputs": {key: array_record(value) for key, value in
                       (("rgb", rgb), ("depth", depth), ("confidence", confidence), ("extrinsic", extrinsic))},
            "ordered_frustum_keys": [[list(key) for key in batch] for batch in batches],
            "confidence_scope": "Same supplied float32 weights, including zeros and fractional values; confidence preparation is outside this activation causal proof."})

    def create():
        # Exact original attribute names/types/channels/resolution/voxel policy.
        return o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(core.float32,)*3, attr_channels=((1,), (1,), (3,)),
            voxel_size=engine.voxel_size, block_resolution=16, block_count=4, device=device)

    def integrate(volume, blocks):
        if name == "cpu-original":
            weighted._integrate_cpu(engine, volume, blocks, rgb, depth, extrinsic, confidence)
        elif name == "cuda-tensor":
            weighted._integrate_tensor(engine, volume, blocks, rgb, depth, extrinsic, confidence)
        else:
            cuda.integrate(engine, volume, blocks, rgb, depth, extrinsic, confidence, cp, kernel)
        if str(device).startswith("CUDA"):
            core.cuda.synchronize(device)

    snapshots = {}
    proxy = None
    for mode in ("original-full", "missing-only"):
        started = time.perf_counter()
        volume = create()
        setup_wall = time.perf_counter()-started
        if mode == "missing-only":
            setup_started = time.perf_counter()
            proxy = MissingOnlyGrid(volume, core=core, device=device, physical_capacity=4,
                logical_limit=4, source_contract=contract,
                configuration=lambda: (engine.settings.confidence_fusion, str(engine.device)))
            setup_wall += time.perf_counter()-setup_started
            actual = proxy
        else:
            actual = volume
        row = {"mode": mode, "initial_capacity": int(volume.hashmap().capacity()),
               "setup_wall_s": setup_wall, "frames": []}
        pair[mode] = row
        for index, batch in enumerate(batches):
            blocks = core.Tensor(batch, dtype=core.int32, device=device)
            frame_started = time.perf_counter()
            integrate(actual, blocks)
            row["frames"].append({"index": index, "frustum_rows": len(batch),
                "integration_wall_s": time.perf_counter()-frame_started,
                "capacity": int(volume.hashmap().capacity()), "active_keys": int(volume.hashmap().size())})
        row["final_capacity"], row["final_blocks"] = int(volume.hashmap().capacity()), int(volume.hashmap().size())
        if mode == "missing-only":
            row["key_proof"] = proxy.hashmap().verify_final(4)
            row["activation"] = proxy.hashmap().report()
        snapshots[mode] = key_mapped_attributes(volume, keys, np, core)
        row["attribute_snapshots"] = {name: array_record(value) for name, value in snapshots[mode].items()}
        row["all_in_wall_s"] = time.perf_counter()-started
        row["integration_wall_s"] = sum(frame["integration_wall_s"] for frame in row["frames"])
    pair["inputs_after"] = {key: array_record(value) for key, value in
                            (("rgb", rgb), ("depth", depth), ("confidence", confidence), ("extrinsic", extrinsic))}
    pair["inputs_unchanged"] = pair["inputs_after"] == pair["inputs"]
    if not pair["inputs_unchanged"]:
        raise RuntimeError("Original synthetic RGB/depth/confidence/extrinsic changed during fusion")
    comparison = {}
    pair["comparison"] = comparison
    for attr in ("tsdf", "weight", "color"):
        control, selected = snapshots["original-full"][attr], snapshots["missing-only"][attr]
        errors = int(np.count_nonzero(control.view(np.uint32) != selected.view(np.uint32)))
        comparison[attr] = {"elements": control.size, "bit_mismatches": errors}
        if errors:
            raise RuntimeError(f"{name}: original/proxy key-mapped {attr} bits differ ({errors})")
    if not np.count_nonzero(snapshots["missing-only"]["weight"] > 0):
        raise RuntimeError("Synthetic fusion produced no measured voxel updates")
    if (pair["original-full"]["final_capacity"] <= 4 or pair["missing-only"]["final_capacity"] != 4
            or pair["missing-only"]["final_blocks"] != 4):
        raise RuntimeError("Actual selected hashmap did not demonstrate the controlled capacity distinction")
    contract.unchanged()
    pair.update(complete=True, exact_key_mapped_attribute_bits=True, comparison=comparison,
                nonzero_weight_voxels=int(np.count_nonzero(snapshots["missing-only"]["weight"] > 0)))
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_allocated:
        parser.error("Require an explicitly allocated exclusive native/GPU slot")
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT/"benchmark-output")
    except ValueError:
        parser.error("Keep private proof reports under ignored benchmark-output")
    if args.output.exists():
        parser.error("Refuse overwriting immutable proof output")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle server before an allocated native/GPU experiment")
    report = {"kind": KIND, "status": "running", "start_utc": now(), "pairs": [],
        "performance_claim": False, "whole_session_quality_proven": False,
        "measurement": "Original weighted CPU/tensor/fused calls, unchanged full per-frame input integration. Key-mapped attribute bits within each backend, selected exact physical capacity and original pessimistic growth are the gates. Small synthetic component walls include proxy checks/copies/synchronization; no scan speed claim.",
        "source_sha256": source_hash(), "artifacts_sha256": artifacts()}
    if report["source_sha256"] != FROZEN_CORE:
        parser.error("Current frozen core differs; requires a new explicit experiment")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    before_env = {key: os.environ.get(key) for key in THREAD_ENV}
    cp = o3d = binding = None
    failure = None
    try:
        save()
        os.environ.update(THREAD_ENV)
        import numpy as np
        import open3d as o3d
        import cupy as cp
        from scanner_server import weighted_fusion as weighted, cuda_fusion as cuda
        from shared.native import kernels
        o3d.utility.set_max_threads(20)
        if not o3d.core.cuda.is_available():
            raise RuntimeError("Actual CUDA Open3D support is required")
        if kernels() is None:
            raise RuntimeError("Original native weighted CPU kernel is required for this policy proof")
        setup_started = time.perf_counter()
        fused_cp, kernel, unavailable = cuda._kernel(0)
        if fused_cp is not cp or kernel is None or unavailable is not None:
            raise RuntimeError("Actual original fused CUDA kernel is unavailable")
        report["fused_setup_compile_s"] = time.perf_counter()-setup_started
        # Load the actual DLPack transport extension before runtime closure;
        # this is setup, not a voxel update or a comparative timed warmup.
        with cp.cuda.Device(0), cp.cuda.Stream.null:
            transport = o3d.core.Tensor([0.], dtype=o3d.core.float32, device=o3d.core.Device("CUDA:0"))
            borrowed = cp.from_dlpack(transport.to_dlpack())
            cp.asarray(np.zeros(1, np.float32))
            cp.cuda.Stream.null.synchronize()
        o3d.core.cuda.synchronize(o3d.core.Device("CUDA:0"))
        del borrowed, transport
        contract = WeightedActivationSourceContract()
        binding = native_binding(np, o3d, cp)
        report["runtime_binding"] = binding
        for backend in ("cpu-original", "cuda-tensor", "cuda-fused"):
            pair = {}
            report["pairs"].append(pair)
            compare_backend(backend, np=np, o3d=o3d, cp=cp,
                weighted=weighted, cuda=cuda, kernel=kernel, contract=contract, pair=pair)
            save()
        report["runtime_binding_after"] = native_binding(np, o3d, cp)
        if report["runtime_binding_after"] != binding:
            raise RuntimeError("Actual native runtime/thread/hardware bindings changed")
        contract.unchanged()
    except BaseException as error:
        failure = error
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        cleanup = []
        if o3d is not None:
            try:
                o3d.core.cuda.synchronize(o3d.core.Device("CUDA:0"))
            except BaseException as error:
                cleanup.append({"action": "final device synchronization", "type": type(error).__name__, "message": str(error)})
                if failure is None:
                    failure = error
        try:
            report["source_sha256_after"] = source_hash()
            report["artifacts_sha256_after"] = artifacts()
            report["source_unchanged"] = (report["source_sha256_after"] == report["source_sha256"]
                and report["artifacts_sha256_after"] == report["artifacts_sha256"])
            if not report["source_unchanged"]:
                raise RuntimeError("Experiment sources changed before closure")
            report["peak_process_rss_bytes"] = peak_rss_bytes()
        except BaseException as error:
            cleanup.append({"action": "final provenance/RSS", "type": type(error).__name__, "message": str(error)})
            if failure is None:
                failure = error
        for key, value in before_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report.update(cleanup_failures=cleanup, environment_restored=all(os.environ.get(key) == value
            for key, value in before_env.items()), end_utc=now())
        passed = (failure is None and not cleanup and len(report["pairs"]) == 3
                  and all(pair.get("complete") is True for pair in report["pairs"])
                  and report.get("source_unchanged") is True and report["environment_restored"])
        report["status"] = "passed" if passed else "failed"
        try:
            save()
        except BaseException as write_error:
            if failure is not None:
                failure.add_note(f"Final proof write also failed: {write_error}")
                raise failure from write_error
            raise
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "backends": [row.get("backend", "setup-failed") for row in report["pairs"]]}), flush=True)
    if failure is not None:
        raise failure
    finish_cuda_worker()


if __name__ == "__main__":
    main()
