"""Allocated synthetic CPU-Eigen shadows for a separate GPU LDLT/Euler port.

This narrow experiment has no ICP/nearest/trajectory or whole-Finish authority.
No numerical imports occur before explicit allocation and CLI validation.
It preserves all inputs/raw outputs and status/tolerance failures for diagnosis.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash

ARTIFACTS = ("scripts/research/benchmark_device_ldlt.py", "scripts/research/research_device_ldlt.py",
    "scripts/research/research_device_ldlt.cu", "scripts/research/validate_resident_equation_fixture.py",
    "scripts/research/archive/research_resident_icp.py",
    "scripts/profile_session.py", "scripts/process_metrics.py", "scripts/tool_paths.py", "scripts/tool-catalog.json")
TOLERANCE = 1e-8


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def actual_runtime(np, cp, device):
    binaries = {}
    for name, module in list(sys.modules.items()):
        if name.startswith("numpy.") and name.endswith("._multiarray_umath"):
            path = Path(module.__file__).resolve()
            binaries[str(path)] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    if not binaries:
        raise RuntimeError("Cannot bind actual NumPy metric/pose-product binary")
    return {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "cupy": cp.__version__,
        "numpy_binary": binaries, "numpy_err_policy": np.geterr(),
        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "driver_version": cp.cuda.runtime.driverGetVersion(), "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "device": device.provenance["device"], "device_name": device.provenance["device_name"],
        "arithmetic_options": device.provenance["options"]}


def cases(np):
    """Regular SPD plus zero/rank/condition/pivot/nonfinite diagnostic systems."""
    rng = np.random.default_rng(271828)
    rows = []

    def add(name, matrix, gradient, pose=None, expected_extra_status=None):
        rows.append((name, np.asarray(matrix, np.float64), np.asarray(gradient, np.float64),
            np.eye(4) if pose is None else np.asarray(pose, np.float64), expected_extra_status))

    for k in range(48):
        basis = rng.normal(size=(6, 6))
        matrix = basis.T @ basis + np.eye(6)*.1
        scale = 2.**(k%25-12)
        desired = rng.normal(size=6)*.01
        gradient = -(matrix @ desired)
        pose = np.eye(4)
        angle = .15*(k-24)
        pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
        pose[:3, 3] = rng.normal(size=3)
        add("spd-pivot-scale-"+str(k), matrix*scale, gradient*scale, pose)
    for value in (0., 1e-14, np.nextafter(1e-12, 0.), 1e-12, np.nextafter(1e-12, np.inf), 1e-10):
        diagonal = np.asarray([1., .8, .5, .3, .1, value])
        for order in (np.arange(6), np.arange(6)[::-1], np.asarray([1, 3, 5, 0, 2, 4])):
            add("condition-"+value.hex()+"-"+str(order.tolist()), np.diag(diagonal[order]), np.zeros(6))
    add("zero", np.zeros((6, 6)), np.zeros(6))
    add("rank-one", np.ones((6, 6)), np.zeros(6))
    add("negative", -np.eye(6), np.zeros(6))
    add("indefinite", np.diag([1., 2., 3., -1., -2., -3.]), np.ones(6)*.01)
    matrix = np.zeros((6, 6));matrix[0, 1] = matrix[1, 0] = 1.
    add("zero-diagonal-nonzero-offdiagonal", matrix, np.zeros(6))
    for exponent in (-1070, -1022, -1000, -500, 500):
        matrix = np.eye(6)*np.ldexp(1., exponent)
        add("diagonal-power-"+str(exponent), matrix, np.zeros(6))
    for name, value in (("nan", np.nan), ("positive-inf", np.inf), ("negative-inf", -np.inf)):
        matrix = np.eye(6);matrix[0, 5] = value
        add("nonfinite-upper-A-"+name, matrix, np.zeros(6))
        gradient = np.zeros(6);gradient[4] = value
        add("nonfinite-gradient-"+name, np.eye(6), gradient)
    # Non-symmetric upper entries are finite but ignored by original Lower LDLT.
    matrix = np.diag(np.arange(1., 7.));matrix[np.triu_indices(6, 1)] = 123.45
    add("lower-triangle-semantics", matrix, np.ones(6)*.01)
    add("signed-zero", np.eye(6), np.asarray([0., -0., 0., -0., 0., -0.]))
    matrix = np.asarray([[2. if i == j else .1 for j in range(6)] for i in range(6)])
    add("mixed-equal-largest-diagonal-pivot-ties", matrix, np.asarray([.001, -.002, .003, -.004, .005, -.006]))
    for angles in ([.3, -.2, .1], [-3., 2., -1.], [np.pi, -np.pi, .5*np.pi]):
        desired = np.asarray([*angles, .01, -.02, .03])
        add("quaternion-angle-stress-"+str(angles), np.eye(6), -desired)
    matrix = np.zeros((6, 6));matrix[0, 0] = 1.;matrix[0, 1] = matrix[1, 0] = 1e308
    add("finite-factor-overflow", matrix, np.zeros(6))
    previous = np.eye(4);previous[0, 3] = 1.5e308
    add("finite-composition-overflow", np.eye(6), np.asarray([0., 0., 0., -1.5e308, 0., 0.]), previous, 9)
    for name, value in (("nan", np.nan), ("positive-inf", np.inf), ("negative-inf", -np.inf)):
        previous = np.eye(4);previous[0, 3] = value
        add("nonfinite-previous-prototype-input-"+name, np.eye(6), np.zeros(6), previous, 8)
    return rows


def difference_evidence(value):
    """A broken extreme result must remain a serializable numerical failure."""
    value = float(value)
    return {"absolute_delta": value if math.isfinite(value) else None,
            "nonfinite_difference": not math.isfinite(value)}


def absolute_difference(np, left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        return difference_evidence(np.max(np.abs(left - right)))


def actual_equations(np, cp, device, eigen, fixture, output_path):
    """Original observed systems only; each original DLL gold is rechecked first."""
    from scripts.research.research_device_ldlt import MAX_BATCH

    fixture.check_unchanged()
    with np.load(fixture.equation_path, allow_pickle=False) as archive:
        data = {name: np.ascontiguousarray(archive[name]) for name in
                ("matrices", "gradients", "cpu_update", "cpu_status")}
    fixture.check_unchanged()
    count = fixture.count
    if (data["matrices"].shape != (count, 6, 6) or data["gradients"].shape != (count, 6)
            or data["cpu_update"].shape != (count, 4, 4) or data["cpu_status"].shape != (count,)
            or any(data[name].dtype != np.float64 for name in ("matrices", "gradients", "cpu_update"))
            or data["cpu_status"].dtype != np.int32 or not np.equal(data["cpu_status"], 0).all()
            or not all(np.isfinite(data[name]).all() for name in ("matrices", "gradients", "cpu_update"))):
        raise RuntimeError("Observed original equation numeric arrays are outside the declared accepted domain")
    before = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in data.items()}
    fresh_status = np.empty(count, np.int32)
    fresh_update = np.empty((count, 4, 4), np.float64)
    started = time.perf_counter()
    for index in range(count):
        fresh_status[index] = eigen.library.resident_solve(data["matrices"][index].ctypes.data,
            data["gradients"][index].ctypes.data, fresh_update[index].ctypes.data)
    cpu_s = time.perf_counter() - started
    cpu_exact = bool(np.array_equal(fresh_status, data["cpu_status"])
                     and np.array_equal(fresh_update.view(np.uint64), data["cpu_update"].view(np.uint64)))
    fields = {"status": np.empty(count, np.int32), "step": np.empty((count, 6), np.float64),
        "update": np.empty((count, 4, 4), np.float64), "composed": np.empty((count, 4, 4), np.float64),
        "pivots": np.empty((count, 6), np.int32), "diagonal": np.empty((count, 6), np.float64)}
    # Preserve a failed fresh original-DLL check before any GPU comparison.
    if not cpu_exact:
        np.savez_compressed(output_path, fresh_cpu_status=fresh_status, fresh_cpu_update=fresh_update)
        return {"passed": False, "count": count, "fresh_cpu_gold_bitwise_equal": False,
            "fresh_original_dll_s": cpu_s, "gpu_executed": False,
            "results": {"path": str(output_path), "sha256": file_hash(output_path)},
            "scope": "Fresh original DLL output differs from recorded original bits; GPU proof is blocked"}
    launched = []
    for start in range(0, count, MAX_BATCH):
        end = min(count, start + MAX_BATCH)
        with cp.cuda.Device(device.device), cp.cuda.Stream.null:
            matrices = cp.asarray(data["matrices"][start:end])
            gradients = cp.asarray(data["gradients"][start:end])
            previous = cp.asarray(np.broadcast_to(np.eye(4), (end - start, 4, 4)).copy())
            output = device.allocate(end - start)
            device.synchronize()
            started = time.perf_counter()
            device.launch(matrices, gradients, previous, output)
            device.synchronize()
            elapsed = time.perf_counter() - started
            for name in fields:
                fields[name][start:end] = cp.asnumpy(output[name])
        launched.append({"first_row": start, "end_row": end, "launch_and_sync_s": elapsed})
        del matrices, gradients, previous, output
    np.savez_compressed(output_path, fresh_cpu_status=fresh_status, fresh_cpu_update=fresh_update, **fields)
    rows, passed = [], True
    for index in range(count):
        update = absolute_difference(np, fresh_update[index], fields["update"][index])
        composed = absolute_difference(np, fresh_update[index], fields["composed"][index])
        accepted = (int(fields["status"][index]) == 0 and not update["nonfinite_difference"]
                    and not composed["nonfinite_difference"] and update["absolute_delta"] <= TOLERANCE
                    and composed["absolute_delta"] <= TOLERANCE and np.isfinite(fields["step"][index]).all())
        rows.append({"record_index": index, "cpu_status": int(fresh_status[index]),
            "gpu_status": int(fields["status"][index]), "update": update, "identity_composition": composed,
            "passed": bool(accepted)})
        passed &= bool(accepted)
    if before != {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in data.items()}:
        raise RuntimeError("Original observed numeric systems/gold were mutated during microcomparison")
    fixture.check_unchanged()
    return {"passed": passed, "count": count, "fresh_cpu_gold_bitwise_equal": cpu_exact,
        "fresh_original_dll_s": cpu_s, "gpu_executed": True, "bounded_batches": launched,
        "results": {"path": str(output_path), "sha256": file_hash(output_path)}, "rows": rows,
        "composition_scope": "Identity previous operand only. Actual previous accumulated trajectory poses were not observed.",
        "timing_scope": "Observed matrices are independent batched micro-solves; batch launch/sync excludes upload, setup, CPU shadows and D2H. No sequential-ICP or whole-Finish claim.",
        "diagnostic_scope": "Step/pivots/D are saved; exact original Eigen intermediate values are not exposed/certified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver-dll", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll")
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--equations", type=Path, help="Closed output-only original all-nine equation manifest")
    parser.add_argument("--singleton-repeats", type=int, default=0,
        help="Optional bounded one-system launch+sync observation; no sequential-ICP speed authority")
    args = parser.parse_args()
    if not args.run_allocated or not 1 <= args.repeats <= 10000 or not 0 <= args.singleton_repeats <= 10000:
        parser.error("Require allocated hardware and bounded repeat count")
    args.output = args.output.resolve()
    inputs_path, results_path = args.output.with_suffix(".inputs.npz"), args.output.with_suffix(".results.npz")
    actual_results_path = args.output.with_suffix(".actual-results.npz")
    try:
        args.output.relative_to(ROOT/"benchmark-output")
        args.solver_dll.resolve(strict=True).relative_to(ROOT)
    except (OSError, ValueError):
        parser.error("Keep fresh private outputs/solve DLL in workspace benchmark-output")
    for path in (args.output, inputs_path, results_path, actual_results_path):
        if path.exists():
            parser.error("Preserve old artifact; require fresh "+str(path))
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle field server before isolated numerical research")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    core = source_hash()
    artifacts = {path: file_hash(ROOT/path) for path in ARTIFACTS}
    solver_path = args.solver_dll.resolve(strict=True)
    solver_pins = {str(path): file_hash(path) for path in (solver_path, solver_path.with_suffix(".build.json"))}
    report = {"kind": "standalone-pinned-eigen-device-ldlt-shadow-v1", "status": "running", "start_utc": now(),
        "source_sha256": core, "artifacts_sha256": artifacts, "solver_sha256": solver_pins,
        "absolute_update_tolerance": TOLERANCE, "absolute_composition_tolerance": TOLERANCE,
        "bitwise_equivalence_claim": False, "nn_or_full_finish_authority": False,
        "scope": "Identical synthetic FP64 6x6/gradient inputs compared to the pinned standalone original resident Eigen bridge, with matching acceptance/rejection codes. Quaternion Rz*Ry*Rx incremental output and a separately original NumPy update@previous pose compared. Scalar evaluator/libdevice trig and explicit RN pose product are new unproved rounding. No original ICP/NN/gates/convergence/batching or graphs altered.",
        "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    save()
    device = None
    fixture = None
    try:
        if args.equations is not None:
            from scripts.research.validate_resident_equation_fixture import validate_equation_fixture
            fixture = validate_equation_fixture(args.equations, expected_solver_path=solver_path)
            report["observed_equations"] = {"manifest_path": fixture.manifest_path,
                "manifest_sha256": fixture.manifest_sha256, "equation_path": fixture.equation_path,
                "equation_sha256": fixture.equation_sha256, "count": fixture.count,
                "authority": "Unchanged original 6x6 inputs/update comparison only; no changed trajectory authority"}
        import numpy as np
        import cupy as cp
        from scripts.research.archive.research_resident_icp import EigenSolve, CPU_BRIDGE_SOURCE
        from scripts.research.research_device_ldlt import DeviceLDLT
        from scripts.process_metrics import peak_rss_bytes
        eigen = EigenSolve(solver_path)
        cpp = solver_path.with_suffix(".cpp")
        if file_hash(cpp) != hashlib.sha256(CPU_BRIDGE_SOURCE.encode()).hexdigest():
            raise RuntimeError("Actual emitted CPU bridge CPP differs from pinned source")
        solver_pins[str(cpp)] = file_hash(cpp)
        report["solver_sha256"] = dict(solver_pins)
        device = DeviceLDLT(device=0)
        report.update(eigen=eigen.metadata, device=device.provenance)
        runtime = actual_runtime(np, cp, device)
        report["runtime_binding"] = runtime
        if fixture is not None:
            observed = fixture.manifest
            binding = observed["proof_bindings"]
            if (np.__version__ != binding["versions"]["numpy"] or cp.__version__ != binding["versions"]["cupy"]
                    or runtime["driver_version"] != binding["cuda_driver_version"]
                    or runtime["runtime_version"] != binding["cuda_runtime_version"]
                    or runtime["device"] != binding["resident_configuration"]["device"]
                    or runtime["device_name"] != binding["gpu"][0]["name"]
                    or binding["numpy_core_binary_sha256"] not in {
                        value["sha256"] for value in runtime["numpy_binary"].values()}
                    or eigen.metadata != observed["solve_metadata"]):
                raise RuntimeError("Actual original equation NumPy/CuPy/device/driver/loaded Eigen runtime differs")
        data = cases(np)
        names = [row[0] for row in data]
        matrices = np.ascontiguousarray([row[1] for row in data])
        gradients = np.ascontiguousarray([row[2] for row in data])
        previous = np.ascontiguousarray([row[3] for row in data])
        np.savez_compressed(inputs_path, matrices=matrices, gradients=gradients, previous=previous, names=np.asarray(names))
        report["input_artifact"] = {"path": str(inputs_path), "sha256": file_hash(inputs_path)}
        input_hashes = [hashlib.sha256(array.tobytes()).hexdigest() for array in (matrices, gradients, previous)]
        cpu_status, cpu_update = np.zeros(len(data), np.int32), np.full((len(data), 4, 4), np.nan)
        cpu_start = time.perf_counter()
        for i in range(len(data)):
            cpu_status[i] = eigen.library.resident_solve(matrices[i].ctypes.data, gradients[i].ctypes.data, cpu_update[i].ctypes.data)
        cpu_s = time.perf_counter()-cpu_start
        with cp.cuda.Device(0), cp.cuda.Stream.null:
            a, b, p = cp.asarray(matrices), cp.asarray(gradients), cp.asarray(previous)
            output = device.allocate(len(data))
            device.synchronize()
            started = time.perf_counter()
            device.launch(a, b, p, output)
            device.synchronize()
            first_s = time.perf_counter()-started
            host = {key: cp.asnumpy(value) for key, value in output.items()}
        np.savez_compressed(results_path, cpu_status=cpu_status, cpu_update=cpu_update, **host)
        report["result_artifact"] = {"path": str(results_path), "sha256": file_hash(results_path)}
        passed = True
        for i, name in enumerate(names):
            expected_extra = data[i][4]
            accepted = int(cpu_status[i]) == 0 and int(host["status"][i]) == 0
            row = {"name": name, "cpu_status": int(cpu_status[i]), "gpu_status": int(host["status"][i]),
                "status_equal": int(cpu_status[i]) == int(host["status"][i]), "update_absolute_delta": None,
                "composition_absolute_delta": None, "passed": False, "prototype_pose_input_expected_status": expected_extra}
            if expected_extra is not None:
                row["passed"] = int(cpu_status[i]) == 0 and int(host["status"][i]) == expected_extra
                row["scope"] = "Additional prototype previous/composition operand rejection; not an Eigen 6x6 rejection-code equivalence claim"
                passed &= row["passed"]
                report["rows"].append(row)
                continue
            if accepted:
                update = absolute_difference(np, cpu_update[i], host["update"][i])
                row["update_absolute_delta"] = update["absolute_delta"]
                expected_pose = cpu_update[i] @ previous[i]
                composed = absolute_difference(np, expected_pose, host["composed"][i])
                row["composition_absolute_delta"] = composed["absolute_delta"]
                row["nonfinite_difference"] = update["nonfinite_difference"] or composed["nonfinite_difference"]
                row["passed"] = bool(np.isfinite(host["step"][i]).all() and np.isfinite(host["update"][i]).all() and np.isfinite(host["composed"][i]).all()
                    and not row["nonfinite_difference"]
                    and row["update_absolute_delta"] <= TOLERANCE and row["composition_absolute_delta"] <= TOLERANCE)
            else:
                row["passed"] = row["status_equal"]
            passed &= row["passed"]
            report["rows"].append(row)
        if input_hashes != [hashlib.sha256(array.tobytes()).hexdigest() for array in (matrices, gradients, previous)]:
            raise RuntimeError("Original CPU shadow inputs were mutated")
        report.update(cpu_initial_shadow_s=cpu_s, gpu_first_launch_sync_wall_s=first_s, all_cases_passed=passed,
            diagnostic_scope="GPU step/pivots/diagonal are saved for diagnosis; finite accepted step is checked, but exact Eigen pivot/diagonal/solution-vector values are not exposed by original DLL and not certified. CPU acceptance code/incremental matrix and original NumPy composition are the actual shadows.")
        save()
        if not passed:
            raise RuntimeError("GPU LDLT/Euler status or numerical shadows failed; raw outputs preserved")
        if fixture is not None:
            report["actual_equation_comparison"] = actual_equations(np, cp, device, eigen, fixture, actual_results_path)
            save()
            if report["actual_equation_comparison"]["passed"] is not True:
                raise RuntimeError("Actual original equation status/update shadows failed; raw outputs preserved")
        # Narrow launch/sync latency only. CPU comparisons and data uploads are
        # excluded explicitly; this is not a PCIe or whole-ICP speed claim.
        with cp.cuda.Device(0), cp.cuda.Stream.null:
            device.synchronize()
            started = time.perf_counter()
            for _ in range(args.repeats):
                device.launch(a, b, p, output)
            device.synchronize()
            elapsed = time.perf_counter()-started
            final = {key: cp.asnumpy(value) for key, value in output.items()}
        if any(not np.array_equal(host[key].view(np.uint8), final[key].view(np.uint8)) for key in host):
            raise RuntimeError("Repeated resident micro-solve outputs changed")
        if args.singleton_repeats:
            with cp.cuda.Device(0), cp.cuda.Stream.null:
                one_a, one_b, one_p = cp.asarray(matrices[:1]), cp.asarray(gradients[:1]), cp.asarray(previous[:1])
                one = device.allocate(1)
                device.synchronize()
                started = time.perf_counter()
                for _ in range(args.singleton_repeats):
                    device.launch(one_a, one_b, one_p, one)
                    device.synchronize()
                single_s = time.perf_counter() - started
                single = {key: cp.asnumpy(value) for key, value in one.items()}
            if any(not np.array_equal(host[key][:1].view(np.uint8), single[key].view(np.uint8)) for key in host):
                raise RuntimeError("One-system launch produced a different previously shadowed result")
            report["singleton_observation"] = {"systems_per_launch": 1, "repeats": args.singleton_repeats,
                "launch_and_sync_wall_s": single_s,
                "scope": "Python shape/alias guards, launch and explicit synchronization per solve on preuploaded synthetic SPD case0. No NN, equations, PCIe inputs, convergence or trajectory."}
        report.update(status="passed", repeated_batch_launch_and_terminal_sync_s=elapsed, repeated_batches=args.repeats,
            systems_per_batch=len(data), peak_process_rss_bytes=peak_rss_bytes(),
            timing_scope="Python launch/shape/disjoint-output guards plus repeated kernel enqueue and one terminal sync; initial upload, CPU shadows, D2H and setup excluded. No graph, single-iteration or whole-ICP claim.")
        report["runtime_binding_after"] = actual_runtime(np, cp, device)
        if report["runtime_binding_after"] != runtime:
            raise RuntimeError("Actual metric/device/driver/runtime/thread configuration changed")
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        primary = sys.exception()
        failures = []
        if device is not None:
            try:
                device.close()
            except BaseException as error:
                failures.append(error)
        try:
            if fixture is not None:
                fixture.check_unchanged()
            report["source_after_sha256"] = source_hash()
            report["artifacts_after_sha256"] = {path: file_hash(ROOT/path) for path in ARTIFACTS}
            report["solver_after_sha256"] = {path: file_hash(Path(path)) for path in solver_pins}
            if (report["source_after_sha256"] != core or report["artifacts_after_sha256"] != artifacts
                    or report["solver_after_sha256"] != solver_pins):
                raise RuntimeError("Bound numerical/core/solver artifacts changed")
        except BaseException as error:
            failures.append(error)
        if failures:
            report.update(status="failed", cleanup_failures=[{"type": type(error).__name__, "message": str(error)} for error in failures])
        report["end_utc"] = now()
        try:
            save()
        except BaseException as error:
            failures.append(error)
        if failures:
            if primary is not None:
                primary.add_note(f"Device LDLT closure also failed: {failures}")
            else:
                raise failures[0]


if __name__ == "__main__":
    main()
