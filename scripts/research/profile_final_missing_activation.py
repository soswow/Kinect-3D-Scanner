"""New weighted Final supervisor over the frozen original CPU Finish producer.

Require a fresh closed CPU/tensor/fused activation proof. Native-original keeps
original activation; exact-missing-key selects only the new private allocator.
Forward session/checkpoint/current component proofs and logical Final budget.
Windows worker exit is deferred until this distinct envelope and hooks close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

KIND = "offline-original-cpu-finish-weighted-missing-activation-v2"
NEW_ARTIFACTS = (
    "scripts/research/final_missing_activation.py", "scripts/research/profile_final_missing_activation.py",
    "scripts/research/benchmark_missing_activation.py", "tests/test_final_missing_activation.py")
ORIGINAL_PRODUCER = "scripts/research/profile_final_allocation.py"
ORIGINAL_PRODUCER_SHA256 = "ce9376d8003c570ba3e5584253a8c7cd616f3324ab7c5ab412a737948d0cd042"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def activation_proof(path):
    """Physical component prerequisite only; no inherited mesh/timing token."""
    from scripts.research.benchmark_missing_activation import ARTIFACTS, FROZEN_CORE, KIND as physical_kind, THREAD_ENV
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    pins = {name: sha(ROOT/name) for name in ARTIFACTS}
    check(value.get("kind") == physical_kind and value.get("status") == "passed"
          and value.get("failure") is None and value.get("cleanup_failures") == []
          and value.get("environment_restored") is True and value.get("source_unchanged") is True
          and value.get("source_sha256") == value.get("source_sha256_after") == FROZEN_CORE
          and value.get("artifacts_sha256") == value.get("artifacts_sha256_after") == pins,
          "Fresh current-source CPU/tensor/fused activation proof is incomplete")
    runtime = value.get("runtime_binding")
    check(isinstance(runtime, dict) and runtime == value.get("runtime_binding_after")
          and runtime.get("open3d_threads") == 20 and isinstance(runtime.get("binaries"), dict)
          and runtime["binaries"], "Physical native runtime closure is absent")
    versions, cuda, gpu = runtime.get("versions", {}), runtime.get("cuda", {}), runtime.get("gpu")
    check(set(versions) == {"numpy", "open3d", "cupy"}
          and all(isinstance(version, str) and version for version in versions.values())
          and runtime.get("environment") == THREAD_ENV
          and cuda.get("device") == 0 and isinstance(cuda.get("name"), str) and cuda["name"]
          and all(type(cuda.get(key)) is int and cuda[key] > 0
                  for key in ("driver_version", "runtime_version"))
          and isinstance(gpu, list) and gpu
          and all(isinstance(row, dict) and row.get("name") and row.get("driver") for row in gpu),
          "Physical actual version/thread/CUDA/hardware identity is incomplete")
    for name, record in runtime["binaries"].items():
        binary = Path(name).resolve(strict=True)
        check(binary.stat().st_size == record["bytes"] and sha(binary) == record["sha256"],
              "Actual physical proof native library changed")
    pairs = value.get("pairs")
    check(isinstance(pairs, list) and [row.get("backend") for row in pairs] ==
          ["cpu-original", "cuda-tensor", "cuda-fused"], "Physical weighted backend coverage is incomplete")
    for pair in pairs:
        check(pair.get("complete") is True and pair.get("inputs_unchanged") is True
              and pair.get("inputs") == pair.get("inputs_after")
              and pair.get("exact_key_mapped_attribute_bits") is True
              and type(pair.get("nonzero_weight_voxels")) is int and pair["nonzero_weight_voxels"] > 0,
              "Physical immutable inputs or nonvacuous weighted attribute proof failed")
        comparison = pair.get("comparison", {})
        check(set(comparison) == {"tsdf", "weight", "color"}
              and all(row.get("bit_mismatches") == 0 and type(row.get("elements")) is int
                      and row["elements"] > 0 for row in comparison.values()),
              "Physical key-mapped attribute comparisons are incomplete")
        control, chosen = pair["original-full"], pair["missing-only"]
        keys = chosen.get("key_proof", {})
        check(control.get("initial_capacity") == chosen.get("initial_capacity") == 4
              and control.get("final_capacity", 0) > 4 and chosen.get("final_capacity") == 4
              and control.get("final_blocks") == chosen.get("final_blocks") == 4
              and keys.get("exact_key_union") is True and keys.get("required_blocks") == 4
              and keys.get("initial_capacity") == keys.get("final_capacity") == 4
              and isinstance(keys.get("expected_union_sha256"), str)
              and len(keys["expected_union_sha256"]) == 64
              and keys.get("expected_union_sha256") == keys.get("actual_union_sha256"),
              "Actual original growth versus exact proxy capacity/key proof is incomplete")
    return {"path": str(path), "sha256": sha(path), "runtime_binding": runtime}


def restore(hooks, saved_argv):
    errors = []
    for owner, name, value in reversed(hooks):
        try:
            setattr(owner, name, value)
            check(getattr(owner, name) is value, "Missing-activation supervisor hook failed to restore")
        except BaseException as error:
            errors.append(error)
    try:
        sys.argv = saved_argv
    except BaseException as error:
        errors.append(error)
    return errors


def run(args, forwarded, *, base=None, metrics=None):
    from scripts.research.final_missing_activation import MissingKeyFinalScope, KIND as allocator_kind
    if base is None:
        from scripts.research import profile_final_allocation as base
    if metrics is None:
        from scripts import process_metrics as metrics
    check(sha(ROOT/ORIGINAL_PRODUCER) == ORIGINAL_PRODUCER_SHA256, "Frozen original allocation producer changed")
    check(args.mode in ("native-original", "exact-missing-key"), "Unsupported new allocation mode")
    envelope = args.output.with_suffix(".missing-activation.json")
    sidecar = args.output.with_suffix(".allocation.json")
    check(not any(path.exists() for path in (args.output, envelope, sidecar)), "Require fresh envelope/profile/sidecar paths")
    proof = activation_proof(args.activation_proof)
    pins = {name: sha(ROOT/name) for name in (*base.EXPERIMENT_ARTIFACTS, *NEW_ARTIFACTS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": KIND, "mode": args.mode, "status": "running", "artifacts_sha256": pins,
        "activation_proof": proof, "original_producer_sha256": ORIGINAL_PRODUCER_SHA256,
        "scope": "Original CPU registration/gates/Final body and current checkpoint proof. Exact mode changes activation/candidate state ownership only; original weighted numeric helpers remain. Both modes charge exact isolated union planning. Original control retains its documented shared preparation telemetry/cache effects; exact mode reports private work separately. All-in walls are observations pending independent surface/pose/capacity comparison.",
        "geometry_quality_proven": False, "performance_authority": False, "cleanup_failures": []}

    def save():
        envelope.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    save()
    saved_argv = sys.argv
    hooks, deferred, primary = [], [], None
    original_worker = metrics.finish_cuda_worker
    original_control = base.ObservedOriginalFinalScope

    def replace(owner, name, value):
        old = getattr(owner, name)
        hooks.append((owner, name, old))
        setattr(owner, name, value)

    def control(engine, **kwargs):
        check(engine.settings.confidence_fusion is True, "Matched control requires weighted fusion")
        return original_control(engine, **kwargs)

    try:
        replace(base, "KIND", KIND)
        replace(base, "EXPERIMENT_ARTIFACTS", tuple(pins))
        replace(base, "RightSizedFinalScope", MissingKeyFinalScope)
        replace(base, "ObservedOriginalFinalScope", control)
        replace(metrics, "finish_cuda_worker", lambda: deferred.append(True))
        inherited_mode = "exact-rightsized" if args.mode == "exact-missing-key" else "native-original"
        sys.argv = [str(ROOT/ORIGINAL_PRODUCER), "--mode", inherited_mode, "--output", str(args.output),
                    "--run-allocated", *forwarded]
        base.main()
        check(deferred == [True], "Expected exactly one deferred terminal worker request")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        check(data.get("kind") == KIND and data.get("status") == "complete"
              and data.get("artifacts_sha256") == data.get("artifacts_sha256_after") == pins,
              "Delegated new-kind sidecar/source closure is incomplete")
        selected = data.get("allocation", {})
        expected_kind = allocator_kind if args.mode == "exact-missing-key" else "offline-exact-final-allocation-v1"
        check(selected.get("kind") == expected_kind, "Delegated allocator scope differs from declared mode")
        actual_runtime = data.get("runtime_binding", {})
        binary_hashes = {row["sha256"] for row in proof["runtime_binding"]["binaries"].values()}
        check(actual_runtime == data.get("runtime_binding_after")
              and all(actual_runtime.get("versions", {}).get(name) == version
                      for name, version in proof["runtime_binding"]["versions"].items())
              and actual_runtime.get("gpu") == proof["runtime_binding"]["gpu"]
              and actual_runtime.get("cuda_driver_version") == proof["runtime_binding"]["cuda"]["driver_version"]
              and actual_runtime.get("cuda_runtime_version") == proof["runtime_binding"]["cuda"]["runtime_version"]
              and all(actual_runtime.get(key) in binary_hashes for key in
                      ("numpy_core_binary_sha256", "open3d_backend_binary_sha256")),
              "Actual Finish runtime/GPU differs from the physical activation prerequisite")
        if args.mode == "exact-missing-key" and data.get("outcome") != "expected-early-capacity-failure":
            check(selected.get("restored") is True and selected.get("cleanup_failures") == []
                  and selected.get("weighted_only") is True and selected.get("native_grid_returned") is True
                  and len(selected.get("final_key_proofs", [])) == 1
                  and all(row.get("exact_key_union") is True
                          and row.get("initial_capacity") == row.get("final_capacity")
                          for row in selected["final_key_proofs"])
                  and len(selected.get("preparation_observations", [])) == 1
                  and len(selected.get("activation", [])) == 1
                  and all(row.get("fault") is None and row.get("calls")
                          and all(call.get("complete") is True for call in row["calls"])
                          for row in selected["activation"])
                  and all(row.get("data_owners_and_configuration_unchanged") is True
                          for row in selected.get("preparation_observations", [])),
                  "Exact weighted Final capacity/key/owner closure failed")
        report.update(delegated_sidecar={"path": str(sidecar), "sha256": sha(sidecar)},
            profile={"path": str(args.output), "sha256": sha(args.output)},
            outcome=data.get("outcome"), allocation=selected, status="complete")
    except BaseException as error:
        primary = error
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                                                "traceback": traceback.format_exc()})
        raise
    finally:
        errors = restore(hooks, saved_argv)
        report["supervisor_restored"] = not errors
        report["deferred_terminal_worker_calls"] = len(deferred)
        try:
            report["artifacts_sha256_after"] = {name: sha(ROOT/name) for name in pins}
            check(report["artifacts_sha256_after"] == pins, "Original/new allocation sources changed")
            report["activation_proof_after"] = activation_proof(args.activation_proof)
            check(report["activation_proof_after"] == proof, "Activation proof changed during Finish")
            for name, path in (("delegated_sidecar", sidecar), ("profile", args.output)):
                if name in report:
                    report[name+"_after"] = {"path": str(path), "sha256": sha(path)}
                    check(report[name+"_after"] == report[name], "Delegated report changed during supervisor closure")
        except BaseException as error:
            errors.append(error)
        report["cleanup_failures"] = [{"type": type(error).__name__, "message": str(error)} for error in errors]
        if errors:
            report["status"] = "failed"
        try:
            save()
        except BaseException as write_error:
            cause = primary if primary is not None else (errors[0] if errors else None)
            if cause is not None:
                cause.add_note(f"Missing-activation envelope write also failed: {write_error}")
                raise cause from write_error
            raise
        if errors:
            if primary is not None:
                primary.add_note(f"Missing-activation supervisor cleanup also failed: {errors}")
            else:
                raise errors[0]
    original_worker()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native-original", "exact-missing-key"), required=True)
    parser.add_argument("--activation-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-allocated", action="store_true")
    args, forwarded = parser.parse_known_args()
    if not args.run_allocated:
        parser.error("Require an explicitly allocated exclusive native/GPU slot")
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT/"benchmark-output")
    except ValueError:
        parser.error("Keep fresh private reports inside ignored benchmark-output")
    run(args, forwarded)


if __name__ == "__main__":
    main()
