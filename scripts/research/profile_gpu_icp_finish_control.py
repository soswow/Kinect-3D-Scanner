"""Original, unobserved Finish from the current whole-Finish Live checkpoint.

This separate control retains the frozen controller's raw/checkpoint/runtime,
export and failure closure. It changes only the Final transaction call: no
registration scope, gate observer, dispatch replacement or permit is entered.
The original build and selected-device completion are charged to Finish wall.
It is an overhead control, without GPU, gate-quality or timing authority.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from pathlib import Path
import socket
import sys
import time
from types import CodeType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import profile_gpu_icp_finish as original

KIND = "gpu-icp-whole-finish-unobserved-control-v1"
ORIGINAL = ROOT / "scripts/research/profile_gpu_icp_finish.py"
ORIGINAL_SHA256 = "86b3912f72849735328266d2672e208e88cd62e60c51881b85cf055cb3508984"
CONTROL_FILES = ("scripts/research/profile_gpu_icp_finish_control.py", "tests/test_gpu_icp_finish_control.py")
DESCRIPTION = (
    "Same current fresh-raw Live checkpoint; direct original ScanEngine.build_mesh. "
    "No research registration scope, dispatch substitution, gate observer, CPU result shadow "
    "or proof permit. Original registration, bundle, weighted Final fusion and mesh execute "
    "unchanged. Finish includes direct build, selected-device completion and bounded Final "
    "validation. Setup/materialization, unrounded geometry export and full resource closure "
    "are separate. Call/gate/graph evidence is uncollected; no quality, GPU or speed authority."
)


def require(value, message):
    original.require(value, message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-block-count", type=int)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args(argv)
    args.mode = "native"
    args.checkpoint_directory = args.finish_audit = args.quality_proof = None
    def idle():
        with socket.socket() as probe:
            probe.settimeout(.3)
            return probe.connect_ex(("127.0.0.1", 8000)) != 0
    try:
        args.output = original.preflight(args, idle=idle)
        args.session = args.session.resolve(strict=True)
        args.checkpoint = args.checkpoint.resolve(strict=True)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        parser.error(str(error))
    return args


def derivation_contract(candidate=None):
    """Invert the sole changed call and compare the complete original main AST."""
    require(sha(ORIGINAL) == ORIGINAL_SHA256, "Frozen whole-Finish controller changed")
    source = ORIGINAL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    changed = copy.deepcopy(node) if candidate is None else copy.deepcopy(candidate)
    count = 0
    for item in ast.walk(changed):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            if candidate is None and item.func.id == "execute_finish":
                item.func.id = "original_build_control"; count += 1
            elif candidate is not None and item.func.id == "original_build_control":
                item.func.id = "execute_finish"; count += 1
    require(count == 1, "Control changes exactly one Final transaction call")
    if candidate is not None:
        require(ast.dump(changed, include_attributes=False) == ast.dump(node, include_attributes=False),
                "Control setup, checkpoint, export or closure body changed")
    compiled = compile(source, str(ORIGINAL), "exec", dont_inherit=True)
    expected = next(c for c in compiled.co_consts if isinstance(c, CodeType) and c.co_name == "main")
    require(original.main.__globals__ is original.__dict__
            and original.code_state(original.main.__code__) == original.code_state(expected),
            "Loaded original controller main differs from frozen source")
    return changed if candidate is None else {
        "original_sha256": ORIGINAL_SHA256, "inverse_final_call_count": count,
        "original_main_ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
        "setup_checkpoint_export_and_closure_exact": True,
    }


def source_contract():
    changed = derivation_contract()
    contract = derivation_contract(changed)
    return {"kind": KIND, "derivation": contract,
            "artifacts_sha256": {name: sha(ROOT/name) for name in CONTROL_FILES},
            "checkpoint_artifact_names": list(original.CHECKPOINT_FILES),
            "gate_observation": False, "registration_dispatch_override": False,
            "all_original_bundle_slot_owners_checked": True,
            "gpu_or_quality_authority": False}


def unobserved_loaded_owners(modules, python_modules=(), observed_slots=()):
    """No research wrapper exists here: retain every original Bundle slot."""
    return original.LoadedOwners(modules, python_modules)


def complete_selected_devices():
    """Charge actual selected CUDA completion; no numerical imports until run."""
    import cupy as cp
    import open3d as o3d
    with cp.cuda.Device(0):
        o3d.core.cuda.synchronize()
        cp.cuda.runtime.deviceSynchronize()


def validate_final(engine, value):
    require(type(value) is tuple and len(value) == 2 and value[0] is True
            and type(value[1]) is dict, "Original control Final mesh build failed")
    require(engine.mesh is not None and engine._final_vbg is not None,
            "Original successful weighted Final must commit mesh and Final volume")
    final = engine.final_reconstruction
    capacity = final.get("allocated_blocks")
    require(final.get("applied") is True and final.get("voxel_m") == .005
            and final.get("allocation_strategy") == "exact missing-key activation"
            and type(capacity) is int and capacity > 0
            and final.get("initial_block_capacity") == capacity
            and final.get("requested_block_capacity") == capacity
            and final.get("required_blocks") == final.get("blocks")
            and type(final.get("blocks")) is int and 0 < final["blocks"] <= capacity
            and capacity <= engine.settings.final_block_count,
            "Original exact weighted Final allocation/capacity contract failed")


def original_build_control(engine, *, mode, scope_factory, original_build, protocol,
                           binding, scope_binding, audit_path=None, quality_path=None,
                           clock=time.perf_counter, progress=None, completion=None):
    """Original build exactly once; no supplied scope factory or protocol called."""
    require(mode == "native" and audit_path is None and quality_path is None,
            "Unobserved control cannot consume a GPU/timing permit")
    require(engine.unprocessed_count == 0 and engine.settings.confidence_fusion is True
            and engine.settings.final_voxel_m == .005, "Completed weighted current Live checkpoint required")
    require(binding.get("source_sha256") == original.CURRENT,
            "Control requires unchanged current production source")
    started = clock()
    failures, primary, value = [], None, None
    try:
        value = original_build(engine, progress_cb=progress)
        validate_final(engine, value)
    except BaseException as error:
        primary = error
    finally:
        original.clean("control selected-device completion", completion or complete_selected_devices,
                       failures, primary)
        elapsed = clock()-started
    if primary is None and failures:
        primary = original.FinishProfileFailure("Unobserved control selected-device completion failed")
    registration = {"complete": primary is None, "restored": True, "closed": not failures,
        "failure": None if primary is None else original.error_record(primary),
        "cleanup_failures": failures, "mode": "unobserved-control",
        "scope_installed": False, "original_build_calls": 1,
        "registration_evidence_available": False,
        "calls": [], "events": [], "graphs": [], "final_pose_inventory": [],
        "workspace": None, "cache_receipts": [],
        "uncollected": ["registration_calls", "gate_events", "graph_calls", "scope_final_pose_inventory"],
        "quality_authority": False}
    record = {"finish_s": elapsed, "registration": registration,
        "owned_proof": {"required": False, "closed": True, "failure": None, "cleanup_failures": []},
        "finish_cleanup_failures": failures, "built": value}
    if primary is not None:
        setattr(primary, "finish_profile_record", record)
        raise primary
    return record


def derive_main():
    changed = derivation_contract()
    derivation_contract(changed)
    namespace = dict(original.__dict__)
    initial = original.frozen_json(source_contract())
    owners = original.LoadedOwners((original, sys.modules[__name__]), (original, sys.modules[__name__]))
    original_main = original.main
    original_code = original_main.__code__
    defaults = (repr(original_main.__defaults__), repr(original_main.__kwdefaults__))
    derived_state = None
    def checked_source():
        owners.check()
        require(original.main is original_main and original_main.__code__ is original_code
                and (repr(original_main.__defaults__),repr(original_main.__kwdefaults__)) == defaults,
                "Original loaded controller owner changed")
        if derived_state is not None:
            function, code, function_defaults, aliases = derived_state
            require(function.__globals__ is namespace and function.__code__ is code
                    and (repr(function.__defaults__),repr(function.__kwdefaults__)) == function_defaults
                    and namespace.keys() == aliases.keys()
                    and all(namespace[key] is owner for key,owner in aliases.items()),
                    "Derived control code/global owner changed")
        actual = source_contract()
        require(actual == initial, "Control source/derivation changed during measurement")
        return actual
    def save(path, report, primary=None):
        report.update(kind=KIND, mode="control", scope=DESCRIPTION,
                      whole_finish_authority=False, gate_quality_authority=False)
        report.setdefault("control_source", initial)
        try:
            report["control_source_after"] = checked_source()
        except BaseException as fault:
            report.update(status="failed", cleanup_passed=False)
            if primary is None:
                primary = fault; report["failure"] = original.error_record(fault)
            else:
                primary.add_note("Control source closure: "+repr(fault))
            original.save_report(path, report, primary)
            raise primary
        original.save_report(path, report, primary)
    namespace.update(parse=parse, KINDS={"native": KIND}, save_report=save,
                     original_build_control=original_build_control, LoadedOwners=unobserved_loaded_owners)
    module = ast.Module(body=[changed], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ORIGINAL), "exec", dont_inherit=True), namespace)
    function = namespace["main"]
    derived_state = (function, function.__code__,
        (repr(function.__defaults__),repr(function.__kwdefaults__)), dict(namespace))
    return namespace["main"]


def main(argv=None):
    derive_main()(argv)


if __name__ == "__main__":
    main()
