"""Bounded candidate whole-Finish experiment from the same current Live state.

Only the frozen controller's Final transaction call changes. Original raw,
checkpoint, settings, runtime, mesh export and full failure closure remain.
Audit records exhaustive NN and same-input CPU terminal shadows; shadow records
CPU terminal checks without exhaustive NN instrumentation. Measure requires a
fresh method/source/configuration/runtime qualification, without claiming an
identical later RANSAC history or any production/general-field authority.
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

ORIGINAL = ROOT/"scripts/research/profile_gpu_icp_finish.py"
ORIGINAL_SHA256 = "86b3912f72849735328266d2672e208e88cd62e60c51881b85cf055cb3508984"
KINDS = {mode: "gpu-icp-whole-finish-candidate-"+mode+"-v1"
         for mode in ("capture", "native", "audit", "shadow", "measure")}
CANDIDATE_FILES = (
    "scripts/research/profile_gpu_icp_finish_candidate.py", "tests/test_gpu_icp_finish_candidate_profile.py",
    "scripts/research/gpu_icp_finish_candidate.py", "tests/test_gpu_icp_finish_candidate.py",
    "scripts/research/compare_gpu_icp_candidate_finishes.py", "tests/test_gpu_icp_candidate_finish_quality.py")
CHECKPOINT_FILES = (
    "scripts/research/profile_gpu_icp_finish_candidate.py", "tests/test_gpu_icp_finish_candidate_profile.py",
    "scripts/research/private_live_checkpoint.py", "scripts/profile_session.py", "scripts/process_metrics.py",
    "scripts/research/archive/validate_uniform_grid_proof.py",
    "scripts/research/profile_gpu_icp_finish.py")
DESCRIPTION = (
    "Same current fresh-raw Live checkpoint; bounded bridge-subtree candidate ICP with original "
    "proposal ranking, gates, graph, bundle, exact weighted Final allocation and mesh unchanged. "
    "Finish charges method qualification, context/cache/template setup, original build, all chosen "
    "instrumentation, full candidate cleanup and selected-device completion. Setup/materialization, "
    "unrounded geometry export and final source/resource closure are separate. Audit checks every "
    "actual NN and same-input CPU terminal; shadow omits exhaustive NN checks; measure omits both "
    "after a fresh method/source/runtime qualification. No identical RANSAC-history, whole-field "
    "correctness or production authority is claimed."
)


def require(value, message):
    original.require(value, message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_args(args, *, root=ROOT, idle=None):
    require(args.mode in KINDS, "Candidate mode must be capture, native, audit, shadow or measure")
    require((args.mode == "measure") == (args.candidate_proof is not None),
            "Measure alone consumes a fresh candidate method qualification")
    # Reuse original allocation/idle/fresh-private-checkpoint preflight. Its old
    # timing/quality options are deliberately absent from this new CLI.
    check = argparse.Namespace(**(vars(args) | {"finish_audit":None,"quality_proof":None}))
    check.mode = "capture" if args.mode == "capture" else "native"
    return original.preflight(check, root=root, idle=idle)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--mode", choices=tuple(KINDS), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--candidate-proof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    settings = ast.parse((ROOT/"shared/settings.py").read_text(encoding="utf-8"))
    settings_class = next(n for n in settings.body if isinstance(n,ast.ClassDef) and n.name == "ScanSettings")
    if any(isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)
           and n.target.id == "final_block_count" for n in settings_class.body):
        parser.add_argument("--final-block-count", type=int,
            help="Override the legacy settings field only; current automatic allocation remains original")
    else:
        parser.set_defaults(final_block_count=None)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args(argv)
    def idle():
        with socket.socket() as probe:
            probe.settimeout(.3)
            return probe.connect_ex(("127.0.0.1", 8000)) != 0
    try:
        args.output = validate_args(args, idle=idle)
        args.session = args.session.resolve(strict=True)
        if args.checkpoint is not None:
            args.checkpoint = args.checkpoint.resolve(strict=True)
        if args.checkpoint_directory is not None:
            args.checkpoint_directory = args.checkpoint_directory.resolve()
        if args.candidate_proof is not None:
            args.candidate_proof = args.candidate_proof.resolve(strict=True)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        parser.error(str(error))
    args.finish_audit = args.candidate_proof
    args.quality_proof = None
    return args


def derivation_contract(candidate=None):
    require(sha(ORIGINAL) == ORIGINAL_SHA256, "Frozen whole-Finish controller changed")
    source = ORIGINAL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    changed = copy.deepcopy(node) if candidate is None else copy.deepcopy(candidate)
    count = 0
    for item in ast.walk(changed):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            if candidate is None and item.func.id == "execute_finish":
                item.func.id = "execute_candidate_finish"; count += 1
            elif candidate is not None and item.func.id == "execute_candidate_finish":
                item.func.id = "execute_finish"; count += 1
    require(count == 1, "Candidate changes exactly one Final transaction call")
    if candidate is not None:
        require(ast.dump(changed, include_attributes=False) == ast.dump(node, include_attributes=False),
                "Candidate setup, checkpoint, export or closure body changed")
    compiled = compile(source, str(ORIGINAL), "exec", dont_inherit=True)
    expected = next(c for c in compiled.co_consts if isinstance(c, CodeType) and c.co_name == "main")
    require(original.main.__globals__ is original.__dict__
            and original.code_state(original.main.__code__) == original.code_state(expected),
            "Loaded original controller main differs from frozen source")
    return changed if candidate is None else {
        "original_sha256": ORIGINAL_SHA256, "inverse_final_call_count": count,
        "original_main_ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
        "setup_checkpoint_export_and_closure_exact": True}


def source_contract():
    from scripts.research import gpu_icp_finish_candidate as candidate
    changed = derivation_contract()
    method = candidate.source_contract()
    return {"kinds": dict(KINDS), "derivation": derivation_contract(changed),
        "artifacts_sha256": {name: sha(ROOT/name) for name in CANDIDATE_FILES},
        "candidate_method": method, "configuration":method["configuration"],
        "checkpoint_artifact_names": list(CHECKPOINT_FILES),
        "current_core_policy": "Cold actual source_hash recorded and latched; full before/after closure, no historical digest substitution",
        "checkpoint_policy": "Fresh current-source raw Live; historical checkpoints require exact original source and producer family, never repinning",
        "old_timing_or_quality_permit": False, "whole_finish_authority": False}


def unobserved_loaded_owners(modules, python_modules=(), observed_slots=()):
    # This candidate has no Bundle module gate observers. Retain all original
    # Bundle slot owners instead of inheriting the observed scope's exemption.
    modules, python_modules = list(modules), list(python_modules)
    allocator = sys.modules.get("scanner_server.fusion_memory")
    if allocator is not None:
        if allocator not in modules:
            modules.append(allocator)
        if allocator not in python_modules:
            python_modules.append(allocator)
    return original.LoadedOwners(modules, python_modules)


def complete_selected_devices():
    import cupy as cp
    import open3d as o3d
    with cp.cuda.Device(0):
        o3d.core.cuda.synchronize()
        cp.cuda.runtime.deviceSynchronize()


def validate_final(engine, value):
    require(type(value) is tuple and len(value) == 2 and value[0] is True
            and type(value[1]) is dict and engine.mesh is not None and engine._final_vbg is not None,
            "Original candidate Final mesh build failed")
    final = engine.final_reconstruction
    capacity = final.get("allocated_blocks")
    require(final.get("applied") is True and final.get("voxel_m") == .005
            and final.get("allocation_strategy") == "exact missing-key activation"
            and type(capacity) is int and capacity > 0
            and final.get("initial_block_capacity") == final.get("requested_block_capacity") == capacity
            and final.get("required_blocks") == final.get("blocks")
            and type(final.get("blocks")) is int and 0 < final["blocks"] <= capacity
            and int(engine._final_vbg.hashmap().capacity()) == capacity
            and int(engine._final_vbg.hashmap().size()) == final["blocks"],
            "Original candidate weighted Final allocation/capacity contract failed")
    if final.get("allocation") is not None:
        require(final["allocation"] == "automatic", "Current Final allocation policy changed")


def make_retrieval():
    from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
    owner = object.__new__(DeviceFlatGridICP)
    try:
        owner.__init__(device="CUDA:0", max_clouds=34, max_cache_bytes=256*1024**2,
            audit_nearest=True, audit_misses=True, miss_policy="direct-miss-research-v1")
    except BaseException as error:
        setattr(error,"candidate_retrieval",owner)
        raise
    return owner


def make_context(engine, mode, qualification, qualification_module, binding, configuration):
    import numpy as np
    from scanner_server import cuda_registration, fragments, refinement
    from scripts.research import gpu_icp_finish_candidate as candidate
    owner = object.__new__(candidate.GpuICPFinishCandidate)
    try:
        owner.__init__(cuda_registration, fragments, refinement,
            mode=mode, retrieval_factory=make_retrieval, numpy=np, qualification=qualification,
            protocol=qualification_module, engine=engine, binding=binding)
    except BaseException as error:
        setattr(error,"candidate_context",owner)
        raise
    return owner


def execute_candidate_finish(engine, *, mode, scope_factory, original_build, protocol,
                             binding, scope_binding, audit_path=None, quality_path=None,
                             clock=time.perf_counter, progress=None, candidate_factory=None,
                             qualification_module=None, configuration=None, completion=None):
    require(mode in ("native","audit","shadow","measure") and quality_path is None
            and ((mode == "measure") == (audit_path is not None)),
            "Candidate uses its own method qualification, never an old Finish permit")
    require(engine.unprocessed_count == 0 and engine.settings.confidence_fusion is True
            and engine.settings.final_voxel_m == .005
            and binding.get("source_sha256") == original.source_hash(),
            "Completed weighted current Live checkpoint required")
    started = clock()
    context = qualification = None
    value = registration = None
    failures, primary = [], None
    proof = {"required": mode == "measure", "scope":
        "Observed actual audit/current source/configuration/runtime; no exact later trajectory or general-domain permission"}
    try:
        if mode == "native":
            value = original_build(engine, progress_cb=progress)
            registration = {"complete":True,"restored":True,"closed":True,"failure":None,"cleanup_failures":[],
                "mode":"native","calls":[],"events":[],"graphs":[],"final_pose_inventory":[],
                "workspace":None,"cache_receipts":[],"registration_evidence_available":False,
                "original_build_calls":1,"uncollected":["registration_calls","gate_events","graph_calls"]}
            validate_final(engine,value)
        else:
            value, context, qualification, proof, qualification_module = run_candidate_context(engine, mode, original_build,
                binding, audit_path, clock, progress, candidate_factory, qualification_module,
                configuration, proof)
            validate_final(engine,value)
    except BaseException as error:
        primary = error
        context = getattr(error,"candidate_context",context)
        if hasattr(error,"candidate_method_qualification"):
            proof = error.candidate_method_qualification
        qualification = getattr(error,"candidate_qualification_token",qualification)
        qualification_module = getattr(error,"candidate_qualification_module",qualification_module)
    finally:
        if context is not None:
            def close_context():
                try:
                    context.close(primary=primary)
                except BaseException as fault:
                    # The context may deliberately re-raise the identical
                    # completed primary; that is not a second cleanup fault.
                    if fault is not primary:
                        raise
            original.clean("candidate independent owner cleanup", close_context, failures, primary)
            registration = original.clean("closed candidate report", context.report, failures, primary)
        if qualification is not None:
            receipt = original.clean("candidate method qualification closure", lambda:
                qualification_module.qualification_receipt(qualification,close=True), failures, primary)
            if receipt is not None:
                if receipt.get("report_sha256") != proof.get("report_sha256"):
                    failures.append({"action":"candidate method qualification closure",
                                     "message":"Initial method audit differs from qualified receipt"})
                proof = receipt
        if mode == "measure" and audit_path is not None and "report_sha256" in proof:
            after = original.clean("candidate qualification file closure", lambda: sha(audit_path), failures, primary)
            proof["report_sha256_after"] = after
            if after != proof["report_sha256"]:
                failures.append({"action": "candidate qualification file closure", "message": "Method audit file changed"})
        original.clean("candidate selected-device completion", completion or complete_selected_devices, failures, primary)
        elapsed = clock()-started
    if primary is None and failures:
        primary = original.FinishProfileFailure("Candidate Finish completion/diagnostic closure failed")
    record = {"finish_s": elapsed, "registration": registration,
        "method_qualification": proof, "owned_proof": {"required": False,
            "scope": "Old exact-trajectory Finish permits are not consumed", "closed": True,
            "failure": None, "cleanup_failures": []},
        "finish_cleanup_failures": failures, "built": value}
    if primary is None:
        try:
            require(registration is not None and registration.get("complete") is True
                    and registration.get("restored") is True and registration.get("closed") is True
                    and registration.get("failure") is None and registration.get("cleanup_failures") == [],
                    "Candidate context did not close complete and restored")
        except BaseException as error:
            primary = error
    if primary is not None:
        setattr(primary, "finish_profile_record", record)
        raise primary
    return record


def run_candidate_context(engine, mode, original_build, binding, audit_path, clock, progress,
                          candidate_factory, qualification_module, configuration, proof):
    context = qualification = None
    try:
        if qualification_module is None:
            from scripts.research import compare_gpu_icp_candidate_finishes as qualification_module
        if configuration is None:
            from scripts.research import gpu_icp_finish_candidate as candidate
            configuration = candidate.source_contract()["configuration"]
        if mode == "measure":
            proof.update(report_path=str(audit_path), report_sha256=sha(audit_path))
            qualification = qualification_module.qualify_candidate_method(audit_path, binding, configuration=configuration)
        context = (candidate_factory or make_context)(engine, mode, qualification,
                                                       qualification_module, binding, configuration)
        with context:
            value = context.build(engine, original_build, progress_cb=progress)
            context.finish()
        return value, context, qualification, proof, qualification_module
    except BaseException as error:
        if context is not None:
            setattr(error,"candidate_context",context)
        setattr(error,"candidate_method_qualification",proof)
        setattr(error,"candidate_qualification_token",qualification)
        setattr(error,"candidate_qualification_module",qualification_module)
        raise


def derive_main():
    changed = derivation_contract()
    derivation_contract(changed)
    namespace = dict(original.__dict__)
    current_source = original.source_hash()
    require(type(current_source) is str and len(current_source)==64, "Current production source digest required")
    initial = original.frozen_json(source_contract())
    owners = original.LoadedOwners((original, sys.modules[__name__]), (original, sys.modules[__name__]))
    original_main, original_code = original.main, original.main.__code__
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
                    "Derived candidate code/global owner changed")
        actual = source_contract()
        require(actual == initial, "Candidate method/source family changed during measurement")
        require(original.source_hash() == current_source, "Current production source changed during measurement")
        return actual
    def save(path, report, primary=None):
        report.update(kind=KINDS[report["mode"]], scope=DESCRIPTION,
                      whole_finish_authority=False, exact_native_history_authority=False)
        report.setdefault("candidate_source", initial)
        try:
            report["candidate_source_after"] = checked_source()
        except BaseException as fault:
            report.update(status="failed", cleanup_passed=False)
            if primary is None:
                primary = fault; report["failure"] = original.error_record(fault)
            else:
                primary.add_note("Candidate source closure: "+repr(fault))
            original.save_report(path, report, primary)
            raise primary
        original.save_report(path, report, primary)
    namespace.update(parse=parse, KINDS=dict(KINDS), save_report=save, CURRENT=current_source,
        CHECKPOINT_FILES=CHECKPOINT_FILES,
        FAMILY_FILES=original.FAMILY_FILES+tuple(n for n in CANDIDATE_FILES if n not in original.FAMILY_FILES),
        execute_candidate_finish=execute_candidate_finish, LoadedOwners=unobserved_loaded_owners)
    module = ast.Module(body=[changed], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ORIGINAL), "exec", dont_inherit=True), namespace)
    function = namespace["main"]
    derived_state = (function, function.__code__,
        (repr(function.__defaults__),repr(function.__kwdefaults__)), dict(namespace))
    return function


def main(argv=None):
    derive_main()(argv)


if __name__ == "__main__":
    main()
