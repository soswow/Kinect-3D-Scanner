"""Isolated whole-Finish research with original CPU shadows and hard closure.

Run only in an allocated hardware slot with the idle field server stopped.
Live always replays original raw RGB-D without archived pose seeds. Native
mode records original Finish, audit runs new resident calls plus exact CPU NN
and complete registration shadows, timing requires separate successful whole
Finish/quality authority. Existing component proofs never authorize field
targets by themselves. No production code or original math is modified.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash
from scripts.research.finish_resident_registration import FinishRegistrationScope, FinishResearchFailure
from scripts.research.validate_finish_resident_proof import KIND, FINISH_ARTIFACTS, THREAD_POLICY
from scripts.research.archive.validate_uniform_grid_proof import canonical_hash

PIPELINE = {
    "OMP_NUM_THREADS": "8", "KINECT_NATIVE": "on", "KINECT_BLOCK_COUNT": "5000",
    "KINECT_CUDA_REGISTRATION": "cpu", "KINECT_CUDA_ODOMETRY": "off",
    "KINECT_CUDA_MATCHING": "cuda", "KINECT_CUDA_FUSION": "fused",
    "KINECT_MODEL_REFRESH": "lazy", "KINECT_KEYFRAME_CACHE": "on",
    "KINECT_LIVE_RECOVERY": "full", "KINECT_VISUAL_FEATURES": "adaptive",
    "KINECT_VISUAL_REFINEMENT": "icp", "KINECT_FINAL_VISUAL_FIRST": "off",
    "KINECT_FINAL_LOCAL_REFINEMENT": "icp", "KINECT_ADAPTIVE_EXPERIMENTAL": "off",
    "KINECT_CUDA_INPUT": "auto", "KINECT_CUDA_CONFIDENCE": "off", "PYTHONUTF8": "1"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def pose_digest(poses):
    import numpy as np
    digest = hashlib.sha256()
    for index, pose in poses:
        array = np.asarray(pose)
        if array.dtype != np.dtype("<f8") or array.shape != (4, 4) or not np.isfinite(array).all():
            raise ValueError("Measured Live pose must be original finite FP64")
        digest.update(struct.pack("<Q", index))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def live_scope(engine):
    if engine.unprocessed_count != 0 or getattr(engine, "_pose_seeds_only", False):
        raise ValueError("Finish requires completed fresh raw Live; archived pose seeds forbidden")
    # Discrete decisions are separate from unrounded numeric pose bytes.
    fields = ("index", "success", "method", "message")
    decisions = [{key: row[key] for key in fields if key in row} for row in engine.diagnostics]
    return {"accepted_indices": [index for index, _ in engine.poses],
            "decisions_sha256": canonical_hash(decisions), "poses_sha256": pose_digest(engine.poses),
            "unprocessed_count": 0, "pose_source": "fresh raw Live replay"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--mode", choices=("native", "audit", "timing"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh ignored profile JSON; sidecar/trace beside it")
    parser.add_argument("--component-synthetic", type=Path, required=True)
    parser.add_argument("--component-bridge", type=Path, required=True)
    parser.add_argument("--finish-audit", type=Path)
    parser.add_argument("--quality-proof", type=Path)
    parser.add_argument("--solver-dll", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll")
    parser.add_argument("--final-block-count", type=int)
    parser.add_argument("--final-preplan", action="store_true", help="Observe original exact one-block frustum plan; charge its wall time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if not args.run_allocated or args.stride < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Require exclusive allocation and positive selection")
    if args.final_block_count is not None and not 1 <= args.final_block_count <= 50000:
        parser.error("Final block count must be 1–50000 on both matched controls")
    if (args.mode == "timing") != bool(args.finish_audit and args.quality_proof):
        parser.error("Timing requires closed whole-Finish audit and separate quality proof")
    if args.mode != "timing" and (args.finish_audit or args.quality_proof):
        parser.error("Whole-Finish authority is only consumed by unaudited timing")
    args.session = args.session.resolve(strict=True)
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT/"benchmark-output")
        args.solver_dll.resolve(strict=True).relative_to(ROOT)
    except (ValueError, OSError):
        parser.error("Reports must be ignored benchmark-output; pinned solve DLL must exist in workspace")
    sidecar = args.output.with_suffix(".resident.json")
    trace_path = args.output.with_suffix(".resident.trace.jsonl")
    for path in (args.output, sidecar, trace_path, args.output.with_suffix(".geometry.npz")):
        if path.exists():
            parser.error(f"Require fresh artifacts; preserve previous {path}")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle field server before research replay")
    os.environ.update(PIPELINE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_before, archive_before = source_hash(), file_hash(args.session)
    artifacts = {name: file_hash(ROOT/name) for name in FINISH_ARTIFACTS}
    report = {"kind": KIND, "mode": args.mode, "status": "running", "start_utc": now(),
              "source_sha256": source_before, "archive_sha256": archive_before,
              "artifacts_sha256": artifacts, "performance_attribution_valid": args.mode != "audit",
              "component_proof": {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in
                  (("synthetic", args.component_synthetic), ("bridge", args.component_bridge))},
              "final_budget_plan": [], "registration": None,
              "scope": "Fresh raw Live followed by complete original Finish gates/fusion/mesh; only scoped legacy registration is replaced. Audit timing includes full NN/result shadows and is not a speed measurement. Resident setup, signatures, observation, optional block planning and whole-Finish timing-authority validation are charged inside Finish. Component preflight, initial imports, ZIP decoding, exports and closure are outside; nested setup/validation timers are not subtracted."}

    def save():
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    save()
    trace_file, scope, solver, finalized = None, None, None, False
    original_build, original_final, original_worker, original_argv = None, None, None, sys.argv
    engine_type = profile = None
    template = binding = component_authority = None
    trace_rows = 0

    def restore_runner_hooks():
        nonlocal original_build, original_final, original_worker
        if engine_type is not None and original_build is not None:
            engine_type.build_mesh, engine_type._final_volume = original_build, original_final
        if profile is not None and original_worker is not None:
            profile.finish_cuda_worker = original_worker
        sys.argv = original_argv
        report["runner_hooks_restored"] = True

    def emit(row):
        nonlocal trace_rows
        trace_file.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
        trace_file.flush()
        trace_rows += 1

    def current_binding():
        from scripts.research.benchmark_device_grid_resident import current_runtime_binding
        from scripts.research.benchmark_parallel_fragments import numerical_source_hash
        from scripts.process_metrics import gpu_info
        current = copy.deepcopy(template)
        current.update(source_sha256=source_hash(), component_source_sha256=numerical_source_hash(), gpu=gpu_info(),
            artifacts_sha256={name: file_hash(ROOT/name) for name in template["artifacts_sha256"]},
            versions={"numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__, "cupy": cp.__version__},
            solver_library_path=str(args.solver_dll.resolve()), thread_policy=dict(THREAD_POLICY))
        return current_runtime_binding(current, cp, np, o3d)

    def finalize():
        nonlocal finalized
        if finalized:
            return
        if trace_file and not trace_file.closed:
            trace_file.close()
        report["source_sha256_after"] = source_hash()
        report["archive_sha256_after"] = file_hash(args.session)
        report["artifacts_sha256_after"] = {name: file_hash(ROOT/name) for name in artifacts}
        report["runtime_binding_after"] = current_binding() if binding else None
        report["component_proof_after"] = {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in
            (("synthetic", args.component_synthetic), ("bridge", args.component_bridge))}
        if trace_path.exists():
            report["trace"] = {"path": str(trace_path), "sha256": file_hash(trace_path), "rows": trace_rows}
        if scope:
            report["registration"] = scope.report()
        if (report["source_sha256_after"] != source_before or report["archive_sha256_after"] != archive_before
                or report["artifacts_sha256_after"] != artifacts or (binding and report["runtime_binding_after"] != binding)
                or report["component_proof_after"] != report["component_proof"]):
            raise FinishResearchFailure("Core, raw inputs, helper artifacts, runtime or component proofs changed")
        if report.get("failure") is None:
            if scope is None or not scope.complete or not scope.restored or scope.failure is not None:
                raise FinishResearchFailure("Research context did not complete and restore")
            value = json.loads(args.output.read_text(encoding="utf-8"))
            if (not value["mesh_built"] or value["pose_seeds_used"] or value["source_changed_during_profile"]
                    or value["input_changed_during_profile"]):
                raise FinishResearchFailure("Require completed fresh-raw mesh with unchanged production/input bytes")
            value["research_pipeline_options"] = report["scope_binding"]["pipeline_options"]
            value["research_scope_binding_sha256"] = report["scope_binding_sha256"]
            value["research_finish"] = {"mode": args.mode, "helper_artifacts_sha256": artifacts,
                "registration_implementation": "original legacy CPU" if args.mode == "native" else "offline device-flat resident research",
                "production_backend_scope": "Backend metadata describes original production policy; scoped Finish implementation is recorded separately"}
            args.output.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            geometry = Path(value["geometry"]["artifact"])
            report["profile"] = {"path": str(args.output), "sha256": file_hash(args.output),
                                 "geometry_path": str(geometry), "geometry_sha256": file_hash(geometry)}
            report["status"] = "complete"
        report["end_utc"] = now()
        finalized = True
        save()

    try:
        # All numerical imports occur only after CLI/scope/fresh-output guards.
        import cupy as cp
        import cv2
        import numpy as np
        import open3d as o3d
        from scanner_server.engine import ScanEngine
        from scanner_server import cuda_registration, refinement, fragments, bundle_adjustment
        from scripts import profile_session as profile
        from scripts.research.benchmark_device_grid_resident import bridge_class
        from scripts.research.validate_device_flat_grid_proof import validate_grid_proof
        cv2.setNumThreads(20)
        o3d.utility.set_max_threads(20)
        template = json.loads(args.component_synthetic.read_text(encoding="utf-8"))
        bridge = json.loads(args.component_bridge.read_text(encoding="utf-8"))
        binding = current_binding()
        component_authority = validate_grid_proof(args.component_synthetic, args.component_bridge, binding, bridge["fixture_binding"])
        report["runtime_binding"] = binding
        report["component_authority"] = {"bindings_sha256": component_authority.bindings_sha256,
                                         "target_membership_count": len(component_authority.target_digests),
                                         "scope": "Component preflight only; no field target authorization"}
        with zipfile.ZipFile(args.session) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        selected = list(range(len(manifest["frames"])))[::args.stride][:args.limit]
        trace_file = trace_path.open("x", encoding="utf-8", newline="\n")
        engine_type = ScanEngine
        original_build, original_final = ScanEngine.build_mesh, ScanEngine._final_volume
        original_worker = profile.finish_cuda_worker

        def build(engine, *positional, **keywords):
            nonlocal solver, scope
            live = live_scope(engine)
            if (cv2.getNumThreads() != 20 or o3d.utility.get_max_threads() != 20
                    or os.environ.get("OMP_NUM_THREADS") != "8"):
                raise FinishResearchFailure("Actual original OpenCV/Open3D/OMP thread policy changed")
            report["scope_binding"] = {"archive": {"path": str(args.session), "sha256": archive_before},
                "selected_indices": selected, "seed": args.seed, "settings": engine.settings.to_dict(),
                "pipeline_options": dict(PIPELINE, research_final_preplan=args.final_preplan), "thread_policy": dict(THREAD_POLICY),
                "live": live, "runtime_binding": binding}
            report["measured_live_poses"] = [{"index": index, "pose": np.asarray(pose).tolist()} for index, pose in engine.poses]
            report["scope_binding_sha256"] = canonical_hash(report["scope_binding"])
            finish_authority = None
            if args.mode == "timing":
                from scripts.research.validate_finish_resident_proof import validate_finish_proof
                started = time.perf_counter()
                finish_authority = validate_finish_proof(args.finish_audit, component_authority,
                    report["scope_binding"], artifacts, quality_path=args.quality_proof)
                report["finish_authority_validation_s"] = time.perf_counter()-started
            if args.mode != "native":
                started = time.perf_counter()
                Bridge = bridge_class(args.solver_dll, binding["resident_configuration"]["max_scratch_bytes"],
                                      binding["resident_configuration"]["max_query_bytes"])
                solver = Bridge(device="CUDA:0", max_clouds=binding["cache_policy"]["max_clouds"],
                    max_cache_bytes=binding["cache_policy"]["retained_gpu_bytes"],
                    audit_nearest=args.mode == "audit", audit_misses=args.mode == "audit",
                    miss_policy="direct-miss-research-v1", proof_authority=finish_authority)
                report["resident_setup_s"] = time.perf_counter()-started
            scope = FinishRegistrationScope(cuda_registration, refinement._match, solver,
                mode=args.mode, trace=emit, authority=finish_authority)
            save()
            with scope:
                for name in ("_local_match", "_global_seed", "_verify_bridge", "_verify_partial_bridge",
                             "_verify_visual_bridge", "_validate_bridge_pose", "_pair", "_heldout", "_visual_witness",
                             "propose_fragment_poses"):
                    if hasattr(fragments, name):
                        scope.observe(fragments, name, "fragments."+name)
                for name in ("_trustworthy", "_heldout", "_cost", "_distance", "propose_poses"):
                    if hasattr(refinement, name):
                        scope.observe(refinement, name, "refinement."+name)
                for name in ("_supported", "validate_depth", "propose_bundle_poses"):
                    if hasattr(bundle_adjustment, name):
                        scope.observe(bundle_adjustment, name, "bundle_adjustment."+name)
                scope.observe(refinement.REG, "get_information_matrix_from_point_clouds", "original_information_matrix")
                value = original_build(engine, *positional, **keywords)
                scope.finish()
            report["registration"] = scope.report()
            save()
            return value

        def final(engine, *positional, **keywords):
            if args.final_preplan:
                started = time.perf_counter()
                planner = copy.copy(engine)
                voxel = engine.settings.final_voxel_m or engine.settings.voxel_m
                planner.settings = replace(engine.settings, voxel_m=voxel)
                planner.voxel_size = voxel
                required = planner._required_fusion_blocks(engine.poses, stage="final_budget_plan")
                report["final_budget_plan"].append({"required_blocks": required,
                    "wall_s": time.perf_counter()-started, "budget_blocks": engine.settings.final_block_count,
                    "effective_voxel_m": voxel, "unrounded_pose_sha256": pose_digest(engine.poses),
                    "authority": "Observation only; delegate original Final unchanged; one-block unactivated scratch"})
                save()
            return original_final(engine, *positional, **keywords)

        def finish_worker():
            restore_runner_hooks()
            finalize()
            original_worker()

        ScanEngine.build_mesh, ScanEngine._final_volume = build, final
        profile.finish_cuda_worker = finish_worker
        arguments = [str(ROOT/"scripts/profile_session.py"), str(args.session), "--device", "cuda",
                     "--tracking", "legacy", "--finish", "--seed", str(args.seed), "--output", str(args.output),
                     "--stride", str(args.stride)]
        if args.limit is not None:
            arguments.extend(("--limit", str(args.limit)))
        if args.final_block_count is not None:
            arguments.extend(("--final-block-count", str(args.final_block_count)))
        report["profile_arguments"] = arguments
        save()
        sys.argv = arguments
        profile.main()
        restore_runner_hooks()
        finalize()
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        try:
            restore_runner_hooks()
            finalize()
        except BaseException as secondary:
            report.setdefault("cleanup_failures", []).append({"type": type(secondary).__name__, "message": str(secondary)})
            error.add_note(f"Final provenance preservation also failed: {secondary}")
            try:
                save()
            except BaseException as write_error:
                error.add_note(f"Failed diagnostics write: {write_error}")
        raise
    finally:
        restore_runner_hooks()
        if trace_file is not None and not trace_file.closed:
            trace_file.close()


if __name__ == "__main__":
    main()
