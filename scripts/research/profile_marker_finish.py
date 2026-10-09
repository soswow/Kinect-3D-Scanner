"""Compare original Finish and explicit marker proposals from one Live checkpoint.

Source-only prototype until separately allocated and reviewed. Both modes use
original legacy CPU registration and the unchanged graph/geometry gates. The
marker mode chooses local ICP initialization when SIFT supplies no proposal,
or bounded additional global seeds before original full bridge verification.
CUDA corner lookup is fully CPU-shadowed, so its whole Finish
wall is an accuracy experiment rather than a validated production speedup.
No old checkpoint/resident proof is promoted to authorize marker proposals.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from dataclasses import replace
import datetime as dt
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.profile_session import file_hash, source_hash
from scripts.research.profile_resident_finish import PIPELINE
from scripts.research.profile_checkpoint_resident_finish import GlobalSeedThreadScope
from scripts.research.validate_finish_resident_proof import THREAD_POLICY
from scripts.research.marker_proposal_scope import POLICY as LOCAL_POLICY, MarkerProposalFailure, MarkerSeedScope
from scripts.research.global_marker_proposal_scope import (
    POLICY as GLOBAL_POLICY, GlobalMarkerProposalScope, OriginalMarkerGlobalProvider)
from scripts.research.archive.validate_uniform_grid_proof import canonical_hash

KIND = "offline-checkpoint-marker-proposals-v2"
NEW_ARTIFACTS = (
    "scripts/research/marker_proposal_scope.py",
    "scripts/research/profile_marker_finish.py",
    "scripts/research/native_corner_cuda.py",
    "scripts/research/global_marker_proposal_scope.py",
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def proposal_policy(seed_scope):
    if seed_scope not in ("local", "global"):
        raise ValueError("Require a declared local or global marker proposal scope")
    return LOCAL_POLICY if seed_scope == "local" else GLOBAL_POLICY


def enter_finish_contexts(stack, *, mode, seed_scope, original_scope, fragments, utility,
                          ransac_threads, provider, marker_trace, observations):
    """Original gate observers must see the new global proposal body once.

    Global scope captures the unwrapped original, then original observers and
    scoped RANSAC hooks are installed. Local scope instead wraps the observed
    original local seam last. ExitStack restores every layer in reverse order.
    """
    proposal_policy(seed_scope)
    if mode not in ("original", "marker"):
        raise ValueError("Require a declared original or marker Finish mode")
    marker_scope = None
    if mode == "marker" and seed_scope == "global":
        marker_scope = GlobalMarkerProposalScope(fragments,
            OriginalMarkerGlobalProvider(provider, fragments), trace=marker_trace)
        stack.enter_context(marker_scope)
    stack.enter_context(original_scope)
    for module, name, label in observations:
        original_scope.observe(module, name, label)
    ransac_scope = GlobalSeedThreadScope(fragments, utility, threads=ransac_threads)
    stack.enter_context(ransac_scope)
    if mode == "marker" and seed_scope == "local":
        marker_scope = MarkerSeedScope(fragments, provider, trace=marker_trace)
        stack.enter_context(marker_scope)
    return marker_scope, ransac_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "marker"), required=True)
    parser.add_argument("--seed-scope", choices=("local", "global"), default="local")
    parser.add_argument("--lookup", choices=("cpu", "cuda-audit"), default="cuda-audit")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-synthetic", type=Path, required=True)
    parser.add_argument("--component-bridge", type=Path, required=True)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    policy = proposal_policy(args.seed_scope)
    if not args.run_allocated:
        parser.error("Require explicit hardware allocation; --help performs no numerical imports")
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    try:
        args.checkpoint.relative_to(ROOT/"benchmark-output")
        args.output.relative_to(ROOT/"benchmark-output")
    except ValueError:
        parser.error("Keep private checkpoint and all new reports in ignored benchmark-output")
    if not args.checkpoint.is_file():
        parser.error("Require a closed, freshly measured raw Live checkpoint")
    sidecar = args.output.with_suffix(".marker.json")
    trace_path = args.output.with_suffix(".finish.trace.jsonl")
    marker_trace_path = args.output.with_suffix(".marker.trace.jsonl")
    geometry_path = args.output.with_suffix(".geometry.npz")
    for path in (args.output, sidecar, trace_path, marker_trace_path, geometry_path):
        if path.exists():
            parser.error(f"Preserve old output; require fresh {path}")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle field server before isolated research")
    os.environ.update(PIPELINE)
    from scripts.research.validate_checkpoint_finish_proof import FINISH_ARTIFACTS
    checkpoint_artifacts = {name: file_hash(ROOT/name) for name in FINISH_ARTIFACTS}
    artifacts = {name: file_hash(ROOT/name) for name in (*FINISH_ARTIFACTS, *NEW_ARTIFACTS)}
    source_before = source_hash()
    component_records = {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in
        (("synthetic", args.component_synthetic), ("bridge", args.component_bridge))}
    checkpoint_record = {"path": str(args.checkpoint), "sha256": file_hash(args.checkpoint)}
    report = {"kind": KIND, "mode": args.mode, "status": "running", "start_utc": now(),
        "policy": policy, "seed_scope": args.seed_scope, "lookup_mode": args.lookup, "source_sha256": source_before,
        "artifacts_sha256": artifacts, "checkpoint_artifacts_sha256": checkpoint_artifacts,
        "checkpoint": checkpoint_record, "component_proof": component_records,
        "marker": None, "registration": None, "final_budget_plan": [],
        "new_marker_pose_authority": False, "performance_attribution_valid": args.mode == "original",
        "scope": "Complete Finish from an identical verified private fresh raw Live checkpoint. Original CPU registration, SIFT features/matches/witnesses, original proposal seeds, competing-seed ambiguity and independent-camera gates preserved. Local policy can initialize original local ICP only when original SIFT PnP is absent; global policy adds <=4 ranked marker seeds after original TOP4+two RANSAC and before original uniqueness/full bridge verification. Neither policy accepts marker PnP directly. Existing sequential-camera and independent-global-bridge authority are reported separately. All marker setup/decoder/calibrated projection/lookup and CPU shadows are inside finish_s; imports/materialization and output surface writing outside. Changed coverage is experimental evidence, not ground truth, old checkpoint quality authority or a camera-throughput claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = original_scope = marker_scope = original_final = binding = ransac_scope = None
    trace_file = marker_trace_file = engine_type = None
    archive_path = archive_before = None
    runtime_builder = None

    def save():
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    def restore_final():
        if engine_type is not None and original_final is not None:
            engine_type._final_volume = original_final

    def emit(file, row):
        file.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
        file.flush()

    save()
    try:
        import cupy as cp
        import cv2
        import numpy as np
        import open3d as o3d
        from scanner_server.engine import ScanEngine
        from scanner_server import cuda_registration, refinement, fragments, bundle_adjustment
        from scripts.profile_session import geometry_summary, stage_summary
        from scripts.process_metrics import gpu_info, peak_rss_bytes
        from scripts.research.benchmark_device_grid_resident import current_runtime_binding
        from scripts.research.benchmark_parallel_fragments import numerical_source_hash
        from scripts.research.validate_device_flat_grid_proof import validate_grid_proof
        from scripts.research.validate_checkpoint_finish_proof import validate_checkpoint_manifest
        from scripts.research.private_live_checkpoint import materialize_checkpoint, pose_sha256
        from scripts.research.finish_resident_registration import FinishRegistrationScope
        from scripts.research.marker_proposal_scope import NativeMarkerProvider
        cv2.setNumThreads(20)
        o3d.utility.set_max_threads(20)
        if cv2.getNumThreads() != 20 or o3d.utility.get_max_threads() != 20:
            raise MarkerProposalFailure("Original OpenCV/Open3D thread policy was not applied")
        template = json.loads(args.component_synthetic.read_text(encoding="utf-8"))
        bridge = json.loads(args.component_bridge.read_text(encoding="utf-8"))

        def actual_binding():
            current = copy.deepcopy(template)
            current.update(source_sha256=source_hash(), component_source_sha256=numerical_source_hash(), gpu=gpu_info(),
                artifacts_sha256={name: file_hash(ROOT/name) for name in template["artifacts_sha256"]},
                versions={"numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__, "cupy": cp.__version__},
                solver_library_path=str(Path(template["solver_library_path"]).resolve()), thread_policy=dict(THREAD_POLICY))
            return current_runtime_binding(current, cp, np, o3d)

        runtime_builder = actual_binding
        binding = actual_binding()
        validate_grid_proof(args.component_synthetic, args.component_bridge, binding, bridge["fixture_binding"])
        manifest = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        scope_base = manifest["scope_base"]
        ransac_threads = scope_base["pipeline_options"].get("research_ransac_threads")
        if (scope_base["runtime_binding"] != binding or scope_base["thread_policy"] != THREAD_POLICY
                or scope_base["pipeline_options"] != dict(PIPELINE,
                    research_final_preplan=scope_base["pipeline_options"].get("research_final_preplan"),
                    research_ransac_threads=ransac_threads)
                or type(scope_base["pipeline_options"].get("research_final_preplan")) is not bool
                or type(ransac_threads) is not int or ransac_threads not in (1, 20)):
            raise MarkerProposalFailure("Checkpoint runtime/settings policy is unsupported by the marker scope")
        archive_path = Path(scope_base["archive"]["path"]).resolve(strict=True)
        archive_before = file_hash(archive_path)
        if archive_before != scope_base["archive"]["sha256"]:
            raise MarkerProposalFailure("Original raw archive changed since checkpoint capture")
        seed = scope_base["seed"]
        cv2.setRNGSeed(seed)
        o3d.utility.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        checkpoint_authority = validate_checkpoint_manifest(args.checkpoint, binding, checkpoint_artifacts, scope_base)
        engine, verification = materialize_checkpoint(args.checkpoint, expected_binding=binding,
            expected_artifacts=checkpoint_artifacts)
        if engine.unprocessed_count or getattr(engine, "_pose_seeds_only", False):
            raise MarkerProposalFailure("Marker Finish requires complete fresh raw Live state")
        report.update(runtime_binding=binding, checkpoint_scope_base=scope_base,
            checkpoint_verification=verification, checkpoint_authority=checkpoint_authority.record,
            research_ransac_threads=ransac_threads)
        report["scope_binding"] = {"checkpoint": checkpoint_authority.record, "fresh_live": manifest["fresh_live"],
            "original_scope_base": scope_base, "marker_policy": policy,
            "seed_scope": args.seed_scope, "lookup_mode": args.lookup}
        report["scope_binding_sha256"] = canonical_hash(report["scope_binding"])
        trace_file = trace_path.open("x", encoding="utf-8", newline="\n")
        marker_trace_file = marker_trace_path.open("x", encoding="utf-8", newline="\n")
        live_diagnostics, live_stages = copy.deepcopy(engine.diagnostics), dict(engine.stage_totals_ms)
        before_indices = [index for index, _ in engine.poses]
        engine_type, original_final = ScanEngine, ScanEngine._final_volume

        def final(value, *positional, **keywords):
            if scope_base["pipeline_options"]["research_final_preplan"]:
                started = time.perf_counter()
                planner = copy.copy(value)
                voxel = value.settings.final_voxel_m or value.settings.voxel_m
                planner.settings, planner.voxel_size = replace(value.settings, voxel_m=voxel), voxel
                required = planner._required_fusion_blocks(value.poses, stage="final_budget_plan")
                report["final_budget_plan"].append({"required_blocks": required, "wall_s": time.perf_counter()-started,
                    "budget_blocks": value.settings.final_block_count, "effective_voxel_m": voxel,
                    "unrounded_pose_sha256": pose_sha256(np, value.poses),
                    "authority": "Observation only; original Final delegated unchanged"})
            return original_final(value, *positional, **keywords)

        ScanEngine._final_volume = final
        save()
        started = time.perf_counter()
        original_scope = FinishRegistrationScope(cuda_registration, refinement._match, None,
            mode="native", trace=lambda row: emit(trace_file, row))
        observations = [(fragments, name, "fragments."+name) for name in
            ("_local_match", "_global_seed", "_verify_bridge", "_verify_partial_bridge", "_verify_visual_bridge",
             "_validate_bridge_pose", "_pair", "_heldout", "_visual_witness", "propose_fragment_poses")]
        for name in ("_trustworthy", "_heldout", "_cost", "_distance", "propose_poses"):
            if hasattr(refinement, name):
                observations.append((refinement, name, "refinement."+name))
        for name in ("_supported", "validate_depth", "propose_bundle_poses"):
            if hasattr(bundle_adjustment, name):
                observations.append((bundle_adjustment, name, "bundle_adjustment."+name))
        observations.append((refinement.REG, "get_information_matrix_from_point_clouds", "original_information_matrix"))
        provider = NativeMarkerProvider(engine, lookup_mode=args.lookup, device=0) if args.mode == "marker" else None
        with ExitStack() as contexts:
            marker_scope, ransac_scope = enter_finish_contexts(contexts, mode=args.mode, seed_scope=args.seed_scope,
                original_scope=original_scope, fragments=fragments, utility=o3d.utility,
                ransac_threads=ransac_threads, provider=provider,
                marker_trace=lambda row: emit(marker_trace_file, row), observations=observations)
            progress = lambda current, total, result: print(
                f"Finish {current}/{total}: {result.get('message', '')}", flush=True)
            built, result = engine.build_mesh(progress_cb=progress)
            if marker_scope is not None:
                marker_scope.healthy()
            ransac_scope.healthy()
            original_scope.finish()
        finish_s = time.perf_counter()-started
        restore_final()
        reconstruction = engine.reconstruction_report()
        value = {"schema_version": 2, "session": archive_path.name, "input_sha256": archive_before,
            "source_sha256": source_before, "source_changed_during_profile": source_hash() != source_before,
            "input_changed_during_profile": file_hash(archive_path) != archive_before,
            "selected_indices": scope_base["selected_indices"], "seed": seed,
            "frames": len(engine.raw_frames), "settings": engine.settings.to_dict(), "settings_overrides": {},
            "native_mode": PIPELINE["KINECT_NATIVE"], "omp_threads": "8", "thread_policy": THREAD_POLICY,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__, "open3d": o3d.__version__},
            "gpu_hardware": gpu_info(), "pipeline_options": PIPELINE, "backend": engine.backend,
            "research_pipeline_options": scope_base["pipeline_options"],
            "live_s": None, "finish_s": finish_s, "processing_s": None, "finish_requested": True,
            "fresh_live_measured_s": manifest["capture_report"]["fresh_live_measured_s"],
            "pose_seeds_used": False, "checkpoint": checkpoint_authority.record,
            "accepted_before_finish": len(before_indices), "accepted_indices_before_finish": before_indices,
            "accepted": engine.frame_count, "accepted_indices": [row["index"] for row in reconstruction["poses"]],
            "mesh_built": built, "build_result": result, "poses": reconstruction["poses"],
            "live_diagnostics": live_diagnostics, "live_stages": stage_summary(live_diagnostics, live_stages, np),
            "all_stages": stage_summary(engine.diagnostics, engine.stage_totals_ms, np), "diagnostics": engine.diagnostics,
            "fragment_reconnection": engine.fragment_reconnection, "refinement": engine.refinement,
            "bundle_adjustment": engine.bundle_adjustment, "final_reconstruction": engine.final_reconstruction,
            "peak_process_rss_bytes": peak_rss_bytes(), "geometry": geometry_summary(engine, geometry_path, np),
            "research_marker_scope": {"kind": KIND, "mode": args.mode, "policy": policy,
                "seed_scope": args.seed_scope,
                "scope_binding_sha256": report["scope_binding_sha256"], "artifacts_sha256": artifacts,
                "new_marker_pose_authority": False, "original_sift_witnesses_unchanged": True},
            "measurement": report["scope"]}
        args.output.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        report.update(finish_s=finish_s, mesh_built=built, registration=original_scope.report(),
            marker=None if marker_scope is None else marker_scope.report(),
            profile={"path": str(args.output), "sha256": file_hash(args.output),
                "geometry_path": str(geometry_path), "geometry_sha256": file_hash(geometry_path)})
        if not built:
            raise MarkerProposalFailure("Original Finish did not produce a completed mesh; report retained")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        primary = sys.exception()
        failures = []
        for action in (restore_final,):
            try:
                action()
            except BaseException as error:
                failures.append(error)
        for file, key in ((trace_file, "original_finish_trace"), (marker_trace_file, "marker_trace")):
            if file is not None:
                try:
                    file.close()
                    report[key] = {"path": file.name, "sha256": file_hash(Path(file.name))}
                except BaseException as error:
                    failures.append(error)
        try:
            if original_scope is not None:
                report["registration"] = original_scope.report()
            if marker_scope is not None:
                report["marker"] = marker_scope.report()
            if ransac_scope is not None:
                report["ransac_proposals"] = ransac_scope.report()
            report["source_after_sha256"] = source_hash()
            report["artifacts_after_sha256"] = {name: file_hash(ROOT/name) for name in artifacts}
            report["checkpoint_after_sha256"] = file_hash(args.checkpoint)
            report["component_proof_after"] = {key: {"path": row["path"], "sha256": file_hash(Path(row["path"]))}
                for key, row in component_records.items()}
            report["runtime_binding_after"] = None if runtime_builder is None else runtime_builder()
            if (report["source_after_sha256"] != source_before or report["artifacts_after_sha256"] != artifacts
                    or report["checkpoint_after_sha256"] != checkpoint_record["sha256"]
                    or report["component_proof_after"] != component_records
                    or binding is not None and report["runtime_binding_after"] != binding
                    or archive_path is not None and file_hash(archive_path) != archive_before):
                raise MarkerProposalFailure("Marker/checkpoint/core/runtime/raw provenance changed during Finish")
            report["final_hook_restored"] = engine_type is None or engine_type._final_volume is original_final
            if (not report["final_hook_restored"] or (original_scope and not original_scope.restored)
                    or (marker_scope and not marker_scope.restored) or (ransac_scope and not ransac_scope.restored)):
                raise MarkerProposalFailure("Offline research hooks were not restored")
        except BaseException as error:
            failures.append(error)
        report["end_utc"] = now()
        if failures:
            report.update(status="failed", cleanup_failures=[{"type": type(error).__name__, "message": str(error)} for error in failures])
        try:
            save()
        except BaseException as error:
            failures.append(error)
        if failures:
            if primary is not None:
                primary.add_note(f"Marker Finish closure also failed: {failures}")
            else:
                raise failures[0]
    # Windows numerical-worker lifetime follows the existing verified runners.
    if sys.platform == "win32":
        from scripts.process_metrics import finish_cuda_worker
        try:
            o3d.core.cuda.synchronize()
            finish_cuda_worker()
        except BaseException as error:
            report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)},
                end_utc=now())
            try:
                save()
            except BaseException as secondary:
                error.add_note(f"Late worker failure diagnostics write also failed: {secondary}")
            raise


if __name__ == "__main__":
    main()
