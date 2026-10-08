"""Compare original and exact-size Final allocation from one verified Live state.

This is a new offline allocation experiment, not an old resident-Finish proof.
Both modes retain original CPU registration/gates and observe the same isolated
exact block planner inside Finish. Only exact-rightsized changes candidate VBG
capacity or fails before allocation. No hardware runs without an explicit slot.
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
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.profile_session import file_hash, source_hash
from scripts.research.final_allocation_scope import (
    FinalCapacityError, FinalAllocationContractError, RightSizedFinalScope,
    ObservedOriginalFinalScope, validate_original_functions)
from scripts.research.finish_resident_registration import FinishRegistrationScope, FinishResearchFailure
from scripts.research.private_live_checkpoint import materialize_checkpoint, pose_sha256
from scripts.research.profile_resident_finish import PIPELINE
from scripts.research.validate_finish_resident_proof import THREAD_POLICY
from scripts.research.validate_checkpoint_finish_proof import FINISH_ARTIFACTS
from scripts.research.archive.validate_uniform_grid_proof import canonical_hash

KIND = "offline-original-cpu-finish-exact-final-allocation-v1"
EXPERIMENT_ARTIFACTS = FINISH_ARTIFACTS + (
    "scripts/research/final_allocation_scope.py", "scripts/research/profile_final_allocation.py")
OWNER_FIELDS = ("vbg", "mesh", "point_cloud", "model_pcd", "_final_vbg")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def boundary_state(engine, np):
    """Boundary ownership/pose evidence; not a serialized logical VBG proof."""
    return {"owners": {name: id(getattr(engine, name)) for name in OWNER_FIELDS},
            "poses_sha256": pose_sha256(np, engine.poses),
            "accepted_indices": [index for index, _ in engine.poses],
            "frame_count": engine.frame_count, "processed_count": engine._processed_count,
            "live_fusion_generation": engine._live_fusion_generation,
            "live_capacity": int(engine.vbg.hashmap().capacity()),
            "live_blocks": int(engine.vbg.hashmap().size())}


class FinalBoundaryObserver:
    """Observe only this instance; original build_mesh owns all surface commits."""
    def __init__(self, engine, state):
        self.engine, self.state, self.calls, self.restored = engine, state, [], False

    def __enter__(self):
        self.had_instance = "_final_volume" in self.engine.__dict__
        self.previous = self.engine.__dict__.get("_final_volume")
        self.delegate = self.engine._final_volume
        def final(owner, progress_cb=None):
            if owner is not self.engine:
                raise FinalAllocationContractError("Allocation observer received a different engine")
            row = {"before": self.state(owner), "complete": False}
            self.calls.append(row)
            try:
                result = self.delegate(progress_cb)
                row.update(complete=True, candidate_capacity=int(result.hashmap().capacity()),
                           candidate_blocks=int(result.hashmap().size()))
                return result
            except BaseException as error:
                row["failure"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                primary = sys.exception()
                try:
                    row["after"] = self.state(owner)
                    row["owner_and_pose_preserved"] = row["before"] == row["after"]
                    if not row["owner_and_pose_preserved"]:
                        raise FinalAllocationContractError("Final body changed Live ownership/poses before surface commit")
                except BaseException as secondary:
                    if primary is not None:
                        primary.add_note(f"Final boundary observation also failed: {secondary}")
                        raise primary from secondary
                    raise
        try:
            self.engine._final_volume = MethodType(final, self.engine)
        except BaseException as error:
            try:
                if self.had_instance:
                    self.engine._final_volume = self.previous
                else:
                    self.engine.__dict__.pop("_final_volume", None)
            except BaseException as secondary:
                error.add_note(f"Partial Final boundary installation restore also failed: {secondary}")
            raise
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if self.had_instance:
                self.engine._final_volume = self.previous
            else:
                del self.engine.__dict__["_final_volume"]
            self.restored = True
        except BaseException as secondary:
            if error is not None:
                error.add_note(f"Final boundary hook restoration also failed: {secondary}")
                raise error from secondary
            raise
        return False


def current_component_binding(template, solver_library, cp, np, o3d, cv2):
    from scripts.research.benchmark_device_grid_resident import current_runtime_binding
    from scripts.research.benchmark_parallel_fragments import numerical_source_hash
    from scripts.process_metrics import gpu_info
    current = copy.deepcopy(template)
    current.update(source_sha256=source_hash(), component_source_sha256=numerical_source_hash(), gpu=gpu_info(),
        artifacts_sha256={name: file_hash(ROOT/name) for name in template["artifacts_sha256"]},
        versions={"numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__, "cupy": cp.__version__},
        solver_library_path=str(solver_library.resolve()), thread_policy=dict(THREAD_POLICY))
    return current_runtime_binding(current, cp, np, o3d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--mode", choices=("native-original", "exact-rightsized"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--component-synthetic", type=Path, required=True)
    parser.add_argument("--component-bridge", type=Path, required=True)
    parser.add_argument("--solver-dll", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-block-count", type=int,
                        help="Declared Final-only limit override after bit-verified checkpoint restoration")
    parser.add_argument("--expect-capacity-failure", action="store_true",
                        help="Exact mode only: require early over-capacity error and no candidate allocation")
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    if not args.run_allocated:
        parser.error("Require an explicitly allocated exclusive hardware slot")
    if args.final_block_count is not None and not 1 <= args.final_block_count <= 50000:
        parser.error("Final-only capacity override must be 1–50000")
    if args.expect_capacity_failure and args.mode != "exact-rightsized":
        parser.error("Expected early capacity failure applies only to the exact allocator")
    args.session, args.checkpoint = args.session.resolve(strict=True), args.checkpoint.resolve(strict=True)
    args.output, args.solver_dll = args.output.resolve(), args.solver_dll.resolve(strict=True)
    try:
        args.output.relative_to(ROOT/"benchmark-output")
        args.checkpoint.relative_to(ROOT/"benchmark-output")
        args.solver_dll.relative_to(ROOT)
    except ValueError:
        parser.error("Use private benchmark-output reports/checkpoint and a workspace solve library")
    sidecar, trace_path, geometry_path = (args.output.with_suffix(suffix)
        for suffix in (".allocation.json", ".allocation.trace.jsonl", ".geometry.npz"))
    for path in (args.output, sidecar, trace_path, geometry_path):
        if path.exists():
            parser.error(f"Refuse overwriting immutable experiment artifact: {path}")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle field server before offline reconstruction")
    os.environ.update(PIPELINE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {name: file_hash(ROOT/name) for name in EXPERIMENT_ARTIFACTS}
    checkpoint_artifacts = {name: artifacts[name] for name in FINISH_ARTIFACTS}
    source_before, raw_before = source_hash(), file_hash(args.session)
    checkpoint_record = {"path": str(args.checkpoint), "sha256": file_hash(args.checkpoint)}
    components = {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name,path in
                  (("synthetic", args.component_synthetic), ("bridge", args.component_bridge))}
    report = {"kind": KIND, "mode": args.mode, "status": "running", "start_utc": now(),
        "source_sha256": source_before, "archive_sha256": raw_before, "artifacts_sha256": artifacts,
        "checkpoint_producer_artifacts_sha256": checkpoint_artifacts, "checkpoint": checkpoint_record,
        "component_proof": components, "expected_capacity_failure": args.expect_capacity_failure,
        "geometry_quality_proven": False, "registration_implementation": "original legacy CPU",
        "measurement": "New allocation scope. Materialization/proof/array parity preflight outside Finish. Original CPU registration, output-only gates, declared proposal-only thread policy, both modes' isolated exact planner, allocation setup and boundary evidence all charged inside Finish. Attribute bytes are a logical floor, not sampled peak GPU memory. No prior resident proof or quality approval is relabelled."}

    def save():
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    save()
    trace_file = scope = allocator = observer = proposal_scope = engine = binding = None
    trace_rows = 0

    def emit(row):
        nonlocal trace_rows
        trace_file.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
        trace_file.flush()
        trace_rows += 1

    def validate_policy():
        if cv2.getNumThreads() != 20 or o3d.utility.get_max_threads() != 20 or any(
                os.environ.get(name) != value for name,value in PIPELINE.items()):
            raise FinishResearchFailure("Original thread/environment policy drifted")

    def close_report():
        if trace_file is not None and not trace_file.closed:
            trace_file.close()
        if scope is not None:
            report["registration"] = scope.report()
        if allocator is not None:
            report["allocation"] = allocator.report()
        if observer is not None:
            report["final_boundary"] = {"calls": observer.calls, "hooks_restored": observer.restored,
                "scope": "Same Python/native owners, exact accepted poses, counts and Live hashmap capacity/size at Final entry/return; not a copied/hash-verified full logical VBG comparison"}
        if proposal_scope is not None:
            report["ransac_proposals"] = proposal_scope.report()
        report.update(source_sha256_after=source_hash(), archive_sha256_after=file_hash(args.session),
            artifacts_sha256_after={name: file_hash(ROOT/name) for name in artifacts},
            checkpoint_after={"path": str(args.checkpoint), "sha256": file_hash(args.checkpoint)},
            component_proof_after={name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name,path in
                (("synthetic", args.component_synthetic), ("bridge", args.component_bridge))})
        if binding is not None:
            validate_policy()
            report["runtime_binding_after"] = current_component_binding(template, args.solver_dll, cp, np, o3d, cv2)
            validate_checkpoint_manifest(args.checkpoint, binding, checkpoint_artifacts)
        if trace_path.exists():
            report["trace"] = {"path": str(trace_path), "sha256": file_hash(trace_path), "rows": trace_rows}
        for key in ("source_sha256", "archive_sha256", "artifacts_sha256", "checkpoint", "component_proof", "runtime_binding"):
            if key in report and report[key] != report[key+"_after"]:
                raise FinishResearchFailure("Closed allocation source/raw/checkpoint/component/runtime changed: "+key)
        report["end_utc"] = now()
        save()

    try:
        import cupy as cp
        import cv2
        import numpy as np
        import open3d as o3d
        from scanner_server.engine import ScanEngine
        from scanner_server import cuda_registration, refinement, fragments, bundle_adjustment
        from scripts.profile_session import geometry_summary, stage_summary
        from scripts.process_metrics import finish_cuda_worker, gpu_info, peak_rss_bytes
        from scripts.research.profile_checkpoint_resident_finish import GlobalSeedThreadScope
        from scripts.research.validate_device_flat_grid_proof import validate_grid_proof
        from scripts.research.validate_checkpoint_finish_proof import validate_checkpoint_manifest, validate_proposal_policy
        cv2.setNumThreads(20)
        o3d.utility.set_max_threads(20)
        template = json.loads(args.component_synthetic.read_text(encoding="utf-8"))
        component_bridge = json.loads(args.component_bridge.read_text(encoding="utf-8"))
        binding = current_component_binding(template, args.solver_dll, cp, np, o3d, cv2)
        validate_grid_proof(args.component_synthetic, args.component_bridge, binding, component_bridge["fixture_binding"])
        report["runtime_binding"] = binding
        authority = validate_checkpoint_manifest(args.checkpoint, binding, checkpoint_artifacts)
        checkpoint = authority.manifest
        base = checkpoint["scope_base"]
        if Path(base["archive"]["path"]).resolve() != args.session or base["archive"]["sha256"] != raw_before:
            raise FinishResearchFailure("Requested archive differs from the measured raw checkpoint")
        threads = validate_proposal_policy(base)
        cv2.setRNGSeed(base["seed"])
        o3d.utility.random.seed(base["seed"])
        # The codec restores captured Python/NumPy RNG states; do not replace
        # those with exported poses or independently replay/re-fuse Live data.
        engine, verification = materialize_checkpoint(args.checkpoint, expected_binding=binding,
                                                     expected_artifacts=checkpoint_artifacts)
        if engine.unprocessed_count:
            raise FinishResearchFailure("Checkpoint still contains unprocessed raw input")
        if args.final_block_count is not None:
            engine.settings = replace(engine.settings, final_block_count=args.final_block_count)
        if engine.settings.final_voxel_m is None or engine._final_vbg is not None:
            raise FinishResearchFailure("Require a new explicit Final model from the closed Live checkpoint")
        report.update(checkpoint_verification=verification, checkpoint_scope_base=base,
            execution_settings=json.loads(json.dumps(engine.settings.to_dict(), allow_nan=False)),
            final_only_override={"final_block_count": args.final_block_count} if args.final_block_count is not None else {},
            research_ransac_threads=threads, accepted_before_finish=engine.frame_count,
            accepted_indices_before_finish=[index for index,_ in engine.poses],
            live_pose_sha256_before_finish=pose_sha256(np, engine.poses))
        report["experiment_scope_sha256"] = canonical_hash({key: report[key] for key in
            ("checkpoint", "checkpoint_scope_base", "execution_settings", "final_only_override", "artifacts_sha256")})
        validate_policy()
        # Capture class implementation BEFORE any instance observation hook.
        original_final = ScanEngine._final_volume
        report["original_method_ast_sha256"] = validate_original_functions(engine, original_final)
        trace_file = trace_path.open("x", encoding="utf-8", newline="\n")
        live_diagnostics, live_stages = copy.deepcopy(engine.diagnostics), dict(engine.stage_totals_ms)
        save()
        started = time.perf_counter()
        allocator_class = RightSizedFinalScope if args.mode == "exact-rightsized" else ObservedOriginalFinalScope
        allocator = allocator_class(engine, original_final=original_final)
        observer = FinalBoundaryObserver(engine, lambda owner: boundary_state(owner, np))
        scope = FinishRegistrationScope(cuda_registration, refinement._match, None, mode="native", trace=emit)
        with ExitStack() as stack:
            stack.enter_context(allocator)
            stack.enter_context(observer)
            stack.enter_context(scope)
            for name in ("_local_match", "_global_seed", "_verify_bridge", "_verify_partial_bridge", "_verify_visual_bridge",
                         "_validate_bridge_pose", "_pair", "_heldout", "_visual_witness", "propose_fragment_poses"):
                if hasattr(fragments, name):
                    scope.observe(fragments, name, "fragments."+name)
            for name in ("_trustworthy", "_heldout", "_cost", "_distance", "propose_poses"):
                if hasattr(refinement, name):
                    scope.observe(refinement, name, "refinement."+name)
            for name in ("_supported", "validate_depth", "propose_bundle_poses"):
                if hasattr(bundle_adjustment, name):
                    scope.observe(bundle_adjustment, name, "bundle_adjustment."+name)
            scope.observe(refinement.REG, "get_information_matrix_from_point_clouds", "original_information_matrix")
            proposal_scope = GlobalSeedThreadScope(fragments, o3d.utility, threads=threads)
            stack.enter_context(proposal_scope)
            def progress(current, total, result):
                print(f"Finish {current}/{total}: {result.get('message', '')}", flush=True)
            built, result = engine.build_mesh(progress_cb=progress)
            proposal_scope.healthy()
            scope.finish()
        finish_s = time.perf_counter()-started
        if not (scope.complete and scope.restored and allocator.restored and observer.restored):
            raise FinishResearchFailure("Original registration/allocation/observation hooks did not complete and restore")
        if len(observer.calls) != 1:
            raise FinishResearchFailure("Original Finish did not reach exactly one observed Final allocation")
        if args.expect_capacity_failure:
            if (built or len(allocator.plans) != 1 or allocator.plans[0].capacity_fits or allocator.allocations
                    or observer.calls[0].get("failure", {}).get("type") != "FinalCapacityError"
                    or not observer.calls[0].get("owner_and_pose_preserved") or engine._final_vbg is not None):
                raise FinishResearchFailure("Expected early full-count capacity failure did not preserve the Live boundary")
            report["outcome"] = "expected-early-capacity-failure"
        elif not built:
            raise FinishResearchFailure("Original Finish did not produce its independently testable Final mesh")
        else:
            report["outcome"] = "mesh-built-quality-unproven"
        reconstruction = engine.reconstruction_report()
        value = {"schema_version":2, "session":args.session.name, "input_sha256":raw_before,
            "input_changed_during_profile":file_hash(args.session) != raw_before,
            "source_sha256":source_before, "source_changed_during_profile":source_hash() != source_before,
            "selected_indices":base["selected_indices"], "seed":base["seed"], "frames":len(base["selected_indices"]),
            "settings":report["execution_settings"], "pipeline_options":dict(PIPELINE),
            "research_pipeline_options":base["pipeline_options"], "thread_policy":dict(THREAD_POLICY),
            "finish_requested":True, "pose_seeds_used":False, "mesh_built":built, "build_result":result,
            "accepted_before_finish":report["accepted_before_finish"],
            "accepted_indices_before_finish":report["accepted_indices_before_finish"],
            "accepted":engine.frame_count, "accepted_indices":[row["index"] for row in reconstruction["poses"]],
            "poses":reconstruction["poses"], "unrounded_final_pose_sha256":pose_sha256(np, engine.poses),
            "live_s":None, "finish_s":finish_s, "fresh_live_measured_s":checkpoint["capture_report"]["fresh_live_measured_s"],
            "checkpoint":checkpoint_record, "checkpoint_verification":verification,
            "live_stages":stage_summary(live_diagnostics, live_stages, np), "live_diagnostics":live_diagnostics,
            "all_stages":stage_summary(engine.diagnostics, engine.stage_totals_ms, np), "diagnostics":engine.diagnostics,
            "fragment_reconnection":engine.fragment_reconnection, "refinement":engine.refinement,
            "bundle_adjustment":engine.bundle_adjustment, "final_reconstruction":engine.final_reconstruction,
            "geometry":geometry_summary(engine, geometry_path, np), "backend":engine.backend,
            "gpu_hardware":gpu_info(), "peak_process_rss_bytes":peak_rss_bytes(),
            "research_finish":{"kind":KIND, "mode":args.mode, "helper_artifacts_sha256":artifacts,
                "registration_implementation":"original legacy CPU", "geometry_quality_proven":False},
            "measurement":report["measurement"]}
        args.output.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        report.update(finish_s=finish_s, unrounded_final_pose_sha256=value["unrounded_final_pose_sha256"],
            profile={"path":str(args.output), "sha256":file_hash(args.output),
                     "geometry_path":str(geometry_path), "geometry_sha256":file_hash(geometry_path)})
        o3d.core.cuda.synchronize()
        report["status"] = "complete"
        close_report()
        print(json.dumps({"finish_s":finish_s, "outcome":report["outcome"], "allocation":report["allocation"]}), flush=True)
        if sys.platform == "win32":
            finish_cuda_worker()
    except BaseException as error:
        report.update(status="failed", failure={"type":type(error).__name__, "message":str(error)})
        try:
            close_report()
        except BaseException as secondary:
            report.setdefault("cleanup_failures", []).append({"type":type(secondary).__name__, "message":str(secondary)})
            error.add_note(f"Allocation experiment closure also failed: {secondary}")
        try:
            save()
        except BaseException as secondary:
            error.add_note(f"Allocation failure report write also failed: {secondary}")
        raise
    finally:
        if trace_file is not None and not trace_file.closed:
            primary = sys.exception()
            try:
                trace_file.close()
            except BaseException as secondary:
                if primary is not None:
                    primary.add_note(f"Final trace cleanup also failed: {secondary}")
                else:
                    report.update(status="failed", failure={"type":type(secondary).__name__, "message":str(secondary)})
                    try:
                        save()
                    except BaseException as write_error:
                        secondary.add_note(f"Final cleanup failure write also failed: {write_error}")
                    raise


if __name__ == "__main__":
    main()
