"""Capture actual fresh raw Live once, then compare complete Finish from it.

Capture retains original raw/prepared images, measured poses/decisions, CPU
model/preview/feature order, logical VBG keys/attributes/capacity and session
transaction state. Native/audit/timing materialize that private checkpoint;
they neither reuse exported poses as Live seeds nor rebuild its volume by
re-fusion. No production source or gate is replaced beyond scoped registration.
All numerical work requires an explicitly allocated, idle-server hardware slot.
"""

from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash
from scripts.research.finish_resident_registration import FinishRegistrationScope, FinishResearchFailure
from scripts.research.profile_resident_finish import PIPELINE
from scripts.research.private_live_checkpoint import (
    PreparedInputRecorder, export_checkpoint, materialize_checkpoint, pose_sha256)
from scripts.research.validate_finish_resident_proof import THREAD_POLICY
from scripts.research.archive.validate_uniform_grid_proof import canonical_hash

KIND = "offline-full-finish-device-flat-resident-checkpoint-proposal-threads-v3"
RANSAC_POLICY = "original-fragment-global-seed-scoped-open3d-threads-v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_scope(value):
    """Use the exact finite JSON value schema before capture or comparison.

    Dataclass calibration/settings contain tuples. JSON represents these as
    lists, so normalize the new scope at creation rather than relaxing the
    checkpoint validator's exact comparison after serialization.
    """
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


class GlobalSeedThreadScope:
    """Change only original global RANSAC's declared Open3D thread budget.

    Seeds/checkers/caps/convergence remain the original call. Every call
    restores the normal twenty-thread setting before any subsequent ICP.
    Policy/setup/restoration evidence faults are hard-latched independently
    of ordinary original proposal rejection or exceptions.
    """

    def __init__(self, fragments, utility, *, threads):
        if type(threads) is not int or threads not in (1, 20):
            raise ValueError("Research RANSAC threads must be exactly 1 or 20")
        self.module, self.utility, self.threads = fragments, utility, threads
        self.original = fragments._global_seed
        self.calls, self.failure, self.active, self.restored = [], None, False, True

    def fail(self, message, cause=None):
        if self.failure is None:
            self.failure = FinishResearchFailure(message)
            if cause is not None:
                self.failure.__cause__ = cause
        raise self.failure

    def healthy(self):
        if self.failure is not None:
            raise self.failure

    def call(self, source, target, seed):
        self.healthy()
        started = time.perf_counter()
        row = {"invocation": len(self.calls), "source": source.index, "target": target.index,
            "original_seed": seed, "requested_threads": self.threads, "complete": False,
            "restored_threads": None}
        self.calls.append(row)
        try:
            if self.utility.get_max_threads() != 20:
                self.fail("Original ICP/Open3D thread budget drifted before RANSAC")
            self.utility.set_max_threads(self.threads)
            if self.utility.get_max_threads() != self.threads:
                self.fail("Declared original global RANSAC thread setting was not applied")
        except FinishResearchFailure:
            try:
                self.utility.set_max_threads(20)
            except BaseException as secondary:
                self.failure.add_note(f"RANSAC setup restoration also failed: {secondary}")
            raise
        except BaseException as error:
            try:
                self.utility.set_max_threads(20)
            except BaseException as secondary:
                error.add_note(f"RANSAC setup restoration also failed: {secondary}")
            self.fail("Original RANSAC thread setup failed", error)
        try:
            result = self.original(source, target, seed)
            row.update(complete=True, proposal_present=result is not None)
            return result
        except BaseException as error:
            row["original_exception"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            primary = sys.exception()
            try:
                self.utility.set_max_threads(20)
                row["restored_threads"] = self.utility.get_max_threads()
                if row["restored_threads"] != 20:
                    self.fail("Original twenty-thread ICP budget was not restored after RANSAC")
                row["wall_s"] = time.perf_counter()-started
            except BaseException as error:
                if primary is not None:
                    primary.add_note(f"Original RANSAC thread restoration also failed: {error}")
                if self.failure is None:
                    self.failure = FinishResearchFailure("Original RANSAC thread restoration failed")
                    self.failure.__cause__ = primary if primary is not None else error
                row["thread_restore_exception"] = {"type": type(error).__name__, "message": str(error)}
                raise self.failure

    def __enter__(self):
        if self.active:
            raise FinishResearchFailure("RANSAC thread scope cannot be reentered")
        self.healthy()
        if self.utility.get_max_threads() != 20:
            self.fail("Require original twenty-thread Open3D state before scoped proposals")
        try:
            self.module._global_seed = self.call
            self.active, self.restored = True, False
        except BaseException as error:
            try:
                self.module._global_seed = self.original
            except BaseException as secondary:
                error.add_note(f"RANSAC partial-entry restoration also failed: {secondary}")
            raise
        return self

    def __exit__(self, kind, error, traceback):
        failures = []
        for action in (lambda: setattr(self.module, "_global_seed", self.original),
                       lambda: self.utility.set_max_threads(20)):
            try:
                action()
            except BaseException as secondary:
                failures.append(secondary)
        try:
            if self.utility.get_max_threads() != 20:
                failures.append(RuntimeError("Original twenty-thread Open3D state was not restored"))
        except BaseException as secondary:
            failures.append(secondary)
        self.active, self.restored = False, not failures
        if failures and self.failure is None:
            self.failure = FinishResearchFailure("Original RANSAC scope restoration failed")
            self.failure.__cause__ = error if error is not None else failures[0]
        if self.failure is not None:
            for secondary in failures:
                self.failure.add_note(f"Additional RANSAC restoration error: {secondary}")
            raise self.failure
        return False

    def report(self):
        failure = None if self.failure is None else {"type": type(self.failure).__name__, "message": str(self.failure)}
        if failure is not None and self.failure.__cause__ is not None:
            failure["cause"] = {"type": type(self.failure.__cause__).__name__, "message": str(self.failure.__cause__)}
        return {"policy": RANSAC_POLICY, "requested_threads": self.threads, "outside_threads": 20,
            "calls": self.calls, "call_count": len(self.calls), "failure": failure, "hooks_restored": self.restored,
            "scope": "Only captured original fragments._global_seed uses requested Open3D threads; original seeds/checkers/12000 cap/.999 convergence unchanged. Every call restores20 before original ICP; OMP8/CV20 unchanged. Per-call wall includes thread switching and original proposal work, nested inside finish_s."}


def bulk_bridge_class(solver_library, scratch_bytes, query_bytes, bulk_authority, artifacts):
    """Change only the proven complete CPU auditor; inherit original GPU math."""
    from scripts.research.cuda_bulk_audit_grid_registration import BulkAuditDeviceFlatGridICP
    from scripts.research.research_device_grid_resident_icp import DeviceGridResidentICP
    from scripts.research.archive.research_resident_icp import EigenSolve
    from scripts.research.validate_bulk_resident_audit import BULK_ARTIFACTS

    class BulkResidentBridge(BulkAuditDeviceFlatGridICP):
        def __init__(self, **kwargs):
            super().__init__(max_query_bytes=query_bytes, audit_mode="bulk-shadow",
                bulk_authority=bulk_authority, bulk_chunk_rows=65536, **kwargs)
            try:
                self.solve = EigenSolve(solver_library)
                self.resident = DeviceGridResidentICP(self, self.solve, max_points=1000000,
                    max_scratch_bytes=scratch_bytes, gpu_timing=False, owns_retrieval=False)
                self.resident.provenance["actual_retrieval_sha256"].update({name: artifacts[name] for name in BULK_ARTIFACTS})
                self.resident.provenance["retrieval"] = "Original device-flat math; independently proven original legacy bulk CPU hit/miss shadows, scalar ambiguity resolution"
            except BaseException as error:
                try:
                    super().close()
                except BaseException as secondary:
                    error.add_note(f"Partial bulk resident construction cleanup also failed: {secondary}")
                raise

        def match(self, source, target, initial):
            started = time.perf_counter()
            try:
                return self.resident.match(source, target, initial)
            finally:
                self.statistics["icp_calls"] += 1
                self.statistics["icp_wall_s"] += time.perf_counter()-started

        def close(self):
            primary, failures = sys.exception(), []
            for action in (self.resident.close, super().close):
                try:
                    action()
                except BaseException as error:
                    failures.append(error)
            if failures:
                if primary is not None:
                    primary.add_note(f"Bulk resident cleanup also failed: {failures}")
                else:
                    for error in failures[1:]:
                        failures[0].add_note(f"Additional bulk resident cleanup failure: {error}")
                    raise failures[0]

    return BulkResidentBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Original raw ZIP, checked in every phase")
    parser.add_argument("--mode", choices=("capture", "native", "audit", "timing"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Capture: fresh directory; other modes: closed checkpoint.json")
    parser.add_argument("--output", type=Path, required=True, help="Fresh ignored JSON output")
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--component-synthetic", type=Path, required=True)
    parser.add_argument("--component-bridge", type=Path, required=True)
    parser.add_argument("--finish-audit", type=Path)
    parser.add_argument("--quality-proof", type=Path)
    parser.add_argument("--bulk-audit-proof", type=Path,
                        help="Audit: enable token-gated complete bulk CPU shadows; timing: validate its audited proof only")
    parser.add_argument("--solver-dll", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll")
    parser.add_argument("--final-block-count", type=int)
    parser.add_argument("--final-preplan", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ransac-threads", type=int, choices=(1, 20), default=1,
                        help="Declared original global proposal thread budget; all ICP remains Open3D20")
    args = parser.parse_args()
    if not args.run_allocated:
        parser.error("Require an explicitly allocated exclusive hardware slot")
    if args.final_block_count is not None and not 1 <= args.final_block_count <= 50000:
        parser.error("Matched Final block count must be 1–50000")
    if (args.mode == "timing") != bool(args.finish_audit and args.quality_proof):
        parser.error("Timing requires closed checkpoint Finish audit and separate quality authority")
    if args.mode != "timing" and (args.finish_audit or args.quality_proof):
        parser.error("Timing authority is not consumed by capture/native/audit")
    if args.bulk_audit_proof is not None and args.mode not in ("audit", "timing"):
        parser.error("Bulk proof is used only for audit or timing proof validation")
    args.session = args.session.resolve(strict=True)
    args.output, args.checkpoint = args.output.resolve(), args.checkpoint.resolve()
    try:
        args.output.relative_to(ROOT/"benchmark-output")
        args.checkpoint.relative_to(ROOT/"benchmark-output")
        args.solver_dll.resolve(strict=True).relative_to(ROOT)
    except (OSError, ValueError):
        parser.error("Private outputs/checkpoint must be ignored benchmark-output, solve DLL inside workspace")
    sidecar = args.output.with_suffix(".resident.json")
    trace_path = args.output.with_suffix(".resident.trace.jsonl")
    for path in (args.output, sidecar, trace_path, args.output.with_suffix(".geometry.npz")):
        if path.exists():
            parser.error(f"Require fresh artifacts; preserve {path}")
    if args.mode == "capture":
        if args.checkpoint.exists() and any(args.checkpoint.iterdir()):
            parser.error("Capture requires a fresh empty checkpoint directory")
    elif not args.checkpoint.is_file():
        parser.error("Finish requires a closed private checkpoint.json")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle field server before research replay")
    os.environ.update(PIPELINE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # New scope pins are distinct from immutable v1 and component artifacts.
    from scripts.research.validate_checkpoint_finish_proof import FINISH_ARTIFACTS
    artifacts = {name: file_hash(ROOT/name) for name in FINISH_ARTIFACTS}
    source_before, archive_before = source_hash(), file_hash(args.session)
    component_records = {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in
        (("synthetic", args.component_synthetic), ("bridge", args.component_bridge))}
    report = {"kind": KIND, "mode": args.mode, "status": "running", "start_utc": now(),
        "source_sha256": source_before, "archive_sha256": archive_before,
        "artifacts_sha256": artifacts, "component_proof": component_records,
        "final_budget_plan": [], "registration": None, "cpu_query_auditor": {"mode": "scalar"},
        "research_ransac_threads": args.ransac_threads, "ransac_proposals": None,
        "checkpoint_rng_policy": {"opaque_native_state_serialized": False,
            "opencv": "Explicit cv2.setRNGSeed(scope.seed) before capture or checkpoint materialization; legacy PnP source constructs its own fixed local RNG",
            "open3d": "Explicit utility.random.seed(scope.seed) before capture or materialization; original fragment RANSAC also reseeds each original proposal",
            "python_numpy": "Actual captured global states restored from the checkpoint before Finish",
            "scope": "Shared explicit native seed reset policy, not recovered opaque RNG state; exact ordered original call/gate proofs remain mandatory"},
        "performance_attribution_valid": args.mode in ("native", "timing"),
        "scope": "Complete original Finish from the same verified fresh raw-Live checkpoint under the explicitly declared scoped global-RANSAC thread protocol. Checkpoint materialization, payload/source/runtime preflight and prepared-image parity checks are outside Finish. Resident setup, ordered signatures, output-only gate observers, scoped proposal thread switching/work, optional exact block planning and whole-Finish timing-authority validation are charged inside finish_s. Original proposal seeds/checkers/caps/convergence unchanged; all ICP staysOpen3D20/OMP8 and CV20. Audit includes NN/result shadows and is not a speed measurement. No Live throughput claim from checkpoint Finish."}

    def save():
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    save()
    binding = template = bulk_template = bulk_binding = bulk_authority = engine = scope = solver = trace_file = ransac_scope = None
    original_final = engine_type = None
    trace_rows, finalized = 0, False

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

    def current_bulk_binding():
        from scripts.research.benchmark_bulk_resident_audit import current_runtime_binding
        from scripts.research.benchmark_parallel_fragments import numerical_source_hash
        from scripts.process_metrics import gpu_info
        current = copy.deepcopy(bulk_template)
        api_path = Path(bulk_template["bulk_api_proof"]["path"]).resolve(strict=True)
        current.update(source_sha256=source_hash(), component_source_sha256=numerical_source_hash(), gpu=gpu_info(),
            artifacts_sha256={name: file_hash(ROOT/name) for name in bulk_template["artifacts_sha256"]},
            versions={"numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__, "cupy": cp.__version__},
            solver_library_path=str(args.solver_dll.resolve()), thread_policy=dict(THREAD_POLICY),
            bulk_api_proof={"path": str(api_path), "sha256": file_hash(api_path)})
        return current_runtime_binding(current, cp, np, o3d)

    def memory_observation():
        # Allocator reservation is observed, never confused with the closed
        # graph's logical capacity or claimed to be byte-identical on restore.
        with cp.cuda.Device(0), cp.cuda.Stream.null:
            cp.cuda.Stream.null.synchronize()
            free, total = cp.cuda.runtime.memGetInfo()
            pool = cp.get_default_memory_pool()
            return {"device_free_bytes": int(free), "device_total_bytes": int(total),
                    "cupy_pool_used_bytes": int(pool.used_bytes()), "cupy_pool_reserved_bytes": int(pool.total_bytes()),
                    "scope": "Instantaneous device/global CuPy allocator observation; includes unrelated native/runtime reservations, not serialized private ownership or an independently sampled peak"}

    def validate_actual_policy():
        if (cv2.getNumThreads() != THREAD_POLICY["opencv"]
                or o3d.utility.get_max_threads() != THREAD_POLICY["open3d"]
                or any(os.environ.get(name) != value for name, value in PIPELINE.items())):
            raise FinishResearchFailure("Actual original thread/environment policy changed")

    def restore_hooks():
        if engine_type is not None and original_final is not None:
            engine_type._final_volume = original_final
        report["runner_hooks_restored"] = True

    def finalize():
        nonlocal finalized
        if finalized:
            return
        if ransac_scope is not None:
            report["ransac_proposals"] = ransac_scope.report()
        if trace_file is not None and not trace_file.closed:
            trace_file.close()
        if binding is not None:
            validate_actual_policy()
        report["source_sha256_after"] = source_hash()
        report["archive_sha256_after"] = file_hash(args.session)
        report["artifacts_sha256_after"] = {name: file_hash(ROOT/name) for name in artifacts}
        report["runtime_binding_after"] = current_binding() if binding is not None else None
        report["component_proof_after"] = {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in
            (("synthetic", args.component_synthetic), ("bridge", args.component_bridge))}
        if bulk_authority is not None:
            report["referenced_bulk_audit_proof_after"] = {"path": str(args.bulk_audit_proof.resolve()),
                "sha256": file_hash(args.bulk_audit_proof)}
            report["bulk_runtime_binding_after"] = current_bulk_binding()
            if (report["referenced_bulk_audit_proof_after"] != report["referenced_bulk_audit_proof"]
                    or report["bulk_runtime_binding_after"] != bulk_binding):
                raise FinishResearchFailure("Separate bulk CPU authority/current runtime changed during Finish")
            if args.mode == "audit":
                from scripts.research.bulk_legacy_nn_audit import runtime_binding as actual_native_runtime
                if actual_native_runtime(np, o3d) != report["cpu_query_auditor"]["native_runtime"]:
                    raise FinishResearchFailure("Actual bulk native CPU loader/configuration changed")
        if "checkpoint" in report:
            checkpoint_path = Path(report["checkpoint"]["path"])
            report["checkpoint_after"] = dict(report["checkpoint"], sha256=file_hash(checkpoint_path))
            # Revalidate payloads, not only the manifest. Materialization is
            # private and never writes into the closed input checkpoint.
            validate_checkpoint_manifest(checkpoint_path, binding, artifacts, report["checkpoint_scope_base"])
        if trace_path.exists():
            report["trace"] = {"path": str(trace_path), "sha256": file_hash(trace_path), "rows": trace_rows}
        if scope is not None:
            report["registration"] = scope.report()
        pairs = (("source_sha256", source_before), ("archive_sha256", archive_before),
                 ("artifacts_sha256", artifacts), ("runtime_binding", binding), ("component_proof", component_records))
        if any(report.get(key+"_after") != value for key, value in pairs):
            raise FinishResearchFailure("Production/raw/helper/runtime/component evidence changed")
        if "checkpoint" in report and report["checkpoint_after"] != report["checkpoint"]:
            raise FinishResearchFailure("Closed checkpoint changed during Finish")
        if report.get("failure") is None:
            if args.mode != "capture" and (scope is None or not scope.complete or not scope.restored or scope.failure is not None):
                raise FinishResearchFailure("Finish registration scope did not complete and restore")
            value = json.loads(args.output.read_text(encoding="utf-8"))
            if args.mode != "capture" and not value["mesh_built"]:
                raise FinishResearchFailure("Original Finish did not produce its completed triangle mesh")
            report["profile"] = {"path": str(args.output), "sha256": file_hash(args.output)}
            if args.mode != "capture":
                geometry = Path(value["geometry"]["artifact"])
                report["profile"].update(geometry_path=str(geometry), geometry_sha256=file_hash(geometry))
            report["status"] = "complete"
        report["end_utc"] = now()
        save()
        finalized = True

    try:
        import cupy as cp
        import cv2
        import numpy as np
        import open3d as o3d
        from PIL import Image
        from scanner_server.engine import ScanEngine
        from scanner_server import cuda_registration, refinement, fragments, bundle_adjustment
        from shared.settings import ScanSettings
        from scripts.profile_session import geometry_summary, stage_summary
        from scripts.process_metrics import finish_cuda_worker, gpu_info, peak_rss_bytes
        from scripts.research.benchmark_device_grid_resident import bridge_class
        from scripts.research.validate_device_flat_grid_proof import validate_grid_proof
        from scripts.research.validate_checkpoint_finish_proof import validate_checkpoint_manifest
        cv2.setNumThreads(20)
        o3d.utility.set_max_threads(20)
        cv2.setRNGSeed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        o3d.utility.random.seed(args.seed)
        validate_actual_policy()
        template = json.loads(args.component_synthetic.read_text(encoding="utf-8"))
        bridge = json.loads(args.component_bridge.read_text(encoding="utf-8"))
        binding = current_binding()
        component_authority = validate_grid_proof(args.component_synthetic, args.component_bridge, binding, bridge["fixture_binding"])
        report["runtime_binding"] = binding
        report["component_authority"] = {"bindings_sha256": component_authority.bindings_sha256,
            "target_membership_count": len(component_authority.target_digests),
            "scope": "Component numerical preflight only; new field targets/trajectory need fresh audit"}
        if args.bulk_audit_proof is not None:
            from scripts.research.validate_bulk_resident_audit import validate_bulk_resident_audit, POLICY
            from scripts.research.bulk_legacy_nn_audit import runtime_binding as actual_native_runtime
            bulk_template = json.loads(args.bulk_audit_proof.read_text(encoding="utf-8"))
            bulk_binding = current_bulk_binding()
            if (bulk_template["fixture_binding"] != bridge["fixture_binding"]
                    or bulk_binding["resident_configuration"] != binding["resident_configuration"]
                    or bulk_binding["cache_policy"] != binding["cache_policy"]):
                raise FinishResearchFailure("Bulk proof differs from original component/math/cache configuration")
            bulk_authority = validate_bulk_resident_audit(args.bulk_audit_proof, bulk_binding, bridge["fixture_binding"])
            report["referenced_bulk_audit_proof"] = {"path": str(args.bulk_audit_proof.resolve()),
                "sha256": file_hash(args.bulk_audit_proof)}
            report["bulk_runtime_binding"] = bulk_binding
            if args.mode == "audit":
                report["cpu_query_auditor"] = {"mode": "bulk-shadow", "policy": POLICY,
                    "proof": report["referenced_bulk_audit_proof"], "api_proof": dict(bulk_binding["bulk_api_proof"]),
                    "native_runtime": actual_native_runtime(np, o3d), "chunk_rows": 65536}
        report["memory_before_phase"] = memory_observation()
        with zipfile.ZipFile(args.session) as archive:
            info = archive.getinfo("manifest.json")
            if info.file_size > 16*1024**2:
                raise ValueError("Raw manifest exceeds bounded scope")
            manifest = json.loads(archive.read(info))
            settings = ScanSettings.from_dict(manifest["settings"])
            if args.final_block_count is not None:
                settings = replace(settings, final_block_count=args.final_block_count)
            selected = list(range(len(manifest["frames"])))
            if not selected or len(selected) > ScanEngine.MAX_FRAMES:
                raise ValueError("Require complete bounded original raw archive")
            scope_base = canonical_json_scope({"archive": {"path": str(args.session), "sha256": archive_before},
                "selected_indices": selected, "seed": args.seed, "settings": settings.to_dict(),
                "pipeline_options": dict(PIPELINE, research_final_preplan=args.final_preplan,
                    research_ransac_threads=args.ransac_threads),
                "thread_policy": dict(THREAD_POLICY), "runtime_binding": binding})
            if args.mode == "capture":
                frames = []
                for item in manifest["frames"]:
                    with archive.open(item["rgb"]) as stream, Image.open(stream) as image:
                        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
                    with archive.open(item["depth"]) as stream, Image.open(stream) as image:
                        depth = np.array(image, dtype=np.uint16)
                    frames.append((rgb, depth, item))
        report["checkpoint_scope_base"] = scope_base
        if args.mode == "capture":
            engine = ScanEngine(device="cuda", tracking="legacy")
            engine.reset(settings=settings)
            started = time.perf_counter()
            with PreparedInputRecorder(engine) as recorder:
                for index, (rgb, depth, item) in enumerate(frames):
                    metadata = dict(item.get("metadata", {}))
                    metadata.pop("reference_pose", None)
                    metadata.update(frame_id=index, timestamp_s=item["timestamp_s"])
                    result = engine.store_frame(rgb, depth, metadata)
                    if not result["success"]:
                        raise ValueError("Original raw storage failed: "+result["message"])
                    engine.process_frames()
                    print(f"Capture {index+1}/{len(frames)} accepted={engine.frame_count}", flush=True)
            live_s = time.perf_counter()-started
            capture = {"fresh_live_measured_s": live_s,
                "measurement": "Original storage/Live plus output observer copies; excludes decoding, checkpoint serialization and imports. Observer overhead prevents this from replacing independent Live throughput controls.",
                "backend": engine.backend, "stored_count": engine.stored_count,
                "accepted_indices": [index for index, _ in engine.poses],
                "diagnostics": engine.diagnostics, "live_stages": stage_summary(engine.diagnostics, engine.stage_totals_ms, np),
                "source_sha256": source_before, "archive_sha256": archive_before,
                "observation_healthy": recorder.failure is None,
                "prepared_actual_output_count": len(recorder.inputs)}
            capture["memory_after_live"] = memory_observation()
            path, checkpoint = export_checkpoint(engine, recorder.inputs, args.checkpoint,
                runtime_binding=binding, scope_base=scope_base, artifacts=artifacts, capture_report=capture)
            validate_checkpoint_manifest(path, binding, artifacts, scope_base)
            report["checkpoint"] = {"path": str(path), "sha256": file_hash(path),
                "logical_state_sha256": checkpoint["logical_state_sha256"]}
            capture.update(checkpoint=report["checkpoint"], checkpoint_snapshot_wall_s=checkpoint["snapshot_wall_s"],
                input_changed_during_profile=file_hash(args.session) != archive_before,
                source_changed_during_profile=source_hash() != source_before)
            args.output.write_text(json.dumps(capture, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            restore_hooks()
            finalize()
            print(json.dumps({"checkpoint": report["checkpoint"], "capture_s": live_s}), flush=True)
        else:
            checkpoint_authority = validate_checkpoint_manifest(args.checkpoint, binding, artifacts, scope_base)
            engine, verification = materialize_checkpoint(args.checkpoint, expected_binding=binding, expected_artifacts=artifacts)
            checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
            if checkpoint["scope_base"] != scope_base:
                raise FinishResearchFailure("Requested raw/settings/environment differ from captured checkpoint")
            report["checkpoint"] = {"path": str(args.checkpoint), "sha256": file_hash(args.checkpoint),
                "logical_state_sha256": checkpoint["logical_state_sha256"]}
            report["checkpoint_verification"] = verification
            report["memory_after_materialization"] = memory_observation()
            live = checkpoint["fresh_live"]
            report["scope_binding"] = dict(scope_base, live={
                "accepted_indices": live["accepted_indices"], "decisions_sha256": live["semantic_decisions_sha256"],
                "poses_sha256": live["poses_sha256"], "raw_diagnostics_sha256": live["raw_diagnostics_sha256"],
                "raw_frames_sha256": live["raw_frames_sha256"], "prepared_inputs_sha256": live["prepared_inputs_sha256"],
                "unprocessed_count": 0, "pose_source": "fresh raw Live replay"}, checkpoint=report["checkpoint"])
            report["scope_binding_sha256"] = canonical_hash(report["scope_binding"])
            report["measured_live_poses"] = [{"index": index, "pose": np.asarray(pose).tolist()} for index, pose in engine.poses]
            trace_file = trace_path.open("x", encoding="utf-8", newline="\n")

            def emit(row):
                nonlocal trace_rows
                trace_file.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
                trace_file.flush()
                trace_rows += 1

            engine_type, original_final = ScanEngine, ScanEngine._final_volume

            def final(value, *positional, **keywords):
                if args.final_preplan:
                    started = time.perf_counter()
                    planner = copy.copy(value)
                    voxel = value.settings.final_voxel_m or value.settings.voxel_m
                    planner.settings = replace(value.settings, voxel_m=voxel)
                    planner.voxel_size = voxel
                    required = planner._required_fusion_blocks(value.poses, stage="final_budget_plan")
                    report["final_budget_plan"].append({"required_blocks": required, "wall_s": time.perf_counter()-started,
                        "budget_blocks": value.settings.final_block_count, "effective_voxel_m": voxel,
                        "unrounded_pose_sha256": pose_sha256(np, value.poses),
                        "authority": "Observation only; original Final delegated unchanged; one-block unactivated scratch"})
                    save()
                return original_final(value, *positional, **keywords)

            ScanEngine._final_volume = final
            live_diagnostics, live_stages = copy.deepcopy(engine.diagnostics), dict(engine.stage_totals_ms)
            accepted_before = engine.frame_count
            before_indices = [index for index, _ in engine.poses]
            save()
            finish_started = time.perf_counter()
            validate_actual_policy()
            finish_authority = registration_authority = None
            if args.mode == "timing":
                from scripts.research.validate_checkpoint_finish_proof import validate_checkpoint_finish_proof
                started = time.perf_counter()
                finish_authority = validate_checkpoint_finish_proof(args.finish_audit, component_authority,
                    report["scope_binding"], artifacts, quality_path=args.quality_proof,
                    checkpoint_authority=checkpoint_authority, bulk_authority=bulk_authority)
                registration_authority = finish_authority.registration_authority
                report["finish_authority_validation_s"] = time.perf_counter()-started
            if args.mode != "native":
                started = time.perf_counter()
                if args.mode == "audit" and bulk_authority is not None:
                    Bridge = bulk_bridge_class(args.solver_dll, binding["resident_configuration"]["max_scratch_bytes"],
                        binding["resident_configuration"]["max_query_bytes"], bulk_authority, artifacts)
                else:
                    Bridge = bridge_class(args.solver_dll, binding["resident_configuration"]["max_scratch_bytes"],
                                          binding["resident_configuration"]["max_query_bytes"])
                solver = Bridge(device="CUDA:0", max_clouds=binding["cache_policy"]["max_clouds"],
                    max_cache_bytes=binding["cache_policy"]["retained_gpu_bytes"], audit_nearest=args.mode == "audit",
                    audit_misses=args.mode == "audit", miss_policy="direct-miss-research-v1", proof_authority=registration_authority)
                report["resident_setup_s"] = time.perf_counter()-started
            scope = FinishRegistrationScope(cuda_registration, refinement._match, solver,
                mode=args.mode, trace=emit, authority=registration_authority)
            with scope:
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

                def progress(current, total, result):
                    print(f"Finish {current}/{total}: {result.get('message', '')}", flush=True)

                ransac_scope = GlobalSeedThreadScope(fragments, o3d.utility, threads=args.ransac_threads)
                with ransac_scope:
                    built, result = engine.build_mesh(progress_cb=progress)
                    ransac_scope.healthy()
                report["ransac_proposals"] = ransac_scope.report()
                scope.finish()
            finish_s = time.perf_counter()-finish_started
            restore_hooks()
            reconstruction = engine.reconstruction_report()
            extension_spec = importlib.util.find_spec("_kinect_native")
            extension_path = Path(extension_spec.origin) if extension_spec and extension_spec.origin else None
            value = {"schema_version": 2, "session": args.session.name, "input_sha256": archive_before,
                "input_changed_during_profile": file_hash(args.session) != archive_before,
                "source_sha256": source_before, "source_changed_during_profile": source_hash() != source_before,
                "native_extension": {"path": str(extension_path) if extension_path else None,
                    "sha256": file_hash(extension_path) if extension_path else None, "changed_during_profile": False},
                "selected_indices": selected, "seed": args.seed, "frames": len(selected),
                "native_mode": os.environ["KINECT_NATIVE"], "omp_threads": "8", "thread_policy": dict(THREAD_POLICY),
                "gpu_hardware": gpu_info(), "initial_blocks": os.environ["KINECT_BLOCK_COUNT"],
                "pipeline_options": dict(PIPELINE), "research_pipeline_options": scope_base["pipeline_options"],
                "research_scope_binding_sha256": report["scope_binding_sha256"], "backend": engine.backend,
                "settings": settings.to_dict(), "settings_overrides": {"final_block_count": args.final_block_count} if args.final_block_count else {},
                "versions": {"python": sys.version.split()[0], "open3d": o3d.__version__, "opencv": cv2.__version__, "numpy": np.__version__},
                "live_s": None, "finish_s": finish_s, "processing_s": None,
                "fresh_live_measured_s": checkpoint["capture_report"]["fresh_live_measured_s"],
                "checkpoint": report["checkpoint"], "checkpoint_verification": verification,
                "accepted_before_finish": accepted_before, "accepted_indices_before_finish": before_indices,
                "accepted": engine.frame_count, "accepted_indices": [row["index"] for row in reconstruction["poses"]],
                "finish_requested": True, "pose_seeds_used": False, "mesh_built": built, "build_result": result,
                "live_stages": stage_summary(live_diagnostics, live_stages, np), "live_diagnostics": live_diagnostics,
                "all_stages": stage_summary(engine.diagnostics, engine.stage_totals_ms, np), "poses": reconstruction["poses"],
                "diagnostics": engine.diagnostics, "fragment_reconnection": engine.fragment_reconnection,
                "refinement": engine.refinement, "bundle_adjustment": engine.bundle_adjustment,
                "final_reconstruction": engine.final_reconstruction, "peak_process_rss_bytes": peak_rss_bytes(),
                "geometry": geometry_summary(engine, args.output.with_suffix(".geometry.npz"), np),
                "research_finish": {"mode": args.mode, "helper_artifacts_sha256": artifacts,
                    "registration_implementation": "original legacy CPU" if args.mode == "native" else "offline device-flat resident research",
                    "production_backend_scope": "Backend describes original production policy; scoped Finish separately recorded"},
                "measurement": report["scope"]}
            args.output.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            report["finish_s"] = finish_s
            report["registration"] = scope.report()
            report["memory_after_finish"] = memory_observation()
            finalize()
            print(json.dumps({"finish_s": finish_s, "mesh_built": built, "checkpoint_verification": verification,
                              "registration_calls": report["registration"]["calls"]}), flush=True)
        if sys.platform == "win32":
            o3d.core.cuda.synchronize()
            finish_cuda_worker()
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        try:
            restore_hooks()
            finalize()
        except BaseException as secondary:
            report.setdefault("cleanup_failures", []).append({"type": type(secondary).__name__, "message": str(secondary)})
            error.add_note(f"Checkpoint Finish closure also failed: {secondary}")
        # A late failure after successful closure, or a closure save failure,
        # must replace local status while retaining the actual primary error.
        try:
            save()
        except BaseException as write_error:
            error.add_note(f"Failed diagnostics write: {write_error}")
        raise
    finally:
        restore_hooks()
        if trace_file is not None and not trace_file.closed:
            trace_file.close()


if __name__ == "__main__":
    main()
