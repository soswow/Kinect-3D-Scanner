"""Audit one complete original fragment-pair proposal set using graph4 ICP.

Fresh raw-derived fixtures only. Serial proposal jobs preserve all original
forward/reverse/camera/witness/gate steps and ordered ambiguity consumption.
Every actual candidate match has an exhaustive NN audit and original native
ICP shadow on its exact actual input and unrounded seed. A separate registered
permit from that closed audit authorizes the identical ordered trajectory for
timing. Both modes grant only prepared-pair component authority.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import microbatch_bridge_scope as scope

KIND = "gpu-icp-complete-bridge-proposal-audit-v1"
TIMING_KIND = "gpu-icp-complete-bridge-proposal-timing-v1"
CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"
ENVIRONMENT = {"OMP_NUM_THREADS": "8", "KINECT_NATIVE": "on", "KINECT_CUDA_REGISTRATION": "cpu"}
OWN_FILES = ("scripts/research/microbatch_bridge_scope.py",
    "scripts/research/microbatch_bridge_driver.py", "tests/test_microbatch_bridge.py")
MAX_CLOUDS = 34
CACHE_BYTES = 256*1024**2
TOTAL_BYTES = 512*1024**2


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--mode", choices=("audit", "timing"), default="audit")
    p.add_argument("--proof", type=Path)
    p.add_argument("--run-allocated", action="store_true")
    args = p.parse_args()
    args.capture, args.output = args.capture.resolve(), args.output.resolve()
    if (not args.run_allocated or not 0 <= args.task <= 7 or args.output.exists()
            or args.mode == "timing" and args.proof is None
            or not args.output.is_relative_to(ROOT/"benchmark-output")):
        p.error("Require exclusive hardware allocation, bounded task and fresh private output")
    with socket.socket() as check:
        check.settimeout(.3)
        if check.connect_ex(("127.0.0.1", 8000)) == 0:
            p.error("Stop the idle field server before this offline experiment")
    return args


def all_train_clouds(source, target):
    clouds = {}
    for fragment in (source, target):
        for cloud in [fragment.train]+[v.train for v in fragment.keys]:
            clouds[id(cloud)] = cloud
    if not 2 <= len(clouds) <= MAX_CLOUDS:
        raise scope.BridgeFailure("Bounded immutable aggregate/view cloud inventory required")
    return list(clouds.values())


def packet_descriptor(np, packet):
    """Direct misses legitimately carry positive-infinite FP64 minima."""
    array = np.asarray(packet)
    if (array.dtype != np.float64 or array.ndim != 2 or array.shape[1] != 8
            or not np.isfinite(array[:, :6]).all()
            or not ((np.isfinite(array[:, 6:]) & (array[:, 6:] >= 0)) | (array[:, 6:] == np.inf)).all()
            or (array[:, 6] > array[:, 7]).any()):
        raise scope.BridgeFailure("Malformed audited NN packet cannot enter trace")
    return {"dtype": array.dtype.str, "shape": list(array.shape), "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


class SharedBridgeCache:
    """One fixed pair's immutable train grids/XYZ/normals; no job-time eviction."""
    def __init__(self, retrieval, clouds):
        self.retrieval, self.cp, self.np = retrieval, retrieval.cp, retrieval.np if hasattr(retrieval, "np") else None
        self.items, self.normals, self.streams, self.leases = {}, {}, {}, []
        self.lease_streams = {}
        self.peak_active_streams = 0
        self.closed, self.failure = False, None
        self.owned_bytes = 0
        import numpy as np
        self.np = np
        # Conservative three-grid storage plus original-order XYZ and normals.
        forecast = sum(168*len(c.points)+12 for c in clouds)
        if forecast > CACHE_BYTES:
            raise scope.BridgeFailure("Fixed pair cache byte preflight rejects before target uploads")
        cp = self.cp
        begin = time.perf_counter()
        for cloud in clouds:
            points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
            if (points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 3
                    or not 1 <= len(points) <= 1_000_000 or normals.dtype != np.float64
                    or normals.shape != points.shape or not np.isfinite(points).all()
                    or not np.isfinite(normals).all()):
                raise scope.BridgeFailure("Every possible dynamic target needs finite original FP64 normals")
            self.items[id(cloud)] = {radius: retrieval._dataset(cloud, radius)
                for radius in (.12, .06, .03)}
            with cp.cuda.Device(0), cp.cuda.Stream.null:
                self.normals[id(cloud)] = cp.asarray(np.ascontiguousarray(normals))
        with cp.cuda.Device(0), cp.cuda.Stream.null:
            cp.cuda.Stream.null.synchronize()
        # Cache builder may evict under its cap. Reject rather than use any
        # stale/CPU-only/incomplete pointer as a complete device trajectory.
        for cloud in clouds:
            items = self.items[id(cloud)]
            for radius, item in items.items():
                shift = -math.frexp(radius)[1]
                if (retrieval.cache.get(id(cloud)) is not item or item.get("target") is not cloud
                        or item.get("cpu_only") or item.get("data") is None
                        or item["grids"].get(shift) is None):
                    raise scope.BridgeFailure("All three retained target grids must exist before proposal jobs")
        self.owned_bytes = int(retrieval.cache_bytes+sum(a.nbytes for a in self.normals.values()))
        if self.owned_bytes > forecast or self.owned_bytes > CACHE_BYTES:
            raise scope.BridgeFailure("Declared shared pair memory bound underestimated actual arrays")
        self.setup_wall_s = time.perf_counter()-begin
        self.cache_members = {key: value for key, value in retrieval.cache.items()}

    def register_lease_stream(self, lease, stream):
        if (self.closed or self.failure is not None or lease.get("owner") is not self
                or lease.get("closed") or stream.device_id != 0):
            raise scope.BridgeFailure("Closed/foreign/damaged shared cache lease")
        if int(stream.ptr) not in self.streams and len(self.streams) >= 4:
            raise scope.BridgeFailure("At most four owned lane streams can hold this fixed pair")
        self.streams[int(stream.ptr)] = stream
        self.lease_streams[id(lease)] = stream
        self.peak_active_streams = max(self.peak_active_streams, len(self.streams))

    def release_lease(self, lease):
        """Remove an active stream only after independently proving completion."""
        if lease.get("owner") is not self or not any(v is lease for v in self.leases):
            raise scope.BridgeFailure("Cannot release foreign proposal lease")
        stream = self.lease_streams.get(id(lease))
        if stream is not None:
            try:
                with self.cp.cuda.Device(0), stream: stream.synchronize()
            except BaseException as error:
                self.failure = self.failure or error
                raise
            del self.lease_streams[id(lease)]
            if not any(int(s.ptr) == int(stream.ptr) for s in self.lease_streams.values()):
                del self.streams[int(stream.ptr)]
        lease["closed"] = True
        self.leases = [v for v in self.leases if v is not lease]

    def lease(self, source, target):
        if self.closed or self.failure is not None:
            raise scope.BridgeFailure("Shared fixed pair cannot execute after failure/release")
        if (set(self.cache_members) != set(self.retrieval.cache)
                or any(self.retrieval.cache[k] is not v for k, v in self.cache_members.items())):
            raise scope.BridgeFailure("Shared target cache mutated during original proposal jobs")
        if id(source) not in self.items or id(target) not in self.items:
            raise scope.BridgeFailure("Dynamic original match uses a cloud outside the fixed pair inventory")
        value = {"owner": self, "target": target, "items": self.items[id(target)],
            "source_points": self.items[id(source)][.12]["data"], "normals": self.normals[id(target)],
            "device_id": 0, "owned_bytes": self.owned_bytes, "closed": False}
        self.leases.append(value)
        return value

    def close(self):
        if self.closed: return
        try:
            with self.cp.cuda.Device(0):
                for stream in self.streams.values(): stream.synchronize()
                self.cp.cuda.Stream.null.synchronize()
        except BaseException as error:
            self.failure = self.failure or error
            # Preserve every grid/numeric/stream owner while completion is unproved.
            raise
        self.retrieval.close()
        self.normals.clear()
        for lease in self.leases: lease["closed"] = True
        self.items.clear()
        self.closed = True

    def report(self):
        return {"closed": self.closed, "failure": None if self.failure is None else repr(self.failure),
            "owned_numeric_bytes": self.owned_bytes, "cache_cap_bytes": CACHE_BYTES,
            "active_streams": len(self.streams), "peak_active_streams": self.peak_active_streams,
            "maximum_streams": 4,
            "normal_grid_setup_wall_s": getattr(self, "setup_wall_s", None),
            "no_job_time_cache_mutation": True,
            "excluded_memory": "CPU KDTree/Python objects, CUDA allocator reservation, native workspace and opaque CUDA graph handles"}


def train_inventory(np, source, target):
    rows = {}
    for fragment in (source, target):
        prefix = str(fragment.index)
        def cloud(key, value):
            rows[key] = {k: scope.descriptor(np, getattr(value, k)) for k in ("points", "normals", "colors")}
        cloud(prefix+"/train", fragment.train)
        cloud(prefix+"/heldout", fragment.heldout)
        for i, view in enumerate(fragment.keys):
            key = prefix+f"/key{i}/raw{view.index}"
            cloud(key+"/train", view.train); cloud(key+"/heldout", view.heldout)
            rows[key+"/pose"] = scope.descriptor(np, view.pose)
            features = {}
            for name in ("pixels", "points", "descriptors"):
                value = getattr(view.features, name)
                if value is None and name != "descriptors":
                    raise scope.BridgeFailure("Only absent feature descriptors may use exact None")
                features[name] = None if value is None else scope.descriptor(np, value)
            rows[key+"/features"] = features
    return rows


class ProposalMatcher:
    def __init__(self, module, np, method, guard, row, *, shared=None, workspace=None,
                 permit=None, protocol=None, terminal=None):
        self.module, self.np, self.method, self.guard, self.row = module, np, method, guard, row
        self.shared, self.workspace = shared, workspace
        self.permit, self.protocol, self.terminal = permit, protocol, terminal
        self.loops = []
        self.failure = None

    def __call__(self, source, target, initial):
        if self.failure is not None: raise scope.BridgeFailure("Damaged original proposal cannot continue") from self.failure
        self.guard.check()
        before = scope.input_binding(self.np, source, target, initial)
        row = {"call_index": len(self.row["calls"]), "proposal_index": self.row["proposal_index"],
            "caller": sys._getframe(2).f_code.co_name, "input_binding": before, "complete": False,
            "query_trace": []}
        self.row["calls"].append(row)
        begin = time.perf_counter()
        loop, primary, lease = None, None, None
        try:
            if self.shared is None:
                result = self.guard.cpu_match(source, target, initial)
            else:
                if self.permit is not None:
                    self.protocol.validate_expected_bridge_call(self.permit, row["proposal_index"], row["call_index"], before)
                def observe(query):
                    packet = query["packet"]
                    order = self.np.argsort(packet[:, 0])
                    row["query_trace"].append({"query_index": query["query_index"], "stage": query["stage"],
                        "radius": query["radius"], "target_sha256": query["target_sha256"],
                        "packet": packet_descriptor(self.np, packet[order]),
                        "corrected_ids_sha256": hashlib.sha256(query["corrected_ids"][order].tobytes()).hexdigest(),
                        "counters": query["counters"]})
                loop = self.workspace.new_lane(audit_observer=observe if self.permit is None else None)
                self.loops.append(loop)
                lease = self.shared.lease(source, target)
                result = loop.match(source, target, initial, chunk_iterations=4,
                    pair_lease=lease)
                row["device_match_wall_s_inclusive"] = time.perf_counter()-begin
                row["source_binding"], row["consumed_input_binding"] = dict(loop.provenance), loop.input_binding
                self.workspace.check_lane(loop)
                row["terminal"] = self.terminal(self.np, result, loop.statistics, len(source.points), len(target.points))
                if self.permit is None:
                    native_begin = time.perf_counter()
                    native = self.guard.cpu_match(source, target, initial)
                    row["original_cpu_shadow_wall_s"] = time.perf_counter()-native_begin
                    row["native_shadow"] = scope.result_shadow(self.np, native, result,
                        len(source.points), len(target.points))
                    if not row["native_shadow"]["passed"]:
                        raise scope.BridgeFailure("Actual original CPU ICP result shadow failed")
                else:
                    self.protocol.validate_bridge_terminal(self.permit, row["proposal_index"], row["call_index"], before, row["terminal"])
                    row["native_shadow"] = {"collected": False, "scope": "Original native same-input result shadows belong to the closed fresh complete-bridge audit"}
            row["result"] = scope.result_evidence(self.np, result, len(source.points), len(target.points))
            row["input_bytes_unchanged"] = before == scope.input_binding(self.np, source, target, initial)
            if not row["input_bytes_unchanged"]: raise scope.BridgeFailure("Actual original call inputs/seed changed")
            row["query_trace_sha256"] = hashlib.sha256(scope.canonical(row["query_trace"]).encode()).hexdigest()
            row["complete"] = True
            return result  # Actual candidate object reaches every original gate.
        except BaseException as error:
            self.failure = primary = error
            row["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            if loop is not None:
                try:
                    self.workspace.close_lane(loop, primary=primary)
                    if lease is not None: self.shared.release_lease(lease)
                except BaseException as cleanup:
                    self.failure = self.failure or cleanup
                    row["cleanup_failure"] = repr(cleanup)
                    if primary is not None:
                        note = getattr(primary, "add_note", None)
                        if callable(note): note("Proposal lane cleanup: "+repr(cleanup))
                        raise primary from cleanup
                    raise
                finally: row["loop_report"] = loop.report()
            row["whole_call_wall_s_inclusive"] = time.perf_counter()-begin


def run(args):
    producer_begin = time.perf_counter()
    report = {"kind": KIND if args.mode == "audit" else TIMING_KIND, "policy": scope.POLICY, "status": "running", "mode": args.mode,
        "failure": None, "cleanup_failures": [], "native_proposals": [], "gpu_proposals": [],
        "whole_finish_authority": False, "timing_authority": False,
        "performance_attribution_valid": False, "schedule": "serial-complete-proposals",
        "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "One fixed original-ranked fragment pair; all genuine deduplicated original proposals, original verifier and ordered ambiguity. No adaptive frontier, optimizer, Live, fusion or mesh authority.",
        "timers": "Whole proposal wall includes private helper setup, original CPU gates, fresh private graph/buffers/capture, selected stream completion, input/result/permit validation and metadata. Audit adds exhaustive original CPU NN/native ICP shadows; timing omits only those observational shadows. gpu_cold_setup_and_proposals_wall_s charges workspace compiled setup and immutable target cache once, but is a setup+proposals subtotal; final owner cleanup and producer closure are separately recorded. The proposal-loop clocks also include interproposal boundary checks and progress publication. Gate/call/graph/setup timers are nested; never sum/subtract them for a speed claim. Prepared-pair component only."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save()
    primary, shared, retrieval, guard, workspace = None, None, None, None, None
    old_env = {k: os.environ.get(k) for k in ENVIRONMENT}
    threads = None
    try:
        for k, v in ENVIRONMENT.items(): os.environ[k] = v
        from scripts.profile_session import source_hash
        from scripts.research import device_loop_icp as method
        from scripts.research import device_loop_workspace as workspace_module
        from scripts.research import microbatch_bridge_protocol as protocol
        from scripts.research.gpu_icp_experiment_capture import fixture_records, sha as file_sha
        from scripts.research.benchmark_parallel_fragments import unpack_fragment, verification_dependencies
        from scripts.research.gpu_icp_device_loop_experiment import LoadedCodeGuard, terminal
        capture = json.loads(args.capture.read_text(encoding="utf-8"))
        if (capture.get("kind") != "gpu-icp-current-field-pair-fixture-v2" or capture.get("status") != "passed"
                or capture.get("cleanup_passed") is not True or capture.get("pose_seeds_used_for_live_tracking") is not False
                or capture.get("source_sha256") != CURRENT or capture.get("source_sha256_after") != CURRENT
                or capture.get("artifacts_sha256") != capture.get("artifacts_sha256_after")):
            raise scope.BridgeFailure("Require closed current fresh raw-Live-derived capture")
        fixed = {str(args.capture): file_sha(args.capture)}
        for name in ("fixture", "runtime_snapshot", "session", "profile", "native_extension"):
            value = capture[name]
            path = str(Path(value["path"]).resolve())
            if file_sha(path) != value["sha256"]: raise scope.BridgeFailure("Captured fixed resource changed")
            fixed[path] = value["sha256"]
        fixture_path = Path(capture["fixture"]["path"]).resolve()
        if not fixture_path.is_relative_to(ROOT/"benchmark-output"):
            raise scope.BridgeFailure("Only locally owned bound fixture pickle can be materialized")
        for name, digest in capture["artifacts_sha256"].items():
            if file_sha(ROOT/name) != digest: raise scope.BridgeFailure("Captured preparation source changed")
        import numpy as np
        import cupy as cp
        import open3d as o3d
        import cv2
        from scanner_server import fragments as module
        from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
        threads = (cv2, o3d, cv2.getNumThreads(), o3d.utility.get_max_threads())
        cv2.setNumThreads(20); o3d.utility.set_max_threads(20)
        guard, loop_guard = scope.OriginalVerifierGuard(module), LoadedCodeGuard(method)
        binary_modules = (sys.modules[o3d.geometry.PointCloud.__module__.split(".geometry")[0]],
            importlib.import_module("numpy._core._multiarray_umath"), importlib.import_module("cupy._core.core"))
        modules = {m.__name__: (m, str(Path(m.__file__).resolve())) for m in
            (np, cp, o3d, cv2, module, method, sys.modules[DeviceFlatGridICP.__module__])+binary_modules}
        paths = {"open3d": Path(binary_modules[0].__file__).resolve(),
            "numpy": Path(binary_modules[1].__file__).resolve(),
            "cupy": Path(binary_modules[2].__file__).resolve(),
            "native": Path(capture["native_extension"]["path"]).resolve()}
        binaries = {k: {"path": str(v), "sha256": file_sha(v)} for k, v in paths.items()}
        fixed.update({v["path"]: v["sha256"] for v in binaries.values()})
        with cp.cuda.Device(0): properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        hardware = {"name": name.decode() if isinstance(name, bytes) else str(name),
            "compute_capability": [int(properties["major"]), int(properties["minor"])],
            "total_global_mem": int(properties["totalGlobalMem"])}
        artifact_names = set(OWN_FILES+tuple(capture["artifacts_sha256"])+tuple(method.source_contract()["artifacts"])
            +tuple(workspace_module.source_contract()["artifacts"])+(
            "scripts/research/cuda_device_flat_grid_registration.py", "scripts/research/validate_device_flat_grid_proof.py",
            "scripts/research/archive/cuda_uniform_grid_registration.py", "scripts/research/archive/cuda_flat_grid_registration.py",
            "scripts/research/archive/validate_flat_grid_proof.py", "scripts/research/archive/validate_uniform_grid_proof.py",
            "scripts/research/gpu_icp_device_loop_experiment.py", "scripts/research/gpu_icp_device_loop_protocol.py",
            "tests/test_microbatch_bridge_protocol.py"))
        def runtime():
            return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve()),
                "numpy": np.__version__, "cupy": cp.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__,
                "driver": cp.cuda.runtime.driverGetVersion(), "cuda": cp.cuda.runtime.runtimeGetVersion(),
                "nvrtc": list(cp.cuda.nvrtc.getVersion()), "hardware": hardware, "binaries": binaries,
                "numpy_build_sha256": hashlib.sha256(scope.canonical(getattr(np.__config__, "CONFIG", {})).encode()).hexdigest(),
                "numpy_cpu_features": {k: bool(v) for k, v in getattr(importlib.import_module("numpy._core._multiarray_umath"), "__cpu_features__", {}).items()},
                "thread_policy": {"open3d": o3d.utility.get_max_threads(), "opencv": cv2.getNumThreads(), "omp": os.environ.get("OMP_NUM_THREADS")},
                "environment": {k: os.environ.get(k) for k in ENVIRONMENT}, "device": "CUDA:0"}
        def binding():
            return {"source_sha256": source_hash(), "artifacts_sha256": {p: file_sha(ROOT/p) for p in artifact_names},
                "fixed_files": {p: file_sha(p) for p in fixed}, "runtime": runtime(),
                "method_source": method.source_contract(), "verification_dependencies": verification_dependencies(),
                "workspace_source": workspace_module.source_contract(),
                "configuration": {"cache_bytes": CACHE_BYTES, "max_clouds": MAX_CLOUDS, "max_total_bytes": TOTAL_BYTES,
                    "graph": True, "chunk_iterations": 4}}
        report["binding"] = binding()
        if report["binding"]["source_sha256"] != CURRENT: raise scope.BridgeFailure("Current core differs from captured source")
        def boundary():
            guard.check(source=True); loop_guard.check()
            if runtime() != report["binding"]["runtime"]: raise scope.BridgeFailure("Actual runtime/thread/configuration changed")
            for k, (m, path) in modules.items():
                if sys.modules.get(k) is not m or str(Path(m.__file__).resolve()) != path:
                    raise scope.BridgeFailure("Actual loaded numerical/helper owner changed")
        with fixture_path.open("rb") as stream: fixture = pickle.load(stream)
        if fixture_records(fixture) != capture["tasks"]: raise scope.BridgeFailure("Materialized fixture differs from capture")
        task = fixture["tasks"][args.task]
        if not 2 <= len(task["proposals"]) <= 6: raise scope.BridgeFailure("Require all genuine competing proposals; never pad/subset")
        a, b = task["pair"]
        source, target = (unpack_fragment(fixture["fragments"][i]) for i in (a, b))
        # Exactly reinstall available fixture matches with this process's feature
        # identities. Each subsequent proposal gets a copy of these logical caches.
        for key, fragment in ((a, source), (b, target)):
            for original, view in zip(fixture["fragments"][key]["keys"], fragment.keys):
                for other in source.keys+target.keys:
                    matches = original["matches"].get(other.index)
                    if matches is not None: module._cache_matches(view, other, matches.copy())
        before = train_inventory(np, source, target)
        seeds = [np.ascontiguousarray(s, dtype=np.float64) for s in task["proposals"]]
        original_seeds = [scope.descriptor(np, s) for s in task["proposals"]]
        if original_seeds != [scope.descriptor(np, s) for s in seeds]: raise scope.BridgeFailure("Seed contiguous copy changed original values")
        report["pair_binding"] = {"task": args.task, "pair": [a, b], "inventory": before,
            "seeds": original_seeds, "camera": json.loads(scope.canonical(asdict(fixture["camera"]))),
            "pose_source": fixture["metadata"]["local_pose_source"]}
        if report["pair_binding"]["pose_source"] != "measured Finish fragment report; no archived ZIP poses":
            raise scope.BridgeFailure("Only freshly measured original local poses can reproduce these component inputs")
        permit = None
        if args.mode == "timing":
            permit = protocol.validate_bridge_audit(args.proof, report["binding"], report["pair_binding"])
            report["audit_proof"] = {"path": str(Path(args.proof).resolve()), "sha256": file_sha(args.proof)}
        native_values, gpu_values = [], []
        for mode, collection, values in (("native", report["native_proposals"], native_values),
                ("gpu", report["gpu_proposals"], gpu_values)):
            if mode == "gpu":
                boundary(); begin = time.perf_counter()
                retrieval = DeviceFlatGridICP("CUDA:0", max_clouds=MAX_CLOUDS, max_cache_bytes=CACHE_BYTES,
                    audit_nearest=True, audit_misses=True, miss_policy="direct-miss-research-v1")
                # Retain the partially constructed owner even when setup fails,
                # so asynchronous uploads cannot escape final synchronization.
                shared = SharedBridgeCache.__new__(SharedBridgeCache)
                shared.__init__(retrieval, all_train_clouds(source, target))
                # Retain a partially constructed compiled owner on setup fault.
                workspace = workspace_module.DeviceLoopWorkspace.__new__(workspace_module.DeviceLoopWorkspace)
                workspace.__init__(retrieval, max_total_bytes=TOTAL_BYTES, audit=args.mode == "audit", timing_permit=permit)
                report["shared_constructor_prepare_wall_s"] = time.perf_counter()-begin
            start = time.perf_counter()
            for index, seed in enumerate(seeds):
                boundary()
                row = {"proposal_index": index, "calls": [], "gates": [], "complete": False}
                collection.append(row)
                begun = time.perf_counter()
                private_source, private_target = scope.private_fragments(source, target)
                matcher = ProposalMatcher(module, np, method, guard, row, shared=shared if mode == "gpu" else None,
                    workspace=workspace if mode == "gpu" else None, permit=permit if mode == "gpu" else None,
                    protocol=protocol, terminal=terminal)
                verifier = scope.PrivateBridgeScope(module, matcher, trace=row["gates"], guard=guard)
                value = verifier.verify(private_source, private_target, seed, fixture["camera"])
                verifier.healthy()
                if matcher.failure is not None or any(not c["complete"] for c in row["calls"]):
                    raise scope.BridgeFailure("Dynamic complete proposal call suffix failed")
                row["result"] = scope.semantic(np, value)
                row["whole_proposal_wall_s"] = time.perf_counter()-begun
                row["complete"] = True
                values.append(value)
                boundary(); save()
                print(f"{mode} complete pair{a}/{b} proposal{index}: {len(row['calls'])} calls, {row['whole_proposal_wall_s']:.3f}s", flush=True)
            report[mode+"_whole_proposals_wall_s"] = time.perf_counter()-start
            report[mode+"_pair_verdict"] = scope.original_pair_verdict(module, values)
        report["gpu_cold_setup_and_proposals_wall_s"] = report["shared_constructor_prepare_wall_s"]+report["gpu_whole_proposals_wall_s"]
        comparisons = []
        for old, new in zip(report["native_proposals"], report["gpu_proposals"]):
            gates = lambda row: [{k: v for k, v in r.items() if k not in ("wall_s_inclusive",)} for r in row["gates"]]
            differences = scope.compare_evidence(old["result"], new["result"], path="result")
            left, right = gates(old), gates(new)
            if len(left) != len(right):
                differences.append({"path": "gates", "reason": "ordered complete gate count"})
            else:
                for i, (x, y) in enumerate(zip(left, right)):
                    scope.compare_evidence(x, y, tolerance=1e-5 if x["name"] ==
                        "REG.get_information_matrix_from_point_clouds" else 1e-8,
                        path=f"gates[{i}]", differences=differences)
            comparisons.append({"proposal_index": old["proposal_index"], "native_calls": len(old["calls"]),
                "gpu_calls": len(new["calls"]), "passed": not differences, "differences": differences[:100],
                "difference_count": len(differences)})
        report["proposal_quality"] = comparisons
        report["pair_quality_passed"] = (all(c["passed"] for c in comparisons)
            and report["native_pair_verdict"] == report["gpu_pair_verdict"])
        if permit is not None: protocol.validate_complete_bridge(permit, report)
        report["input_bytes_unchanged"] = before == train_inventory(np, source, target)
        report["original_seed_bytes_unchanged"] = original_seeds == [scope.descriptor(np, s) for s in task["proposals"]]
        if not report["pair_quality_passed"] or not report["input_bytes_unchanged"] or not report["original_seed_bytes_unchanged"]:
            raise scope.BridgeFailure("Complete original proposal gates/input/verdict audit failed; diagnostics retained")
        boundary(); report["loaded_owners_unchanged"] = True
    except BaseException as error:
        primary = error
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        cleanup_begin = time.perf_counter()
        failures = []
        def clean(label, action):
            try: return action()
            except BaseException as error:
                failures.append({"action": label, "type": type(error).__name__, "message": str(error)})
                return None
        if workspace is not None:
            clean("workspace stream completion", lambda: workspace.close(primary=primary))
            report["workspace"] = clean("workspace diagnostic closure", workspace.report)
        if shared is not None and (workspace is None or getattr(workspace, "closed", False)):
            clean("shared stream/cache completion", shared.close)
            report["shared_cache"] = shared.report()
        elif shared is not None:
            report["shared_cache"] = shared.report()
            failures.append({"action": "shared cache retained", "message": "Workspace completion unproved; cache owners retained"})
        elif retrieval is not None: clean("unused cache release", retrieval.close)
        report["owner_cleanup_wall_s"] = time.perf_counter()-cleanup_begin
        closure_begin = time.perf_counter()
        if guard is not None: clean("original verifier closure", lambda: guard.check(source=True))
        if "binding" in report: clean("all loaded loop/numerical/helper owners closure", boundary)
        if "binding" in report: report["binding_after"] = clean("complete source/runtime/fixed resource closure", binding)
        if threads is not None:
            cv2, o3d, cvt, o3t = threads
            clean("OpenCV restore", lambda: cv2.setNumThreads(cvt))
            clean("Open3D restore", lambda: o3d.utility.set_max_threads(o3t))
            if clean("restored thread values", lambda: (cv2.getNumThreads(), o3d.utility.get_max_threads())) != (cvt, o3t):
                failures.append({"action": "thread restoration", "message": "Actual values differ"})
        for k, v in old_env.items():
            clean("environment restore", lambda k=k, v=v: os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v))
        report["cleanup_failures"] = failures
        report["cleanup_passed"] = not failures and all(os.environ.get(k) == v for k, v in old_env.items())
        report["status"] = "passed" if (primary is None and report["cleanup_passed"]
            and report.get("binding_after") == report.get("binding")
            and report.get("pair_quality_passed") is True and report.get("input_bytes_unchanged") is True
            and report.get("original_seed_bytes_unchanged") is True and report.get("loaded_owners_unchanged") is True
            and report.get("shared_cache", {}).get("closed") is True
            and report.get("workspace", {}).get("closed") is True
            and report.get("workspace", {}).get("failure") is None) else "failed"
        report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if report["status"] == "passed" and args.mode == "audit":
            try: protocol.validate_report(report, report["binding"], report["pair_binding"])
            except BaseException as error:
                primary = error; report["status"] = "failed"
                report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if report["status"] == "passed" and args.mode == "timing": report["performance_attribution_valid"] = True
        report["producer_closure_wall_s"] = time.perf_counter()-closure_begin
        report["producer_wall_s_before_final_publication"] = time.perf_counter()-producer_begin
        try: save()
        except BaseException as write_error:
            if primary is not None:
                note = getattr(primary, "add_note", None)
                if callable(note): note("Final bridge diagnostic write failed: "+repr(write_error))
                raise primary from write_error
            raise
    if report["status"] != "passed": raise RuntimeError("Complete bridge proposal audit failed; private output preserved") from primary
    from scripts.process_metrics import finish_cuda_worker
    finish_cuda_worker()


if __name__ == "__main__":
    run(parse())
