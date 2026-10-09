"""Fresh current-input device-loop audit, then separately authorized timing.

No scanner hooks or old permits. Genuine captured proposals retain their exact
arrays and unrounded seeds. Audits shadow every actual NN row and the terminal
original native ICP result. Timing compares exact own-audit terminal bytes.
This is a prepared-pair component experiment, without whole-Finish authority.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time
import traceback
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import gpu_icp_device_loop_protocol as protocol

OWN_FILES = ("scripts/research/gpu_icp_device_loop_experiment.py",
    "scripts/research/gpu_icp_device_loop_protocol.py", "tests/test_gpu_icp_device_loop_protocol.py")
DEPENDENCIES = ("scripts/research/cuda_device_flat_grid_registration.py",
    "scripts/research/validate_device_flat_grid_proof.py",
    "scripts/research/archive/cuda_flat_grid_registration.py",
    "scripts/research/archive/cuda_uniform_grid_registration.py",
    "scripts/research/archive/validate_flat_grid_proof.py",
    "scripts/research/archive/validate_uniform_grid_proof.py",
    "scripts/research/archive/research_uniform_grid_nn.cu",
    "scripts/research/benchmark_parallel_fragments.py", "scripts/profile_session.py",
    "scripts/process_metrics.py", "scripts/tool_paths.py", "scripts/tool-catalog.json")
ENVIRONMENT = {"OMP_NUM_THREADS": "8", "KINECT_NATIVE": "on", "KINECT_CUDA_REGISTRATION": "cpu"}


def descriptor(np, value):
    array = np.asarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def code_identity(code):
    """Compare loaded code to its source without executing another module."""
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags,
        tuple(code_identity(v) if isinstance(v, CodeType) else v for v in code.co_consts))


class LoadedCodeGuard:
    def __init__(self, module):
        self.module, self.path = module, Path(module.__file__).resolve()
        compiled = compile(self.path.read_text(encoding="utf-8"), str(self.path), "exec")
        expected = {c.co_name: c for c in compiled.co_consts if isinstance(c, CodeType)}
        self.functions = []
        for name, code in expected.items():
            value = getattr(module, name)
            if isinstance(value, FunctionType):
                self.add(module, name, value, code)
            elif isinstance(value, type):
                for method in code.co_consts:
                    if isinstance(method, CodeType):
                        self.add(value, method.co_name, getattr(value, method.co_name), method)
        self.objects = {name: getattr(module, name) for name in ("DeviceLoopICP", "STAGES", "POLICY", "MAX_POINTS", "CONTROL", "RESIDENT", "FLAT", "CLASSIFIER", "LDLT")}

    def add(self, owner, name, function, expected):
        if not isinstance(function, FunctionType) or code_identity(function.__code__) != code_identity(expected):
            raise ValueError("Loaded device-loop method differs from bound source: "+name)
        self.functions.append((owner, name, function, function.__code__,
            repr(function.__defaults__), repr(function.__kwdefaults__)))

    def check(self):
        if sys.modules.get(self.module.__name__) is not self.module or Path(self.module.__file__).resolve() != self.path:
            raise ValueError("Loaded device-loop module ownership changed")
        for name, value in self.objects.items():
            if getattr(self.module, name) is not value:
                raise ValueError("Loaded device-loop control/source ownership changed")
        for owner, name, function, code, defaults, kwdefaults in self.functions:
            if (getattr(owner, name) is not function or function.__code__ is not code
                    or repr(function.__defaults__) != defaults or repr(function.__kwdefaults__) != kwdefaults):
                raise ValueError("Loaded device-loop function/code/default ownership changed")
        return True


def terminal(np, result, stats, n, m):
    pose, pairs = np.asarray(result.transformation), np.asarray(result.correspondence_set)
    if (pose.dtype != np.float64 or pose.shape != (4,4) or not np.isfinite(pose).all()
            or pairs.dtype != np.int32 or pairs.ndim != 2 or pairs.shape[1] != 2
            or len(np.unique(pairs[:,0])) != len(pairs)
            or (pairs[:,0] < 0).any() or (pairs[:,0] >= n).any()
            or (pairs[:,1] < 0).any() or (pairs[:,1] >= m).any()):
        raise ValueError("Malformed original terminal pose/correspondence mapping")
    value = {"pose": descriptor(np, pose), "correspondences": descriptor(np, pairs),
        "fitness": float(result.fitness), "inlier_rmse": float(result.inlier_rmse),
        "queries": stats["queries"], "updates": stats["updates"]}
    protocol.terminal_contract(value, n, m)
    return value


def native_shadow(np, native, candidate, n, m):
    def mapping(result):
        pairs = np.asarray(result.correspondence_set)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu"
                or len(np.unique(pairs[:,0])) != len(pairs) or (pairs[:,0]<0).any()
                or (pairs[:,0]>=n).any() or (pairs[:,1]<0).any() or (pairs[:,1]>=m).any()):
            raise ValueError("Malformed native/candidate source-to-target mapping")
        return pairs[np.lexsort((pairs[:,1], pairs[:,0]))].astype(np.int32)
    poses = (np.asarray(native.transformation), np.asarray(candidate.transformation))
    metrics = [native.fitness, candidate.fitness, native.inlier_rmse, candidate.inlier_rmse]
    if any(p.shape != (4,4) or not np.isfinite(p).all() for p in poses) or not np.isfinite(metrics).all():
        raise ValueError("Nonfinite original native shadow cannot pass")
    a, b = mapping(native), mapping(candidate)
    value = {"transform_max_abs_delta": float(np.max(np.abs(poses[0]-poses[1]))),
        "fitness_abs_delta": abs(float(native.fitness)-float(candidate.fitness)),
        "rmse_abs_delta": abs(float(native.inlier_rmse)-float(candidate.inlier_rmse)),
        "correspondence_ids_equal": bool(np.array_equal(a,b)),
        "native_mapping": descriptor(np,a), "candidate_mapping": descriptor(np,b)}
    value["passed"] = value["correspondence_ids_equal"] and all(value[k] <= 1e-8 for k in
        ("transform_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta"))
    return value


def close_lane(loop, retrieval, *, primary=None):
    """Retain cache pointers if stream completion is unproved; never retry math."""
    failures = []
    completed = loop is None
    if loop is not None:
        try:
            loop.close(); completed = True
        except BaseException as error:
            failures.append(error)
    if retrieval is not None and completed:
        try:
            retrieval.close()
        except BaseException as error:
            failures.append(error)
    if failures:
        if primary is not None:
            for error in failures:
                note = getattr(primary, "add_note", None)
                if callable(note): note("Device-loop experiment cleanup: "+repr(error))
            raise primary from failures[0]
        raise failures[0]


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("audit", "timing"), default="audit")
    p.add_argument("--proof", type=Path)
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--seeds", type=int, choices=(2,4), default=4)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--configurations", nargs="+", choices=("step1","graph1","graph2","graph4"), default=["step1","graph4"])
    p.add_argument("--run-allocated", action="store_true")
    args = p.parse_args()
    args.capture, args.output = args.capture.resolve(), args.output.resolve()
    if (not args.run_allocated or not 1 <= args.pairs <= 8 or not 1 <= args.repeats <= 5
            or len(args.configurations) != len(set(args.configurations))
            or args.mode == "audit" and "step1" not in args.configurations
            or args.mode == "timing" and args.proof is None
            or not args.output.is_relative_to(ROOT/"benchmark-output") or args.output.exists()):
        p.error("Require exclusive allocation, bounded genuine pairs/configurations and a fresh private output; timing needs own audit")
    with socket.socket() as check:
        check.settimeout(.3)
        if check.connect_ex(("127.0.0.1",8000)) == 0: p.error("Stop idle field server before experiment")
    return args


def run(args):
    run_begin = time.perf_counter()
    report = {"kind": protocol.KIND if args.mode == "audit" else protocol.TIMING_KIND,
        "mode": args.mode, "status": "running", "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "failure": None, "cleanup_failures": [], "rows": [], "native_controls": [],
        "performance_attribution_valid": False, "whole_finish_authority": False,
        "configurations": args.configurations, "scope": "Exact prepared-pair genuine proposal prefixes; original native terminal ICP shadows. No complete bridge/witness/frontier/Finish/mesh authority.",
        "timer_scope": "Sequential host wall. Constructor compilation, start/cache/build/upload plus selected stream completion, bounded advance/control copies, terminal download, result shadow/permit validation and cleanup are disjoint. Graph capture is nested within advance. Cold top-level file/binary rehash and row-boundary owner/thread checks are outside all_in. Audits include all original CPU query shadows. Every trajectory owns a fresh graph; no stale pointer graph reuse. Later repeats warm process compiler/allocator caches only; target indexes are rebuilt in every trajectory. Native controls precede device variants; device order reverses on odd timing repeats."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save()
    failure, guard = None, None
    fixed, modules, loop_reports = {}, {}, []
    old_env = {k: os.environ.get(k) for k in ENVIRONMENT}
    old_threads = None
    try:
        for key, value in ENVIRONMENT.items(): os.environ[key] = value
        from scripts.profile_session import source_hash
        from scripts.research import device_loop_icp as method
        from scripts.research.archive.research_resident_icp import ResidentICP
        from scripts.research.benchmark_parallel_fragments import unpack_cloud
        from scripts.research.gpu_icp_experiment_capture import fixture_records
        capture = json.loads(args.capture.read_text(encoding="utf-8"))
        if (capture.get("kind") != "gpu-icp-current-field-pair-fixture-v2" or capture.get("status") != "passed"
                or capture.get("cleanup_passed") is not True or capture.get("pose_seeds_used_for_live_tracking") is not False
                or capture.get("source_sha256") != protocol.CURRENT or capture.get("source_sha256_after") != protocol.CURRENT
                or capture.get("artifacts_sha256") != capture.get("artifacts_sha256_after")):
            raise ValueError("Require closed current raw-derived field capture; no archived-seeded Live")
        for name in ("fixture","runtime_snapshot","session","profile","native_extension"):
            record = capture[name]
            if protocol.sha(record["path"]) != record["sha256"]: raise ValueError("Captured raw/fixture/profile/native changed")
            fixed[str(Path(record["path"]).resolve())] = record["sha256"]
        fixed[str(args.capture)] = protocol.sha(args.capture)
        fixture_path = Path(capture["fixture"]["path"]).resolve()
        if not fixture_path.is_relative_to(ROOT/"benchmark-output"):
            raise ValueError("Only own bound private fixture is trusted for local pickle loading")
        for path, digest in capture["artifacts_sha256"].items():
            if protocol.sha(ROOT/path) != digest: raise ValueError("Captured original preparation source changed")
        import numpy as np
        import cupy as cp
        import open3d as o3d
        import cv2
        from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
        old_threads = (cv2, o3d, cv2.getNumThreads(), o3d.utility.get_max_threads())
        cv2.setNumThreads(20); o3d.utility.set_max_threads(20)
        guard = LoadedCodeGuard(method)
        modules = {m.__name__: (m, str(Path(m.__file__).resolve())) for m in
            (method, np, cp, o3d, cv2, sys.modules[ResidentICP.__module__], sys.modules[DeviceFlatGridICP.__module__])}
        paths = {"open3d": Path(sys.modules[o3d.geometry.PointCloud.__module__.split(".geometry")[0]].__file__).resolve(),
            "numpy": Path(importlib.import_module("numpy._core._multiarray_umath").__file__).resolve(),
            "cupy": Path(importlib.import_module("cupy._core.core").__file__).resolve(),
            "native": Path(capture["native_extension"]["path"]).resolve()}
        numpy_core = importlib.import_module("numpy._core._multiarray_umath")
        for module in (numpy_core,importlib.import_module("cupy._core.core"),
                sys.modules[o3d.geometry.PointCloud.__module__.split(".geometry")[0]]):
            modules[module.__name__] = (module,str(Path(module.__file__).resolve()))
        binaries = {key: {"path": str(path), "sha256": protocol.sha(path)} for key,path in paths.items()}
        fixed.update({r["path"]:r["sha256"] for r in binaries.values()})
        fixed[str(Path(sys.executable).resolve())] = protocol.sha(sys.executable)
        with cp.cuda.Device(0): properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        hardware = {"name":name.decode() if isinstance(name,bytes) else str(name),
            "compute_capability":[int(properties["major"]),int(properties["minor"])],
            "total_global_mem":int(properties["totalGlobalMem"])}
        artifacts = {str(Path(name)).replace("\\","/"): protocol.sha(ROOT/name) for name in
            set(OWN_FILES+DEPENDENCIES+tuple(capture["artifacts_sha256"])+tuple(method.source_contract()["artifacts"]))}
        def runtime():
            numpy_configuration = protocol.canonical(getattr(np.__config__,"CONFIG",{}))
            return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve()),
                "numpy":np.__version__, "cupy":cp.__version__, "open3d":o3d.__version__, "opencv":cv2.__version__,
                "driver":cp.cuda.runtime.driverGetVersion(), "cuda":cp.cuda.runtime.runtimeGetVersion(),
                "device":"CUDA:0", "thread_policy":{"open3d":o3d.utility.get_max_threads(),"opencv":cv2.getNumThreads(),"omp":os.environ.get("OMP_NUM_THREADS")},
                "environment": {key:os.environ.get(key) for key in ENVIRONMENT}, "binaries":binaries,"hardware":hardware,
                "nvrtc_version":list(cp.cuda.nvrtc.getVersion()),
                "numpy_build_configuration_sha256":hashlib.sha256(numpy_configuration.encode()).hexdigest(),
                "numpy_cpu_features":{key:bool(value) for key,value in getattr(numpy_core,"__cpu_features__",{}).items()},
                "native_binary_scope":"Captured original preparation extension bytes; this prepared-pair evaluator does not call it"}
        def binding():
            return {"source_sha256":source_hash(),"method_source":method.source_contract(),
                "artifacts_sha256": {p:protocol.sha(ROOT/p) for p in artifacts}, "runtime":runtime(),
                "fixed_files": {p:protocol.sha(p) for p in fixed}}
        report["binding"] = binding()
        protocol.actual_resource_closure(report["binding"])
        def boundary():
            guard.check()
            if runtime() != report["binding"]["runtime"]: raise ValueError("Actual loaded runtime/thread/environment changed")
            for name,(module,path) in modules.items():
                if sys.modules.get(name) is not module or str(Path(module.__file__).resolve()) != path:
                    raise ValueError("Numerical/helper module owner changed")
        with fixture_path.open("rb") as stream: fixture = pickle.load(stream)
        if fixture_records(fixture) != capture["tasks"]:
            raise ValueError("Actual fixture point/normal/proposal descriptors differ from closed capture")
        tasks = [t for t in fixture["tasks"] if len(t["proposals"]) >= args.seeds][:args.pairs]
        if len(tasks) != args.pairs: raise ValueError("Insufficient genuine seed groups; never pad or substitute")
        permit = None
        if args.mode == "timing":
            permit = protocol.validate_device_loop_audit(args.proof, report["binding"])
            report["audit_proof"] = {"path":str(Path(args.proof).resolve()),"sha256":protocol.sha(args.proof)}
        report["process_preparation_wall_s"] = time.perf_counter()-run_begin
        report["process_preparation_scope"] = "Shared lazy numerical imports/CUDA context, captured input parsing, cold source/binary/raw checks and fresh permit validation. Separate from per-trajectory all_in; charged in total run wall. No resident CPU-solve comparison is claimed."
        refs = {}
        original_arrays = []
        for task in tasks:
            a,b = task["pair"]
            source = unpack_cloud(fixture["fragments"][a]["train"])
            target = unpack_cloud(fixture["fragments"][b]["train"])
            for index, initial in enumerate(task["proposals"][:args.seeds]):
                original_seed = np.asarray(initial)
                if original_seed.dtype != np.float64 or original_seed.shape != (4,4) or not np.isfinite(original_seed).all():
                    raise ValueError("Only finite original unrounded float64 proposal seeds are accepted")
                seed = np.ascontiguousarray(initial, dtype=np.float64)
                original = {"source":descriptor(np,source.points),"target":descriptor(np,target.points),
                    "normals":descriptor(np,target.normals),"seed":descriptor(np,seed)}
                original_arrays.append((source,target,seed,original))
                native = None
                for repeat in range(1 if args.mode == "audit" else args.repeats):
                    boundary()
                    begin = time.perf_counter()
                    native = ResidentICP._original_cpu_match(source,target,seed)
                    elapsed = time.perf_counter()-begin
                    report["native_controls"].append({"pair":[a,b],"seed_index":index,"repeat":repeat,
                        "wall_s":elapsed,"pose":descriptor(np,native.transformation),"fitness":float(native.fitness),
                        "inlier_rmse":float(native.inlier_rmse)})
                    order = list(args.configurations)
                    if args.mode == "timing" and repeat%2: order.reverse()
                    for label in order:
                        boundary()
                        graph,chunk = (False,1) if label == "step1" else (True,int(label[-1]))
                        row = {"pair":[a,b],"seed_index":index,"repeat":repeat,"configuration":label,
                            "variant_order":order,"query_trace":[]}
                        report["rows"].append(row)
                        retrieval = loop = None
                        primary = None
                        begin = time.perf_counter()
                        def observe(query):
                            packet = query["packet"]
                            order = np.argsort(packet[:,0])
                            row["query_trace"].append({"stage":query["stage"],"query_index":query["query_index"],
                                "radius":query["radius"],"target_sha256":query["target_sha256"],
                                "packet":descriptor(np,packet[order]),
                                "corrected_ids_sha256":hashlib.sha256(query["corrected_ids"][order].tobytes()).hexdigest(),
                                "counters":query["counters"]})
                        try:
                            # Cache builder only. Its observational flags grant no loop timing authority;
                            # nearest_device is never called by this complete-device evaluator.
                            retrieval = DeviceFlatGridICP("CUDA:0",max_clouds=1,audit_nearest=True,audit_misses=True,
                                miss_policy="direct-miss-research-v1")
                            loop = method.DeviceLoopICP(retrieval,cuda_graph=graph,
                                audit_nearest=args.mode=="audit",audit_misses=args.mode=="audit",
                                audit_observer=observe if args.mode=="audit" else None,
                                timing_authorizer=None if permit is None else lambda inputs,code:
                                    protocol.validate_device_loop_start(permit,inputs,code,report["binding"]))
                            row["constructor_wall_s"] = time.perf_counter()-begin
                            begin = time.perf_counter()
                            loop.start(source,target,seed,chunk_iterations=chunk)
                            with cp.cuda.Device(0),loop.stream: loop.stream.synchronize()
                            row["start_cache_build_upload_sync_wall_s"] = time.perf_counter()-begin
                            row["input_binding"],row["source_binding"] = loop.input_binding,dict(loop.provenance)
                            begin = time.perf_counter()
                            for _ in range(94):
                                if loop.advance()["phase"] == "done": break
                            else: raise RuntimeError("Bounded complete loop did not finish")
                            with cp.cuda.Device(0),loop.stream: loop.stream.synchronize()
                            row["advance_sync_wall_s"] = time.perf_counter()-begin
                            begin = time.perf_counter()
                            result = loop.result()
                            with cp.cuda.Device(0),loop.stream: loop.stream.synchronize()
                            row["terminal_copy_sync_wall_s"] = time.perf_counter()-begin
                            begin = time.perf_counter()
                            row["terminal"] = terminal(np,result,loop.statistics,len(source.points),len(target.points))
                            row["native_shadow"] = native_shadow(np,native,result,len(source.points),len(target.points))
                            if not row["native_shadow"]["passed"]: raise RuntimeError("Fresh original native terminal shadow failed")
                            if permit is not None:
                                protocol.validate_timing_accounting(loop.statistics,len(source.points))
                                protocol.validate_device_loop_terminal(permit,loop.input_binding,loop.provenance,row["terminal"],report["binding"])
                            else:
                                key = (a,b,index)
                                evidence = (row["terminal"],row["query_trace"])
                                if key in refs and refs[key] != evidence: raise RuntimeError("Graph/chunk changed exact query trajectory or terminal bytes")
                                refs[key] = evidence
                            row["query_trace_sha256"] = hashlib.sha256(protocol.canonical(row["query_trace"]).encode()).hexdigest()
                            row["result_validation_wall_s"] = time.perf_counter()-begin
                        except BaseException as error:
                            primary = error
                            row["failure"] = {"type":type(error).__name__,"message":str(error)}
                            raise
                        finally:
                            begin = time.perf_counter()
                            try: close_lane(loop,retrieval,primary=primary)
                            finally:
                                row["cleanup_wall_s"] = time.perf_counter()-begin
                                if loop is not None:
                                    row["loop_report"] = loop.report(); loop_reports.append(row["loop_report"])
                        row["all_in_wall_s"] = sum(row[k] for k in ("constructor_wall_s","start_cache_build_upload_sync_wall_s",
                            "advance_sync_wall_s","terminal_copy_sync_wall_s","result_validation_wall_s","cleanup_wall_s"))
                        boundary(); save()
                        print(f"{args.mode} pair{a}/{b} seed{index} {label} repeat{repeat}: {row['all_in_wall_s']:.6f}s closed",flush=True)
        report["input_bytes_unchanged"] = all(original == {"source":descriptor(np,s.points),"target":descriptor(np,t.points),
            "normals":descriptor(np,t.normals),"seed":descriptor(np,i)} for s,t,i,original in original_arrays)
        boundary()
        report["loaded_owners_unchanged"] = True
        report["binding_after"] = binding()
        if report["binding_after"] != report["binding"] or not report["input_bytes_unchanged"]:
            raise RuntimeError("Current source/runtime/raw/input closure changed")
        if permit is not None: protocol.registered(permit)
    except BaseException as error:
        failure = error
        report["failure"] = {"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
    finally:
        close_begin = time.perf_counter()
        cleanup = []
        def clean(label, action):
            try: return action()
            except BaseException as error:
                cleanup.append({"action":label,"type":type(error).__name__,"message":str(error)})
                return None
        if guard is not None: clean("loaded code closure",guard.check)
        if "binding" in report: clean("all loaded runtime owners and threads",boundary)
        if "binding" in report:
            report["binding_after"] = clean("current source/runtime/resource closure",binding)
            clean("actual resource closure",lambda:protocol.actual_resource_closure(report["binding_after"]))
        if old_threads is not None:
            cv2,o3d,cvt,o3t = old_threads
            clean("OpenCV restoration",lambda:cv2.setNumThreads(cvt))
            clean("Open3D restoration",lambda:o3d.utility.set_max_threads(o3t))
            if clean("actual restored thread values",lambda:(cv2.getNumThreads(),o3d.utility.get_max_threads())) != (cvt,o3t):
                cleanup.append({"action":"thread restoration","message":"actual thread values differ"})
        for key,value in old_env.items():
            clean("environment restoration",lambda key=key,value=value:
                os.environ.pop(key,None) if value is None else os.environ.__setitem__(key,value))
        report["cleanup_failures"] = cleanup
        report["cleanup_passed"] = not cleanup and all(os.environ.get(k)==v for k,v in old_env.items())
        complete = (failure is None and report["cleanup_passed"] and report.get("binding_after")==report.get("binding")
            and report.get("input_bytes_unchanged") is True and report.get("loaded_owners_unchanged") is True
            and bool(loop_reports) and all(r["closed"] and r["failure"] is None for r in loop_reports))
        report["status"] = "passed" if complete else "failed"
        report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if complete and args.mode=="audit":
            try: protocol.validate_audit_report(report,report["binding"])
            except BaseException as error:
                failure=error;report["status"]="failed"
                report["failure"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        if report["status"]=="passed" and args.mode=="timing": report["performance_attribution_valid"]=True
        report["phase_closure_wall_s"] = time.perf_counter()-close_begin
        report["total_run_wall_s_before_final_save"] = time.perf_counter()-run_begin
        try: save()
        except BaseException as write_error:
            if failure is not None:
                note=getattr(failure,"add_note",None)
                if callable(note): note("Final diagnostic write also failed: "+repr(write_error))
                raise failure from write_error
            raise
    if report["status"]!="passed": raise RuntimeError("Device-loop experiment failed; own fresh diagnostics retained") from failure
    from scripts.process_metrics import finish_cuda_worker
    finish_cuda_worker()


if __name__=="__main__":
    run(parse())
