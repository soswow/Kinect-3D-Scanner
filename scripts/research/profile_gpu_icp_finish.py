"""Fresh current-source Live checkpoint and original whole-Finish GPU experiment.

Imports are stdlib-only until an explicit allocated, idle-server invocation.
Capture uses every raw view and original Live processing; no exported pose seeds.
Native/audit/timing start from the same exact logical Live state. Finish wall
includes owned permit/setup, scope entry, original build, gates and all cleanup.
Existing component and historical Finish receipts never authorize this runner.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from types import CodeType, FunctionType
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash as _profile_file_hash, source_hash

CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"
CAPTURE_KIND = "gpu-icp-whole-finish-fresh-live-capture-v1"
KINDS = {"native": "gpu-icp-whole-finish-native-v1", "audit": "gpu-icp-whole-finish-audit-v1",
         "timing": "gpu-icp-whole-finish-timing-v1", "capture": CAPTURE_KIND}
PIPELINE = {
    "OMP_NUM_THREADS": "8", "KINECT_NATIVE": "on", "KINECT_BLOCK_COUNT": "5000",
    "KINECT_CUDA_REGISTRATION": "cpu", "KINECT_CUDA_ODOMETRY": "off",
    "KINECT_CUDA_MATCHING": "cuda", "KINECT_CUDA_FUSION": "fused",
    "KINECT_MODEL_REFRESH": "lazy", "KINECT_KEYFRAME_CACHE": "on",
    "KINECT_LIVE_RECOVERY": "full", "KINECT_VISUAL_FEATURES": "adaptive",
    "KINECT_VISUAL_REFINEMENT": "icp", "KINECT_FINAL_VISUAL_FIRST": "off",
    "KINECT_FINAL_LOCAL_REFINEMENT": "icp", "KINECT_ADAPTIVE_EXPERIMENTAL": "off",
    "KINECT_CUDA_INPUT": "auto", "KINECT_CUDA_CONFIDENCE": "off", "PYTHONUTF8": "1"}
THREAD_ENV = ("OMP_WAIT_POLICY", "KMP_BLOCKTIME", "OPENCV_FOR_THREADS_NUM", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
CHECKPOINT_FILES = ("scripts/research/profile_gpu_icp_finish.py", "tests/test_gpu_icp_finish_profile.py",
    "scripts/research/private_live_checkpoint.py", "scripts/profile_session.py", "scripts/process_metrics.py",
    "scripts/research/archive/validate_uniform_grid_proof.py")
FAMILY_FILES = CHECKPOINT_FILES + (
    "scripts/research/gpu_icp_finish_scope.py", "tests/test_gpu_icp_finish_scope.py",
    "scripts/research/device_loop_finish_workspace.py", "tests/test_device_loop_finish_workspace.py",
    "scripts/research/gpu_icp_finish_protocol.py", "tests/test_gpu_icp_finish_protocol.py",
    "scripts/research/compare_gpu_icp_finishes.py", "tests/test_gpu_icp_finish_quality.py",
    "scripts/research/compare_resident_finishes.py",
    "scripts/research/cuda_device_flat_grid_registration.py", "scripts/research/research_device_flat_grid_nn.cu",
    "scripts/research/validate_device_flat_grid_proof.py", "scripts/research/archive/cuda_uniform_grid_registration.py",
    "scripts/research/archive/cuda_flat_grid_registration.py", "scripts/research/archive/validate_flat_grid_proof.py",
    "scripts/research/gpu_icp_device_loop_experiment.py", "scripts/research/gpu_icp_device_loop_protocol.py",
    "scripts/research/microbatch_bridge_scope.py", "scripts/research/microbatch_bridge_driver.py",
    "scripts/research/field_finish_conformance_scope.py", "scripts/research/finish_resident_registration.py",
    "scripts/research/validate_finish_resident_proof.py", "scripts/tool_paths.py", "scripts/tool-catalog.json")


class FinishProfileFailure(BaseException):
    """Controller faults cannot become an ordinary rejected engine stage."""


def file_hash(path):
    return _profile_file_hash(Path(path))


def require(value, message):
    if not value:
        raise FinishProfileFailure(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def frozen_json(value):
    return json.loads(canonical(value))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def error_record(error):
    return {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exception(error)}


def save_report(path, report, primary=None):
    """A late publication fault may not replace the actual numerical fault."""
    try:
        Path(path).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    except BaseException as write_error:
        if primary is not None:
            primary.add_note("Failed whole-Finish diagnostics write: "+repr(write_error))
            raise primary from write_error
        raise


def clean(label, action, failures, primary=None):
    try:
        return action()
    except BaseException as error:
        failures.append({"action": label, "type": type(error).__name__, "message": str(error)})
        if primary is not None:
            primary.add_note(label+": "+repr(error))
        return None


def preflight(args, *, root=ROOT, idle=None):
    require(args.run_allocated, "Require an exclusive hardware allocation")
    require(args.mode in KINDS, "Explicit capture/native/audit/timing mode required")
    require(args.final_block_count is None or type(args.final_block_count) is int
            and 1 <= args.final_block_count <= 50000, "Bounded common Final logical limit required")
    require(args.mode == "capture" or args.checkpoint is not None, "Finish requires a fresh Live checkpoint")
    require((args.mode == "capture") == (args.checkpoint_directory is not None), "Capture alone creates a checkpoint directory")
    require(args.mode != "capture" or args.checkpoint is None, "Capture cannot load a checkpoint")
    require((args.mode == "timing") == bool(args.finish_audit and args.quality_proof),
            "Timing alone consumes a fresh whole-Finish audit and independent native quality")
    require(args.mode == "timing" or not (args.finish_audit or args.quality_proof), "No historical authority in other modes")
    output = Path(args.output).resolve()
    require(output.is_relative_to(Path(root).resolve()/"benchmark-output"), "Keep private data in ignored benchmark-output")
    paths = (output, output.with_suffix(".geometry.npz"), output.with_suffix(".trace.jsonl"))
    require(not any(path.exists() for path in paths), "Require fresh outputs; preserve all prior measurements")
    if args.checkpoint_directory is not None:
        folder = Path(args.checkpoint_directory).resolve()
        require(folder.is_relative_to(Path(root).resolve()/"benchmark-output") and not folder.exists(),
                "Require a new private checkpoint directory")
    if idle is not None:
        require(idle(), "Stop the idle field server before offline research")
    return output


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", type=Path)
    p.add_argument("--mode", choices=tuple(KINDS), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--finish-audit", type=Path)
    p.add_argument("--quality-proof", type=Path)
    p.add_argument("--final-block-count", type=int)
    p.add_argument("--run-allocated", action="store_true")
    args = p.parse_args(argv)
    def idle():
        with socket.socket() as probe:
            probe.settimeout(.3)
            return probe.connect_ex(("127.0.0.1", 8000)) != 0
    try:
        args.output = preflight(args, idle=idle)
        args.session = args.session.resolve(strict=True)
        for name in ("checkpoint", "finish_audit", "quality_proof"):
            if getattr(args, name) is not None:
                setattr(args, name, getattr(args, name).resolve(strict=True))
        if args.checkpoint_directory is not None:
            args.checkpoint_directory = args.checkpoint_directory.resolve()
    except (BaseException) as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        p.error(str(error))
    return args


def core_artifacts():
    """Explicit production inventory, in addition to the authoritative core hash."""
    result = {}
    for directory in ("scanner_server", "shared", "native"):
        for path in sorted((ROOT/directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".toml") and "build" not in path.parts:
                result[path.relative_to(ROOT).as_posix()] = file_hash(path)
    require(bool(result), "Nonempty current production source inventory required")
    return result


def code_state(code):
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags,
        tuple(code_state(v) if isinstance(v, CodeType) else v for v in code.co_consts))


class LoadedOwners:
    """Explicit actual module/function owners, checked before and after work."""
    def __init__(self, modules, python_modules=(), observed_slots=()):
        self.modules = [(m.__name__, m, Path(m.__file__).resolve()) for m in modules]
        self.functions, self.classes, self.properties, self.wrapped, self.closures = [], [], [], [], []
        self.globals=[]
        observed=set(observed_slots)
        self.used_sources={}
        for module in python_modules:
            compiled = compile(Path(module.__file__).read_text(encoding="utf-8"), module.__file__, "exec", dont_inherit=True)
            for item in compiled.co_consts:
                if isinstance(item, CodeType):
                    fn = getattr(module, item.co_name, None)
                    if isinstance(fn, type):
                        self.classes.append((module,item.co_name,fn))
                        methods=[(fn,c.co_name,getattr(fn,c.co_name,None),c) for c in item.co_consts if isinstance(c,CodeType)]
                    else:methods=[(module,item.co_name,fn,item)]
                    for owner,name,method,expected in methods:
                        installed=method
                        if isinstance(method,property):
                            descriptor=method;method=descriptor.fget
                            self.properties.append((owner,name,descriptor,method))
                        while hasattr(method,"__wrapped__"):
                            require(isinstance(method,FunctionType),"Original decorator must remain an owned Python callable")
                            original=method.__wrapped__
                            defining=sys.modules.get(method.__globals__.get("__name__"))
                            require(defining is not None and method.__globals__ is defining.__dict__,"Decorator globals owner changed")
                            path=Path(defining.__file__).resolve()
                            self.used_sources[str(path)]=file_hash(path)
                            tree=compile(path.read_text(encoding="utf-8"),str(path),"exec",dont_inherit=True)
                            def codes(code):
                                yield code
                                for child in code.co_consts:
                                    if isinstance(child,CodeType):yield from codes(child)
                            require(any(code_state(c)==code_state(method.__code__) for c in codes(tree)),
                                    "Loaded original decorator differs from its actual source")
                            self.wrapped.append((method,original,method.__code__,repr(method.__defaults__),repr(method.__kwdefaults__)))
                            self.closures.append((method,tuple(c.cell_contents for c in method.__closure__ or ())))
                            method=original
                        require(isinstance(method,FunctionType) and method.__globals__ is module.__dict__
                            and code_state(method.__code__)==code_state(expected),
                                "Loaded controller/codec function differs from its source")
                        aliases=tuple((key,module.__dict__[key]) for key in method.__code__.co_names
                            if key in module.__dict__ and (module,key) not in observed)
                        self.globals.append((method,module.__dict__,aliases))
                        if not isinstance(installed,property):
                            slot_owner=None if (owner,name) in observed else owner
                            self.functions.append((slot_owner,name,getattr(owner,name),getattr(owner,name).__code__,repr(getattr(owner,name).__defaults__),repr(getattr(owner,name).__kwdefaults__)))
                            if getattr(owner,name) is not method:
                                self.functions.append((None,name,method,method.__code__,repr(method.__defaults__),repr(method.__kwdefaults__)))
                        else:
                            self.functions.append((None,name,method,method.__code__,repr(method.__defaults__),repr(method.__kwdefaults__)))
        self.check()

    def check(self):
        for name, module, path in self.modules:
            require(sys.modules.get(name) is module and Path(module.__file__).resolve() == path,
                    "Loaded numerical/helper module ownership changed")
        for module,name,owner in self.classes:
            require(getattr(module,name) is owner,"Loaded controller/codec class owner changed")
        for owner,name,descriptor,fn in self.properties:
            require(getattr(owner,name) is descriptor and descriptor.fget is fn,"Loaded engine property owner changed")
        for fn,original,code,defaults,keywords in self.wrapped:
            require(fn.__wrapped__ is original and fn.__code__ is code and repr(fn.__defaults__)==defaults
                and repr(fn.__kwdefaults__)==keywords,"Original decorator target/code/default owner changed")
        for fn,cells in self.closures:
            actual=tuple(c.cell_contents for c in fn.__closure__ or ())
            require(len(actual)==len(cells) and all(a is b for a,b in zip(actual,cells)),"Original decorator closure changed")
        for fn,namespace,aliases in self.globals:
            require(fn.__globals__ is namespace and all(namespace.get(name) is owner for name,owner in aliases),
                    "Original controller/codec/bundle imported helper owner changed")
        for module, name, fn, code, defaults, kwdefaults in self.functions:
            require((module is None or getattr(module, name) is fn) and fn.__code__ is code and repr(fn.__defaults__) == defaults
                    and repr(fn.__kwdefaults__) == kwdefaults, "Loaded controller/codec code/default changed")
        return True


def backend_contract(backend):
    require(backend.get("device") == "CUDA:0" and backend.get("requested") == "cuda"
        and backend.get("cuda_available") is True and backend.get("fallback_reason") is None
        and backend.get("tracking") == "legacy" and backend.get("tracking_device") == "CPU:0"
        and backend.get("native_kernels", {}).get("active") is True, "Actual original CUDA volume/native CPU tracking required")
    require(backend.get("cuda_input", {}).get("implementation") == "cuda"
        and backend.get("cuda_input", {}).get("fallback_batches") == 0
        and backend.get("depth_confidence", {}).get("requested") == "off"
        and backend.get("depth_confidence", {}).get("implementation") == "cpu"
        and backend.get("confidence_cuda", {}).get("implementation") == "fused"
        and backend.get("confidence_cuda", {}).get("fallback_reason") is None,
        "Original field input/CPU confidence algorithm/fused weighted fusion required")
    return True


def execute_finish(engine, *, mode, scope_factory, original_build, protocol, binding, scope_binding,
                   audit_path=None, quality_path=None, clock=time.perf_counter, progress=None):
    """One charged Final transaction; even swallowed stage faults remain fatal."""
    started = clock()
    permit = scope = None
    failures, primary, value = [], None, None
    try:
        if mode == "timing":
            permit = protocol.validate_finish_audit(audit_path, binding, scope_binding, quality_path=quality_path)
        scope = scope_factory(permit)
        with scope:
            value = scope.build(engine, original_build, progress_cb=progress)
            scope.finish()
        require(type(value) is tuple and len(value) == 2 and value[0] is True, "Original Final mesh build failed")
    except BaseException as error:
        primary = error
    finally:
        if permit is not None:
            clean("owned Finish proof close", lambda: protocol.close_permit(permit, primary), failures, primary)
        registration = clean("closed registration report", scope.report, failures, primary) if scope is not None else None
        ownership = clean("closed permit receipt", lambda: protocol.permit_report(permit), failures, primary)
        elapsed = clock()-started
    if primary is None and failures:
        primary = FinishProfileFailure("Finish transaction cleanup failed")
    record = {"finish_s": elapsed, "registration": registration, "owned_proof": ownership,
              "finish_cleanup_failures": failures, "built": value}
    if primary is not None:
        setattr(primary, "finish_profile_record", record)
        raise primary
    try:
        require(registration is not None and registration.get("complete") is True and registration.get("restored") is True
            and registration.get("closed") is True and registration.get("failure") is None
            and registration.get("cleanup_failures") == [], "Original scope did not close complete and restored")
        if mode == "timing":
            require(ownership.get("closed") is True and ownership.get("failure") is None
                and ownership.get("cleanup_failures") == [], "Timing proof did not close")
    except BaseException as error:
        setattr(error,"finish_profile_record",record)
        raise
    return record


def scope_identity(manifest, checkpoint, checkpoint_sha256, scope_base):
    require(manifest.get("scope_base") == scope_base, "Fresh checkpoint raw/settings/runtime scope differs")
    fresh = manifest["fresh_live"]
    require(fresh.get("unprocessed_count") == 0 and fresh.get("pose_source") == "fresh raw Live replay",
            "Checkpoint cannot use exported poses or outstanding processing")
    return frozen_json(dict(scope_base, checkpoint={"path": str(checkpoint), "sha256": checkpoint_sha256,
        "logical_state_sha256": manifest["logical_state_sha256"]}, fresh_live=fresh))


def export_geometry(engine, destination, np):
    """Original unrounded mesh arrays, with no alignment/quantized signature."""
    require(engine.mesh is not None, "Successful Final must have its original mesh")
    points=np.asarray(engine.mesh.vertices);faces=np.asarray(engine.mesh.triangles);colors=np.asarray(engine.mesh.vertex_colors)
    require(points.ndim==2 and points.shape[1]==3 and len(points)>0 and np.isfinite(points).all()
        and faces.ndim==2 and faces.shape[1]==3 and len(faces)>0 and faces.dtype.kind in "iu"
        and (faces>=0).all() and (faces<len(points)).all(), "Finite nonempty original Final mesh required")
    np.savez_compressed(destination,points=points,faces=faces,colors=colors)
    return {"artifact":str(Path(destination).resolve()),"sha256":file_hash(destination),
        "vertices":len(points),"triangles":len(faces),"bounds_m":[points.min(axis=0).tolist(),points.max(axis=0).tolist()]}


def main(argv=None):
    args = parse(argv)
    started = time.perf_counter()
    report = {"kind": KINDS[args.mode], "mode": args.mode, "status": "running", "start_utc": now(),
        "failure": None, "cleanup_failures": [], "cleanup_passed": False, "whole_finish_authority": False,
        "scope": "Fresh current production Live checkpoint. GPU changes bridge-subtree ICP only. Original proposal/frontier/gates/graph/Final weighted fusion/mesh remain in charge. Audit adds all query and same-input complete CPU ICP shadows; audit wall is not speed. Finish includes permit load, factory/cache/workspace setup, scope enter/build/exit and owned proof cleanup. Imports/ZIP decode/materialization, geometry export and final resource closure are separately recorded. No archived poses, approximate seeds, surface fitting or production promotion."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_report(args.output, report)
    old_env = {key: os.environ.get(key) for key in PIPELINE}
    os.environ.update(PIPELINE)
    threads = runtime = binding = boundary = engine = trace = None
    numerical_loaded=False
    primary, failures = None, report["cleanup_failures"]
    try:
        import cv2
        import numpy as np
        import cupy as cp
        import open3d as o3d
        numerical_loaded=True
        from PIL import Image
        from scanner_server.engine import ScanEngine
        from scanner_server import fragments, refinement, cuda_registration, bundle_adjustment
        from shared.settings import ScanSettings
        from scripts import profile_session as profile, process_metrics
        from scripts.research import private_live_checkpoint as codec, device_loop_icp as method
        from scripts.research import gpu_icp_finish_scope as injection, device_loop_finish_workspace as workspace
        from scripts.research import gpu_icp_finish_protocol as protocol
        from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
        threads = (cv2, o3d, cv2.getNumThreads(), o3d.utility.get_max_threads())
        cv2.setNumThreads(20); o3d.utility.set_max_threads(20)
        cv2.setRNGSeed(0); np.random.seed(0); random.seed(0); o3d.utility.random.seed(0)
        require(source_hash() == CURRENT, "Require the unchanged current production source")
        with zipfile.ZipFile(args.session) as archive:
            require(archive.getinfo("manifest.json").file_size <= 8*1024**2, "Bounded original archive manifest required")
            manifest = json.loads(archive.read("manifest.json"))
        selected = list(range(len(manifest["frames"])))
        require(1 <= len(selected) <= ScanEngine.MAX_FRAMES, "Complete bounded archive selection required")
        settings = ScanSettings.from_dict(manifest["settings"])
        original_settings_sha = hashlib.sha256(json.dumps(settings.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()
        if args.final_block_count is not None:
            settings = replace(settings, final_block_count=args.final_block_count)
        require(settings.confidence_fusion and settings.live_reconstruction and settings.reconnect_fragments,
                "Original weighted Live and Final fragment pipeline required")
        require(settings.voxel_m == .01 and settings.final_voxel_m == .005, "Exact captured 10mm Live/5mm Final field policy required")
        core = core_artifacts()
        checkpoint_artifacts = dict(core, **{name: file_hash(ROOT/name) for name in CHECKPOINT_FILES})
        artifact_names = set(FAMILY_FILES)|set(core)|set(method.source_contract()["artifacts"])|set(workspace.source_contract()["artifacts"])|set(protocol.source_contract()["artifacts"])
        binaries_modules = (sys.modules[o3d.geometry.PointCloud.__module__.split(".geometry")[0]],
            importlib.import_module("numpy._core._multiarray_umath"), importlib.import_module("cupy._core.core"),
            importlib.import_module("_kinect_native"))
        binaries = {key: {"path": str(Path(module.__file__).resolve()), "sha256": file_hash(module.__file__)}
                    for key, module in zip(("open3d", "numpy", "cupy", "native"), binaries_modules)}
        fixed = {str(args.session): file_hash(args.session), **{r["path"]:r["sha256"] for r in binaries.values()}}
        # Loaded OpenCV and compiler helpers are additional independent files.
        for module in (cv2, importlib.import_module("cupy.cuda.compiler")):
            path = str(Path(module.__file__).resolve()); fixed[path] = file_hash(path)
        cv_binaries=list(Path(cv2.__file__).resolve().parent.glob("cv2*.pyd"))
        require(len(cv_binaries)==1,"One actual loaded OpenCV backend binary required")
        cv_binary={"path":str(cv_binaries[0].resolve()),"sha256":file_hash(cv_binaries[0])}
        fixed[cv_binary["path"]]=cv_binary["sha256"]
        for path in (ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.dll",
                     ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.cpp",
                     ROOT/"benchmark-output/cuda-pipeline/resident-icp/resident_solve.build.json"):
            require(path.is_file(),"Original pinned Eigen bridge artifact required")
            fixed[str(path.resolve())] = file_hash(path)
        modules = [np, cp, o3d, cv2, fragments, refinement, cuda_registration, bundle_adjustment, codec, profile, process_metrics,
                   method, injection, workspace, protocol, sys.modules[__name__], sys.modules[ScanEngine.__module__],
                   sys.modules[DeviceFlatGridICP.__module__], *binaries_modules]
        # The source-bound scope owns these intentional output-only module-slot
        # replacements. Cold original bodies/defaults and all other imported
        # aliases remain checked here; scope checks original+installed wrappers.
        observed_slots=tuple((bundle_adjustment,name) for name in tuple(injection.BUNDLE_GATES)+("make_problem",))
        owners = LoadedOwners(modules, (codec, profile, process_metrics, sys.modules[__name__],sys.modules[ScanEngine.__module__],bundle_adjustment),
                             observed_slots=observed_slots)
        fixed.update(owners.used_sources)
        def runtime_value():
            with cp.cuda.Device(0): props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve()),
                "numpy": np.__version__, "cupy": cp.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__,
                "driver": cp.cuda.runtime.driverGetVersion(), "cuda": cp.cuda.runtime.runtimeGetVersion(),
                "nvrtc": list(cp.cuda.nvrtc.getVersion()), "hardware": {"name": name.decode() if isinstance(name, bytes) else str(name),
                    "compute_capability": [int(props["major"]),int(props["minor"])], "total_global_mem": int(props["totalGlobalMem"])},
                "binaries": binaries, "opencv_binary":cv_binary,"device": "CUDA:0",
                "numpy_build_sha256": hashlib.sha256(canonical(getattr(np.__config__, "CONFIG", {})).encode()).hexdigest(),
                "numpy_cpu_features": {k:bool(v) for k,v in getattr(binaries_modules[1],"__cpu_features__",{}).items()},
                "thread_policy": {"open3d": o3d.utility.get_max_threads(), "opencv": cv2.getNumThreads(), "omp": os.environ.get("OMP_NUM_THREADS")},
                "environment": {k:os.environ.get(k) for k in tuple(PIPELINE)+THREAD_ENV}}
        runtime = runtime_value
        def binding_value():
            return {"source_sha256": source_hash(), "artifacts_sha256": {name:file_hash(ROOT/name) for name in sorted(artifact_names)},
                "fixed_files": {name:file_hash(name) for name in sorted(fixed)}, "runtime": runtime(),
                "method_source": method.source_contract(), "workspace_source": workspace.source_contract(),
                "finish_proof_source": protocol.source_contract(), "finish_scope_source": injection.source_contract(),
                "configuration": {"graph":True,"chunk_iterations":4,"max_points":1_000_000,
                    "max_scratch_bytes":256*1024**2,"max_total_bytes":512*1024**2,"pair_cache_bytes":256*1024**2,"max_jobs":4096}}
        binding = binding_value
        report["binding"] = frozen_json(binding())
        def boundary_value():
            owners.check()
            require(runtime() == report["binding"]["runtime"], "Actual selected runtime/thread/environment changed")
        boundary = boundary_value
        boundary()
        scope_base = frozen_json({"archive": {"path":str(args.session),"sha256":fixed[str(args.session)]},
            "selected_indices":selected,"seed":0,"settings":settings.to_dict(),"pipeline_options":PIPELINE,
            "thread_policy":report["binding"]["runtime"]["thread_policy"],"runtime_binding":report["binding"]["runtime"]})
        report.update(scope_base=scope_base, checkpoint_artifacts_sha256=checkpoint_artifacts,
                      initial_preflight_wall_s=time.perf_counter()-started)
        save_report(args.output, report)
        if args.mode == "capture":
            decode_start=time.perf_counter(); frames=[]
            with zipfile.ZipFile(args.session) as archive:
                for item in manifest["frames"]:
                    with archive.open(item["rgb"]) as stream, Image.open(stream) as image:
                        rgb=np.array(image.convert("RGB"),dtype=np.uint8)
                    with archive.open(item["depth"]) as stream, Image.open(stream) as image:
                        depth=np.array(image,dtype=np.uint16)
                    frames.append((rgb,depth,item))
            report["image_decode_wall_s"]=time.perf_counter()-decode_start
            engine=ScanEngine(device="cuda",tracking="legacy"); engine.reset(settings=settings)
            live_start=time.perf_counter()
            with codec.PreparedInputRecorder(engine) as recorder:
                for index,(rgb,depth,item) in enumerate(frames):
                    metadata=dict(item.get("metadata",{}));metadata.pop("reference_pose",None)
                    metadata.update(frame_id=index,timestamp_s=item["timestamp_s"])
                    stored=engine.store_frame(rgb,depth,metadata)
                    require(stored.get("success") is True,"Original raw observation storage failed")
                    engine.process_frames()
                    print(f"Live {index+1}/{len(frames)} accepted={engine.frame_count}",flush=True)
                require(recorder.failure is None,"Original prepared-input observation failed")
            live_s=time.perf_counter()-live_start
            backend_contract(engine.backend); boundary()
            capture={"fresh_live_measured_s":live_s,"raw_selected_indices":selected,"pose_seeds_used":False,
                "stored":engine.stored_count,"accepted":engine.frame_count,"backend":engine.backend,
                "diagnostics":engine.diagnostics,"stage_totals_ms":engine.stage_totals_ms,
                "prepared_inputs":len(recorder.inputs),"input_observer_restored":engine.__dict__.get("_prepare_input") is None}
            checkpoint, checkpoint_manifest=codec.export_checkpoint(engine,recorder.inputs,args.checkpoint_directory,
                runtime_binding=report["binding"]["runtime"],scope_base=scope_base,artifacts=checkpoint_artifacts,capture_report=capture)
            report.update(capture=capture, checkpoint={"path":str(checkpoint),"sha256":file_hash(checkpoint)},
                scope_binding=scope_identity(checkpoint_manifest,checkpoint,file_hash(checkpoint),scope_base))
        else:
            material_start=time.perf_counter()
            engine,verification=codec.materialize_checkpoint(args.checkpoint,
                expected_binding=report["binding"]["runtime"],expected_artifacts=checkpoint_artifacts)
            checkpoint_manifest,checkpoint_sha=codec.validate_checkpoint_files(args.checkpoint,report["binding"]["runtime"],checkpoint_artifacts)
            report.update(checkpoint_verification=verification,materialization_wall_s=time.perf_counter()-material_start,
                checkpoint={"path":str(args.checkpoint),"sha256":checkpoint_sha},
                scope_binding=scope_identity(checkpoint_manifest,args.checkpoint,checkpoint_sha,scope_base))
            require(frozen_json(engine.settings.to_dict())==scope_base["settings"],"Restored original settings differ")
            backend_contract(engine.backend);boundary()
            before_indices=[i for i,_ in engine.poses]; live_diagnostics=frozen_json(engine.diagnostics);live_stages=dict(engine.stage_totals_ms)
            trace=args.output.with_suffix(".trace.jsonl").open("x",encoding="utf-8")
            def emit(row):
                trace.write(canonical(row)+"\n");trace.flush()
            def factory(permit):
                retrieval_factory = None if args.mode=="native" else lambda: DeviceFlatGridICP(device="CUDA:0",max_clouds=34,
                    max_cache_bytes=256*1024**2,audit_nearest=True,audit_misses=True,miss_policy="direct-miss-research-v1")
                workspace_factory = None if args.mode=="native" else workspace.DeviceLoopWorkspace
                return injection.GpuICPFinishScope(cuda_registration,fragments,refinement,mode=args.mode,
                    retrieval_factory=retrieval_factory,workspace_factory=workspace_factory,protocol=protocol,permit=permit,
                    trace=emit,boundary=boundary,numpy=np,engine=engine,bundle_module=bundle_adjustment)
            def progress(current,total,result):
                print(f"Finish {current}/{total}: {result.get('message','')}",flush=True)
            transaction=execute_finish(engine,mode=args.mode,scope_factory=factory,original_build=ScanEngine.build_mesh,
                protocol=protocol,binding=report["binding"],scope_binding=report["scope_binding"],
                audit_path=args.finish_audit,quality_path=args.quality_proof,progress=progress)
            report.update(transaction)
            registration=transaction["registration"]
            for output,key in (("calls","calls"),("events","events"),("graph_calls","graphs"),
                               ("final_pose_inventory","final_pose_inventory"),("workspace","workspace"),("cache_receipts","cache_receipts")):
                report[output]=registration[key]
            backend_contract(engine.backend)
            reconstruction=engine.reconstruction_report()
            export_start=time.perf_counter()
            geometry=export_geometry(engine,args.output.with_suffix(".geometry.npz"),np)
            report["geometry"]={"path":geometry["artifact"],"sha256":geometry["sha256"]}
            # Original reconstruction_report rounds presentation poses. Preserve
            # unrounded engine Final values for independent quality comparisons.
            poses=[{"index":index,"camera_to_world":np.asarray(pose).tolist()} for index,pose in engine.poses]
            report["profile"]={"schema_version":1,"session":args.session.name,"input_sha256":fixed[str(args.session)],
                "source_sha256":CURRENT,"input_changed_during_profile":False,"source_changed_during_profile":False,
                "selected_indices":selected,"seed":0,"frames":len(selected),"settings":settings.to_dict(),
                "settings_overrides":{"final_block_count":args.final_block_count} if args.final_block_count is not None else {},
                "original_settings_sha256":original_settings_sha,"pipeline_options":dict(PIPELINE),"backend":engine.backend,
                "thread_policy":report["binding"]["runtime"]["thread_policy"],"native_extension":dict(binaries["native"],changed_during_profile=False),
                "versions":{k:report["binding"]["runtime"][k] for k in ("python","open3d","opencv","numpy")},
                "live_s":checkpoint_manifest["capture_report"]["fresh_live_measured_s"],"finish_s":transaction["finish_s"],
                "finish_requested":True,"pose_seeds_used":False,"mesh_built":True,"build_result":transaction["built"][1],
                "accepted_before_finish":len(before_indices),"accepted_indices_before_finish":before_indices,
                "accepted":engine.frame_count,"accepted_indices":[i for i,_ in engine.poses],"poses":poses,"geometry":geometry,
                "live_stages":profile.stage_summary(live_diagnostics,live_stages,np),"live_diagnostics":live_diagnostics,
                "all_stages":profile.stage_summary(engine.diagnostics,engine.stage_totals_ms,np),"diagnostics":engine.diagnostics,
                "fragment_reconnection":engine.fragment_reconnection,"refinement":engine.refinement,
                "bundle_adjustment":engine.bundle_adjustment,"final_reconstruction":engine.final_reconstruction,
                "peak_process_rss_bytes":process_metrics.peak_rss_bytes()}
            report["geometry_export_wall_s"]=time.perf_counter()-export_start
            boundary()
            codec.validate_checkpoint_files(args.checkpoint,report["binding"]["runtime"],checkpoint_artifacts)
        report["loaded_owners_unchanged"]=True
    except BaseException as error:
        primary=error
        report["failure"]=error_record(error)
        if hasattr(error,"finish_profile_record"):
            report.update(error.finish_profile_record)
    finally:
        cleanup_start=time.perf_counter()
        if trace is not None:
            clean("trace close",trace.close,failures,primary)
            clean("trace SHA",lambda:report.update(trace={"path":str(args.output.with_suffix('.trace.jsonl')),
                "sha256":file_hash(args.output.with_suffix('.trace.jsonl'))}),failures,primary)
        if numerical_loaded:
            clean("original Open3D selected device completion",o3d.core.cuda.synchronize,failures,primary)
            clean("research CuPy selected device completion",cp.cuda.runtime.deviceSynchronize,failures,primary)
        if boundary is not None: clean("loaded runtime/thread/source closure",boundary,failures,primary)
        if binding is not None: report["binding_after"]=clean("full actual source/binary/raw closure",binding,failures,primary)
        if "binding" in report and report.get("binding_after")!=report["binding"]:
            failures.append({"action":"binding equality","message":"Before/after actual source/runtime/resources differ"})
        if args.mode=="capture" and report.get("checkpoint"):
            clean("closed checkpoint payload/source verification",lambda:codec.validate_checkpoint_files(
                Path(report["checkpoint"]["path"]),report["binding"]["runtime"],report["checkpoint_artifacts_sha256"]),failures,primary)
        if threads is not None:
            clean("OpenCV thread restoration",lambda:threads[0].setNumThreads(threads[2]),failures,primary)
            clean("Open3D thread restoration",lambda:threads[1].utility.set_max_threads(threads[3]),failures,primary)
            restored=clean("thread restoration getters",lambda:threads[0].getNumThreads()==threads[2]
                and threads[1].utility.get_max_threads()==threads[3],failures,primary)
            report["threads_restored"]=restored is True
            if restored is not True:failures.append({"action":"thread restoration","message":"Original thread policy not restored"})
        else:report["threads_restored"]=True
        for key,value in old_env.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        report["environment_restored"]={key:os.environ.get(key) for key in old_env}==old_env
        report["producer_cleanup_wall_s"]=time.perf_counter()-cleanup_start
        report["producer_wall_s_before_final_publication"]=time.perf_counter()-started
        report["end_utc"]=now()
        report["cleanup_passed"]=not failures and report["threads_restored"] and report["environment_restored"]
        passed=primary is None and report["cleanup_passed"] and report.get("loaded_owners_unchanged") is True
        report["status"]="passed" if passed else "failed"
        if primary is None and not passed:
            primary=FinishProfileFailure("Current whole-Finish input/source/runtime/cleanup closure failed")
            report["failure"]=error_record(primary)
        save_report(args.output,report,primary)
    if primary is not None:raise primary
    print(json.dumps({"mode":args.mode,"status":report["status"],"finish_s":report.get("finish_s"),
                      "accepted":report.get("profile",{}).get("accepted")}),flush=True)
    # Completion and all report/hash/restoration work MUST precede termination.
    # The isolated Windows CUDA wheel has unsafe DLL finalizers; root separately
    # records actual worker exit/wait. This runner never invents that receipt.
    from scripts.process_metrics import finish_cuda_worker
    finish_cuda_worker()


if __name__ == "__main__":
    main()
