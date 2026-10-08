"""Allocation-only boundary probe from a fresh CURRENT-core full raw replay.

Final poses come exclusively from the supplied completed unseeded raw profile,
never from the ZIP reconstruction. They authorize only the allocation boundary:
no tracking, ICP, graph, geometry or performance authority is created. Native
execution requires an explicit exclusive slot and a stopped field server.
Version 2 compares reconstructed settings as exact finite canonical JSON:
Python tuple arrays and persisted JSON lists share representation; numeric
values, types, signed zero, key membership and ordering within arrays stay exact.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.profile_session import file_hash, source_hash
from scripts.process_metrics import finish_cuda_worker, gpu_info

KIND = "current-full-raw-final-budget-boundary-probe-v2"
ARTIFACTS = ("scripts/research/probe_final_budget.py", "tests/test_final_budget_probe.py",
    "scripts/profile_session.py", "scripts/process_metrics.py")


class BudgetProbeFault(RuntimeError):
    pass


def require(value, message):
    if not value:raise BudgetProbeFault(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def validate_reconstructed_settings(actual, expected):
    """Exact JSON boundary, preserving numeric spelling and array order."""
    try:
        actual_json=json.dumps(actual,sort_keys=True,separators=(",",":"),allow_nan=False)
        expected_json=json.dumps(expected,sort_keys=True,separators=(",",":"),allow_nan=False)
    except (TypeError,ValueError) as error:
        raise BudgetProbeFault("Reconstructed settings require finite JSON values") from error
    require(actual_json == expected_json,"Declared actual settings do not reconstruct the fresh profile")
    return {"comparison":"exact finite canonical JSON; tuple arrays serialize as lists",
        "sha256":hashlib.sha256(actual_json.encode("utf-8")).hexdigest(),"numeric_tolerance":0}


def pose_serializer_contract(engine_class=None):
    """Original p.tolist() expression, with loaded method/source code consistency."""
    path=ROOT/"scanner_server/engine.py"
    module=ast.parse(path.read_text(encoding="utf-8"))
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name == "ScanEngine")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "reconstruction_report")
    require(len(method.body) == 1 and isinstance(method.body[0],ast.Return)
        and isinstance(method.body[0].value,ast.Dict),"Full-precision reconstruction serializer scope changed")
    dictionary=method.body[0].value
    pairs=[value for key,value in zip(dictionary.keys,dictionary.values) if isinstance(key,ast.Constant) and key.value == "poses"]
    expected=ast.parse('[{"index": i, "camera_to_world": p.tolist()} for i, p in self.poses]',mode="eval").body
    require(len(pairs) == 1 and ast.dump(pairs[0],include_attributes=False) == ast.dump(expected,include_attributes=False),
        "Original full-precision pose serializer changed")
    if engine_class is not None:
        tree=ast.Module(body=[method],type_ignores=[])
        # Do not inherit this helper's future-annotations flag into the
        # original engine module, which has no future-annotations directive.
        compiled=compile(tree,str(path),"exec",dont_inherit=True)
        source_code=next(c for c in compiled.co_consts if hasattr(c,"co_code"))
        def code_state(code):
            return (code.co_code,code.co_names,code.co_varnames,code.co_argcount,code.co_kwonlyargcount,
                code.co_posonlyargcount,code.co_freevars,code.co_cellvars,code.co_flags,
                tuple(code_state(c) if hasattr(c,"co_code") else c for c in code.co_consts))
        require(code_state(engine_class.reconstruction_report.__code__) == code_state(source_code),
            "Loaded full-precision serializer code differs from current source")
    return {"path":"scanner_server/engine.py", "method_ast_sha256":hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest(),
        "original_p_tolist_expression":True,"loaded_code_checked":engine_class is not None}


def validate_profile(value, *, current_source, input_digest, test_limit, expected_required):
    """Stdlib preflight rejects old cores/checkpoints/archived seeds/partial views."""
    require(type(test_limit) is int and type(expected_required) is int and 1 <= test_limit < expected_required <= 50000,
        "Require an intentionally insufficient bounded logical budget")
    require(value.get("schema_version") == 1 and value.get("finish_requested") is True
        and value.get("mesh_built") is True and value.get("pose_seeds_used") is False
        and value.get("input_changed_during_profile") is False and value.get("source_changed_during_profile") is False
        and value.get("source_sha256") == current_source and value.get("input_sha256") == input_digest,
        "Require a completed current-source/input unseeded full raw profile")
    require(not value.get("checkpoint") and not value.get("research_finish")
        and value.get("experimental_visual_fallback") is False
        and type(value.get("live_s")) in (int,float) and math.isfinite(value["live_s"]) and value["live_s"] > 0,
        "Checkpoint/research/archived reconstruction poses cannot authorize this raw allocation probe")
    count=value.get("frames")
    require(type(count) is int and 1 <= count <= 500 and value.get("selected_indices") == list(range(count))
        and all(type(i) is int for i in value["selected_indices"]) and len(value.get("live_diagnostics",[])) == count,
        "Require all original raw views in original order")
    settings=value["settings"];final=value["final_reconstruction"]
    require(all(type(final.get(k)) is int for k in ("required_blocks","blocks","initial_block_capacity","allocated_blocks")),
        "Require actual integer Final capacity/counts")
    require(settings.get("confidence_fusion") is True and settings.get("final_voxel_m") == .005
        and type(settings.get("final_block_count")) is int and settings["final_block_count"] >= expected_required
        and final.get("applied") is True and final.get("voxel_m") == .005
        and final.get("allocation_strategy") == "exact missing-key activation"
        and final.get("required_blocks") == final.get("blocks") == final.get("initial_block_capacity")
            == final.get("allocated_blocks") == expected_required,
        "Current completed weighted 5mm exact Final plan does not match the requested probe")
    overrides=value.get("settings_overrides",{})
    require(type(overrides) is dict and set(overrides) <= {"final_block_count"}, "Only declared logical budget override is supported")
    require(value["native_mode"] == "on" and value["native_extension"]["changed_during_profile"] is False
        and value["native_extension"].get("sha256"), "Completed original native runtime required")
    poses=value.get("poses");indices=value.get("accepted_indices")
    require(type(poses) is list and type(indices) is list and 0 < len(poses) == len(indices) == value["accepted"]
        and indices == sorted(set(indices)) and all(type(i) is int and 0 <= i < count for i in indices)
        and [p.get("index") for p in poses] == indices, "Final pose membership/order is invalid")
    for pose in poses:
        matrix=pose.get("camera_to_world")
        require(type(matrix) is list and len(matrix) == 4 and all(type(row) is list and len(row) == 4 for row in matrix)
            and all(type(x) in (int,float) and math.isfinite(x) for row in matrix for x in row)
            and matrix[3] == [0,0,0,1], "Require finite full-precision camera-to-world matrices")
    policy=value["thread_policy"]
    require(all(type(policy.get(k)) is int and 1 <= policy[k] <= 64 for k in ("open3d_threads","opencv_threads"))
        and type(value["omp_threads"]) is str and value["omp_threads"].isdigit(), "Actual raw thread policy missing")
    return {"frames":count,"final_views":len(poses),"required_blocks":expected_required,
        "settings_sha256":canonical(settings),"pose_provenance":"Full-precision p.tolist() from completed CURRENT-core unseeded full raw replay; allocation-only, never tracking seeds."}


def restore_hooks(owned):
    errors=[]
    for owner,name,original in reversed(owned):
        try:
            setattr(owner,name,original)
            require(getattr(owner,name) is original,"Hook identity not restored")
        except BaseException as error:errors.append({"action":name,"type":type(error).__name__,"message":str(error)})
    return errors


class AllocationBoundary:
    """Observe one unchanged original planner; deny every candidate/fusion call."""
    def __init__(self, owner, engine_class, *, logical_limit, required_blocks, pose_digest):
        self.owner=owner;self.cls=engine_class;self.limit=logical_limit;self.required=required_blocks
        self.pose_digest=pose_digest;self.owned=[];self.creations=[];self.plans=[];self.cleanup=[];self.fault=None

    def fail(self,message):
        error=BudgetProbeFault(message)
        if self.fault is None:self.fault=error
        raise error

    def install(self,name,value):
        self.owned.append((self.cls,name,getattr(self.cls,name)))
        setattr(self.cls,name,value)

    def __enter__(self):
        create=self.cls._create_vbg;planner=self.cls._required_fusion_blocks
        def guarded_create(candidate,block_count=None):
            if block_count != 1 or self.creations:self.fail("Final candidate or extra grid creation attempted")
            if candidate.voxel_size != .005 or candidate.settings.confidence_fusion is not True:
                self.fail("Scratch uses different Final voxel/confidence settings")
            if getattr(candidate,"_fusion_block_limit",None) != self.limit:self.fail("Logical budget not propagated")
            self.creations.append({"requested_blocks":block_count,"voxel_m":candidate.voxel_size,"complete":False})
            volume=create(candidate,block_count=block_count)
            capacity=int(volume.hashmap().capacity())
            if capacity != 1:self.fail("Original scratch native capacity differs")
            self.creations[-1].update(actual_capacity=capacity,complete=True)
            return volume
        def guarded_planner(candidate,proposals,progress_cb=None,*,stage="fragment_reconnection"):
            if self.plans or proposals is not self.owner.poses or self.pose_digest(self.owner.poses) != self.initial_pose_digest:
                self.fail("Planner inputs/order changed or planner repeated")
            if stage != "final_capacity_preflight":self.fail("Different planner stage used")
            self.plans.append({"stage":stage,"views":len(proposals),"complete":False})
            required=planner(candidate,proposals,progress_cb,stage=stage)
            if type(required) is not int or required != self.required:self.fail("Fresh raw Final required count differs")
            self.plans[-1].update(required_blocks=required,complete=True)
            return required
        def forbidden(*args,**kwargs):self.fail("Tracking, mesh construction or Final fusion attempted during allocation-only probe")
        self.initial_pose_digest=self.pose_digest(self.owner.poses)
        try:
            for name,value in (("_create_vbg",guarded_create),("_required_fusion_blocks",guarded_planner),
                ("_integrate_vbg",forbidden),("process_frames",forbidden),("build_mesh",forbidden)):
                self.install(name,value)
        except BaseException as primary:
            self.cleanup=restore_hooks(self.owned)
            for error in self.cleanup:primary.add_note(str(error))
            raise
        return self

    def __exit__(self,kind,error,traceback):
        self.cleanup=restore_hooks(self.owned)
        if self.cleanup:
            if error is not None:
                for item in self.cleanup:error.add_note("Allocation-only hook restoration: "+str(item))
            else:raise BudgetProbeFault("Allocation-only hook restoration failed: "+str(self.cleanup))
        return False

    def close(self,error):
        expected=(f"Final 0.005 m model needs {self.required} blocks; increase final_block_count from {self.limit} "
            f"to at least {self.required}, or choose a coarser final voxel. No Final fusion candidate was allocated.")
        require(self.fault is None and not self.cleanup and type(error) is ValueError and str(error) == expected,
            "Original typed early capacity error is absent or differs")
        require(len(self.creations) == len(self.plans) == 1 and self.creations[0]["complete"] is True
            and self.plans[0]["complete"] is True,"Original single scratch/planner closure absent")
        return {"create_calls":self.creations,"planner_calls":self.plans,"hooks_restored":True,
            "original_error_type":"ValueError","required_blocks":self.required,"logical_limit":self.limit,
            "candidate_allocated":False,"tracking_or_fusion_called":False}


def arrays_digest(np, pairs):
    digest=hashlib.sha256()
    for pair in pairs:
        for array in pair:
            require(array.flags.c_contiguous and not array.dtype.hasobject,"Require original contiguous numeric inputs")
            digest.update(str((array.shape,array.dtype.str)).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def owner_state(np,engine):
    return {"poses":arrays_digest(np,[(np.asarray(p),) for _,p in engine.poses]),
        "pose_indices":tuple(i for i,_ in engine.poses),"pose_owner":id(engine.poses),
        "raw":arrays_digest(np,engine.raw_frames),"raw_owner":id(engine.raw_frames),
        "raw_array_owners":tuple(tuple(id(a) for a in pair) for pair in engine.raw_frames),
        "metadata_sha256":canonical(engine.frame_metadata),"settings_owner":id(engine.settings),
        "volume_owner":id(engine.vbg),"volume_capacity":int(engine.vbg.hashmap().capacity()),
        "volume_size":int(engine.vbg.hashmap().size()),"mesh_owner":id(engine.mesh),
        "model_owner":id(engine.model_pcd),"point_cloud_owner":id(engine.point_cloud),
        "final_owner":id(engine._final_vbg),"final_reconstruction_sha256":canonical(engine.final_reconstruction),
        "input_adapter_owner":id(engine._input_preparation),"confidence_adapter_owner":id(engine._confidence_preparation),
        "cumulative_pose":arrays_digest(np,[(np.asarray(engine.cumulative_T),)])}


def run(args):
    require(args.run_allocated,"Require an explicitly allocated exclusive native/GPU slot")
    require(not args.output.exists(),"Preserve previous reports; require fresh output")
    args.output.resolve().relative_to(ROOT/"benchmark-output")
    with socket.socket() as probe:
        probe.settimeout(.3)
        require(probe.connect_ex(("127.0.0.1",8000)) != 0,"Stop the field server before isolated native allocation probes")
    inputs={"profile":file_hash(args.profile),"raw_zip":file_hash(args.session)}
    current=source_hash();profile=json.loads(args.profile.read_text(encoding="utf-8"))
    preflight=validate_profile(profile,current_source=current,input_digest=inputs["raw_zip"],
        test_limit=args.test_block_count,expected_required=args.expected_required_blocks)
    artifacts={name:file_hash(ROOT/name) for name in ARTIFACTS}
    report={"kind":KIND,"status":"running","source_sha256":current,"inputs_sha256":inputs,
        "artifacts_sha256":artifacts,"preflight":preflight,"geometry_authority":False,
        "registration_authority":False,"performance_claim":False,
        "scope":"Allocation-only fresh current raw replay poses. Fresh empty Live owners, not a restored Live checkpoint. Native global calibration caches/pools may warm; opaque state rollback is not claimed.",
        "cleanup_failures":[]}
    report["pose_serializer"]=pose_serializer_contract()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    environment=dict(profile["pipeline_options"],KINECT_NATIVE=profile["native_mode"],
        KINECT_BLOCK_COUNT=profile["initial_blocks"],OMP_NUM_THREADS=profile["omp_threads"])
    environment.update({k:profile["thread_policy"].get(k) for k in
        ("OMP_WAIT_POLICY","KMP_BLOCKTIME","OPENCV_FOR_THREADS_NUM","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")})
    previous={key:os.environ.get(key) for key in environment}
    failure=None;o3d=None;threads=None;boundary=None;extension_path=None
    try:
        save()
        for key,value in environment.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=str(value)
        import cv2
        import numpy as np
        import open3d as o3d
        from PIL import Image
        from scanner_server.engine import ScanEngine
        from scanner_server.fragments import _rigid
        from shared.settings import ScanSettings
        report["pose_serializer"]=pose_serializer_contract(ScanEngine)
        threads=(cv2.getNumThreads(),o3d.utility.get_max_threads())
        cv2.setNumThreads(profile["thread_policy"]["opencv_threads"])
        o3d.utility.set_max_threads(profile["thread_policy"]["open3d_threads"])
        require({"python":sys.version.split()[0],"numpy":np.__version__,"opencv":cv2.__version__,"open3d":o3d.__version__}
            == profile["versions"],"Actual loaded versions differ from the fresh raw profile")
        extension=importlib.util.find_spec("_kinect_native")
        require(extension is not None and extension.origin,"Actual native extension unavailable")
        extension_path=Path(extension.origin)
        report["native_extension_sha256"]=file_hash(extension_path)
        require(report["native_extension_sha256"] == profile["native_extension"]["sha256"],
            "Actual native extension bytes differ from the fresh raw profile")
        report["gpu_hardware"]=gpu_info()
        require(report["gpu_hardware"] == profile["gpu_hardware"],"Actual GPU/driver identity differs from the fresh raw profile")
        with zipfile.ZipFile(args.session) as archive:
            manifest=json.loads(archive.read("manifest.json"))
            require(len(manifest["frames"]) == preflight["frames"],"Raw ZIP selection differs")
            settings=ScanSettings.from_dict(manifest["settings"])
            require(hashlib.sha256(json.dumps(settings.to_dict(),sort_keys=True,allow_nan=False).encode()).hexdigest()
                == profile["original_settings_sha256"],"Original raw loader/settings differ")
            settings=replace(settings,**profile.get("settings_overrides",{}))
            report["settings_reconstruction"]=validate_reconstructed_settings(settings.to_dict(),profile["settings"])
            engine=ScanEngine(device="cuda" if profile["backend"]["device"] == "CUDA:0" else "cpu",tracking=profile["backend"]["tracking"])
            engine.reset(settings=replace(settings,final_block_count=args.test_block_count))
            require(str(engine.device) == profile["backend"]["device"],"Actual selected allocation device differs")
            for index,item in enumerate(manifest["frames"]):
                require(all(archive.getinfo(item[key]).file_size <= 20*1024**2 for key in ("rgb","depth")),"Unbounded encoded raw image")
                with archive.open(item["rgb"]) as stream,Image.open(stream) as image:
                    rgb=np.array(image.convert("RGB"),dtype=np.uint8)
                with archive.open(item["depth"]) as stream,Image.open(stream) as image:
                    depth=np.array(image,dtype=np.uint16)
                metadata=dict(item.get("metadata",{}));metadata.pop("reference_pose",None)
                metadata.update(frame_id=index,timestamp_s=item["timestamp_s"])
                require(engine.store_frame(rgb,depth,metadata)["success"] is True,"Original raw storage failed")
        engine.poses=[(row["index"],np.asarray(row["camera_to_world"],dtype=np.float64).copy()) for row in profile["poses"]]
        require(all(_rigid(pose) for _,pose in engine.poses),"Fresh Final full-precision poses are not rigid")
        engine.frame_count=len(engine.poses);engine.cumulative_T=engine.poses[-1][1].copy()
        before=owner_state(np,engine)
        digest=lambda poses:arrays_digest(np,[(np.asarray(pose),) for _,pose in poses])
        expected_error=None
        boundary=AllocationBoundary(engine,ScanEngine,logical_limit=args.test_block_count,
            required_blocks=args.expected_required_blocks,pose_digest=digest)
        with boundary:
            try:engine._final_volume()
            except ValueError as error:expected_error=error
        report["boundary"]=boundary.close(expected_error)
        after=owner_state(np,engine)
        require(before == after,"Fresh allocation-owner/raw-input/pose bytes changed on early failure")
        report.update(owner_state_unchanged=True,raw_input_sha256=before["raw"],
            final_pose_bytes_sha256=before["poses"],actual_versions=profile["versions"],
            actual_native_sha256=file_hash(Path(extension.origin)),actual_device=str(engine.device),
            actual_threads={"open3d":o3d.utility.get_max_threads(),"opencv":cv2.getNumThreads(),"omp":os.environ.get("OMP_NUM_THREADS")})
    except BaseException as error:
        failure=error;report["failure"]={"type":type(error).__name__,"message":str(error)}
    finally:
        actions=[]
        if o3d is not None:actions.append(("device synchronization",lambda:o3d.core.cuda.synchronize() if o3d.core.cuda.is_available() else None))
        if threads is not None:actions.extend((("OpenCV thread restore",lambda:cv2.setNumThreads(threads[0])),
            ("Open3D thread restore",lambda:o3d.utility.set_max_threads(threads[1]))))
        actions.append(("source/input closure",lambda:report.update(source_sha256_after=source_hash(),
            inputs_sha256_after={"profile":file_hash(args.profile),"raw_zip":file_hash(args.session)},
            artifacts_sha256_after={name:file_hash(ROOT/name) for name in ARTIFACTS})))
        if extension_path is not None:
            actions.append(("native/runtime closure",lambda:report.update(
                native_extension_sha256_after=file_hash(extension_path),gpu_hardware_after=gpu_info())))
        for name,action in actions:
            try:action()
            except BaseException as error:
                report["cleanup_failures"].append({"action":name,"type":type(error).__name__,"message":str(error)})
                if failure is None:failure=error
        for key,value in previous.items():
            try:
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
            except BaseException as error:
                report["cleanup_failures"].append({"action":"environment restore "+key,"type":type(error).__name__,"message":str(error)})
                if failure is None:failure=error
        report["environment_restored"]=all(os.environ.get(k)==v for k,v in previous.items())
        passed=(failure is None and not report["cleanup_failures"] and report["environment_restored"]
            and report.get("source_sha256_after") == current and report.get("inputs_sha256_after") == inputs
            and report.get("artifacts_sha256_after") == artifacts and report.get("owner_state_unchanged") is True
            and report.get("native_extension_sha256_after") == report.get("native_extension_sha256")
                == profile["native_extension"]["sha256"]
            and report.get("gpu_hardware_after") == report.get("gpu_hardware") == profile["gpu_hardware"])
        report["status"]="passed" if passed else "failed"
        try:save()
        except BaseException as secondary:
            if failure is not None:failure.add_note(f"Final budget probe write also failed: {secondary}");raise failure from secondary
            raise
    if not passed:raise BudgetProbeFault("Allocation-only budget probe failed; report preserved") from failure
    print(json.dumps({"output":str(args.output),"status":report["status"],"candidate_allocated":False}),flush=True)
    finish_cuda_worker()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session",type=Path);parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--run-allocated",action="store_true")
    parser.add_argument("--test-block-count",type=int,default=10000)
    parser.add_argument("--expected-required-blocks",type=int,default=13302)
    args=parser.parse_args()
    if args.output.exists():parser.error("Preserve prior reports; use fresh output")
    run(args)


if __name__ == "__main__":main()
