"""Fresh physical smoke of installed weighted Final missing-key activation.

Both engine policies call the same current CPU/tensor/fused fusion functions
on native VBGs. No proxy or historical proof token is used. This establishes
only within-backend synthetic attribute bits/capacity, never speed or mesh
quality. Numerical imports require an explicitly allocated exclusive slot.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.profile_session import file_hash,source_hash
from scripts.process_metrics import finish_cuda_worker,peak_rss_bytes
from scripts.research import benchmark_missing_activation as frozen

KIND="production-weighted-final-missing-activation-physical-smoke-v1"
HELPER="scripts/research/benchmark_missing_activation.py"
HELPER_SHA256="97f4e3810087d71d452513ac5673fe5e0d5ea7c41979a65588b9baf48dea3723"
ARTIFACTS=("scripts/research/benchmark_production_missing_activation.py",
    "tests/test_production_missing_activation_smoke.py",HELPER,
    "scanner_server/fusion_allocation.py","scanner_server/engine.py",
    "scanner_server/weighted_fusion.py","scanner_server/cuda_fusion.py","scanner_server/weighted_fusion.cu",
    "shared/native.py","native/kernels.cpp","native/weighted_fusion.h",
    "scripts/profile_session.py","scripts/process_metrics.py")
BACKENDS=("cpu-original","cuda-tensor","cuda-fused")
KEYS=((-1,-1,12),(0,-1,12),(-1,0,12),(0,0,12))
BATCHES=(KEYS[:3],(KEYS[1],KEYS[3],KEYS[0]),KEYS)


def pins():return {name:file_hash(ROOT/name) for name in ARTIFACTS}


def policy_engine(device,candidate):
    return SimpleNamespace(device=device,max_depth_m=3.,sdf_trunc=.08,voxel_size=.005,
        settings=SimpleNamespace(confidence_fusion=True,camera=SimpleNamespace(
            width=32,height=24,fx=40.,fy=40.,cx=15.5,cy=11.5)),
        _final_missing_only_activation=candidate,_final_allocated_blocks=4,_fusion_block_limit=4)


def validate_pair(pair):
    if (pair.get("inputs_unchanged") is not True or pair.get("nonzero_weight_voxels",0)<=0
            or pair.get("fractional_confidence_pixels",0)<=0 or pair.get("fractional_weight_voxels",0)<=0
            or pair["ordinary"]["initial_capacity"]!=4 or pair["ordinary"]["final_capacity"]!=8
            or pair["candidate"]["initial_capacity"]!=4 or pair["candidate"]["final_capacity"]!=4
            or any(pair[mode]["final_blocks"]!=4 or len(pair[mode]["frames"])!=3 for mode in ("ordinary","candidate"))
            or any([frame.get("frustum_rows") for frame in pair[mode]["frames"]]!=[3,3,4]
                or [frame.get("active_keys") for frame in pair[mode]["frames"]]!=[3,4,4]
                or [frame.get("capacity") for frame in pair[mode]["frames"]]!=(
                    [4,8,8] if mode=="ordinary" else [4,4,4]) for mode in ("ordinary","candidate"))
            or any(pair["comparison"][name]["bit_mismatches"]!=0
                or pair["comparison"][name]["elements"]<=0 for name in ("tsdf","weight","color"))):
        raise RuntimeError("Current production activation capacity/input/nonzero/fractional/attribute gate failed")
    pair["complete"]=True


def compare_backend(name,np,o3d,cp,weighted,cuda,kernel,pair):
    core=o3d.core;device=core.Device("CPU:0" if name=="cpu-original" else "CUDA:0")
    rgb,depth,confidence,extrinsic=frozen.input_arrays(np)
    arrays=dict(rgb=rgb,depth=depth,confidence=confidence,extrinsic=extrinsic)
    pair.update(backend=name,device=str(device),complete=False,
        ordered_frustum_keys=[[list(key) for key in batch] for batch in BATCHES],
        inputs={key:frozen.array_record(value) for key,value in arrays.items()},
        fractional_confidence_pixels=int(np.count_nonzero((confidence>0)&(confidence<1))))
    snapshots={}
    for mode,candidate in (("ordinary",False),("candidate",True)):
        engine=policy_engine(device,candidate)
        volume=o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf","weight","color"),
            attr_dtypes=(core.float32,)*3,attr_channels=((1,),(1,),(3,)),
            voxel_size=.005,block_resolution=16,block_count=4,device=device)
        row=dict(initial_capacity=int(volume.hashmap().capacity()),frames=[])
        pair[mode]=row
        for batch in BATCHES:
            blocks=core.Tensor(batch,dtype=core.int32,device=device)
            if name=="cpu-original":weighted._integrate_cpu(engine,volume,blocks,rgb,depth,extrinsic,confidence)
            elif name=="cuda-tensor":weighted._integrate_tensor(engine,volume,blocks,rgb,depth,extrinsic,confidence)
            else:cuda.integrate(engine,volume,blocks,rgb,depth,extrinsic,confidence,cp,kernel)
            if name!="cpu-original":core.cuda.synchronize(device)
            row["frames"].append(dict(frustum_rows=len(batch),capacity=int(volume.hashmap().capacity()),
                active_keys=int(volume.hashmap().size())))
        row.update(final_capacity=int(volume.hashmap().capacity()),final_blocks=int(volume.hashmap().size()))
        snapshots[mode]=frozen.key_mapped_attributes(volume,KEYS,np,core)
        row["attribute_snapshots"]={key:frozen.array_record(value) for key,value in snapshots[mode].items()}
    pair["inputs_after"]={key:frozen.array_record(value) for key,value in arrays.items()}
    pair["inputs_unchanged"]=pair["inputs_after"]==pair["inputs"]
    pair["comparison"]={name:dict(elements=snapshots["ordinary"][name].size,
        bit_mismatches=int(np.count_nonzero(snapshots["ordinary"][name].view(np.uint32)
            !=snapshots["candidate"][name].view(np.uint32)))) for name in ("tsdf","weight","color")}
    weights=snapshots["candidate"]["weight"]
    pair.update(nonzero_weight_voxels=int(np.count_nonzero(weights>0)),
        fractional_weight_voxels=int(np.count_nonzero((weights>0)&(weights!=np.floor(weights)))))
    validate_pair(pair)


def run(args):
    if not args.run_allocated:raise ValueError("Require an explicitly allocated exclusive native/GPU slot")
    if args.output.exists():raise ValueError("Require fresh output; preserve prior physical reports")
    args.output.resolve().relative_to(ROOT/"benchmark-output")
    if file_hash(ROOT/HELPER)!=HELPER_SHA256:raise RuntimeError("Immutable synthetic input/key mapping helper changed")
    if not (ROOT/"scanner_server/fusion_allocation.py").is_file():
        raise RuntimeError("Production allocation change is not installed; this smoke cannot run on the old core")
    report=dict(kind=KIND,status="running",pairs=[],performance_claim=False,whole_session_quality_proven=False,
        helper_sha256=HELPER_SHA256,source_sha256=source_hash(),artifacts_sha256=pins(),
        start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope="Same installed CPU/tensor/fused functions; native VBG ordinary False versus private Final True/cap4/limit4. Exact within-backend per-key bits; no historical authority, proxy or speed claim.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    previous={key:os.environ.get(key) for key in frozen.THREAD_ENV};failure=None;cleanup=[];o3d=None
    try:
        save();os.environ.update(frozen.THREAD_ENV)
        import numpy as np
        import open3d as o3d
        import cupy as cp
        from scanner_server import weighted_fusion as weighted,cuda_fusion as cuda
        from shared.native import kernels
        o3d.utility.set_max_threads(20)
        if not o3d.core.cuda.is_available() or kernels() is None:raise RuntimeError("Actual CUDA and native CPU kernel required")
        fused,kernel,reason=cuda._kernel(0)
        if fused is not cp or kernel is None or reason is not None:raise RuntimeError("Actual fused CUDA kernel unavailable")
        with cp.cuda.Device(0),cp.cuda.Stream.null:
            transport=o3d.core.Tensor([0.],dtype=o3d.core.float32,device=o3d.core.Device("CUDA:0"))
            borrowed=cp.from_dlpack(transport.to_dlpack());cp.asarray(np.zeros(1,np.float32))
            cp.cuda.Stream.null.synchronize()
        o3d.core.cuda.synchronize(o3d.core.Device("CUDA:0"));del borrowed,transport
        report["runtime_binding"]=frozen.native_binding(np,o3d,cp)
        for name in BACKENDS:
            pair={};report["pairs"].append(pair)
            compare_backend(name,np,o3d,cp,weighted,cuda,kernel,pair);save()
        report["runtime_binding_after"]=frozen.native_binding(np,o3d,cp)
        if report["runtime_binding_after"]!=report["runtime_binding"]:raise RuntimeError("Actual runtime/native/thread binding changed")
    except BaseException as error:
        failure=error;report["failure"]=dict(type=type(error).__name__,message=str(error))
    finally:
        for name,action in (("device synchronization",lambda:o3d.core.cuda.synchronize(o3d.core.Device("CUDA:0")) if o3d else None),
                ("source closure",lambda:report.update(source_sha256_after=source_hash(),artifacts_sha256_after=pins())),
                ("peak RSS",lambda:report.update(peak_process_rss_bytes=peak_rss_bytes()))):
            try:action()
            except BaseException as error:
                cleanup.append(dict(action=name,type=type(error).__name__,message=str(error)))
                if failure is None:failure=error
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        report.update(cleanup_failures=cleanup,environment_restored=all(os.environ.get(k)==v for k,v in previous.items()))
        passed=(failure is None and not cleanup and len(report["pairs"])==3 and all(p.get("complete") is True for p in report["pairs"])
            and report.get("source_sha256_after")==report["source_sha256"]
            and report.get("artifacts_sha256_after")==report["artifacts_sha256"] and report["environment_restored"])
        report.update(status="passed" if passed else "failed",end_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        try:save()
        except BaseException as secondary:
            if failure is not None:failure.add_note(f"Final physical diagnostics write also failed: {secondary}");raise failure from secondary
            raise
    if not passed:raise RuntimeError("Current production physical smoke failed; report preserved") from failure
    print(json.dumps(dict(output=str(args.output),status=report["status"])),flush=True)
    finish_cuda_worker()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-allocated",action="store_true");parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("Require fresh output")
    run(args)


if __name__=="__main__":main()
