"""Separate warp + host-bound raw-kernel screening proof/timing; no ICP/Finish authority.

Historical NPZ arrays are input only. Every row compares all five raw columns
bit-for-bit with the original exhaustive shader, then original Open3D CPU
nearest hit/miss/ambiguity resolution is audited. Float shadow is freshly
prepared by RN conversion, validated and bound before/after each case.
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OLD=ROOT/"scripts/research/archive/research_flat_grid_nn.cu"
NEW=Path(__file__).with_name("research_warp_flat_grid_filtered_nn.cu")
WARP=Path(__file__).with_name("microbatch_warp_flat_grid_nn.cu")
QUERY_CAP=4.
FILES=(str(Path(__file__).relative_to(ROOT)),str(OLD.relative_to(ROOT)),str(NEW.relative_to(ROOT)),
    "scripts/research/flat_grid_filter_bound.py","tests/test_flat_grid_filter_bound.py",
    "scripts/research/research_flat_grid_filtered_nn.cu","scripts/research/benchmark_flat_grid_filtered_nn.py",
    "scripts/research/microbatch_warp_flat_grid_nn.cu","tests/test_warp_flat_grid_filter.py",
    "scripts/research/archive/cuda_uniform_grid_registration.py","scripts/research/archive/research_uniform_grid_nn.cu","scanner_server/cuda_nn_registration.py",
    "scripts/research/archive/validate_uniform_grid_proof.py","scripts/profile_session.py","scripts/process_metrics.py")
# CuPy's _compile_with_cache_cuda already adds -ftz=true. A second request
# is rejected by NVRTC; the independent bound covers either FTZ convention.
OPTIONS=("--std=c++11","--fmad=false")
CAP=96*1024**2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def descriptor(array):
    return {"dtype":array.dtype.str,"shape":list(array.shape),"nbytes":int(array.nbytes),
        "sha256":hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--trace",type=Path)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--run-allocated",action="store_true")
    args=p.parse_args();args.output=args.output.resolve()
    if not args.run_allocated or not 3<=args.repeats<=20 or bool(args.trace)!=bool(args.manifest):
        p.error("Require exclusive slot,3..20 repeats and both trace/manifest if supplied")
    if args.output.exists() or not args.output.is_relative_to(ROOT/"benchmark-output"):
        p.error("Require fresh private benchmark-output report")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1",8000))==0:p.error("Stop idle server before allocated experiment")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={"kind":"warp-host-bound-float-screen-original-double-grid-neighbours-v3","status":"running",
        "start_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"failure":None,"cleanup_failures":[],
        "authority":"Raw lookup component only; no typed ICP/proposal/Finish/production authority.",
        "timers":"Preuploaded preallocated kernel enqueue+selected-stream completion, GPU event separately. Output and diagnostic D2H outside these timers. Grid/shadow/query setup and cleanup separately charged; no end-to-end speed claim.",
        "compiler_options":list(OPTIONS),"declared_query_max_abs_cap":QUERY_CAP,"gpu_numeric_memory_cap_bytes":CAP,
        "host_fixture_scope":"Numeric NPZ uncompressed cap128MiB; host/native/Python allocations are separate from the GPU numeric cap.","cases":[],"repeats":args.repeats}
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    primary=None;adapter=None;cp=None;threads=None
    old_omp=os.environ.get("OMP_NUM_THREADS")
    try:
        save();os.environ["OMP_NUM_THREADS"]="8"
        import numpy as np
        import cupy as cp
        import open3d as o3d
        from scripts.profile_session import source_hash
        from scripts.research.flat_grid_filter_bound import error_bound
        from scripts.research.archive.cuda_uniform_grid_registration import UniformGridICP,dyadic_shift
        threads=(o3d,o3d.utility.get_max_threads());o3d.utility.set_max_threads(20)
        compiler=importlib.import_module("cupy.cuda.compiler")
        compiler_path=Path(compiler.__file__).resolve();compiler_before=sha(compiler_path)
        compiler_tree=ast.parse(compiler_path.read_text(encoding="utf-8"))
        compile_function=next(n for n in compiler_tree.body if isinstance(n,ast.FunctionDef) and n.name=="_compile_with_cache_cuda")
        ftz_append=ast.parse("options += ('-ftz=true',)").body[0]
        if sum(ast.dump(n,include_attributes=False)==ast.dump(ftz_append,include_attributes=False) for n in compile_function.body)!=1:
            raise ValueError("Actual CuPy CUDA wrapper must append exactly one explicit FTZ option")
        report["compiler_policy"]={"helper_path":str(compiler_path),"helper_sha256":compiler_before,
            "requested_options":list(OPTIONS),"wrapper_appended_option":"-ftz=true",
            "effective_numeric_options":list(OPTIONS)+["-ftz=true"],
            "scope":"Numeric options only; CuPy also supplies architecture/header/backend options."}
        source_before=source_hash();artifacts={name:sha(ROOT/name) for name in FILES}
        modules={m.__name__:(m,str(Path(m.__file__).resolve())) for m in (np,cp,o3d,compiler)}
        binary_modules=(sys.modules[o3d.geometry.PointCloud.__module__.split(".geometry")[0]],
            importlib.import_module("numpy._core._multiarray_umath"),importlib.import_module("cupy._core.core"))
        modules.update({m.__name__:(m,str(Path(m.__file__).resolve())) for m in binary_modules})
        binaries={str(Path(m.__file__).resolve()):sha(m.__file__) for m in binary_modules}
        def runtime():
            with cp.cuda.Device(0):props=cp.cuda.runtime.getDeviceProperties(0)
            name=props["name"]
            return {"python":sys.version.split()[0],"executable":str(Path(sys.executable).resolve()),
                "numpy":np.__version__,"cupy":cp.__version__,"open3d":o3d.__version__,
                "driver":cp.cuda.runtime.driverGetVersion(),"cuda":cp.cuda.runtime.runtimeGetVersion(),
                "nvrtc":list(cp.cuda.nvrtc.getVersion()),"device":"CUDA:0",
                "gpu_name":name.decode() if isinstance(name,bytes) else str(name),
                "compute_capability":[int(props["major"]),int(props["minor"])],
                "open3d_threads":o3d.utility.get_max_threads(),"omp":os.environ.get("OMP_NUM_THREADS")}
        report.update(source_sha256=source_before,artifacts_sha256=artifacts,runtime=runtime(),binaries_sha256=binaries)
        fixed={str(compiler_path):compiler_before};cases=[]
        if args.trace:
            args.trace,args.manifest=args.trace.resolve(),args.manifest.resolve()
            if not (args.trace.is_relative_to(ROOT/"benchmark-output") and args.manifest.is_relative_to(ROOT/"benchmark-output")):
                raise ValueError("Only locally owned private data fixtures")
            if args.trace.stat().st_size>128*1024**2 or args.manifest.stat().st_size>8*1024**2:raise ValueError("Bounded trace")
            with zipfile.ZipFile(args.trace) as zipped:
                if len(zipped.infolist())>64 or sum(i.file_size for i in zipped.infolist())>128*1024**2:raise ValueError("Bounded numeric NPZ")
            manifest=json.loads(args.manifest.read_text(encoding="utf-8-sig"))
            fixed.update({str(args.trace):sha(args.trace),str(args.manifest):sha(args.manifest)})
            if fixed[str(args.trace)]!=manifest["trace_sha256"]:raise ValueError("Trace manifest hash mismatch")
            if not 1<=len(manifest["batches"])<=12:raise ValueError("At most12 actual batches")
            with np.load(args.trace,allow_pickle=False) as data:
                for row in manifest["batches"]:
                    arrays=[]
                    for key in ("target_points","queries"):
                        value=data[f"batch{row['index']}_{key}"];bound=row["arrays"][key]
                        record=descriptor(value)
                        if (record["dtype"]!=bound["dtype"] or record["shape"]!=bound["shape"]
                                or record["nbytes"]!=bound["bytes"] or record["sha256"]!=bound["sha256"]):raise ValueError("Trace exact bytes mismatch")
                        arrays.append(np.ascontiguousarray(value))
                    cases.append((f"trace-{row['index']}",*arrays,row["radius_m"],None))
            report["trace_input"]={"npz_sha256":fixed[str(args.trace)],"manifest_sha256":fixed[str(args.manifest)],"scope":"Historical arrays only; fresh reference/shadow below"}
        report["fixed_files_sha256"]=fixed
        rng=np.random.default_rng(1234)
        for radius in (.12,.06,.03):
            points=rng.uniform(-.2,.2,(4096,3));queries=points[rng.integers(0,len(points),2048)]+rng.normal(0,.002,(2048,3))
            cases.append((f"random-{radius}",points,queries,radius,None))
        tiny=np.nextafter(0.,1.)
        focused=[("duplicates",[[0,0,0],[0,0,0],[.03,0,0]],[[0,0,0],[.015,0,0],[.06,0,0]],.03,None),
            ("subnormal",[[-tiny,0,0],[tiny,0,0]],[[0,0,0],[tiny,0,0]],2.**-20,None),
            ("float-halfway",[[1.+2.**-25,0,0],[1.,0,0],[1.-2.**-26,0,0]],[[0,0,0],[1,0,0]],1.,None),
            ("cell-boundaries",[[.0625,0,0],[.0625,.0625,0],[-.0625,0,0]],[[np.nextafter(.0625,0),0,0],[np.nextafter(.0625,1),0,0]],.06,None),
            ("fewer-two",[[0,0,0]],[[0,0,0],[.1,0,0]],.03,None),
            ("declined-bound",[[0,0,0],[.01,0,0],[.02,0,0]],[[.001,0,0]],.03,float("inf")),
            ("edge-cell",[[1048575.,0,0],[-1048576.,0,0]],[[1048575.5,0,0],[-1048575.5,0,0]],1.,None),
            ("original-unsupported-query",[[0,0,0],[.01,0,0]],[[2000000.,0,0]],1.,None),
            ("outside-host-cap",[[4.1,0,0],[4.2,0,0]],[[4.11,0,0],[4.19,0,0]],.12,None)]
        cases.extend((label,np.array(points,dtype=np.float64),np.array(queries,dtype=np.float64),radius,forced) for label,points,queries,radius,forced in focused)
        adapter=UniformGridICP(max_clouds=1,max_cache_bytes=64*1024**2,audit_nearest=True,audit_misses=True,miss_policy="direct-miss-research-v1")
        with cp.cuda.Device(0),cp.cuda.Stream.null:
            kernels={}
            for key,path,entry in (("exhaustive",OLD,"flat_grid_nearest_two"),("warp-exhaustive",WARP,"flat_grid_nearest_two"),("filtered",NEW,"warp_filtered_grid_nearest_two"),("shadow",NEW,"make_float_shadow")):
                begun=time.perf_counter();kernel=cp.RawKernel(path.read_text(encoding="utf-8"),entry,options=OPTIONS);kernel.compile();kernels[key]=kernel
                report.setdefault("kernel_setup",[]).append({"variant":key,"compile_wall_s":time.perf_counter()-begun})
        for label,points,queries,radius,forced in cases:
            if (points.dtype!=np.float64 or queries.dtype!=np.float64 or points.shape[1:]!=(3,) or queries.shape[1:]!=(3,)
                    or not points.flags.c_contiguous or not queries.flags.c_contiguous or not np.isfinite(points).all() or not np.isfinite(queries).all()
                    or not 1<=len(points)<=1_000_000 or not 1<=len(queries)<=100_000):raise ValueError("Bounded finite original-double case")
            before={"points":descriptor(points),"queries":descriptor(queries)};target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            begun=time.perf_counter();item=adapter._dataset(target,radius);shift=dyadic_shift(radius);grid=item["grids"].get(shift)
            if grid is None:raise ValueError("Pilot requires original supported grid; CPU-only target separately unsupported")
            sorted_points,ids,keys,offsets=grid
            forecast=int(adapter.cache_bytes+12*len(points)+len(queries)*(24+40+12))
            if forecast>CAP:raise ValueError("Bounded grid+shadow+query+output+diagnostic arrays")
            with cp.cuda.Device(0),cp.cuda.Stream.null:
                shadow=cp.empty(sorted_points.shape,dtype=cp.float32);kernels["shadow"](((3*len(points)+255)//256,),(256,),(sorted_points,np.uint32(3*len(points)),shadow))
                gpu_query=cp.asarray(queries);output=cp.empty((len(queries),5),dtype=cp.float64);diagnostic=cp.empty((len(queries),3),dtype=cp.uint32)
                cp.cuda.Stream.null.synchronize();host_sorted=cp.asnumpy(sorted_points);host_shadow=cp.asnumpy(shadow)
            target_max=float(np.max(np.abs(points)))
            bound_begin=time.perf_counter();certified_error=error_bound(target_max,QUERY_CAP,int(shift));bound_wall=time.perf_counter()-bound_begin
            # Each cached shadow coordinate must satisfy the cast+FTZ envelope.
            cast_limit=np.nextafter(np.abs(host_sorted)*(2.**-24)+2.**-126,np.inf)
            if not np.isfinite(host_shadow).all() or np.any(np.abs(host_shadow.astype(np.float64)-host_sorted)>cast_limit):raise ValueError("Shadow not a valid RN/FTZ conversion enclosure")
            shadow_before=descriptor(host_shadow);sorted_before=descriptor(host_sorted)
            row={"label":label,"target_rows":len(points),"query_rows":len(queries),"radius_m":radius,"target_max_abs":target_max,
                "bound_declined_deliberately":forced is not None,"certified_error":certified_error,
                "host_bound_setup_wall_s":bound_wall,"query_cap":QUERY_CAP,
                "actual_query_max_abs":float(np.max(np.abs(queries))),"input_arrays":before,"sorted_points":sorted_before,"float_shadow":shadow_before,
                "numeric_owned_bytes":forecast,"grid_shadow_query_setup_wall_s":time.perf_counter()-begun,"samples":[]}
            report["cases"].append(row);reference=None
            with cp.cuda.Device(0),cp.cuda.Stream.null:events=(cp.cuda.Event(),cp.cuda.Event())
            common=(sorted_points,ids,keys,offsets,np.uint32(len(keys)),gpu_query,np.uint32(len(queries)),np.int32(shift),output)
            filtered=(sorted_points,shadow,ids,keys,offsets,np.uint32(len(keys)),gpu_query,np.uint32(len(queries)),np.int32(shift),np.float64(QUERY_CAP),np.float64(certified_error if forced is None else forced),output,diagnostic)
            for repeat in range(args.repeats+1):
                values={}
                order=("exhaustive","warp-exhaustive","filtered");offset=repeat%3;order=order[offset:]+order[:offset]
                for name in order:
                    with cp.cuda.Device(0),cp.cuda.Stream.null:
                        begun=time.perf_counter();events[0].record();kernels[name]((len(queries) if name=="exhaustive" else (len(queries)+3)//4,),(128,),filtered if name=="filtered" else common);events[1].record();cp.cuda.Stream.null.synchronize()
                        wall=time.perf_counter()-begun;event=float(cp.cuda.get_elapsed_time(*events))/1000.
                        copy_begin=time.perf_counter();value=cp.asnumpy(output);counts=cp.asnumpy(diagnostic) if name=="filtered" else None;copy_wall=time.perf_counter()-copy_begin
                    values[name]=value
                    if reference is None and name=="exhaustive":reference=value.copy()
                    sample={"variant":name,"variant_order":list(order),"repeat":repeat,"warmup":repeat==0,"kernel_enqueue_event_sync_wall_s":wall,"gpu_event_s":event,
                        "output_diagnostic_copy_wall_s":copy_wall,"original_candidate_visits":int(value[:,4].sum())}
                    if counts is not None:
                        if (np.any(counts[:,0]>value[:,4]) or np.any(counts[:,1]>2*value[:,4]) or np.any(counts[:,2]>1)):
                            raise ValueError("Malformed screening counts")
                        sample.update(exact_double_evaluations=int(counts[:,0].sum()),float_evaluations=int(counts[:,1].sum()),screened_queries=int(counts[:,2].sum()))
                        if label=="outside-host-cap" and (np.any(counts[:,2]) or not np.array_equal(counts[:,0].astype(np.float64),value[:,4])):
                            raise ValueError("Supported out-of-cap queries must execute full original exact scan")
                    row["samples"].append(sample)
                for name,value in values.items():
                    if not np.array_equal(value.copy().view(np.uint64),reference.copy().view(np.uint64)):raise RuntimeError(label+": original five output columns changed")
            begun=time.perf_counter()
            adapter.statistics["query_rows"]+=len(queries)
            adapter.statistics["candidate_visits"]+=int(reference[:,4].sum())
            adapter._resolve(queries,item,radius,reference[:,0].astype(np.int32),reference[:,2],reference[:,3])
            with cp.cuda.Device(0),cp.cuda.Stream.null:
                shadow_after=descriptor(cp.asnumpy(shadow));sorted_after=descriptor(cp.asnumpy(sorted_points));cp.cuda.Stream.null.synchronize()
            row.update(cpu_audit_wall_s=time.perf_counter()-begun,all_five_raw_uint64_exact=True,original_cpu_hit_miss_ambiguity_audit=True,
                input_bytes_unchanged=before=={"points":descriptor(points),"queries":descriptor(queries)} and descriptor(np.asarray(target.points))==before["points"],
                float_shadow_unchanged=shadow_after==shadow_before,sorted_double_points_unchanged=sorted_after==sorted_before)
            if not all(row[k] for k in ("input_bytes_unchanged","float_shadow_unchanged","sorted_double_points_unchanged")):raise RuntimeError("Immutable case owner changed")
            save();print(label+": exact raw5 and fresh original CPU audit passed",flush=True)
            with cp.cuda.Device(0),cp.cuda.Stream.null:
                cp.cuda.Stream.null.synchronize()
                adapter.clear_cache()
                # Argument tuples retain numeric pointers; release them too,
                # only after completion, before the next case budget preflight.
                del common,filtered,shadow,gpu_query,output,diagnostic,sorted_points,ids,keys,offsets,grid,item,target
        report["cpu_audit_statistics"]=dict(adapter.statistics)
        stats=report["cpu_audit_statistics"]
        if not (stats["direct_gpu_hits"]==stats["audited_hits"]>0
                and stats["declared_gpu_misses"]==stats["audited_misses"]>0
                and stats["audit_index_mismatches"]==stats["audit_false_misses"]==0
                and stats["uncertain_queries"]>0 and stats["unsupported_queries"]>0):
            raise RuntimeError("Nonvacuous original CPU hit/miss/tie/support coverage required")
        report["source_sha256_after"]=source_hash();report["artifacts_sha256_after"]={name:sha(ROOT/name) for name in FILES}
        report["fixed_files_sha256_after"]={name:sha(name) for name in fixed};report["binaries_sha256_after"]={name:sha(name) for name in binaries};report["runtime_after"]=runtime()
        report["loaded_owners_unchanged"]=all(sys.modules.get(name) is module and str(Path(module.__file__).resolve())==path for name,(module,path) in modules.items())
        if not (report["source_sha256_after"]==source_before and report["artifacts_sha256_after"]==artifacts and report["fixed_files_sha256_after"]==fixed
                and report["binaries_sha256_after"]==binaries and report["runtime_after"]==report["runtime"] and report["loaded_owners_unchanged"]):raise RuntimeError("Actual source/runtime/input/binary closure changed")
    except BaseException as error:
        primary=error;report["failure"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
    finally:
        cleanup_begin=time.perf_counter()
        def clean(label,action):
            try:action()
            except BaseException as error:report["cleanup_failures"].append({"action":label,"type":type(error).__name__,"message":str(error)})
        if cp is not None:
            def finish_stream():
                with cp.cuda.Device(0),cp.cuda.Stream.null:cp.cuda.Stream.null.synchronize()
            clean("selected stream completion",finish_stream)
        if adapter is not None and not report["cleanup_failures"]:clean("original cache release",adapter.close)
        if threads is not None:
            o3d,previous=threads;clean("thread restore",lambda:o3d.utility.set_max_threads(previous))
            try:
                if o3d.utility.get_max_threads()!=previous:raise RuntimeError("Thread values differ")
            except BaseException as error:report["cleanup_failures"].append({"action":"thread verification","message":repr(error)})
        clean("OMP restore",lambda:os.environ.pop("OMP_NUM_THREADS",None) if old_omp is None else os.environ.__setitem__("OMP_NUM_THREADS",old_omp))
        report["owner_cleanup_wall_s"]=time.perf_counter()-cleanup_begin;report["cleanup_passed"]=not report["cleanup_failures"] and os.environ.get("OMP_NUM_THREADS")==old_omp
        report["status"]="passed" if primary is None and report["cleanup_passed"] else "failed"
        report["end_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
        try:save()
        except BaseException as error:
            if primary is not None:raise primary from error
            raise
    if report["status"]!="passed":raise RuntimeError("Filtered NN pilot failed; report retained") from primary
    from scripts.process_metrics import finish_cuda_worker
    finish_cuda_worker()


if __name__=="__main__":main()
