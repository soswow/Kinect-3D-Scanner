"""Sampled same-kernel HOST diagnostic on/off ablation; no full NN authority.

Run only after exclusive hardware allocation. Reuse the closed original-query
trace and its original CPU gold, capture/source/raw/runtime proof guards. Both
serial and parallel staged shaders return identical seven-column output in both
modes. Default CPU missing/tie/support fallback remains active. This measures
the adapter including transfers/fallback, plus the nested host decode timer;
it makes no whole bridge or unmeasured full-query improvement claim.

The original five-kernel driver is imported only after main's allocation guard.
Its existing trace/provenance validation stays intact; this separate wrapper
binds its own helper/adapter/validator bytes and preserves failure diagnostics.
Use a closed trace made by the EXACT current five-kernel harness bytes (currently
grid-lookup-ablation-order-v2/real.npz); the older v1 trace is incompatible with
the updated producer. This wrapper does not relax the producer-hash guard.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.tool_paths import tool_path

import argparse
import hashlib
import json
import time
import traceback



def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def benchmark_host(args,report,runtime,manifest):
    np,o3d,cv2,cp=runtime
    from scripts.research.archive import benchmark_grid_lookup_ablation as shared
    from scripts.research.archive.cuda_parallel_staged_grid_registration import ParallelStagedUniformGridICP
    cases=[]
    with np.load(args.trace,allow_pickle=False) as archive:
        for row in manifest["batches"]:
            arrays={name:archive[f"batch{row['index']}_{name}"] for name in row["arrays"]}
            for name,value in arrays.items():
                shared.require(shared.array_binding(np,value)==row["arrays"][name],"Trace arrays changed")
            shared.require(arrays["queries"].dtype==np.float64 and arrays["queries"].shape==(row["query_count"],3)
                and 1<=row["query_count"]<=2048,"Bounded original sampled queries required")
            cases.append((row,arrays))
    retained=sum(array.nbytes for _,arrays in cases for array in arrays.values())
    shared.require(retained==manifest["retained_array_bytes"] and retained<=128*1024**2 and 1<=len(cases)<=12,
        "Original trace extent/budget changed")
    variants=[]
    solvers={}
    report["host_diagnostic_measurements"]=[]
    report["variant_setup"]=[]
    try:
        for family,shader,entry in (("serial_staged","research_staged_grid_nn.cu","staged_grid_nearest_two"),
                ("parallel_staged","research_parallel_staged_grid_nn.cu","parallel_staged_grid_nearest_two")):
            for collect in (True,False):
                name=family+("_collect" if collect else "_skip")
                started=time.perf_counter()
                solver=ParallelStagedUniformGridICP(max_clouds=12,max_cache_bytes=args.cache_mib*1024**2,
                    miss_policy="cpu-fallback",record_stage_diagnostics=collect)
                solvers[name]=solver
                with cp.cuda.Device(0),cp.cuda.Stream.null:
                    solver.kernel=cp.RawKernel((tool_path(shader)).read_text(),entry,options=("--std=c++11","--fmad=false"))
                    solver.kernel.compile()
                report["variant_setup"].append({"variant":name,"setup_compile_s":time.perf_counter()-started,
                    "shader_sha256":file_hash(tool_path(shader)),"record_stage_diagnostics":collect})
                variants.append((name,family,collect))
        for row,arrays in cases:
            queries=arrays["queries"]
            before={name:shared.array_binding(np,array) for name,array in arrays.items()}
            target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arrays["target_points"]))
            target.normals=o3d.utility.Vector3dVector(arrays["target_normals"])
            tree=o3d.geometry.KDTreeFlann(target)
            gold=shared.cpu_ids(np,tree,queries,row["radius_m"])
            shared.require(np.array_equal(gold,arrays["gold_ids"]),"Fresh CPU gold differs from trace")
            items={}
            raw_by_family={}
            for name,family,collect in variants:
                solver=solvers[name]
                started=time.perf_counter()
                item=solver._dataset(target,row["radius_m"])
                with cp.cuda.Device(0),cp.cuda.Stream.null:
                    cp.cuda.Stream.null.synchronize()
                    shared.require(item["tables"].get(row["radius_m"]) is not None,"Actual GPU table required")
                    device=cp.asarray(queries)
                    raw=cp.asnumpy(solver._raw_device(device,item,row["radius_m"]))
                bits=np.ascontiguousarray(raw).view(np.uint64).tobytes()
                if collect:raw_by_family[family]=bits
                shared.require(raw_by_family[family]==bits,"Same-kernel HOST toggle changed raw output bits")
                shared.check_output(np,solver,item,row["radius_m"],queries,gold,raw)
                items[name]=(item,time.perf_counter()-started)
                # Warm host adapter separately, preserving default CPU fallback.
                shared.require(np.array_equal(solver.nearest_indices(queries,item,row["radius_m"]),gold),"Warm adapter changed gold")
            samples={name:[] for name,_,_ in variants}
            # Reverse on/off order every repetition to reduce drift confounding.
            for repeat in range(args.repeats):
                order=variants if repeat%2==0 else list(reversed(variants))
                for name,family,collect in order:
                    solver=solvers[name]
                    item,_=items[name]
                    previous=dict(solver.statistics)
                    started=time.perf_counter()
                    ids=solver.nearest_indices(queries,item,row["radius_m"])
                    elapsed=time.perf_counter()-started
                    shared.require(np.array_equal(ids,gold),"HOST diagnostic mode changed sampled original CPU IDs")
                    delta={key:solver.statistics[key]-previous[key] for key in solver.statistics}
                    shared.require(delta["query_rows"]==len(queries) and delta["audited_hits"]==delta["audited_misses"]==0,
                        "Timing query/audit accounting changed")
                    record={"repeat":repeat,"variant_order":[item[0] for item in order],"adapter_wall_s":elapsed,
                        "gpu_search_transfer_s":delta["gpu_search_transfer_s"],"cpu_fallback_s":delta["cpu_fallback_s"],
                        "empty_cpu_audit_loop_s":delta["cpu_audit_s"],"host_stage_decode_s":delta["host_stage_diagnostics_s"],
                        "query_rows":delta["query_rows"],"candidate_visits":delta["candidate_visits"],
                        "cpu_fallback_queries":delta["exact_cpu_queries"],"original_cpu_ids_equal":True,
                        "stage_work":"collected" if collect else "uncollected"}
                    if collect:
                        record["stage_counters"]={key:value for key,value in delta.items() if key.startswith("stage")
                            or key in ("pruned_candidates","double_evaluations")}
                    else:
                        shared.require(not any(key.startswith("stage") or key in ("pruned_candidates","double_evaluations") for key in delta),
                            "Uncollected stage work must be omitted")
                    samples[name].append(record)
            for name,family,collect in variants:
                report["host_diagnostic_measurements"].append({"variant":name,"family":family,
                    "record_stage_diagnostics":collect,"trace_batch":row["index"],"pair_position":row["pair_position"],
                    "scope":row["scope"],"radius_m":row["radius_m"],"query_count":len(queries),
                    "original_full_query_count":row["source_query_count"],"target_count":len(target.points),
                    "same_kernel_raw_output_bits_equal":True,"setup_warm_and_raw_quality_wall_s":items[name][1],"samples":samples[name]})
            shared.require(before=={name:shared.array_binding(np,array) for name,array in arrays.items()},"Sample arrays mutated")
            shared.save(args.output,report)
            print(f"Host diagnostics batch{row['index']}: collect/skip sampled gold and raw bits pass",flush=True)
        report["input_arrays_unchanged"]=True
        report["host_diagnostic_totals"]={name:{
            "adapter_wall_s":sum(sample["adapter_wall_s"] for row in report["host_diagnostic_measurements"] if row["variant"]==name for sample in row["samples"]),
            "host_stage_decode_s":sum(sample["host_stage_decode_s"] for row in report["host_diagnostic_measurements"] if row["variant"]==name for sample in row["samples"]),
            "gpu_search_transfer_s":sum(sample["gpu_search_transfer_s"] for row in report["host_diagnostic_measurements"] if row["variant"]==name for sample in row["samples"]),
            "cpu_fallback_s":sum(sample["cpu_fallback_s"] for row in report["host_diagnostic_measurements"] if row["variant"]==name for sample in row["samples"]),
            "stage_work":"collected" if collect else "uncollected"} for name,_,collect in variants}
        report["scope_limits"]="Same seven-column staged kernel per on/off pair; transfers, host decode and original CPU fallback included in adapter wall. Decode/search/fallback timers are nested and cannot be added again to adapter wall. Alternating variant order; bounded sampled original queries only, no full-bridge/direct-miss authority or linear full-query extrapolation."
    finally:
        primary=sys.exception();failures=[]
        for name,solver in solvers.items():
            try:solver.close()
            except BaseException as error:failures.append({"variant":name,"type":type(error).__name__,"message":str(error)})
        report["solver_cleanup_failures"]=failures
        if failures:
            if primary is not None:primary.add_note(f"Secondary host diagnostic cleanup failed: {failures}")
            else:raise RuntimeError("Host diagnostic owned cleanup failed")


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode",choices=("benchmark",))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-allocated",action="store_true")
    args,_=parser.parse_known_args()
    if not args.run_allocated or args.output.exists():parser.error("Require explicit allocation and fresh report path")
    paths=[Path(__file__),ROOT/"scripts/tool_paths.py",ROOT/"scripts/tool-catalog.json",ROOT/"scripts/research/archive/cuda_parallel_staged_grid_registration.py",
        ROOT/"scripts/research/archive/validate_parallel_staged_grid_proof.py",ROOT/"scripts/research/archive/benchmark_grid_lookup_ablation.py",
        ROOT/"scripts/research/archive/research_parallel_staged_grid_nn.cu",ROOT/"scripts/research/archive/research_staged_grid_nn.cu"]
    before={str(path.relative_to(ROOT)):file_hash(path) for path in paths}
    from scripts.research.archive import benchmark_grid_lookup_ablation as harness
    from scripts import process_metrics
    original_benchmark,original_finish=harness.benchmark_trace,process_metrics.finish_cuda_worker
    def finalize():
        report=json.loads(args.output.read_text())
        after={str(path.relative_to(ROOT)):file_hash(path) for path in paths}
        report["kind"]="bounded-original-query-host-stage-diagnostics-ablation"
        report["host_diagnostic_helper"]={"artifacts_sha256":before,"artifacts_sha256_after":after,"unchanged":before==after,
            "scope":"Replace sampled lookup loop only; unchanged closed trace/source/raw/runtime/proof guards retained; no full new-shader authority"}
        if before!=after:report["status"]="failed"
        harness.save(args.output,report)
        return before==after
    def guarded_finish():
        if not finalize():raise RuntimeError("Host diagnostic helper provenance changed")
        original_finish()
    harness.benchmark_trace=benchmark_host
    process_metrics.finish_cuda_worker=guarded_finish
    try:harness.main()
    except BaseException as error:
        primary={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        secondary=[];report={"kind":"host-diagnostic-helper-early-failure"}
        if args.output.exists():
            try:finalize();report=json.loads(args.output.read_text())
            except BaseException as cleanup_error:
                secondary.append(str(cleanup_error))
                try:
                    backup=args.output.with_name(args.output.name+".partial-"+str(time.time_ns()))
                    backup.write_bytes(args.output.read_bytes())
                    report["partial_report_preserved_at"]=str(backup)
                except BaseException as backup_error:secondary.append("Partial report preservation failed: "+str(backup_error))
        report.update(status="failed",host_diagnostic_current_failure=primary,host_diagnostic_secondary_errors=secondary)
        try:harness.save(args.output,report)
        except BaseException as write_error:raise error.with_traceback(error.__traceback__) from write_error
        raise
    finally:
        harness.benchmark_trace=original_benchmark
        process_metrics.finish_cuda_worker=original_finish


if __name__=="__main__":main()
