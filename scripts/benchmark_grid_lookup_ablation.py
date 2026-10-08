"""Bounded actual-query kernel ablation; never a whole-component speed claim.

Run only in an explicitly allocated hardware slot. ``capture`` executes the
already-proved full-radius bridge on all nine original competing proposals,
then stores at most twelve stratified FP64 query batches and unrounded seeds.
``benchmark`` reuses those immutable arrays for causal shader comparisons.
Both modes bind current raw/fixture/proof/runtime/source bytes. New shaders
have no full-bridge authority: every sampled direct hit/miss is checked against
gold and fresh original CPU nearest IDs before any ranking is reported.

Capture example (supply the actual separate full-radius proof paths):
  python scripts/benchmark_grid_lookup_ablation.py capture --run-allocated \
    --trace benchmark-output/cuda-pipeline/grid-ablation/real.npz \
    --output benchmark-output/cuda-pipeline/grid-ablation/capture.json \
    --proof-synthetic SYNTHETIC.json --proof-bridge BRIDGE.json
Benchmark uses the same proof flags, trace, and a distinct fresh --output.
Optional --include-flat adds a reviewed fifth full-radius iterator variant.
Numerical imports, compilation and device work are deferred until main().
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_binding(np, array):
    array = np.asarray(array)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.nbytes,
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def cpu_ids(np, tree, queries, radius):
    result = np.empty(len(queries), np.int32)
    for row, point in enumerate(queries):
        count, indices, _ = tree.search_hybrid_vector_3d(point, radius, 1)
        result[row] = indices[0] if count else -1
    return result


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")


class TraceCapture:
    def __init__(self, np, recorder, args):
        self.np, self.recorder, self.args = np, recorder, args
        self.arrays, self.rows, self.seen = {}, [], set()
        self.retained_bytes = 0
        self.capture_s = self.cpu_audit_s = 0.
        self.match_context = None

    def consider(self, queries, item, radius, gold):
        np, rec = self.np, self.recorder
        scope = next((row.get("cloud_scope") for row in reversed(rec.stack)
                      if row["name"] == "match"), None)
        require(scope in ("camera", "fragment_union"), "Real batch lacks registered camera/union provenance")
        key = (rec.pair_position, scope, float(radius))
        if key in self.seen or len(self.rows) >= self.args.max_batches:
            return
        started = time.perf_counter()
        target = item["target"]
        points, normals = np.asarray(target.points), np.asarray(target.normals)
        require(queries.dtype == np.float64 and points.dtype == np.float64 and normals.dtype == np.float64
                and points.shape == normals.shape and np.isfinite(queries).all(), "Original FP64 geometry required")
        # Evenly select original rows, preserving their exact coordinate bytes.
        selected = np.unique(np.linspace(0, len(queries)-1, min(len(queries), self.args.queries), dtype=np.int64))
        subset, selected_gold = np.ascontiguousarray(queries[selected]), np.asarray(gold[selected], np.int32).copy()
        arrays = {"queries": subset, "target_points": points.copy(), "target_normals": normals.copy(),
                  "gold_ids": selected_gold, "source_rows": selected}
        needed = sum(value.nbytes for value in arrays.values())
        require(self.retained_bytes+needed <= self.args.trace_mib*1024**2, "Trace byte budget exceeded; preserve partial diagnostics")
        audit_started = time.perf_counter()
        expected = cpu_ids(np, item["cpu"], subset, radius)
        self.cpu_audit_s += time.perf_counter()-audit_started
        require(np.array_equal(expected, selected_gold), "Captured proved-grid IDs differ from fresh original CPU IDs")
        index = len(self.rows)
        self.arrays.update({f"batch{index}_{name}": value for name, value in arrays.items()})
        self.rows.append({"index": index, "pair_position": rec.pair_position,
            "proposal_index": rec.proposal_index, "scope": scope, "branch_path": rec.scope(),
            "radius_m": float(radius), "icp_radius_stage": { .12:"coarse", .06:"middle", .03:"fine" }.get(float(radius)),
            "source_query_count": len(queries), "query_count": len(subset),
            "complete_queries_binding": array_binding(np, queries), "match_context": self.match_context,
            "arrays": {name: array_binding(np, value) for name, value in arrays.items()},
            "original_cpu_gold_checked": True})
        self.retained_bytes += needed
        self.seen.add(key)
        self.capture_s += time.perf_counter()-started


def capture_trace(args, report, runtime, fixture, tasks, authority, bridge, binding):
    np, o3d, cv2, cp = runtime
    from scanner_server import fragments as module, refinement, appearance
    from scripts import benchmark_parallel_fragments as baseline, profile_fragment_verification as profiler
    from scripts.benchmark_selective_fragment_threads import fresh_fragments
    from scripts.benchmark_optix_nn import evidence_agreement, canonical_evidence, pair_verdict
    from scripts.cuda_uniform_grid_registration import UniformGridICP
    rec = profiler.Recorder()
    packed = fresh_fragments(fixture, baseline, module)
    for index, fragment in packed.items():
        for name in ("train", "heldout"):
            rec.register(getattr(fragment, name), f"fragment:{index}:union:{name}", kind=name)
        for view in fragment.keys:
            for name in ("train", "heldout"):
                rec.register(getattr(view, name), f"fragment:{index}:view:{view.index}:{name}", kind=name)
    collector = TraceCapture(np, rec, args)
    class CapturingGrid(UniformGridICP):
        def match(self, source, target, initial):
            collector.match_context = {"source": rec.cloud(source), "target": rec.cloud(target),
                "source_points": array_binding(np, np.asarray(source.points)),
                "source_normals": array_binding(np, np.asarray(source.normals)),
                "initial_fp64": array_binding(np, np.asarray(initial)),
                "initial_unrounded_bytes_hex": np.asarray(initial).tobytes().hex()}
            return super().match(source, target, initial)
        def nearest_indices(self, queries, item, radius):
            before = array_binding(np, queries)
            result = super().nearest_indices(queries, item, radius)
            collector.consider(queries, item, radius, result)
            require(before == array_binding(np, queries), "Original lookup queries mutated during capture")
            return result
    solver = CapturingGrid(max_clouds=64, max_cache_bytes=256*1024**2,
                           miss_policy="direct-miss-research-v1", proof_authority=authority)
    original = module._match
    before = (profiler.array_digest(fixture), profiler.cloud_digest(packed))
    cpu_run = next(run for run in bridge["real_runs"] if run["mode"] == "native_cpu")
    report["capture_proposals"] = []
    try:
        module._match = solver.match
        started = time.perf_counter()
        with profiler.instrumentation(rec, module, refinement, appearance, cv2):
            rec.phase = "fixed_verification"
            for task, gold_pair in zip(tasks, cpu_run["pairs"]):
                rec.pair_position = task["position"]
                values = []
                for index, initial in enumerate(task["proposals"]):
                    rec.proposal_index = index
                    value = module._verify_bridge(packed[task["pair"][0]], packed[task["pair"][1]], initial, fixture["camera"])
                    quality = evidence_agreement(gold_pair["proposal_results"][index]["evidence"], value)
                    report["capture_proposals"].append({"position": task["position"], "proposal_index": index,
                        "input_sha256": fileless_hash(initial.tobytes()), "quality": quality,
                        "evidence": canonical_evidence(value)})
                    require(quality["passed"], "Capture changed original independent proposal evidence")
                    values.append(value)
                verdict = pair_verdict(values)
                require(all(verdict[key] == gold_pair["verdict"][key] for key in
                            ("accepted", "ambiguous", "verified_proposals")), "Capture changed ordered competing-proposal verdict")
                print(f"Captured original pair{task['pair']} with{len(collector.rows)} bounded batches", flush=True)
        report["capture_bridge_wall_s"] = time.perf_counter()-started
        require(before == (profiler.array_digest(fixture), profiler.cloud_digest(packed)), "Original fixture/cloud arrays changed")
        require(len(report["capture_proposals"]) == 9, "Every original competing proposal must execute")
        require({row["pair_position"] for row in collector.rows} == {0,4}, "Accepted/rejected query coverage absent")
        require({row["radius_m"] for row in collector.rows} == {.12,.06,.03}, "Coarse/middle/fine radius coverage absent")
        require({row["scope"] for row in collector.rows} == {"camera","fragment_union"}, "Camera/union query coverage absent")
        report["capture_scope_counts"] = {f"{position}/{scope}/{radius}": sum(
            row["pair_position"] == position and row["scope"] == scope and row["radius_m"] == radius
            for row in collector.rows) for position in (0,4) for scope in ("camera","fragment_union") for radius in (.12,.06,.03)}
        report.update(trace_batches=collector.rows, retained_array_bytes=collector.retained_bytes,
            capture_observer_wall_s=collector.capture_s, sampled_original_cpu_audit_s=collector.cpu_audit_s,
            fixture_and_cloud_arrays_unchanged=True, all_nine_original_gates_passed=True)
        args.trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez(args.trace, **collector.arrays)
        manifest = {"kind":"bounded-original-real-grid-query-trace", "status":"pending_final_guards", "producer_sha256":report["script_sha256"],
            "source_sha256":FROZEN, "full_radius_proof_binding":binding, "fixture_binding":report["fixture_binding"],
            "input_binding":report["input_binding"], "proof_report_hashes":report["proof_report_hashes"],
            "trace_sha256":file_hash(args.trace), "batches":collector.rows,
            "retained_array_bytes":collector.retained_bytes, "limits":report["limits"],
            "reverse_order":report["reverse_order"],
            "capture_authority":"All nine original gates and fresh originalCPU nearest IDs on retained real rows; no new shader authority",
            "capture_report":str(args.output.resolve())}
        save(args.trace.with_suffix(".json"), manifest)
        report["trace_sha256"] = manifest["trace_sha256"]
        report["trace_manifest_preclosure_sha256"] = file_hash(args.trace.with_suffix(".json"))
    finally:
        module._match = original
        primary=sys.exception()
        try:solver.close()
        except BaseException as cleanup_error:
            report["capture_cleanup_failure"]={"type":type(cleanup_error).__name__,"message":str(cleanup_error)}
            if primary is not None:primary.add_note(f"Capture cleanup also failed: {cleanup_error}")
            else:raise


def fileless_hash(payload):
    return hashlib.sha256(payload).hexdigest()


def check_output(np, solver, item, radius, queries, gold, output):
    require(output.shape in ((len(queries),5),(len(queries),7)),"Unexpected raw shader output shape")
    require(np.isfinite(output[:,0:2]).all() and np.all(output[:,0:2]==np.floor(output[:,0:2]))
            and np.all((output[:,0]==-3)|(output[:,0]==-1)|((output[:,0]>=0)&(output[:,0]<len(item["target"].points)))),
            "Invalid original-index shader output")
    second=output[:,1].astype(np.int32)
    target_n=len(item["target"].points)
    require(np.all((second==-1)|((second>=0)&(second<target_n)))
            and np.all((second<0)|(second!=output[:,0])),"Invalid/distinct-second shader index")
    require(np.all((np.isfinite(output[:,2:4])&(output[:,2:4]>=0))|np.isposinf(output[:,2:4]))
            and np.all(output[:,2]<=output[:,3]),"Invalid/nonordered original FP64 distance minima")
    first, a, b = output[:,0].astype(np.int32), output[:,2], output[:,3]
    r2 = radius*radius
    with np.errstate(invalid="ignore"):
        margin = 64*np.finfo(np.float64).eps*np.maximum(np.maximum(np.abs(a),abs(r2)),np.finfo(np.float64).tiny)
        boundary = np.isfinite(a) & (np.abs(a-r2) <= margin)
        tie = np.isfinite(a) & np.isfinite(b) & (b-a <= 64*np.finfo(np.float64).eps*np.maximum(np.maximum(np.abs(a),np.abs(b)),abs(r2)))
    direct = (first != -3) & ~boundary & ~tie
    raw_ids = np.where(np.isfinite(a) & (a < r2), first, -1)
    require(np.array_equal(raw_ids[direct],gold[direct]), "Sampled direct GPU hit/miss differs from original CPU/gold")
    started = time.perf_counter()
    resolved = solver._resolve(queries,item,radius,first,a,b)
    resolve_s = time.perf_counter()-started
    require(np.array_equal(resolved,gold), "Resolved sampled GPU nearest IDs differ from original CPU/gold")
    visits = output[:,4]
    require(np.isfinite(visits).all() and np.all(visits>=0) and np.all(visits==np.floor(visits))
            and np.all(visits<=target_n*(3 if output.shape[1]==7 else 1)),"Invalid bounded candidate counters")
    if output.shape[1] == 7:
        packed, pruned = output[:,5].view(np.uint64), output[:,6].view(np.uint64)
        total = np.zeros(len(output),np.uint64)
        for stage in range(3):
            mask = np.uint64((1<<20)-1)
            seen = (packed >> np.uint64(20*stage)) & mask
            skipped = (pruned >> np.uint64(20*stage)) & mask
            require(np.all(skipped<=seen), "Packed stage prune counter exceeds visited count")
            total += seen
        require(np.array_equal(total,visits.astype(np.uint64)), "Packed stage counts differ from visit total")
    return {"passed":True,"direct_hits":int(np.count_nonzero(direct & (raw_ids>=0))),
            "direct_misses":int(np.count_nonzero(direct & (raw_ids<0))),
            "cpu_uncertain_or_unsupported":int(np.count_nonzero(~direct)),
            "candidate_visits":int(visits.sum()),"cpu_resolution_s":resolve_s}


def benchmark_trace(args, report, runtime, manifest):
    np,o3d,cv2,cp = runtime
    from scripts.cuda_uniform_grid_registration import UniformGridICP
    from scripts.cuda_staged_grid_registration import StagedUniformGridICP
    from scripts.benchmark_optix_nn import gpu_memory_snapshot
    variants = [("serial_full",UniformGridICP,"research_uniform_grid_nn.cu","grid_nearest_two"),
        ("parallel_full",UniformGridICP,"research_parallel_grid_nn.cu","parallel_grid_nearest_two"),
        ("serial_staged",StagedUniformGridICP,"research_staged_grid_nn.cu","staged_grid_nearest_two"),
        ("parallel_staged",StagedUniformGridICP,"research_parallel_staged_grid_nn.cu","parallel_staged_grid_nearest_two")]
    if args.include_flat:
        variants.append(("flat_full",UniformGridICP,"research_flat_grid_nn.cu","flat_grid_nearest_two"))
    if args.reverse_order:
        variants.reverse()
    report["variant_order"]=[name for name,_,_,_ in variants]
    report["raw_family_reference_variants"]={}
    require(args.trace.stat().st_size <= manifest["retained_array_bytes"]+1024**2, "Trace archive extent exceeds declared bound")
    data = np.load(args.trace,allow_pickle=False)
    cases = []
    retained = 0
    for row in manifest["batches"]:
        arrays = {name: data[f"batch{row['index']}_{name}"] for name in row["arrays"]}
        for name,value in arrays.items():
            require(array_binding(np,value)==row["arrays"][name], "Trace array bytes/dtype/shape changed")
            retained += value.nbytes
        require(arrays["queries"].dtype==np.float64 and arrays["queries"].shape==(row["query_count"],3)
                and row["query_count"]<=2048 and arrays["target_points"].dtype==np.float64
                and arrays["target_points"].shape==arrays["target_normals"].shape
                and arrays["target_points"].shape[1:]==(3,), "Original bounded FP64 trace geometry required")
        cases.append((row,arrays))
    data.close()
    require(retained==manifest["retained_array_bytes"] and retained<=128*1024**2 and 1<=len(cases)<=12, "Trace retained-byte/case bound differs")
    # Focused fresh original-CPU cases supplement sampled real parity.
    tiny = np.nextafter(np.float64(0.),np.float64(1.))
    base = np.nextafter(np.float64(32768.),np.float64(0.))
    focused = [([ [0.,0.,0.],[0.,0.,0.],[.03,0.,0.] ],[[0.,0.,0.],[.015,0.,0.],[.06,0.,0.]],.03,"duplicates-ties-boundary"),
        ([[-tiny,0.,0.],[tiny,0.,0.]],[[0.,0.,0.],[tiny,0.,0.]],2.**-20,"minimum-radius-subnormal"),
        ([[base,0.,0.],[base-.01,0.,0.]],[[base+.01,0.,0.],[base+.1,0.,0.],[base+.2,0.,0.]],.12,"unsupported-inner-supported-full"),
        ([[1e7,0.,0.]],[[1e7,0.,0.],[0.,0.,0.]],.12,"unsupported-original-domain")]
    for points,queries,radius,label in focused:
        points,queries = np.array(points,np.float64),np.array(queries,np.float64)
        cases.append(({"index":label,"radius_m":radius,"scope":"focused_original_cpu","query_count":len(queries)},
            {"target_points":points,"target_normals":np.zeros_like(points),"queries":queries,"gold_ids":None}))
    solvers = {}
    report["variant_setup"], report["measurements"] = [],[]
    try:
        for name,cls,shader,entry in variants:
            started = time.perf_counter()
            solver = cls(max_clouds=12,max_cache_bytes=args.cache_mib*1024**2,miss_policy="cpu-fallback")
            base_setup = time.perf_counter()-started
            solvers[name] = solver
            compile_s = 0.
            if name not in ("serial_full","serial_staged"):
                started = time.perf_counter()
                with cp.cuda.Device(0),cp.cuda.Stream.null:
                    solver.kernel = cp.RawKernel((ROOT/"scripts"/shader).read_text(),entry,options=("--std=c++11","--fmad=false"))
                    solver.kernel.compile()
                compile_s = time.perf_counter()-started
            report["variant_setup"].append({"variant":name,"base_adapter_setup_s":base_setup,
                "replacement_shader_compile_s":compile_s,"shader":shader,"shader_sha256":file_hash(ROOT/"scripts"/shader),
                "compile_scope":"Replacement variants also compile the baseline adapter kernel; this setup is separate from lookup timing"})
        for row,arrays in cases:
            points,normals,queries = arrays["target_points"],arrays["target_normals"],arrays["queries"]
            before = {key:array_binding(np,arrays[key]) for key in ("target_points","target_normals","queries")}
            target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            target.normals = o3d.utility.Vector3dVector(normals)
            tree = o3d.geometry.KDTreeFlann(target)
            started = time.perf_counter()
            gold = cpu_ids(np,tree,queries,row["radius_m"])
            cpu_audit_s = time.perf_counter()-started
            if arrays["gold_ids"] is not None:
                require(np.array_equal(gold,arrays["gold_ids"]), "Trace gold differs from current original CPU nearest IDs")
            raw_baselines={}
            for name,_,_,_ in variants:
                solver = solvers[name]
                started = time.perf_counter()
                item = solver._dataset(target,row["radius_m"])
                with cp.cuda.Device(0),cp.cuda.Stream.null:cp.cuda.Stream.null.synchronize()
                build_s = time.perf_counter()-started
                if row["scope"] != "focused_original_cpu":
                    from scripts.cuda_uniform_grid_registration import dyadic_shift
                    present = (item.get("tables",{}).get(row["radius_m"]) is not None if "tables" in item
                               else item["grids"].get(dyadic_shift(row["radius_m"])) is not None)
                    require(present,"Real microbenchmark requires an actual GPU dataset; CPU-only fallback cannot rank as fast")
                samples = []
                with cp.cuda.Device(0),cp.cuda.Stream.null:
                    started = time.perf_counter()
                    query_device = cp.asarray(np.ascontiguousarray(queries))
                    cp.cuda.Stream.null.synchronize()
                    h2d_s = time.perf_counter()-started
                    # One warm/correctness lookup is kept separate from repeats.
                    for repeat in range(args.repeats+1):
                        begin,end = cp.cuda.Event(),cp.cuda.Event()
                        started = time.perf_counter()
                        begin.record()
                        output_device = solver._raw_device(query_device,item,row["radius_m"])
                        end.record(); end.synchronize()
                        lookup_wall_s = time.perf_counter()-started
                        gpu_s = cp.cuda.get_elapsed_time(begin,end)/1000.
                        started = time.perf_counter()
                        output = cp.asnumpy(output_device)
                        d2h_s = time.perf_counter()-started
                        # Compare every byte, including second minima and raw
                        # packed words that may encode floating-point NaNs.
                        family="serial_staged" if output.shape[1]==7 else "serial_full"
                        raw_bytes=np.ascontiguousarray(output).view(np.uint64).tobytes()
                        if family not in raw_baselines:
                            raw_baselines[family]=raw_bytes
                            report["raw_family_reference_variants"][family]=name
                        require(raw_baselines.get(family)==raw_bytes,
                                f"{name} changed same-family raw IDs/minima/visits/stage payload bits")
                        quality = check_output(np,solver,item,row["radius_m"],queries,gold,output)
                        quality["same_family_raw_uint64_bits_equal"]=True
                        if row["scope"] != "focused_original_cpu":
                            require(quality["direct_hits"]+quality["direct_misses"]>0,"Real lookup lacks direct GPU coverage")
                        samples.append({"repeat":repeat,"warmup":repeat==0,"gpu_lookup_event_s":gpu_s,
                            "lookup_enqueue_and_sync_wall_s":lookup_wall_s,"output_d2h_wall_s":d2h_s,
                            "output_bytes":output.nbytes,"quality":quality})
                report["measurements"].append({"variant":name,"trace_batch":row["index"],"scope":row["scope"],
                    "pair_position":row.get("pair_position"),"proposal_index":row.get("proposal_index"),
                    "radius_m":row["radius_m"],"query_count":len(queries),"target_count":len(points),
                    "dataset_hash_tree_sort_upload_wall_s":build_s,"query_h2d_wall_s":h2d_s,
                    "fresh_original_cpu_gold_audit_s":cpu_audit_s,"samples":samples})
                save(args.output,report)
                print(f"{name} batch{row['index']}: exact sampled IDs pass",flush=True)
            require(before == {key:array_binding(np,arrays[key]) for key in before}, "Trace input arrays mutated by benchmark")
        require(len(report["measurements"])==len(cases)*len(variants)
                and {row["variant"] for row in report["measurements"]}==set(report["variant_order"]),
                "Every case must include every required variant and original serial family reference")
        report["input_arrays_unchanged"] = True
        report["gpu_memory_before_close"] = gpu_memory_snapshot(0)
    finally:
        primary=sys.exception()
        failures=[]
        for name,solver in solvers.items():
            try:
                solver.close()
            except BaseException as error:
                failures.append({"variant":name,"type":type(error).__name__,"message":str(error)})
        report["solver_cleanup_failures"] = failures
        if failures:
            if primary is not None:primary.add_note(f"Owned solver cleanup also failed: {failures}")
            else:raise RuntimeError("Owned solver cleanup failed; no performance conclusion")
    real = [row for row in report["measurements"] if row["scope"] != "focused_original_cpu"]
    report["real_sample_lookup_totals"] = {name:{
        "gpu_event_s":sum(sample["gpu_lookup_event_s"] for row in real if row["variant"]==name for sample in row["samples"] if not sample["warmup"]),
        "lookup_enqueue_and_sync_wall_s":sum(sample["lookup_enqueue_and_sync_wall_s"] for row in real if row["variant"]==name for sample in row["samples"] if not sample["warmup"]),
        "output_d2h_wall_s":sum(sample["output_d2h_wall_s"] for row in real if row["variant"]==name for sample in row["samples"] if not sample["warmup"]),
        "lookup_and_output_transfer_wall_s":sum(sample["lookup_enqueue_and_sync_wall_s"]+sample["output_d2h_wall_s"]
            for row in real if row["variant"]==name for sample in row["samples"] if not sample["warmup"])} for name,_,_,_ in variants}
    report["scope_limits"] = "Preuploaded bounded real rows with warmed kernels, one event-timed raw lookup/host output copy; CUDA event includes output allocation/enqueue gaps and is not pure shader instruction time. Recorded forward/reverse variant order. Exact raw-family equality is order independent and includes the original serial output. No whole component speedup or all-query proof. GPU events overlap lookup wall and cannot be added to it. Sorting/build/H2D/CPU resolution are separate; no ICP math or gate speed claim."


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("capture","benchmark"))
    parser.add_argument("--run-allocated",action="store_true")
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--proof-synthetic",type=Path,required=True)
    parser.add_argument("--proof-bridge",type=Path,required=True)
    parser.add_argument("--fixture",type=Path,default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle")
    parser.add_argument("--reference",type=Path,default=ROOT/"benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.json")
    parser.add_argument("--max-batches",type=int,default=12)
    parser.add_argument("--queries",type=int,default=2048)
    parser.add_argument("--trace-mib",type=int,default=128)
    parser.add_argument("--cache-mib",type=int,default=32)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--include-flat",action="store_true")
    parser.add_argument("--reverse-order",action="store_true")
    args=parser.parse_args()
    if not args.run_allocated or not (1<=args.max_batches<=12 and 1<=args.queries<=2048 and 1<=args.trace_mib<=128
                                    and 1<=args.cache_mib<=256 and 1<=args.repeats<=20):
        parser.error("Require allocated slot and bounded positive batch/query/trace/cache/repeat settings")
    if args.trace.suffix.lower()!=".npz" or args.output.exists():
        parser.error("Require .npz trace and fresh output report")
    if args.mode=="capture" and (args.trace.exists() or args.trace.with_suffix(".json").exists()):
        parser.error("Preserve existing trace; capture requires fresh paths")
    require(args.output.resolve() not in (args.trace.resolve(),args.trace.with_suffix(".json").resolve()),"Report must not overwrite trace")
    os.environ.setdefault("OMP_NUM_THREADS","8")
    os.environ["KINECT_CUDA_REGISTRATION"]="cpu"
    from scripts.profile_session import source_hash
    require(source_hash()==FROZEN,"Frozen production9331 required before numerical imports")
    report={"kind":"bounded-real-grid-lookup-ablation","status":"running","mode":args.mode,
        "reverse_order":args.reverse_order,
        "script_sha256":file_hash(__file__),"source_sha256":FROZEN,
        "cpu_hardware":{"platform":platform.platform(),"processor":platform.processor(),"logical_cpus":os.cpu_count()},
        "limits":{"max_batches":args.max_batches,"queries_per_batch":args.queries,"trace_bytes":args.trace_mib*1024**2,
                  "variant_cache_bytes":args.cache_mib*1024**2,"variant_clouds":12,"lookup_repeats":args.repeats},
        "authority":"Sampled kernel candidate selection only. Original raw-derived measured local poses, no archived seeds or coordinate rounding. No whole-component/Finish/live-FPS gain or new shader full-proof authority.",
        "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    save(args.output,report)
    failure=None
    try:
        import numpy as np,open3d as o3d,cv2,cupy as cp
        from scripts import benchmark_uniform_grid_nn as original,benchmark_parallel_fragments as baseline
        from scripts.cuda_uniform_grid_registration import DOMAIN
        from scripts.validate_uniform_grid_proof import validate_grid_proof
        from scripts.process_metrics import gpu_info
        o3d.utility.set_max_threads(20);cv2.setNumThreads(20)
        require(os.environ["OMP_NUM_THREADS"]=="8","OriginalOMP8 proof policy required")
        bridge=json.loads(args.proof_bridge.read_text())
        fixture,reference,raw,tasks,fixture_binding=original.load_fixture_scope(args,baseline.numerical_source_hash())
        report["fixture_binding"]=fixture_binding
        report["proof_report_hashes"]={"synthetic":file_hash(args.proof_synthetic),"bridge":file_hash(args.proof_bridge)}
        report["input_binding"]={name:{"path":str(path.resolve()),"sha256":file_hash(path)} for name,path in
            (("fixture",args.fixture),("reference",args.reference),("raw",raw),("synthetic_proof",args.proof_synthetic),("bridge_proof",args.proof_bridge))}
        proof_report={"source_sha256":FROZEN,"component_source_sha256":baseline.numerical_source_hash(),
            "artifacts_sha256":{name:file_hash(ROOT/name) for name in bridge["proof_bindings"]["artifacts_sha256"]},
            "gpu":gpu_info(),"domain":DOMAIN,
            "versions":{"numpy":np.__version__,"open3d":o3d.__version__,"opencv":cv2.__version__,"cupy":cp.__version__},
            "thread_policy":{"open3d":20,"opencv":20,"omp":"8"},"cache_policy":bridge["proof_bindings"]["cache_policy"]}
        require(proof_report["cache_policy"]["max_clouds"]==64
                and proof_report["cache_policy"]["retained_gpu_bytes"]==256*1024**2,
                "Capture owns the exact original64-cloud/256MiB proof policy")
        require(proof_report["gpu"] is not None,"Actual GPU/driver identity unavailable")
        binding=original.current_runtime_binding(proof_report,cp,np,o3d)
        authority=validate_grid_proof(args.proof_synthetic,args.proof_bridge,binding,fixture_binding)
        report["full_radius_proof_binding"]=binding
        paths=[Path(__file__),ROOT/"scripts/cuda_staged_grid_registration.py",ROOT/"scripts/research_staged_grid_nn.cu",
               ROOT/"scripts/research_parallel_grid_nn.cu",ROOT/"scripts/research_parallel_staged_grid_nn.cu"]
        if args.include_flat:paths.append(ROOT/"scripts/research_flat_grid_nn.cu")
        report["ablation_artifacts_sha256"]={str(path.relative_to(ROOT)):file_hash(path) for path in paths}
        if args.mode=="capture":
            capture_trace(args,report,(np,o3d,cv2,cp),fixture,tasks,authority,bridge,binding)
        else:
            manifest=json.loads(args.trace.with_suffix(".json").read_text())
            capture_report=json.loads(Path(manifest["capture_report"]).read_text())
            require(manifest["batches"]==capture_report["trace_batches"]
                    and manifest["trace_sha256"]==capture_report["trace_sha256"]
                    and manifest["retained_array_bytes"]==capture_report["retained_array_bytes"]
                    and manifest["limits"]==capture_report["limits"]
                    and manifest["reverse_order"]==capture_report["reverse_order"]
                    and manifest["producer_sha256"]==capture_report["script_sha256"]
                    and manifest["source_sha256"]==capture_report["source_sha256"]
                    and manifest["full_radius_proof_binding"]==capture_report["full_radius_proof_binding"]
                    and manifest["fixture_binding"]==capture_report["fixture_binding"]
                    and manifest["input_binding"]==capture_report["input_binding"]
                    and manifest["proof_report_hashes"]==capture_report["proof_report_hashes"],
                    "Manifest scope/arrays/provenance differs from the exact closed capture report")
            require(manifest["status"]=="passed" and manifest.get("all_final_capture_guards_passed") is True
                    and manifest["capture_report_sha256"]==file_hash(manifest["capture_report"])
                    and capture_report["status"]=="passed" and capture_report["all_provenance_guards_passed"] is True
                    and capture_report["all_nine_original_gates_passed"] is True
                    and manifest["producer_sha256"]==report["script_sha256"]
                    and manifest["source_sha256"]==FROZEN and manifest["full_radius_proof_binding"]==binding
                    and manifest["fixture_binding"]==fixture_binding and manifest["input_binding"]==report["input_binding"]
                    and manifest["proof_report_hashes"]==report["proof_report_hashes"]
                    and manifest["trace_sha256"]==file_hash(args.trace),"Trace producer/source/hardware/raw/proof bindings changed")
            report["trace_sha256"],report["trace_manifest_sha256"]=file_hash(args.trace),file_hash(args.trace.with_suffix(".json"))
            benchmark_trace(args,report,(np,o3d,cv2,cp),manifest)
        with cp.cuda.Device(0),cp.cuda.Stream.null:cp.cuda.Stream.null.synchronize()
        report["gpu_after"]=gpu_info()
        require(report["gpu_after"]==proof_report["gpu"],"GPU/driver changed during run")
        require(original.current_runtime_binding(proof_report,cp,np,o3d)==binding,"CPU/CUDA runtime binary/configuration changed")
        require(source_hash()==FROZEN and baseline.numerical_source_hash()==binding["component_source_sha256"],"Core/math source changed")
        require(all(file_hash(ROOT/name)==digest for name,digest in report["ablation_artifacts_sha256"].items()),"Ablation shader/producer source changed")
        require(all(file_hash(item["path"])==item["sha256"] for item in report["input_binding"].values()),"Original raw/fixture/reference/proof files changed")
        require(all(file_hash(ROOT/name)==digest for name,digest in binding["artifacts_sha256"].items()),"Original proved producer/helper changed")
        if args.mode=="benchmark":
            require(file_hash(args.trace)==report["trace_sha256"] and file_hash(args.trace.with_suffix(".json"))==report["trace_manifest_sha256"],"Trace changed during ablation")
        report["all_provenance_guards_passed"]=True
        report["status"]="passed"
    except BaseException as error:
        failure=error
        report["status"]="failed"
        report["failure"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        if args.mode=="capture" and args.trace.with_suffix(".json").exists():
            try:
                manifest=json.loads(args.trace.with_suffix(".json").read_text());manifest["status"]="failed"
                manifest["capture_failure"]=report["failure"];save(args.trace.with_suffix(".json"),manifest)
            except BaseException as cleanup_error:
                report["trace_failure_marking_error"]=str(cleanup_error)
    finally:
        report["ended_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        try:save(args.output,report)
        except BaseException as write_error:
            if failure is not None:raise failure.with_traceback(failure.__traceback__) from write_error
            raise
    if failure is not None:raise failure
    if args.mode=="capture":
        # The closed report is the authority, not a provisional trace emitted
        # before source/hardware/input/cleanup guards have finished.
        try:
            manifest=json.loads(args.trace.with_suffix(".json").read_text())
            manifest.update(status="passed",all_final_capture_guards_passed=True,
                            capture_report_sha256=file_hash(args.output))
            save(args.trace.with_suffix(".json"),manifest)
        except BaseException as close_error:
            report["status"]="failed"
            report["failure"]={"type":type(close_error).__name__,"message":str(close_error),"traceback":traceback.format_exc()}
            save(args.output,report)
            raise
    print(json.dumps({"output":str(args.output),"status":report["status"],"mode":args.mode}),flush=True)
    from scripts.process_metrics import finish_cuda_worker
    finish_cuda_worker()


if __name__=="__main__":
    main()
