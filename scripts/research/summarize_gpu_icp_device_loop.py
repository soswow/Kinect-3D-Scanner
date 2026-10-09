"""Stdlib scalar publication of closed own-loop audit and timing reports.

This creates no permit and performs no numerical imports/calls. The exact
original audit schema is reused through a private function/global copy; only
its resource reader is publication-scoped to current helper source bytes and
recorded before/after fixed-resource manifests. Private ZIP/binary bytes are
not rescanned. Original numerical producers performed those expensive hashes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research import gpu_icp_device_loop_protocol as protocol

KIND="gpu-icp-device-loop-scalar-summary-v1"
OWN_FILES=("scripts/research/summarize_gpu_icp_device_loop.py","tests/test_gpu_icp_device_loop_summary.py")
PARTITION=("constructor_wall_s","start_cache_build_upload_sync_wall_s","advance_sync_wall_s",
    "terminal_copy_sync_wall_s","result_validation_wall_s","cleanup_wall_s")
PRODUCERS=("scripts/research/gpu_icp_device_loop_experiment.py",
    "scripts/research/gpu_icp_device_loop_protocol.py","tests/test_gpu_icp_device_loop_protocol.py")


def require(value,message):
    if not value: raise ValueError(message)


def digest(value):
    return hashlib.sha256(protocol.canonical(value).encode("utf-8")).hexdigest()


def seconds(value, *, positive=False):
    require(type(value) in (int,float) and math.isfinite(value) and
        (value>0 if positive else value>=0),"Finite measured host wall required")
    return float(value)


def distribution(values):
    require(bool(values),"Positive measured sample coverage required")
    values=[seconds(v) for v in values]
    return {"samples":len(values),"first_s":values[0],"median_s":statistics.median(values),
        "min_s":min(values),"max_s":max(values),"all_s":values}


def publication_resources(binding):
    """Report-domain checks, not fresh numerical-runtime or fixed-byte authority."""
    require(type(binding) is dict and set(binding)=={"source_sha256","method_source","artifacts_sha256","runtime","fixed_files"}
        and binding["source_sha256"]==protocol.CURRENT,"Complete measured current-source binding required")
    protocol.normalized_source(binding["method_source"])
    artifacts,fixed,runtime=binding["artifacts_sha256"],binding["fixed_files"],binding["runtime"]
    require(type(artifacts) is dict and set(PRODUCERS)<=set(artifacts) and type(fixed) is dict and fixed,
        "Complete actual producer/helper and fixed-resource inventory required")
    for name,expected in binding["method_source"]["artifacts"].items():
        require(artifacts.get(name)==expected,"Every mathematical source must belong to the measured inventory")
    for name,expected in artifacts.items():
        require(type(name) is str and "\\" not in name and ":" not in name and ".." not in Path(name).parts
            and not Path(name).is_absolute() and (ROOT/name).resolve().is_relative_to(ROOT)
            and protocol.is_digest(expected) and protocol.sha(ROOT/name)==expected,"Current measured helper source changed")
    require(all(type(name) is str and name and protocol.is_digest(expected) for name,expected in fixed.items()),
        "Recorded fixed-resource manifest malformed")
    require(runtime.get("device")=="CUDA:0" and runtime.get("open3d")=="0.20.0"
        and runtime.get("thread_policy")=={"open3d":20,"opencv":20,"omp":"8"},"Actual measured device/library/thread policy required")
    binaries=runtime.get("binaries")
    require(type(binaries) is dict and set(binaries)=={"numpy","open3d","cupy","native"},"Every loaded numerical binary must be recorded")
    require(all(type(r) is dict and set(r)=={"path","sha256"} and fixed.get(r["path"])==r["sha256"]
        and protocol.is_digest(r["sha256"]) for r in binaries.values()),"Actual loaded binary identity must belong to fixed-resource closure")


def audit_schema(report):
    # The measured validator itself is unmodified. This read-only clone cannot
    # call its token constructor/registry; the substituted resource domain is
    # explicitly recorded in the public output and supplies no runtime permit.
    globals_copy=dict(protocol.__dict__)
    globals_copy["actual_resource_closure"]=publication_resources
    reader=FunctionType(protocol.validate_audit_report.__code__,globals_copy,
        "publication_audit_schema",protocol.validate_audit_report.__defaults__,protocol.validate_audit_report.__closure__)
    return reader(report,report["binding"])


def closed(report):
    require(report.get("status")=="passed" and report.get("failure") is None
        and report.get("cleanup_failures")==[] and report.get("cleanup_passed") is True
        and report.get("input_bytes_unchanged") is True and report.get("loaded_owners_unchanged") is True,
        "Every actual input/owner/runtime/cleanup closure must pass")
    require(report.get("binding") is not None and report.get("binding_after")==report["binding"]
        and report.get("whole_finish_authority") is False,"Changed current sources/resources or whole-Finish claim refused")
    require(type(report.get("configurations")) is list and report["configurations"]
        and len(set(report["configurations"]))==len(report["configurations"]),"Unique measured configuration list required")
    publication_resources(report["binding"])


def row_key(row):
    pair=row.get("pair")
    require(type(pair) is list and len(pair)==2 and all(type(n) is int and n>=0 for n in pair)
        and type(row.get("seed_index")) is int and 0<=row["seed_index"]<6,"Bounded original pair/seed identity required")
    return tuple(pair)+(row["seed_index"],)


def row_contract(row):
    consumed=row["input_binding"]
    protocol.normalized_input(consumed)
    source=protocol.normalized_source(row["source_binding"])
    loop=row["loop_report"]
    require(loop.get("closed") is True and loop.get("failure") is None
        and loop.get("input_binding")==consumed and protocol.normalized_source(loop.get("provenance"))==source
        and loop.get("whole_finish_authority") is False,"Actual selected-stream lane must close with exact claimed inputs/source")
    label=row.get("configuration")
    require(label in ("step1","graph1","graph2","graph4"),"Unknown graph/chunk configuration")
    cfg=consumed["configuration"]
    require((cfg["cuda_graph"],cfg["chunk_iterations"])==((False,1) if label=="step1" else (True,int(label[-1])))
        and loop.get("cuda_graph")==cfg["cuda_graph"],"Recorded graph/chunk label differs from actual execution")
    shadow=row.get("native_shadow")
    require(type(shadow) is dict and shadow.get("passed") is True and shadow.get("correspondence_ids_equal") is True,
        "Original native same-input terminal result shadow required")
    for name in ("transform_max_abs_delta","fitness_abs_delta","rmse_abs_delta"):
        require(type(shadow.get(name)) in (int,float) and math.isfinite(shadow[name]) and 0<=shadow[name]<=1e-8,
            "Fresh original native result tolerance failed")
    values=[seconds(row.get(k)) for k in PARTITION]
    require(sum(values)==seconds(row.get("all_in_wall_s"),positive=True),"Disjoint host-wall partition does not match actual all-in sum")
    stats=loop["statistics"]
    require(type(stats.get("control_copies")) is int and stats["control_copies"]>0
        and type(stats.get("peak_lane_bytes")) is int and 0<stats["peak_lane_bytes"]<=cfg["max_total_bytes"]
        and type(stats.get("maximum_graph_nodes")) is int and 0<=stats["maximum_graph_nodes"]<=11*cfg["chunk_iterations"]
        and type(stats.get("graph_captures")) is int and type(stats.get("graph_launches")) is int,
        "Measured bounded graph/control-copy/lane ownership counters required")
    capture=seconds(stats.get("graph_capture_s"))
    require(capture<=row["advance_sync_wall_s"] and
        ((stats["graph_captures"]==1 and stats["graph_launches"]>0 and stats["maximum_graph_nodes"]>0 and capture>0)
        if cfg["cuda_graph"] else (stats["graph_captures"]==stats["graph_launches"]==stats["maximum_graph_nodes"]==0 and capture==0)),
        "Actual one-owned-graph capture must be nested in advance; step mode cannot claim graph work")
    return protocol.canonical(protocol.normalized_input(consumed)),source


def compact_case(name,audit,timing,*,audit_sha):
    require(type(name) is str and name and all(c.isalnum() or c in "-_" for c in name),"Safe public scalar case label required")
    closed(audit);closed(timing)
    scopes=audit_schema(audit)
    require(timing.get("kind")==protocol.TIMING_KIND and timing.get("mode")=="timing"
        and timing.get("performance_attribution_valid") is True and timing["binding"]==audit["binding"]
        and timing["configurations"]==audit["configurations"],"Exact same current-source own-audit scope required for timed rows")
    require(timing.get("audit_proof",{}).get("sha256")==audit_sha,"Timing must reference this exact closed audit")
    references={protocol.canonical(s["input"]):s for s in scopes}
    audit_rows={}
    for row in audit["rows"]:
        key,source=row_contract(row)
        require(key not in audit_rows,"Duplicate exact audited input/configuration")
        audit_rows[key]=row
    rows=timing.get("rows")
    require(type(rows) is list and rows,"Positive measured timing trajectories required")
    repeats=sorted({r.get("repeat") for r in rows})
    require(repeats==list(range(len(repeats))) and 1<=len(repeats)<=5
        and all(type(r.get("repeat")) is int for r in rows),"Bounded complete repeat sequence required")
    coverage=set();identities=set()
    for row in rows:
        key,source=row_contract(row)
        ref=references.get(key)
        require(ref is not None and ref["source"]==source and row["terminal"]==ref["terminal"],
            "Actual timed pose/correspondence/metric/query/update bytes differ from own audited trajectory")
        protocol.validate_timing_accounting(row["loop_report"]["statistics"],row["input_binding"]["source"]["shape"][0])
        require(row["query_trace"]==[] and row["query_trace_sha256"]==digest([]),"Full NN audit cannot be hidden in the timed path")
        stats,gold=row["loop_report"]["statistics"],audit_rows[key]["loop_report"]["statistics"]
        require(all(stats[k]==gold[k] for k in ("queries","updates","query_rows","direct_hits","direct_misses","cpu_ambiguity_rows")),
            "Timed actual NN and update accounting differs from own full CPU-shadowed trajectory")
        identity=row_key(row);identities.add(identity)
        require(identity==row_key(audit_rows[key]),"Original pair/seed label cannot substitute for another input scope")
        record=(row["repeat"],identity,row["configuration"])
        require(record not in coverage,"Duplicate row cannot substitute for missing timing coverage");coverage.add(record)
    require(coverage=={(r,i,c) for r in repeats for i in identities for c in timing["configurations"]},
        "Every selected trajectory/configuration/repeat must be timed")
    require(identities=={row_key(r) for r in audit["rows"]},"Timing silently dropped an actual audited pair/seed")
    controls=timing.get("native_controls")
    require(type(controls) is list and controls,"Fresh same-input original native controls required")
    cpu_coverage=set()
    for row in controls:
        require(type(row.get("repeat")) is int and row["repeat"] in repeats,"Malformed original control repeat")
        identity=row_key(row);record=(row["repeat"],identity)
        require(identity in identities and record not in cpu_coverage,"Duplicate/foreign original native control")
        cpu_coverage.add(record);seconds(row.get("wall_s"),positive=True)
        protocol.descriptor(row.get("pose"),4,"<f8")
        require(row["pose"]["shape"]==[4,4] and all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) and row[k]>=0
            for k in ("fitness","inlier_rmse")) and row["fitness"]<=1,"Finite original CPU control metrics required")
        for candidate in rows:
            if candidate["repeat"]==row["repeat"] and row_key(candidate)==identity:
                require(abs(row["fitness"]-candidate["terminal"]["fitness"])==candidate["native_shadow"]["fitness_abs_delta"]
                    and abs(row["inlier_rmse"]-candidate["terminal"]["inlier_rmse"])==candidate["native_shadow"]["rmse_abs_delta"],
                    "CPU control metrics do not belong to their same-input/repeat actual result shadow")
    require(cpu_coverage=={(r,i) for r in repeats for i in identities},"Missing matched original control coverage")
    cpu=distribution([sum(r["wall_s"] for r in controls if r["repeat"]==repeat) for repeat in repeats])
    groups=[]
    for label in timing["configurations"]:
        selected=[r for r in rows if r["configuration"]==label]
        def totals(key): return [sum(r[key] for r in selected if r["repeat"]==repeat) for repeat in repeats]
        partition={key:distribution(totals(key)) for key in PARTITION}
        all_in=distribution(totals("all_in_wall_s"))
        capture=[sum(seconds(r["loop_report"]["statistics"]["graph_capture_s"]) for r in selected if r["repeat"]==repeat) for repeat in repeats]
        groups.append({"configuration":label,"trajectories_per_repeat":len(identities),
            "all_in_host_wall":all_in,"partition":partition,"nested_graph_capture_wall":distribution(capture),
            "all_in_gpu_to_native_median_ratio":all_in["median_s"]/cpu["median_s"],
            "advance_and_terminal_host_wall":distribution([sum(r["advance_sync_wall_s"]+r["terminal_copy_sync_wall_s"] for r in selected if r["repeat"]==repeat) for repeat in repeats]),
            "maximum_graph_nodes":max(r["loop_report"]["statistics"]["maximum_graph_nodes"] for r in selected),
            "peak_lane_owned_bytes":max(r["loop_report"]["statistics"]["peak_lane_bytes"] for r in selected),
            "control_copies_per_repeat":[sum(r["loop_report"]["statistics"]["control_copies"] for r in selected if r["repeat"]==repeat) for repeat in repeats]})
    trajectory_medians=[]
    for index,identity in enumerate(sorted(identities)):
        selected=[r for r in rows if row_key(r)==identity]
        original=selected[0]["input_binding"]
        native_median=statistics.median(r["wall_s"] for r in controls if row_key(r)==identity)
        configurations=[]
        for label in timing["configurations"]:
            matching=[r for r in selected if r["configuration"]==label]
            configurations.append({"configuration":label,
                "all_in_median_s":statistics.median(r["all_in_wall_s"] for r in matching),
                "partition_median_s":{k:statistics.median(r[k] for r in matching) for k in PARTITION},
                "nested_graph_capture_median_s":statistics.median(r["loop_report"]["statistics"]["graph_capture_s"] for r in matching)})
        trajectory_medians.append({"trajectory_index":index,"source_points":original["source"]["shape"][0],
            "target_points":original["target"]["shape"][0],"native_cpu_median_s":native_median,
            "configurations":configurations})
    fields=("query_rows","audited_hits","audited_misses","cpu_ambiguity_rows")
    return {"case":name,"status":"passed","source_sha256":timing["binding"]["source_sha256"],
        "trajectories":len(identities),"repeats":len(repeats),
        "audit_coverage":{k:sum(r["loop_report"]["statistics"][k] for r in audit["rows"]) for k in fields},
        "native_cpu_matched_host_wall":cpu,"configurations":groups,
        "per_trajectory_medians":trajectory_medians,
        "median_partition_scope":"Stage medians are descriptive separately and need not sum to the median all-in wall. Every underlying row's actual stage sum is checked exactly.",
        "fresh_native_and_exact_own_terminal_shadows_passed":True,
        "complete_proposal_bridge_gates_measured":False,"whole_finish_authority":False,
        "process_preparation_wall_s":seconds(timing["process_preparation_wall_s"]),
        "phase_closure_wall_s":seconds(timing["phase_closure_wall_s"]),
        "total_run_wall_s_before_final_save":seconds(timing["total_run_wall_s_before_final_save"]),
        "binding_sha256":digest(timing["binding"]),"fixed_resource_manifest_sha256":digest(timing["binding"]["fixed_files"]),
        "fixed_resource_count":len(timing["binding"]["fixed_files"]),
        "method_generated_sha256":timing["binding"]["method_source"]["generated_sha256"],
        "helper_source_artifacts_sha256":timing["binding"]["artifacts_sha256"],
        "numerical_binary_sha256":{k:r["sha256"] for k,r in timing["binding"]["runtime"]["binaries"].items()},
        "thread_policy":timing["binding"]["runtime"]["thread_policy"]}


def run(args):
    output=args.output.resolve()
    require(output.is_relative_to(ROOT/"benchmark-output") and not output.exists(),"Fresh private publication output required")
    result={"kind":KIND,"status":"running","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"cases":[],
        "whole_finish_authority":False,"authority_token_created":False,"fixed_resource_bytes_rehashed_by_summary":False,
        "validation_scope":"Read-only exact own-loop audit schema using a private function/global copy; current helper bytes and recorded before/after fixed-resource manifests. No token or numerical-runtime authority. Producer hashed actual private resources before/after numerical jobs; controller exit completion is separate.",
        "timer_scope":"Per-repeat sums over identical selected trajectories. All-in is exactly constructor+start/index/build/upload/sync+advance/sync+terminal/copy/sync+native-result-and-permit-validation+cleanup. Graph capture is nested inside advance, not additive. Every trajectory rebuilds target indexes and owns a fresh graph; later repeats warm only process/compiler/allocator caches. Native controls run first and device order reverses on alternate repeats. Cold process file/library proof preparation and final closure are separately recorded. Advance+terminal is a partial host timer, not GPU event time or a complete implementation speed claim.",
        "limitations":["Selected prepared-pair ICP seed trajectories only; no complete proposal/witness/frontier/Finish/tracking/mesh gates were replayed by this experiment.",
            "Tight original same-input native result shadows and exact own-audit pose/correspondence/metrics/query-update bytes validate this method-specific component scope only.",
            "Three small repeated controls and two field-session cases do not establish general throughput or scanning FPS.",
            "Per-trajectory setup/graph capture/source validation remain charged. Potential future reuse must be measured separately."]}
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save()
    try:
        names=set()
        for name,audit_name,timing_name in args.case:
            require(name not in names,"Duplicate public case label");names.add(name)
            a,t=(Path(p).resolve(strict=True) for p in (audit_name,timing_name))
            require(a.is_relative_to(ROOT/"benchmark-output") and t.is_relative_to(ROOT/"benchmark-output"),"Only local private measured reports accepted")
            ah,th=protocol.sha(a),protocol.sha(t)
            audit,timing=(json.loads(p.read_text(encoding="utf-8")) for p in (a,t))
            require(Path(timing["audit_proof"]["path"]).resolve()==a,"Timing references a different audit file")
            row=compact_case(name,audit,timing,audit_sha=ah)
            require(protocol.sha(a)==ah and protocol.sha(t)==th,"Measured report changed during publication")
            row["reports"]={"audit":{"path":a.relative_to(ROOT).as_posix(),"sha256":ah},
                "timing":{"path":t.relative_to(ROOT).as_posix(),"sha256":th}}
            result["cases"].append(row)
        result["audited_query_rows"]=sum(c["audit_coverage"]["query_rows"] for c in result["cases"])
        result["producer_artifacts_sha256"]={p:protocol.sha(ROOT/p) for p in OWN_FILES}
        result["status"]="passed";save()
    except BaseException as error:
        result["status"]="failed";result["failure"]={"type":type(error).__name__,"message":str(error)}
        try:save()
        except BaseException as write_error:
            error.add_note("Scalar diagnostic write also failed: "+repr(write_error));raise error from write_error
        raise
    print(f"Closed scalar device-loop summary: {len(result['cases'])} cases, {result['audited_query_rows']} audited rows")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",nargs=3,action="append",required=True,metavar=("NAME","AUDIT","TIMING"))
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
