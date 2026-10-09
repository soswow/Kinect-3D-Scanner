"""Publish scalar evidence from closed independent-seed microbatch experiments.

Stdlib only: no CUDA/native imports, tracking, image, pose, graph or permit
creation. This validates current helper bytes and the measured reports' complete
before/after resource closure. It does not rescan private ZIPs or loaded binary
bytes; those were hashed by the numerical producers outside their batch timers.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import gpu_icp_experiment_protocol as protocol

KIND = "gpu-icp-microbatch-scalar-summary-v1"
TIMING_KIND = "gpu-icp-seed-microbatch-timing-v1"
OWN_FILES = ("scripts/research/summarize_gpu_icp_microbatch.py",
    "tests/test_gpu_icp_microbatch_summary.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(protocol.canonical(value).encode("utf-8")).hexdigest()


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
        "Positive finite measured batch wall required")
    return float(value)


def distribution(rows):
    values = [seconds(r["batch_wall_s"]) for r in rows]
    return {"samples": len(values), "first_s": values[0], "median_s": statistics.median(values),
        "min_s": min(values), "max_s": max(values), "all_s": values}


def closed(report):
    require(report.get("status") == "passed" and report.get("failure") is None
        and report.get("validation_failure") is None and report.get("cleanup_failures") == [],
        "A running/failed/contradictory report cannot be published as measured success")
    require(all(report.get(k) is True for k in ("cleanup_passed", "source_unchanged", "fixture_unchanged", "input_bytes_unchanged")),
        "Complete measured owner/input/source/cleanup closure required")
    require(report.get("source_sha256") == report.get("source_sha256_after") == protocol.CURRENT
        and report.get("binding") == report.get("binding_after")
        and report.get("pair_binding") == report.get("pair_binding_after")
        and report.get("producer_artifacts_sha256") == report.get("producer_artifacts_sha256_after")
        and report.get("fixed_files_sha256") == report.get("fixed_files_sha256_after"),
        "Changed actual inputs, resources or helpers cannot support a timing table")
    for name in ("producer_artifacts_sha256", "fixed_files_sha256"):
        require(type(report.get(name)) is dict and bool(report[name]), "Nonempty source/resource closure required")
    helpers = report.get("helpers")
    require(type(helpers) is list and helpers and all(h.get("closed") is True
        and h.get("source_unchanged") is True and h.get("failure") is None
        and h.get("cleanup_failures") == [] and h.get("binding") == report["binding"]
        and h.get("whole_finish_authority") is False for h in helpers), "Every actual helper owner must close")
    require(report.get("whole_finish_authority") is False, "Microbatch reports cannot authorize whole Finish")


def shadows(row, size, *, native=False):
    require(row.get("passed") is True, "A failed measured row must stay visible as a failure")
    values = row.get("shadows") if native else row.get("native_shadows")
    require(type(values) is list and [s.get("seed_index") for s in values] == list(range(size)),
        "Each original ordered seed needs its fresh native result shadow")
    for shadow in values:
        require(all(shadow.get(k) is True for k in ("passed", "correspondence_ids_equal", "strong_gate_equal")),
            "Original native correspondence/strong-gate result changed")
        for key in ("transform_max_abs_diff", "fitness_abs_diff", "inlier_rmse_abs_diff"):
            value = shadow.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e-8,
                "Original native result tolerance failed")


def timing_rows(report, audit):
    require(report.get("kind") == TIMING_KIND and report.get("performance_attribution_valid") is True,
        "Separate authorized nonaudit timing required")
    require(report.get("binding") == audit["binding"] and report.get("pair_binding") == audit["pair_binding"]
        and report.get("batch_sizes") == audit["batch_sizes"]
        and report.get("producer_artifacts_sha256") == audit["producer_artifacts_sha256"],
        "Timing and audit must bind the same actual arrays, source, method and resources")
    # The timing producer additionally pins the exact already-closed audit file.
    proof = report["audited_proof"]
    expected_files = dict(audit["fixed_files_sha256"])
    expected_files[str(Path(proof["path"]).resolve())] = proof["sha256"]
    require(report.get("fixed_files_sha256") == expected_files,
        "Only the exact closed audit file may extend the original fixed resource set")
    protocol.binding_contract(report["binding"])
    pair, sizes = report["pair_binding"], report["batch_sizes"]
    rows, native = report.get("timing_rows"), report.get("native_timing_rows")
    require(type(rows) is list and rows and type(native) is list and native, "Complete matched timing rows required")
    repeats = sorted({r.get("repeat") for r in rows})
    require(repeats == list(range(len(repeats))) and 1 <= len(repeats) <= 5
        and all(type(r.get("repeat")) is int for r in rows), "Bounded complete repeat sequence required")
    coverage, native_coverage = set(), set()
    for row in rows:
        size, schedule, repeat = row.get("batch_size"), row.get("schedule"), row["repeat"]
        require(size in sizes and schedule in ("serial", "concurrent") and (repeat,size,schedule) not in coverage,
            "Unique genuine serial/concurrent timing rows required")
        coverage.add((repeat,size,schedule))
        require(row.get("pair_binding") == protocol.prefix_pair(pair,size)
            and row.get("pair_verdict_equal") is True and row.get("full_original_proposal_count") == len(pair["seeds"]),
            "Original ordered-prefix/full competing-proposal verdict changed")
        shadows(row,size)
        gates = row.get("bridge_gate_shadows")
        require(type(gates) is list and [g.get("seed_index") for g in gates] == list(range(size))
            and all(g.get("passed") is True and g.get("original_forward_consumed_once") is True for g in gates),
            "Original complete proposal witness gates must consume each scoped result exactly once")
        batch = row["batch_record"]
        require(batch.get("inputs") == row["pair_binding"] and batch.get("schedule") == schedule
            and batch.get("lanes") == size and batch.get("input_bytes_unchanged") is True
            and batch.get("full_call_fallbacks") == 0,
            "Measured batch differs from its claimed actual scope/wall")
        require(seconds(batch.get("wall_s")) <= seconds(row["batch_wall_s"]),
            "Inner adapter timer cannot exceed the enclosing complete driver call timer")
        lanes = batch.get("lane_deltas")
        require(type(lanes) is list and [l.get("index") for l in lanes] == list(range(size)), "Complete ordered timing lanes required")
        for lane in lanes:
            nn, icp = lane["retrieval_statistics"], lane["statistics"]
            require(icp.get("calls") == 1 and icp.get("cpu_fallback_calls") == 0
                and all(nn.get(k) == 0 for k in ("audited_hits", "audited_misses", "audit_index_mismatches", "audit_false_misses", "device_malformed_results")),
                "Nonaudit timing must omit only authorized shadows; no hidden fallback or malformed result")
            require(all(type(nn.get(k)) is int and nn[k] >= 0 for k in
                ("query_rows", "direct_gpu_hits", "declared_gpu_misses", "exact_cpu_queries"))
                and nn["query_rows"] > 0
                and nn["query_rows"] == nn["direct_gpu_hits"]+nn["declared_gpu_misses"]+nn["exact_cpu_queries"],
                "Actual timing correspondence/ambiguity accounting required")
    for row in native:
        size, repeat = row.get("batch_size"), row.get("repeat")
        require(type(repeat) is int and repeat in repeats and size in sizes and (repeat,size) not in native_coverage,
            "Unique matched contemporary native timing controls required")
        native_coverage.add((repeat,size)); shadows(row,size,native=True); seconds(row["batch_wall_s"])
    require(coverage == {(r,n,s) for r in repeats for n in sizes for s in ("serial","concurrent")}
        and native_coverage == {(r,n) for r in repeats for n in sizes}, "Missing matched native/GPU timing coverage")
    require(all(h.get("audit") is False or h["statistics"].get("successful_batches") == 0 for h in report["helpers"]),
        "Actual timed batches must use the authorized unaudited helper")
    return rows, native


def compact_case(name, audit, timing, *, audit_sha):
    require(type(name) is str and name and all(c.isalnum() or c in "-_" for c in name), "Safe public case label required")
    closed(audit); closed(timing)
    protocol.validate_report(audit,audit["binding"],audit["pair_binding"])
    proof = timing.get("audited_proof")
    require(type(proof) is dict and proof.get("sha256") == audit_sha, "Timing must name this exact closed audit report")
    rows, native = timing_rows(timing,audit)
    counts = {k:0 for k in ("query_rows","audited_hits","audited_misses","exact_cpu_queries")}
    for row in audit["audit_rows"]:
        for lane in row["batch_record"]["lane_deltas"]:
            for key in counts: counts[key] += lane["retrieval_statistics"][key]
    groups = []
    for size in timing["batch_sizes"]:
        selected = sorted((r for r in native if r["batch_size"]==size),key=lambda r:r["repeat"])
        cpu = distribution(selected)
        group = {"seeds":size,"native_cpu":cpu}
        for schedule in ("serial","concurrent"):
            selected = sorted((r for r in rows if r["batch_size"]==size and r["schedule"]==schedule),key=lambda r:r["repeat"])
            values = distribution(selected)
            values["nested_adapter_wall_s"] = [seconds(r["batch_record"]["wall_s"]) for r in selected]
            values["enclosing_policy_and_call_wall_s"] = [r["batch_wall_s"]-r["batch_record"]["wall_s"] for r in selected]
            values["gpu_to_native_median_wall_ratio"] = values["median_s"]/cpu["median_s"]
            group[schedule] = values
        groups.append(group)
    stats = [h["statistics"] for h in timing["helpers"] if h.get("audit") is False]
    require(stats,"Positive actual unaudited helper required")
    for record in stats:
        require(all(type(record.get(k)) in (int,float) and math.isfinite(record[k]) and record[k] >= 0
            for k in ("cold_setup_s","source_checks_s"))
            and type(record.get("source_checks")) is int and record["source_checks"] >= 0
            and type(record.get("peak_combined_owned_bytes")) is int and record["peak_combined_owned_bytes"] > 0,
            "Measured finite setup/source-check cost and explicit owned memory required")
    return {"case":name,"status":"passed","source_sha256":timing["source_sha256"],
        "source_points":timing["pair_binding"]["source"]["points"]["shape"][0],
        "target_points":timing["pair_binding"]["target"]["points"]["shape"][0],
        "genuine_proposals":len(timing["pair_binding"]["seeds"]),
        "audit_coverage":counts,"original_native_and_complete_proposal_gates_passed":True,
        "timing_groups":groups,"cold_unaudited_helper_setup_s":sum(s["cold_setup_s"] for s in stats),
        "setup_scope":"Unaudited helper construction only; audit-validation probe construction, full proof validation and original gold/gate work are outside this value and the batch timer.",
        "nested_source_checks_s":sum(s["source_checks_s"] for s in stats),
        "source_checks":sum(s["source_checks"] for s in stats),
        "peak_owned_gpu_bytes":max(s["peak_combined_owned_bytes"] for s in stats),
        "binding_sha256":digest(timing["binding"]),"pair_binding_sha256":digest(timing["pair_binding"]),
        "fixed_resource_manifest_sha256":digest(timing["fixed_files_sha256"]),
        "fixed_resource_count":len(timing["fixed_files_sha256"]),
        "helper_source_artifacts_sha256":timing["binding"]["artifacts_sha256"],
        "method_policy":timing["binding"]["method_policy"],
        "thread_policy":timing["binding"]["runtime"]["thread_policy"],
        "gpu_event_statistics_uncollected":["gpu_transform_ms","gpu_equations_ms"],
        "whole_finish_authority":False}


def run(args):
    output = args.output.resolve()
    require(output.is_relative_to(ROOT/"benchmark-output") and not output.exists(), "Fresh private summary output required")
    result = {"kind":KIND,"status":"running","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "whole_finish_authority":False,"performance_authority_token_created":False,
        "fixed_resource_bytes_rehashed_by_summary":False,"cases":[],
        "timer_scope":"Enclosing driver call host wall includes permit/source/policy checks, shared preparation/upload, target cache misses, lane iterations, CPU Eigen solve, final result copies and stream/thread completion. The adapter's narrower timer is nested; their difference measures enclosing call/policy work, not a separate GPU stage. Cold helper setup and original CPU result/complete proposal gate shadows are outside the batch timer. Persistent target cache is shared; source/normals are uploaded per batch. Source-check time is nested, not additive. First sample is the first occurrence for that seed count/schedule, not an isolated cold start.",
        "limitations":["Two selected prepared field pairs, genuine original seed prefixes and three small repeated controls do not establish whole scanning/Finish/FPS performance.",
            "Original CPU solve and dependent reverse/camera/witness/gate calls remain on CPU. Concurrent streams batch only independent initial-forward seeds.",
            "Measured report resource manifests were hashed before/after numerical jobs. Publication rechecks helper source bytes and report bindings without rescanning private ZIP/library bytes.",
            "Later samples share the same helper target cache but do not assume monotonically faster warm timings. Per-group minima/maxima are retained."]}
    output.parent.mkdir(parents=True,exist_ok=True)
    def save(): output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save()
    try:
        names = set()
        for name,audit_name,timing_name in args.case:
            require(name not in names,"Duplicate public case label");names.add(name)
            a,t = Path(audit_name).resolve(strict=True),Path(timing_name).resolve(strict=True)
            require(a.is_relative_to(ROOT/"benchmark-output") and t.is_relative_to(ROOT/"benchmark-output"), "Only local private measured reports accepted")
            ah,th = protocol.sha(a),protocol.sha(t)
            audit,timing = (json.loads(p.read_text(encoding="utf-8")) for p in (a,t))
            require(Path(timing["audited_proof"]["path"]).resolve()==a,"Timing references another audit path")
            row = compact_case(name,audit,timing,audit_sha=ah)
            require(protocol.sha(a)==ah and protocol.sha(t)==th,"Measured report changed during publication")
            row["reports"]={"audit":{"path":a.relative_to(ROOT).as_posix(),"sha256":ah},
                "timing":{"path":t.relative_to(ROOT).as_posix(),"sha256":th}}
            result["cases"].append(row)
        result["audited_query_rows"] = sum(c["audit_coverage"]["query_rows"] for c in result["cases"])
        result["producer_artifacts_sha256"]={p:protocol.sha(ROOT/p) for p in OWN_FILES}
        result["status"]="passed";save()
    except BaseException as error:
        result["status"]="failed";result["failure"]={"type":type(error).__name__,"message":str(error)}
        try: save()
        except BaseException as write_error:
            error.add_note("Summary failure preservation also failed: "+repr(write_error))
            raise error from write_error
        raise
    print(f"Closed scalar microbatch summary: {len(result['cases'])} cases, {result['audited_query_rows']} audited rows")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",nargs=3,action="append",required=True,metavar=("NAME","AUDIT","TIMING"))
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
