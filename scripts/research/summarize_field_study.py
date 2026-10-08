"""Publishable scalar evidence from closed local field reports; stdlib only.

This is a retrospective report, never a numerical/quality/backend authority.
No native imports, raw ZIP reads, image/pose/graph arrays or absolute personal
paths are copied. Historical source hashes stay historical; failed parent and
quality records are retained rather than relabelled successful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.summarize_field_latency import summarize as latency_summary

STUDY = ROOT/"benchmark-output/field-cuda-study"
RAW = {
    "chest-5": ("raw-baselines-v1/chest-5-scan-session-cpu-0.json",
                "raw-baselines-v1/chest-5-scan-session-cuda-0.json", "raw-baselines-v1/chest-5-scan-session-quality.json"),
    "chest-6": ("raw-baselines-v1/chest-6-scan-session_20261009_002019-cpu-0.json",
                "raw-baselines-v1/chest-6-scan-session_20261009_002019-cuda-0.json", "raw-baselines-v1/chest-6-scan-session_20261009_002019-quality.json"),
    "chest-7": ("raw-baselines-20k-v1/chest-7-scan-session_20261009_003259-cpu-0.json",
                "raw-baselines-20k-cuda-v1/chest-7-scan-session_20261009_003259-cuda-0.json", "raw-baselines-20k-cuda-v1/chest-7-quality.json"),
}
DEFERRED = {
    "chest-5": ("deferred-policy-v1/chest-5-scan-session-deferred.json", "deferred-policy-v1/chest-5-quality.json"),
    "chest-6": ("deferred-policy-v1/chest-6-scan-session_20261009_002019-deferred.json", "deferred-policy-v1/chest-6-quality.json"),
    "chest-7": ("deferred-policy-20k-v1/chest-7-scan-session_20261009_003259-deferred.json", "deferred-policy-20k-v1/chest-7-quality.json"),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def check(value, message):
    if not value:
        raise ValueError(message)


def number(value):
    check(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Nonfinite/negative scalar metric")
    return value


def count(value, positive=False):
    check(type(value) is int and value >= int(positive), "Malformed integer counter")
    return value


def surface(value):
    keys = ("threshold_m", "samples_per_surface", "precision", "completeness", "surface_p95_m", "surface_rmse_m")
    result = {key:number(value[key]) for key in keys}
    check(result["threshold_m"] == .005 and result["samples_per_surface"] == 30000
          and .999 <= result["precision"] <= 1 and .999 <= result["completeness"] <= 1
          and result["surface_p95_m"] <= .0005
          and value["alignment"] == "fixed input coordinate frame; no scale or trajectory fitting", "Unexpected surface scope")
    result["alignment"] = "fixed input coordinates; no fitting"
    return result


class Reports:
    def __init__(self, root=STUDY):
        self.root = Path(root)
        self.files, self.values = {}, {}

    def read(self, name):
        check(type(name) is str and bool(name) and not name.startswith("/") and not Path(name).is_absolute()
              and "\\" not in name and ":" not in name and ".." not in Path(name).parts,
              "Require a repository-relative report reference")
        if name not in self.values:
            path = self.root/name
            raw = path.read_bytes()
            check(len(raw) <= 32*1024**2, "Report exceeds bounded JSON size")
            value = json.loads(raw)
            check(isinstance(value, dict), "Report must be a JSON object")
            check(value.get("status") in (None, "profile", "running", "complete", "passed", "failed"), "Unexpected report status")
            self.values[name] = value
            self.files[name] = {"path":"benchmark-output/field-cuda-study/"+name, "sha256":sha(raw),
                "bytes":len(raw), "reported_status":value.get("status", "profile")}
        return self.values[name]

    def record(self, name):
        self.read(name)
        return dict(self.files[name])

    def close(self):
        check(all(sha((self.root/name).read_bytes()) == value["sha256"] for name,value in self.files.items()),
              "Input report bytes changed during aggregation")


def pair_scope(a, b, allowed_options):
    """Close actual raw selection/settings/thread scope before deriving a ratio."""
    check(all(a[key] == b[key] for key in ("source_sha256", "input_sha256", "selected_indices", "settings",
        "thread_policy", "omp_threads", "seed", "versions")), "Replay actual source/input/selection/settings/thread scope differs")
    options_a, options_b = a["pipeline_options"], b["pipeline_options"]
    changes = {key for key in set(options_a)|set(options_b) if options_a.get(key) != options_b.get(key)}
    check(changes == set(allowed_options) and all((options_a[key], options_b[key]) == values
          for key,values in allowed_options.items()), "Unrelated requested workflow/backend changes")
    check(a["native_extension"]["sha256"] == b["native_extension"]["sha256"]
          and all(row["native_extension"]["changed_during_profile"] is False for row in (a,b)), "Native kernel identity changed")
    check(all(row.get("pose_seeds_used") is False and row.get("mesh_built") is True
          and row.get("input_changed_during_profile") is False and row.get("source_changed_during_profile") is False
          and row["settings"]["confidence_fusion"] is True for row in (a,b)), "Incomplete/unseeded weighted raw replay required")
    check(a["accepted_indices"] == b["accepted_indices"], "Actual Final view membership differs")
    for row in (a,b):
        indices = row["selected_indices"]
        count(row["frames"],True)
        check(type(indices) is list and len(indices) == row["frames"] and indices == list(range(row["frames"]))
              and all(type(index) is int for index in indices), "Complete ordered raw selection required")


def profile(reports, name):
    value = reports.read(name)
    row = latency_summary(reports.root/name)
    check(row["profile_sha256"] == reports.record(name)["sha256"], "Latency input differs from closed report bytes")
    return {"evidence":reports.record(name), "source_sha256":row["source_sha256"], "input_sha256":row["input_sha256"],
        "settings_sha256":row["settings_sha256"], "selected_indices_sha256":canonical(value["selected_indices"]),
        "frames":row["frames"], "live_views":row["accepted_before_finish"], "final_views":row["accepted_after_finish"],
        "mesh_built":row["mesh_built"], "live_s":number(row["live_s"]), "finish_s":number(row["finish_s"]),
        "processing_s":number(row["processing_s"]), "live_latency_ms":row["live_latency_ms"],
        "final_block_limit":value["settings"]["final_block_count"], "confidence_fusion":True,
        "cuda_confidence_acceleration":value["pipeline_options"]["KINECT_CUDA_CONFIDENCE"]}


def actual_backend(value, cuda):
    backend = value["backend"]
    inp,match = backend["cuda_input"],backend["descriptor_matching"]
    for container,keys in ((inp,("gpu_batches","cpu_batches","fallback_batches")),(match,("cuda_batches","cpu_batches"))):
        for key in keys:
            count(container[key])
    check(backend["geometric_verification"]["implementation"] == "legacy"
          and backend["depth_confidence"]["requested"] == "off"
          and backend["depth_confidence"]["implementation"] == "cpu"
          and value["settings"]["confidence_fusion"] is True, "Original geometry/CPU confidence algorithm required")
    if cuda:
        check(inp.get("requested") == "auto" and inp.get("implementation") == "cuda" and inp.get("device") == "CUDA:0"
            and inp.get("probe_passed") is True and inp.get("gpu_batches",0) > 0
            and inp.get("cpu_batches") == inp.get("fallback_batches") == 0
            and match.get("requested") == match.get("implementation") == "cuda"
            and match.get("cuda_batches",0) > 0 and match.get("cpu_batches") == 0 and match.get("fallback_reason") is None,
            "Actual CUDA input/descriptor stages differ")
    else:
        check(inp.get("requested") == "off" and inp.get("implementation") == "cpu" and inp.get("device") == "CPU:0"
            and inp.get("cpu_batches",0) > 0 and inp.get("gpu_batches") == inp.get("fallback_batches") == 0
            and match.get("requested") == match.get("implementation") == "cpu"
            and match.get("cpu_batches",0) > 0 and match.get("cuda_batches") == 0 and match.get("fallback_reason") is None,
            "Actual CPU input/descriptor stages differ")


def matched_quality(reports, name, baseline, candidate, observable=False):
    q = reports.read(name)
    if observable:
        check(q.get("status") == "complete" and q.get("observable_geometry_passed") is True
              and q["reference"]["sha256"] == reports.record(baseline)["sha256"]
              and q["candidate"]["sha256"] == reports.record(candidate)["sha256"], "Observable surface references/closure differ")
        metrics = q["triangle_surface_metrics"]
    else:
        check(q["baseline_report_sha256"] == reports.record(baseline)["sha256"]
              and q["candidate_report_sha256"] == reports.record(candidate)["sha256"], "Stored physical quality references differ")
        metrics = q["triangle_surface_metrics_vs_cpu"]
    comparison = q.get("comparison", q)
    check(comparison.get("same_mesh_success") is True and comparison.get("same_accepted_indices") is True,
          "Stored mesh/view quality comparison failed")
    return {"evidence":reports.record(name), "surface":surface(metrics), "whole_pipeline_equivalence_authority":False}


def ratio(a, b):
    check(number(a) > 0 and number(b) > 0, "Positive measured durations required")
    return {"baseline_over_candidate":a/b, "less_processing_percent":100*(1-b/a)}


def finish_pair_scope(a, b):
    """Checkpoint comparisons use the same raw-derived Live state, without copying it."""
    check(all(a[key] == b[key] for key in ("source_sha256", "input_sha256", "selected_indices", "settings",
        "pipeline_options", "thread_policy", "seed", "accepted_indices", "accepted_indices_before_finish"))
        and a["checkpoint"]["sha256"] == b["checkpoint"]["sha256"], "Finish actual raw/checkpoint/options/views scope differs")
    check(all(row.get("finish_requested") is True and row.get("pose_seeds_used") is False
        and row.get("input_changed_during_profile") is False and row.get("source_changed_during_profile") is False
        and row.get("mesh_built") is True and row["settings"]["confidence_fusion"] is True for row in (a,b)),
        "Closed unseeded weighted checkpoint Finish required")


def activation_trial(reports, name):
    value = reports.read(name)
    state = value.get("status")
    item = {"evidence":reports.record(name), "reported_status":state,
        "whole_mesh_quality_authority":False, "failure_type":value.get("failure",{}).get("type") if value.get("failure") else None}
    if state != "complete":
        return item
    check(value.get("kind") == "offline-original-cpu-finish-weighted-missing-activation-v2"
        and value["artifacts_sha256"] == value["artifacts_sha256_after"] and value["supervisor_restored"] is True
        and value["deferred_terminal_worker_calls"] == 1 and value["cleanup_failures"] == []
        and value["activation_proof"] == value["activation_proof_after"]
        and value["delegated_sidecar"] == value["delegated_sidecar_after"] and value["profile"] == value["profile_after"],
        "Whole activation envelope before/after source/input/cleanup closure failed")
    profile_name = name.replace(".missing-activation.json", ".json")
    allocation_name = name.replace(".missing-activation.json", ".allocation.json")
    physical_name = "missing-activation-v2/weighted-proof.json"
    check(value["profile"]["sha256"] == reports.record(profile_name)["sha256"]
        and value["delegated_sidecar"]["sha256"] == reports.record(allocation_name)["sha256"]
        and value["activation_proof"]["sha256"] == reports.record(physical_name)["sha256"],
        "Whole activation named evidence bytes differ")
    raw_profile,physical = reports.read(profile_name),reports.read(physical_name)
    check(raw_profile["source_sha256"] == physical["source_sha256"]
        and raw_profile.get("source_changed_during_profile") is False and raw_profile.get("input_changed_during_profile") is False
        and raw_profile.get("pose_seeds_used") is False and raw_profile.get("finish_requested") is True
        and raw_profile["settings"]["confidence_fusion"] is True, "Allocation profile raw/source/weighted scope differs")
    allocation = value["allocation"]
    check(allocation["restored"] is True and allocation["cleanup_failures"] == [], "Candidate scope restoration failed")
    check(value["mode"] in ("native-original", "exact-missing-key")
        and value["outcome"] in ("mesh-built-quality-unproven", "expected-early-capacity-failure"), "Unexpected allocation outcome")
    item.update(mode=value["mode"], outcome=value["outcome"], finish_s=number(raw_profile["finish_s"]))
    if value["mode"] == "exact-missing-key" and value["outcome"] == "mesh-built-quality-unproven":
        rows = allocation.get("final_key_proofs", [])
        check(len(rows) == 1 and allocation["native_grid_returned"] is True, "Exact candidate native/key proof absent")
        row = rows[0]
        check(row["exact_key_union"] is True and row["expected_union_sha256"] == row["actual_union_sha256"]
            and type(row["required_blocks"]) is int and row["required_blocks"] > 0
            and row["required_blocks"] == row["initial_capacity"] == row["final_capacity"] == row["final_blocks"],
            "Actual key union/capacity changed")
        item["capacity_evidence"] = {key:row[key] for key in ("required_blocks","initial_capacity","final_capacity","exact_key_union")}
        item["attribute_mib"] = row["final_capacity"]*4096*20/2**20
    return item


def activation_comparison(reports, label):
    base = "missing-activation-v2/"+label
    original, exact = base+"-original.json", base+"-exact.json"
    # Explicitly selected declared-geometry report, never a fallback that
    # silently overwrites a failed bit-identity comparison.
    quality = base+("-quality-1.json" if label == "chest-7" else "-quality.json")
    if not (reports.root/quality).exists():
        return {"session":label,"independent_quality_status":"pending","whole_mesh_quality_authority":False}
    a,b = reports.read(original), reports.read(exact)
    finish_pair_scope(a,b)
    q = reports.read(quality)
    check(q.get("same_pipeline_options") is True and q.get("same_live_accepted_indices") is True,
          "Whole allocation options/Live evidence differs")
    comparison = q["comparison"]
    check(number(comparison["max_pose_translation_delta_m"]) <= .0005
          and number(comparison["max_pose_rotation_delta_deg"]) <= .1, "Whole allocation declared Final pose limits failed")
    if label == "chest-5":
        check(q.get("identical_unrounded_final_poses") is True, "C5 exact pose comparison failed")
    first,second = activation_trial(reports,base+"-original.missing-activation.json"),activation_trial(reports,base+"-exact.missing-activation.json")
    check(first["reported_status"] == second["reported_status"] == "complete", "Allocation comparison trials incomplete")
    item = {"session":label, "independent_quality_status":"complete", "quality":matched_quality(reports,quality,original,exact,True),
        "original_finish_s":first["finish_s"], "exact_finish_s":second["finish_s"], "final_views":len(b["accepted_indices"]),
        "capacity_evidence":second["capacity_evidence"], "attribute_mib":second["attribute_mib"],
        "configured_attribute_mib":b["settings"]["final_block_count"]*4096*20/2**20,
        "bit_identical_unrounded_final_poses_proven":q.get("identical_unrounded_final_poses") is True,
        "max_final_translation_delta_m":comparison["max_pose_translation_delta_m"],
        "max_final_rotation_delta_deg":comparison["max_pose_rotation_delta_deg"],
        "declared_pose_limits":{"translation_m":.0005,"rotation_deg":.1},
        "whole_mesh_quality_authority":False,"production_speed_claim":False,
        "scope":"One paired original-CPU checkpoint Finish, including planner/observers. Attribute floor excludes hashmap, Live, scratch and pools; independent observed geometry only."}
    if label == "chest-7":
        failed_name = base+"-quality.json"
        failed = reports.read(failed_name)
        check(failed.get("status") == "failed"
              and failed["reference"]["sha256"] == reports.record(original)["sha256"]
              and failed["candidate"]["sha256"] == reports.record(exact)["sha256"], "Preserved C7 exact-pose failure references differ")
        item["preserved_strict_pose_failure"] = {"evidence":reports.record(failed_name),"reported_status":"failed",
            "scope":"Bit-identical unrounded pose requirement failed; separately named declared geometry comparison does not relabel it."}
    return item


def source_guard_component(reports):
    variants = []
    originals = []
    for mode in ("original-ast","whole-file-sha"):
        name = "combined-sync-source-guard-v1/"+mode+".json"
        value = reports.read(name)
        scope = value["source_guard_scope"]
        rows = value["real_runs"]
        check(value["status"] == "passed" and value["performance_attribution_valid"] is True
            and value["authority_unchanged"] is True and value["cleanup_failures"] == []
            and value["original_hook_restored"] is True and value["fixture_arrays_unchanged"] is True
            and value["cloud_arrays_unchanged"] == {kind:True for kind in ("native_cpu","device_resident","combined_sync")}
            and value["new_whole_finish_authority"] is False,
            "Source-guard original authority/cleanup closure failed")
        for key in ("source_sha256","component_source_sha256","artifacts_sha256","input_files",
                "combined_runtime_binding","device_runtime_binding","gpu"):
            check(value[key] == value[key+"_after"], "Source-guard actual before/after identity differs")
        check(scope["mode"] == mode and scope["sources_unchanged"] is True and scope["owned_hooks_restored"] is True
            and scope["cleanup_failures"] == [] and scope["artifacts_sha256"] == scope["artifacts_sha256_after"]
            and scope["new_kernel"] is False and scope["graph_capture"] is False
            and scope["whole_finish_authority"] is False and scope["identical_clocks_in_both_modes"] is True,
            "Source-guard new scope did not close")
        check(len(rows) == 9 and {(r["mode"],r["repeat"]) for r in rows} ==
            {(kind,repeat) for kind in ("native_cpu","device_resident","combined_sync") for repeat in range(3)}
            and all(r["complete"] is True and r["quality_complete"] is True for r in rows),
            "Source-guard all-nine/all-three completed authority required")
        for row in rows:
            if row["mode"] != "combined_sync":continue
            delta = row["source_guard_statistics_delta"]
            check(delta["source_checks"] == 236 and delta["iteration_config_checks"] == delta["preenqueue_calls"] == 11569,
                "Source-guard actual call coverage differs")
            for key in ("source_check_s","iteration_config_check_s","preenqueue_bookkeeping_s"):
                number(delta[key])
        medians = {kind:statistics.median(number(r["elapsed_s"]) for r in rows if r["mode"] == kind)
            for kind in ("native_cpu","device_resident","combined_sync")}
        variants.append({"mode":mode,"evidence":reports.record(name),"component_median_s":medians,
            "runs":[{"mode":r["mode"],"repeat":r["repeat"],"wall_s":r["elapsed_s"]} for r in rows]})
        originals.append(value)
    check(all(originals[0][key] == originals[1][key] for key in ("source_sha256","component_source_sha256",
        "artifacts_sha256","input_files","fixture_binding","combined_runtime_binding","device_runtime_binding","gpu"))
        and originals[0]["source_guard_scope"]["artifacts_sha256"] == originals[1]["source_guard_scope"]["artifacts_sha256"],
        "AST/SHA causal scope differs beyond requested guard mode")
    sha_medians = variants[1]["component_median_s"]
    return {"variants":variants,"cpu_over_sha_component":sha_medians["native_cpu"]/sha_medians["combined_sync"],
        "device_over_sha_component":sha_medians["device_resident"]/sha_medians["combined_sync"],
        "less_component_wall_vs_contemporary_device_percent":100*(1-sha_medians["combined_sync"]/sha_medians["device_resident"]),
        "whole_finish_authority":False,"whole_scan_speed_claim":False,
        "scope":"Separate source-guard A/B, each with contemporary native/device/combined three rotating all-nine rounds. Same numerical kernels/iteration and inclusive clocks; median is descriptive, not a statistical confidence interval. AST parsing cost is a research safety overhead."}


def aggregate(reports):
    out = {"schema_version":1, "kind":"publishable-scalar-field-cuda-study-v1",
        "authority":False, "privacy":"Explicit scalar allowlist only; no images, poses, per-view arrays, graph inventories, native paths or raw ZIP contents.",
        "scope":"Retrospective stored evidence, not current proof-token validation. Raw rows are one selected-view replay per mode; no camera FPS or statistical interval. Component gains cannot be applied to total scanning.",
        "raw_cpu_cuda_pairs":[], "deferred_recovery_pairs":[]}
    for label,(cpu,cuda,quality) in RAW.items():
        a,b = reports.read(cpu), reports.read(cuda)
        pair_scope(a,b,{"KINECT_CUDA_MATCHING":("cpu","cuda"), "KINECT_CUDA_INPUT":("off","auto")})
        actual_backend(a,False)
        actual_backend(b,True)
        first,second = profile(reports,cpu),profile(reports,cuda)
        out["raw_cpu_cuda_pairs"].append({"session":label, "cpu":first, "cuda":second,
            "quality":matched_quality(reports,quality,cpu,cuda), **ratio(first["processing_s"],second["processing_s"])})
        deferred,quality = DEFERRED[label]
        pair_scope(b,reports.read(deferred),{"KINECT_LIVE_RECOVERY":("full","deferred")})
        actual_backend(reports.read(deferred),True)
        third = profile(reports,deferred)
        out["deferred_recovery_pairs"].append({"session":label, "full":second, "deferred":third,
            "quality":matched_quality(reports,quality,cuda,deferred,True), **ratio(second["processing_s"],third["processing_s"]),
            "tradeoff":"Faster stored-view rejection may make Live preview sparser; Final coverage checked only for this replay."})
    out["failed_original_10k_chest7"] = [profile(reports,"raw-baselines-v1/chest-7-scan-session_20261009_003259-"+mode+"-0.json")
                                            for mode in ("cpu","cuda")]
    check(all(row["mesh_built"] is False for row in out["failed_original_10k_chest7"]), "Historical failed Final changed")
    out["parent_executions"] = []
    for name in ("raw-baselines-v1/execution.json", "raw-baselines-20k-v1/execution.json", "raw-baselines-20k-cuda-v1/execution.json"):
        value = reports.read(name)
        out["parent_executions"].append({"evidence":reports.record(name), "status":value["status"],
            "child_exit_codes":[row.get("exit_code") for row in value["runs"]], "failure_type":value.get("failure",{}).get("type"),
            "note":"The successful20k CPU child closed; its parent Unicode console-print failure remains failed." if "20k-v1" in name else None})

    name = "native-corner-cpu-cuda-v2/report.json"
    value = reports.read(name)
    check(value["status"] == "passed" and value["all_actual_queries_original_brute_shadowed"] is True
          and value["all_cpu_fast_ids_and_squared_bits_exact"] is True and value["source_sha256"] == value["source_sha256_after"], "Corner comparison not closed")
    out["exact_corner_component"] = {"evidence":reports.record(name), "actual_queries":value["actual_queries"],
        "synthetic_queries":value["synthetic_queries"], "whole_scan_speed_claim":False,
        "scope":"All-in efficient CPU versus CUDA lookup; fresh original brute shadow once/view; detection/support/ICP/Finish excluded.",
        "sessions":[{"session":"chest-"+str(index+5), "queries":row["summary"]["actual_queries"],
            "cpu_fast_all_in_s":number(row["summary"]["cpu_fast_all_in_s"]),
            "cuda_all_in_s":number(row["summary"]["gpu_component_s"]),
            "original_brute_shadow_s":number(row["summary"]["original_brute_shadow_s"]),
            "cuda_gain_over_improved_cpu":ratio(row["summary"]["cpu_fast_all_in_s"],row["summary"]["gpu_component_s"])["baseline_over_candidate"]}
            for index,row in enumerate(value["sessions"])]}

    name = "combined-sync-timing-v1/report.json"
    value = reports.read(name)
    check(value["status"] == "passed" and value["authority_unchanged"] is True and value["cleanup_failures"] == []
          and value["source_sha256"] == value["source_sha256_after"] and value["artifacts_sha256"] == value["artifacts_sha256_after"]
          and len(value["real_runs"]) == 9 and all(r["complete"] and r["quality_complete"] for r in value["real_runs"]), "Combined timing not closed")
    out["combined_sync_component"] = {"evidence":reports.record(name), "whole_finish_authority":False,
        "runs":[{"mode":r["mode"], "repeat":r["repeat"], "wall_s":number(r["elapsed_s"])} for r in value["real_runs"]],
        "scope":"Contemporary native/device/combined all-nine component, three rotating orders. Preflight/setup excluded and separately recorded.",
        "conclusion":"Combined sync was slower than original device-resident in all three rounds; no promotion."}
    out["combined_source_guard_component"] = source_guard_component(reports)

    out["marker_proposal_trials"] = []
    for label in ("chest-5","chest-7"):
        original = "marker-finish-v2/"+label+"-global-original.json"
        marker = "marker-finish-v2/"+label+"-global-marker-audit.json"
        a,b = reports.read(original),reports.read(marker)
        finish_pair_scope(a,b)
        out["marker_proposal_trials"].append({"session":label, "original_evidence":reports.record(original),
            "marker_evidence":reports.record(marker), "original_finish_s":number(a["finish_s"]),
            "marker_audited_finish_s":number(b["finish_s"]), "final_views":len(b["accepted_indices"]),
            "whole_backend_authority":False,"production_speed_claim":False,
            "scope":"Marker proposals add original verification work. Original brute corner shadows remain charged in this audit; no live-tracking or scan-throughput claim."})

    name = "device-ldlt-v4/actual-and-synthetic.json"
    value = reports.read(name)
    actual,single = value["actual_equation_comparison"],value["singleton_observation"]
    check(value["status"] == "passed" and actual["passed"] is True and actual["fresh_cpu_gold_bitwise_equal"] is True,
          "Device solve comparison not closed")
    out["device_solve_component"] = {"evidence":reports.record(name), "synthetic_systems":len(value["rows"]),
        "actual_systems":actual["count"], "fresh_cpu_dll_s":number(actual["fresh_original_dll_s"]),
        "gpu_singleton_launch_and_sync_mean_us":number(single["launch_and_sync_wall_s"])*1e6/single["repeats"],
        "scope":"Independent systems/identity previous pose only; batch/singleton micro-scopes exclude different overheads; no full ICP/trajectory claim.",
        "conclusion":"Moving only the small solver to GPU is not supported by this observation."}

    name = "bulk-nn-audit/old-trajectory-dual-proof.json"
    value = reports.read(name)
    stats = value["real_runs"][-1]["statistics_delta"]
    check(value["status"] == "passed" and value["cleanup_passed"] is True
          and stats["dual_scalar_bulk_queries"] == 104123989
          and stats["dual_id_mismatches"] == stats["dual_metric_bit_mismatches"] == 0, "Bulk dual proof not closed")
    out["bulk_cpu_audit_infrastructure"] = {"evidence":reports.record(name),
        "original_queries":stats["dual_scalar_bulk_queries"], "hits":stats["audited_hits"], "misses":stats["audited_misses"],
        "id_mismatches":0, "metric_bit_mismatches":0, "scan_speed_claim":False,
        "scope":"Complete old all-nine original resident trajectory dual scalar/bulk shadow; new field audits still require actual input proofs."}

    out["preserved_failures"] = []
    for name,cause in (
        ("resident-finish-v4/chest-5-serial-quality.json","Strict ordered native/GPU history differs; no old Finish timing authority."),
        ("resident-finish-v4/chest-6-serial-quality.json","Strict intermediate evidence differs despite matching final observables; no old authority."),
        ("final-allocation-v1/chest-5-rightsized.allocation.json","Original activation grew exact3328 candidate to6656; trial rejected growth.")):
        value = reports.read(name)
        check(value.get("status") == "failed", "Historical failed trial must not be relabelled")
        out["preserved_failures"].append({"evidence":reports.record(name), "status":"failed",
            "failure_type":value.get("failure",{}).get("type"), "scope":cause})
    name = "final-allocation-v2/chest-5-headroom.allocation.json"
    value = reports.read(name)
    allocations = value["allocation"]["allocations"]
    check(value["status"] == "complete" and value["allocation"]["restored"] is True
          and value["allocation"]["cleanup_failures"] == [] and len(allocations) == 1, "Headroom allocation not closed")
    row = allocations[0]
    quality_name = "final-allocation-v2/chest-5-headroom-quality.json"
    quality = reports.read(quality_name)
    check(quality["status"] == "complete" and quality["observable_geometry_passed"] is True
          and quality["identical_unrounded_final_poses"] is True, "Headroom observable quality failed")
    headroom_quality = matched_quality(reports,quality_name,"final-allocation-v1/chest-5-original.json",
        "final-allocation-v2/chest-5-headroom.json",True)
    out["reserve_headroom_memory"] = {"evidence":reports.record(name), "quality_evidence":reports.record(quality_name),
        "logical_limit":row["configured_logical_limit"], "unique_blocks":row["final_unique_blocks"],
        "initial_capacity":row["actual_capacity"], "final_capacity":row["final_capacity"],
        "attribute_mib":row["final_capacity"]*4096*20/2**20, "configured_attribute_mib":row["configured_logical_limit"]*4096*20/2**20,
        "surface":headroom_quality["surface"],
        "scope":"TSDF/weight/colour attribute floor only; hashmap, Live, scratch and pools additional. Observable quality is not new backend authority."}

    out["activation_proofs"] = []
    for version in ("v1","v2"):
        name = "missing-activation-"+version+"/weighted-proof.json"
        if not (reports.root/name).exists():
            out["activation_proofs"].append({"version":version,"status":"pending"});continue
        value = reports.read(name)
        check(value["status"] == "passed" and value["source_unchanged"] is True and value["environment_restored"] is True
              and value["cleanup_failures"] == [] and len(value["pairs"]) == 3
              and all(row["complete"] is True and row["inputs_unchanged"] is True and row["inputs"] == row["inputs_after"]
                  and row["exact_key_mapped_attribute_bits"] is True and row["nonzero_weight_voxels"] > 0
                  and row["original-full"]["final_capacity"] == 8 and row["missing-only"]["final_capacity"] == 4
                  and all(m["bit_mismatches"] == 0 and m["elements"] > 0 for m in row["comparison"].values())
                  for row in value["pairs"]), "Physical activation proof not closed")
        out["activation_proofs"].append({"version":version, "evidence":reports.record(name), "status":"passed",
            "whole_session_quality":False, "scan_speed_claim":False,
            "backends":[{"backend":row["backend"], "original_capacity":row["original-full"]["final_capacity"],
                "missing_only_capacity":row["missing-only"]["final_capacity"],
                "nonzero_weight_voxels":row["nonzero_weight_voxels"],
                "compared_float32_values":sum(m["elements"] for m in row["comparison"].values()),
                "bit_mismatches":sum(m["bit_mismatches"] for m in row["comparison"].values())} for row in value["pairs"]]})
    name = "missing-activation-v1/chest-5-exact.missing-activation.json"
    value = reports.read(name)
    out["preserved_observer_failure"] = {"evidence":reports.record(name), "status":value["status"],
        "failure_type":value.get("failure",{}).get("type"),
        "cause":"v1 observer treated module.shape function as array metadata before planner/candidate; v2 needs fresh proofs."}
    out["new_whole_activation_trials"] = []
    folder = reports.root/"missing-activation-v2"
    # Bounded, nonrecursive scalar envelopes only. Never follow trace/NPZ/raw
    # references or infer mesh/quality authority from an allocation envelope.
    paths = sorted(folder.glob("*.missing-activation.json")) if folder.exists() else []
    check(len(paths) <= 32, "Unbounded activation envelope inventory")
    for path in paths:
        name = "missing-activation-v2/"+path.name
        out["new_whole_activation_trials"].append(activation_trial(reports,name))
    out["new_whole_activation_comparisons"] = [activation_comparison(reports,label) for label in ("chest-5","chest-7")]
    out["new_whole_activation_status"] = "See closed scalar trial statuses; independent whole mesh/pose quality is never inferred from allocation."
    reports.close()
    out["evidence_files"] = sorted(reports.files.values(),key=lambda row:row["path"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior compact evidence; use a fresh output")
    producer = Path(__file__)
    dependency = ROOT/"scripts/research/summarize_field_latency.py"
    before = {"scripts/research/summarize_field_study.py":sha(producer.read_bytes()),
        "scripts/research/summarize_field_latency.py":sha(dependency.read_bytes())}
    result = aggregate(Reports())
    check(all(sha((ROOT/name).read_bytes()) == digest for name,digest in before.items()), "Summary source changed during reporting")
    result["summary_producer"] = {"path":"scripts/research/summarize_field_study.py", "sha256":before["scripts/research/summarize_field_study.py"]}
    result["summary_dependencies_sha256"] = before
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"evidence_files":len(result["evidence_files"]),"authority":False}))


if __name__ == "__main__":
    main()
