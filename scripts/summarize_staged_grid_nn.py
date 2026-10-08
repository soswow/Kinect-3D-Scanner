"""Pure-stdlib derived summary of immutable original-double grid reports."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_staged_grid_proof import validate_grid_proof


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT/"benchmark-output/cuda-pipeline/uniform-grid-nearest/staged-pruned-v1"
    paths = {name: folder/(name+".json") for name in ("synthetic-proof", "bridge-audit", "timing")}
    hashes_before = {name: digest(path) for name, path in paths.items()}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    timing, audit, synthetic = (reports[name] for name in ("timing", "bridge-audit", "synthetic-proof"))
    authority = validate_grid_proof(paths["synthetic-proof"], paths["bridge-audit"],
                                   timing["proof_bindings"], timing["fixture_binding"])
    assert timing["status"] == "passed" and timing["performance_attribution_valid"] is True
    assert timing["timing_driver"]["unchanged"] is True
    assert timing["source_sha256"] == timing["source_sha256_after"]
    assert timing["artifacts_sha256"] == timing["artifacts_sha256_after"]
    assert timing["gpu"] == timing["gpu_after"] and timing["cleanup_passed"] is True
    assert timing["fixture_arrays_unchanged"] and timing["cloud_arrays_unchanged"]
    assert timing["original_fixture_authority"]["same_decisions_and_support"]
    for field in ("fixture_sha256", "reference_sha256", "original_raw_input_sha256"):
        assert timing[field] == timing[field+"_after"]
    native = timing["real_runs"][0]
    assert native["mode"] == "native_cpu" and native["complete"]
    grid = timing["real_runs"][1:]
    assert len(grid) == 2
    runs = []
    max_pose = max_info = 0.
    for run in grid:
        assert run["mode"] == "grid" and run["complete"]
        assert [pair["pair"] for pair in run["pairs"]] == [[8,12],[0,2]]
        assert all(pair["pair_verdict_same"] for pair in run["pairs"])
        for pair in run["pairs"]:
            for proposal in pair["proposal_results"]:
                q = proposal["quality"]
                assert q["passed"] and q["same_decision"] and not q.get("exact_field_differences", [])
                max_pose = max(max_pose, q.get("max_transform_entry_delta") or 0.)
                max_info = max(max_info, q.get("max_information_delta") or 0.)
        stats = run["statistics_delta"]
        assert stats["query_rows"] == audit["real_runs"][1]["statistics_delta"]["query_rows"]
        assert stats["audited_hits"] == stats["audited_misses"] == stats["exact_cpu_queries"] == 0
        assert stats["proof_membership_cpu_clouds"] == 0
        # Nested wall timers: subtract contained groups before adding self costs.
        groups = {
            "gpu_search_and_host_transfers": stats["gpu_search_transfer_s"],
            "cpu_fallback_loop_overhead_no_queries": stats["cpu_fallback_s"],
            "cpu_shadow_loop_overhead_no_queries": stats["cpu_audit_s"],
            "other_correspondence_work": stats["correspondence_wall_s"]-stats["gpu_search_transfer_s"]-stats["cpu_fallback_s"]-stats["cpu_audit_s"],
            "dataset_preparation": stats["dataset_wall_s"],
            "cpu_estimator_transform_and_icp_loop_combined": stats["icp_wall_s"]-stats["correspondence_wall_s"]-stats["dataset_wall_s"],
            "bridge_work_outside_icp": run["elapsed_s"]-stats["icp_wall_s"],
        }
        assert all(value >= 0 for value in groups.values())
        assert abs(sum(groups.values())-run["elapsed_s"]) < 1e-8
        runs.append({"repeat":run["repeat"], "elapsed_s":run["elapsed_s"],
            "ratio_to_native":run["elapsed_s"]/native["elapsed_s"],
            "self_groups_s":groups, "self_sum_s":sum(groups.values()),
            "inclusive_groups_s":{key:stats[key] for key in ("icp_wall_s","correspondence_wall_s","dataset_wall_s")},
            "counters":{key:stats[key] for key in ("query_rows","candidate_visits","searches","pose_iterations","icp_calls",
                "direct_gpu_hits","declared_gpu_misses","index_builds","cloud_uploads","cache_hits","cache_evictions",
                "exact_cpu_queries","audited_hits","audited_misses","proof_membership_cpu_clouds",
                "pruned_candidates","double_evaluations","grid_slot_evictions","pointer_table_evictions",
                *[f"stage{stage}_{name}" for stage in range(3) for name in ("rows","candidate_visits","pruned_candidates","double_evaluations","early_certificates")])},
            "gpu_memory_snapshot":run["gpu_memory_after"]})
    warm = runs[-1]
    warm_stats = grid[-1]["statistics_delta"]
    search_budget = native["elapsed_s"]-(warm["elapsed_s"]-warm_stats["gpu_search_transfer_s"])
    raw_snapshots = {name:digest(path) for name,path in paths.items()}
    assert raw_snapshots == hashes_before
    summary = {
        "kind":"derived-staged-pruned-original-double-uniform-grid-summary", "status":"passed",
        "result":"Exact but slower: staged/pruned warm bridge is approximately twice contemporary native CPU wall time; research only",
        "production_promotion":False,
        "source_sha256":timing["source_sha256"], "component_source_sha256":timing["component_source_sha256"],
        "raw_report_sha256":raw_snapshots, "proof_bindings_sha256":authority.bindings_sha256,
        "fixture_binding_sha256":authority.fixture_binding_sha256, "proof_target_count":len(authority.target_digests),
        "fixture_binding":timing["fixture_binding"], "artifact_sha256":timing["artifacts_sha256"],
        "timing_driver":timing["timing_driver"], "metadata":timing["proof_bindings"],
        "synthetic":{"host_device_cases":len(synthetic["synthetic_indices"]),
            "extra_boundary_cases":len(synthetic["synthetic_grid_boundaries"]),
            "host_enclosure":synthetic["host_enclosure"], "statistics":synthetic["synthetic_statistics"],
            "interval_guards":synthetic["synthetic_interval_guards"],
            "budget_fallback":synthetic["synthetic_budget_fallback"],"ownership_guards":synthetic["synthetic_ownership_guards"]},
        "real_audit":{"elapsed_s":audit["real_runs"][1]["elapsed_s"],
            "statistics":audit["real_runs"][1]["statistics_delta"], "all_nine_original_proposals_passed":True},
        "timing":{"native_elapsed_s":native["elapsed_s"], "grid":runs,
            "peak_retained_gpu_bytes":timing["statistics"]["peak_retained_gpu_bytes"],
            "required_warm_search_budget_to_match_native_s":search_budget,
            "required_warm_search_improvement_factor":warm_stats["gpu_search_transfer_s"]/search_budget if search_budget>0 else None,
            "candidate_visit_ratio_to_full_radius":warm_stats["candidate_visits"]/14097651238,
            "FP64_evaluation_ratio_to_full_radius":warm_stats["double_evaluations"]/14097651238,
            "peak_grids_per_cloud":timing["statistics"]["peak_grids_per_cloud"],
            "peak_tables_per_cloud":timing["statistics"]["peak_tables_per_cloud"]},
        "quality":{"accepted_and_rejected_decisions_unchanged":True,"all_original_witness_fields_unchanged":True,
            "max_pose_coefficient_delta":max_pose,"max_information_delta":max_info,
            "arrays_raw_inputs_fixture_reference_source_gpu_unchanged":True,"cleanup_passed":True},
        "timer_limits":"Inclusive groups overlap; self groups subtract nested timers and sum to wall. Native C++ NN/solve internals are unexposed. GPU search includes launches, host/device transfers and synchronization; no kernel-only claim. Disabled fallback/shadow timers still include empty-loop setup overhead.",
        "authority_limits":"Original-raw fixed bridge component with measured local poses only; no archived live seeds, camera-FPS or whole-Finish improvement claim. Timing reuses independently audited proofs; no fresh timing-phase synthetic GPU coverage.",
        "source_snapshot_directory":"benchmark-output/cuda-pipeline/uniform-grid-nearest/staged-pruned-v1/source",
        "summary_script_sha256":digest(Path(__file__)),
    }
    output = folder/"research-summary.json"
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output.relative_to(ROOT)),"status":summary["status"],
                      "native_s":native["elapsed_s"],"grid_s":[run["elapsed_s"] for run in runs],"search_factor_required":summary["timing"]["required_warm_search_improvement_factor"]}))


if __name__ == "__main__":
    main()
