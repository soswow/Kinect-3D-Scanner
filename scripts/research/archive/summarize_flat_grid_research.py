"""Stdlib-only compact evidence for the separate measured flat host bridge."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import argparse
import hashlib
import json
import math

from scripts.research.archive.validate_flat_grid_proof import validate_grid_proof, require


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"benchmark-output/cuda-pipeline/uniform-grid-nearest/flat-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.directory/"research-summary.json"
    require(not out.exists(), "Require a fresh derived summary path")
    paths = {name: args.directory/filename for name, filename in (
        ("synthetic", "synthetic-proof.json"), ("audit", "bridge-audit.json"), ("timing", "timing.json"))}
    hashes = {name: digest(path) for name, path in paths.items()}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    audit, timing = reports["audit"], reports["timing"]
    authority = validate_grid_proof(paths["synthetic"], paths["audit"], audit["proof_bindings"], audit["fixture_binding"])
    require(all(digest(ROOT/name) == value for name, value in authority.artifact_sha256), "Measured source artifacts changed")
    require(timing["status"] == "passed" and timing["cleanup_passed"] is True
            and not timing.get("failure") and timing["performance_attribution_valid"] is True
            and timing["cpu_hit_audit"] is False and timing["cpu_miss_audit"] is False
            and timing["proof_bindings"] == audit["proof_bindings"] == timing["proof_bindings_after"]
            and timing["fixture_binding"] == audit["fixture_binding"]
            and timing["source_sha256"] == timing["source_sha256_after"]
            and timing["artifacts_sha256"] == timing["artifacts_sha256_after"]
            and timing["gpu"] == timing["gpu_after"] and timing["timing_driver"]["unchanged"] is True,
            "Timing source/runtime/input/hardware/cleanup or attribution contract failed")
    auth = timing["validated_proof_authority"]
    require(auth["bindings_sha256"] == authority.bindings_sha256
            and auth["fixture_binding_sha256"] == authority.fixture_binding_sha256
            and auth["synthetic_report_sha256"] == hashes["synthetic"]
            and auth["bridge_report_sha256"] == hashes["audit"], "Timing does not bind the actual fresh proof files")
    cpu = [run for run in timing["real_runs"] if run["mode"] == "native_cpu"]
    gpu = [run for run in timing["real_runs"] if run["mode"] == "grid"]
    require(len(cpu) == 1 and cpu[0]["complete"] is True and [run["repeat"] for run in gpu] == [0, 1], "Expected one native and two flat runs")
    native = cpu[0]["elapsed_s"]
    require(math.isfinite(native) and native > 0, "Invalid contemporary native wall time")
    qualities, rows = [], []
    for run in gpu:
        require(run["complete"] is True and [(pair["position"], pair["pair"]) for pair in run["pairs"]] == [(0, [8, 12]), (4, [0, 2])], "Changed accepted/rejected pair order")
        for pair, task in zip(run["pairs"], audit["fixture_binding"]["tasks"]):
            proposals = pair["proposal_results"]
            require(pair["pair_verdict_same"] is True
                    and [entry["input_sha256"] for entry in proposals] == task["proposal_sha256"]
                    and [entry["proposal_index"] for entry in proposals] == list(range(len(task["proposal_sha256"])))
                    and all(entry["quality"]["passed"] is True and entry["quality"]["same_decision"] is True
                        and (entry["evidence"] is None or entry["quality"].get("exact_field_differences") == []) for entry in proposals),
                    "Original ordered proposal witnesses/decision/pose/information gates failed")
            qualities.extend(entry["quality"] for entry in proposals)
        stats = run["statistics_delta"]
        require(all(stats[name] == 0 for name in ("audited_hits", "audited_misses", "audit_index_mismatches", "audit_false_misses", "exact_cpu_queries")), "Timing includes CPU shadows/fallback queries/errors")
        partition = {"gpu_search_transfer_output_parse_s": stats["gpu_search_transfer_s"],
            "empty_cpu_fallback_bookkeeping_s": stats["cpu_fallback_s"],
            "empty_cpu_audit_bookkeeping_s": stats["cpu_audit_s"],
            "other_correspondence_s": stats["correspondence_wall_s"]-stats["gpu_search_transfer_s"]-stats["cpu_fallback_s"]-stats["cpu_audit_s"],
            "dataset_s": stats["dataset_wall_s"],
            "cpu_estimator_transform_loop_s": stats["icp_wall_s"]-stats["correspondence_wall_s"]-stats["dataset_wall_s"],
            "outside_icp_verification_s": run["elapsed_s"]-stats["icp_wall_s"]}
        require(all(math.isfinite(value) and value >= 0 for value in partition.values())
                and abs(sum(partition.values())-run["elapsed_s"]) < 1e-8, "Disjoint timing partition is invalid")
        floor = run["elapsed_s"]-stats["gpu_search_transfer_s"]
        rows.append({"repeat": run["repeat"], "cache_state": "first" if run["repeat"] == 0 else "warm",
            "full_bridge_wall_s": run["elapsed_s"], "slower_than_native_ratio": run["elapsed_s"]/native,
            "native_over_flat_speed_ratio": native/run["elapsed_s"], "disjoint_partition_s": partition,
            "non_search_floor_s": floor, "search_budget_to_equal_native_s": max(0., native-floor),
            "query_rows": stats["query_rows"], "candidate_visits": stats["candidate_visits"],
            "index_builds": stats["index_builds"], "cache_hits": stats["cache_hits"]})
    audit_run = next(run for run in audit["real_runs"] if run["mode"] == "grid")
    stats = audit_run["statistics_delta"]
    summary = {"kind": "flat-grid-host-bridge-research-summary", "status": "passed",
        "producer": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(__file__)},
        "claim": "Fresh flat-specific synthetic and complete CPU hit/miss proofs pass; both full host bridge timings are slower than contemporary native CPU.",
        "whole_archive_speedup_measured": False, "production_promotion": False,
        "reports": {name: {"path": str(path.resolve()), "sha256": hashes[name]} for name, path in paths.items()},
        "proof_bindings": audit["proof_bindings"], "fixture_binding": audit["fixture_binding"],
        "native_cpu_full_bridge_s": native, "flat_runs": rows,
        "audit": {"query_rows": stats["query_rows"], "direct_hits": stats["direct_gpu_hits"], "audited_hits": stats["audited_hits"],
            "direct_misses": stats["declared_gpu_misses"], "audited_misses": stats["audited_misses"],
            "index_mismatches": stats["audit_index_mismatches"], "false_misses": stats["audit_false_misses"],
            "candidate_visits": stats["candidate_visits"], "target_membership_count": len(authority.target_digests),
            "wall_s": audit_run["elapsed_s"], "cpu_shadow_s": stats["cpu_audit_s"], "performance_attribution_valid": False},
        "quality": {"all_nine_original_gates_per_run_passed": True,
            "max_pose_coefficient_difference": max(q["max_transform_entry_delta"] for q in qualities if q.get("max_transform_entry_delta") is not None),
            "max_information_matrix_difference": max(q["max_information_delta"] for q in qualities if q.get("max_information_delta") is not None),
            "metric_scope": "Available accepted-proposal pose/information matrices; rejected None evidence agrees exactly and has no such matrix."},
        "solver_setup": timing["setup_partition"],
        "memory": {"retained_index_cap_bytes": timing["cache_policy"]["retained_gpu_bytes"],
            "peak_retained_index_bytes": timing["statistics"]["peak_retained_gpu_bytes"],
            "peak_retained_index_MiB": timing["statistics"]["peak_retained_gpu_bytes"]/1024**2,
            "peak_process_rss_bytes": timing["peak_process_rss_bytes"],
            "peak_process_rss_MiB": timing["peak_process_rss_bytes"]/1024**2,
            "scope": "Retained index arrays exclude transient query/output buffers and CuPy pools; process RSS and GPU snapshots are separate."},
        "limits": "Fixed original-raw component with measured local poses; accepted/rejected nine-proposal gates, not complete Finish/live tracking/mesh. Inclusive timers overlap. GPU-search timer includes host upload/download/enqueue/output extraction, not pure shader instruction time. CPU native search/solve internals are unexposed. Different resident methods/trajectories require separate fresh proofs.",
        "cleanup_passed": True}
    require({name: digest(path) for name, path in paths.items()} == hashes, "Measured reports changed during derivation")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(out), "status": summary["status"], "native_s": native,
        "flat_s": [row["full_bridge_wall_s"] for row in rows]}))


if __name__ == "__main__":
    main()
