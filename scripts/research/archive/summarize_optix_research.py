"""Summarize preserved RTX component evidence without importing CUDA/Open3D."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import hashlib
import json

DIRECTORY = ROOT / "benchmark-output/cuda-pipeline/optix-nearest"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = DIRECTORY / name
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    return path, report


def summarize(label, names, direct=False):
    loaded = [load(name) for name in names]
    synthetic, audit, timing = [report for _, report in loaded]
    provenance_fields = ("source_sha256", "verification_dependencies_sha256",
                         "artifacts_sha256", "radius_bins", "device", "gpu", "versions")
    if direct:
        provenance_fields += ("miss_policy", "certified_domain")
    shared_provenance = all(report[field] == synthetic[field]
                            for report in (audit, timing) for field in provenance_fields)
    guards = ("passed", "source_unchanged", "math_unchanged", "artifacts_unchanged",
              "fixture_unchanged", "raw_input_unchanged")
    reports_valid = all(report.get(field) is True for _, report in loaded for field in guards) \
        and timing.get("performance_attribution_valid") is True \
        and timing.get("cpu_index_audit") is False and timing.get("cpu_miss_audit", False) is False
    raw_bound = (audit["fixture_sha256"] == timing["fixture_sha256"]
                 and audit["raw_input_sha256"] == timing["raw_input_sha256"])
    native_audit = [run["statistics_delta"] for pair in audit["real_pairs"] for run in pair["rtx_runs"]]
    counts = {key: sum(row.get(key, 0) for row in native_audit)
              for key in ("query_rows", "direct_rt_hit_queries", "audited_rt_hit_queries",
                          "audit_index_mismatches", "full_radius_rt_misses", "rt_miss_cpu_hits",
                          "declared_rt_miss_queries", "audited_rt_miss_queries", "audit_false_misses",
                          "candidate_visits", "rt_uncertain_queries", "exact_cpu_queries")}
    hits_valid = (audit.get("cpu_index_audit") is True
                  and synthetic.get("synthetic_device_adapter_checked") is True
                  and counts["direct_rt_hit_queries"] > 0
                  and counts["direct_rt_hit_queries"] == counts["audited_rt_hit_queries"]
                  and counts["audit_index_mismatches"] == 0)
    misses_valid = counts["rt_miss_cpu_hits"] == 0
    if direct:
        misses_valid = misses_valid and audit.get("cpu_miss_audit") is True \
            and counts["declared_rt_miss_queries"] > 0 \
            and counts["declared_rt_miss_queries"] == counts["audited_rt_miss_queries"] \
            and counts["audit_false_misses"] == 0
    else:
        misses_valid = misses_valid and counts["exact_cpu_queries"] >= counts["full_radius_rt_misses"]
    pair_rows = []
    for pair in timing["real_pairs"]:
        pair_rows.append({"pair": pair["pair"], "accepted": pair["original_pair_verdict"]["accepted"],
            "proposals": pair["proposals"], "native_cpu_s": pair["cpu_original_s"],
            "rtx_first_s": pair["rtx_runs"][0]["elapsed_s"],
            "rtx_warm_s": [run["elapsed_s"] for run in pair["rtx_runs"][1:]],
            "all_proposal_checks_passed": all(check["passed"] for run in pair["rtx_runs"]
                                               for check in run["agreement"]),
            "ordered_pair_verdicts_agree": all(run["pair_verdict_agrees"] for run in pair["rtx_runs"])})
    passes = min(len(pair["rtx_runs"]) for pair in timing["real_pairs"])
    native_s = sum(pair["cpu_original_s"] for pair in timing["real_pairs"])
    timing_rows = []
    for repeat in range(passes):
        runs = [pair["rtx_runs"][repeat] for pair in timing["real_pairs"]]
        times = {key: sum(run["statistics_delta"].get(key, 0) for run in runs)
                 for key in ("rt_search_s", "cpu_fallback_s", "dataset_wall_s", "index_build_s",
                             "icp_wall_s", "correspondence_wall_s")}
        rtx_s = sum(run["elapsed_s"] for run in runs)
        timing_rows.append({"repeat": repeat, "state": "first-needed-indexes" if repeat == 0 else "persistent-indexes",
            "native_cpu_s": native_s, "rtx_full_bridge_s": rtx_s, "speedup_native_over_rtx": native_s / rtx_s,
            "slowdown_rtx_over_native": rtx_s / native_s, "timers_s": times,
            "correspondence_array_metric_other_s": times["correspondence_wall_s"] - times["rt_search_s"] - times["cpu_fallback_s"],
            "cpu_estimator_transform_python_loop_residual_s": times["icp_wall_s"] - times["dataset_wall_s"] - times["correspondence_wall_s"]})
    return {"label": label, "proof_status": "passed-component-only" if all((reports_valid, shared_provenance, raw_bound, hits_valid, misses_valid)) else "invalid",
        "reports": [{"path": str(path), "sha256": digest(path)} for path, _ in loaded],
        "provenance_agrees": shared_provenance, "recorded_unchanged_guards_pass": reports_valid,
        "same_fixture_and_raw_input": raw_bound, "native_hit_audit_complete": hits_valid,
        "native_miss_cpu_resolution_or_audit_complete": misses_valid,
        "source_sha256": timing["source_sha256"], "verification_dependencies_sha256": timing["verification_dependencies_sha256"],
        "artifacts_sha256": timing["artifacts_sha256"], "radius_bins": timing["radius_bins"],
        "miss_policy": timing.get("miss_policy", "cpu-fallback"), "device": timing["device"], "gpu": timing["gpu"],
        "fixture_sha256": timing["fixture_sha256"], "raw_input_sha256": timing["raw_input_sha256"],
        "native_audit_counts": counts, "pairs": pair_rows, "timing": timing_rows,
        "setup_s": timing["setup_s"], "fixture_load_and_hash_s": timing["fixture_load_and_hash_s"],
        "fixture_unpack_and_cache_s": timing["fixture_unpack_and_cache_s"],
        "retained_cache_peak_bytes": timing["final_statistics"]["peak_allocated_cache_gpu_bytes"],
        "bvh_peak_bytes": timing["final_statistics"]["peak_bvh_bytes"],
        "process_rss_peak_bytes": timing["peak_process_rss_bytes"],
        "memory_scope": "Retained points/AABB/BVH cache; excludes query buffers, temporary build workspace and CuPy pool reservations.",
        "promotion": "Research only: slower full bridge than original native CPU; no whole-session mesh validation."}


def main():
    rows = [summarize("conservative CPU miss fallback", (
                "conservative-v1/synthetic.json", "conservative-v1/real-pairs-audit.json", "conservative-v1/real-pairs.json")),
            summarize("research certified direct misses with staged radii", (
                "synthetic-direct-staged.json", "real-pairs-direct-staged-audit.json", "real-pairs-direct-staged.json"), direct=True),
            summarize("single-launch staged RTX traversal with register payload", (
                "staged-v1/synthetic.json", "staged-v1/real-pairs-audit.json", "staged-v1/real-pairs.json"), direct=True)]
    report = {"kind": "standalone-rtx-nearest-research-summary", "summary_script_sha256": digest(Path(__file__)),
        "experiments": rows,
        "scope": "Same original CPU estimator/convergence and independent proposal gates on fixed archive component inputs. Fixture local poses are component inputs only; no live tracking, adaptive frontier or mesh proof.",
        "next_research": "Exact uniform-grid search is a separate pending structure. Single-launch RTX staging is measured here and remains slower than native CPU."}
    path = DIRECTORY / "research-summary.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "status": [row["proof_status"] for row in rows]}))
    if any(row["proof_status"] == "invalid" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
