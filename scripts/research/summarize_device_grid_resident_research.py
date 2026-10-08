"""Stdlib-only evidence for the fresh device-flat resident component.

Run after all reports close. Inclusive GPU waits are labelled as host wall;
disabled GPU events have no duration authority. This cannot establish complete
archive Finish/mesh performance or authorize production selection.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
import math

from scripts.research.validate_device_flat_grid_proof import validate_grid_proof
from scripts.research.archive.validate_uniform_grid_proof import require


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
        default=ROOT/"benchmark-output/cuda-pipeline/uniform-grid-nearest/device-resident-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.directory/"research-summary.json"
    require(not output.exists(), "Require a fresh derived summary path")
    paths = {name: args.directory/filename for name, filename in (
        ("synthetic", "synthetic-proof.json"), ("audit", "bridge-audit.json"), ("timing", "timing.json"))}
    hashes = {name: digest(path) for name, path in paths.items()}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    audit, timing = reports["audit"], reports["timing"]
    authority = validate_grid_proof(paths["synthetic"], paths["audit"], audit["proof_bindings"], audit["fixture_binding"])
    artifacts = dict(authority.artifact_sha256)
    driver, producer = timing["timing_driver"], ROOT/"scripts/research/benchmark_device_grid_resident.py"
    driver_path = ROOT/"scripts/research/benchmark_device_grid_resident_timing.py"
    require(driver["path"] == str(driver_path.relative_to(ROOT))
        and driver["sha256"] == driver["sha256_after"] == artifacts[str(driver_path.relative_to(ROOT))]
        and driver["proof_producing_harness_path"] == str(producer.relative_to(ROOT))
        and driver["proof_producing_harness_sha256"] == driver["proof_producing_harness_sha256_after"]
        == artifacts[str(producer.relative_to(ROOT))], "Actual timing driver/producer fingerprints differ from proof authority")
    require(timing["status"] == "passed" and timing["cleanup_passed"] is True
        and not timing.get("failure") and not timing.get("cleanup_failures")
        and timing["performance_attribution_valid"] is True
        and timing["cpu_hit_audit"] is False and timing["cpu_miss_audit"] is False
        and timing["proof_bindings"] == audit["proof_bindings"] == timing["proof_bindings_after"]
        and timing["fixture_binding"] == audit["fixture_binding"]
        and timing["fixture_binding_sha256"] == authority.fixture_binding_sha256
        and timing["source_sha256"] == timing["source_sha256_after"]
        and timing["artifacts_sha256"] == timing["artifacts_sha256_after"]
        and timing["gpu"] == timing["gpu_after"] and timing["timing_driver"]["unchanged"] is True
        and timing["resident_configuration"] == timing["resident_configuration_after"]
        == audit["proof_bindings"]["resident_configuration"]
        and timing["resident_math"] == timing["resident_math_after"] == audit["proof_bindings"]["resident_math"]
        and timing["resident_configuration"]["gpu_timing"] is False,
        "Timing actual source/configuration/runtime/hardware/cleanup or attribution contract failed")
    for field in ("fixture_sha256", "reference_sha256", "original_raw_input_sha256"):
        require(timing[field] == timing[field+"_after"] == audit["fixture_binding"][field], "Timing inputs changed")
    require(timing["fixture_arrays_unchanged"] is True and timing["cloud_arrays_unchanged"] is True
        and timing["original_fixture_authority"]["same_decisions_and_support"] is True,
        "Actual native fixture/array authority failed")
    auth = timing["validated_proof_authority"]
    require(auth["bindings_sha256"] == authority.bindings_sha256
        and auth["fixture_binding_sha256"] == authority.fixture_binding_sha256
        and auth["synthetic_report_sha256"] == hashes["synthetic"]
        and auth["bridge_report_sha256"] == hashes["audit"], "Timing does not bind these exact fresh proof reports")
    cpu = [run for run in timing["real_runs"] if run["mode"] == "native_cpu"]
    gpu = [run for run in timing["real_runs"] if run["mode"] == "grid"]
    require(len(cpu) == 1 and cpu[0]["complete"] is True and [run["repeat"] for run in gpu] == [0, 1]
        and len(timing["real_runs"]) == 3, "Require a native control and two resident repeats")
    native = cpu[0]["elapsed_s"]
    require(math.isfinite(native) and native > 0, "Invalid native full-bridge wall time")
    rows, qualities = [], []
    event_fields = {"gpu_transform_ms", "gpu_equations_ms"}
    for run in gpu:
        require(run["complete"] is True and [(p["position"], p["pair"]) for p in run["pairs"]]
            == [(0, [8, 12]), (4, [0, 2])], "Original accepted/rejected pair scope or order changed")
        for pair, task in zip(run["pairs"], audit["fixture_binding"]["tasks"]):
            proposals = pair["proposal_results"]
            require(pair["pair_verdict_same"] is True
                and [p["proposal_index"] for p in proposals] == list(range(len(task["proposal_sha256"])))
                and [p["input_sha256"] for p in proposals] == task["proposal_sha256"]
                and all(p["quality"]["passed"] is True and p["quality"]["same_decision"] is True
                    and (p["evidence"] is None or p["quality"].get("exact_field_differences") == []) for p in proposals),
                "Original proposal order, witnesses, pose/information or decision quality failed")
            qualities.extend(p["quality"] for p in proposals)
        stats, resident = run["statistics_delta"], run["resident_statistics_delta"]
        require(all(stats[field] == 0 for field in ("audited_hits", "audited_misses", "audit_index_mismatches",
            "audit_false_misses", "device_malformed_results"))
            and resident["cpu_fallback_calls"] == 0 and resident["calls"] > 0 and resident["pose_iterations"] > 0,
            "Timing contains CPU shadows, numerical errors or full-call fallback")
        require(not event_fields.intersection(resident)
            and set(run["resident_uncollected_statistics"]) == event_fields,
            "Uncollected event durations were presented as measurements")
        resident_report = run["device_resident"]
        require(resident_report["gpu_timing"] is False and not event_fields.intersection(resident_report["statistics"])
            and set(resident_report["uncollected_statistics"]) == event_fields
            and all(resident_report["provenance"].get(k) is True for k in (
                "source_unchanged", "actual_retrieval_unchanged", "original_resident_math_unchanged")),
            "Actual resident math/retrieval or instrumentation changed")
        # These sequential host scopes partition wall time. GPU work queued in
        # an earlier scope may finish during a later blocking copy/counter scope.
        nn = {"nn_raw_classification_counter_wait_s": stats["device_raw_classify_counter_s"],
            "nn_flagged_packet_copy_wait_s": stats["device_flagged_copy_s"],
            "nn_original_cpu_resolution_s": stats["device_cpu_resolution_s"],
            "nn_correction_enqueue_s": stats["device_correction_enqueue_s"]}
        nn["nn_remaining_host_control_s"] = resident["nn_wall_s"]-sum(nn.values())
        partition = dict(nn, target_dataset_s=stats["dataset_wall_s"],
            source_normal_allocation_upload_enqueue_s=resident["upload_s"],
            transform_enqueue_s=resident["transform_enqueue_s"], equation_enqueue_s=resident["equation_enqueue_s"],
            system_copy_and_queued_gpu_wait_s=resident["system_copy_s"], original_eigen_solve_s=resident["solve_s"],
            final_correspondence_selection_copy_wait_s=resident["final_pair_copy_s"],
            disabled_event_collection_bookkeeping_s=resident["event_collection_s"])
        partition["remaining_resident_host_control_s"] = resident["wall_s"]-sum(partition.values())
        partition["icp_wrapper_s"] = stats["icp_wall_s"]-resident["wall_s"]
        partition["outside_icp_verification_s"] = run["elapsed_s"]-stats["icp_wall_s"]
        require(all(math.isfinite(v) and v >= 0 for v in partition.values())
            and abs(sum(partition.values())-run["elapsed_s"]) < 1e-8, "Invalid disjoint host-wall partition")
        floor = run["elapsed_s"]-resident["nn_wall_s"]
        rows.append({"repeat": run["repeat"], "cache_state": "first" if run["repeat"] == 0 else "warm",
            "full_bridge_wall_s": run["elapsed_s"], "native_over_resident_speed_ratio": native/run["elapsed_s"],
            "resident_over_native_wall_ratio": run["elapsed_s"]/native, "disjoint_host_wall_partition_s": partition,
            "nn_inclusive_wall_s": resident["nn_wall_s"], "non_nn_floor_s": floor,
            "nn_budget_to_equal_native_s": max(0., native-floor),
            "query_rows": stats["query_rows"], "candidate_visits": stats["candidate_visits"],
            "pose_iterations": resident["pose_iterations"], "icp_calls": resident["calls"],
            "device_calls": stats["device_calls"], "original_cpu_query_resolutions": stats["exact_cpu_queries"],
            "index_builds": stats["index_builds"], "cache_hits": stats["cache_hits"],
            "flagged_query_download_rows": stats["device_query_download_rows"],
            "flagged_query_packet_download_bytes": stats["device_query_download_bytes"],
            "counter_download_bytes": stats["device_counter_syncs"]*80,
            "corrected_id_or_metric_upload_bytes": stats["device_correction_upload_bytes"],
            "resident_source_normal_pose_upload_bytes": resident["host_to_device_bytes"],
            "resident_equation_system_and_final_pairs_download_bytes": resident["device_to_host_bytes"]})
    audit_run = next(run for run in audit["real_runs"] if run["mode"] == "grid")
    stats = audit_run["statistics_delta"]
    manifest_path = args.directory/"final-manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_path = args.directory/"source-snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "passed" and manifest["source_files_unchanged"] is True
        and manifest["source_sha256"] == snapshot["source_sha256"] == audit["source_sha256"]
        and manifest["artifacts_sha256"] == snapshot["artifacts_sha256"] == audit["artifacts_sha256"]
        and manifest["source_snapshot_sha256"] == digest(snapshot_path)
        and manifest["hardware_jobs_complete"] is True and manifest["hardware_released_to_parent"] is True,
        "Final source/execution closure manifest failed")
    executions = {}
    for name, phase in (("synthetic", "synthetic-proof"), ("audit", "bridge-audit"), ("timing", "timing")):
        row = manifest["phases"][phase]
        execution_path, log_path = args.directory/(phase+"-execution.json"), args.directory/(phase+".log")
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        require(row["report_sha256"] == execution["report_sha256"] == hashes[name]
            and row["execution_sha256"] == digest(execution_path)
            and row["log_sha256"] == execution["log_sha256"] == digest(log_path)
            and execution["runner_sha256"] == digest(args.directory/"run-phase.py")
            and row["exit_code"] == execution["exit_code"] == 0 and execution["child_wait_completed"] is True
            and execution["status"] == "exited" and execution["phase"] == phase
            and row["child_pid"] == execution["child_pid"] and row["command"] == execution["command"]
            and row["end_utc"] == execution["end_utc"] and row["elapsed_s"] == execution["elapsed_s"]
            and row["source_input_runtime_cleanup_gates_passed"] is True
            and execution["environment_overrides"] == {"OMP_NUM_THREADS": "8", "PYTHONUTF8": "1"},
            "Actual exit/wait/command/environment/report/log/runner closure disagrees")
        executions[name] = {"metadata_path": str(execution_path.resolve()), "metadata_sha256": row["execution_sha256"],
            "log_path": str(log_path.resolve()), "log_sha256": row["log_sha256"],
            "child_pid": execution["child_pid"], "exit_code": 0, "child_wait_completed": True,
            "command": execution["command"], "process_wall_s": execution["elapsed_s"],
            "process_wall_scope": "Includes process/import/setup/report/cleanup time; component bridge wall is separate."}
    faster = all(row["full_bridge_wall_s"] < native for row in rows)
    summary = {"kind": "device-flat-resident-bridge-research-summary", "status": "passed",
        "producer": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(__file__)},
        "claim": ("Fresh device-resident synthetic and full-trajectory CPU hit/miss proofs pass; both full component timings are faster than contemporary native CPU."
            if faster else "Fresh device-resident synthetic and full-trajectory CPU hit/miss proofs pass; the two full component timings do not establish a repeatable improvement over contemporary native CPU."),
        "whole_archive_speedup_measured": False, "production_promotion": False,
        "reports": {name: {"path": str(path.resolve()), "sha256": hashes[name]} for name, path in paths.items()},
        "execution_manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_hash,
            "source_snapshot_sha256": manifest["source_snapshot_sha256"], "phases": executions,
            "hardware_jobs_complete": True, "hardware_released_to_parent": True},
        "proof_bindings": audit["proof_bindings"], "fixture_binding": audit["fixture_binding"],
        "native_cpu_full_bridge_s": native, "resident_runs": rows,
        "audit": {"query_rows": stats["query_rows"], "direct_hits": stats["direct_gpu_hits"],
            "audited_hits": stats["audited_hits"], "direct_misses": stats["declared_gpu_misses"],
            "audited_misses": stats["audited_misses"], "original_cpu_resolutions": stats["exact_cpu_queries"],
            "index_mismatches": stats["audit_index_mismatches"], "false_misses": stats["audit_false_misses"],
            "candidate_visits": stats["candidate_visits"], "target_membership_count": len(authority.target_digests),
            "wall_s": audit_run["elapsed_s"], "cpu_shadow_s": stats["cpu_audit_s"],
            "all_query_packet_rows": stats["device_query_download_rows"],
            "performance_attribution_valid": False},
        "quality": {"all_nine_original_gates_per_run_passed": True,
            "max_pose_coefficient_difference": max(q["max_transform_entry_delta"] for q in qualities
                if q.get("max_transform_entry_delta") is not None),
            "max_information_matrix_difference": max(q["max_information_delta"] for q in qualities
                if q.get("max_information_delta") is not None),
            "metric_scope": "Available accepted-proposal matrices; rejected None witnesses have no matrix metric."},
        "solver_setup": timing["setup_partition"], "resident_configuration": timing["resident_configuration"],
        "memory": {"retained_grid_and_original_xyz_cap_bytes": timing["cache_policy"]["retained_gpu_bytes"],
            "peak_retained_grid_and_original_xyz_bytes": timing["statistics"]["peak_retained_gpu_bytes"],
            "peak_query_scratch_bytes": timing["statistics"]["peak_device_query_bytes"],
            "peak_combined_resident_and_query_scratch_bytes": timing["device_resident"]["statistics"]["peak_combined_resident_and_query_scratch_bytes"],
            "peak_process_rss_bytes": timing["peak_process_rss_bytes"],
            "scope": "Separate retained/index and per-query/resident caps; external queries, host packets and CuPy pools are not a total memory cap."},
        "limits": "Raw-derived fixed component with measured local poses; accepted/rejected nine-proposal gates do not establish complete Finish/live tracking/mesh quality or throughput. Resident reductions can change roundoff. Host-wait partitions are not GPU kernel durations; queued work can complete in a later scope. CUDA events are disabled/uncollected. Native internal NN/solve split is unexposed. A whole-archive experiment needs separate quality and latency proof.",
        "cleanup_passed": True}
    require(hashes == {name: digest(path) for name, path in paths.items()}, "Measured reports changed while deriving summary")
    require(digest(manifest_path) == manifest_hash, "Closure manifest changed while deriving summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": "passed", "native_s": native,
        "resident_s": [row["full_bridge_wall_s"] for row in rows]}))


if __name__ == "__main__":
    main()
