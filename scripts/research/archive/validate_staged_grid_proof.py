"""Pure-stdlib validation of separate staged/pruned grid proof artifacts.

This validates recorded evidence and exact runtime/input scope; it does not
claim universal mathematical/software compatibility or modify scanner policy.
No numerical imports, device initialization or builds occur in this module.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import hashlib
import json
from dataclasses import dataclass


class GridProofError(ValueError):
    pass


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GridProofAuthority:
    bindings_sha256: str
    fixture_binding_sha256: str
    synthetic_report_sha256: str
    bridge_report_sha256: str
    target_digests: frozenset
    artifact_sha256: tuple


def require(condition, reason):
    if not condition:
        raise GridProofError(reason)


def positive(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def zero(value):
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


PATH_RESOLUTION_ARTIFACTS = ("scripts/tool_paths.py", "scripts/tool-catalog.json")


def validate_path_resolution_artifacts(mapping):
    """Relocated research must bind the resolver and the catalog it consumes."""
    require(isinstance(mapping, dict), "Path resolution artifact fingerprints absent")
    require(all(isinstance(name, str) for name in mapping), "Malformed path resolution artifact name")
    normalized = {name.replace("\\", "/"): digest for name, digest in mapping.items()}
    require(len(normalized) == len(mapping), "Aliased path resolution artifact names")
    for name in PATH_RESOLUTION_ARTIFACTS:
        path = ROOT / name
        require(path.is_file() and normalized.get(name) == file_hash(path),
                f"Current path resolver/catalog is missing or changed: {name}")


def audit_coverage(stats):
    require(positive(stats.get("query_rows")) and positive(stats.get("direct_gpu_hits"))
            and positive(stats.get("declared_gpu_misses")) and positive(stats.get("audited_hits"))
            and positive(stats.get("audited_misses")), "Proof lacks positive query/hit/miss coverage")
    require(stats["query_rows"] >= stats["direct_gpu_hits"]+stats["declared_gpu_misses"], "Audit coverage exceeds actual query rows")
    require(stats.get("direct_gpu_hits") == stats.get("audited_hits"), "Not every direct hit was CPU audited")
    require(stats.get("declared_gpu_misses") == stats.get("audited_misses"), "Not every direct miss was CPU audited")
    require(zero(stats.get("audit_index_mismatches")) and zero(stats.get("audit_false_misses")), "Proof contains native nearest errors")
    require(positive(stats.get("stage0_early_certificates")) and positive(stats.get("stage1_early_certificates")), "Both narrow-stage certificate branches need positive proof coverage")
    require(positive(stats.get("pruned_candidates")) and positive(stats.get("double_evaluations")), "Conservative pruning and original FP64 branches both need positive coverage")
    require(stats["candidate_visits"] == stats["pruned_candidates"]+stats["double_evaluations"], "Visit/prune/FP64 accounting differs")
    for stage in range(3):
        require(stats.get(f"stage{stage}_candidate_visits") == stats.get(f"stage{stage}_pruned_candidates",-1)+stats.get(f"stage{stage}_double_evaluations",-1), "Per-stage evaluation accounting differs")
        require(positive(stats.get(f"stage{stage}_rows")) and stats[f"stage{stage}_rows"]<=stats["query_rows"], "Stage execution coverage invalid")
        if stage<2:
            require(stats[f"stage{stage}_early_certificates"]<=stats[f"stage{stage}_rows"], "Early certificate count exceeds executed rows")
    require(stats["candidate_visits"]==sum(stats[f"stage{stage}_candidate_visits"] for stage in range(3)), "Per-stage visits do not cover total visits")


def common_checks(report, expected_bindings):
    validate_path_resolution_artifacts(expected_bindings["artifacts_sha256"])
    require(report.get("kind") == "standalone-staged-pruned-original-double-uniform-grid" and report.get("status") == "passed", "Require a successful measured grid report")
    require(report.get("performance_attribution_valid") is False, "Numerical proof must be fully audited, not a performance report")
    require(report.get("cpu_hit_audit") is True and report.get("cpu_miss_audit") is True, "Both explicit CPU audit flags are required")
    require(report.get("miss_policy") == "direct-miss-research-v1", "Require the audited complete-radius miss policy")
    require(report.get("proof_bindings") == expected_bindings and report.get("proof_bindings_sha256") == canonical_hash(expected_bindings), "Current CUDA/CPU/source/domain/policy binding differs")
    require(report.get("source_sha256") == expected_bindings["source_sha256"] == report.get("source_sha256_after"), "Core source changed")
    require(report.get("component_source_sha256") == expected_bindings["component_source_sha256"] == report.get("component_source_sha256_after"), "Verification math changed")
    require(report.get("artifacts_sha256") == expected_bindings["artifacts_sha256"] == report.get("artifacts_sha256_after"), "Helper/kernel/harness bytes changed")
    require(report.get("gpu") == expected_bindings["gpu"] == report.get("gpu_after"), "GPU/driver identity changed")
    require(report.get("domain") == expected_bindings["domain"], "Dyadic signed-domain definition changed")
    require(report.get("cleanup_passed") is True and not report.get("failure") and not report.get("cleanup_failures"), "Proof setup/work/cleanup failed")
    host = report.get("host_enclosure", {})
    require(host.get("unsupported_guard_passed") is True and positive(host.get("packing_unique_cases")), "Host domain/packing guards did not pass")
    cases = host.get("domain_boundary_cases", [])
    require(len(cases) >= 11 and all(case.get("passed") is True and positive(case.get("supported_boundary_pairs")) for case in cases), "Host enclosure boundary coverage incomplete")
    require({case.get("radius") for case in cases}.issuperset({2.**-20, .0075, .015, .03, .06, .12, .125, .25, .5, 1.}), "Required inner-stage/dyadic radius bounds absent")


def validate_grid_proof(synthetic_path, bridge_path, expected_bindings, expected_fixture_binding):
    synthetic_path, bridge_path = Path(synthetic_path).resolve(), Path(bridge_path).resolve()
    require(synthetic_path != bridge_path, "Synthetic and real bridge proof must be separate artifacts")
    before = (file_hash(synthetic_path), file_hash(bridge_path))
    try:
        synthetic = json.loads(synthetic_path.read_text())
        bridge = json.loads(bridge_path.read_text())
        common_checks(synthetic, expected_bindings)
        common_checks(bridge, expected_bindings)
        require(synthetic.get("synthetic_only") is True and not synthetic.get("real_runs"), "Require dedicated synthetic proof")
        require(synthetic.get("synthetic_device_adapter_checked") is True, "Separate host/device-adapter proof missing")
        cases = synthetic.get("synthetic_indices", [])
        require(len(cases) >= 16 and all(positive(case.get("queries")) and zero(case.get("index_mismatches"))
                and zero(case.get("device_adapter_index_mismatches")) and isinstance(case.get("device_adapter_max_squared_distance_delta"), (int,float))
                and 0 <= case["device_adapter_max_squared_distance_delta"] <= 1e-12 for case in cases), "Synthetic host/device nearest proof incomplete")
        boundaries = synthetic.get("synthetic_grid_boundaries", [])
        require(len(boundaries) >= 43 and all(positive(case.get("queries")) and zero(case.get("index_mismatches")) for case in boundaries), "Signed/subnormal/min-radius device proof incomplete")
        require(any(case.get("case") == "original-double-subnormal-and-minimum-radius" for case in boundaries), "Subnormal native evidence absent")
        audit_coverage(synthetic.get("synthetic_statistics", {}))
        intervals=synthetic.get("synthetic_interval_guards",[])
        require(len(intervals)>=7 and all(positive(case.get("queries")) and zero(case.get("index_mismatches")) for case in intervals), "Restricted interval/tie/inner-stage boundary device proof incomplete")
        require(any(case.get("inner_query_unsupported_full_supported") is True for case in intervals), "Uniform unsupported-inner-stage synchronization proof missing")
        require(synthetic.get("synthetic_budget_fallback",{}).get("passed") is True, "Missing final/budget CPU-only fallback proof incomplete")
        ownership=synthetic.get("synthetic_ownership_guards",{})
        require(ownership.get("passed") is True and ownership.get("absent_final_kernel_guard_passed") is True
                and ownership.get("four_table_cap_exercised") is True
                and positive(ownership.get("grid_slot_evictions")) and positive(ownership.get("pointer_table_evictions"))
                and positive(ownership.get("global_cache_evictions")) and ownership.get("peak_grids_per_cloud",99)<=5
                and ownership.get("peak_tables_per_cloud",99)<=4, "Five-grid/table/global-cache eviction and incomplete-final fallback proof missing")
        require(bridge.get("synthetic_only") is False, "Require a separate actual bridge proof")
        require(bridge.get("fixture_binding") == expected_fixture_binding and bridge.get("fixture_binding_sha256") == canonical_hash(expected_fixture_binding), "Fixture/raw input/task/proposal membership differs")
        require(bridge.get("fixture_sha256") == expected_fixture_binding["fixture_sha256"] == bridge.get("fixture_sha256_after"), "Fixture changed")
        require(bridge.get("reference_sha256") == expected_fixture_binding["reference_sha256"] == bridge.get("reference_sha256_after"), "Reference changed")
        require(bridge.get("original_raw_input_sha256") == expected_fixture_binding["original_raw_input_sha256"] == bridge.get("original_raw_input_sha256_after"), "Raw archive changed")
        require(bridge.get("fixture_arrays_unchanged") is True and bridge.get("cloud_arrays_unchanged") is True, "Original arrays changed")
        require(bridge.get("fixture_provenance", {}).get("local_pose_source") == "measured Finish fragment report; no archived ZIP poses", "Archived/unproven pose fixture is not supported")
        require(bridge.get("original_fixture_authority", {}).get("same_decisions_and_support") is True, "Native fixture authority differs")
        runs = bridge.get("real_runs", [])
        cpu = [run for run in runs if run.get("mode") == "native_cpu"]
        gpu = [run for run in runs if run.get("mode") == "grid"]
        require(len(cpu) == 1 and cpu[0].get("complete") is True and gpu, "Native baseline or complete GPU runs missing")
        wanted_tasks = expected_fixture_binding["tasks"]
        require([(task["position"], task["pair"], len(task["proposal_sha256"])) for task in wanted_tasks] == [(0,[8,12],5),(4,[0,2],4)], "Require the actual accepted/rejected nine-proposal membership")
        for run in runs:
            require(run.get("complete") is True and len(run.get("pairs", [])) == 2, "Incomplete pair run")
            for pair, task in zip(run["pairs"], wanted_tasks):
                require(pair.get("position") == task["position"] and pair.get("pair") == task["pair"], "Changed original pair order")
                proposals = pair.get("proposal_results", [])
                require([p.get("proposal_index") for p in proposals] == list(range(len(task["proposal_sha256"])))
                        and [p.get("input_sha256") for p in proposals] == task["proposal_sha256"], "Changed competing proposal order/input")
                require(pair.get("verdict", {}).get("accepted") is (task["position"] == 0), "Accepted/rejected task decision changed")
                require(pair.get("pair_verdict_same") is True, "Ambiguity or verified-proposal evidence changed")
                if run["mode"] == "grid":
                    require(all(p.get("quality", {}).get("passed") is True for p in proposals), "Original proposal witnesses/validation/pose/information proof failed")
            if run["mode"] == "grid":
                audit_coverage(run.get("statistics_delta", {}))
        targets = bridge.get("real_target_membership_sha256", [])
        require(targets and len(set(targets)) == len(targets)
                and all(isinstance(digest,str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) for digest in targets), "Real target point-byte membership absent")
        after = (file_hash(synthetic_path), file_hash(bridge_path))
        require(before == after, "Proof files changed during validation")
        validate_path_resolution_artifacts(expected_bindings["artifacts_sha256"])
        return GridProofAuthority(canonical_hash(expected_bindings), canonical_hash(expected_fixture_binding), before[0], before[1],
                                  frozenset(targets), tuple(sorted(expected_bindings["artifacts_sha256"].items())))
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        raise GridProofError(f"Incomplete/malformed proof artifact: {error}") from error
