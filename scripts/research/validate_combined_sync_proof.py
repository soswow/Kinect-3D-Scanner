"""Distinct stdlib-only component authority for fully audited combined copies.

Old Device/Finish/Bulk tokens do not authorize this orchestration. A returned
token is deliberately not a subclass of any old token. This first validator
records component evidence only; no current helper accepts unaudited timing.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from scripts.research import validate_device_flat_grid_proof as original
from scripts.research.research_combined_sync_icp import source_contract
from scripts.research.combined_sync_synthetic import REQUIRED_CHECKS
from scripts.research.archive.validate_uniform_grid_proof import (
    GridProofError, canonical_hash, file_hash, require, positive, zero)

ROOT = Path(__file__).resolve().parents[2]
KIND = "standalone-original-double-combined-sync-resident"
TRAVERSAL = "combined-counter-normal-single-copy-v1"
EXPECTED_QUERIES = 104123989
NEW_ARTIFACTS = (
    "scripts/research/research_combined_sync_icp.py", "scripts/research/combined_sync_synthetic.py",
    "scripts/research/benchmark_combined_sync_icp.py", "scripts/research/validate_combined_sync_proof.py",
    "tests/test_combined_sync_contract.py", "scripts/research/cuda_bulk_audit_grid_registration.py",
    "scripts/research/bulk_legacy_nn_audit.py", "scripts/research/benchmark_bulk_resident_audit.py",
    "scripts/research/validate_bulk_resident_audit.py", "scripts/research/finish_resident_registration.py",
    "scripts/research/validate_finish_resident_proof.py")


@dataclass(frozen=True)
class CombinedSyncProofAuthority:
    """Independent immutable component-only evidence; no Finish/timing promotion."""
    synthetic_report_path: str
    synthetic_report_sha256: str
    bridge_report_path: str
    bridge_report_sha256: str
    runtime_binding_json: str
    fixture_binding_sha256: str
    target_digests: frozenset
    artifact_sha256: tuple

    @property
    def runtime_binding(self):
        return json.loads(self.runtime_binding_json)


def combined_counts(stats, query_rows):
    require(type(query_rows) is int and query_rows > 0, "Actual combined queries absent")
    integer_keys = ("iteration_calls", "primary_counter_term_syncs", "primary_download_bytes",
        "provisional_queries", "provisional_candidate_visits", "common_calls", "common_queries",
        "flagged_calls", "flagged_queries", "discarded_placeholder_calls", "malformed_results",
        "full_query_audited_calls", "full_query_audited_rows", "id_bit_comparison_rows",
        "metric_bit_comparison_rows", "filtered_bit_comparison_rows", "term_bit_comparison_calls",
        "equation_bit_mismatches", "nearest_bit_mismatches", "peak_combined_owned_query_bytes")
    require(all(type(stats.get(key)) is int and stats[key] >= 0 for key in integer_keys),
            "Malformed combined integer counters")
    calls = stats["iteration_calls"]
    require(positive(calls) and stats["provisional_queries"] == stats["full_query_audited_rows"] == query_rows,
            "Every actual changed trajectory query needs its complete original CPU shadow")
    require(stats["primary_counter_term_syncs"] == stats["full_query_audited_calls"] == calls
            and stats["primary_download_bytes"] == 320 * calls,
            "Combined counter/normal transport was not exactly one320-byte copy per iteration")
    require(stats["common_calls"] + stats["flagged_calls"] == calls
            and stats["discarded_placeholder_calls"] == stats["flagged_calls"]
            and stats["term_bit_comparison_calls"] == stats["common_calls"]
            and positive(stats["common_calls"]) and positive(stats["common_queries"])
            and stats["common_queries"] <= query_rows and stats["flagged_queries"] <= query_rows - stats["common_queries"],
            "Original common math or CPU-resolution-before-equation coverage incomplete")
    require(stats["id_bit_comparison_rows"] == stats["metric_bit_comparison_rows"]
            == stats["filtered_bit_comparison_rows"] == stats["common_queries"],
            "Every common ID/metric/filter bit was not checked")
    require(all(zero(stats[key]) for key in ("malformed_results", "nearest_bit_mismatches", "equation_bit_mismatches")),
            "Combined numerical/transport failure was hidden")


def combined_report_checks(report, binding, artifacts):
    metadata = report["combined_sync"]
    require(metadata.get("policy") == binding["combined_sync_contract"]["policy"]
            and metadata.get("source_contract") == binding["combined_sync_contract"]
            and metadata.get("audit_only") is True and metadata.get("new_timing_authority") is False
            and metadata.get("graph_capture") is False and report.get("performance_attribution_valid") is False,
            "Combined research policy or attribution changed")
    require(metadata.get("source_sha256") == metadata.get("source_sha256_before")
            == artifacts["scripts/research/research_combined_sync_icp.py"] and metadata.get("source_unchanged") is True,
            "Actual combined helper source changed")


def focused_checks(value):
    require(value.get("passed") is True and set(value.get("required_checks", [])) == set(REQUIRED_CHECKS)
            and all(value.get(key) is True for key in REQUIRED_CHECKS), "New focused combined guard/math proof absent")
    cases = value.get("cases", [])
    expected = ("common_hit_equation_bits", "common_miss_equation_bits", "partial_block_equation_bits",
        "tie_cpu_resolution_before_equations", "strict_boundary_cpu_resolution_before_equations",
        "unsupported_cpu_resolution_before_equations", "malformed_hard_fault_before_cpu")
    require([row.get("case") for row in cases] == list(expected), "Focused common/flagged/malformed case scope changed")
    for row in cases:
        require(row.get("complete") is True and row.get("inputs_unchanged") is True
                and row.get("owned_scratch_released") is True and not row.get("failure")
                and not row.get("cleanup_failures"), "Focused combined input/ownership/failure closure absent")
        stats = row["statistics"]
        require(stats.get("primary_counter_term_syncs") == 1 and stats.get("primary_download_bytes") == 320,
                "Focused primary transfer differs")
        if row["expected_branch"] == "common":
            combined_counts(stats, row["queries"])
        elif row["expected_branch"] == "flagged":
            require(stats.get("flagged_calls") == stats.get("discarded_placeholder_calls") == 1
                    and stats.get("full_query_audited_rows") == row["queries"]
                    and zero(stats.get("common_calls")) and zero(stats.get("term_bit_comparison_calls")),
                    "Focused CPU-resolution guard did not discard all placeholders")
        else:
            require(row["expected_branch"] == "malformed" and stats.get("malformed_results") == 1
                    and zero(stats.get("full_query_audited_calls")) and row.get("expected_hard_fault"),
                    "Malformed numerical state reached CPU resolution")


def common_checks(report, binding, artifacts, manifest):
    require(report.get("kind") == KIND and report.get("traversal") == TRAVERSAL,
            "Old device/Finish/bulk or kernel-only proof cannot authorize combined synchronization")
    # Reuse original closed math/input/resource predicates, never the old token.
    original.common_checks(dict(report, kind=original.KIND, traversal=original.TRAVERSAL), binding, artifacts, manifest)
    require(report.get("performance_attribution_valid") is False and report.get("new_whole_finish_authority") is False
            and not any(report.get(key) for key in ("failure", "producer_current_failure", "cleanup_failures", "full_icp_shadow_failure")),
            "Combined proof hid failure or promoted diagnostic wall time")
    require(report.get("cpu_audit_proof") == report.get("cpu_audit_proof_after") == binding["cpu_audit_proof"]
            and report.get("cpu_audit_authority_unchanged") is True, "Complete CPU auditor authority changed")
    audit = binding["cpu_audit_proof"]
    require(file_hash(Path(audit["path"])) == audit["sha256"], "Closed CPU auditor proof file changed")
    # Independently reconstruct the CPU-only token, rather than trusting the
    # producer's success Boolean or minting an old GPU/Finish authority.
    from scripts.research.validate_bulk_resident_audit import validate_bulk_resident_audit
    dual = json.loads(Path(audit["path"]).read_text(encoding="utf-8"))
    authority = validate_bulk_resident_audit(audit["path"], dual["proof_bindings"], dual["fixture_binding"])
    require(binding["bulk_cpu_runtime"] == authority.runtime_binding["bulk_cpu_runtime"],
            "Combined queries did not use the proven original CPU auditor/native runtime")
    require(report.get("cpu_audit_authority") == {"api_path": authority.api_report_path,
                "api_sha256": authority.api_report_sha256, "binding_sha256": canonical_hash(authority.runtime_binding)},
            "Referenced full dual/native API authority differs")
    combined_report_checks(report["device_resident"], binding, artifacts)
    focused_checks(report["combined_sync_synthetic"])
    require(report.get("original_nine_loop_binding") == report.get("original_nine_loop_binding_after")
            and report.get("original_nine_loop_binding", {}).get("original_nine_proposal_loop_unchanged") is True,
            "Original proposal verification loop source changed")


def validate_combined_sync_proof(synthetic_path, bridge_path, expected_binding, expected_fixture):
    synthetic_path, bridge_path = Path(synthetic_path).resolve(), Path(bridge_path).resolve()
    require(synthetic_path != bridge_path, "New synthetic and real proofs must be separate files")
    before = file_hash(synthetic_path), file_hash(bridge_path)
    try:
        artifacts = original.validate_configuration(expected_binding)
        require(set(NEW_ARTIFACTS).issubset(artifacts), "Combined orchestration/CPU auditor dependency missing")
        require(expected_binding.get("combined_sync_contract") == source_contract(),
                "Actual guard-prefix/math/source/compiler contract differs")
        manifest = original.validate_current_artifacts(expected_binding, artifacts)
        synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        for report in (synthetic, bridge):
            common_checks(report, expected_binding, artifacts, manifest)
        require(synthetic.get("synthetic_only") is True and not synthetic.get("real_runs")
                and synthetic.get("synthetic_device_adapter_checked") is True,
                "Dedicated fresh combined synthetic proof absent")
        from scripts.research.validate_bulk_resident_audit import validate_dual_synthetics
        validate_dual_synthetics(synthetic)  # Original actual NN/cache/packet predicates only.
        require(bridge.get("synthetic_only") is False and bridge.get("fixture_binding") == expected_fixture
                and bridge.get("fixture_binding_sha256") == canonical_hash(expected_fixture), "Actual raw fixture/proposals differ")
        for field in ("fixture_sha256", "reference_sha256", "original_raw_input_sha256"):
            require(bridge.get(field) == expected_fixture[field] == bridge.get(field + "_after"), "Original inputs changed")
        require(bridge.get("fixture_arrays_unchanged") is True and bridge.get("cloud_arrays_unchanged") is True
                and bridge.get("original_fixture_authority", {}).get("same_decisions_and_support") is True,
                "Original immutable input/support authority absent")
        tasks = expected_fixture["tasks"]
        require([(row["position"], row["pair"], len(row["proposal_sha256"])) for row in tasks]
                == [(0, [8, 12], 5), (4, [0, 2], 4)], "Require original accepted/rejected all-nine component scope")
        runs = bridge["real_runs"]
        require([row.get("mode") for row in runs] == ["native_cpu", "grid"] and all(row.get("repeat") == 0 for row in runs),
                "Require one contemporary original CPU and one fully audited changed trajectory")
        for run in runs:
            require(run.get("complete") is True and len(run.get("pairs", [])) == 2, "Component run incomplete")
            for pair, task in zip(run["pairs"], tasks):
                require(pair.get("position") == task["position"] and pair.get("pair") == task["pair"]
                        and pair.get("pair_verdict_same") is True and pair.get("verdict", {}).get("accepted") is (task["position"] == 0),
                        "Original pair order/decision/support changed")
                rows = pair["proposal_results"]
                require([row.get("proposal_index") for row in rows] == list(range(len(task["proposal_sha256"])))
                        and [row.get("input_sha256") for row in rows] == task["proposal_sha256"], "Original proposal order/bytes changed")
                if run["mode"] == "grid":
                    require(all(row.get("quality", {}).get("passed") is True for row in rows),
                            "Original pose/witness/heldout/information gates failed")
            if run["mode"] == "grid":
                stats = run["statistics_delta"]
                original.device_audit_coverage(stats, complete_trajectory=True)
                require(stats["query_rows"] == EXPECTED_QUERIES, "Complete unchanged original query trajectory not audited")
                require(stats["device_calls"] == 11569 and run["resident_statistics_delta"]["calls"] == 236
                        and run["resident_statistics_delta"]["pose_iterations"] == 10861,
                        "Original complete call/iteration/NN order scope changed")
                original.resident_checks(run, expected_binding, artifacts)
                combined_report_checks(run["device_resident"], expected_binding, artifacts)
                combined_counts(run["combined_statistics_delta"], EXPECTED_QUERIES)
                from scripts.research.validate_finish_resident_proof import validate_shadow
                rows = bridge["full_icp_shadows"]
                require(run.get("full_icp_shadow_range") == [0,len(rows)] and len(rows) == run["resident_statistics_delta"]["calls"],
                        "Every original actual ICP call needs its complete original CPU result shadow")
                for index,row in enumerate(rows):
                    require(row.get("index") == index and row.get("complete") is True and not row.get("failure"),
                            "Original ICP result shadow was incomplete or reordered")
                    validate_shadow(row)
                require(run["combined_statistics_delta"]["peak_combined_owned_query_bytes"]
                        <= expected_binding["resident_configuration"]["max_scratch_bytes"], "Combined scratch exceeded its budget")
        targets = bridge["real_target_membership_sha256"]
        require(targets and len(set(targets)) == len(targets) and all(original.digest(value) for value in targets),
                "Actual immutable target membership absent")
        original.validate_current_artifacts(expected_binding, artifacts)
        require(before == (file_hash(synthetic_path), file_hash(bridge_path)), "New combined proof files changed")
        return CombinedSyncProofAuthority(str(synthetic_path), before[0], str(bridge_path), before[1],
            json.dumps(expected_binding, sort_keys=True, separators=(",", ":"), allow_nan=False),
            canonical_hash(expected_fixture), frozenset(targets), tuple(sorted(artifacts.items())))
    except GridProofError:
        raise
    except (KeyError, TypeError, IndexError, ValueError, AttributeError, OSError) as error:
        raise GridProofError(f"Incomplete/malformed/stale combined proof: {error}") from error
