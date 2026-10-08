"""Pure-stdlib flat traversal proof validator; original proofs cannot authorize it.

Reuses the original strict artifact/runtime/domain/input and complete nine-gate
contract, requiring the separate flat report kind, traversal binding and cache
ownership proof. Numerical/device imports and source mutations never occur.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import json
from dataclasses import dataclass
from scripts.research.archive.validate_uniform_grid_proof import (
    GridProofAuthority, GridProofError, canonical_hash, file_hash, require,
    positive, zero, audit_coverage, validate_path_resolution_artifacts, common_checks as original_common_checks)


@dataclass(frozen=True)
class FlatGridProofAuthority(GridProofAuthority):
    """Distinct type prevents an old serial/staged proof selecting flat math."""


def common_checks(report, expected_bindings):
    require(report.get("kind") == "standalone-original-double-flat-grid", "Require a separate measured flat-grid report")
    require(report.get("traversal") == "flat-full-radius-v1", "Flat traversal contract absent")
    require(expected_bindings.get("domain", {}).get("version") == "flat-dyadic27-original-double-v1",
            "Serial/staged grid bindings cannot authorize flat traversal")
    normalized = dict(report, kind="standalone-original-double-uniform-grid")
    original_common_checks(normalized, expected_bindings)
    require(report.get("proof_bindings_after") == expected_bindings, "Installed CPU/CUDA runtime or actual settings changed")


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
        owned = synthetic.get("flat_cache_ownership", {})
        require(owned.get("passed") is True and positive(owned.get("cloud_evictions"))
                and owned.get("bounded_retained_bytes") is True and owned.get("clear_releases_all_owned_arrays") is True
                and owned.get("one_byte_budget_original_cpu_fallback") is True,
                "Flat inherited cache eviction/reset/budget ownership checks incomplete")
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
        require(all(run.get("mode") in ("native_cpu", "grid") for run in runs), "Unknown real proof run mode")
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
        return FlatGridProofAuthority(canonical_hash(expected_bindings), canonical_hash(expected_fixture_binding), before[0], before[1],
                                  frozenset(targets), tuple(sorted(expected_bindings["artifacts_sha256"].items())))
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        raise GridProofError(f"Incomplete/malformed proof artifact: {error}") from error
