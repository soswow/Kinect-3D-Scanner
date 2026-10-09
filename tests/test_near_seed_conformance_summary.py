"""Scalar publication phase/refusal contracts; no numerical imports."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from scripts.research import summarize_near_seed_conformance as publisher
from tests.test_near_seed_reuse import closed_audit, evidence


def phase():
    audit, _, captures = closed_audit()
    value = audit["rounds"][0]["native"]
    item = copy.deepcopy(value["pairs"][0]["proposals"][0]["calls"][0])
    value["pairs"][0]["proposals"] = [{"proposal_index": 0, "calls": [dict(item, call_index=i) for i in range(714)],
        "gates": [{"complete": True}], "complete": True}]
    value["call_count"] = 714
    value["cold_whole_phase_wall_s"] = 2.; value["whole_phase_wall_s"] = 1.9
    value["pairs"][0]["inclusive_pair_wall_s"] = 1.8
    task = captures[0]["tasks"][0]; task["proposal_count"] = 1
    return value, captures


class SummaryTests(unittest.TestCase):
    def test_closed_phase_requires_every_call_gate_input_cleanup_and_clock(self):
        value, captures = phase()
        self.assertEqual(len(publisher.phase_closed(value, captures)), 714)
        for key in ("call", "gate", "input", "cleanup", "clock", "count", "nested"):
            bad = copy.deepcopy(value)
            if key == "call": bad["pairs"][0]["proposals"][0]["calls"][-1]["complete"] = False
            elif key == "gate": bad["pairs"][0]["proposals"][0]["gates"][0]["complete"] = False
            elif key == "input": bad["pairs"][0]["seed_bytes_unchanged"] = False
            elif key == "cleanup": bad["cleanup_passed"] = False
            elif key == "clock": bad["cold_whole_phase_wall_s"] = float("nan")
            elif key == "count": bad["call_count"] = 713
            else: bad["cold_whole_phase_wall_s"] = .1
            with self.assertRaises(publisher.scope.BridgeFailure): publisher.phase_closed(bad, captures)

    def test_strict_refusal_is_original_cpu_first_miss_not_a_reused_hit(self):
        with tempfile.TemporaryDirectory() as folder:
            audit, binding, captures = closed_audit(); ap = Path(folder)/"audit.json"; fp = Path(folder)/"failure.json"
            binding["artifacts_sha256"] = dict(publisher.driver.V1_PINS); audit["binding_after"] = copy.deepcopy(binding)
            for _, path in publisher.strict.LoadedMethodGuard().modules:
                binding["artifacts_sha256"][path.relative_to(publisher.ROOT).as_posix()] = publisher.scope.sha(path)
            audit["binding_after"] = copy.deepcopy(binding)
            ap.write_text(json.dumps(audit), encoding="utf-8"); sha = hashlib.sha256(ap.read_bytes()).hexdigest()
            old = publisher.driver.phase_calls(audit["rounds"][0]["cache"])[0]
            actual = copy.deepcopy(old); actual["result"] = evidence(1e-16); actual["complete"] = False
            failure = {"kind": publisher.strict.TIMING_KIND, "status": "failed", "failure": {"type": "RuntimeError"},
                "cleanup_passed": True, "cleanup_failures": [], "binding": binding, "binding_after": binding, "captures": captures,
                "audit_proof": {"sha256": sha}, "rounds": [{"cache": {"pairs": [{"proposals": [{"calls": [actual]}]}],
                    "cache": {"closed": True, "statistics": {"calls": 1, "misses": 1, "original_cpu_calls": 1, "hits": 0, "audited_hits": 0}}}}]}
            fp.write_text(json.dumps(failure), encoding="utf-8")
            result = publisher.strict_failure(ap, fp)
            self.assertEqual(result["reused_calls"], 0); self.assertEqual(result["pose_max_abs_delta"], 1e-16)
            self.assertFalse(result["independent_process_exit_observed"])
            failure["rounds"][0]["cache"]["cache"]["statistics"]["hits"] = 1
            fp.write_text(json.dumps(failure), encoding="utf-8")
            with self.assertRaises(publisher.scope.BridgeFailure): publisher.strict_failure(ap, fp)

    def test_producer_pin_mismatch_refused_before_any_input_read(self):
        with patch.object(publisher, "PINS", {"scripts/research/benchmark_near_seed_conformance.py": "bad"}):
            with self.assertRaises(publisher.scope.BridgeFailure): publisher.compact(None, None, None, None)

    def test_recorded_producer_source_pins_are_required_not_only_current_files(self):
        pins = dict(publisher.driver.V1_PINS, **publisher.PINS)
        publisher.artifact_receipt({"artifacts_sha256": pins}, pins)
        missing = dict(pins); missing.pop(next(iter(missing)))
        for binding in ({}, {"artifacts_sha256": missing}, {"artifacts_sha256": dict(pins, **{next(iter(pins)): "wrong"})}):
            with self.assertRaises(publisher.scope.BridgeFailure): publisher.artifact_receipt(binding, pins)

    def test_all_consumed_pure_helper_sources_match_recorded_audit(self):
        path = publisher.ROOT/"scripts/research/microbatch_bridge_scope.py"
        guard = SimpleNamespace(check=lambda: None, modules=[(publisher.scope, path)])
        name = path.relative_to(publisher.ROOT).as_posix()
        publisher.guard_artifacts((guard,), {"artifacts_sha256": {name: publisher.scope.sha(path)}})
        for record in ({}, {name: "wrong"}):
            with self.assertRaises(publisher.scope.BridgeFailure): publisher.guard_artifacts((guard,), {"artifacts_sha256": record})


if __name__ == "__main__": unittest.main()
