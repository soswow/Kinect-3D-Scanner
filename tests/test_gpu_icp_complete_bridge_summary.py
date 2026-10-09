"""Artificial recorded receipts; no numerical modules or token minting."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import summarize_gpu_icp_complete_bridge as summary
from tests.test_microbatch_bridge_protocol import fixture


def reports():
    audit, _, _ = fixture()
    audit["binding"]["fixed_files"] = {"recorded-private-fixture": "b"*64}
    artifacts = audit["binding"]["artifacts_sha256"]
    artifacts.pop("scripts/fake.cu")
    artifacts["scripts/research/device_loop_control.cu"] = summary.sha(summary.ROOT/"scripts/research/device_loop_control.cu")
    for proposal in audit["gpu_proposals"]:
        for call in proposal["calls"]:
            call["source_binding"]["artifacts"] = copy.deepcopy(audit["binding"]["method_source"]["artifacts"])
            call["loop_report"]["provenance"] = copy.deepcopy(call["source_binding"])
    audit["workspace"]["jobs"] = [{"index": i, "closed": True, "failure": None,
        "report": copy.deepcopy(call["loop_report"])} for i, call in enumerate(
            c for p in audit["gpu_proposals"] for c in p["calls"])]
    timing = copy.deepcopy(audit)
    timing.update(kind=summary.TIMING_KIND, mode="timing", performance_attribution_valid=True,
        audit_proof={"path": "unused", "sha256": "a"*64},
        native_whole_proposals_wall_s=.5, shared_constructor_prepare_wall_s=.1,
        gpu_whole_proposals_wall_s=.4, gpu_cold_setup_and_proposals_wall_s=.1+.4,
        owner_cleanup_wall_s=.01, producer_closure_wall_s=.05,
        producer_wall_s_before_final_publication=1.1)
    timing["workspace"]["audit"] = False
    for proposal in timing["gpu_proposals"]:
        for call in proposal["calls"]:
            call["consumed_input_binding"]["configuration"].update(audit_nearest=False, audit_misses=False)
            call["loop_report"]["input_binding"] = copy.deepcopy(call["consumed_input_binding"])
            call["loop_report"]["statistics"].update(audited_hits=0, audited_misses=0, flagged_rows=0, packet_bytes=0)
            call["query_trace"] = []
            call["native_shadow"] = {"collected": False}
    timing["workspace"]["jobs"] = [{"index": i, "closed": True, "failure": None,
        "report": copy.deepcopy(call["loop_report"])} for i, call in enumerate(
            c for p in timing["gpu_proposals"] for c in p["calls"])]
    return audit, timing


class SummaryTests(unittest.TestCase):
    def check(self, audit, timing): return summary.compact(audit, timing, "a"*64)

    def test_valid_path_free_scalar_counts_and_charged_ratio(self):
        result = self.check(*reports())
        self.assertEqual(result["actual_gpu_calls"], 2)
        self.assertEqual(result["proposal_count"], 2)
        self.assertEqual(result["audit_coverage"]["query_rows"], 48)
        self.assertEqual(result["timing"]["gpu_subtotal_plus_cleanup_s"], .51)
        self.assertEqual(result["timing"]["gpu_subtotal_plus_cleanup_to_native_wall_ratio"], 1.02)
        self.assertEqual(result["timing"]["producer_closure_s"], .05)
        self.assertNotIn("path", summary.scope.canonical(result))
        self.assertFalse(result["whole_finish_authority"])

    def test_legacy_kind_running_or_failed_report_refused(self):
        for key, value in (("kind", "gpu-icp-seed-microbatch-audit-v1"), ("status", "running"), ("failure", "fault")):
            a, t = reports(); a[key] = value
            with self.assertRaises(ValueError): self.check(a, t)

    def test_missing_or_contradictory_cleanup_refused(self):
        for key, value in (("cleanup_passed", False), ("cleanup_failures", ["fault"]),
                ("input_bytes_unchanged", False), ("whole_finish_authority", True)):
            a, t = reports(); t[key] = value
            with self.assertRaises(ValueError): self.check(a, t)

    def test_changed_source_and_fixed_resource_receipts_refused(self):
        a, t = reports(); t["binding_after"] = dict(t["binding"], source_sha256="f"*64)
        with self.assertRaises(ValueError): self.check(a, t)
        a, t = reports(); t["binding"] = copy.deepcopy(t["binding"])
        t["binding"]["fixed_files"]["another"] = "b"*64; t["binding_after"] = t["binding"]
        with self.assertRaises(ValueError): self.check(a, t)

    def test_current_source_file_changed_refused(self):
        a, t = reports()
        with patch.object(summary, "sha", return_value="f"*64):
            with self.assertRaises(ValueError): self.check(a, t)

    def test_missing_mandatory_own_source_refused(self):
        a, t = reports()
        for r in (a, t):
            r["binding"]["artifacts_sha256"].pop(summary.protocol.BRIDGE_FILES[0])
        with self.assertRaises(ValueError): self.check(a, t)

    def test_full_workspace_source_family_and_graph4_required(self):
        a, t = reports()
        a["binding"]["artifacts_sha256"].pop("tests/test_device_loop_workspace.py")
        with self.assertRaises(ValueError): self.check(a, t)
        for key, value in (("graph", False), ("chunk_iterations", 2)):
            a, t = reports(); a["binding"]["configuration"][key] = value
            with self.assertRaises(ValueError): self.check(a, t)
        a, t = reports(); t["gpu_proposals"][0]["calls"][0]["source_binding"]["setup_reuse"] = {}
        with self.assertRaises(ValueError): self.check(a, t)

    def test_wrong_closed_audit_sha_refused(self):
        a, t = reports(); t["audit_proof"]["sha256"] = "b"*64
        with self.assertRaises(ValueError): self.check(a, t)

    def test_foreign_seed_or_target_input_refused(self):
        for role, name in (("source", "points"), ("target", "normals")):
            a, t = reports(); t["gpu_proposals"][0]["calls"][0]["input_binding"][role][name]["sha256"] = "f"*64
            with self.assertRaises(ValueError): self.check(a, t)

    def test_dynamic_suffix_and_order_must_be_complete(self):
        a, t = reports(); t["gpu_proposals"][0]["calls"] = []
        with self.assertRaises(ValueError): self.check(a, t)
        a, t = reports(); t["gpu_proposals"][1]["calls"][0]["call_index"] = 1
        with self.assertRaises(ValueError): self.check(a, t)

    def test_workspace_job_exact_report_and_order_required(self):
        for field, value in (("index", 1), ("closed", False), ("report", {})):
            a, t = reports(); t["workspace"]["jobs"][0][field] = value
            with self.assertRaises(ValueError): self.check(a, t)

    def test_cache_lane_or_template_failure_cannot_be_hidden(self):
        for owner, field, value in (("shared_cache", "closed", False), ("workspace", "template_started", True),
                ("workspace", "failure", "fault"), ("workspace", "active_lane", True)):
            a, t = reports(); t[owner][field] = value
            with self.assertRaises(ValueError): self.check(a, t)

    def test_gate_decision_witness_suffix_and_ambiguity_changed_refused(self):
        a, t = reports(); t["gpu_proposals"][0]["gates"][0]["result"] = False
        with self.assertRaises(ValueError): self.check(a, t)
        a, t = reports(); t["gpu_proposals"][0]["gates"] = []
        with self.assertRaises(ValueError): self.check(a, t)
        a, t = reports(); t["gpu_pair_verdict"]["ambiguous"] = True
        with self.assertRaises(ValueError): self.check(a, t)

    def test_own_terminal_one_ulp_metric_change_refused(self):
        a, t = reports(); t["gpu_proposals"][0]["calls"][0]["terminal"]["fitness"] += 1e-16
        with self.assertRaises(ValueError): self.check(a, t)

    def test_failed_actual_native_shadow_and_canonical_id_refused(self):
        for key, value in (("correspondence_ids_equal", False), ("transform_max_abs_delta", 1e-6)):
            a, t = reports(); a["gpu_proposals"][0]["calls"][0]["native_shadow"][key] = value
            with self.assertRaises(ValueError): self.check(a, t)

    def test_timing_observations_and_hidden_fallback_counts_refused(self):
        for key, value in (("audited_hits", 1), ("solve_blocks", 1), ("query_rows", 1), ("packet_bytes", 64)):
            a, t = reports(); t["gpu_proposals"][0]["calls"][0]["loop_report"]["statistics"][key] = value
            # Keep the claimed owner report consistent; the actual accounting is still invalid.
            t["workspace"]["jobs"][0]["report"] = copy.deepcopy(t["gpu_proposals"][0]["calls"][0]["loop_report"])
            with self.assertRaises(ValueError): self.check(a, t)

    def test_inner_audit_flag_cannot_masquerade_as_timing(self):
        a, t = reports(); t["gpu_proposals"][0]["calls"][0]["consumed_input_binding"]["configuration"]["audit_nearest"] = True
        with self.assertRaises(ValueError): self.check(a, t)

    def test_timing_subtotal_never_subtracts_audit_or_closure(self):
        a, t = reports(); t["gpu_cold_setup_and_proposals_wall_s"] -= .01
        with self.assertRaises(ValueError): self.check(a, t)
        for key in ("native_whole_proposals_wall_s", "owner_cleanup_wall_s", "producer_closure_wall_s"):
            a, t = reports(); t[key] = float("nan")
            with self.assertRaises(ValueError): self.check(a, t)

    def test_fresh_import_has_no_numerical_modules(self):
        code = "from scripts.research import summarize_gpu_icp_complete_bridge; import sys; assert not {'numpy','cupy','open3d','cv2'} & set(sys.modules)"
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=summary.ROOT,
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_publisher_never_mints_or_changes_a_registry(self):
        registries = (summary.protocol.values._REGISTRY, summary.protocol.workspace_tokens._REGISTRY,
            summary.protocol._BRIDGE_RECORDS)
        before = [dict(r) for r in registries]
        self.check(*reports())
        self.assertEqual(before, [dict(r) for r in registries])

    def test_three_matched_receipts_publish_once_counted_audit(self):
        with tempfile.TemporaryDirectory(dir=summary.ROOT/"benchmark-output") as folder:
            directory = Path(folder); a, t = reports()
            audit = directory/"audit.json"
            audit.write_text(json.dumps(a), encoding="utf-8")
            timings = []
            for i in range(3):
                value = copy.deepcopy(t)
                value["audit_proof"] = {"path": str(audit), "sha256": summary.sha(audit)}
                value["native_whole_proposals_wall_s"] += i*.1
                path = directory/f"timing{i}.json"
                path.write_text(json.dumps(value), encoding="utf-8"); timings.append(path)
            output = directory/"summary.json"
            summary.run(SimpleNamespace(audit=audit, timing=timings, output=output))
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "passed")
            self.assertEqual(value["timing_group"]["samples"], 3)
            self.assertEqual(value["case"]["audit_coverage"]["query_rows"], 48)
            self.assertEqual(value["timing_group"]["median_native_proposals_wall_s"], .6)
            self.assertFalse(value["performance_authority_token_created"])
            self.assertNotIn(str(directory), output.read_text(encoding="utf-8"))

    def test_failed_publication_preserves_diagnostics_and_existing_output(self):
        with tempfile.TemporaryDirectory(dir=summary.ROOT/"benchmark-output") as folder:
            directory = Path(folder)
            audit, timing, output = (directory/name for name in ("audit.json", "timing.json", "summary.json"))
            audit.write_text("{", encoding="utf-8"); timing.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(audit=audit, timing=[timing], output=output)
            with self.assertRaises(json.JSONDecodeError): summary.run(args)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["failure"]["type"], "JSONDecodeError")
            original = output.read_bytes()
            with self.assertRaises(ValueError): summary.run(args)
            self.assertEqual(original, output.read_bytes())


if __name__ == "__main__": unittest.main()
