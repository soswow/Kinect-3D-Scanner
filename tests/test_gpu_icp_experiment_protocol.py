"""Meaningful stdlib-only negative contracts for new batch timing authority."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research import gpu_icp_experiment_protocol as p
from scripts.research.gpu_icp_experiment_driver import equal_evidence, correspondence_bounds, final_closure


class ProtocolContracts(unittest.TestCase):
    def fixture(self):
        pin = p.sha(p.ROOT/"README.md")
        binding = {"source_sha256": p.CURRENT, "artifacts_sha256": {"README.md": pin},
            "method_policy": p.POLICY, "solve_metadata": {"library_sha256": "a"*64},
            "runtime": {"device": "CUDA:0", "open3d": "0.20.0",
                "thread_policy": {"open3d_threads": 20, "opencv_threads": 20, "omp_threads": "8"}},
            "configuration": {"device": "CUDA:0", "gpu_timing": False, "stages": [[.12, 40], [.06, 30], [.03, 20]],
                "max_points": 1000000, "max_query_bytes": 136*1024**2, "max_cache_bytes": 256*1024**2, "max_total_bytes": 512*1024**2}}
        def desc(rows, cols, key):
            return {"shape": [rows, cols], "dtype": "<f8", "nbytes": rows*cols*8, "sha256": key*64}
        cloud = {name: desc(12, 3, "b") for name in ("points", "normals", "colors")}
        pair = {"source": copy.deepcopy(cloud), "target": copy.deepcopy(cloud),
            "seeds": [desc(4, 4, key) for key in ("c", "d", "e", "f")]}
        report = {"kind": p.KIND, "status": "passed", "binding": binding, "binding_after": copy.deepcopy(binding),
            "pair_binding": pair, "pair_binding_after": copy.deepcopy(pair), "cleanup_passed": True,
            "source_unchanged": True, "fixture_unchanged": True, "input_bytes_unchanged": True,
            "cleanup_failures": [],
            "whole_finish_authority": False, "performance_attribution_valid": False, "batch_sizes": [2, 4],
            "producer_artifacts_sha256": {name:p.sha(p.ROOT/name) for name in p.PRODUCERS},
            "producer_artifacts_sha256_after": {name:p.sha(p.ROOT/name) for name in p.PRODUCERS},
            "helpers": [{"audit": True, "binding": copy.deepcopy(binding), "source_unchanged": True,
                "closed": True, "failure": None, "cleanup_failures": []}],
            "fixed_files_sha256": {str(p.ROOT/"README.md"): pin}, "fixed_files_sha256_after": {str(p.ROOT/"README.md"): pin}, "audit_rows": []}
        for n in (2, 4):
            for schedule in ("serial", "concurrent"):
                actual = p.prefix_pair(pair, n)
                nn = {"query_rows": 10, "direct_gpu_hits": 5, "declared_gpu_misses": 4, "audited_hits": 5,
                    "audited_misses": 4, "exact_cpu_queries": 1, "audit_index_mismatches": 0,
                    "audit_false_misses": 0, "device_malformed_results": 0}
                lane = {"statistics": {"calls": 1, "cpu_fallback_calls": 0}, "retrieval_statistics": nn}
                report["audit_rows"].append({"batch_size": n, "schedule": schedule, "pair_binding": actual,
                    "passed": True, "pair_verdict_equal": True, "full_original_proposal_count": 4,
                    "batch_record": {"inputs": actual, "schedule": schedule, "lanes": n,
                        "input_bytes_unchanged": True, "full_call_fallbacks": 0,
                        "lane_deltas": [dict(copy.deepcopy(lane), index=i) for i in range(n)]},
                    "native_shadows": [{"seed_index": i, "passed": True, "correspondence_ids_equal": True,
                        "strong_gate_equal": True, "transform_max_abs_diff": 0., "fitness_abs_diff": 0., "inlier_rmse_abs_diff": 0.} for i in range(n)],
                    "bridge_gate_shadows": [{"seed_index": i, "passed": True, "original_forward_consumed_once": True} for i in range(n)]})
        return report, binding, pair

    def bad(self, mutation):
        report, binding, pair = self.fixture()
        mutation(report)
        with self.assertRaises(ValueError): p.validate_report(report, binding, pair)

    def test_positive_exact_both_schedules_prefixes(self):
        r,b,v = self.fixture()
        self.assertEqual(len(p.validate_report(r,b,v)), 4)

    def test_old_kind_refused(self): self.bad(lambda r: r.update(kind="standalone-original-double-device-flat-resident"))
    def test_incomplete_status_refused(self): self.bad(lambda r: r.update(status="running"))
    def test_missing_schedule_refused(self): self.bad(lambda r: r["audit_rows"].pop())
    def test_duplicate_schedule_refused(self): self.bad(lambda r: r["audit_rows"].__setitem__(1, copy.deepcopy(r["audit_rows"][0])))
    def test_permuted_seed_refused(self): self.bad(lambda r: r["audit_rows"][0]["pair_binding"].update(seeds=list(reversed(r["pair_binding"]["seeds"][:2]))))
    def test_missing_cpu_hits_refused(self): self.bad(lambda r: r["audit_rows"][0]["batch_record"]["lane_deltas"][0]["retrieval_statistics"].update(audited_hits=4))
    def test_missing_cpu_misses_refused(self): self.bad(lambda r: r["audit_rows"][0]["batch_record"]["lane_deltas"][0]["retrieval_statistics"].update(audited_misses=3))
    def test_hidden_fallback_refused(self): self.bad(lambda r: r["audit_rows"][0]["batch_record"]["lane_deltas"][0]["statistics"].update(cpu_fallback_calls=1))
    def test_device_malformed_refused(self): self.bad(lambda r: r["audit_rows"][0]["batch_record"]["lane_deltas"][0]["retrieval_statistics"].update(device_malformed_results=1))
    def test_native_changed_correspondence_refused(self): self.bad(lambda r: r["audit_rows"][0]["native_shadows"][0].update(correspondence_ids_equal=False))
    def test_native_roundoff_limit_refused(self): self.bad(lambda r: r["audit_rows"][0]["native_shadows"][0].update(transform_max_abs_diff=1e-7))
    def test_nonfinite_metric_refused(self): self.bad(lambda r: r["audit_rows"][0]["native_shadows"][0].update(inlier_rmse_abs_diff=float("nan")))
    def test_original_gate_bypass_refused(self): self.bad(lambda r: r["audit_rows"][0]["bridge_gate_shadows"][0].update(original_forward_consumed_once=False))
    def test_competing_proposal_drop_refused(self): self.bad(lambda r: r["audit_rows"][0].update(full_original_proposal_count=2))
    def test_ambiguity_verdict_changed_refused(self): self.bad(lambda r: r["audit_rows"][0].update(pair_verdict_equal=False))
    def test_cleanup_fault_refused(self): self.bad(lambda r: r.update(cleanup_passed=False))
    def test_contradictory_failure_refused(self): self.bad(lambda r: r.update(failure={"type": "DeviceFault"}))
    def test_contradictory_cleanup_refused(self): self.bad(lambda r: r.update(cleanup_failures=[{"type": "DeviceFault"}]))
    def test_helper_source_late_drift_refused(self): self.bad(lambda r: r["helpers"][0].update(source_unchanged=False))
    def test_helper_owner_not_released_refused(self): self.bad(lambda r: r["helpers"][0].update(closed=False))
    def test_source_after_drift_refused(self): self.bad(lambda r: r["binding_after"].update(source_sha256="9"*64))
    def test_producer_after_drift_refused(self): self.bad(lambda r: r["producer_artifacts_sha256_after"].update(extra="a"*64))
    def test_native_resource_drift_refused(self): self.bad(lambda r: r["fixed_files_sha256_after"].update(extra="a"*64))

    def test_registered_token_and_exact_prefix_scope(self):
        r,b,v = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"audit.json"; path.write_text(json.dumps(r), encoding="utf-8")
            permit = p.validate_microbatch_audit(path,b,v)
            p.assert_registered_microbatch_permit(permit)
            p.validate_microbatch_permit(permit,binding=b,pair_binding=p.prefix_pair(v,2),schedule="concurrent")
            with self.assertRaises(ValueError): p.assert_registered_microbatch_permit(replace(permit))
            wrong = copy.deepcopy(p.prefix_pair(v,2)); wrong["seeds"].reverse()
            with self.assertRaises(ValueError): p.validate_microbatch_permit(permit,binding=b,pair_binding=wrong,schedule="serial")
            path.write_text(json.dumps(r)+" ", encoding="utf-8")
            with self.assertRaises(ValueError): p.assert_registered_microbatch_permit(permit)

    def test_old_or_arbitrary_constructor_token_refused(self):
        with self.assertRaises(ValueError): p.assert_registered_microbatch_permit(object())

    def test_numeric_evidence_never_masks_boolean_decisions(self):
        self.assertFalse(equal_evidence(True,1))
        self.assertFalse(equal_evidence({"support": [1,2]}, {"support": [2,1]}))
        self.assertFalse(equal_evidence(float("nan"),float("nan")))
        self.assertTrue(equal_evidence({"rmse": .01},{"rmse": .01000000001}))

    def test_actual_cloud_correspondence_bounds(self):
        correspondence_bounds([[0,0],[11,12]],12,13)
        for rows in ([[12,0]], [[0,13]], [[-1,0]], [[0,-1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError): correspondence_bounds(rows,12,13)

    def closed(self):
        r,b,v = self.fixture()
        r["helpers"] = [{"source_unchanged": True, "closed": True, "failure": None,
            "cleanup_failures": [], "binding": copy.deepcopy(b)}]
        return r

    def test_both_modes_require_late_source_closure(self):
        self.assertTrue(final_closure(self.closed()))
        for mutate in (lambda r: r["producer_artifacts_sha256_after"].update(extra="a"*64),
                lambda r: r["helpers"][0].update(source_unchanged=False),
                lambda r: r["helpers"][0]["binding"].update(source_sha256="9"*64),
                lambda r: r["binding_after"].update(source_sha256="9"*64),
                lambda r: r["helpers"][0].update(closed=False)):
            r = self.closed(); mutate(r)
            self.assertFalse(final_closure(r))


if __name__ == "__main__": unittest.main()
