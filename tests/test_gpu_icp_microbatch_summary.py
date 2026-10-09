"""Stdlib publication guards; reuse the exact original authority's fake domain."""
import copy
import importlib.util
from pathlib import Path
import unittest

from scripts.research import summarize_gpu_icp_microbatch as summary

spec=importlib.util.spec_from_file_location("batch_original_contract_fixture",
    Path(__file__).with_name("test_gpu_icp_experiment_protocol.py"))
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


class SummaryContracts(unittest.TestCase):
    def fixture(self):
        audit,binding,pair=original.ProtocolContracts().fixture()
        audit.update(source_sha256=summary.protocol.CURRENT,source_sha256_after=summary.protocol.CURRENT)
        audit["helpers"][0]["whole_finish_authority"]=False
        timing=copy.deepcopy(audit)
        timing.update(kind=summary.TIMING_KIND,performance_attribution_valid=True,
            audited_proof={"path":"unused-private-audit.json","sha256":"a"*64},
            audit_rows=[],timing_rows=[],native_timing_rows=[])
        timing["fixed_files_sha256"][str(Path(timing["audited_proof"]["path"]).resolve())]="a"*64
        timing["fixed_files_sha256_after"]=copy.deepcopy(timing["fixed_files_sha256"])
        helper=timing["helpers"][0]
        helper.update(audit=False,statistics={"successful_batches":12,"cold_setup_s":.01,
            "source_checks_s":.02,"source_checks":36,"peak_combined_owned_bytes":1024})
        for repeat in range(3):
            for size in (2,4):
                for schedule in ("serial","concurrent"):
                    row=copy.deepcopy(next(r for r in audit["audit_rows"] if r["batch_size"]==size and r["schedule"]==schedule))
                    row.update(repeat=repeat,batch_wall_s=.2+repeat*.01)
                    row["batch_record"]["wall_s"]=row["batch_wall_s"]-.02
                    for lane in row["batch_record"]["lane_deltas"]:
                        lane["retrieval_statistics"].update(audited_hits=0,audited_misses=0)
                    timing["timing_rows"].append(row)
                timing["native_timing_rows"].append({"repeat":repeat,"batch_size":size,
                    "batch_wall_s":.1,"passed":True,
                    "shadows":copy.deepcopy(row["native_shadows"])})
        return audit,timing

    def good(self,audit,timing):
        return summary.compact_case("case-6",audit,timing,audit_sha="a"*64)

    def bad(self,mutate):
        a,t=self.fixture();mutate(a,t)
        with self.assertRaises((ValueError,KeyError,TypeError)): self.good(a,t)

    def test_exact_matched_success_and_units(self):
        a,t=self.fixture();row=self.good(a,t)
        self.assertEqual(row["audit_coverage"],{"query_rows":120,"audited_hits":60,"audited_misses":48,"exact_cpu_queries":12})
        self.assertEqual(row["timing_groups"][0]["native_cpu"]["median_s"],.1)
        self.assertAlmostEqual(row["timing_groups"][0]["serial"]["gpu_to_native_median_wall_ratio"],2.1)
        self.assertFalse(row["whole_finish_authority"])
        self.assertNotIn("pose",row)

    def test_audit_actual_hit_omission(self): self.bad(lambda a,t:a["audit_rows"][0]["batch_record"]["lane_deltas"][0]["retrieval_statistics"].update(audited_hits=4))
    def test_foreign_closed_audit(self): self.bad(lambda a,t:t["audited_proof"].update(sha256="b"*64))
    def test_timing_native_row_missing(self): self.bad(lambda a,t:t["native_timing_rows"].pop())
    def test_timing_concurrent_row_missing(self): self.bad(lambda a,t:t["timing_rows"].pop())
    def test_duplicate_schedule(self): self.bad(lambda a,t:t["timing_rows"].__setitem__(1,copy.deepcopy(t["timing_rows"][0])))
    def test_native_shadow_failed(self): self.bad(lambda a,t:t["native_timing_rows"][0]["shadows"][0].update(correspondence_ids_equal=False))
    def test_scoped_gate_bypassed(self): self.bad(lambda a,t:t["timing_rows"][0]["bridge_gate_shadows"][0].update(original_forward_consumed_once=False))
    def test_competing_proposal_dropped(self): self.bad(lambda a,t:t["timing_rows"][0].update(full_original_proposal_count=2))
    def test_permuted_seeds(self): self.bad(lambda a,t:t["timing_rows"][0]["pair_binding"].update(seeds=list(reversed(t["pair_binding"]["seeds"][:2]))))
    def test_batch_wall_mismatch(self): self.bad(lambda a,t:t["timing_rows"][0].update(batch_wall_s=.01))
    def test_nonfinite_inner_wall(self): self.bad(lambda a,t:t["timing_rows"][0]["batch_record"].update(wall_s=float("nan")))
    def test_nonfinite_wall(self): self.bad(lambda a,t:t["native_timing_rows"][0].update(batch_wall_s=float("nan")))
    def test_hidden_cpu_fallback(self): self.bad(lambda a,t:t["timing_rows"][0]["batch_record"]["lane_deltas"][0]["statistics"].update(cpu_fallback_calls=1))
    def test_unomitted_audit_overhead(self): self.bad(lambda a,t:t["timing_rows"][0]["batch_record"]["lane_deltas"][0]["retrieval_statistics"].update(audited_hits=5))
    def test_late_helper_drift(self): self.bad(lambda a,t:t["helpers"][0].update(source_unchanged=False))
    def test_changed_resources(self): self.bad(lambda a,t:t["fixed_files_sha256_after"].update(extra="b"*64))
    def test_arbitrary_extra_fixed_resource(self):
        def change(a,t):
            t["fixed_files_sha256"]["another-resource"]="b"*64
            t["fixed_files_sha256_after"]=copy.deepcopy(t["fixed_files_sha256"])
        self.bad(change)
    def test_status_with_primary_failure(self): self.bad(lambda a,t:t.update(failure={"type":"DeviceError"}))
    def test_infinite_nested_setup(self): self.bad(lambda a,t:t["helpers"][0]["statistics"].update(cold_setup_s=float("inf")))
    def test_no_absolute_private_paths_or_raw_payload(self):
        a,t=self.fixture();r=self.good(a,t)
        encoded=summary.protocol.canonical(r)
        self.assertNotIn(str(summary.ROOT),encoded)
        self.assertNotIn("fixed_files_sha256",encoded)
        self.assertNotIn("pair_binding\":",encoded)
        self.assertNotIn("seeds\":[",encoded)


if __name__=="__main__":unittest.main()
