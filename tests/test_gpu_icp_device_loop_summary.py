"""Report-only device-loop publication guards, stdlib and no native imports."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import summarize_gpu_icp_device_loop as s

spec=importlib.util.spec_from_file_location("loop_original_contract_fixture",
    Path(__file__).with_name("test_gpu_icp_device_loop_protocol.py"))
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


class LoopSummaryContracts(unittest.TestCase):
    def fixture(self):
        audit,binding=original.fixture()
        binding["runtime"]={"device":"CUDA:0","open3d":"0.20.0",
            "thread_policy":{"open3d":20,"opencv":20,"omp":"8"},
            "binaries":{name:{"path":"private-binary","sha256":"a"*64} for name in ("numpy","open3d","cupy","native")}}
        binding["fixed_files"]={"private-binary":"a"*64}
        audit["configurations"]=["step1","graph4"]
        graph=audit["rows"][0]
        graph.update(pair=[0,1],seed_index=0,repeat=0,configuration="graph4")
        graph["loop_report"].update(cuda_graph=True,whole_finish_authority=False)
        stats=graph["loop_report"]["statistics"]
        stats.update(flagged_rows=24,packet_bytes=24*64,graph_capture_s=.001,
            maximum_graph_nodes=44,peak_lane_bytes=1024,control_copies=6,graph_captures=1,graph_launches=2)
        step=copy.deepcopy(graph)
        step["configuration"]="step1"
        step["input_binding"]["configuration"].update(cuda_graph=False,chunk_iterations=1)
        step["source_binding"]["cuda_graph"]=False
        step["loop_report"]["provenance"]["cuda_graph"]=False
        step["loop_report"].update(cuda_graph=False)
        step["loop_report"]["statistics"].update(graph_capture_s=0.,maximum_graph_nodes=0,graph_captures=0,graph_launches=0)
        audit["rows"]=[step,graph]
        for row in audit["rows"]:
            for field in s.PARTITION:row[field]=.01
            row["all_in_wall_s"]=sum(row[field] for field in s.PARTITION)
        timing=copy.deepcopy(audit)
        timing.update(kind=s.protocol.TIMING_KIND,mode="timing",performance_attribution_valid=True,
            audit_proof={"path":"unused-private-proof.json","sha256":"a"*64},rows=[],native_controls=[],
            process_preparation_wall_s=1.,phase_closure_wall_s=.1,total_run_wall_s_before_final_save=2.)
        for repeat in range(3):
            for reference in audit["rows"]:
                row=copy.deepcopy(reference)
                row.update(repeat=repeat,query_trace=[],query_trace_sha256=s.digest([]))
                row["native_shadow"].update(fitness_abs_delta=0.,rmse_abs_delta=0.)
                row["input_binding"]["configuration"].update(audit_nearest=False,audit_misses=False)
                row["loop_report"]["statistics"].update(audited_hits=0,audited_misses=0,flagged_rows=0,packet_bytes=0)
                timing["rows"].append(row)
            timing["native_controls"].append({"pair":[0,1],"seed_index":0,"repeat":repeat,
                "wall_s":.03,"pose":copy.deepcopy(reference["terminal"]["pose"]),"fitness":.5,"inlier_rmse":.001})
        return audit,timing

    def good(self,a,t):
        with patch.object(s,"publication_resources"):
            return s.compact_case("case-6",a,t,audit_sha="a"*64)

    def bad(self,mutate):
        a,t=self.fixture();mutate(a,t)
        with self.assertRaises((ValueError,KeyError,TypeError)):self.good(a,t)

    def test_positive_exact_refs_partition_units(self):
        a,t=self.fixture();r=self.good(a,t)
        self.assertEqual(r["audit_coverage"]["query_rows"],48)
        self.assertEqual(r["native_cpu_matched_host_wall"]["median_s"],.03)
        self.assertEqual(r["configurations"][0]["all_in_host_wall"]["median_s"],.06)
        self.assertEqual(r["configurations"][0]["all_in_gpu_to_native_median_ratio"],2.)
        self.assertFalse(r["whole_finish_authority"])
        self.assertFalse(r["complete_proposal_bridge_gates_measured"])
        self.assertEqual(r["per_trajectory_medians"][0]["native_cpu_median_s"],.03)
        self.assertEqual(r["per_trajectory_medians"][0]["configurations"][0]["partition_median_s"]["constructor_wall_s"],.01)

    def test_private_reader_never_mutates_original_or_registry(self):
        a,t=self.fixture();before=dict(s.protocol._REGISTRY)
        original_reader=s.protocol.actual_resource_closure
        self.good(a,t)
        self.assertIs(s.protocol.actual_resource_closure,original_reader)
        self.assertEqual(s.protocol._REGISTRY,before)

    def test_foreign_audit_digest(self):self.bad(lambda a,t:t["audit_proof"].update(sha256="b"*64))
    def test_late_source_binding(self):self.bad(lambda a,t:t.update(binding_after=dict(t["binding"],source_sha256="b"*64)))
    def test_partial_cpu_query_audit(self):self.bad(lambda a,t:a["rows"][0]["loop_report"]["statistics"].update(audited_hits=11))
    def test_full_cpu_audit_left_in_timing(self):self.bad(lambda a,t:t["rows"][0]["loop_report"]["statistics"].update(audited_hits=12))
    def test_scope_chunk_mismatch(self):self.bad(lambda a,t:t["rows"][1]["input_binding"]["configuration"].update(chunk_iterations=2))
    def test_configuration_label_false(self):self.bad(lambda a,t:t["rows"][1].update(configuration="step1"))
    def test_exact_terminal_not_tolerance(self):self.bad(lambda a,t:t["rows"][0]["terminal"].update(inlier_rmse=.001000000000001))
    def test_changed_intermediate_hit_counts(self):
        def change(a,t):
            t["rows"][0]["loop_report"]["statistics"].update(direct_hits=11,direct_misses=13)
        self.bad(change)
    def test_native_correspondence_shadow_failed(self):self.bad(lambda a,t:t["rows"][0]["native_shadow"].update(correspondence_ids_equal=False))
    def test_native_metric_nan(self):self.bad(lambda a,t:t["rows"][0]["native_shadow"].update(fitness_abs_delta=float("nan")))
    def test_missing_repeat_configuration(self):self.bad(lambda a,t:t["rows"].pop())
    def test_duplicate_actual_row(self):self.bad(lambda a,t:t["rows"].__setitem__(1,copy.deepcopy(t["rows"][0])))
    def test_missing_native_control(self):self.bad(lambda a,t:t["native_controls"].pop())
    def test_foreign_native_label(self):self.bad(lambda a,t:t["native_controls"][0].update(seed_index=1))
    def test_finite_control(self):self.bad(lambda a,t:t["native_controls"][0].update(wall_s=float("nan")))
    def test_mislabelled_native_metrics(self):self.bad(lambda a,t:t["native_controls"][0].update(inlier_rmse=.002))
    def test_lane_owners_not_completed(self):self.bad(lambda a,t:t["rows"][0]["loop_report"].update(closed=False))
    def test_changed_disjoint_partition(self):self.bad(lambda a,t:t["rows"][0].update(all_in_wall_s=.07))
    def test_fake_lane_peak(self):self.bad(lambda a,t:t["rows"][0]["loop_report"]["statistics"].update(peak_lane_bytes=True))
    def test_graph_capture_exceeds_advance(self):self.bad(lambda a,t:t["rows"][1]["loop_report"]["statistics"].update(graph_capture_s=1.))
    def test_hidden_primary_failure(self):self.bad(lambda a,t:t.update(failure={"type":"DeviceError"}))
    def test_query_audit_trace_left_in_timing(self):self.bad(lambda a,t:t["rows"][0].update(query_trace=[{}]))
    def test_no_absolute_arrays_or_inventory_publication(self):
        a,t=self.fixture();r=self.good(a,t);encoded=s.protocol.canonical(r)
        for value in (str(s.ROOT),'"input_binding":','"fixed_files":','"pose":','"query_trace":'):
            self.assertNotIn(value,encoded)


class PublicationResourceTests(unittest.TestCase):
    def fixture(self,root):
        for name in s.PRODUCERS+("scripts/fake.cu",):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
        paths={name:s.protocol.sha(root/name) for name in s.PRODUCERS+("scripts/fake.cu",)}
        _,binding=original.fixture()
        binding["method_source"]["artifacts"]={"scripts/fake.cu":paths["scripts/fake.cu"]}
        binding["artifacts_sha256"]=paths
        # Deliberately absent binary bytes: this scope validates the recorded
        # producer closure, never fabricates freshly rehashed library authority.
        binary=str(root/"missing-private-binary")
        binding["fixed_files"]={binary:"a"*64}
        binding["runtime"]={"device":"CUDA:0","open3d":"0.20.0",
            "thread_policy":{"open3d":20,"opencv":20,"omp":"8"},
            "binaries":{name:{"path":binary,"sha256":"a"*64} for name in ("numpy","open3d","cupy","native")}}
        return binding

    def test_declared_source_only_resource_domain_and_actual_source_drift(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(s,"ROOT",Path(folder)):
            root=Path(folder);binding=self.fixture(root);s.publication_resources(binding)
            (root/"scripts/fake.cu").write_bytes(b"changed")
            with self.assertRaises(ValueError):s.publication_resources(binding)

    def test_omitted_shader_or_mismatched_library_manifest(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(s,"ROOT",Path(folder)):
            root=Path(folder);binding=self.fixture(root)
            bad=copy.deepcopy(binding);bad["artifacts_sha256"].pop("scripts/fake.cu")
            with self.assertRaises(ValueError):s.publication_resources(bad)
            bad=copy.deepcopy(binding);bad["runtime"]["binaries"]["numpy"]["sha256"]="b"*64
            with self.assertRaises(ValueError):s.publication_resources(bad)


if __name__=="__main__":unittest.main()
