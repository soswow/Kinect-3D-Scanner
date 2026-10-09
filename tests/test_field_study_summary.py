"""Meaningful stdlib-only reporting scope and privacy regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.research import summarize_field_study as summary


def raw_profile(cuda=False):
    return {"schema_version":1,"source_sha256":"a"*64,"input_sha256":"b"*64,
        "selected_indices":[0,1],"frames":2,"settings":{"confidence_fusion":True,"final_block_count":10000},
        "thread_policy":{"open3d":20,"opencv":20,"omp":"8"},"omp_threads":8,"seed":0,"versions":{"numpy":"test"},
        "native_extension":{"sha256":"c"*64,"changed_during_profile":False},
        "pose_seeds_used":False,"mesh_built":True,"finish_requested":True,
        "input_changed_during_profile":False,"source_changed_during_profile":False,"accepted_indices":[0],
        "accepted_indices_before_finish":[0],"accepted_before_finish":1,"accepted":1,
        "session":"C:/Users/private/session","poses":{"private_trajectory":[[99.0]]},
        "graph_inventory":{"private_vertices":[1,2]},"checkpoint":{"sha256":"d"*64},
        "live_diagnostics":[{"elapsed_ms":1.0},{"elapsed_ms":3.0}],"live_stages":{},
        "live_s":1.,"finish_s":2.,"processing_s":3.,
        "pipeline_options":{"KINECT_CUDA_MATCHING":"cuda" if cuda else "cpu",
            "KINECT_CUDA_INPUT":"auto" if cuda else "off","KINECT_CUDA_CONFIDENCE":"off"},
        "backend":{"geometric_verification":{"implementation":"legacy"},
            "depth_confidence":{"requested":"off","implementation":"cpu"},
            "cuda_input":{"requested":"auto" if cuda else "off","implementation":"cuda" if cuda else "cpu",
                "device":"CUDA:0" if cuda else "CPU:0","probe_passed":True if cuda else None,
                "gpu_batches":2 if cuda else 0,"cpu_batches":0 if cuda else 2,"fallback_batches":0},
            "descriptor_matching":{"requested":"cuda" if cuda else "cpu","implementation":"cuda" if cuda else "cpu",
                "cuda_batches":2 if cuda else 0,"cpu_batches":0 if cuda else 2,"fallback_reason":None}}}


def write(folder, name, value):
    path = Path(folder)/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8")
    return path


class ScopeTests(unittest.TestCase):
    allowed = {"KINECT_CUDA_MATCHING":("cpu","cuda"),"KINECT_CUDA_INPUT":("off","auto")}

    def test_genuine_cpu_cuda_scope(self):
        summary.pair_scope(raw_profile(),raw_profile(True),self.allowed)
        summary.actual_backend(raw_profile(),False)
        summary.actual_backend(raw_profile(True),True)

    def test_core_input_settings_thread_selection_drift(self):
        for key,value in (("source_sha256","e"*64),("input_sha256","e"*64),
            ("settings",{"confidence_fusion":True,"final_block_count":20000}),
            ("thread_policy",{"open3d":8}),("selected_indices",[1,0]),("versions",{"numpy":"other"})):
            with self.subTest(key=key):
                b=raw_profile(True);b[key]=value
                with self.assertRaises(ValueError):summary.pair_scope(raw_profile(),b,self.allowed)

    def test_same_duplicate_or_boolean_selection_cannot_pass_pair(self):
        for indices in ([0,0],[False,1]):
            a,b=raw_profile(),raw_profile(True);a["selected_indices"]=b["selected_indices"]=indices
            with self.assertRaises(ValueError):summary.pair_scope(a,b,self.allowed)

    def test_unrelated_option_change(self):
        b=raw_profile(True);b["pipeline_options"]["KINECT_CUDA_CONFIDENCE"]="fused"
        with self.assertRaises(ValueError):summary.pair_scope(raw_profile(),b,self.allowed)

    def test_seeds_failures_native_change_and_confidence_disabled(self):
        for key,value in (("pose_seeds_used",True),("mesh_built",False),("input_changed_during_profile",True),
            ("source_changed_during_profile",True),("accepted_indices",[1])):
            b=raw_profile(True);b[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):summary.pair_scope(raw_profile(),b,self.allowed)
        b=raw_profile(True);b["native_extension"]["changed_during_profile"]=True
        with self.assertRaises(ValueError):summary.pair_scope(raw_profile(),b,self.allowed)
        a,b=raw_profile(),raw_profile(True)
        a["settings"]["confidence_fusion"]=b["settings"]["confidence_fusion"]=False
        with self.assertRaises(ValueError):summary.pair_scope(a,b,self.allowed)

    def test_actual_backend_not_only_requested_label(self):
        for key,value in (("implementation","cpu"),("probe_passed",False),("fallback_batches",1),
            ("cpu_batches",1),("gpu_batches",0),("gpu_batches",True),("device","CUDA:1")):
            b=raw_profile(True);b["backend"]["cuda_input"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):summary.actual_backend(b,True)
        b=raw_profile(True);b["backend"]["descriptor_matching"]["cpu_batches"]=1
        with self.assertRaises(ValueError):summary.actual_backend(b,True)
        a=raw_profile();a["backend"]["cuda_input"]["gpu_batches"]=1
        with self.assertRaises(ValueError):summary.actual_backend(a,False)

    def test_checkpoint_state_identity_and_seed_closure(self):
        a,b=raw_profile(True),raw_profile(True)
        summary.finish_pair_scope(a,b)
        for key,value in (("checkpoint",{"sha256":"e"*64}),("accepted_indices_before_finish",[1]),("pose_seeds_used",True)):
            changed=copy.deepcopy(b);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):summary.finish_pair_scope(a,changed)


class ReportingTests(unittest.TestCase):
    def test_report_bytes_closed_against_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path=write(folder,"closed.json",{"status":"passed"})
            reports=summary.Reports(folder);reports.record("closed.json");reports.close()
            path.write_text('{"status":"failed"}',encoding="utf-8")
            with self.assertRaises(ValueError):reports.close()

    def test_relative_allowlist_does_not_read_external_path(self):
        reports=summary.Reports()
        for name in ("../external.json","C:/Users/private/a.json","a\\b.json","/tmp/report.json"):
            with self.subTest(name=name),self.assertRaises(ValueError):reports.read(name)

    def test_profile_allowlist_excludes_private_paths_poses_and_graphs(self):
        with tempfile.TemporaryDirectory() as folder:
            value=raw_profile();value["settings"]["private_path"]="F:/Projects/private/calibration"
            write(folder,"profile.json",value)
            row=summary.profile(summary.Reports(folder),"profile.json")
            text=json.dumps(row)
            for forbidden in ("C:/Users","F:/Projects","private_trajectory","private_vertices","graph_inventory","poses"):
                self.assertNotIn(forbidden,text)
            self.assertEqual(row["confidence_fusion"],True)
            self.assertEqual(row["cuda_confidence_acceleration"],"off")
            self.assertEqual(row["live_latency_ms"]["p95"],2.9)

    def test_latency_read_must_match_stored_report_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            write(folder,"profile.json",raw_profile())
            original=summary.latency_summary
            def wrong(path):
                row=original(path);row["profile_sha256"]="e"*64;return row
            with mock.patch.object(summary,"latency_summary",wrong),self.assertRaises(ValueError):
                summary.profile(summary.Reports(folder),"profile.json")

    def test_surface_contract_rejects_nonfinite_wrong_frame_or_resolution(self):
        good={"threshold_m":.005,"samples_per_surface":30000,"precision":1.,"completeness":1.,
            "surface_p95_m":1e-8,"surface_rmse_m":1e-8,"alignment":"fixed input coordinate frame; no scale or trajectory fitting"}
        summary.surface(good)
        for key,value in (("surface_p95_m",float("nan")),("surface_rmse_m",float("inf")),("precision",1.01),
            ("completeness",.98),("surface_p95_m",.01),("samples_per_surface",1000),("alignment","best fit")):
            bad=dict(good);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):summary.surface(bad)

    def test_quality_references_require_exact_current_report_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            write(folder,"a.json",{"status":"complete"});write(folder,"b.json",{"status":"complete"})
            reports=summary.Reports(folder)
            q={"status":"complete","observable_geometry_passed":True,
                "reference":{"sha256":"e"*64},"candidate":{"sha256":reports.record("b.json")["sha256"]}}
            write(folder,"quality.json",q)
            with self.assertRaises(ValueError):summary.matched_quality(reports,"quality.json","a.json","b.json",True)

    def test_zero_bool_nonfinite_duration_rejected_and_regression_retained(self):
        for value in (0,True,float("nan"),float("inf"),-1):
            with self.subTest(value=value),self.assertRaises(ValueError):summary.ratio(value,1.)
        self.assertLess(summary.ratio(1.,2.)["less_processing_percent"],0)

    def test_missing_quality_is_pending_not_geometry_authority(self):
        with tempfile.TemporaryDirectory() as folder:
            row=summary.activation_comparison(summary.Reports(folder),"chest-7")
            self.assertEqual(row["independent_quality_status"],"pending")
            self.assertIs(row["whole_mesh_quality_authority"],False)

    def activation_fixture(self, folder):
        prefix="missing-activation-v2/"
        value=raw_profile(True)
        write(folder,prefix+"chest-5-exact.json",value)
        write(folder,prefix+"chest-5-exact.allocation.json",{"status":"complete"})
        write(folder,prefix+"weighted-proof.json",{"status":"passed","source_sha256":value["source_sha256"]})
        reports=summary.Reports(folder)
        envelope={"kind":"offline-original-cpu-finish-weighted-missing-activation-v2","status":"complete",
            "artifacts_sha256":{"helper":"a"*64},"artifacts_sha256_after":{"helper":"a"*64},
            "supervisor_restored":True,"deferred_terminal_worker_calls":1,"cleanup_failures":[],
            "mode":"exact-missing-key","outcome":"mesh-built-quality-unproven",
            "profile":{"sha256":reports.record(prefix+"chest-5-exact.json")["sha256"]},
            "delegated_sidecar":{"sha256":reports.record(prefix+"chest-5-exact.allocation.json")["sha256"]},
            "activation_proof":{"sha256":reports.record(prefix+"weighted-proof.json")["sha256"]},
            "allocation":{"restored":True,"cleanup_failures":[],"native_grid_returned":True,
                "final_key_proofs":[{"expected_union_sha256":"b"*64,"actual_union_sha256":"b"*64,
                    "exact_key_union":True,"required_blocks":4,"initial_capacity":4,"final_capacity":4,"final_blocks":4}]}}
        for key in ("profile","delegated_sidecar","activation_proof"):
            envelope[key+"_after"]=copy.deepcopy(envelope[key])
        name=prefix+"chest-5-exact.missing-activation.json"
        write(folder,name,envelope)
        return name,envelope

    def test_allocation_envelope_never_grants_mesh_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            name,_=self.activation_fixture(folder)
            row=summary.activation_trial(summary.Reports(folder),name)
            self.assertEqual(row["capacity_evidence"]["final_capacity"],4)
            self.assertIs(row["whole_mesh_quality_authority"],False)
            self.assertNotIn("surface",row)

    def test_allocation_named_input_source_and_restore_closure(self):
        for key,value in (("supervisor_restored",False),("artifacts_sha256_after",{"helper":"c"*64}),
            ("deferred_terminal_worker_calls",0),("profile_after",{"sha256":"c"*64}),
            ("cleanup_failures",[{"private_path":"C:/Users/private"}])):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                name,envelope=self.activation_fixture(folder);envelope[key]=value;write(folder,name,envelope)
                with self.assertRaises(ValueError):summary.activation_trial(summary.Reports(folder),name)
        with tempfile.TemporaryDirectory() as folder:
            name,envelope=self.activation_fixture(folder)
            envelope["profile"]["sha256"]=envelope["profile_after"]["sha256"]="c"*64;write(folder,name,envelope)
            with self.assertRaises(ValueError):summary.activation_trial(summary.Reports(folder),name)

    def test_allocation_growth_or_wrong_exact_union_is_rejected(self):
        for key,value in (("final_capacity",8),("actual_union_sha256","c"*64),("exact_key_union",False),("required_blocks",True)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                name,envelope=self.activation_fixture(folder)
                envelope["allocation"]["final_key_proofs"][0][key]=value;write(folder,name,envelope)
                with self.assertRaises(ValueError):summary.activation_trial(summary.Reports(folder),name)

    def test_failed_trial_stays_failed_without_demanding_successful_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            write(folder,"failed.json",{"status":"failed","failure":{"type":"TypeError"}})
            row=summary.activation_trial(summary.Reports(folder),"failed.json")
            self.assertEqual(row["reported_status"],"failed")
            self.assertEqual(row["failure_type"],"TypeError")
            self.assertNotIn("capacity_evidence",row)


if __name__ == "__main__":
    unittest.main()
