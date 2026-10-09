"""Pure synthetic candidate evidence contracts; no numerical authority."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import compare_gpu_icp_candidate_finishes as quality


def descriptor(rows,columns,dtype="<f8",digest="a"):
    return {"dtype":dtype,"shape":[rows,columns],"sha256":digest*64}


def fixture(mode="audit", calls=True):
    identity=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    cfg={"device":"CUDA:0","max_points":1_000_000,"max_scratch_bytes":268435456,
        "max_total_bytes":536870912,"stages":[[.12,40],[.06,30],[.03,20]],
        "cuda_graph":True,"chunk_iterations":4,"audit_nearest":mode=="audit","audit_misses":mode=="audit"}
    consumed={"source":descriptor(2,3),"target":descriptor(2,3,digest="b"),
        "normals":descriptor(2,3,digest="c"),"seed":descriptor(4,4,digest="d"),"configuration":cfg}
    source={"policy":quality.loop.POLICY,"original_math_guard_inverse":True,
        "stages":[[.12,40],[.06,30],[.03,20]],"graph_option_supported":True,
        "options":["--std=c++11","--fmad=false"],"artifacts":{"shader.cu":"f"*64},
        "generated_sha256":"e"*64,"device":"CUDA:0","cuda_graph":True}
    def full(record):return dict(record,nbytes=8*record["shape"][0]*record["shape"][1])
    inputs={side:{"points":full(consumed[side]),"normals":full(consumed["normals"]),
        "colors":full(descriptor(0,3,digest="e"))} for side in ("source","target")}
    inputs["seed"]=full(consumed["seed"])
    stats={"queries":6,"updates":3,"query_rows":12,"direct_hits":6,"direct_misses":6,
        "audited_hits":6 if mode=="audit" else 0,"audited_misses":6 if mode=="audit" else 0,
        "cpu_ambiguity_rows":0,"solve_blocks":0,"flagged_rows":12 if mode=="audit" else 0,
        "packet_bytes":768 if mode=="audit" else 0}
    trace=[{"query_index":index,"stage":index//2,"radius":(.12,.06,.03)[index//2],
        "target_sha256":consumed["target"]["sha256"],"packet":full(descriptor(2,8)),
        "corrected_ids_sha256":"0"*64} for index in range(6)] if mode=="audit" else []
    pose_descriptor=descriptor(4,4);pose_descriptor["sha256"]=hashlib.sha256(struct.pack("<16d",*(v for row in identity for v in row))).hexdigest()
    terminal={"pose":pose_descriptor,"correspondences":descriptor(2,2,"<i4"),
        "fitness":1.,"inlier_rmse":.001,"queries":6,"updates":3}
    result={"transformation":identity,"fitness":1.,"inlier_rmse":.001,
        "correspondence_mapping":dict(descriptor(2,2,"<i4"),nbytes=16),
        "pose":full(pose_descriptor),"raw_correspondences":dict(descriptor(2,2,"<i4"),nbytes=16)}
    shadow={"passed":True,"correspondence_ids_equal":True,"native":copy.deepcopy(result),
        "candidate":copy.deepcopy(result),"transform_max_abs_delta":0.,"fitness_abs_delta":0.,"rmse_abs_delta":0.}
    row={"call_index":0,"pair_index":0,"pair":[0,1],"proposal_index":0,
        "input_binding":inputs,"consumed_input_binding":consumed,"source_binding":source,
        "terminal":terminal,"result":result,"loop_report":{"closed":True,"failure":None,
            "input_binding":consumed,"provenance":source,"statistics":stats},
        "query_trace":trace,"query_trace_sha256":hashlib.sha256(quality.canonical(trace).encode()).hexdigest(),
        "native_shadow":shadow if mode!="measure" else {"collected":False},
        "authorization":{"cpu_first":mode=="shadow"},"input_bytes_unchanged":True,"complete":True,"cleanup_failures":[]}
    if mode=="shadow":row["cpu_first_certificate"]={"input":copy.deepcopy(consumed),"completed":True,
        "input_bytes_unchanged":True,"result":copy.deepcopy(shadow["native"])}
    rows=[row] if calls else []
    workspace={"closed":True,"failure":None,"template_started":False,"active_lane":False,
        "jobs":[{"index":0,"closed":True,"report":row["loop_report"]}]} if calls else None
    profile={"input_sha256":"1"*64,"selected_indices":[0],"seed":0,"settings":{"bundle_adjustment":True},
        "pipeline_options":{"tracking":"legacy"},"thread_policy":{"open3d":20,"opencv":20,"omp":"8"},
        "source_sha256":"2"*64,"accepted_indices":[0],"poses":[{"index":0,"camera_to_world":copy.deepcopy(identity)}],
        "mesh_built":True,"finish_requested":True,"pose_seeds_used":False,
        "input_changed_during_profile":False,"source_changed_during_profile":False,
        "fragment_reconnection":{"applied":False,"fragments":[{"id":0,"connected":True,"frame_indices":[0],"context_frame_indices":[]}],
            "verified_bridges":[],"ambiguous_pairs":[],"invalid_indices":[],"unassigned_indices":[],"excluded_frames":[],
            "connected_fragments":[0],"unconnected_fragments":[],"recent_reference_links":[],"loop_closures":[],
            "pair_budget":256,"fragment_limit":32,"budget_limited":False}}
    binding={"source_sha256":"2"*64,"artifacts_sha256":dict(source["artifacts"]),"fixed_files":{},"runtime":{"device":"CUDA:0"},"method_source":source}
    contract={"configuration":dict(quality.CONFIGURATION),"candidate_method":{
        "policy":"bridge-only-original-device-loop-minimal-candidate-v1","configuration":dict(quality.CONFIGURATION),
        "historical_trajectory_authority":False,"general_domain_proof":False}}
    pose_record=copy.deepcopy(pose_descriptor);pose_record["values"]=copy.deepcopy(identity)
    final_rows=[{"index":0,"camera_to_world":pose_record}]
    inventory={"pose_convention":"camera_to_world","length_unit":"metres","captured_after_successful_build":True,
        "rows":final_rows,"pose_inventory_sha256":hashlib.sha256(quality.canonical(final_rows).encode()).hexdigest()}
    registration={"mode":mode,"complete":True,"restored":True,"closed":True,"failure":None,
        "cleanup_failures":[],"successful_builds":1,"default_evidence":True,"configuration":dict(quality.CONFIGURATION),
        "calls":rows,"workspace":workspace,"cache_receipts":[{"closed":True,"failure":None,"active_streams":0,"owned_numeric_bytes":96}] if calls else [],"final_pose_inventory":inventory}
    return {"kind":quality.PREFIX+mode+"-v1","mode":mode,"status":"passed","failure":None,
        "cleanup_passed":True,"cleanup_failures":[],"binding":binding,"binding_after":copy.deepcopy(binding),
        "candidate_source":contract,"candidate_source_after":copy.deepcopy(contract),"registration":registration,
        "calls":rows,"scope_binding":{"checkpoint":"exact-fresh-live"},"profile":profile,"final_pose_inventory":inventory,
        "method_qualification":{"required":mode=="measure","closed":True,"failure":None,
            "report_sha256":"8"*64,"report_sha256_after":"8"*64,
            "historical_trajectory_authority":False,"general_domain_authority":False}}


def native_fixture(candidate):
    native=copy.deepcopy(candidate)
    native.update(kind=quality.PREFIX+"native-v1",mode="native")
    native["registration"]={"complete":True,"restored":True,"closed":True,"failure":None,
        "cleanup_failures":[],"original_build_calls":1,"registration_evidence_available":False,"graphs":[]}
    return native


class CandidateQualityTests(unittest.TestCase):
    def test_positive_exhaustive_actual_hit_miss_and_terminal_evidence(self):
        result=quality.report_evidence(fixture(),positive=True)
        self.assertEqual((result["actual_gpu_calls"],result["query_rows"]),(1,12))
        self.assertTrue(result["exhaustive_nn_audit"])

    def test_shadow_is_actual_cpu_terminal_only_without_nn_claim(self):
        result=quality.report_evidence(fixture("shadow"))
        self.assertFalse(result["exhaustive_nn_audit"])
        self.assertEqual(result["actual_cpu_terminal_shadows"],1)
        with self.assertRaisesRegex(ValueError,"positive exhaustive"):
            quality.report_evidence(fixture("shadow"),positive=True)

    def test_actual_controller_nested_source_envelope_and_default_evidence(self):
        from scripts.research import profile_gpu_icp_finish_candidate as controller
        report=fixture();contract=controller.source_contract()
        self.assertIn("candidate_method",contract)
        report["candidate_source"]=contract;report["candidate_source_after"]=copy.deepcopy(contract)
        quality.report_evidence(report)
        report["registration"]["default_evidence"]=False
        with self.assertRaisesRegex(ValueError,"fully closed"):
            quality.report_evidence(report,positive=True)

    def test_flat_source_envelope_cannot_qualify_actual_controller_method(self):
        report=fixture();report["candidate_source"]=report["candidate_source"]["candidate_method"]
        report["candidate_source_after"]=copy.deepcopy(report["candidate_source"])
        with self.assertRaisesRegex(ValueError,"source contract"):
            quality.report_evidence(report)
        report=fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"audit.json";path.write_text(json.dumps(report),encoding="utf-8")
            with patch.object(quality,"check_sources"):
                with self.assertRaisesRegex(ValueError,"actual current controller"):
                    quality.qualify_candidate_method(path,report["binding"])

    def test_measure_has_no_unperformed_cpu_shadows(self):
        self.assertEqual(quality.report_evidence(fixture("measure"))["actual_cpu_terminal_shadows"],0)
        report=fixture("measure");report["calls"][0]["native_shadow"]={"passed":True}
        with self.assertRaisesRegex(ValueError,"unperformed"):
            quality.report_evidence(report)

    def test_measure_cannot_omit_or_change_method_qualification(self):
        for key,value in (("required",False),("closed",False),("report_sha256_after","9"*64)):
            report=fixture("measure");report["method_qualification"][key]=value
            with self.assertRaisesRegex(ValueError,"method qualification receipt"):
                quality.report_evidence(report)

    def test_zero_gpu_is_valid_negative_and_cannot_qualify_method(self):
        report=fixture(calls=False)
        self.assertEqual(quality.report_evidence(report)["actual_gpu_calls"],0)
        with self.assertRaisesRegex(ValueError,"positive exhaustive"):
            quality.report_evidence(report,positive=True)

    def test_actual_input_normals_and_seed_must_equal_consumed_values(self):
        for key in ("normals","seed"):
            report=fixture()
            record=report["calls"][0]["input_binding"]["target"]["normals"] if key=="normals" else report["calls"][0]["input_binding"]["seed"]
            record["sha256"]="9"*64
            with self.assertRaisesRegex(ValueError,"normals/unrounded seed"):
                quality.report_evidence(report)

    def test_query_trace_scale_order_and_counter_coverage_fail_closed(self):
        for kind in ("trace","counter","auditflag"):
            report=fixture();row=report["calls"][0]
            if kind=="trace":row["query_trace"][0]["stage"]=2
            elif kind=="counter":row["loop_report"]["statistics"]["audited_misses"]-=1
            else:row["consumed_input_binding"]["configuration"]["audit_nearest"]=False
            with self.assertRaises(ValueError):quality.report_evidence(report)

    def test_declared_shadow_deltas_are_recomputed_and_ids_exact(self):
        for kind in ("pose","ids","delta"):
            report=fixture();shadow=report["calls"][0]["native_shadow"]
            if kind=="pose":shadow["candidate"]["transformation"][0][3]=.01
            elif kind=="ids":shadow["candidate"]["correspondence_mapping"]["sha256"]="9"*64
            else:shadow["fitness_abs_delta"]=1e-12
            with self.assertRaisesRegex(ValueError,"metrics/canonical"):
                quality.report_evidence(report)

    def test_cpu_first_certificate_and_actual_result_receipts_cannot_disagree(self):
        for kind in ("certificate","candidate","packet","terminal"):
            report=fixture("shadow" if kind=="certificate" else "audit");row=report["calls"][0]
            if kind=="certificate":row["cpu_first_certificate"]["input"]["seed"]["sha256"]="9"*64
            elif kind=="candidate":row["native_shadow"]["candidate"]["raw_correspondences"]["sha256"]="9"*64
            elif kind=="packet":row["query_trace"][0]["packet"]["nbytes"]-=1
            else:row["terminal"]["pose"]["sha256"]="9"*64
            with self.assertRaises(ValueError):quality.report_evidence(report)

    def test_owner_workspace_job_suffix_and_cleanup_must_match(self):
        for kind in ("job","active","cache","fault"):
            report=fixture();registration=report["registration"]
            if kind=="job":registration["workspace"]["jobs"][0]["index"]=1
            elif kind=="active":registration["workspace"]["active_lane"]=True
            elif kind=="cache":registration["cache_receipts"][0]["closed"]=False
            else:report["cleanup_failures"].append("stream fault")
            with self.assertRaises(ValueError):quality.report_evidence(report)

    def test_actual_math_source_and_helper_must_match_closed_binding(self):
        for kind in ("method","artifact"):
            report=fixture()
            if kind=="method":report["binding"]["method_source"]=dict(report["binding"]["method_source"],generated_sha256="9"*64)
            else:report["binding"]["artifacts_sha256"]={}
            report["binding_after"]=copy.deepcopy(report["binding"])
            with self.assertRaisesRegex(ValueError,"evaluated helper/shader"):
                quality.report_evidence(report)

    def test_typed_fixed_configuration_rejects_loosened_or_boolean_alias(self):
        for key,value in (("max_jobs",5000),("device",False),("cuda_graph",1)):
            cfg=dict(quality.CONFIGURATION);cfg[key]=value
            with self.assertRaises(ValueError):quality.configuration_contract(cfg)

    def test_new_method_registry_rejects_constructed_or_damaged_token(self):
        report=fixture()
        report["candidate_source"]=quality.current_candidate_contract()
        report["candidate_source_after"]=copy.deepcopy(report["candidate_source"])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"audit.json";path.write_text(json.dumps(report),encoding="utf-8")
            with patch.object(quality,"check_sources"):
                token=quality.qualify_candidate_method(path,report["binding"])
                quality.validate_candidate_method(token,binding=report["binding"],configuration=dict(quality.CONFIGURATION))
                foreign=quality.CandidateMethodQualification(**token.__dict__)
                with self.assertRaisesRegex(ValueError,"constructed"):
                    quality.assert_registered_candidate_qualification(foreign)
                object.__setattr__(token,"configuration_json","{}")
                with self.assertRaisesRegex(ValueError,"unchanged"):
                    quality.assert_registered_candidate_qualification(token)

    def test_method_authorizes_new_bounded_inputs_not_exact_audit_replay(self):
        report=fixture()
        report["candidate_source"]=quality.current_candidate_contract()
        report["candidate_source_after"]=copy.deepcopy(report["candidate_source"])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"audit.json";path.write_text(json.dumps(report),encoding="utf-8")
            with patch.object(quality,"check_sources"):
                token=quality.qualify_candidate_method(path,report["binding"])
                row=fixture("measure")["calls"][0];row["consumed_input_binding"]["seed"]["sha256"]="9"*64
                quality.validate_candidate_call(token,consumed=row["consumed_input_binding"],source=row["source_binding"],configuration=dict(quality.CONFIGURATION))
                row["consumed_input_binding"]["source"]["shape"][0]=1_000_001
                with self.assertRaises(ValueError):
                    quality.validate_candidate_call(token,consumed=row["consumed_input_binding"],source=row["source_binding"],configuration=dict(quality.CONFIGURATION))

    def test_final_observables_do_not_invent_native_graph_or_gate_history(self):
        candidate=fixture("shadow");native=native_fixture(candidate)
        result=quality.observable_quality(native,candidate)
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["retained_graph_comparison_available"])
        self.assertIsNone(result["retained_graph_equal"])
        self.assertFalse(result["native_ransac_history_equivalence_claimed"])

    def test_final_original_view_membership_pose_and_current_source_scope(self):
        for kind in ("view","pose","source","checkpoint"):
            candidate=fixture();native=native_fixture(candidate)
            if kind=="view":candidate["profile"]["accepted_indices"]=[1]
            elif kind=="pose":
                candidate["profile"]["poses"][0]["camera_to_world"][0][3]=.0006
                inventory=candidate["final_pose_inventory"]
                record=inventory["rows"][0]["camera_to_world"];record["values"][0][3]=.0006
                record["sha256"]=hashlib.sha256(struct.pack("<16d",*(v for row in record["values"] for v in row))).hexdigest()
                inventory["pose_inventory_sha256"]=hashlib.sha256(quality.canonical(inventory["rows"]).encode()).hexdigest()
            elif kind=="source":candidate["profile"]["source_sha256"]="9"*64
            else:candidate["scope_binding"]={"checkpoint":"other"}
            if kind=="pose":self.assertFalse(quality.observable_quality(native,candidate)["checks"]["pose_bounds"])
            else:
                with self.assertRaises(ValueError):quality.observable_quality(native,candidate)

    def test_portable_current_source_does_not_repin_historical_core(self):
        candidate=fixture();native=native_fixture(candidate)
        self.assertNotEqual(candidate["binding"]["source_sha256"],"07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c")
        self.assertTrue(quality.observable_quality(native,candidate)["checks"]["pose_bounds"])

    def test_original_final_inventory_hash_and_retained_graph_discreteness(self):
        candidate=fixture();native=native_fixture(candidate)
        candidate["final_pose_inventory"]["rows"][0]["camera_to_world"]["sha256"]="9"*64
        with self.assertRaises(ValueError):quality.observable_quality(native,candidate)
        candidate=fixture();native=native_fixture(candidate)
        graph={"complete":True,"context":{"owner":"refinement.propose_poses","node_raw_view_indices":[0,1]},
            "options":{"edge_prune_threshold":.25},"after":{"nodes":[{},{}],"edges":[
                {"source_node":0,"target_node":1,"uncertain":True,"confidence":.25}]}}
        native["registration"].update(registration_evidence_available=True,graphs=[copy.deepcopy(graph)])
        candidate["registration"]["graphs"]=[copy.deepcopy(graph)]
        self.assertTrue(quality.observable_quality(native,candidate)["retained_graph_equal"])
        candidate["registration"]["graphs"][0]["after"]["edges"][0]["confidence"]=.249999999999
        with self.assertRaisesRegex(ValueError,"retained graph"):
            quality.observable_quality(native,candidate)

    def test_common_helper_and_native_resources_must_match(self):
        for kind in ("artifacts_sha256","fixed_files"):
            candidate=fixture();native=native_fixture(candidate)
            native["binding"][kind]["shared"]="a"*64
            candidate["binding"][kind]["shared"]="b"*64
            native["binding_after"]=copy.deepcopy(native["binding"])
            candidate["binding_after"]=copy.deepcopy(candidate["binding"])
            with self.assertRaisesRegex(ValueError,"Common current"):
                quality.observable_quality(native,candidate)

    def test_published_witness_change_fails_without_claiming_internal_graph(self):
        candidate=fixture();native=native_fixture(candidate)
        for report in (candidate,native):
            report["profile"]["selected_indices"]=[0,1]
            fragments=report["profile"]["fragment_reconnection"]
            fragments["fragments"].append({"id":1,"connected":True,"frame_indices":[1],"context_frame_indices":[]})
            fragments["connected_fragments"]=[0,1]
            fragments["verified_bridges"]=[{"source":0,"target":1,"support":[[0,1]],
                "validation_scope":"independent camera pairs","connected_to_scan":True,"visual_constraint":True}]
        result=quality.observable_quality(native,candidate)
        self.assertFalse(result["published_memberships_are_internal_optimizer_trace"])
        self.assertFalse(result["retained_graph_comparison_available"])
        candidate["profile"]["fragment_reconnection"]["verified_bridges"][0]["support"]=[[1,0]]
        with self.assertRaisesRegex(ValueError,"witness/frontier"):
            quality.observable_quality(native,candidate)

    def test_published_membership_ids_and_missing_schema_fail_or_disclose(self):
        candidate=fixture();native=native_fixture(candidate)
        candidate["profile"]["fragment_reconnection"]["excluded_frames"]=[999]
        with self.assertRaisesRegex(ValueError,"membership IDs"):
            quality.observable_quality(native,candidate)
        candidate=fixture();native=native_fixture(candidate)
        del candidate["profile"]["fragment_reconnection"]["loop_closures"]
        with self.assertRaisesRegex(ValueError,"memberships changed"):
            quality.observable_quality(native,candidate)

    def test_loaded_pure_guard_rejects_code_and_policy_alias_change(self):
        guard=quality.LoadedOwners()
        old=quality.native_shadow.__code__
        try:
            quality.native_shadow.__code__=(lambda row:None).__code__
            with self.assertRaisesRegex(ValueError,"owner/code/default"):guard.check()
        finally:quality.native_shadow.__code__=old
        with patch.object(quality,"CONFIGURATION",dict(quality.CONFIGURATION)):
            with self.assertRaisesRegex(ValueError,"owner/code/default"):guard.check()

    def test_relative_artifact_paths_reject_traversal_and_aliases(self):
        for artifacts in ({"../x":"a"*64},{"C:/x":"a"*64},{"a\\b":"a"*64,"a/b":"a"*64}):
            with self.assertRaises(ValueError):quality.normalized_artifacts({"artifacts_sha256":artifacts})


if __name__=="__main__":unittest.main()
