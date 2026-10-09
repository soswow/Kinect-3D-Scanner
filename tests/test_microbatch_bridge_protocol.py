"""Artificial stdlib proofs for complete-bridge workspace timing contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.research import microbatch_bridge_protocol as bridge
from scripts.research import device_loop_workspace_protocol as tokens
from scripts.research import device_loop_workspace as workspace
from scripts.research import gpu_icp_device_loop_protocol as values
from tests.test_gpu_icp_device_loop_protocol import fixture as loop_fixture


def full_desc(value):
    result=copy.deepcopy(value);result["nbytes"]=8*result["shape"][0]*result["shape"][1]
    return result


def fixture():
    common,binding=loop_fixture();contract=workspace.source_contract()
    binding["workspace_source"]=contract
    binding["configuration"]={"graph":True,"chunk_iterations":4}
    binding["artifacts_sha256"].update(contract["artifacts"])
    binding["artifacts_sha256"].update({p:values.sha(values.ROOT/p) for p in bridge.BRIDGE_FILES})
    calls=[]
    for proposal in range(2):
        row=copy.deepcopy(common["rows"][0]);consumed=row["input_binding"]
        row["source_binding"]["setup_reuse"]=copy.deepcopy(contract)
        row["loop_report"]["provenance"]=copy.deepcopy(row["source_binding"])
        full={"source":{name:full_desc(consumed["source"]) for name in ("points","normals","colors")},
            "target":{name:full_desc(consumed["target"] if name!="normals" else consumed["normals"]) for name in ("points","normals","colors")},
            "seed":full_desc(consumed["seed"])}
        for query in row["query_trace"]:query["packet"]=full_desc(query["packet"])
        row["query_trace_sha256"]=hashlib.sha256(values.canonical(row["query_trace"]).encode()).hexdigest()
        call={"proposal_index":proposal,"call_index":0,"input_binding":full,"consumed_input_binding":consumed,
            "source_binding":row["source_binding"],"terminal":row["terminal"],"native_shadow":row["native_shadow"],
            "loop_report":row["loop_report"],"query_trace":row["query_trace"],"query_trace_sha256":row["query_trace_sha256"],
            "complete":True,"input_bytes_unchanged":True,"result":{"accepted":True}}
        calls.append({"proposal_index":proposal,"complete":True,"calls":[call],"result":{"accepted":True},
            "gates":[{"name":"original","complete":True,"result":True,"wall_s_inclusive":.1}]})
    pair={"seeds":[calls[0]["calls"][0]["input_binding"]["seed"],calls[1]["calls"][0]["input_binding"]["seed"]]}
    report={"kind":bridge.KIND,"status":"passed","mode":"audit","binding":binding,"binding_after":binding,
        "pair_binding":pair,"failure":None,"cleanup_failures":[],"cleanup_passed":True,
        "input_bytes_unchanged":True,"original_seed_bytes_unchanged":True,"loaded_owners_unchanged":True,
        "whole_finish_authority":False,"performance_attribution_valid":False,
        "workspace":{"policy":workspace.POLICY,"provenance":contract,"closed":True,"failure":None,
            "active_lane":False,"retained_failed_lanes":0,"template_started":False,"audit":True,
            "jobs":[{"index":index,"closed":True,"failure":None,"report":proposal["calls"][0]["loop_report"]}
                for index,proposal in enumerate(calls)]},
        "shared_cache":{"closed":True,"failure":None},"native_proposals":copy.deepcopy(calls),"gpu_proposals":calls,
        "native_pair_verdict":{"accepted":True,"ambiguous":False},"gpu_pair_verdict":{"accepted":True,"ambiguous":False},
        "pair_quality_passed":True}
    return report,binding,pair


class AuditGuardTests(unittest.TestCase):
    def check(self,report,binding,pair):
        with patch.object(values,"actual_resource_closure"):
            return bridge.validate_report(report,binding,pair)

    def test_repeated_dynamic_call_inputs_are_legitimate_not_coverage_substitution(self):
        report,binding,pair=fixture();_,calls=self.check(report,binding,pair)
        self.assertEqual(len(calls),2)

    def test_inherited_checker_global_require_is_not_mutated(self):
        original=values.require;report,binding,pair=fixture();self.check(report,binding,pair)
        self.assertIs(values.require,original)

    def test_old_proof_kind_or_missing_owner_source_refused(self):
        report,binding,pair=fixture();report["kind"]=values.KIND
        with self.assertRaises(ValueError):self.check(report,binding,pair)
        report,binding,pair=fixture();binding["artifacts_sha256"].pop(bridge.BRIDGE_FILES[0])
        with self.assertRaises(ValueError):self.check(report,binding,pair)

    def test_started_template_unclosed_lane_or_cache_refused(self):
        for key,value in (("template_started",True),("closed",False),("retained_failed_lanes",1),("audit",False)):
            report,binding,pair=fixture();report["workspace"][key]=value
            with self.assertRaises(ValueError):self.check(report,binding,pair)
        report,binding,pair=fixture();report["shared_cache"]["closed"]=False
        with self.assertRaises(ValueError):self.check(report,binding,pair)

    def test_exact_ordered_workspace_report_and_graph4_policy_required(self):
        report,binding,pair=fixture()
        report["workspace"]["jobs"][0]["report"]=copy.deepcopy(report["workspace"]["jobs"][0]["report"])
        report["workspace"]["jobs"][0]["report"]["statistics"]["updates"]+=1
        with self.assertRaises(ValueError):self.check(report,binding,pair)
        report,binding,pair=fixture();report["workspace"]["jobs"].reverse()
        with self.assertRaises(ValueError):self.check(report,binding,pair)
        report,binding,pair=fixture();binding["configuration"]["chunk_iterations"]=1
        with self.assertRaises(ValueError):self.check(report,binding,pair)
        report,binding,pair=fixture()
        report["gpu_proposals"][0]["calls"][0]["consumed_input_binding"]["configuration"]["chunk_iterations"]=1
        with self.assertRaises(ValueError):self.check(report,binding,pair)

    def test_omitted_genuine_proposal_or_dynamic_suffix_refused(self):
        report,binding,pair=fixture();report["gpu_proposals"].pop()
        with self.assertRaises(ValueError):self.check(report,binding,pair)
        report,binding,pair=fixture();report["gpu_proposals"][0]["calls"]=[]
        with self.assertRaises(ValueError):self.check(report,binding,pair)

    def test_gate_outcome_or_final_ambiguity_changed_refused(self):
        report,binding,pair=fixture();report["gpu_proposals"][0]["gates"][0]["result"]=False
        with self.assertRaises(ValueError):self.check(report,binding,pair)
        report,binding,pair=fixture();report["gpu_pair_verdict"]["ambiguous"]=True
        with self.assertRaises(ValueError):self.check(report,binding,pair)

    def test_original_vector_owner_and_consumed_seed_must_agree(self):
        report,binding,pair=fixture();report["gpu_proposals"][0]["calls"][0]["input_binding"]["seed"]["sha256"]="f"*64
        with self.assertRaises(ValueError):self.check(report,binding,pair)

    def test_actual_query_audit_native_id_and_completion_faults_refused(self):
        for field,value in (("complete",False),("input_bytes_unchanged",False),("cleanup_failure","sync")):
            report,binding,pair=fixture();report["gpu_proposals"][0]["calls"][0][field]=value
            with self.assertRaises(ValueError):self.check(report,binding,pair)
        report,binding,pair=fixture();report["gpu_proposals"][0]["calls"][0]["native_shadow"]["correspondence_ids_equal"]=False
        with self.assertRaises(ValueError):self.check(report,binding,pair)

    def test_each_call_full_shadow_but_positive_hit_miss_coverage_aggregate(self):
        report,binding,pair=fixture()
        for index,proposal in enumerate(report["gpu_proposals"]):
            stats=proposal["calls"][0]["loop_report"]["statistics"]
            stats.update(direct_hits=24 if index==0 else 0,audited_hits=24 if index==0 else 0,
                direct_misses=24 if index==1 else 0,audited_misses=24 if index==1 else 0)
        self.check(report,binding,pair)
        for proposal in report["gpu_proposals"]:
            proposal["calls"][0]["loop_report"]["statistics"].update(direct_hits=24,audited_hits=24,direct_misses=0,audited_misses=0)
        with self.assertRaises(ValueError):self.check(report,binding,pair)


class RegisteredTimingTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/"bridge.json"
        self.report,self.binding,self.pair=fixture()
        self.path.write_text(json.dumps(self.report),encoding="utf-8")
        with patch.object(values,"actual_resource_closure"):
            self.token=bridge.validate_bridge_audit(self.path,self.binding,self.pair)

    def test_private_mint_without_full_validator_context_refused(self):
        with self.assertRaises(ValueError):tokens._mint(self.path,self.binding,workspace.source_contract(),[])

    def test_constructed_old_or_modified_permit_refused(self):
        fake=tokens.WorkspaceTimingPermit(**self.token.__dict__)
        with self.assertRaises(ValueError):tokens.registered(fake)
        object.__setattr__(self.token,"calls_json","[]")
        with self.assertRaises(ValueError):tokens.registered(self.token)

    def test_exact_ordered_call_and_terminal_passes(self):
        call=self.report["gpu_proposals"][0]["calls"][0]
        bridge.validate_expected_bridge_call(self.token,0,0,call["input_binding"])
        bridge.validate_bridge_terminal(self.token,0,0,call["input_binding"],call["terminal"])
        consumed=copy.deepcopy(call["consumed_input_binding"])
        consumed["configuration"]["audit_nearest"]=consumed["configuration"]["audit_misses"]=False
        tokens.validate_start(self.token,0,consumed,call["source_binding"])
        self.assertTrue(bridge.validate_complete_bridge(self.token,self.timing_report()))

    def timing_report(self):
        actual=copy.deepcopy(self.report)
        for proposal in actual["gpu_proposals"]:
            for call in proposal["calls"]:
                consumed=call["consumed_input_binding"]
                consumed["configuration"]["audit_nearest"]=consumed["configuration"]["audit_misses"]=False
                call["loop_report"]["input_binding"]=copy.deepcopy(consumed)
                call["loop_report"]["statistics"].update(audited_hits=0,audited_misses=0,flagged_rows=0,packet_bytes=0)
        return actual

    def test_foreign_payload_or_missing_suffix_refused(self):
        payload=copy.deepcopy(self.report["gpu_proposals"][0]["calls"][0]["input_binding"])
        payload["source"]["colors"]["sha256"]="e"*64
        with self.assertRaises(ValueError):bridge.validate_expected_bridge_call(self.token,0,0,payload)
        with self.assertRaises(ValueError):bridge.validate_complete_bridge(self.token,[1])
        actual=self.timing_report();actual["gpu_proposals"][0]["gates"].pop()
        with self.assertRaises(ValueError):bridge.validate_complete_bridge(self.token,actual)

    def test_timing_cannot_hide_audit_or_damaged_lane_metadata(self):
        for field,value in (("closed",False),("failure","damaged")):
            actual=self.timing_report();actual["gpu_proposals"][0]["calls"][0]["loop_report"][field]=value
            with self.assertRaises(ValueError):bridge.validate_complete_bridge(self.token,actual)
        actual=self.timing_report()
        actual["gpu_proposals"][0]["calls"][0]["loop_report"]["statistics"]["audited_hits"]=12
        with self.assertRaises(ValueError):bridge.validate_complete_bridge(self.token,actual)

    def test_timing_graph_and_original_stage_budget_are_bound(self):
        token=self.token
        config={"device":"CUDA:0","cuda_graph":True,"max_points":1_000_000,
            "max_scratch_bytes":256*1024**2,"max_total_bytes":512*1024**2}
        tokens.constructor_authority(token,workspace.source_contract(),config)
        with self.assertRaises(ValueError):tokens.constructor_authority(token,workspace.source_contract(),dict(config,cuda_graph=False))

    def test_proof_rewrite_or_one_ulp_output_refused(self):
        call=self.report["gpu_proposals"][0]["calls"][0];terminal=copy.deepcopy(call["terminal"])
        terminal["fitness"]+=1e-16
        with self.assertRaises(ValueError):bridge.validate_bridge_terminal(self.token,0,0,call["input_binding"],terminal)
        self.path.write_text("{}",encoding="utf-8")
        with self.assertRaises(ValueError):tokens.registered(self.token)


if __name__=="__main__":unittest.main()
