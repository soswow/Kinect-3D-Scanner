"""Owned scalar receipts only; no token mint, numerical import or raw rescan."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import summarize_gpu_icp_owned_bridge as summary
from tests.test_gpu_icp_complete_bridge_summary import reports as old_reports


def reports():
    audit,timing=old_reports()
    proof=summary.owned.source_contract();driver=summary.driver.source_contract();workspace=summary.workspace.source_contract()
    for report,mode in ((audit,"audit"),(timing,"timing")):
        report["kind"]=summary.owned.KIND if mode=="audit" else summary.TIMING_KIND
        binding=report["binding"]
        binding.update(owned_proof_source=proof,owned_driver_source=driver,workspace_source=workspace)
        for contract in (proof,driver,workspace):binding["artifacts_sha256"].update(contract["artifacts"])
        report["binding_after"]=copy.deepcopy(binding)
        report["workspace"].update(policy=summary.workspace.POLICY,provenance=workspace)
        for proposal in report["gpu_proposals"]:
            for call in proposal["calls"]:
                call["source_binding"]["setup_reuse"]=workspace
                call["source_binding"]["artifacts"]=copy.deepcopy(binding["method_source"]["artifacts"])
                call["loop_report"]["provenance"]=copy.deepcopy(call["source_binding"])
        report["workspace"]["jobs"]=[{"index":i,"closed":True,"failure":None,"report":copy.deepcopy(call["loop_report"])}
            for i,call in enumerate(c for p in report["gpu_proposals"] for c in p["calls"])]
    audit["owned_proof"]={"required":False}
    timing["owned_proof"]={"policy":summary.owned.POLICY,"required":True,"closed":True,"failure":None,"hash_initial":"a"*64,"hash_final":"a"*64,
        "share_read_only":True,"decoded_immutable":True,"held_handle_reads":2,"cleanup_failures":[],"old_token_consumed":False}
    return audit,timing


class OwnedSummaryTests(unittest.TestCase):
    def check(self,audit,timing):return summary.compact(audit,timing,"a"*64)

    def test_valid_preserves_original_exact_counts_and_charged_ratio(self):
        result=self.check(*reports())
        self.assertEqual(result["actual_gpu_calls"],2)
        self.assertEqual(result["audit_coverage"]["query_rows"],48)
        self.assertEqual(result["timing"]["gpu_subtotal_plus_cleanup_to_native_wall_ratio"],1.02)
        self.assertEqual(result["owned_proof_receipt"]["held_handle_reads"],2)
        self.assertFalse(result["whole_finish_authority"])

    def test_missing_failed_unclosed_or_foreign_owned_lock_refused(self):
        for key,value in (("policy","foreign-v1"),("required",False),("closed",False),("failure","fault"),("cleanup_failures",["fault"]),
            ("hash_initial","b"*64),("hash_final","b"*64),("share_read_only",False),("decoded_immutable",False),
            ("old_token_consumed",True),("held_handle_reads",3),("held_handle_reads",True)):
            audit,timing=reports();timing["owned_proof"][key]=value
            with self.assertRaises(ValueError):self.check(audit,timing)

    def test_wrong_source_field_and_omitted_new_source_refused(self):
        for field in ("owned_proof_source","owned_driver_source","workspace_source"):
            audit,timing=reports()
            for report in (audit,timing):report["binding"][field]={};report["binding_after"]=copy.deepcopy(report["binding"])
            with self.assertRaises(ValueError):self.check(audit,timing)
        audit,timing=reports()
        for report in (audit,timing):
            report["binding"]["artifacts_sha256"].pop(summary.driver.NEW_FILES[0]);report["binding_after"]=copy.deepcopy(report["binding"])
        with self.assertRaises(ValueError):self.check(audit,timing)

    def test_old_kind_or_audit_timing_lock_claim_refused(self):
        audit,timing=reports();audit["kind"]=summary.original.protocol.KIND
        with self.assertRaises(ValueError):self.check(audit,timing)
        audit,timing=reports();audit["owned_proof"]={"required":True}
        with self.assertRaises(ValueError):self.check(audit,timing)

    def test_original_dynamic_order_terminal_gate_and_query_accounting_still_required(self):
        for mutation in (lambda t:t["gpu_proposals"][0]["calls"][0].update(call_index=1),
            lambda t:t["gpu_proposals"][0]["calls"][0]["terminal"].update(updates=100),
            lambda t:t["gpu_proposals"][0]["gates"][0].update(result=False),
            lambda t:t["gpu_proposals"][0]["calls"][0]["loop_report"]["statistics"].update(audited_hits=1)):
            audit,timing=reports();mutation(timing)
            with self.assertRaises(ValueError):self.check(audit,timing)

    def test_loaded_original_and_private_global_code_owners_pinned(self):
        with patch.object(summary.original,"wall",lambda value:0.):
            with self.assertRaises(ValueError):summary.check_loaded()
        with patch.dict(summary._NAMESPACE,{"wall":lambda value:0.}):
            with self.assertRaises(ValueError):summary.check_loaded()
        with patch.object(summary._PURE_PROTOCOL,"_validate_one",lambda *args:None):
            with self.assertRaises(ValueError):summary.check_loaded()

    def test_in_place_borrowed_guard_code_and_default_mutation_refused(self):
        for function in (summary.guards._validate_one,summary.guards.values.validate_audit_report):
            code=function.__code__;defaults=function.__defaults__
            try:
                function.__code__=(lambda *args:None).__code__
                with self.assertRaises(ValueError):summary.check_loaded()
            finally:function.__code__=code
            try:
                function.__defaults__=(None,)
                with self.assertRaises(ValueError):summary.check_loaded()
            finally:function.__defaults__=defaults

    def test_publication_has_no_registry_change_and_existing_file_refusal(self):
        audit,timing=reports()
        with tempfile.TemporaryDirectory(dir=summary.ROOT/"benchmark-output") as folder:
            folder=Path(folder);a=folder/"a.json";t=folder/"t.json";output=folder/"summary.json"
            a.write_text(json.dumps(audit),encoding="utf-8");ah=summary.sha(a)
            timing["audit_proof"]={"path":str(a),"sha256":ah}
            timing["owned_proof"].update(hash_initial=ah,hash_final=ah)
            t.write_text(json.dumps(timing),encoding="utf-8")
            before=dict(summary.owned._REGISTRY)
            args=SimpleNamespace(audit=a,timing=[t],output=output)
            summary.run(args);record=json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"],"passed");self.assertEqual(record["kind"],summary.KIND)
            self.assertFalse(record["fixed_resource_bytes_rehashed_by_summary"])
            self.assertFalse(record["os_child_exit_independently_verified"])
            self.assertEqual(before,summary.owned._REGISTRY)
            bytes_before=output.read_bytes()
            with self.assertRaises(ValueError):summary.run(args)
            self.assertEqual(output.read_bytes(),bytes_before)

    def test_fresh_import_has_no_numerical_modules(self):
        code="from scripts.research import summarize_gpu_icp_owned_bridge; import sys; assert not any(k.split('.')[0] in ('numpy','cupy','open3d') for k in sys.modules)"
        result=subprocess.run([sys.executable,"-S","-c",code],cwd=summary.ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=="__main__":unittest.main()
