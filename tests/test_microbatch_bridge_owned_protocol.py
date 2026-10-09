"""Artificial stdlib-only owned-proof contracts; no CUDA/native evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import microbatch_bridge_owned_protocol as p
from scripts.research import gpu_icp_device_loop_protocol as values
from tests.test_microbatch_bridge_protocol import fixture


class FakeLock:
    def __init__(self,path):self.path=path;self.handle=object();self.api=object();self.data=Path(path).read_bytes();self.closed=False;self.reads=0;self.close_error=None
    def read(self):
        if self.closed:raise ValueError("closed")
        self.reads+=1;return self.data
    def close(self):
        if self.close_error is not None:raise self.close_error
        self.closed=True


class OwnedProofTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/"proof.json"
        self.report,self.binding,self.pair=fixture()
        with patch.object(values,"actual_resource_closure"):
            self.contract,self.calls=p.original.validate_report(self.report,self.binding,self.pair)
        self.report["kind"]=p.KIND
        self.path.write_text(json.dumps(self.report),encoding="utf-8")
        self.mocks=[patch.object(p,"WindowsProofLock",FakeLock),patch.object(p,"_OwnerGuard",lambda:SimpleNamespace(check=lambda:None)),
            patch.object(p,"validate_report",return_value=(self.contract,self.calls))]
        for mock in self.mocks:mock.start();self.addCleanup(mock.stop)
        self.token=p.validate_bridge_audit(self.path,self.binding,self.pair)
        self.addCleanup(lambda:p.close_permit(self.token) if not p._state(self.token)["closed"] else None)

    def timing(self):
        result=copy.deepcopy(self.report)
        for proposal in result["gpu_proposals"]:
            for call in proposal["calls"]:
                consumed=call["consumed_input_binding"]
                consumed["configuration"]["audit_nearest"]=consumed["configuration"]["audit_misses"]=False
                call["loop_report"]["input_binding"]=copy.deepcopy(consumed)
                call["loop_report"]["statistics"].update(audited_hits=0,audited_misses=0,flagged_rows=0,packet_bytes=0)
        return result

    def test_proof_is_read_once_despite_repeated_ordered_checks(self):
        row=self.timing()["gpu_proposals"][0]["calls"][0]
        for _ in range(12):
            p.validate_expected_bridge_call(self.token,0,0,row["input_binding"])
            p.validate_terminal(self.token,0,row["consumed_input_binding"],row["source_binding"],row["terminal"])
        self.assertEqual(p._state(self.token)["lock"].reads,1)
        p.close_permit(self.token)
        self.assertEqual(p.permit_report(self.token)["held_handle_reads"],2)

    def test_cached_decoded_values_are_recursively_immutable_and_detached(self):
        state=p._state(self.token)
        with self.assertRaises(TypeError):state["refs"].indexed[(0,0)]=99
        self.report["gpu_proposals"][0]["calls"][0]["terminal"]["queries"]=80
        self.assertEqual(p._call(self.token,0)["terminal"]["queries"],6)
        copied=p._call(self.token,0);copied["terminal"]["queries"]=82
        self.assertEqual(p._call(self.token,0)["terminal"]["queries"],6)

    def test_old_constructed_or_modified_token_refused(self):
        for fake in (None,SimpleNamespace(report_path=str(self.path)),p.OwnedBridgeTimingPermit(**self.token.__dict__)):
            with self.assertRaises(ValueError):p.registered(fake)
        object.__setattr__(self.token,"report_path","other")
        with self.assertRaises(ValueError):p.registered(self.token)
        object.__setattr__(self.token,"report_path",str(self.path))

    def test_replaced_reference_owner_refused(self):
        state=p._state(self.token);old=state["refs"]
        state["refs"]=copy.copy(old)
        with self.assertRaises(ValueError):p.registered(self.token)
        state["refs"]=old

    def test_original_exact_order_seed_and_terminal_are_still_checked(self):
        actual=self.timing();call=actual["gpu_proposals"][0]["calls"][0]
        p.validate_start(self.token,0,call["consumed_input_binding"],call["source_binding"])
        self.assertTrue(p.validate_complete_bridge(self.token,actual))
        changed=copy.deepcopy(call["input_binding"]);changed["seed"]["sha256"]="f"*64
        with self.assertRaises(ValueError):p.validate_expected_bridge_call(self.token,0,0,changed)
        wrong=copy.deepcopy(call["terminal"]);wrong["fitness"]+=1e-16
        with self.assertRaises(ValueError):p.validate_bridge_terminal(self.token,0,0,call["input_binding"],wrong)
        with self.assertRaises(ValueError):p.validate_expected_bridge_call(self.token,9,0,call["input_binding"])

    def test_complete_original_gates_and_lane_cleanup_cannot_be_omitted(self):
        for mutate in (lambda a:a["gpu_proposals"][0]["gates"].pop(),
            lambda a:a["gpu_proposals"][0]["calls"][0]["loop_report"].update(closed=False),
            lambda a:a["gpu_pair_verdict"].update(ambiguous=True)):
            actual=self.timing();mutate(actual)
            with self.assertRaises(ValueError):p.validate_complete_bridge(self.token,actual)

    def test_initial_and_terminal_hash_change_is_failure_and_handle_still_closes(self):
        state=p._state(self.token);state["lock"].data=b"changed"
        with self.assertRaises(ValueError):p.close_permit(self.token)
        receipt=p.permit_report(self.token)
        self.assertTrue(receipt["closed"]);self.assertIsNotNone(receipt["failure"])
        self.assertNotEqual(receipt["hash_initial"],receipt["hash_final"])
        with self.assertRaises(ValueError):p.registered(self.token)

    def test_primary_failure_preserved_and_cleanup_independently_attempted(self):
        state=p._state(self.token);state["lock"].data=b"changed"
        primary=RuntimeError("GPU primary")
        with self.assertRaises(RuntimeError) as caught:p.close_permit(self.token,primary=primary)
        self.assertIs(caught.exception,primary);self.assertIsInstance(primary.__cause__,ValueError)
        self.assertTrue(state["lock"].closed)

    def test_close_failure_retains_owner_for_cleanup_retry(self):
        state=p._state(self.token);error=OSError("CloseHandle")
        state["lock"].close_error=error
        with self.assertRaises(OSError):p.close_permit(self.token)
        self.assertFalse(state["closed"]);self.assertFalse(state["lock"].closed)
        with self.assertRaises(ValueError):p.registered(self.token)
        state["lock"].close_error=None;p.close_permit(self.token)
        self.assertTrue(state["closed"]);self.assertIsNotNone(state["failure"])

    def test_closed_owner_is_idempotent_and_never_reauthorizes(self):
        p.close_permit(self.token);p.close_permit(self.token)
        with self.assertRaises(ValueError):p.registered(self.token)

    def test_float_signed_zero_and_numeric_types_survive_frozen_cache(self):
        value={"a":[-0.,0.,1,1.,True]};saved=p._freeze(value);value["a"].append(2)
        self.assertEqual(p._thaw(saved)["a"],[-0.,0.,1,1.,True])
        self.assertNotEqual(p._freeze(-0.),p._freeze(0.));self.assertNotEqual(p._freeze(1),p._freeze(1.))
        with self.assertRaises(ValueError):p._freeze(float("nan"))

    def test_late_loaded_owner_damage_refuses_math(self):
        state=p._state(self.token);state["guard"].check=lambda:(_ for _ in ()).throw(ValueError("owner replaced"))
        with self.assertRaises(ValueError):p.registered(self.token)
        with self.assertRaises(ValueError):p.close_permit(self.token)
        self.assertTrue(state["lock_owner"].closed);self.assertIsNotNone(state["failure"])

    def test_each_frozen_reference_field_owner_is_checked(self):
        state=p._state(self.token);refs=state["refs"]
        for name in ("workspace","calls","report","indexed"):
            old=getattr(refs,name)
            object.__setattr__(refs,name,object())
            with self.assertRaises(ValueError):p.registered(self.token)
            object.__setattr__(refs,name,old)

    def test_damaged_cache_still_closes_original_handle_and_preserves_primary(self):
        state=p._state(self.token);refs=state["refs"];old=refs.calls
        object.__setattr__(refs,"calls",())
        primary=RuntimeError("original GPU fault")
        with self.assertRaises(RuntimeError) as caught:p.close_permit(self.token,primary)
        self.assertIs(caught.exception,primary);self.assertIsInstance(primary.__cause__,ValueError)
        self.assertTrue(state["lock_owner"].closed);object.__setattr__(refs,"calls",old)

    def test_lock_replacement_and_handle_damage_refuse_work_then_close_original(self):
        state=p._state(self.token);old=state["lock"]
        state["lock"]=FakeLock(self.path)
        with self.assertRaises(ValueError):p.registered(self.token)
        state["lock"]=old;handle=old.handle;old.handle=object()
        with self.assertRaises(ValueError):p.registered(self.token)
        with self.assertRaises(ValueError):p.close_permit(self.token)
        self.assertTrue(old.closed);self.assertIs(old.handle,handle)


class SourceAndLockTests(unittest.TestCase):
    def test_new_complete_guard_accepts_only_its_own_closed_source_family(self):
        from scripts.research import device_loop_owned_workspace as workspace
        report,binding,pair=fixture();contract=workspace.source_contract()
        binding["artifacts_sha256"]=copy.deepcopy(binding["artifacts_sha256"])
        binding["workspace_source"]=contract;binding["owned_proof_source"]=p.source_contract()
        binding["artifacts_sha256"].update(contract["artifacts"])
        report["kind"]=p.KIND;report["workspace"].update(policy=workspace.POLICY,provenance=contract)
        for proposal in report["gpu_proposals"]:
            for call in proposal["calls"]:
                call["source_binding"]["setup_reuse"]=contract
                call["loop_report"]["provenance"]=copy.deepcopy(call["source_binding"])
        with patch.object(values,"actual_resource_closure"):
            actual,calls=p.validate_report(report,binding,pair)
            self.assertEqual(actual,contract);self.assertEqual(len(calls),2)
            report["kind"]=p.original.KIND
            with self.assertRaises(ValueError):p.validate_report(report,binding,pair)
            report["kind"]=p.KIND;binding["artifacts_sha256"].pop(p.NEW_FILES[0])
            with self.assertRaises(ValueError):p.validate_report(report,binding,pair)

    def test_actual_loaded_guard_rejects_foreign_function_globals(self):
        guard=p._OwnerGuard();guard.check()
        original=p.original.validate_complete_bridge
        replacement=FunctionType(original.__code__,dict(original.__globals__),original.__name__,original.__defaults__)
        with patch.object(p.original,"validate_complete_bridge",replacement):
            with self.assertRaises(ValueError):guard.check()
            with self.assertRaises(ValueError):p._OwnerGuard()

    def test_actual_loaded_guard_rejects_late_code_and_default_damage(self):
        guard=p._OwnerGuard();original=p.close_permit
        code=original.__code__;defaults=original.__defaults__
        try:
            original.__defaults__=("changed",)
            with self.assertRaises(ValueError):guard.check()
            original.__defaults__=defaults
            original.__code__=p.permit_report.__code__
            with self.assertRaises(ValueError):guard.check()
        finally:original.__code__=code;original.__defaults__=defaults

    def test_actual_loaded_guard_rejects_class_alias_replacement(self):
        guard=p._OwnerGuard()
        with patch.object(p,"WindowsProofLock",object):
            with self.assertRaises(ValueError):guard.check()

    def test_source_derivations_recover_original_complete_guard_bodies(self):
        guard=p._derived("validate_report",(("from scripts.research import device_loop_workspace as workspace",
            "from scripts.research import device_loop_owned_workspace as workspace"),))
        self.assertTrue(callable(guard));self.assertIsNot(guard,p.original.validate_report)
        self.assertEqual(p.original.KIND,"gpu-icp-complete-bridge-proposal-audit-v1")

    def test_no_arbitrary_or_mutable_nonjson_cache_values(self):
        for value in (object(),{1:"key"},(1,2),float("inf")):
            with self.assertRaises(ValueError):p._freeze(value)

    def test_initial_failed_validation_closes_owned_read_handle(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bad.json";path.write_text("{}",encoding="utf-8");locks=[]
            def make(path):lock=FakeLock(path);locks.append(lock);return lock
            with (patch.object(p,"WindowsProofLock",make),patch.object(p,"_OwnerGuard",lambda:SimpleNamespace(check=lambda:None)),
                patch.object(p,"validate_report",side_effect=ValueError("bad audit"))):
                with self.assertRaises(ValueError):p.validate_bridge_audit(path,{}, {})
            self.assertTrue(locks[0].closed)


if __name__=="__main__":unittest.main()
