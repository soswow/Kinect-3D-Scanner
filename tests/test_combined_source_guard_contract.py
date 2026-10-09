"""Stdlib-only causal source-check, owned dispatch and failure contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import combined_source_guard_adapter as adapter
from scripts.research import validate_combined_source_guard_timing as guard
from scripts.research import benchmark_combined_source_guard_timing as driver
from scripts.research.archive.validate_uniform_grid_proof import GridProofError
from tests.test_combined_sync_timing_contract import fake_iteration,authorities


def source_statistics():
    return {"source_checks":0,"iteration_config_checks":0,"preenqueue_calls":0,
        "source_check_s":0.,"iteration_config_check_s":0.,"preenqueue_bookkeeping_s":0.}


class AuthorityContracts(unittest.TestCase):
    def setUp(self):guard._ACTIVE.clear()
    def token(self,mode="whole-file-sha",pins=None):
        a,b,binding,fixture,cfg,cache=authorities()
        binding["combined_sync_contract"]={"original_math_bytes_unchanged":True}
        a=guard.parent.combined.CombinedSyncProofAuthority("cs","0"*64,"cb","0"*64,
            json.dumps(binding),"f"*64,frozenset({"t"}),((guard.SOURCE,"0"*64),))
        old=guard.parent.CombinedSyncTimingAuthority(a,b,json.dumps(binding),(),"f"*64,())
        with patch.object(guard.parent,"validate_activation",return_value=old),patch.object(guard,"file_hash",return_value="0"*64):
            token=guard.validate_source_guard_authority(old,mode,pins or {name:"0"*64 for name in guard.ARTIFACTS})
            guard.validate_source_guard_activation(token,old,mode)
        return token,old
    def test_both_declared_modes_register_separate_exact_tokens(self):
        for mode in guard.MODES:
            token,old=self.token(mode)
            self.assertIs(type(token),guard.SourceGuardTimingAuthority)
            self.assertIs(token.original_timing_authority,old)
            self.assertEqual(token.source_contract_json,'{"original_math_bytes_unchanged":true}')
    def test_equal_but_unregistered_token_rejected(self):
        token,old=self.token()
        with self.assertRaises(GridProofError):guard.validate_source_guard_activation(copy.copy(token),old,"whole-file-sha")
    def test_mode_and_original_token_identity_cannot_drift(self):
        token,old=self.token()
        with self.assertRaises(GridProofError):guard.validate_source_guard_activation(token,old,"original-ast")
        with self.assertRaises(GridProofError):guard.validate_source_guard_activation(token,copy.copy(old),"whole-file-sha")
    def test_current_original_and_new_source_bytes_required(self):
        token,old=self.token()
        for changed in (guard.SOURCE,guard.ARTIFACTS[0]):
            def digest(path):return "1"*64 if Path(path)==guard.ROOT/changed else "0"*64
            with patch.object(guard,"file_hash",side_effect=digest),self.assertRaises(GridProofError):
                guard.validate_source_guard_activation(token,old,"whole-file-sha")
    def test_missing_new_source_dependency_rejected(self):
        with self.assertRaises(GridProofError):self.token(pins={guard.ARTIFACTS[0]:"0"*64})
    def test_wrong_old_token_and_unknown_mode_rejected(self):
        with self.assertRaises(GridProofError):guard.validate_source_guard_authority(None,"whole-file-sha",{})
        token,old=self.token()
        with self.assertRaises(GridProofError):guard.validate_source_guard_authority(old,"skip",{})


class GuardContracts(unittest.TestCase):
    def instance(self,mode="whole-file-sha"):
        obj=object.__new__(adapter.SourceGuardResidentICP)
        obj.source_check_mode=mode
        obj.source_guard_authority=SimpleNamespace(source_check_mode=mode,original_source_sha256="0"*64,
            source_contract_json='{"math":"same"}')
        obj.source_guard_statistics=source_statistics()
        obj.contract={"math":"same"}
        obj._source_guard_generator_state=adapter.generator_state()
        return obj
    def test_sha_check_preserves_configuration_and_cached_contract(self):
        obj=self.instance()
        with (patch.object(adapter.parent.CombinedSyncTimedResidentICP,"_check_configuration") as original,
                patch.object(adapter.Path,"read_bytes",return_value=b"original"),
                patch.object(adapter.hashlib,"sha256",return_value=SimpleNamespace(hexdigest=lambda:"0"*64))):
            obj._check_configuration(check_source=True)
        original.assert_called_once_with(check_source=False)
        self.assertEqual(obj.source_guard_statistics["source_checks"],1)
    def test_mutated_contract_whole_source_and_generator_stop(self):
        for fault in ("contract","source","generator"):
            obj=self.instance()
            if fault=="contract":obj.contract["math"]="changed"
            digest="1"*64 if fault=="source" else "0"*64
            state=("changed",) if fault=="generator" else adapter.generator_state()
            with (patch.object(adapter.parent.CombinedSyncTimedResidentICP,"_check_configuration"),
                    patch.object(adapter.Path,"read_bytes",return_value=b"original"),
                    patch.object(adapter.hashlib,"sha256",return_value=SimpleNamespace(hexdigest=lambda:digest)),
                    patch.object(adapter,"generator_state",return_value=state),self.assertRaises(RuntimeError)):
                obj._check_configuration(check_source=True)
            self.assertEqual(obj.source_guard_statistics["source_checks"],1)
    def test_actual_configuration_fault_is_not_recovered(self):
        obj=self.instance()
        with (patch.object(adapter.parent.CombinedSyncTimedResidentICP,"_check_configuration",side_effect=RuntimeError("kernel changed")),
                self.assertRaisesRegex(RuntimeError,"kernel changed")):
            obj._check_configuration(check_source=True)
    def test_iteration_guard_retains_original_and_does_not_read_source(self):
        obj=self.instance()
        with (patch.object(adapter.parent.CombinedSyncTimedResidentICP,"_check_configuration") as original,
                patch.object(adapter.Path,"read_bytes",side_effect=AssertionError("unexpected source read"))):
            obj._check_configuration()
        original.assert_called_once_with(check_source=False)
        self.assertEqual(obj.source_guard_statistics["iteration_config_checks"],1)
    def test_original_ast_control_calls_original_source_checker(self):
        obj=self.instance("original-ast")
        with patch.object(adapter.parent.CombinedSyncTimedResidentICP,"_check_configuration") as original:
            obj._check_configuration(check_source=True)
        original.assert_called_once_with(check_source=True)
    def test_policy_mutation_fails_before_parent_dispatch(self):
        obj=self.instance();obj.source_check_mode="original-ast"
        with patch.object(adapter.parent.CombinedSyncTimedResidentICP,"_check_configuration") as original,self.assertRaises(RuntimeError):
            obj._check_configuration(check_source=True)
        original.assert_not_called()
    def check_generator_change(self,attribute,value):
        obj=self.instance()
        function=adapter.audited.original_gpu_source
        before=getattr(function,attribute)
        try:
            setattr(function,attribute,value)
            with (patch.object(adapter.parent.CombinedSyncTimedResidentICP,"_check_configuration"),
                    patch.object(adapter.Path,"read_bytes",return_value=b"original"),
                    patch.object(adapter.hashlib,"sha256",return_value=SimpleNamespace(hexdigest=lambda:"0"*64)),
                    self.assertRaisesRegex(RuntimeError,"source contract changed")):
                obj._check_configuration(check_source=True)
        finally:setattr(function,attribute,before)
    def test_function_code_mutation_rejected_despite_same_owner(self):
        self.check_generator_change("__code__",adapter.audited.original_gpu_source.__code__.replace(co_firstlineno=999))
    def test_function_default_path_mutation_rejected_despite_same_owner(self):
        self.check_generator_change("__defaults__",(Path("different-source.py"),))
    def test_keyword_default_mutation_rejected_despite_same_owner(self):
        self.check_generator_change("__kwdefaults__",{"source":"changed"})
    def test_nested_default_snapshot_is_immutable(self):
        value={"nested":["original"]};snapshot=adapter.frozen_generator_value(value)
        value["nested"][0]="changed"
        self.assertNotEqual(adapter.frozen_generator_value(value),snapshot)


class IterationContracts(unittest.TestCase):
    def run_iteration(self,**options):
        obj,args,events=fake_iteration(**options)
        obj.source_guard_statistics=source_statistics()
        return obj,args,events
    def test_entire_original_iteration_and_bridge_ast_recover(self):
        contract=adapter.ITERATION_CONTRACT
        self.assertTrue(contract["original_numerical_and_control_statements_unchanged"])
        self.assertTrue(contract["only_preenqueue_clock_insertions"])
        self.assertFalse(contract["new_kernel"] or contract["graph_capture"])
        _,bridge=driver.owned_bridge_constructor(lambda:None)
        self.assertTrue(bridge["constructor_import_only"] and bridge["original_entire_bridge_body_recoverable"])
    def test_common_one_copy_and_new_nested_bookkeeping(self):
        obj,args,events=self.run_iteration()
        self.assertEqual(adapter._ITERATION(obj,*args),"original-normal-terms")
        self.assertEqual(events,["raw","classify","normal-guard","collapse-guard","copy","normalize"])
        self.assertEqual(obj.source_guard_statistics["preenqueue_calls"],1)
        self.assertGreaterEqual(obj.source_guard_statistics["preenqueue_bookkeeping_s"],0)
    def test_flags_still_resolve_before_original_equations(self):
        obj,args,events=self.run_iteration(flagged=True)
        with patch.object(adapter.parent.DeviceGridResidentICP,"_equations",side_effect=lambda *args:(events.append("gold") or "gold")):
            self.assertEqual(adapter._ITERATION(obj,*args),"gold")
        self.assertEqual(events[-2:],["cpu-resolve","gold"])
    def test_malformed_stops_before_cpu_or_solve(self):
        obj,args,events=self.run_iteration(malformed=True)
        with self.assertRaises(RuntimeError):adapter._ITERATION(obj,*args)
        self.assertNotIn("cpu-resolve",events);self.assertNotIn("normalize",events)


class DriverContracts(unittest.TestCase):
    def test_allocation_rejected_before_numerical_imports(self):
        with self.assertRaisesRegex(ValueError,"allocated"):driver.run(SimpleNamespace(run_allocated=False))
    def test_partial_cleanup_attempts_every_owner(self):
        calls=[]
        class Owner:
            def __setattr__(self,name,value):
                calls.append(name)
                if name=="bad":raise RuntimeError("restore failed")
                object.__setattr__(self,name,value)
        obj=Owner();errors=driver.restore_hooks([(obj,"ok",object()),(obj,"bad",object())])
        self.assertEqual(calls,["bad","ok"])
        self.assertEqual(len(errors),1)
    def test_constructor_uses_captured_original_after_scoped_hook(self):
        captured=driver.parent.bridge_class
        with patch.object(driver.parent,"bridge_class",lambda *args:None):
            _,contract=driver.owned_bridge_constructor(lambda:None,captured)
        self.assertTrue(contract["original_entire_bridge_body_recoverable"])
    def invoke_failure(self,parent_run):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"report.json"
            args=SimpleNamespace(run_allocated=True,output=path,source_check_mode="whole-file-sha")
            saved_bridge=driver.parent.bridge_class;saved_kind=driver.parent.KIND
            from scripts import process_metrics
            saved_worker=process_metrics.finish_cuda_worker
            with patch.object(driver.parent,"run",side_effect=parent_run),patch.object(driver,"file_hash",return_value="0"*64):
                with self.assertRaises(RuntimeError):driver.run(args)
            self.assertIs(driver.parent.bridge_class,saved_bridge)
            self.assertEqual(driver.parent.KIND,saved_kind)
            self.assertIs(process_metrics.finish_cuda_worker,saved_worker)
            return json.loads(path.read_text(encoding="utf-8"))
    def test_early_current_failure_is_saved_and_all_hooks_restored(self):
        value=self.invoke_failure(RuntimeError("primary early fault"))
        self.assertEqual(value["status"],"failed")
        self.assertEqual(value["timing_driver_current_failure"]["message"],"primary early fault")
        self.assertEqual(value["kind"],driver.KIND)
        self.assertTrue(value["source_guard_scope"]["current_failure"])
    def test_passed_partial_report_cannot_mask_late_failure(self):
        def late(args):
            args.output.write_text('{"status":"passed"}',encoding="utf-8")
            raise RuntimeError("late actual fault")
        value=self.invoke_failure(late)
        self.assertEqual(value["status"],"failed")
        self.assertEqual(value["timing_driver_current_failure"]["message"],"late actual fault")
    def test_missing_worker_completion_never_publishes_success(self):
        value=self.invoke_failure(lambda args:args.output.write_text('{"status":"passed"}',encoding="utf-8"))
        self.assertEqual(value["status"],"failed")
        self.assertIn("deferred",value["timing_driver_current_failure"]["message"])


if __name__=="__main__":unittest.main()
