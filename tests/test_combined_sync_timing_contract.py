"""Stdlib-only authority, original-math seam and hard-failure contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import combined_sync_timing_adapter as adapter
from scripts.research import validate_combined_sync_timing as guard
from scripts.research import benchmark_combined_sync_timing as driver
from scripts.research.archive.validate_uniform_grid_proof import GridProofError,canonical_hash


def authorities():
    cfg={"device":"CUDA:0","max_points":1000000,"max_scratch_bytes":256*1024**2,
        "max_query_bytes":136*1024**2,"gpu_timing":False}
    cache={"max_clouds":64,"retained_gpu_bytes":256*1024**2}
    binding={key:"same" for key in guard.SHARED_BINDING_KEYS}
    binding.update(resident_configuration=cfg,cache_policy=cache)
    text=json.dumps(binding,sort_keys=True,separators=(",",":"))
    a=guard.combined.CombinedSyncProofAuthority("cs","0"*64,"cb","0"*64,text,"f"*64,frozenset({"t"}),(("old","0"*64),))
    b=guard.device.DeviceFlatGridProofAuthority("a"*64,"f"*64,"0"*64,"0"*64,frozenset({"t"}),(("old","0"*64),),text)
    return a,b,binding,{"fixture":"fresh"},cfg,cache


class AuthorityContracts(unittest.TestCase):
    def setUp(self):guard._ACTIVE.clear()
    def activate(self,transform=None):
        a,b,binding,fixture,cfg,cache=authorities()
        second=copy.deepcopy(binding)
        if transform:transform(second)
        with patch.object(guard.combined,"validate_combined_sync_proof",return_value=a),patch.object(
                guard.device,"validate_grid_proof",return_value=b),patch.object(guard,"file_hash",return_value="0"*64):
            value=guard.validate_timing_authority("cs","cb","ds","db",binding,second,fixture,
                {name:"0"*64 for name in guard.ARTIFACTS})
            guard.validate_activation(value,b,cfg,cache)
        return value,b,cfg,cache
    def test_both_fresh_independent_proofs_required(self):
        token,device,cfg,cache=self.activate()
        self.assertIs(type(token),guard.CombinedSyncTimingAuthority)
        self.assertIsNot(type(token),guard.combined.CombinedSyncProofAuthority)
        self.assertEqual(len(token.proof_report_files),4)
        self.assertEqual(token.fixture_binding_sha256,canonical_hash({"fixture":"fresh"}))
    def test_manufactured_equal_token_cannot_activate(self):
        token,device,cfg,cache=self.activate()
        manufactured=copy.copy(token)
        self.assertEqual(token,manufactured)
        with self.assertRaises(GridProofError):guard.validate_activation(manufactured,device,cfg,cache)
    def test_device_configuration_drift_rejected(self):
        token,device,cfg,cache=self.activate();cfg["gpu_timing"]=True
        with self.assertRaises(GridProofError):guard.validate_activation(token,device,cfg,cache)
    def test_runtime_math_drift_rejected(self):
        with self.assertRaises(GridProofError):self.activate(lambda value:value.update(resident_math="different"))
    def test_missing_timing_dependency_rejected(self):
        a,b,binding,fixture,cfg,cache=authorities()
        with patch.object(guard.combined,"validate_combined_sync_proof",return_value=a),patch.object(
                guard.device,"validate_grid_proof",return_value=b),patch.object(guard,"file_hash",return_value="0"*64),self.assertRaises(GridProofError):
            guard.validate_timing_authority("cs","cb","ds","db",binding,binding,fixture,{guard.ARTIFACTS[0]:"0"*64})
    def test_proof_bytes_change_before_construction_rejected(self):
        token,device,cfg,cache=self.activate()
        with patch.object(guard,"file_hash",return_value="1"*64),self.assertRaises(GridProofError):
            guard.validate_activation(token,device,cfg,cache)
    def test_inherited_source_change_rejected(self):
        token,device,cfg,cache=self.activate()
        def hashed(path):return "1"*64 if Path(path).name=="old" else "0"*64
        with patch.object(guard,"file_hash",side_effect=hashed),self.assertRaises(GridProofError):
            guard.validate_activation(token,device,cfg,cache)
    def test_runtime_json_is_immutable(self):
        token,_,_,_=self.activate();value=token.runtime_binding
        value["resident_configuration"]["gpu_timing"]=True
        self.assertFalse(token.runtime_binding["resident_configuration"]["gpu_timing"])


class FakeArray:
    def __init__(self,n=1,dtype="f8",shape=None):
        self.dtype=dtype;self.shape=shape or (n,3);self.flags=SimpleNamespace(c_contiguous=True)
        self.device=SimpleNamespace(id=0);self.n=n
    def __len__(self):return self.n


class FakeCombined:
    def __init__(self,counters):self.counters=counters;self.nbytes=320
    def __setitem__(self,key,value):pass
    def __getitem__(self,key):return self.counters if key.stop==10 else self
    def view(self,dtype):return self
    def copy(self):return self


def fake_iteration(flagged=False,malformed=False):
    counters=[1,0,0,1,0,1,1,0,0,0] if flagged else [0,1,0,0,0,0,1,0,0,0]
    if malformed:counters[7]=1
    combined=FakeCombined(counters);events=[]
    cp=SimpleNamespace(ndarray=FakeArray,float64="f8",int32="i4",asnumpy=lambda value:(events.append("copy") or value))
    np=SimpleNamespace(float64=float,uint32=int,int32=int)
    nearest=FakeArray(dtype="i4",shape=(1,))
    def nearest_device(*args):events.append("cpu-resolve");return nearest,None
    retrieval=SimpleNamespace(_raw_device=lambda *args:(events.append("raw") or object()),
        classify=lambda *args:events.append("classify"),miss_policy="direct-miss-research-v1",nearest_device=nearest_device)
    stats={key:0 for key in ("primary_enqueue_s","primary_copy_wait_s","iteration_calls","primary_counter_term_syncs",
        "primary_download_bytes","malformed_results","provisional_queries","provisional_candidate_visits","flagged_calls",
        "flagged_queries","discarded_placeholder_calls","common_calls","common_queries","primary_normalize_s")}
    self=SimpleNamespace(_check_configuration=lambda:None,cp=cp,np=np,combined_statistics=stats,retrieval=retrieval,
        device_id=0,max_points=1000000,_buffers=lambda count:{"combined":combined,"ids":nearest,"squared":object(),
            "reasons":object(),"flagged":object()},combined_partial_kernel=lambda *args:events.append("normal-guard"),
        combined_collapse_kernel=lambda *args:events.append("collapse-guard"),
        _normalize_equations=lambda values,count:(events.append("normalize") or "original-normal-terms"))
    args=(FakeArray(),{"target":SimpleNamespace(points=[0]),"data":FakeArray()},object(),object(),.03,[0],object())
    return self,args,events


class IterationContracts(unittest.TestCase):
    def test_ast_reuses_original_numerical_seams(self):
        c=adapter.ITERATION_CONTRACT
        self.assertTrue(c["original_provisional_prefix_unchanged"] and c["original_common_normalization_unchanged"])
        self.assertFalse(c["new_math"] or c["graph_capture"])
    def test_common_path_only_one_copy_and_no_full_shadow(self):
        selfobj,args,events=fake_iteration()
        value=adapter._TIMED_ITERATION(selfobj,*args)
        self.assertEqual(value,"original-normal-terms")
        self.assertEqual(events,["raw","classify","normal-guard","collapse-guard","copy","normalize"])
        self.assertEqual(selfobj.combined_statistics["primary_download_bytes"],320)
        self.assertEqual(selfobj.combined_statistics["common_queries"],1)
    def test_flagged_original_cpu_before_original_equations(self):
        selfobj,args,events=fake_iteration(flagged=True)
        with patch.object(adapter.DeviceGridResidentICP,"_equations",side_effect=lambda *args:(events.append("original-equations") or "gold")):
            self.assertEqual(adapter._TIMED_ITERATION(selfobj,*args),"gold")
        self.assertEqual(events[-2:],["cpu-resolve","original-equations"])
        self.assertNotIn("normalize",events)
        self.assertEqual(selfobj.combined_statistics["discarded_placeholder_calls"],1)
    def test_malformed_stops_before_resolution_or_solve(self):
        selfobj,args,events=fake_iteration(malformed=True)
        with self.assertRaises(RuntimeError):adapter._TIMED_ITERATION(selfobj,*args)
        self.assertNotIn("cpu-resolve",events);self.assertNotIn("normalize",events)
        self.assertEqual(selfobj.combined_statistics["malformed_results"],1)
    def test_full_call_fallback_is_baseexception(self):
        with self.assertRaises(adapter.CombinedSyncTimingFailure):adapter.CombinedSyncTimedResidentICP._forbid_full_fallback()
        self.assertFalse(issubclass(adapter.CombinedSyncTimingFailure,Exception))


class DriverContracts(unittest.TestCase):
    def test_allocation_rejected_before_native_import(self):
        with self.assertRaisesRegex(ValueError,"allocated"):
            driver.run(SimpleNamespace(run_allocated=False))
    def test_three_round_order_rotates_each_path(self):
        # Fixed source protocol, not an arbitrary caller-supplied ordering.
        import inspect
        source=inspect.getsource(driver.run)
        self.assertIn('["native_cpu","device_resident","combined_sync"]',source)
        self.assertIn('["combined_sync","native_cpu","device_resident"]',source)
        self.assertIn('["device_resident","combined_sync","native_cpu"]',source)
    def test_contemporary_control_quality_failure_preserved(self):
        pair={"position":0,"pair":[8,12],"proposal_results":[{"proposal_index":0}],
            "verdict":{"accepted":True,"ambiguous":False,"verified_proposals":1}}
        a={"mode":"native_cpu","pairs":[copy.deepcopy(pair)]}
        b={"mode":"combined_sync","pairs":[copy.deepcopy(pair)]}
        reference={"results":[{"configuration":"1x20","rows":[{"position":0}]}]}
        baseline=SimpleNamespace(compare_rows=lambda *args:{"same_decisions_and_support":True})
        values={("native_cpu",0,0):"native",("combined_sync",0,0):"GPU"}
        with self.assertRaisesRegex(RuntimeError,"evidence"):
            driver.compare_round([b,a],values,reference,baseline,lambda *args:{"passed":False})
        self.assertFalse(b["pairs"][0]["proposal_results"][0]["quality"]["passed"])
    def test_driver_failed_report_never_keeps_passed_status(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"report.json";path.write_text('{"status":"passed"}',encoding="utf-8")
            error=RuntimeError("primary-current-run")
            driver.preserve_failure(path,error)
            value=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["status"],"failed")
            self.assertEqual(value["timing_driver_current_failure"]["message"],str(error))
    def test_malformed_partial_bytes_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"report.json";path.write_bytes(b"incomplete{")
            driver.preserve_failure(path,RuntimeError("primary"))
            value=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(Path(value["partial_report_preserved_at"]).read_bytes(),b"incomplete{")


if __name__=="__main__":unittest.main()
