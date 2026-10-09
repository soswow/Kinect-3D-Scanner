"""Pure stdlib ownership/authorization contracts for the minimal candidate."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research import gpu_icp_finish_candidate as candidate
from scripts.research import gpu_icp_finish_scope as owners


def full_binding():
    array={"dtype":"<f8","shape":[3,3],"nbytes":72,"sha256":"1"*64}
    cloud={name:copy.deepcopy(array) for name in ("points","normals","colors")}
    return {"source":copy.deepcopy(cloud),"target":copy.deepcopy(cloud),
        "seed":{"dtype":"<f8","shape":[4,4],"nbytes":128,"sha256":"2"*64}}


class Context:
    def __enter__(self):return self
    def __exit__(self,*args):return False


class Stream(Context):
    def __init__(self,log):self.log=log;self.error=None
    def synchronize(self):
        self.log.append("sync")
        if self.error is not None:raise self.error


class FakeGuard:
    def __init__(self,*args):self.slots=[]
    def add(self,module,name,original,installed):self.slots.append((module,name,installed))
    def check(self,**kwargs):
        for module,name,installed in self.slots:candidate.require(getattr(module,name) is installed,"slot replaced")


class Template:
    def __init__(self,retrieval,**kwargs):
        self.log=retrieval.log;self.device=0;self.cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda d:Context()))
        self.stream=Stream(self.log);self.closed=False;self.started=False
        self.configuration=tuple(sorted(kwargs.items()))
        self.audit_nearest=kwargs["audit_nearest"];self.authorizer=kwargs["timing_authorizer"]
        self.kernels={"math":object()};self.provenance={"method":"original"};self.statistics={}
    def close(self):self.stream.synchronize();self.closed=True;self.log.append("template-close")


class Lane:
    def __init__(self,template,authorizer,observe):
        self.log=template.log;self.template=template;self.authorizer=authorizer
        self.configuration=template.configuration;self.kernels=template.kernels
        self.provenance=template.provenance;self.closed=False;self.statistics={};self.error=None
    def match(self,source,target,seed,**kwargs):
        self.log.append("gpu")
        self.input_binding=candidate.normalized_consumed(full_binding(),dict(candidate.CONFIGURATION))
        if self.template.audit_nearest:
            self.input_binding["configuration"].update(audit_nearest=True,audit_misses=True)
        else:self.authorizer(self.input_binding,self.provenance)
        return SimpleNamespace(value="gpu")
    def close(self):
        self.log.append("lane-close")
        if self.error is not None:raise self.error
        self.closed=True
    def report(self):return {"closed":self.closed,"failure":None,"statistics":{}}


class Cache:
    def __init__(self,retrieval,clouds):
        self.log=retrieval.log;self.closed=False;self.failure=None;self.owned_bytes=100
        self.report_error=None;self.leases=[]
    def lease(self,source,target):
        result={"owner":self,"closed":False,"device_id":0};self.leases.append(result);return result
    def release_lease(self,lease):
        self.log.append("lease-release");lease["closed"]=True
    def close(self):self.log.append("cache-close");self.closed=True
    def report(self):
        if self.report_error is not None:raise self.report_error
        return {"closed":self.closed,"failure":None,"owned_numeric_bytes":self.owned_bytes}


class Fixture:
    def __init__(self,test,mode="shadow",protocol=None,qualification=None):
        self.test=test;self.log=[];self.result=SimpleNamespace(value="original")
        self.registration=SimpleNamespace(_device=SimpleNamespace(get=lambda:None))
        def dispatch(*args):self.log.append("dispatch-cpu");return None
        self.registration.match=dispatch
        def cpu(source,target,seed):
            self.log.append("cpu")
            test.assertIsNone(self.registration.match(source,target,seed))
            return self.result
        def bridge(source,target,seed):
            for value in ("forward","reverse","witness"):
                self.log.append(value);self.registration.match(source,target,seed)
            return self.result
        self.fragments=SimpleNamespace(REG=object(),_match=cpu,_verify_bridge=bridge)
        self.refinement=SimpleNamespace(REG=self.fragments.REG,_match=cpu)
        self.retrieval=SimpleNamespace(log=self.log,close=lambda:self.log.append("retrieval-close"))
        def new_lane(template,*,timing_authorizer=None,audit_observer=None):
            lane=Lane(template,timing_authorizer,audit_observer);self.last_lane=lane;return lane
        clone=SimpleNamespace(SourceOwnerGuard=lambda m:FakeGuard(),
            snapshot=lambda t:{"kernels":dict(t.kernels)},verify_snapshot=lambda *a:None,fresh_clone=new_lane)
        fakeowners=SimpleNamespace(WrapperOwnerGuard=FakeGuard,HelperOwners=owners.HelperOwners,
            function_state=owners.function_state,check_function=owners.check_function)
        self.adapters=SimpleNamespace(method=SimpleNamespace(DeviceLoopICP=Template),clone=clone,owners=fakeowners,
            capture=lambda *args:full_binding(),result=lambda *args:{"canonical_ids":"same"},
            shadow=lambda *args:{"passed":True},semantic=lambda r:{"value":r.value},
            clouds=lambda *args:[],cache_factory=Cache,packet=lambda p:{},
            terminal=lambda *args:{"complete":True},final_poses=lambda e:{"views":[1]})
        self.scope=candidate.GpuICPFinishCandidate(self.registration,self.fragments,self.refinement,mode=mode,
            retrieval_factory=lambda:self.retrieval,adapters=self.adapters,qualification=qualification,
            protocol=protocol,binding={"current":True})
        self.source=SimpleNamespace(index=1);self.target=SimpleNamespace(index=2)
    def run(self):
        with self.scope:
            result=self.fragments._verify_bridge(self.source,self.target,object())
            self.test.assertIs(result,self.result)
            self.scope.build(SimpleNamespace(unprocessed_count=0),lambda e:(True,"mesh"))
            self.scope.finish()
        return self.scope.report()


class AuthorizationTests(unittest.TestCase):
    def authorize(self,mode="shadow",**kwargs):
        full=full_binding();source={"math":"current"};owner=SimpleNamespace(closed=False,failure=None)
        lease={"owner":owner,"closed":False,"device_id":0}
        consumed=candidate.normalized_consumed(full,dict(candidate.CONFIGURATION))
        certificate={"input":consumed,"completed":True,"input_bytes_unchanged":True,"result":{}}
        args={"certificate":certificate} if mode=="shadow" else kwargs
        auth=candidate.CallAuthorization(mode,full,source,dict(candidate.CONFIGURATION),lease,owner,**args)
        return auth,consumed,source,lease
    def test_shadow_certificate_exact_call_and_single_use(self):
        auth,consumed,source,_=self.authorize()
        self.assertTrue(auth(consumed,source)["cpu_first"])
        with self.assertRaises(candidate.CandidateFailure):auth(consumed,source)
    def test_actual_seed_or_source_drift_rejected_before_use(self):
        for field in ("seed","source"):
            auth,consumed,source,_=self.authorize();consumed[field]["sha256"]="f"*64
            with self.assertRaises(candidate.CandidateFailure):auth(consumed,source)
            self.assertFalse(auth.used)
        auth,consumed,source,_=self.authorize()
        with self.assertRaises(candidate.CandidateFailure):auth(consumed,{"math":"changed"})
    def test_foreign_closed_lease_and_authorization_owner_rejected(self):
        for change in ("owner","closed","device_id"):
            auth,consumed,source,lease=self.authorize()
            lease[change]={"owner":object(),"closed":True,"device_id":1}[change]
            with self.assertRaises(candidate.CandidateFailure):auth(consumed,source)
        auth,consumed,source,_=self.authorize();auth.state=auth.state._replace(mode="measure")
        with self.assertRaises(candidate.CandidateFailure):auth(consumed,source)
    def test_missing_or_incomplete_cpu_first_certificate_rejected(self):
        auth,consumed,source,_=self.authorize()
        auth.state=auth.state_owner=auth.state._replace(certificate=None)
        with self.assertRaises(candidate.CandidateFailure):auth(consumed,source)
        auth,consumed,source,_=self.authorize();cert=__import__("json").loads(auth.state.certificate);cert["completed"]=False
        auth.state=auth.state_owner=auth.state._replace(certificate=candidate.canonical(cert))
        with self.assertRaises(candidate.CandidateFailure):auth(consumed,source)
    def test_measure_invokes_own_method_validator_on_actual_input(self):
        calls=[];token=object()
        protocol=SimpleNamespace(validate_candidate_call=lambda t,**kw:calls.append((t,kw)))
        auth,consumed,source,_=self.authorize("measure",protocol=protocol,qualification=token)
        receipt=auth(consumed,source)
        self.assertFalse(receipt["cpu_first"]);self.assertEqual(len(calls),1)
        self.assertIs(calls[0][0],token);self.assertEqual(calls[0][1]["consumed"],consumed)
        auth,consumed,source,_=self.authorize("measure")
        with self.assertRaises(candidate.CandidateFailure):auth(consumed,source)
    def test_measure_validator_slot_and_in_place_code_mutation_rejected(self):
        protocol=SimpleNamespace(validate_candidate_call=lambda *a,**kw:True)
        state=owners.function_state(protocol.validate_candidate_call)
        for mutate in ("slot","code"):
            auth,consumed,source,_=self.authorize("measure",protocol=protocol,qualification=object(),validator_state=state)
            original=protocol.validate_candidate_call;code=original.__code__
            try:
                if mutate=="slot":protocol.validate_candidate_call=lambda *a,**kw:True
                else:original.__code__=(lambda *a,**kw:False).__code__
                with self.assertRaises(BaseException):auth(consumed,source)
                self.assertFalse(auth.used)
            finally:protocol.validate_candidate_call=original;original.__code__=code


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{"KINECT_CUDA_REGISTRATION":"cpu"});self.env.start()
        self.source=patch.object(candidate,"source_contract",return_value={"portable":True});self.source.start()
        # Cold source-own guards are exercised separately against the real module.
        self.helpers=patch.object(owners,"HelperOwners",return_value=FakeGuard());self.helpers.start()
    def tearDown(self):self.helpers.stop();self.source.stop();self.env.stop()
    def test_shadow_runs_cpu_first_each_call_and_returns_original_proposal_object(self):
        f=Fixture(self);report=f.run()
        self.assertEqual([v for v in f.log if v in ("cpu","gpu")],["cpu","gpu"]*3)
        self.assertTrue(report["complete"] and report["closed"] and report["restored"])
        self.assertEqual(len(report["calls"]),3);self.assertEqual(len(report["workspace"]["jobs"]),3)
        self.assertEqual(report["gates"],[]);self.assertTrue(all(r["authorization"]["cpu_first"] for r in report["calls"]))
        f.scope.close();self.assertEqual(f.log.count("cache-close"),1)
    def test_audit_original_math_clone_and_full_native_shadow(self):
        f=Fixture(self,"audit");report=f.run()
        self.assertEqual([v for v in f.log if v in ("cpu","gpu")],["gpu","cpu"]*3)
        self.assertTrue(all(r["native_shadow"]["passed"] for r in report["calls"]))
        self.assertEqual(report["workspace"]["original_constructor_fields"],27)
    def test_current_method_measure_qualification_is_required(self):
        with self.assertRaises(candidate.CandidateFailure):Fixture(self,"measure")
        calls=[];protocol=SimpleNamespace(validate_candidate_method=lambda *a,**k:calls.append("qualify"),
            validate_candidate_call=lambda *a,**k:calls.append("authorize"))
        f=Fixture(self,"measure",protocol=protocol,qualification=object());report=f.run()
        self.assertEqual(calls,["qualify"]+["authorize"]*3);self.assertNotIn("cpu",f.log)
        self.assertTrue(all(r["native_shadow"]=={"collected":False} for r in report["calls"]))
    def test_original_cpu_dispatch_outside_bridge_is_unchanged(self):
        f=Fixture(self)
        with f.scope:
            self.assertIsNone(f.registration.match(f.source,f.target,object()))
            self.assertEqual(f.scope.stats["non_bridge_dispatch_calls"],1)
            f.scope.build(SimpleNamespace(unprocessed_count=0),lambda e:(True,"mesh"));f.scope.finish()
        self.assertEqual(f.scope.calls,[])
    def test_all_original_competitors_preserve_order_and_pair_lifetime(self):
        f=Fixture(self)
        with f.scope:
            for _ in range(2):self.assertIs(f.fragments._verify_bridge(f.source,f.target,object()),f.result)
            self.assertEqual(len(f.scope.pairs[0]["proposals"]),2)
            self.assertEqual([r["proposal_index"] for r in f.scope.calls],[0]*3+[1]*3)
            f.scope.build(SimpleNamespace(unprocessed_count=0),lambda e:(True,"mesh"));f.scope.finish()
        self.assertEqual(f.log.count("cache-close"),1)
    def test_failed_entry_restores_both_original_slots(self):
        f=Fixture(self);old=(f.registration.match,f.fragments._verify_bridge)
        with patch.object(candidate,"source_contract",side_effect=RuntimeError("cold fault")):
            with self.assertRaisesRegex(RuntimeError,"cold fault"):f.scope.__enter__()
        self.assertIs(f.registration.match,old[0]);self.assertIs(f.fragments._verify_bridge,old[1])
    def test_partial_constructor_owner_closes_and_reports_original_fault(self):
        scope=object.__new__(candidate.GpuICPFinishCandidate)
        with self.assertRaises(candidate.CandidateFailure) as caught:
            scope.__init__(None,None,None,mode="measure",retrieval_factory=None)
        with self.assertRaises(candidate.CandidateFailure) as cleaned:scope.close(caught.exception)
        self.assertIs(cleaned.exception,caught.exception)
        receipt=scope.report();self.assertEqual(receipt["calls"],[]);self.assertTrue(receipt["restored"])
        self.assertFalse(receipt["complete"]);self.assertIsNotNone(receipt["failure"])
    def test_failed_retrieval_constructor_retains_owned_partial_object(self):
        f=Fixture(self);partial=SimpleNamespace(close=lambda:f.log.append("partial-close"))
        failure=RuntimeError("retrieval init")
        def factory():failure.candidate_retrieval=partial;raise failure
        f.scope.factory=factory
        f.scope.owners=tuple(factory if value is f.scope.owners[4] else value for value in f.scope.owners)
        with self.assertRaises(candidate.CandidateFailure):
            with f.scope:f.fragments._verify_bridge(f.source,f.target,object())
        self.assertIs(f.scope.retrieval,partial);self.assertIn("partial-close",f.log)
    def test_completion_fault_retains_target_and_template_owners(self):
        f=Fixture(self);f.scope.__enter__();f.fragments._verify_bridge(f.source,f.target,object())
        cache=f.scope.shared;template=f.scope.template;template.stream.error=RuntimeError("sync fault")
        primary=candidate.CandidateFailure("original fault")
        with self.assertRaises(candidate.CandidateFailure) as caught:f.scope.close(primary)
        self.assertIs(caught.exception,primary);self.assertIs(f.scope.shared,cache);self.assertIs(f.scope.template,template)
        self.assertFalse(cache.closed);self.assertNotIn("cache-close",f.log)
        self.assertFalse(f.scope.closed);self.assertIsNotNone(f.scope.failure)
    def test_cache_report_fault_preserves_primary_and_partial_receipt(self):
        f=Fixture(self);f.scope.__enter__();f.fragments._verify_bridge(f.source,f.target,object())
        cache=f.scope.shared;cache.report_error=RuntimeError("report fault")
        primary=candidate.CandidateFailure("original fault")
        with self.assertRaises(candidate.CandidateFailure) as caught:f.scope.close(primary)
        self.assertIs(caught.exception,primary);self.assertTrue(cache.closed);self.assertIs(f.scope.shared,cache)
        self.assertTrue(any("report fault" in r for r in f.scope.cleanup_failures));self.assertFalse(f.scope.closed)
    def test_resource_and_configuration_replacement_refused(self):
        f=Fixture(self);f.scope.configuration=("changed",)
        with self.assertRaises(candidate.CandidateFailure):f.scope._critical()
        f=Fixture(self);f.scope.retrieval=object()
        with self.assertRaises(candidate.CandidateFailure):f.scope._critical()
        for name,value in (("mode","measure"),("adapters",object()),("original_match",lambda *a:None)):
            f=Fixture(self);setattr(f.scope,name,value)
            with self.assertRaises(candidate.CandidateFailure):f.scope._critical()
    def test_loaded_critical_code_or_class_slot_change_refused(self):
        f=Fixture(self)
        with patch.object(candidate.GpuICPFinishCandidate,"_cpu",lambda *a:None):
            with self.assertRaises(candidate.CandidateFailure):f.scope._critical()
    def test_ordinary_owner_fault_is_latched_even_if_body_catches_it(self):
        f=Fixture(self);f.scope.__enter__();original=f.scope.guard.check
        f.scope.guard.check=lambda **kw:(_ for _ in ()).throw(RuntimeError("transient owner fault"))
        with self.assertRaises(candidate.CandidateFailure) as caught:
            f.registration.match(f.source,f.target,object())
        f.scope.guard.check=original
        self.assertIsInstance(caught.exception.__cause__,RuntimeError)
        with self.assertRaises(candidate.CandidateFailure) as later:f.scope.close()
        self.assertIs(later.exception,caught.exception)
        self.assertFalse(f.scope.report()["complete"])


class LoadedSourceTests(unittest.TestCase):
    def test_real_own_cold_compiled_class_and_defaults_guard(self):
        guard=owners.HelperOwners(((candidate,("require","canonical","source_contract","normalized_consumed",
            "CallAuthorization","dependencies","GpuICPFinishCandidate")),))
        guard.check(source=True)
        fn=candidate.normalized_consumed;old=fn.__defaults__
        try:
            fn.__defaults__=("changed",)
            with self.assertRaises(owners.FinishGPUFailure):guard.check()
        finally:fn.__defaults__=old
    def test_import_is_stdlib_only_in_fresh_process(self):
        code="from scripts.research import gpu_icp_finish_candidate; import sys; assert not any(n in sys.modules for n in ('numpy','cupy','open3d','cv2'))"
        result=subprocess.run([sys.executable,"-S","-c",code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_method_scope_has_no_historical_core_hash(self):
        source=Path(candidate.__file__).read_text(encoding="utf-8")
        self.assertNotIn("07a948e8",source);self.assertNotIn("CURRENT =",source)
        self.assertNotIn("gpu_icp_finish_protocol",source)


if __name__=="__main__":unittest.main()
