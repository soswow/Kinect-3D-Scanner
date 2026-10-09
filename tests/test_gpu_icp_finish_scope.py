"""Stdlib ownership/fault/return-object contracts; no numerical packages."""
import contextvars
import copy
import importlib.util
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import gpu_icp_finish_scope as subject


CORE = '''
from contextvars import ContextVar
_device=ContextVar("test_original_device",default=None)
native_results=[]
def match(source,target,initial):
    return None
def _match(source,target,initial):
    result=registration.match(source,target,initial)
    if result is not None:return result
    value=native_factory(source,target,initial)
    native_results.append(value)
    return value
def _verify_bridge(source,target,initial,*args,**kwargs):
    return _pair(source.train,target.train,initial)
def _pair(source,target,initial):
    return _match(source,target,initial)
def global_optimization(*args,**kwargs):
    return graph_value
def get_information_matrix_from_point_clouds(*args,**kwargs):
    return information_value
'''

BUNDLE_CORE='''
from dataclasses import dataclass
@dataclass
class BundleProblem:
    poses:list
    landmarks:object
    cameras:object
    tracks:object
    pixels:object
    depths:object
class _BudgetExceeded(RuntimeError):pass
def build_tracks(*args,**kwargs):return [[(0,1),(1,2),(2,3)]]
def _verified_matches(*args,**kwargs):return []
def make_problem(*args,**kwargs):return problem_value
def _supported(problem,*args,**kwargs):return True
def solve_bundle(problem,*args,**kwargs):
    if solve_error is not None:raise solve_error
    if mutate_problem:problem.cameras.values[0]=99
    return solver_value
def validate_depth(*args,**kwargs):return True,{"validation_pairs":1}
def propose_bundle_poses(engine,progress_cb=None):
    try:
        build_tracks(None,None);_verified_matches(None,None,None)
        problem=make_problem(None,None,None)
        if _supported(problem,None):
            solve_bundle(problem,None);validate_depth(None,None,None,None)
        return proposal_value
    except (_BudgetExceeded,ImportError) as error:
        caught.append(error)
        return None,{"reason":str(error),"budget_exceeded":isinstance(error,_BudgetExceeded)}
'''

class Array:
    """Artificial descriptor carrier; never a native/numerical array."""
    def __init__(self,values,shape,kind="f"):
        self.values=values;self.shape=shape;self.dtype=SimpleNamespace(str="<"+kind+"8")
    def tolist(self):return copy.deepcopy(self.values)


def bundle_semantic(value):
    if isinstance(value,Array):
        count=1
        for size in value.shape:count*=size
        return {"array":{"dtype":value.dtype.str,"shape":list(value.shape),"nbytes":count*8,
            "sha256":hashlib.sha256(subject.canonical(value.values).encode()).hexdigest()}}
    if isinstance(value,(list,tuple)):return [bundle_semantic(v) for v in value]
    return semantic(value)


class Context:
    def __enter__(self):return self
    def __exit__(self,*args):return False


class Stream(Context):
    def __init__(self,log):self.log=log;self.failure=None
    def synchronize(self):
        self.log.append("sync")
        if self.failure:raise self.failure


class Loop:
    def __init__(self,workspace,observer):
        self.workspace=workspace;self.closed=False;self.failure=None
        self.provenance={"method":"original"};self.input_binding={"exact":"actual"}
        self.statistics={"queries":6,"updates":3};self.observer=observer
    def match(self,source,target,initial,**kwargs):
        self.workspace.log.append("gpu")
        if self.workspace.match_failure:raise self.workspace.match_failure
        return self.workspace.result
    def report(self):return {"closed":self.closed,"failure":None if self.failure is None else repr(self.failure),
        "statistics":dict(self.statistics),"provenance":self.provenance,"input_binding":self.input_binding}


class Workspace:
    def __init__(self,retrieval,**configuration):
        self.retrieval=retrieval;self.log=retrieval.log;self.closed=False;self.active=None
        self.configuration=configuration;self.jobs=[];self.result=SimpleNamespace(value="gpu")
        self.match_failure=self.lane_close_failure=self.close_failure=None
        self.template=SimpleNamespace(device=0,stream=Stream(self.log),cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda _:Context())))
    def new_lane(self,*,audit_observer=None):
        self.log.append("lane");self.active=Loop(self,audit_observer);self.jobs.append(self.active);return self.active
    def check_lane(self,lane):assert lane is self.active
    def close_lane(self,lane,*,primary=None):
        self.log.append("lane-close")
        if self.lane_close_failure:raise self.lane_close_failure
        lane.closed=True;self.active=None
    def close(self,*,primary=None):
        self.log.append("workspace-close")
        if self.close_failure:raise self.close_failure
        if self.active:self.close_lane(self.active,primary=primary)
        self.closed=True
    def report(self):return {"closed":self.closed,"jobs":len(self.jobs)}


class Cache:
    instances=[]
    def __init__(self,retrieval,clouds):
        self.log=retrieval.log;self.closed=False;self.owned_bytes=100;self.leases=[]
        self.close_failure=self.release_failure=None;type(self).instances.append(self);self.log.append("cache")
    def lease(self,source,target):
        value={"closed":False};self.leases.append(value);return value
    def release_lease(self,value):
        self.log.append("lease-release")
        if self.release_failure:raise self.release_failure
        value["closed"]=True;self.leases.remove(value)
    def close(self):
        self.log.append("cache-close")
        if self.close_failure:raise self.close_failure
        self.closed=True
    def report(self):return {"closed":self.closed,"leases":len(self.leases)}


def capture(a,b,s):return {"source":copy.deepcopy(a.points),"target":copy.deepcopy(b.points),"seed":copy.deepcopy(s)}
def semantic(value):return {"value":value.value} if hasattr(value,"value") else copy.deepcopy(value)
def pair_clouds(a,b):return [a.train,b.train]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.log=[];Cache.instances=[]
        path=Path(self.directory.name)/"finish_core.py";path.write_text(CORE,encoding="utf-8")
        self.name="_finish_core_contract"
        spec=importlib.util.spec_from_file_location(self.name,path);self.module=importlib.util.module_from_spec(spec)
        sys.modules[self.name]=self.module;self.addCleanup(sys.modules.pop,self.name,None);spec.loader.exec_module(self.module)
        self.module.registration=self.module
        self.module.native_factory=lambda *args:SimpleNamespace(value="gpu")
        self.module.REG=SimpleNamespace(global_optimization=self.module.global_optimization,
            get_information_matrix_from_point_clouds=self.module.get_information_matrix_from_point_clouds)
        self.module.graph_value=object();self.module.information_value=object()
        self.original_match=self.module.match;self.original_bridge=self.module._verify_bridge
        self.original_pair=self.module._pair
        self.env=patch.dict(os.environ,{"KINECT_CUDA_REGISTRATION":"cpu"});self.env.start();self.addCleanup(self.env.stop)
        self.a=SimpleNamespace(index=0,train=SimpleNamespace(points=[[0.,0.,0.]]),keys=[])
        self.b=SimpleNamespace(index=1,train=SimpleNamespace(points=[[1.,0.,0.]]),keys=[])
        self.c=SimpleNamespace(index=2,train=SimpleNamespace(points=[[2.,0.,0.]]),keys=[])
        self.adapters=SimpleNamespace(capture=capture,result=lambda r,a,b:semantic(r),
            shadow=lambda *args:{"passed":True},semantic=semantic,
            terminal=lambda *args:{"terminal":"actual"},packet=lambda q:{"packet":True},
            pair_clouds=pair_clouds,cache_factory=Cache,
            graph_snapshot=lambda g:copy.deepcopy(g),graph_context=lambda:{"owner":"original"},
            final_poses=lambda engine:{"indices":[1]})
        self.protocol=SimpleNamespace(next_workspace_job=lambda permit,payload:len(self.scope.calls)-1,
            validate_finish_terminal=lambda *args:self.log.append("terminal-after-close"),
            validate_finish_event=lambda *args:None,
            validate_complete_finish=lambda permit,report:self.log.append("complete-after-close"))
        self.engine=SimpleNamespace(unprocessed_count=0)

    def make_scope(self,mode="audit",**kwargs):
        retrieval=SimpleNamespace(log=self.log,close=lambda:self.log.append("retrieval-close"))
        self.scope=subject.GpuICPFinishScope(self.module,self.module,self.module,mode=mode,
            retrieval_factory=lambda:retrieval,workspace_factory=Workspace,protocol=self.protocol,
            permit=object() if mode=="timing" else None,numpy=SimpleNamespace(eye=lambda _:"identity"),
            adapters=self.adapters,**kwargs)
        return self.scope

    def make_bundle(self):
        path=Path(self.directory.name)/"bundle_core.py";path.write_text(BUNDLE_CORE,encoding="utf-8")
        name="_finish_bundle_contract"
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;self.addCleanup(sys.modules.pop,name,None);spec.loader.exec_module(module)
        pose=Array([[float(i==j) for j in range(4)] for i in range(4)],(4,4))
        module.problem_value=module.BundleProblem([pose,pose,pose],Array([[0.,0.,1.]],(1,3)),
            Array([0,1,2],(3,),"i"),Array([0,0,0],(3,),"i"),Array([[1.,2.]]*3,(3,2)),Array([1.]*3,(3,)))
        module.solver_value=([pose]*3,Array([[0.,0.,1.]],(1,3)),{"solver_converged":True})
        module.proposal_value=(None,{"applied":False,"time_budget_s":45.,"elapsed_ms":1.})
        module.solve_error=None;module.mutate_problem=False;module.caught=[]
        self.adapters.semantic=bundle_semantic
        return module

    @staticmethod
    def build(engine):return (True,{"original":True})
    def finish(self,scope):scope.build(self.engine,self.build);scope.finish()

    def test_native_and_gpu_return_actual_objects_and_preserve_aliases(self):
        for mode in ("native","audit"):
            with self.make_scope(mode) as scope:
                original_alias=self.module._match
                value=self.module._verify_bridge(self.a,self.b,[1])
                self.assertIs(self.module._match,original_alias)
                self.assertIs(value,self.module.native_results[-1] if mode=="native" else scope.workspace.result)
                self.finish(scope)
            self.assertIs(self.module.match,self.original_match);self.assertIs(self.module._verify_bridge,self.original_bridge)
            self.assertTrue(scope.report()["complete"])

    def test_non_bridge_dispatch_is_original_cpu_and_no_gpu_setup(self):
        with self.make_scope() as scope:
            value=self.module._match(self.a.train,self.b.train,[1])
            self.assertIs(value,self.module.native_results[-1]);self.assertIsNone(scope.workspace)
            self.finish(scope)
        self.assertEqual(scope.native_calls,1)

    def test_same_pair_all_proposals_reuse_cache_and_new_pair_synchronizes(self):
        with self.make_scope() as scope:
            self.module._verify_bridge(self.a,self.b,[1]);self.module._verify_bridge(self.a,self.b,[2])
            self.assertEqual(len(Cache.instances),1);self.assertEqual(len(scope.workspace.jobs),2)
            self.module._verify_bridge(self.a,self.c,[3]);self.assertEqual(len(Cache.instances),2)
            self.assertTrue(Cache.instances[0].closed);self.assertLess(self.log.index("sync"),self.log.index("cache-close"))
            self.finish(scope)
        self.assertEqual([len(p["proposals"]) for p in scope.pairs],[2,1])

    def test_audit_original_cpu_shadow_runs_each_gpu_call(self):
        with self.make_scope() as scope:
            for seed in ([1],[2]):self.module._verify_bridge(self.a,self.b,seed)
            self.assertEqual(len(self.module.native_results),2);self.assertEqual(len(scope.calls),2)
            self.assertTrue(all(c["native_shadow"]["passed"] for c in scope.calls));self.finish(scope)

    def test_native_zero_bridge_full_build_is_valid(self):
        with self.make_scope("native") as scope:self.finish(scope)
        self.assertEqual(scope.calls,[]);self.assertTrue(scope.closed)

    def test_timing_terminal_and_complete_follow_selected_cleanup(self):
        with self.make_scope("timing") as scope:
            self.module._verify_bridge(self.a,self.b,[1]);self.finish(scope)
        self.assertLess(self.log.index("lease-release"),self.log.index("terminal-after-close"))
        self.assertLess(self.log.index("cache-close"),self.log.index("complete-after-close"))
        self.assertEqual(self.module.native_results,[])

    def test_failed_cpu_shadow_latches_even_if_body_swallows(self):
        self.adapters.shadow=lambda *args:{"passed":False}
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope() as scope:
                try:self.module._verify_bridge(self.a,self.b,[1])
                except subject.FinishGPUFailure:pass
        self.assertIs(self.module.match,self.original_match);self.assertTrue(scope.workspace.closed)

    def test_trace_failure_is_hard_and_hooks_restore(self):
        def trace(row):raise ValueError("recording fault")
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope(trace=trace):self.module._verify_bridge(self.a,self.b,[1])
        self.assertIs(self.module._pair,self.original_pair)

    def test_selected_failure_retains_lease_until_workspace_close(self):
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope() as scope:
                self.module._verify_bridge(self.a,self.b,[1])
                scope.workspace.lane_close_failure=RuntimeError("completion failed")
                scope.workspace.close_failure=RuntimeError("completion still failed")
                self.module._verify_bridge(self.a,self.b,[2])
        self.assertIsNotNone(scope.shared);self.assertFalse(scope.shared.closed)
        self.assertFalse(scope.workspace.closed);self.assertTrue(scope.shared.leases)

    def test_primary_failure_preserved_with_secondary_close(self):
        with self.assertRaises(subject.FinishGPUFailure) as raised:
            with self.make_scope() as scope:
                self.module._verify_bridge(self.a,self.b,[1])
                scope.workspace.close_failure=RuntimeError("secondary")
                scope.fail(ValueError("primary"))
        self.assertIn("primary",str(raised.exception));self.assertTrue(scope.cleanup_failures)

    def test_live_pending_build_rejected(self):
        self.engine.unprocessed_count=1
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native") as scope:self.finish(scope)

    def test_build_failure_and_missing_suffix_refuse_success(self):
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native") as scope:scope.build(self.engine,lambda engine:(False,{}))
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native"):pass

    def test_original_successful_build_tuple_identity(self):
        expected=(True,{"same":object()})
        with self.make_scope("native") as scope:
            self.assertIs(scope.build(self.engine,lambda engine:expected),expected);scope.finish()

    def test_pair_owner_replacement_rejected(self):
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope() as scope:
                self.module._verify_bridge(self.a,self.b,[1])
                self.module._verify_bridge(copy.copy(self.a),self.b,[2])

    def test_pair_value_mutation_rejected_before_second_gpu(self):
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope() as scope:
                self.module._verify_bridge(self.a,self.b,[1]);self.a.train.points[0][0]=3.
                self.module._verify_bridge(self.a,self.b,[2])
        self.assertEqual(self.log.count("gpu"),1)

    def test_dispatch_owner_mutation_rejected(self):
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native") as scope:
                self.module.match=lambda *args:None;scope.healthy()
        self.assertTrue(scope.cleanup_failures)

    def test_alias_or_device_selection_mutation_rejected(self):
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native") as scope:
                token=self.module._device.set("CUDA:0")
                try:scope.healthy()
                finally:self.module._device.reset(token)

    def test_original_loaded_code_mutation_rejected(self):
        original=self.module._match;code=original.__code__
        try:
            with self.assertRaises(subject.FinishGPUFailure):
                with self.make_scope("native") as scope:
                    original.__code__=(lambda *args:None).__code__;scope.healthy()
        finally:original.__code__=code

    def test_graph_optimizer_returns_original_object_and_records_before_after(self):
        self.adapters.semantic=lambda value:"original" if value is self.module.graph_value else semantic(value)
        graph={"nodes":[1]};option=SimpleNamespace(max_correspondence_distance=.03,edge_prune_threshold=.25,reference_node=0)
        with self.make_scope("native") as scope:
            self.assertIs(self.module.REG.global_optimization(graph,object(),object(),option),self.module.graph_value)
            self.assertEqual(scope.graphs[0]["before"],scope.graphs[0]["after"]);self.finish(scope)

    def test_no_numerical_imports_required(self):
        for name in ("numpy","cupy","open3d"):self.assertNotIn(name,sys.modules)

    def test_gpu_fault_preserved_if_stage_cleanup_replaces_it(self):
        with self.assertRaises(subject.FinishGPUFailure) as raised:
            with self.make_scope() as scope:
                self.module._verify_bridge(self.a,self.b,[1])
                scope.workspace.match_failure=ValueError("gpu primary")
                try:self.module._verify_bridge(self.a,self.b,[2])
                except subject.FinishGPUFailure:raise RuntimeError("stage cleanup replaces it")
        self.assertIn("gpu primary",str(raised.exception))

    def test_partial_workspace_constructor_retained_and_primary_diagnostic(self):
        class BrokenWorkspace(Workspace):
            def __init__(self,*args,**kwargs):raise ValueError("constructor primary")
        scope=self.make_scope()
        scope.workspace_factory=BrokenWorkspace
        scope.owners=scope.owners[:7]+(BrokenWorkspace,)+scope.owners[8:]
        with self.assertRaises(subject.FinishGPUFailure) as raised:
            with scope:self.module._verify_bridge(self.a,self.b,[1])
        self.assertIn("constructor primary",str(raised.exception));self.assertIsNotNone(scope.workspace)
        self.assertIn("partial_report_failure",scope.report()["workspace"])

    def test_loaded_scope_method_in_place_mutation_rejected(self):
        fn=subject.GpuICPFinishScope._cpu;code=fn.__code__
        try:
            with self.assertRaises(subject.FinishGPUFailure):
                with self.make_scope("native") as scope:
                    fn.__code__=(lambda *args:None).__code__;scope.healthy()
        finally:fn.__code__=code

    def test_workspace_owner_replacement_rejected(self):
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope() as scope:
                self.module._verify_bridge(self.a,self.b,[1])
                scope.workspace=copy.copy(scope.workspace);scope.healthy()

    def test_graph_options_change_rejected_before_original_optimizer(self):
        graph={"nodes":[1]};option=SimpleNamespace(max_correspondence_distance=.04,edge_prune_threshold=.25,reference_node=0)
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native") as scope:self.module.REG.global_optimization(graph,object(),object(),option)
        self.assertFalse(scope.graphs[0]["complete"])

    def test_adapter_in_place_code_mutation_rejected(self):
        fn=self.adapters.shadow;code=fn.__code__
        try:
            with self.assertRaises(subject.FinishGPUFailure):
                with self.make_scope() as scope:
                    self.module._verify_bridge(self.a,self.b,[1])
                    fn.__code__=(lambda *args:{"passed":True,"changed":True}).__code__;scope.healthy()
        finally:fn.__code__=code

    def test_late_owner_fault_cannot_be_swallowed_after_finish(self):
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native") as scope:
                self.finish(scope)
                self.module.match=lambda *args:None
                try:scope.healthy()
                except subject.FinishGPUFailure:pass
        self.assertFalse(scope.report()["complete"])
        self.assertIs(self.module.match,self.original_match)

    def test_bound_build_uses_only_its_actual_engine_and_return_object(self):
        class Engine:
            unprocessed_count=0
            def build(self):return self.value
        engine=Engine();engine.value=(True,{"original":object()})
        with self.make_scope("native") as scope:
            self.assertIs(scope.build(engine,engine.build),engine.value);scope.finish()
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native") as scope:scope.build(Engine(),engine.build)

    def test_consumed_pure_helper_source_code_defaults_and_aliases_are_owned(self):
        guard=subject.HelperOwners(((self.module,("_match",)),))
        guard.check(source=True)
        function=self.module._match;original_code=function.__code__
        try:
            function.__code__=(lambda *args:None).__code__
            with self.assertRaises(subject.FinishGPUFailure):guard.check()
        finally:function.__code__=original_code
        function.__defaults__=("changed",)
        with self.assertRaises(subject.FinishGPUFailure):guard.check()
        function.__defaults__=None
        old_registration=self.module.registration
        self.module.registration=SimpleNamespace()
        with self.assertRaises(subject.FinishGPUFailure):guard.check()
        self.module.registration=old_registration
        path=Path(self.module.__file__);path.write_text(CORE+"\n# changed bytes\n",encoding="utf-8")
        with self.assertRaises(subject.FinishGPUFailure):guard.check(source=True)

    def test_copied_registration_namespace_cannot_bypass_original_observers(self):
        original=self.module.REG
        try:
            with self.assertRaises(subject.FinishGPUFailure):
                with self.make_scope("native") as scope:
                    self.module.REG=SimpleNamespace(**vars(original))
                    try:scope.healthy()
                    except subject.FinishGPUFailure:pass
        finally:self.module.REG=original

    def test_bundle_original_objects_constraints_and_all_observers_preserved(self):
        bundle=self.make_bundle();original=bundle.propose_bundle_poses
        with self.make_scope("native",bundle_module=bundle) as scope:
            scope.np.asarray=lambda array:array
            self.assertIs(bundle.propose_bundle_poses(self.engine),bundle.proposal_value)
            make=next(r for r in scope.gates if r["function"]=="bundle_adjustment.make_problem")
            solve=next(r for r in scope.gates if r["function"]=="bundle_adjustment.solve_bundle")
            self.assertEqual(make["result"],solve["consumed_problem"])
            self.assertEqual(solve["consumed_problem"]["constraint_camera_ids"],[0,1,2])
            self.assertEqual(solve["consumed_problem"]["constraint_landmark_ids"],[0,0,0])
            self.assertTrue(all(r["complete"] for r in scope.gates));self.finish(scope)
        self.assertIs(bundle.propose_bundle_poses,original)

    def test_original_bundle_budget_and_import_routes_keep_exception_identity(self):
        for kind in ("budget","import"):
            bundle=self.make_bundle();error=bundle._BudgetExceeded("budget") if kind=="budget" else ImportError("optional")
            bundle.solve_error=error
            with self.make_scope("native",bundle_module=bundle) as scope:
                scope.np.asarray=lambda array:array
                value=bundle.propose_bundle_poses(self.engine)
                self.assertIs(bundle.caught[-1],error)
                self.assertEqual(value[1]["budget_exceeded"],kind=="budget")
                solve=next(r for r in scope.gates if r["function"]=="bundle_adjustment.solve_bundle")
                self.assertEqual(solve["result"]["original_exception"]["message"],str(error))
                self.finish(scope)

    def test_bundle_malformed_ids_and_input_mutation_are_hard_faults(self):
        for fault in ("ids","mutation"):
            bundle=self.make_bundle()
            if fault=="ids":bundle.problem_value.cameras.values[0]=3
            else:bundle.mutate_problem=True
            with self.assertRaises(subject.FinishGPUFailure):
                with self.make_scope("native",bundle_module=bundle) as scope:
                    scope.np.asarray=lambda array:array
                    bundle.propose_bundle_poses(self.engine)
            self.assertIsNotNone(scope.failure)

    def test_enabled_bundle_without_observers_fails_before_original_build(self):
        self.engine.settings=SimpleNamespace(bundle_adjustment=True)
        calls=[]
        with self.assertRaises(subject.FinishGPUFailure):
            with self.make_scope("native") as scope:scope.build(self.engine,lambda engine:calls.append(engine))
        self.assertEqual(calls,[])

    def test_bundle_evidence_import_errors_are_not_original_optional_noops(self):
        for phase in ("pre_capture","post_result"):
            bundle=self.make_bundle()
            def broken_evidence(value):raise ImportError("observer evidence fault")
            self.adapters.semantic=broken_evidence
            with self.assertRaises(subject.FinishGPUFailure) as raised:
                with self.make_scope("native",bundle_module=bundle) as scope:
                    scope.np.asarray=lambda array:array
                    if phase=="pre_capture":bundle._supported(bundle.problem_value,None)
                    else:bundle._verified_matches(None,None,None)
            self.assertIn("observer evidence fault",str(raised.exception))
            self.assertEqual(bundle.caught,[])


if __name__=="__main__":unittest.main()
