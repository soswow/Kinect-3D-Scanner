"""Minimal research-only whole-Finish bridge GPU candidate.

Original proposal order, competitors, witnesses, acceptance, optimization and
fusion remain in charge. Audit records exhaustive nearest shadows. Shadow
authorizes each current call from a CPU-first same-input certificate. Measure
uses a separately qualified METHOD, never a historical trajectory token.
No source/runtime inventory sweep or gate-result logging occurs at each gate.
"""
from __future__ import annotations
import contextvars
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType, SimpleNamespace
from typing import NamedTuple

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
POLICY="bridge-only-original-device-loop-minimal-candidate-v1"
MODES=("audit","shadow","measure")
CONFIGURATION=MappingProxyType({"device":0,"cuda_graph":True,"chunk_iterations":4,
    "max_points":1_000_000,"max_scratch_bytes":256*1024**2,"max_total_bytes":512*1024**2,
    "pair_cache_bytes":256*1024**2,"max_jobs":4096})
FILES=("scripts/research/gpu_icp_finish_candidate.py","tests/test_gpu_icp_finish_candidate.py")
HELPERS=("scripts/research/device_loop_icp.py","scripts/research/device_loop_workspace.py",
    "scripts/research/microbatch_bridge_driver.py","scripts/research/microbatch_bridge_scope.py",
    "scripts/research/gpu_icp_device_loop_experiment.py","scripts/research/gpu_icp_finish_scope.py",
    "scripts/research/field_finish_conformance_scope.py","scripts/research/finish_resident_registration.py",
    "scanner_server/cuda_registration.py","scanner_server/fragments.py","scanner_server/refinement.py")


class CandidateFailure(BaseException):
    """Research faults cannot become a rejected proposal or successful mesh."""


class AuthorizationState(NamedTuple):
    mode:str
    expected:str
    source:str
    configuration:tuple
    lease:object
    owner:object
    certificate:object
    protocol:object
    qualification:object
    validator:object


def require(value,message):
    if not value:raise CandidateFailure(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def source_contract():
    from scripts.research import device_loop_icp as method
    pins=dict(method.source_contract()["artifacts"])
    pins.update({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES+HELPERS})
    return {"policy":POLICY,"configuration":dict(CONFIGURATION),"artifacts":pins,
        "original_device_math":method.source_contract(),"original_clone_fields":27,
        "proposal_and_gate_bodies":"Captured original _verify_bridge and all helpers; no gate patches",
        "authorization":"Actual CPU-first certificate in shadow; fresh method qualification in measure",
        "historical_trajectory_authority":False,"general_domain_proof":False}


def normalized_consumed(full,configuration):
    def narrow(record):return {k:record[k] for k in ("dtype","shape","sha256")}
    return {"source":narrow(full["source"]["points"]),"target":narrow(full["target"]["points"]),
        "normals":narrow(full["target"]["normals"]),"seed":narrow(full["seed"]),
        "configuration":{"device":"CUDA:0","max_points":configuration["max_points"],
            "max_scratch_bytes":configuration["max_scratch_bytes"],"max_total_bytes":configuration["max_total_bytes"],
            "stages":[[.12,40],[.06,30],[.03,20]],"cuda_graph":True,"chunk_iterations":4,
            "audit_nearest":False,"audit_misses":False}}


class CallAuthorization:
    """One actual call; immutable expected bytes, source and CPU certificate."""
    def __init__(self,mode,full,source,configuration,lease,owner,*,certificate=None,
                 protocol=None,qualification=None,validator_state=None):
        require(mode in ("shadow","measure"),"Only explicit unaudited candidate modes authorize")
        self.state=AuthorizationState(mode,canonical(normalized_consumed(full,configuration)),canonical(source),
            tuple(sorted(configuration.items())),lease,owner,None if certificate is None else canonical(certificate),
            protocol,qualification,validator_state)
        self.state_owner=self.state
        self.used=False

    def __call__(self,consumed,source):
        require(self.state is self.state_owner,"Actual call authorization owner changed")
        state=self.state
        require(not self.used and canonical(consumed)==state.expected and canonical(source)==state.source,
            "Current cloud/normal/unrounded seed/config/source differs before GPU")
        require(state.lease.get("owner") is state.owner and state.lease.get("closed") is False
            and state.owner.closed is False and state.owner.failure is None
            and state.lease.get("device_id")==0,"Actual bounded target lease is not active")
        if state.mode=="shadow":
            require(state.certificate is not None,"CPU-first actual-call certificate required")
            certificate=json.loads(state.certificate)
            require(certificate["input"]==consumed and certificate["input_bytes_unchanged"] is True
                and certificate["completed"] is True and type(certificate["result"]) is dict,
                "Fresh original CPU call did not certify identical inputs before GPU")
        else:
            require(state.protocol is not None and state.qualification is not None,"Own method qualification required")
            if state.validator is not None:
                from scripts.research.gpu_icp_finish_scope import check_function
                require(state.protocol.validate_candidate_call is state.validator[0],"Own method authorizer slot changed")
                check_function(state.validator)
            state.protocol.validate_candidate_call(state.qualification,consumed=consumed,source=source,
                configuration=dict(state.configuration))
        self.used=True
        return {"method_policy":POLICY,"mode":state.mode,"actual_input_bound":True,
            "cpu_first":state.mode=="shadow","historical_input_reference":False}


def dependencies(np):
    from scripts.research import device_loop_icp as method, device_loop_workspace as clone
    from scripts.research import gpu_icp_finish_scope as owners
    from scripts.research import microbatch_bridge_scope as evidence
    from scripts.research import microbatch_bridge_driver as shared
    from scripts.research import gpu_icp_device_loop_experiment as experiment
    from scripts.research import field_finish_conformance_scope as field
    helper_guard=owners.HelperOwners(((evidence,("descriptor","input_binding","result_evidence","result_shadow","semantic")),
        (clone,("require","canonical","sha","code_state","SourceOwnerGuard","pristine","snapshot","verify_snapshot","fresh_clone")),
        (shared,("SharedBridgeCache","all_train_clouds","packet_descriptor")),
        (experiment,("terminal","descriptor")),
        (field,("final_pose_inventory","matrix_evidence")),
        (field.original,("evidence",))))
    return SimpleNamespace(method=method,clone=clone,owners=owners,helper_guard=helper_guard,
        capture=lambda a,b,s:evidence.input_binding(np,a,b,s),
        result=lambda r,a,b:evidence.result_evidence(np,r,len(a.points),len(b.points)),
        shadow=lambda a,b,s,t:evidence.result_shadow(np,a,b,len(s.points),len(t.points)),
        semantic=lambda value:evidence.semantic(np,value),clouds=shared.all_train_clouds,
        cache_factory=shared.SharedBridgeCache,packet=lambda p:shared.packet_descriptor(np,p),
        terminal=lambda r,l,a,b:experiment.terminal(np,r,l.statistics,len(a.points),len(b.points)),
        final_poses=field.final_pose_inventory)


class GpuICPFinishCandidate:
    def __init__(self,registration,fragments,refinement,*,mode="shadow",retrieval_factory=None,
                 numpy=None,engine=None,binding=None,boundary=None,trace=None,
                 qualification=None,protocol=None,adapters=None):
        # A retained object.__new__ owner can close/report even if cold setup rejects.
        self.mode=mode;self.failure=None;self.cleanup_failures=[];self.patches=[]
        self.retrieval=self.template=self.shared=self.active=None
        self.retrieval_owner=self.template_owner=self.shared_owner=None
        self.calls=[];self.pairs=[];self.jobs=[];self.cache_receipts=[];self.final_poses=None
        self.successful_builds=0;self.complete=self.restored=self.closed=self.entered=False
        self.default_evidence=adapters is None
        self.stats={"cold_setup_wall_s":0.,"pair_setup_wall_s":0.,"pair_close_wall_s":0.,
            "cpu_shadow_wall_s":0.,"actual_call_wall_s":0.,"cleanup_wall_s":0.,"non_bridge_dispatch_calls":0}
        require(mode in MODES and callable(retrieval_factory),"Explicit candidate mode and retrieval factory required")
        require((mode=="measure")== (qualification is not None),"Only measure consumes fresh method qualification")
        self.module,self.fragments,self.refinement=registration,fragments,refinement
        self.mode,self.factory,self.np,self.engine=mode,retrieval_factory,numpy,engine
        self.binding,self.boundary,self.trace=binding,boundary or (lambda:None),trace or (lambda r:None)
        self.qualification,self.protocol=qualification,protocol
        self.default_evidence=adapters is None
        self.adapters=dependencies(numpy) if adapters is None else adapters
        self.configuration=tuple(sorted(CONFIGURATION.items()));self.policy_owner=CONFIGURATION
        self.original_dispatch,self.original_match,self.original_bridge=registration.match,refinement._match,fragments._verify_bridge
        require(fragments._match is self.original_match and fragments.REG is refinement.REG,"Original registration aliases required")
        self.original_reg=refinement.REG;self.selector=registration._device
        self.owners=(registration,fragments,refinement,mode,retrieval_factory,numpy,engine,binding,
            self.boundary,self.trace,qualification,protocol,self.adapters,self.original_dispatch,
            self.original_match,self.original_bridge,self.original_reg,self.selector)
        self.bypass=contextvars.ContextVar("candidate-original-cpu",default=False)
        self.context=contextvars.ContextVar("candidate-original-bridge",default=None)
        self.guard=self.adapters.owners.WrapperOwnerGuard((registration,fragments,refinement))
        self.guard.add(refinement,"_match",self.original_match,self.original_match)
        self.guard.add(fragments,"_match",self.original_match,self.original_match)
        self.expected=None;self.pair_key=None;self.pair_objects=None
        self.contract=source_contract()
        self.method_guard=self.method_validator_state=None
        self.own_guard=self.adapters.owners.HelperOwners(((sys.modules[__name__],
            ("require","canonical","source_contract","normalized_consumed","CallAuthorization","dependencies","GpuICPFinishCandidate")),))
        self.hot_states=tuple((owner,name,self.adapters.owners.function_state(getattr(owner,name))) for owner,name in (
            (GpuICPFinishCandidate,"dispatch"),(GpuICPFinishCandidate,"_critical"),
            (GpuICPFinishCandidate,"_critical_checks"),
            (GpuICPFinishCandidate,"_cpu"),(GpuICPFinishCandidate,"_lease"),
            (CallAuthorization,"__call__")))
        if mode=="measure":
            require(protocol is not None and binding is not None,"Fresh candidate method binding required")
            if self.default_evidence:
                from scripts.research import compare_gpu_icp_candidate_finishes as own_protocol
                require(protocol is own_protocol,"Only the new candidate method qualification registry is accepted")
                self.method_guard=self.adapters.owners.HelperOwners(((protocol,
                    ("assert_registered_candidate_qualification","validate_candidate_method","validate_candidate_call")),))
                self.method_guard.check(source=True)
                protocol.assert_registered_candidate_qualification(qualification)
            protocol.validate_candidate_method(qualification,binding=binding,configuration=dict(CONFIGURATION))
            self.method_validator_state=self.adapters.owners.function_state(protocol.validate_candidate_call)

    def fail(self,error):
        if self.failure is None:
            self.failure=error if isinstance(error,CandidateFailure) else CandidateFailure("GPU candidate failed: "+repr(error))
            if self.failure is not error:self.failure.__cause__=error
        raise self.failure

    def _critical(self):
        try:self._critical_checks()
        except BaseException as error:self.fail(error)

    def _critical_checks(self):
        if self.failure is not None:raise self.failure
        require(CONFIGURATION is self.policy_owner and tuple(sorted(CONFIGURATION.items()))==self.configuration,
            "Candidate configuration changed")
        actual=(self.module,self.fragments,self.refinement,self.mode,self.factory,self.np,self.engine,self.binding,
            self.boundary,self.trace,self.qualification,self.protocol,self.adapters,self.original_dispatch,
            self.original_match,self.original_bridge,self.original_reg,self.selector)
        require(all(a is b for a,b in zip(actual,self.owners)),"Candidate method/callback owner changed")
        require(self.fragments._match is self.original_match and self.refinement._match is self.original_match
            and self.fragments.REG is self.original_reg and self.refinement.REG is self.original_reg
            and self.module._device is self.selector and self.selector.get() is None
            and os.environ.get("KINECT_CUDA_REGISTRATION")=="cpu","Original CPU dispatch/gate owners changed")
        require(self.retrieval is self.retrieval_owner and self.template is self.template_owner
            and self.shared is self.shared_owner,"Candidate resource owner changed")
        for owner,name,state in self.hot_states:
            require(getattr(owner,name) is state[0],"Critical candidate class slot changed")
            self.adapters.owners.check_function(state)
        self.guard.check()  # Four installed/original callable slots, not a process inventory.

    def _cold(self):
        self._critical();self.boundary()
        require(source_contract()==self.contract,"Candidate/math/helper source bytes changed")
        self.own_guard.check(source=True)
        if self.default_evidence:self.adapters.helper_guard.check(source=True)
        if self.method_guard is not None:self.method_guard.check(source=True)
        self.guard.check(source=True)
        if self.template is not None:
            self.math_guard.check();self.adapters.clone.verify_snapshot(self.template,self.expected)

    def _cpu(self,source,target,seed):
        token=self.bypass.set(True)
        try:return self.original_match(source,target,seed)
        finally:self.bypass.reset(token)

    def _install(self,module,name,value):
        old=getattr(module,name);self.guard.add(module,name,old,value)
        self.patches.append((module,name,old,value));setattr(module,name,value)

    def __enter__(self):
        require(not self.entered,"Candidate context cannot be reused")
        self.entered=True
        try:
            self._install(self.module,"match",self.dispatch)
            self._install(self.fragments,"_verify_bridge",self.bridge)
            self._cold();return self
        except BaseException as error:
            self.failure=error;self.__exit__(type(error),error,error.__traceback__)

    def _setup(self):
        if self.template is not None:return
        begin=time.perf_counter()
        try:self.retrieval=self.factory()
        except BaseException as error:
            self.retrieval=getattr(error,"candidate_retrieval",None)
            self.retrieval_owner=self.retrieval
            raise
        self.retrieval_owner=self.retrieval
        method=self.adapters.method;self.math_guard=self.adapters.clone.SourceOwnerGuard(method)
        self.template=object.__new__(method.DeviceLoopICP);self.template_owner=self.template
        def forbidden(*args):raise CandidateFailure("Pristine template can never start")
        self.template.__init__(self.retrieval,device=0,cuda_graph=True,
            max_points=CONFIGURATION["max_points"],max_scratch_bytes=CONFIGURATION["max_scratch_bytes"],
            max_total_bytes=CONFIGURATION["max_total_bytes"],audit_nearest=self.mode=="audit",audit_misses=self.mode=="audit",
            timing_authorizer=None if self.mode=="audit" else forbidden)
        self.expected=self.adapters.clone.snapshot(self.template)
        self.stats["cold_setup_wall_s"]+=time.perf_counter()-begin

    def _close_pair(self):
        if self.shared is None:return
        begin=time.perf_counter()
        require(self.active is None,"Active lane forbids pair cache release")
        t=self.template
        with t.cp.cuda.Device(t.device),t.stream:t.stream.synchronize()
        self.shared.close();require(self.shared.closed,"Pair stream completion unproved")
        self.cache_receipts.append(self.shared.report());self.shared=self.shared_owner=None
        self.stats["pair_close_wall_s"]+=time.perf_counter()-begin

    def bridge(self,source,target,seed,*args,**kwargs):
        self._critical();require(self.context.get() is None,"Nested original verification unsupported")
        key=(int(source.index),int(target.index))
        require(key[0]!=key[1] and min(key)>=0,"Directed fragment IDs required")
        try:
            if key!=self.pair_key:
                self._close_pair();self._cold();self.pair_key=key;self.pair_objects=(source,target)
                self.pairs.append({"pair_index":len(self.pairs),"pair":list(key),"proposals":[]})
            require(self.pair_objects[0] is source and self.pair_objects[1] is target,"Pair owners replaced")
            pair=self.pairs[-1];row={"proposal_index":len(pair["proposals"]),"complete":False}
            pair["proposals"].append(row)
            token=self.context.set({"pair_index":pair["pair_index"],"pair":pair["pair"],"proposal_index":row["proposal_index"]})
            try:
                result=self.original_bridge(source,target,seed,*args,**kwargs)
                self._critical();row.update(result=self.adapters.semantic(result),complete=True)
                return result
            finally:self.context.reset(token)
        except BaseException as error:self.fail(error)

    def _lease(self,source,target):
        self._setup()
        if self.shared is None:
            begin=time.perf_counter();factory=self.adapters.cache_factory
            self.shared=object.__new__(factory);self.shared_owner=self.shared
            self.shared.__init__(self.retrieval,self.adapters.clouds(*self.pair_objects))
            require(self.shared.owned_bytes<=CONFIGURATION["pair_cache_bytes"],"Bounded immutable pair cache exceeded")
            self.stats["pair_setup_wall_s"]+=time.perf_counter()-begin
        return self.shared.lease(source,target)

    def dispatch(self,source,target,seed):
        if self.bypass.get():return None
        self._critical();context=self.context.get()
        if context is None:
            self.stats["non_bridge_dispatch_calls"]+=1
            return self.original_dispatch(source,target,seed)
        require(len(self.calls)<CONFIGURATION["max_jobs"],"Candidate actual call cap exceeded")
        row=dict(context,call_index=len(self.calls),complete=False,query_trace=[],cleanup_failures=[])
        self.calls.append(row);lane=lease=primary=result=native=None;begin=time.perf_counter()
        try:
            before=self.adapters.capture(source,target,seed);row["input_binding"]=before
            if self.mode=="shadow":
                cpu_begin=time.perf_counter();native=self._cpu(source,target,seed)
                self.stats["cpu_shadow_wall_s"]+=time.perf_counter()-cpu_begin
                require(self.adapters.capture(source,target,seed)==before,"CPU-first call mutated inputs")
                row["cpu_first_certificate"]={"input":normalized_consumed(before,dict(CONFIGURATION)),
                    "result":self.adapters.result(native,source,target),"input_bytes_unchanged":True,"completed":True}
            lease=self._lease(source,target)
            certificate=row.get("cpu_first_certificate")
            auth=None if self.mode=="audit" else CallAuthorization(self.mode,before,self.template.provenance,
                dict(CONFIGURATION),lease,self.shared,certificate=certificate,protocol=self.protocol,qualification=self.qualification,
                validator_state=self.method_validator_state)
            def observe(query):
                packet=query["packet"];order=self.np.argsort(packet[:,0])
                row["query_trace"].append({"query_index":query["query_index"],"stage":query["stage"],"radius":query["radius"],
                    "target_sha256":query["target_sha256"],"packet":self.adapters.packet(packet[order]),
                    "corrected_ids_sha256":hashlib.sha256(query["corrected_ids"][order].tobytes()).hexdigest(),
                    "counters":list(query["counters"])})
            lane=self.adapters.clone.fresh_clone(self.template,audit_observer=observe if self.mode=="audit" else None,
                timing_authorizer=auth)
            self.active=lane
            result=lane.match(source,target,seed,chunk_iterations=4,pair_lease=lease)
            require(lane.configuration==self.template.configuration and lane.authorizer is auth
                and all(lane.kernels[k] is v for k,v in self.expected["kernels"].items()),"Lane configuration/kernel/authorizer changed")
            row.update(consumed_input_binding=copy.deepcopy(lane.input_binding),source_binding=copy.deepcopy(lane.provenance),
                terminal=self.adapters.terminal(result,lane,source,target),result=self.adapters.result(result,source,target))
            if self.mode=="audit":
                cpu_begin=time.perf_counter();native=self._cpu(source,target,seed)
                self.stats["cpu_shadow_wall_s"]+=time.perf_counter()-cpu_begin
            if native is not None:
                row["native_shadow"]=self.adapters.shadow(native,result,source,target)
                require(row["native_shadow"].get("passed") is True,"Actual complete original CPU shadow failed")
            else:row["native_shadow"]={"collected":False}
            row["authorization"]={"mode":self.mode,"actual_input_bound":self.mode=="audit" or auth.used,
                "cpu_first":self.mode=="shadow","method_qualification":self.mode=="measure","historical_input_reference":False}
            require(self.adapters.capture(source,target,seed)==before,"Actual GPU call mutated original inputs/seed")
            row["input_bytes_unchanged"]=True
        except BaseException as error:primary=error;row["failure"]={"type":type(error).__name__,"message":str(error)}
        finally:
            if lane is not None:
                try:
                    lane.close();require(lane.closed,"Lane selected stream completion unproved")
                    if lease is not None:self.shared.release_lease(lease)
                    self.active=None
                except BaseException as error:
                    row["cleanup_failures"].append(repr(error));primary=primary or error
                try:row["loop_report"]=lane.report()
                except BaseException as error:
                    row["cleanup_failures"].append(repr(error));primary=primary or error
            row["actual_call_wall_s_inclusive"]=time.perf_counter()-begin
            self.stats["actual_call_wall_s"]+=row["actual_call_wall_s_inclusive"]
        try:
            if primary is not None:self.fail(primary)
            row["query_trace_sha256"]=hashlib.sha256(canonical(row["query_trace"]).encode()).hexdigest()
            row["complete"]=True;self.jobs.append({"index":row["call_index"],"closed":True,"report":row["loop_report"]})
            self._critical();self.trace(row);return result
        except BaseException as error:
            row["complete"]=False;row.setdefault("failure",{"type":type(error).__name__,"message":str(error)})
            self.fail(error)

    def build(self,engine,original_build,*args,**kwargs):
        try:
            self._critical();require(engine.unprocessed_count==0 and self.successful_builds==0,"Complete fresh Live before one Finish")
            value=original_build(engine,*args,**kwargs)
            self._critical();require(type(value) is tuple and len(value)==2 and value[0] is True,"Original Final build failed")
            self.final_poses=self.adapters.final_poses(engine);self.successful_builds=1
            return value
        except BaseException as error:self.fail(error)

    def finish(self):
        try:
            self._cold();require(self.successful_builds==1 and self.final_poses is not None
                and all(r["complete"] for r in self.calls)
                and all(r["complete"] for p in self.pairs for r in p["proposals"]),"Incomplete original proposal/call/build suffix")
            self.complete=True
        except BaseException as error:self.fail(error)

    def __exit__(self,kind,error,traceback):
        begin=time.perf_counter();primary=self.failure or error
        if primary is None:
            try:self._cold()
            except BaseException as fault:primary=fault
        self.template,self.retrieval,self.shared=self.template_owner,self.retrieval_owner,self.shared_owner
        pending=[]
        for module,name,old,installed in reversed(self.patches):
            try:
                require(getattr(module,name) is installed,"Installed hook replaced before cleanup")
            except BaseException as fault:self.cleanup_failures.append(repr(fault))
            try:setattr(module,name,old)
            except BaseException as fault:self.cleanup_failures.append(repr(fault));pending.append((module,name,old,installed))
        self.patches=list(reversed(pending))
        self.restored=not self.cleanup_failures
        completed=True
        if self.active is not None:
            try:self.active.close();require(self.active.closed,"Active lane completion unproved");self.active=None
            except BaseException as fault:completed=False;self.cleanup_failures.append(repr(fault))
        if self.template is not None:
            try:self.template.close();require(self.template.closed,"Template stream completion unproved")
            except BaseException as fault:completed=False;self.cleanup_failures.append(repr(fault))
        if completed:
            if self.shared is not None:
                try:self.shared.close();require(self.shared.closed,"Pair cache completion unproved")
                except BaseException as fault:self.cleanup_failures.append(repr(fault))
                else:
                    try:self.cache_receipts.append(self.shared.report())
                    except BaseException as fault:self.cleanup_failures.append(repr(fault))
                    else:self.shared=self.shared_owner=None
            elif self.retrieval is not None:
                try:self.retrieval.close()
                except BaseException as fault:self.cleanup_failures.append(repr(fault))
        else:self.cleanup_failures.append("Retained numerical owners: stream completion unproved")
        self.closed=completed and self.shared is None and not self.cleanup_failures
        self.stats["cleanup_wall_s"]+=time.perf_counter()-begin
        if primary is None and (self.cleanup_failures or not self.complete):primary=CandidateFailure("Candidate cleanup/suffix incomplete")
        if primary is not None:
            self.failure=primary
            note=getattr(primary,"add_note",None)
            if callable(note):
                for value in self.cleanup_failures:note(value)
            raise primary
        return False

    def close(self,primary=None):
        """Independent cleanup is safe after a successful or failed context exit."""
        if self.closed and self.restored:
            if primary is not None:raise primary
            if self.failure is not None:raise self.failure
            return
        # Constructor rejection creates no numerical resources or hooks.
        if not hasattr(self,"stats"):
            if primary is not None:raise primary
            raise CandidateFailure("Incomplete candidate construction")
        return self.__exit__(None if primary is None else type(primary),primary,
            None if primary is None else primary.__traceback__)

    def report(self):
        return {"policy":POLICY,"mode":self.mode,"configuration":dict(CONFIGURATION),"complete":self.complete and self.failure is None,
            "restored":self.restored,"closed":self.closed,"failure":None if self.failure is None else repr(self.failure),
            "cleanup_failures":list(self.cleanup_failures),"default_evidence":self.default_evidence,
            "calls":copy.deepcopy(self.calls),"pairs":copy.deepcopy(self.pairs),"events":[],"gates":[],"graphs":[],
            "workspace":None if self.template is None else {"closed":getattr(self.template,"closed",False),
                "template_started":getattr(self.template,"started",False),"active_lane":self.active is not None,
                "jobs":copy.deepcopy(self.jobs),"original_constructor_fields":27},
            "cache_receipts":copy.deepcopy(self.cache_receipts),"successful_builds":self.successful_builds,
            "final_pose_inventory":copy.deepcopy(self.final_poses),"statistics":dict(self.stats),
            "uncollected":["gate_events","graph_calls"],"historical_trajectory_authority":False,
            "general_domain_proof":False,"whole_finish_authority":False}
