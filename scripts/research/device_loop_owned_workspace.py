"""Separate owned-proof reuse of unchanged compiled complete-device ICP setup.

The pinned DeviceLoopICP constructor runs once on a never-started template.
Every job receives a fresh, explicitly inventoried original-class instance;
only compiled code handles, immutable configuration and one completed stream
are shared. Original start/advance/result/close methods are not overridden.
This new ownership method has no historical or v3 timing authorization.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import CodeType, FunctionType

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
POLICY="sequential-pristine-template-device-loop-owned-proof-workspace-v2"
LOOP=ROOT/"scripts/research/device_loop_icp.py"
LOOP_SHA256="309260e5048b5578e948e449d8b92e3609402e65f3b7b4f6c9cdf1666ef8cbfb"
OLD=ROOT/"scripts/research/device_loop_workspace.py"
OLD_SHA256="45ee20f5d020fd34c1b9e45886cf57d86f6c4221b3453bccdcf727bb47ab5d4d"
NEW_FILES=('scripts/research/device_loop_owned_workspace.py', 'scripts/research/microbatch_bridge_owned_driver.py', 'scripts/research/microbatch_bridge_owned_protocol.py', 'tests/test_device_loop_owned_workspace.py', 'tests/test_microbatch_bridge_owned_driver.py', 'tests/test_microbatch_bridge_owned_protocol.py')
OLD_PINS={'scripts/research/device_loop_workspace.py': '45ee20f5d020fd34c1b9e45886cf57d86f6c4221b3453bccdcf727bb47ab5d4d', 'scripts/research/device_loop_workspace_protocol.py': '12d5995b19b3d525fd3cf119628936f7bcee8db611b9a1011f0d2c18d6d2214d', 'scripts/research/microbatch_bridge_protocol.py': '1d5a150e634bd21d92264452052123f490fdc1c32174fe282b107d57cc26a04b', 'scripts/research/microbatch_bridge_scope.py': 'a9303b1f85d8eaaa427e8a873b5c32b623ac16d971725c418c8050202014bf80', 'scripts/research/microbatch_bridge_driver.py': 'd048525610ac41d4b591ed5779d3d01ad06876f786d662fe067fcea8e6d2c2ec', 'tests/test_device_loop_workspace.py': '116005ef59d7182c8c6886024cddeccab7d3d689e8527d8abc983bd9a7269b52', 'tests/test_microbatch_bridge_protocol.py': 'b2feff4314ec4f996231c2612a978b96a2fd18f1a1b002dd18a86466fb79f0d6', 'tests/test_microbatch_bridge.py': '72e840905e36fcce7409a668695993e79a4aa0509dc80088b85fab04251aacb4'}
FILES=NEW_FILES+tuple(OLD_PINS)
FIELDS=frozenset(("cp","np","retrieval","device","max_points","max_scratch_bytes","max_total_bytes",
    "audit_nearest","audit_misses","cuda_graph","configuration","authorizer","audit_observer",
    "buffers","lease","failure","grid_owners","graph","graph_chunk","pending_host_packets",
    "started","closed","provenance","statistics","stream","module","kernels"))
IDENTITY_FIELDS=("cp","np","retrieval","stream","module")
VALUE_FIELDS=("device","max_points","max_scratch_bytes","max_total_bytes","audit_nearest",
    "audit_misses","cuda_graph","configuration","authorizer","audit_observer")


class WorkspaceError(RuntimeError):
    """Sticky ownership/configuration/completion fault; math cannot be retried."""


def require(value,message):
    if not value:raise WorkspaceError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def code_state(code):
    return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,
        code.co_argcount,code.co_posonlyargcount,code.co_kwonlyargcount,code.co_flags,
        tuple(code_state(c) if isinstance(c,CodeType) else c for c in code.co_consts))


def derivation_contract(candidate_source=None):
    """Exact held workspace bodies, inverse only the two protocol imports."""
    require(sha(OLD)==OLD_SHA256,"Held v1 workspace changed; no old authority can migrate")
    original=ast.parse(OLD.read_text(encoding="utf-8"))
    candidate=ast.parse(Path(__file__).read_text(encoding="utf-8") if candidate_source is None else candidate_source)
    def node(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    old_class=node(original,"DeviceLoopWorkspace")
    new_class=copy.deepcopy(node(candidate,"DeviceLoopWorkspace"))
    changes=0
    for fn in new_class.body:
        if isinstance(fn,ast.FunctionDef) and fn.name in ("__init__","new_lane"):
            for n in ast.walk(fn):
                if isinstance(n,ast.ImportFrom) and n.module=="scripts.research":
                    for alias in n.names:
                        if alias.name=="microbatch_bridge_owned_protocol" and alias.asname=="protocol":
                            alias.name="device_loop_workspace_protocol";changes+=1
    require(changes==2 and ast.dump(old_class,include_attributes=False)==ast.dump(new_class,include_attributes=False),
        "Owned workspace may change only its two registered-protocol imports; numerical/lifecycle bodies remain exact")
    for name in ("require","sha","canonical","code_state","SourceOwnerGuard","pristine","snapshot","verify_snapshot","fresh_clone"):
        require(ast.dump(node(original,name),include_attributes=False)==ast.dump(node(candidate,name),include_attributes=False),
            "Held workspace guard/clone/native-owner primitive changed: "+name)
    return {"held_workspace_sha256":OLD_SHA256,"inverse_protocol_imports":changes,
        "workspace_class_ast_sha256":hashlib.sha256(ast.dump(old_class,include_attributes=False).encode()).hexdigest(),
        "held_guards_clones_and_lifecycle_exact":True}


def source_contract():
    derived=derivation_contract()
    require(sha(LOOP)==LOOP_SHA256,"Workspace only knows exact held v2 ICP constructor/method inventory")
    require(all(sha(ROOT/name)==digest for name,digest in OLD_PINS.items()),"Any held v1 source changed")
    tree=ast.parse(LOOP.read_text(encoding="utf-8"))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DeviceLoopICP")
    constructor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    stored={n.attr for n in ast.walk(constructor) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)
        and isinstance(n.value,ast.Name) and n.value.id=="self"}
    require(stored==FIELDS,"Constructor fields differ from explicitly reviewed pristine template schema")
    return {"policy":POLICY,"held_loop_sha256":LOOP_SHA256,"constructor_fields":sorted(FIELDS),
        "constructor_ast_sha256":hashlib.sha256(ast.dump(constructor,include_attributes=False).encode()).hexdigest(),
        "artifacts":{name:sha(ROOT/name) for name in FILES},"maximum_active_lanes":1,
        "template_never_started":True,"fresh_graph_per_trajectory":True,"registered_timing_supported":True,
        "proof_ownership":"Separate unified owned-proof registry, locked report and immutable cached reference; no held v1 token",
        "derivation":derived,
        "shared":"compiled module/kernel handles, immutable configuration and selected completed stream",
        "private":"all buffers, phase/control/statistics/provenance, graphs, input/grid/lease owners"}


class SourceOwnerGuard:
    def __init__(self,module):
        self.module=module;self.path=Path(module.__file__).resolve()
        require(self.path==LOOP and sha(self.path)==LOOP_SHA256,"Wrong loaded device-loop source owner")
        expected=compile(self.path.read_text(encoding="utf-8"),str(self.path),"exec")
        self.functions=[]
        for code in expected.co_consts:
            if not isinstance(code,CodeType):continue
            owner=getattr(module,code.co_name)
            if isinstance(owner,FunctionType):self.add(module,code.co_name,owner,code)
            elif isinstance(owner,type):
                for item in code.co_consts:
                    if isinstance(item,CodeType):self.add(owner,item.co_name,getattr(owner,item.co_name),item)
        self.objects={name:getattr(module,name) for name in
            ("DeviceLoopICP","POLICY","STAGES","MAX_POINTS","ROOT","BLOCK","PHASES",
             "CONTROL","RESIDENT","FLAT","CLASSIFIER","LDLT")}

    def add(self,owner,name,function,expected):
        require(isinstance(function,FunctionType) and function.__globals__ is self.module.__dict__
            and code_state(function.__code__)==code_state(expected),
            "Loaded constructor/method differs from exact original source")
        self.functions.append((owner,name,function,function.__code__,repr(function.__defaults__),repr(function.__kwdefaults__)))

    def check(self):
        require(sys.modules.get(self.module.__name__) is self.module and Path(self.module.__file__).resolve()==self.path,
            "Loaded loop module owner changed")
        for name,value in self.objects.items():require(getattr(self.module,name) is value,"Loaded loop control/source owner changed")
        for owner,name,function,code,defaults,kwdefaults in self.functions:
            require(getattr(owner,name) is function and function.__code__ is code
                and repr(function.__defaults__)==defaults and repr(function.__kwdefaults__)==kwdefaults,
                "Loaded constructor/method/code/default owner changed")


def pristine(template):
    require(set(template.__dict__)==FIELDS,"Unknown or already-started template fields cannot be cloned")
    require(template.started is False and template.closed is False and template.failure is None
        and template.buffers=={} and template.grid_owners==[] and template.graph is None
        and template.graph_chunk is None and template.pending_host_packets is None and template.lease is None,
        "Template must remain pristine and never start a trajectory")
    require(type(template.audit_nearest) is bool and template.audit_nearest is template.audit_misses
        and (template.authorizer is None if template.audit_nearest else callable(template.authorizer))
        and template.audit_observer is None,"Template has invalid exhaustive-audit or registered timing policy")
    require(type(template.statistics) is dict and set(template.statistics) and all(
        value==0 for key,value in template.statistics.items() if key!="setup_s"),"Template statistics contain previous work")


def snapshot(template):
    pristine(template)
    return {"identity":{name:getattr(template,name) for name in IDENTITY_FIELDS},
        "values":{name:getattr(template,name) for name in VALUE_FIELDS},
        "provenance":canonical(template.provenance),"statistics":dict(template.statistics),
        "kernels":dict(template.kernels),"container_owners":(template.buffers,template.grid_owners,
            template.provenance,template.statistics,template.kernels)}


def verify_snapshot(template,expected):
    pristine(template)
    require(all(getattr(template,name) is owner for name,owner in expected["identity"].items()),"Shared template native owner replaced")
    require(all(getattr(template,name)==value for name,value in expected["values"].items()),"Immutable template configuration changed")
    require(canonical(template.provenance)==expected["provenance"] and template.statistics==expected["statistics"],
        "Pristine template provenance/counters changed")
    require(set(template.kernels)==set(expected["kernels"]) and all(template.kernels[k] is v for k,v in expected["kernels"].items()),
        "Compiled kernel handle ownership changed")
    require(all(getattr(template,name) is owner for name,owner in zip(
        ("buffers","grid_owners","provenance","statistics","kernels"),expected["container_owners"])),
        "Pristine mutable containers were replaced")


def fresh_clone(template,*,audit_observer=None,timing_authorizer=None):
    """All original constructor fields explicitly initialized; no phase copied."""
    pristine(template)
    lane=object.__new__(type(template))
    for name in IDENTITY_FIELDS+VALUE_FIELDS:setattr(lane,name,getattr(template,name))
    lane.audit_observer=audit_observer
    if not template.audit_nearest:
        require(callable(timing_authorizer),"Every timing clone needs its own registered ordered-call authorizer")
        lane.authorizer=timing_authorizer
    lane.buffers={};lane.lease=None;lane.failure=None;lane.grid_owners=[]
    lane.graph=None;lane.graph_chunk=None;lane.pending_host_packets=None
    lane.started=False;lane.closed=False
    lane.provenance=copy.deepcopy(template.provenance)
    lane.statistics={name:0. if isinstance(value,float) else 0 for name,value in template.statistics.items()}
    lane.kernels=dict(template.kernels)
    require(set(lane.__dict__)==FIELDS,"Clone lacks an explicitly inventoried constructor field")
    return lane


class DeviceLoopWorkspace:
    def __init__(self,retrieval,*,device=0,cuda_graph=True,max_points=1_000_000,
                 max_scratch_bytes=256*1024**2,max_total_bytes=512*1024**2,max_jobs=4096,
                 audit=True,timing_permit=None):
        require(type(max_jobs) is int and 1<=max_jobs<=4096,"Bound sequential workspace jobs")
        self.closed=False;self.failure=None;self.active=None;self.max_jobs=max_jobs
        self.jobs=[];self.retained_failed_lanes=[]
        self.statistics={"cold_setup_wall_s":0.,"clone_guard_wall_s":0.,"lane_close_wall_s":0.,"workspace_close_wall_s":0.}
        begin=time.perf_counter()
        self.provenance=source_contract()
        require(type(audit) is bool,"Declare audit or separately authorized timing mode")
        self.timing_permit=timing_permit
        from scripts.research import microbatch_bridge_owned_protocol as protocol
        if audit:require(timing_permit is None,"Audit cannot consume any timing permit")
        else:protocol.constructor_authority(timing_permit,self.provenance,{"device":f"CUDA:{device}",
            "cuda_graph":cuda_graph,"max_points":max_points,"max_scratch_bytes":max_scratch_bytes,
            "max_total_bytes":max_total_bytes})
        from scripts.research import device_loop_icp as method
        self.guard=SourceOwnerGuard(method)
        self.template=method.DeviceLoopICP(retrieval,device=device,cuda_graph=cuda_graph,max_points=max_points,
            max_scratch_bytes=max_scratch_bytes,max_total_bytes=max_total_bytes,
            audit_nearest=audit,audit_misses=audit,
            timing_authorizer=None if audit else lambda *_: (_ for _ in ()).throw(WorkspaceError("Pristine template must never start")),
            audit_observer=None)
        self.expected=snapshot(self.template)
        self.file_pins=dict(self.provenance["artifacts"],**self.template.provenance["artifacts"])
        self.runtime=self.runtime_binding()
        self.statistics["cold_setup_wall_s"]=time.perf_counter()-begin

    def runtime_binding(self):
        t=self.template;cp,np=t.cp,t.np
        return {"device":t.device,"stream_ptr":int(t.stream.ptr),"numpy":np.__version__,"cupy":cp.__version__,
            "driver":cp.cuda.runtime.driverGetVersion(),"runtime":cp.cuda.runtime.runtimeGetVersion(),
            "nvrtc":list(cp.cuda.nvrtc.getVersion()),
            "numpy_configuration_sha256":hashlib.sha256(canonical(getattr(np.__config__,"CONFIG",{})).encode()).hexdigest()}

    def _healthy(self):
        require(not self.closed and self.failure is None,"Closed/damaged workspace cannot start another trajectory")
        self.guard.check();verify_snapshot(self.template,self.expected)
        require(self.runtime_binding()==self.runtime,"Loaded workspace runtime/compiler changed")
        require(all(sha(ROOT/name)==digest for name,digest in self.file_pins.items()),"Workspace/held mathematical source changed")

    def new_lane(self,*,audit_observer=None):
        begin=time.perf_counter()
        try:
            self._healthy()
            require(self.active is None,"Only one sequential active lane; close it before requesting another")
            require(len(self.jobs)<self.max_jobs,"Bounded sequential job inventory exhausted")
            from scripts.research import microbatch_bridge_owned_protocol as protocol
            index=len(self.jobs)
            authorizer=(None if self.template.audit_nearest else lambda consumed,source:
                protocol.validate_start(self.timing_permit,index,consumed,source))
            lane=fresh_clone(self.template,audit_observer=audit_observer,timing_authorizer=authorizer)
            lane.provenance["setup_reuse"]=copy.deepcopy(self.provenance)
            self.active=lane
            self.active_authorizer=lane.authorizer
            self.jobs.append({"index":len(self.jobs),"closed":False,"failure":None})
            return lane
        except BaseException as error:
            self.failure=self.failure or error
            raise
        finally:self.statistics["clone_guard_wall_s"]+=time.perf_counter()-begin

    def check_lane(self,lane):
        self._healthy()
        require(lane is self.active and type(lane) is type(self.template),"Exact active original-class lane required")
        require(all(getattr(lane,name) is owner for name,owner in self.expected["identity"].items()),
            "Lane shared code/stream/cache/numerical owner changed")
        require(lane.configuration==self.template.configuration and lane.authorizer is self.active_authorizer,
            "Lane execution/authorization configuration changed")
        require(set(lane.kernels)==set(self.expected["kernels"]) and all(lane.kernels[k] is v
            for k,v in self.expected["kernels"].items()),"Lane compiled kernel handle ownership changed")
        require(lane.provenance==dict(self.template.provenance,setup_reuse=self.provenance),
            "Lane source/runtime setup-reuse provenance changed")
        return True

    def close_lane(self,lane,*,primary=None):
        begin=time.perf_counter()
        try:
            require(lane is self.active,"Only the exact workspace-owned active lane can be closed")
            if primary is None and lane.failure is None and self.failure is None:self.check_lane(lane)
            try:
                lane.close()
                require(lane.closed is True,"Original close did not prove selected-stream completion")
            except BaseException as cleanup:
                self.failure=self.failure or primary or cleanup
                if lane not in self.retained_failed_lanes:self.retained_failed_lanes.append(lane)
                if primary is not None:raise primary from cleanup
                raise
            self.jobs[-1].update(closed=True,failure=None if lane.failure is None else repr(lane.failure),
                report=lane.report())
            self.active=None
            if primary is not None or lane.failure is not None:
                self.failure=self.failure or primary or lane.failure
            elif self.failure is None:self._healthy()
        except BaseException as error:
            self.failure=self.failure or primary or error
            raise
        finally:self.statistics["lane_close_wall_s"]+=time.perf_counter()-begin

    def close(self,*,primary=None):
        if self.closed:return
        begin=time.perf_counter()
        try:
            if self.active is not None:self.close_lane(self.active,primary=primary)
            # Original template close synchronizes the shared stream. Cache ownership
            # stays with caller; release it only after workspace.closed is true.
            try:
                self.template.close()
                require(self.template.closed is True,"Workspace selected stream completion is unproved")
            except BaseException as cleanup:
                self.failure=self.failure or primary or cleanup
                if primary is not None:raise primary from cleanup
                raise
            self.retained_failed_lanes.clear();self.closed=True
        finally:self.statistics["workspace_close_wall_s"]+=time.perf_counter()-begin

    def report(self):
        return {"policy":POLICY,"provenance":self.provenance,"statistics":dict(self.statistics),
            "closed":self.closed,"failure":None if self.failure is None else repr(self.failure),
            "jobs":copy.deepcopy(self.jobs),"active_lane":self.active is not None,
            "retained_failed_lanes":len(self.retained_failed_lanes),"template_started":self.template.started,
            "audit":self.template.audit_nearest,"whole_finish_authority":False,
            "ownership":"Caller retains target cache until workspace.closed. One completed shared stream; compiled code reused, every trajectory graph/buffer/phase private."}
