"""Offline whole-Finish bridge ICP injection with original gates and hard closure.

Only calls inside the captured original ``fragments._verify_bridge`` use the
new device workspace. Original imported _match aliases, proposal order, graph
acceptance and Final fusion remain untouched. Numerical imports are lazy.
"""
from __future__ import annotations
import contextvars
import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import CodeType, FunctionType, MethodType, MappingProxyType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POLICY = "original-full-finish-bridge-subtree-device-loop-workspace-v1"
CONFIGURATION = MappingProxyType({"device": 0, "cuda_graph": True,
    "chunk_iterations": 4, "max_points": 1_000_000,
    "max_scratch_bytes": 256*1024**2, "max_total_bytes": 512*1024**2,
    "pair_cache_bytes": 256*1024**2, "max_jobs": 4096})
GATES = ("_local_match", "_global_seed", "_verify_partial_bridge",
    "_verify_visual_bridge", "_validate_bridge_pose", "_pair", "_strong",
    "_heldout", "_visual_witness", "propose_fragment_poses")
REFINEMENT_GATES = ("_trustworthy", "_distance", "propose_poses")
BUNDLE_GATES = ("propose_bundle_poses", "build_tracks", "_supported", "validate_depth",
    "_verified_matches", "solve_bundle")
BUNDLE_FIELDS = ("poses", "landmarks", "cameras", "tracks", "pixels", "depths")
FILES = ("scripts/research/gpu_icp_finish_scope.py", "tests/test_gpu_icp_finish_scope.py")
HELPER_FILES=("scripts/research/microbatch_bridge_scope.py",
    "scripts/research/microbatch_bridge_driver.py",
    "scripts/research/gpu_icp_device_loop_experiment.py",
    "scripts/research/field_finish_conformance_scope.py",
    "scripts/research/finish_resident_registration.py")


class FinishGPUFailure(BaseException):
    """Research faults cannot become an ordinary rejected proposal."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(ok, message):
    if not ok: raise FinishGPUFailure(message)


def code_state(code):
    return (code.co_code, code.co_names, code.co_varnames, code.co_freevars,
        code.co_cellvars, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_flags,
        tuple(code_state(v) if isinstance(v, CodeType) else v for v in code.co_consts))


def function_state(fn):
    bound = fn.__self__ if isinstance(fn, MethodType) else None
    if bound is not None: fn = fn.__func__
    require(isinstance(fn, FunctionType), "Owned Python callable required")
    return (fn, fn.__code__, fn.__globals__, repr(fn.__defaults__),
        repr(fn.__kwdefaults__), tuple(c.cell_contents for c in fn.__closure__ or ()), bound)


def check_function(record):
    fn, code, namespace, defaults, keywords, cells, _ = record
    actual = tuple(c.cell_contents for c in fn.__closure__ or ())
    require(fn.__code__ is code and fn.__globals__ is namespace
        and repr(fn.__defaults__) == defaults and repr(fn.__kwdefaults__) == keywords
        and len(actual) == len(cells) and all(a is b for a, b in zip(actual, cells)),
        "Loaded callable code/default/closure owner changed")


class WrapperOwnerGuard:
    """Explicit original and installed-wrapper owners; no old guard bypass."""
    def __init__(self, modules):
        self.modules, self.slots, self.pins = [], [], {}
        for module in modules:
            path = Path(module.__file__).resolve()
            self.modules.append((module, path))
            self.pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def add(self, module, name, original, installed):
        if isinstance(original, FunctionType):
            path = Path(sys.modules[original.__module__].__file__).resolve()
            compiled = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
            code = next((c for c in compiled.co_consts if isinstance(c, CodeType)
                and c.co_name == original.__name__), None)
            require(code is not None and code_state(code) == code_state(original.__code__)
                and original.__globals__ is sys.modules[original.__module__].__dict__,
                "Captured original callable differs from its actual source")
            old = function_state(original)
        else: old = None  # Native pybind gate identity is retained separately.
        self.slots.append((module, name, original, installed, old,
            function_state(installed)))

    def check(self, *, source=False):
        for module, path in self.modules:
            require(sys.modules.get(module.__name__) is module
                and Path(module.__file__).resolve() == path, "Original module/path owner changed")
        for module, name, original, installed, old, new in self.slots:
            require(getattr(module, name) is installed, "Installed Finish wrapper owner changed")
            if old is not None: check_function(old)
            check_function(new)
        if source:
            require(all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                for path, digest in self.pins.items()), "Finish helper source bytes changed")


class HelperOwners:
    """Pin each consumed pure helper body and its actual global aliases."""
    def __init__(self, groups):
        self.records=[];self.pins={}
        for module,names in groups:
            path=Path(module.__file__).resolve();self.pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            codes={c.co_name:c for c in compile(path.read_text(encoding="utf-8"),str(path),"exec",dont_inherit=True).co_consts
                if isinstance(c,CodeType)}
            for name in names:
                owner=getattr(module,name);expected=codes[name]
                pairs=[(module,name,owner,expected)] if isinstance(owner,FunctionType) else [
                    (owner,c.co_name,getattr(owner,c.co_name),c) for c in expected.co_consts if isinstance(c,CodeType)]
                for parent,key,fn,code in pairs:
                    require(isinstance(fn,FunctionType) and fn.__globals__ is module.__dict__
                        and code_state(fn.__code__)==code_state(code),"Consumed helper differs from source")
                    aliases=tuple((n,module.__dict__[n]) for n in fn.__code__.co_names if n in module.__dict__)
                    self.records.append((module,path,parent,key,function_state(fn),aliases))

    def check(self,*,source=False):
        for module,path,parent,key,record,aliases in self.records:
            require(sys.modules.get(module.__name__) is module and Path(module.__file__).resolve()==path
                and getattr(parent,key) is record[0] and record[0].__globals__ is module.__dict__
                and all(module.__dict__.get(name) is owner for name,owner in aliases),"Consumed helper/module/global owner changed")
            check_function(record)
        if source:require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.pins.items()),
            "Consumed helper source bytes changed")


def source_contract():
    return {"policy": POLICY, "configuration": dict(CONFIGURATION),
        "artifacts": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES+HELPER_FILES},
        "gpu_scope": "Original complete _verify_bridge subtree only; other Finish and all Live calls use original CPU",
        "original_math": "Unchanged device_loop_icp and original bridge/gate/graph/Final bodies",
        "whole_finish_authority": False}


def default_adapters(np):
    from scripts.research import microbatch_bridge_scope as evidence
    from scripts.research.microbatch_bridge_driver import SharedBridgeCache, all_train_clouds, packet_descriptor
    from scripts.research.gpu_icp_device_loop_experiment import terminal
    from scripts.research import field_finish_conformance_scope as field
    from scripts.research import microbatch_bridge_driver as bridge_driver
    from scripts.research import gpu_icp_device_loop_experiment as experiment
    guard=HelperOwners(((evidence,("descriptor","input_binding","result_evidence","result_shadow","semantic","canonical")),
        (bridge_driver,("SharedBridgeCache","all_train_clouds","packet_descriptor")),
        (experiment,("terminal","descriptor")),
        (field,("graph_snapshot","matrix_evidence","graph_context","final_pose_inventory")),
        (field.original,("evidence",))))
    return SimpleNamespace(helper_guard=guard,capture=lambda a,b,s: evidence.input_binding(np,a,b,s),
        result=lambda r,a,b: evidence.result_evidence(np,r,len(a.points),len(b.points)),
        shadow=lambda native,gpu,a,b: evidence.result_shadow(np,native,gpu,len(a.points),len(b.points)),
        semantic=lambda value: evidence.semantic(np,value),
        terminal=lambda r,l,a,b: terminal(np,r,l.statistics,len(a.points),len(b.points)),
        packet=lambda value: packet_descriptor(np,value),
        pair_clouds=all_train_clouds, cache_factory=SharedBridgeCache,
        graph_snapshot=field.graph_snapshot, graph_context=field.graph_context,
        final_poses=field.final_pose_inventory)


def bundle_problem_record(vectors, cameras, tracks):
    """Bound original solver observations and retain exact constraint IDs."""
    require(type(vectors) is dict and set(vectors)==set(BUNDLE_FIELDS), "Exact original BundleProblem fields required")
    poses=vectors["poses"]
    require(type(poses) is list and 1<=len(poses)<=24, "Bounded original bundle camera poses required")
    def array(item,shape,kinds):
        record=item.get("array") if type(item) is dict else None
        require(type(record) is dict and set(record)=={"dtype","shape","nbytes","sha256"}
            and type(record["dtype"]) is str and len(record["dtype"])>=3 and record["dtype"][1] in kinds
            and record["shape"]==shape and type(record["nbytes"]) is int
            and record["nbytes"]==int(record["dtype"][2:])*math.prod(shape)
            and type(record["sha256"]) is str and len(record["sha256"])==64
            and all(c in "0123456789abcdef" for c in record["sha256"]), "Malformed bounded bundle vector evidence")
        return record
    for pose in poses:require(array(pose,[4,4],"f")["dtype"]=="<f8", "Original bundle pose must be FP64")
    landmarks=vectors["landmarks"]["array"]["shape"]
    require(type(landmarks) is list and len(landmarks)==2 and type(landmarks[0]) is int
        and 0<=landmarks[0]<=800 and landmarks[1]==3, "Bounded original bundle landmarks required")
    require(type(cameras) is list and type(tracks) is list and len(cameras)==len(tracks)<=24*800
        and all(type(c) is int and 0<=c<len(poses) for c in cameras)
        and all(type(t) is int and 0<=t<landmarks[0] for t in tracks), "Original bundle constraint IDs are invalid")
    n=len(cameras)
    for name,shape in (("landmarks",landmarks),("pixels",[n,2]),("depths",[n])):
        require(array(vectors[name],shape,"f")["dtype"]=="<f8", "Original bundle measured vectors must be FP64")
    for name in ("cameras","tracks"):array(vectors[name],[n],"iu")
    return {"bundle_problem":copy.deepcopy(vectors), "constraint_camera_ids":list(cameras),
        "constraint_landmark_ids":list(tracks), "observations":n}


class GpuICPFinishScope:
    def __init__(self, registration_module, fragments, refinement, *, mode="audit",
                 retrieval_factory=None, workspace_factory=None, protocol=None,
                 permit=None, trace=None, boundary=None, numpy=None, engine=None,
                 adapters=None, bundle_module=None):
        require(mode in ("native", "audit", "timing"), "Explicit new Finish mode required")
        require(mode == "native" or callable(retrieval_factory), "Owned retrieval factory required")
        require((mode == "timing") == (permit is not None), "Only new Finish timing consumes an authority")
        if protocol is None:
            from scripts.research import gpu_icp_finish_protocol as protocol
        if workspace_factory is None and mode != "native":
            from scripts.research.device_loop_finish_workspace import DeviceLoopFinishWorkspace
            workspace_factory = DeviceLoopFinishWorkspace
        self.module, self.fragments, self.refinement = registration_module, fragments, refinement
        self.original_dispatch, self.original_match = registration_module.match, refinement._match
        require(fragments._match is self.original_match, "Original imported _match aliases must agree")
        self.original_match_state = function_state(self.original_match)
        self.device_context = registration_module._device
        self.bundle_module=bundle_module
        self.bundle_owner=bundle_module
        self.bundle_problem_type=None if bundle_module is None else bundle_module.BundleProblem
        self.registration_owner = refinement.REG
        require(fragments.REG is self.registration_owner,
            "Original bridge and refinement registration module owners must agree")
        self.mode, self.permit, self.protocol = mode, permit, protocol
        self.retrieval_factory, self.workspace_factory = retrieval_factory, workspace_factory
        self.trace, self.boundary = trace or (lambda row: None), boundary or (lambda: None)
        self.np, self.engine, self.adapters = numpy, engine, adapters
        self.default_evidence = adapters is None
        self.adapters_owner = adapters
        self.configuration = tuple(sorted(CONFIGURATION.items()))
        self.policy_owner = CONFIGURATION
        self.owners = (registration_module, fragments, refinement, mode, permit, protocol,
            retrieval_factory, workspace_factory, self.trace, self.boundary, adapters, numpy, engine)
        self.bypass = contextvars.ContextVar("finish_gpu_original_cpu_bypass", default=False)
        self.context = contextvars.ContextVar("finish_gpu_directed_pair", default=None)
        self.gate_context = contextvars.ContextVar("finish_gpu_original_gates", default=())
        self.retrieval, self.workspace, self.shared, self.pair_key = None, None, None, None
        self.retrieval_owner=self.workspace_owner=self.shared_owner=None
        self.pair_objects, self.pair_inventory = None, None
        self.calls, self.pairs, self.gates, self.graphs, self.events, self.cache_receipts = [], [], [], [], [], []
        self.native_calls, self.successful_builds, self.final_poses = 0, 0, None
        self.failure, self.cleanup_failures, self.patches = None, [], []
        self.complete, self.restored, self.entered, self.closed = False, False, False, False
        self.guard = WrapperOwnerGuard((registration_module, fragments, refinement)+(() if bundle_module is None else (bundle_module,)))
        # These aliases remain untouched. Validate their actual loaded body
        # against source before accepting any observer installation.
        self.guard.add(refinement,"_match",self.original_match,self.original_match)
        self.guard.add(fragments,"_match",self.original_match,self.original_match)
        self.native_owners=tuple((name,getattr(refinement.REG,name)) for name in
            ("registration_icp","TransformationEstimationPointToPlane","HuberLoss","ICPConvergenceCriteria")
            if hasattr(refinement.REG,name))
        self.scope_path = Path(__file__).resolve()
        self.scope_digest = hashlib.sha256(self.scope_path.read_bytes()).hexdigest()
        module = sys.modules[__name__]
        compiled = compile(self.scope_path.read_text(encoding="utf-8"),str(self.scope_path),"exec",dont_inherit=True)
        class_code = next(c for c in compiled.co_consts if isinstance(c,CodeType) and c.co_name==type(self).__name__)
        self.loaded_methods=[]
        for code in class_code.co_consts:
            if isinstance(code,CodeType):
                fn=getattr(type(self),code.co_name)
                require(fn.__globals__ is module.__dict__ and code_state(fn.__code__)==code_state(code),
                    "Loaded Finish scope differs from source")
                self.loaded_methods.append((code.co_name,function_state(fn)))
        self.loaded_functions=[]
        for code in compiled.co_consts:
            if isinstance(code,CodeType) and isinstance(getattr(module,code.co_name,None),FunctionType):
                fn=getattr(module,code.co_name)
                require(fn.__globals__ is module.__dict__ and code_state(fn.__code__)==code_state(code),
                    "Loaded scope helper differs from source")
                self.loaded_functions.append((code.co_name,function_state(fn)))
        self.loaded_classes=[]
        for code in compiled.co_consts:
            if isinstance(code,CodeType) and isinstance(getattr(module,code.co_name,None),type):
                cls=getattr(module,code.co_name)
                for method in code.co_consts:
                    if isinstance(method,CodeType):
                        fn=getattr(cls,method.co_name)
                        require(fn.__globals__ is module.__dict__ and code_state(fn.__code__)==code_state(method),
                            "Loaded scope guard class differs from source")
                        self.loaded_classes.append((code.co_name,cls,method.co_name,function_state(fn)))
        self.adapter_states=None
        self.factory_methods=[]
        for factory in (retrieval_factory,workspace_factory):
            if isinstance(factory,type):
                for name,fn in factory.__dict__.items():
                    if isinstance(fn,FunctionType):self.factory_methods.append((factory,name,function_state(fn)))
        self.clock = {"workspace_cold_setup_wall_s": 0., "pair_setup_wall_s": 0.,
            "pair_cleanup_wall_s": 0., "scope_cleanup_wall_s": 0.}

    def fail(self, error):
        if self.failure is None:
            self.failure = error if isinstance(error, FinishGPUFailure) else FinishGPUFailure(
                "Whole-Finish GPU research failed: "+repr(error))
            if self.failure is not error: self.failure.__cause__ = error
        raise self.failure

    def healthy(self):
        if self.failure is not None: raise self.failure
        try:self._check_owners()
        except BaseException as error:self.fail(error)

    def _check_owners(self):
        require(CONFIGURATION is self.policy_owner and tuple(sorted(CONFIGURATION.items())) == self.configuration,
            "Finish configuration owner/values changed")
        require(self.bundle_module is self.bundle_owner and (self.bundle_module is None
            or self.bundle_module.BundleProblem is self.bundle_problem_type), "Original bundle module/problem class owner changed")
        require(all(value is owner for value, owner in zip((self.module,self.fragments,self.refinement,
            self.mode,self.permit,self.protocol,self.retrieval_factory,self.workspace_factory,self.trace,
            self.boundary,self.owners[10],self.np,self.engine),self.owners))
            and self.adapters is self.adapters_owner,"Finish execution owners changed")
        for name,record in self.loaded_methods:
            require(getattr(type(self),name) is record[0],"Loaded Finish method owner changed")
            check_function(record)
        for name,record in self.loaded_functions:
            require(globals().get(name) is record[0],"Loaded scope helper owner changed")
            check_function(record)
        for name,cls,method,record in self.loaded_classes:
            require(globals().get(name) is cls and getattr(cls,method) is record[0],
                "Loaded scope guard class/method owner changed")
            check_function(record)
        if self.adapter_states is not None:
            for name,owner,state in self.adapter_states:
                require(getattr(self.adapters,name) is owner,"Evidence adapter owner changed")
                if state is not None:check_function(state)
            helper=getattr(self.adapters,"helper_guard",None)
            if helper is not None:helper.check()
        for factory,name,record in self.factory_methods:
            require(getattr(factory,name) is record[0],"Owned factory method changed")
            check_function(record)
        require(self.workspace is self.workspace_owner and self.retrieval is self.retrieval_owner
            and self.shared is self.shared_owner,"Workspace/retrieval/pair cache owner changed")
        require(self.fragments._match is self.original_match and self.refinement._match is self.original_match
            and self.fragments.REG is self.registration_owner and self.refinement.REG is self.registration_owner
            and self.module._device is self.device_context and self.device_context.get() is None
            and os.environ.get("KINECT_CUDA_REGISTRATION") == "cpu", "Original CPU dispatch selection/aliases changed")
        check_function(self.original_match_state)
        self.guard.check()
        require(all(getattr(self.refinement.REG,name) is owner for name,owner in self.native_owners),
            "Original CPU estimator/registration callable owner changed")
        self.boundary()

    def _evidence(self):
        if self.adapters is None:
            if self.np is None:
                import numpy as np
                self.np = np
                self.owners = self.owners[:11]+(np,)+self.owners[12:]
            self.adapters = default_adapters(self.np)
            self.adapters_owner = self.adapters
        if self.adapter_states is None:
            self.adapter_states=tuple((name,value,function_state(value) if isinstance(value,(FunctionType,MethodType)) else None)
                for name,value in vars(self.adapters).items())
        return self.adapters

    def _emit(self, row):
        if self.mode == "timing" and row.get("complete"):
            checker = getattr(self.protocol, "validate_finish_event", None)
            require(callable(checker), "New timing must validate every original gate/graph event")
            checker(self.permit, row)
        self.trace(row)
        self.events.append({"event":row["event"], "index":row.get("call_index",
            row.get("gate_index", row.get("graph_index")))})

    def _cpu(self, source, target, initial):
        token = self.bypass.set(True)
        try: return self.original_match(source, target, initial)
        finally: self.bypass.reset(token)

    def _install(self, module, name, value):
        original = getattr(module, name)
        self.guard.add(module, name, original, value)
        self.patches.append((module, name, original, value))
        setattr(module, name, value)

    def _setup(self):
        if self.workspace is not None: return
        begin = time.perf_counter()
        self.retrieval = self.retrieval_factory()
        self.retrieval_owner=self.retrieval
        # Retain the partly constructed owner if __init__ raises.
        self.workspace = object.__new__(self.workspace_factory)
        self.workspace_owner=self.workspace
        self.workspace.__init__(self.retrieval, device=0, cuda_graph=True,
            max_points=CONFIGURATION["max_points"], max_scratch_bytes=CONFIGURATION["max_scratch_bytes"],
            max_total_bytes=CONFIGURATION["max_total_bytes"], max_jobs=CONFIGURATION["max_jobs"],
            audit=self.mode == "audit", timing_permit=self.permit)
        self.clock["workspace_cold_setup_wall_s"] += time.perf_counter()-begin

    def _close_pair(self):
        if self.shared is None: return
        begin = time.perf_counter()
        require(self.workspace is not None and self.workspace.active is None,
            "Pair transition cannot release an active lane")
        template = self.workspace.template
        with template.cp.cuda.Device(template.device), template.stream:
            template.stream.synchronize()
        # The original cache closes its retrieval grids, not the reusable template.
        self.shared.close()
        require(self.shared.closed is True, "Pair cache selected completion not proved")
        self.cache_receipts.append(self.shared.report())
        self.shared = None
        self.shared_owner=None
        self.clock["pair_cleanup_wall_s"] += time.perf_counter()-begin

    def _pair(self, source, target):
        key = (int(source.index), int(target.index))
        require(key[0] != key[1] and min(key) >= 0, "Directed original fragment IDs required")
        if key != self.pair_key:
            self._close_pair()
            self.pair_key, self.pair_objects = key, (source,target)
            self.pair_inventory = None
            self.pairs.append({"pair_index":len(self.pairs), "pair":list(key), "proposals":[]})
        require(self.pair_objects[0] is source and self.pair_objects[1] is target,
            "Same directed pair replaced its original fragment owners")
        return self.pairs[-1]

    def _shared(self, source, target):
        self._setup()
        clouds = self._evidence().pair_clouds(*self.pair_objects)
        current = [self._evidence().capture(c,c,self.np.eye(4)) for c in clouds]
        if self.pair_inventory is not None:
            require(current == self.pair_inventory, "Pair immutable cloud inventory changed between proposals")
        else: self.pair_inventory = current
        if self.shared is None:
            begin = time.perf_counter()
            factory = self._evidence().cache_factory
            self.shared = object.__new__(factory)
            self.shared_owner=self.shared
            self.shared.__init__(self.retrieval, clouds)
            require(self.shared.owned_bytes <= CONFIGURATION["pair_cache_bytes"], "Pair cache bound exceeded")
            self.clock["pair_setup_wall_s"] += time.perf_counter()-begin
        return self.shared

    def dispatch(self, source, target, initial):
        if self.bypass.get(): return None
        self.healthy()
        active = self.context.get()
        if active is None:
            self.native_calls += 1
            return self.original_dispatch(source,target,initial)  # original legacy None route
        index = len(self.calls)
        row = {"event":"match", "call_index":index, "complete":False,
            "query_trace":[], "cleanup_failures":[]}
        self.calls.append(row)
        loop, lease, primary, result = None, None, None, None
        begin = time.perf_counter()
        try:
            evidence = self._evidence()
            before = evidence.capture(source,target,initial)
            payload = {"pair_index":active["pair_index"], "pair":active["pair"],
                "proposal_index":active["proposal_index"], "caller":sys._getframe(1).f_code.co_name,
                "gate_context":list(self.gate_context.get()), "input_binding":before}
            row.update(payload=payload,input_binding=before)
            if self.mode == "native":
                result = self._cpu(source,target,initial)
            else:
                shared = self._shared(source,target)
                if self.mode == "timing":
                    require(self.protocol.next_workspace_job(self.permit,payload) == index,
                        "New Finish ordered workspace index differs")
                def observe(query):
                    packet = query["packet"]; order = self.np.argsort(packet[:,0])
                    row["query_trace"].append({"query_index":query["query_index"], "stage":query["stage"],
                        "radius":query["radius"], "target_sha256":query["target_sha256"],
                        "packet":evidence.packet(packet[order]),
                        "corrected_ids_sha256":hashlib.sha256(query["corrected_ids"][order].tobytes()).hexdigest(),
                        "counters":list(query["counters"])})
                loop = self.workspace.new_lane(audit_observer=observe if self.mode == "audit" else None)
                lease = shared.lease(source,target)
                result = loop.match(source,target,initial,chunk_iterations=4,pair_lease=lease)
                self.workspace.check_lane(loop)
                row.update(source_binding=copy.deepcopy(loop.provenance),consumed_input_binding=copy.deepcopy(loop.input_binding),
                    terminal=evidence.terminal(result,loop,source,target))
                if self.mode == "audit":
                    shadow_begin = time.perf_counter()
                    native = self._cpu(source,target,initial)
                    row["original_cpu_shadow_wall_s"] = time.perf_counter()-shadow_begin
                    row["native_shadow"] = evidence.shadow(native,result,source,target)
                    require(row["native_shadow"].get("passed") is True, "Actual original CPU result shadow failed")
                else: row["native_shadow"] = {"collected":False}
            row["result"] = evidence.result(result,source,target)
            require(evidence.capture(source,target,initial) == before, "Original call inputs/unrounded seed changed")
            row["input_bytes_unchanged"] = True
        except BaseException as error:
            primary = error
            row["failure"] = {"type":type(error).__name__, "message":str(error)}
        finally:
            if loop is not None:
                try:
                    self.workspace.close_lane(loop,primary=primary)
                    require(loop.closed is True, "Lane selected completion is unproved")
                    if lease is not None: self.shared.release_lease(lease)
                except BaseException as cleanup:
                    row["cleanup_failures"].append(repr(cleanup))
                    if primary is None: primary = cleanup
                try: row["loop_report"] = loop.report()
                except BaseException as cleanup:
                    row["cleanup_failures"].append(repr(cleanup))
                    if primary is None: primary = cleanup
            row["whole_call_wall_s_inclusive"] = time.perf_counter()-begin
        try:
            if primary is not None: self.fail(primary)
            if self.mode == "timing":
                self.protocol.validate_finish_terminal(self.permit,index,payload,row["terminal"],row["loop_report"])
            row["query_trace_sha256"] = hashlib.sha256(canonical(row["query_trace"]).encode()).hexdigest()
            row["complete"] = True
            self.healthy(); self._emit(row)
            return result  # The actual GPU/native result reaches unchanged gates.
        except BaseException as error:
            row["complete"] = False
            row.setdefault("failure", {"type":type(error).__name__, "message":str(error)})
            try: self.trace(row)
            except BaseException as secondary: self.cleanup_failures.append("Failed call trace: "+repr(secondary))
            self.fail(error)

    def bridge(self, source, target, initial, *args, **kwargs):
        self.healthy()
        require(self.context.get() is None, "Nested outer fragment verification is unsupported")
        gate={"event":"gate","gate_index":len(self.gates),"function":"fragments._verify_bridge","complete":False}
        self.gates.append(gate)
        gate_token=self.gate_context.set(self.gate_context.get()+({"function":gate["function"],"invocation":gate["gate_index"]},))
        try:
            pair = self._pair(source,target)
            row = {"proposal_index":len(pair["proposals"]), "complete":False}
            pair["proposals"].append(row)
            token = self.context.set({"pair_index":pair["pair_index"],"pair":pair["pair"],
                "proposal_index":row["proposal_index"]})
            try:
                result = self.original_bridge(source,target,initial,*args,**kwargs)
                self.healthy()
                row.update(result=self._evidence().semantic(result),complete=True)
                gate.update(result=copy.deepcopy(row["result"]),complete=True)
                self._emit(gate)
                return result
            finally: self.context.reset(token)
        except BaseException as error:
            gate["failure"]={"type":type(error).__name__,"message":str(error)}
            self.fail(error)
        finally:self.gate_context.reset(gate_token)

    def bundle_problem(self,problem):
        require(self.bundle_module is not None and type(problem) is self.bundle_problem_type
            and dataclasses.is_dataclass(problem)
            and tuple(f.name for f in dataclasses.fields(problem))==BUNDLE_FIELDS,
            "Original bounded BundleProblem dataclass required")
        evidence=self._evidence()
        vectors={name:evidence.semantic(getattr(problem,name)) for name in BUNDLE_FIELDS}
        return bundle_problem_record(vectors,self.np.asarray(problem.cameras).tolist(),
            self.np.asarray(problem.tracks).tolist())

    def observe(self,module,name,label=None,*,capture_problem=False,result_problem=False):
        original = getattr(module,name); label = label or module.__name__+"."+name
        def observed(*args,**kwargs):
            self.healthy()
            row = {"event":"gate","gate_index":len(self.gates),"function":label,"complete":False}
            self.gates.append(row)
            token = self.gate_context.set(self.gate_context.get()+({"function":label,"invocation":row["gate_index"]},))
            original_error=None
            try:
                if capture_problem:
                    problem=args[0] if args else kwargs["problem"]
                    row["consumed_problem"]=self.bundle_problem(problem)
                try:value = original(*args,**kwargs)
                except BaseException as error:
                    original_error=error
                    raise
                self.healthy()
                if capture_problem:require(self.bundle_problem(problem)==row["consumed_problem"],
                    "Original bundle solver/support mutated consumed constraint inputs")
                row.update(result=self.bundle_problem(value) if result_problem else self._evidence().semantic(value),complete=True)
                self._emit(row)
                return value
            except BaseException as error:
                if error is original_error and module is self.bundle_module and name!="propose_bundle_poses" and (
                    isinstance(error,ImportError) or isinstance(error,getattr(module,"_BudgetExceeded",()))):
                    # The original proposal body handles these two legitimate
                    # no-op routes. Keep its exception and budget semantics.
                    try:
                        self.healthy()
                        if capture_problem:require(self.bundle_problem(problem)==row["consumed_problem"],
                            "Original bundle exception changed consumed constraint inputs")
                        row.update(result={"original_exception":{"type":type(error).__name__,"message":str(error)}},complete=True)
                        self._emit(row)
                    except BaseException as fault:self.fail(fault)
                    raise
                row["failure"] = {"type":type(error).__name__,"message":str(error)}
                self.fail(error)
            finally:self.gate_context.reset(token)
        self._install(module,name,observed)

    def observe_graph(self):
        module = self.refinement.REG; original = module.global_optimization
        def optimizer(*args,**kwargs):
            self.healthy()
            row = {"event":"graph_optimization","graph_index":len(self.graphs),
                "gate_context":list(self.gate_context.get()),"complete":False}
            self.graphs.append(row)
            try:
                require(len(args)==4 and not kwargs,"Original four-argument graph call required")
                graph,_,_,options=args
                row.update(context=self._evidence().graph_context(),
                    options={"max_correspondence_distance":float(options.max_correspondence_distance),
                        "edge_prune_threshold":float(options.edge_prune_threshold),"reference_node":int(options.reference_node)},
                    before=self._evidence().graph_snapshot(graph))
                require(row["options"]["max_correspondence_distance"]==.03
                    and row["options"]["edge_prune_threshold"]==.25
                    and 0<=row["options"]["reference_node"]<len(row["before"]["nodes"]),
                    "Original graph optimization settings changed")
                value=original(*args,**kwargs)
                self.healthy();row.update(after=self._evidence().graph_snapshot(graph),complete=True)
                self._emit(row);return value
            except BaseException as error:self.fail(error)
        self._install(module,"global_optimization",optimizer)

    def __enter__(self):
        require(not self.entered,"Finish scope cannot be reused")
        self.entered=True
        try:
            self.original_bridge=self.fragments._verify_bridge
            self._install(self.module,"match",self.dispatch)
            self._install(self.fragments,"_verify_bridge",self.bridge)
            for name in GATES:
                if hasattr(self.fragments,name):self.observe(self.fragments,name,"fragments."+name)
            for name in REFINEMENT_GATES:
                if hasattr(self.refinement,name):self.observe(self.refinement,name,"refinement."+name)
            if self.bundle_module is not None:
                for name in BUNDLE_GATES:
                    self.observe(self.bundle_module,name,"bundle_adjustment."+name,
                        capture_problem=name in ("_supported","solve_bundle"))
                self.observe(self.bundle_module,"make_problem","bundle_adjustment.make_problem",result_problem=True)
            self.observe(self.refinement.REG,"get_information_matrix_from_point_clouds","original_information_matrix")
            self.observe_graph()
            self.healthy();self.guard.check(source=True)
            return self
        except BaseException as error:
            self.failure=error
            self.__exit__(type(error),error,error.__traceback__)

    def build(self,engine,original_build,*args,**kwargs):
        self.healthy()
        try:
            require(getattr(engine,"unprocessed_count",0)==0,"Complete original Live before Finish scope")
            require(not getattr(getattr(engine,"settings",None),"bundle_adjustment",False)
                or self.bundle_module is not None,"Enabled original bundle stage requires complete output/problem observers")
            if isinstance(original_build,MethodType):
                require(original_build.__self__ is engine,"Original bound build must own this engine")
                value=original_build(*args,**kwargs)
            else:value=original_build(engine,*args,**kwargs)
            self.healthy()
            require(type(value) is tuple and len(value)==2 and value[0] is True,
                "Original full transactional Final mesh must succeed")
            require(self.successful_builds==0,"Exactly one successful original build required")
            self.final_poses=self._evidence().final_poses(engine);self.successful_builds=1
            return value
        except BaseException as error:self.fail(error)

    def finish(self):
        self.healthy()
        require(self.successful_builds==1 and self.final_poses is not None,"Original final build/pose inventory absent")
        require(all(row["complete"] for row in self.calls+self.gates+self.graphs)
            and all(p["complete"] for pair in self.pairs for p in pair["proposals"]),"Incomplete actual call/gate/proposal suffix")
        self.guard.check(source=True)
        helper=getattr(self.adapters,"helper_guard",None)
        if helper is not None:helper.check(source=True)
        require(hashlib.sha256(self.scope_path.read_bytes()).hexdigest()==self.scope_digest,
            "Finish scope source bytes changed")
        self.complete=True

    def __exit__(self,kind,error,traceback):
        begin=time.perf_counter()
        primary=self.failure or error
        if primary is None:
            try:
                self.healthy();self.guard.check(source=True)
                helper=getattr(self.adapters,"helper_guard",None)
                if helper is not None:helper.check(source=True)
            except BaseException as failure:primary=self.failure or failure
        # Cleanup uses the captured resource owners even if a public slot was
        # replaced after the last job. Never release a different owner's data.
        self.workspace,self.shared,self.retrieval=self.workspace_owner,self.shared_owner,self.retrieval_owner
        for module,name,original,installed in reversed(self.patches):
            try:
                if getattr(module,name) is not installed:
                    self.cleanup_failures.append("Restore "+name+": wrapper replaced before restoration")
                setattr(module,name,original)
            except BaseException as cleanup:self.cleanup_failures.append("Restore "+name+": "+repr(cleanup))
        self.restored=not self.cleanup_failures
        if self.workspace is not None:
            try:self.workspace.close(primary=primary)
            except BaseException as cleanup:self.cleanup_failures.append("Workspace close: "+repr(cleanup))
        if self.shared is not None:
            if self.workspace is not None and getattr(self.workspace,"closed",False):
                try:
                    self.shared.close();require(self.shared.closed,"Final pair cache close incomplete")
                    self.cache_receipts.append(self.shared.report());self.shared=self.shared_owner=None
                except BaseException as cleanup:self.cleanup_failures.append("Pair cache close: "+repr(cleanup))
            else:self.cleanup_failures.append("Retained pair owners: workspace completion unproved")
        elif self.retrieval is not None and (self.workspace is None or getattr(self.workspace,"closed",False)):
            try:self.retrieval.close()
            except BaseException as cleanup:self.cleanup_failures.append("Retrieval close: "+repr(cleanup))
        self.closed=(self.workspace is None or getattr(self.workspace,"closed",False)) and self.shared is None
        self.clock["scope_cleanup_wall_s"]+=time.perf_counter()-begin
        if primary is None and (self.cleanup_failures or not self.complete):
            primary=FinishGPUFailure("Finish cleanup/suffix incomplete: "+repr(self.cleanup_failures))
        if primary is not None:
            self.failure=primary
            note=getattr(primary,"add_note",None)
            if callable(note):
                for message in self.cleanup_failures:note(message)
            raise primary
        if self.mode=="timing":
            try:self.protocol.validate_complete_finish(self.permit,self.report())
            except BaseException as error:self.fail(error)
        return False

    def report(self):
        workspace_report=None
        if self.workspace is not None:
            try:workspace_report=self.workspace.report()
            except BaseException as error:workspace_report={"closed":getattr(self.workspace,"closed",False),
                "failure":repr(self.failure),"partial_report_failure":repr(error)}
        return {"policy":POLICY,"mode":self.mode,"complete":self.complete and self.failure is None,
            "restored":self.restored,"closed":self.closed,"default_evidence":self.default_evidence,
            "failure":None if self.failure is None else repr(self.failure),"cleanup_failures":list(self.cleanup_failures),
            "configuration":dict(CONFIGURATION),"calls":copy.deepcopy(self.calls),"pairs":copy.deepcopy(self.pairs),
            "gates":copy.deepcopy(self.gates),"graphs":copy.deepcopy(self.graphs),"events":copy.deepcopy(self.events),
            "native_non_bridge_dispatch_calls":self.native_calls,"cache_receipts":copy.deepcopy(self.cache_receipts),
            "workspace":workspace_report,
            "successful_builds":self.successful_builds,"final_pose_inventory":copy.deepcopy(self.final_poses),
            "statistics":dict(self.clock),"whole_finish_authority":False,
            "ownership":"Pair grids/normals/leases retained until selected workspace and cache streams complete; never retry a failed trajectory."}
