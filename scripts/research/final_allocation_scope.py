"""Offline exact Final-capacity experiment, separate from frozen Finish proofs.

Importing this prototype uses no numerical runtime. It delegates the unchanged
Final body and allocator after a private exact frustum-union plan. A smaller
hashmap can change enumeration/surface ordering: new mesh quality evidence is
required; this is neither bit-exact authority nor a production selection.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import ast
import copy
from dataclasses import dataclass,asdict,replace
from functools import lru_cache
import hashlib
import math
import time
from types import FunctionType,MethodType,SimpleNamespace

KIND = "offline-exact-final-allocation-v1"
ATTRIBUTE_BYTES_PER_BLOCK = 16**3 * 20
ORIGINAL_AST = {
    "_create_vbg":"75e538cd8a573d7c410aadc1874080e583a6249735ea196c0d28ec683708a0f6",
    "_required_fusion_blocks":"ede657e4586a2bb2a1dc1352034e3db728346998269cd5dc531e304a65cbd58b",
    "_final_volume":"c4172cdb6a182eddbee68143e6dd42099fb11d72a18deea08890164136ef48a4"}
PRIVATE_MODULES = {"shared.calibration","shared.depth","shared.native"}


class FinalAllocationContractError(RuntimeError):
    pass


class FinalCapacityError(ValueError):
    def __init__(self,plan):
        self.plan = plan
        super().__init__(f"Final {plan.final_voxel_m:g} m model needs {plan.required_blocks} blocks; "
            f"increase final_block_count from {plan.configured_limit} to at least {plan.required_blocks}, "
            "or choose a coarser Final voxel. No Final candidate was allocated.")


@dataclass(frozen=True)
class FinalAllocationPlan:
    required_blocks: int
    configured_limit: int
    allocated_blocks: int
    final_voxel_m: float
    sdf_trunc_m: float
    accepted_views: int
    planning_wall_s: float
    capacity_fits: bool

    @property
    def report(self):
        return dict(asdict(self),kind=KIND,
            allocated_attribute_bytes=self.allocated_blocks*ATTRIBUTE_BYTES_PER_BLOCK if self.capacity_fits else 0,
            configured_attribute_bytes=self.configured_limit*ATTRIBUTE_BYTES_PER_BLOCK,
            scope="Attribute floor only: original TSDF/weight/color float32,16^3 voxels; excludes hashmap, Live volume, images, native scratch/global allocator reservations. No geometry authority.")


def check(condition,message):
    if not condition:
        raise FinalAllocationContractError(message)


def code_signature(code):
    """Compare loaded implementation with compiled source, ignoring locations."""
    def constant(value):
        if hasattr(value,"co_code"):
            return code_signature(value)
        if isinstance(value,tuple):
            return [constant(x) for x in value]
        if isinstance(value,float):
            return ("float",value.hex())
        return (type(value).__name__,repr(value))
    return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,
        code.co_argcount,code.co_kwonlyargcount,code.co_posonlyargcount,code.co_flags,
        tuple(constant(x) for x in code.co_consts))


def validate_original_functions(engine,original_final):
    path = ROOT/"scanner_server/engine.py"
    text = path.read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name == "ScanEngine")
    compiled = compile(text,str(path),"exec",dont_inherit=True)
    class_code = next(c for c in compiled.co_consts if hasattr(c,"co_name") and c.co_name == "ScanEngine")
    loaded = {"_final_volume":original_final,"_create_vbg":engine._create_vbg.__func__,
        "_required_fusion_blocks":engine._required_fusion_blocks.__func__}
    result = {}
    for name,expected in ORIGINAL_AST.items():
        node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == name)
        actual = hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        check(actual == expected,"Original Final/planner/allocator source changed; requires a new experimental scope")
        function = loaded[name]
        expected_code = next(c for c in class_code.co_consts if hasattr(c,"co_name") and c.co_name == name)
        check(function.__module__ == "scanner_server.engine" and function.__closure__ is None
            and code_signature(function.__code__) == code_signature(expected_code),
            "Loaded Final/planner/allocator differs from pinned original source")
        result[name] = actual
    return result


def private_depth_preparation(function):
    """Same function code, private Python globals/LRU caches, no module loading.

    Camera maps/LUTs are recomputed in the private cache, never inserted into
    shared calibration caches. Existing native kernels retain their original
    policy/implementation; a planner may not import a new extension globally.
    Native scratch/pool reservations remain measured runtime effects.
    """
    namespaces,functions = {},{}

    def loaded_module(name):
        if name != "_kinect_native" or name not in sys.modules:
            raise ImportError("Private planner will not load a native module")
        return sys.modules[name]

    def clone(value):
        if id(value) in functions:
            return functions[id(value)]
        raw = getattr(value,"__wrapped__",value)
        check(isinstance(raw,FunctionType) and raw.__module__ in PRIVATE_MODULES and raw.__closure__ is None,
              "Unknown depth-preparation closure cannot be privately planned")
        key = id(raw.__globals__)
        first = key not in namespaces
        if first:
            namespaces[key] = dict(raw.__globals__)
        local = namespaces[key]
        result = FunctionType(raw.__code__,local,raw.__name__,raw.__defaults__)
        result.__kwdefaults__ = raw.__kwdefaults__
        if hasattr(value,"cache_parameters"):
            result = lru_cache(**value.cache_parameters())(result)
        functions[id(value)] = result
        if first:
            if raw.__module__ == "shared.native":
                local["importlib"] = SimpleNamespace(import_module=loaded_module)
            for name,item in list(local.items()):
                item_raw = getattr(item,"__wrapped__",item)
                if isinstance(item_raw,FunctionType) and item_raw.__module__ in PRIVATE_MODULES:
                    local[name] = clone(item)
        return result

    return clone(function)


def private_containers(value,memo=None):
    """Detach Python container state, retaining read-only native/array owners."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value,dict):
        result = {}
        memo[id(value)] = result
        result.update((key,private_containers(item,memo)) for key,item in value.items())
        return result
    if isinstance(value,list):
        result = []
        memo[id(value)] = result
        result.extend(private_containers(item,memo) for item in value)
        return result
    if isinstance(value,tuple):
        return tuple(private_containers(item,memo) for item in value)
    if isinstance(value,set):
        return set(value)
    return value


def plan_final(engine,*,original_final):
    validate_original_functions(engine,original_final)
    check(engine.settings.final_voxel_m is not None and engine._final_vbg is None,
          "Plan only a new explicit Final resolution, not a cached or Live volume")
    limit = engine.settings.final_block_count
    voxel = engine.settings.final_voxel_m
    check(type(limit) is int and 1 <= limit <= 50000 and isinstance(voxel,(int,float))
          and not isinstance(voxel,bool) and math.isfinite(voxel) and voxel > 0,"Invalid Final capacity/voxel policy")
    started = time.perf_counter()
    planner = copy.copy(engine)
    planner.__dict__ = private_containers(engine.__dict__)
    planner.settings = replace(engine.settings,voxel_m=voxel)
    planner.voxel_size = voxel
    check(planner.sdf_trunc == engine.sdf_trunc,"Final allocation planning changed original SDF truncation")
    method = engine._required_fusion_blocks.__func__
    globals_copy = dict(method.__globals__)
    globals_copy["prepare_metric_depth"] = private_depth_preparation(globals_copy["prepare_metric_depth"])
    private_method = FunctionType(method.__code__,globals_copy,method.__name__,method.__defaults__)
    private_method.__kwdefaults__ = method.__kwdefaults__
    # Pose matrices are tiny; copy them without rounding. Recorded image arrays
    # stay original read-only inputs to the source-bound depth implementation.
    poses = [(index,pose.copy()) for index,pose in engine.poses]
    required = private_method(planner,poses,None,stage="final_capacity_preflight")
    check(type(required) is int and required >= 0,"Planner returned malformed exact union size")
    fits = required <= limit
    return FinalAllocationPlan(required,limit,max(1,required) if fits else 0,float(voxel),
        float(engine.sdf_trunc),len(poses),time.perf_counter()-started,fits)


class RightSizedFinalScope:
    """Instance-only reversible wrapper; build_mesh retains commit authority."""
    def __init__(self,engine,*,original_final,on_plan=None):
        self.engine,self.original_final,self.on_plan = engine,original_final,on_plan
        self.plans,self.allocations,self.cleanup_failures = [],[],[]
        self.entered,self.restored = False,False

    def final(self,owner,progress_cb=None):
        check(owner is self.engine,"Final allocator scope received another engine")
        if owner.settings.final_voxel_m is None or owner._final_vbg is not None:
            return self.original_final(owner,progress_cb)
        plan = plan_final(owner,original_final=self.original_final)
        self.plans.append(plan)
        if self.on_plan is not None:
            self.on_plan(plan.report)
        if not plan.capacity_fits:
            raise FinalCapacityError(plan)
        original_create = owner._create_vbg.__func__
        allocations = self.allocations

        class PrivateFinal(type(owner)):
            def _create_vbg(candidate,block_count=None):
                check(block_count == plan.configured_limit and len(allocations) == allocation_start,
                      "Original Final allocation count/cap changed unexpectedly")
                value = original_create(candidate,block_count=plan.allocated_blocks)
                actual_capacity = int(value.hashmap().capacity())
                allocations.append({"requested_blocks":block_count,"allocated_blocks":plan.allocated_blocks,
                    "actual_capacity":actual_capacity})
                check(actual_capacity == plan.allocated_blocks,
                      "Final volume capacity differs from the exact planned allocation")
                return value

        allocation_start = len(allocations)
        proxy = copy.copy(owner)
        proxy.__class__ = PrivateFinal
        # Actual integration intentionally retains original helper/backend/stage
        # sharing and fault behavior. Only the earlier planner was isolated.
        result = self.original_final(proxy,progress_cb)
        check(len(allocations) == allocation_start+1,"Original Final omitted its planned allocation")
        allocations[-1]["final_capacity"] = int(result.hashmap().capacity())
        check(allocations[-1]["final_capacity"] == plan.allocated_blocks,
              "Original Final resized beyond the planned capacity; allocation experiment failed")
        owner.final_reconstruction = proxy.final_reconstruction
        owner.final_reconstruction.update(required_blocks=plan.required_blocks,
            allocated_blocks=plan.allocated_blocks,configured_block_limit=plan.configured_limit,
            attribute_budget_mib=plan.allocated_blocks*ATTRIBUTE_BYTES_PER_BLOCK/2**20,
            configured_attribute_budget_mib=plan.configured_limit*ATTRIBUTE_BYTES_PER_BLOCK/2**20,
            planning_wall_s=plan.planning_wall_s,allocator_scope=KIND)
        return result

    def __enter__(self):
        check(not self.entered,"A Final allocation scope cannot be reused")
        validate_original_functions(self.engine,self.original_final)
        self.entered = True
        self.had_instance = "_final_volume" in self.engine.__dict__
        self.previous = self.engine.__dict__.get("_final_volume")
        self.wrapper = MethodType(self.final,self.engine)
        self.engine._final_volume = self.wrapper
        return self

    def __exit__(self,kind,error,traceback):
        try:
            if self.had_instance:
                self.engine._final_volume = self.previous
            else:
                del self.engine.__dict__["_final_volume"]
            self.restored = True
        except BaseException as secondary:
            self.cleanup_failures.append(str(secondary))
            if error is not None:
                error.add_note(f"Final allocation scope restore also failed: {secondary}")
                raise error
            raise
        return False

    def report(self):
        return {"kind":KIND,"restored":self.restored,"plans":[plan.report for plan in self.plans],
            "allocations":list(self.allocations),"cleanup_failures":list(self.cleanup_failures),
            "actual_allocated_attribute_bytes":[row.get("final_capacity",row["actual_capacity"])*ATTRIBUTE_BYTES_PER_BLOCK
                                                for row in self.allocations],
            "performance_measured":False,"geometry_proven":False,
            "note":"Source prototype only; new independently bound native/mesh experiment required. Existing v1/v2 timing authority cannot authorize this allocator."}


class ObservedOriginalFinalScope(RightSizedFinalScope):
    """Matched planner overhead, then unchanged original allocation/error path.

    The exact union is evidence only in this control. Even an overflowing plan
    delegates the original body, which retains its original partial-fusion
    capacity check and failure semantics.
    """
    def final(self,owner,progress_cb=None):
        check(owner is self.engine,"Final allocator scope received another engine")
        if owner.settings.final_voxel_m is None or owner._final_vbg is not None:
            return self.original_final(owner,progress_cb)
        plan = plan_final(owner,original_final=self.original_final)
        self.plans.append(plan)
        if self.on_plan is not None:
            self.on_plan(plan.report)
        result = self.original_final(owner,progress_cb)
        self.allocations.append({"requested_blocks":plan.configured_limit,
            "allocated_blocks":int(result.hashmap().capacity()),
            "actual_capacity":int(result.hashmap().capacity()),"final_capacity":int(result.hashmap().capacity())})
        return result

    def report(self):
        result = super().report()
        result["control"] = "Matched isolated planner observation; unchanged original Final allocation and error path"
        return result
