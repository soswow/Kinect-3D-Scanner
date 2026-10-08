"""New reserve-headroom Final experiment; original activation/math unchanged.

The measured v1 exact-unique allocation grew automatically on C5. This separate
policy plans both exact unique blocks and the pessimistic incoming-row reserve
bound used by Open3D HashMap::Activate. No numerical imports occur here.
"""

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from types import FunctionType, MethodType

from scripts.research import final_allocation_scope as original

ROOT = Path(__file__).resolve().parents[2]
KIND = "offline-original-final-reserve-headroom-v2"
V1_SOURCE_SHA256 = "a6c6591a8d10352df81adb7e0177889b1e2ee94c316aad873c4d3be1b0f62153"
MAX_PLANNED_BLOCKS = 100000
MAX_PLANNED_VIEWS = 1000
ACTIVATION_SOURCE = "https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/core/hashmap/HashMap.cpp"
check = original.check


class PhysicalFinalCapacityError(ValueError):
    def __init__(self, plan):
        self.plan = plan
        super().__init__(f"Original Final activation needs physical reserve capacity {plan.physical_blocks}, "
            f"above the separately declared physical budget {plan.physical_budget}; "
            f"its exact unique blocks {plan.required_blocks} fit logical limit {plan.configured_limit}. "
            "No Final candidate was allocated.")


@dataclass(frozen=True)
class HeadroomPlan:
    required_blocks: int
    configured_limit: int
    physical_blocks: int
    physical_budget: int
    maximum_frustum_rows: int
    final_voxel_m: float
    sdf_trunc_m: float
    accepted_views: int
    planning_wall_s: float
    capacity_fits: bool
    physical_fits: bool
    rows_json: str

    @property
    def report(self):
        result = asdict(self)
        result.pop("rows_json")
        result.update(kind=KIND, rows=json.loads(self.rows_json),
            allocated_attribute_bytes=self.physical_blocks*original.ATTRIBUTE_BYTES_PER_BLOCK
                if self.capacity_fits and self.physical_fits else 0,
            configured_attribute_bytes=self.configured_limit*original.ATTRIBUTE_BYTES_PER_BLOCK,
            policy="max(1, max(prefix unique BEFORE each original frame + full incoming frustum rows))",
            memory_scope="Physical TSDF/weight/color attribute floor only; logical unique hardcap remains configured. Native hashmap, scratch, images and pools additional.",
            activation_source=ACTIVATION_SOURCE,
            source_scope="Official v0.19 reserve rule explains actual v0.20 C5 growth; actual selected backend capacity needs new runtime proof")
        return result


class FrustumReserveRecorder:
    def __init__(self, indices):
        check(isinstance(indices,(list,tuple)) and len(indices) <= MAX_PLANNED_VIEWS
            and all(type(index) is int and index >= 0 for index in indices)
            and len(set(indices)) == len(indices),"Unsupported exact accepted-view order")
        self.indices = tuple(indices)
        self.blocks, self.rows = set(), []
        self.maximum_reserve, self.maximum_frustum = 0, 0

    def observe(self, coordinates):
        check(len(self.rows) < len(self.indices),"Planner emitted an extra frustum")
        rows = list(coordinates)
        check(len(rows) <= MAX_PLANNED_BLOCKS,"Unbounded original frustum coordinate rows")
        keys = []
        for row in rows:
            check(len(row) == 3,"Malformed original frustum coordinates")
            values = tuple(int(value) for value in row)
            check(all(value == converted and not isinstance(value,bool) and -(2**31) <= converted < 2**31
                      for value,converted in zip(row,values)),"Noninteger/out-of-domain original block coordinate")
            keys.append(values)
        unique = set(keys)
        check(len(unique) == len(keys),"Original compute_unique_block_coordinates returned duplicate rows")
        before = len(self.blocks)
        pessimistic = before+len(keys)
        self.blocks.update(unique)
        check(len(self.blocks) <= MAX_PLANNED_BLOCKS,"Planned union exceeds bounded experimental domain")
        self.maximum_reserve = max(self.maximum_reserve,pessimistic)
        self.maximum_frustum = max(self.maximum_frustum,len(keys))
        self.rows.append({"frame_index":self.indices[len(self.rows)],"prefix_unique_before":before,
            "frustum_rows":len(keys),"new_unique":len(self.blocks)-before,"prefix_unique_after":len(self.blocks),
            "reserve_bound":pessimistic,
            "ordered_coordinates_sha256":hashlib.sha256(b"".join(struct.pack("<iii",*key) for key in keys)).hexdigest()})

    def finish(self, required):
        check(type(required) is int and required == len(self.blocks) and len(self.rows) == len(self.indices),
              "Observed frustum union/order differs from unchanged original planner")
        check(self.maximum_reserve >= required,"Malformed pessimistic reserve bound")
        return max(1,self.maximum_reserve)


class ObservedCoordinateTensor:
    def __init__(self,tensor,recorder):
        self.tensor,self.recorder = tensor,recorder

    def cpu(self):
        return ObservedCpuCoordinates(self.tensor.cpu(),self.recorder)


class ObservedCpuCoordinates:
    def __init__(self,tensor,recorder):
        self.tensor,self.recorder = tensor,recorder

    def numpy(self):
        values = self.tensor.numpy()
        check(getattr(values,"ndim",None) == 2 and values.shape[1] == 3
            and getattr(values.dtype,"kind",None) == "i" and values.dtype.itemsize == 4,
              "Unsupported original frustum tensor dtype/shape")
        self.recorder.observe(values)
        return values  # identical original array, without changing its order/storage


class ObservedScratchGrid:
    def __init__(self,volume,recorder):
        self.volume,self.recorder = volume,recorder

    def __getattr__(self,name):
        return getattr(self.volume,name)

    def compute_unique_block_coordinates(self,*args,**kwargs):
        return ObservedCoordinateTensor(self.volume.compute_unique_block_coordinates(*args,**kwargs),self.recorder)


def plan_final_headroom(engine,*,original_final,physical_budget):
    check(hashlib.sha256((ROOT/"scripts/research/final_allocation_scope.py").read_bytes()).hexdigest() == V1_SOURCE_SHA256,
          "Measured original allocation helper changed; do not relabel its dependency")
    original.validate_original_functions(engine,original_final)
    check(engine.settings.final_voxel_m is not None and engine._final_vbg is None,"Plan only a new explicit Final volume")
    limit,voxel = engine.settings.final_block_count,engine.settings.final_voxel_m
    check(type(limit) is int and 1 <= limit <= 50000 and type(physical_budget) is int
          and 1 <= physical_budget <= MAX_PLANNED_BLOCKS and isinstance(voxel,(float,int))
          and not isinstance(voxel,bool) and math.isfinite(voxel) and voxel > 0,"Invalid logical/physical/voxel policy")
    started = time.perf_counter()
    recorder = FrustumReserveRecorder([index for index,_ in engine.poses])
    planner = copy.copy(engine)
    planner.__dict__ = original.private_containers(engine.__dict__)
    planner.settings = replace(engine.settings,voxel_m=voxel)
    planner.voxel_size = voxel
    check(planner.sdf_trunc == engine.sdf_trunc,"Headroom planner changed original SDF truncation")
    create = engine._create_vbg.__func__
    scratch = []
    def create_observed(owner,block_count=None):
        check(owner is planner and block_count == 1 and not scratch,"Original planner scratch allocation changed")
        value = create(owner,block_count=1)
        check(int(value.hashmap().capacity()) == 1 and int(value.hashmap().size()) == 0,"Planner scratch grid is not empty capacity1")
        scratch.append(value)
        return ObservedScratchGrid(value,recorder)
    planner._create_vbg = MethodType(create_observed,planner)
    method = engine._required_fusion_blocks.__func__
    local = dict(method.__globals__)
    local["prepare_metric_depth"] = original.private_depth_preparation(local["prepare_metric_depth"])
    private = FunctionType(method.__code__,local,method.__name__,method.__defaults__)
    private.__kwdefaults__ = method.__kwdefaults__
    poses = [(index,pose.copy()) for index,pose in engine.poses]
    required = private(planner,poses,None,stage="final_reserve_headroom_preflight")
    physical = recorder.finish(required)
    check(len(scratch) == 1 and int(scratch[0].hashmap().size()) == 0,"Planner activated voxel blocks")
    return HeadroomPlan(required,limit,physical,physical_budget,recorder.maximum_frustum,float(voxel),
        float(engine.sdf_trunc),len(poses),time.perf_counter()-started,required <= limit,physical <= physical_budget,
        json.dumps(recorder.rows,sort_keys=True,separators=(",",":"),allow_nan=False))


class HeadroomFinalScope(original.RightSizedFinalScope):
    def __init__(self,engine,*,original_final,physical_budget,on_plan=None):
        super().__init__(engine,original_final=original_final,on_plan=on_plan)
        self.physical_budget = physical_budget

    def final(self,owner,progress_cb=None):
        check(owner is self.engine,"Headroom scope received another engine")
        if owner.settings.final_voxel_m is None or owner._final_vbg is not None:
            return self.original_final(owner,progress_cb)
        plan = plan_final_headroom(owner,original_final=self.original_final,physical_budget=self.physical_budget)
        self.plans.append(plan)
        if self.on_plan is not None:
            self.on_plan(plan.report)
        if not plan.capacity_fits:
            raise original.FinalCapacityError(plan)
        if not plan.physical_fits:
            raise PhysicalFinalCapacityError(plan)
        original_create = owner._create_vbg.__func__
        allocations,start = self.allocations,len(self.allocations)
        class PrivateFinal(type(owner)):
            def _create_vbg(candidate,block_count=None):
                check(block_count == plan.configured_limit and len(allocations) == start,
                      "Original Final logical allocation count/cap changed")
                value = original_create(candidate,block_count=plan.physical_blocks)
                actual = int(value.hashmap().capacity())
                allocations.append({"requested_logical_blocks":block_count,"allocated_physical_blocks":plan.physical_blocks,
                    "actual_capacity":actual,"configured_logical_limit":plan.configured_limit})
                check(actual == plan.physical_blocks,"Backend initial physical capacity differs from planned reserve")
                return value
        proxy = copy.copy(owner)
        proxy.__class__ = PrivateFinal
        value = self.original_final(proxy,progress_cb)
        check(len(allocations) == start+1,"Original Final omitted or repeated its allocation")
        allocations[-1].update(final_capacity=int(value.hashmap().capacity()),final_unique_blocks=int(value.hashmap().size()))
        check(allocations[-1]["final_capacity"] == plan.physical_blocks,"Backend still resized beyond declared reserve; experiment failed")
        check(allocations[-1]["final_unique_blocks"] == plan.required_blocks,"Actual integrated unique keys differ from exact planner")
        owner.final_reconstruction = proxy.final_reconstruction
        owner.final_reconstruction.update(required_blocks=plan.required_blocks,physical_block_capacity=plan.physical_blocks,
            configured_block_limit=plan.configured_limit,physical_budget_blocks=plan.physical_budget,
            physical_attribute_budget_mib=plan.physical_blocks*original.ATTRIBUTE_BYTES_PER_BLOCK/2**20,
            configured_attribute_budget_mib=plan.configured_limit*original.ATTRIBUTE_BYTES_PER_BLOCK/2**20,
            planning_wall_s=plan.planning_wall_s,allocator_scope=KIND)
        return value

    def report(self):
        result = super().report()
        result.update(kind=KIND,physical_budget_blocks=self.physical_budget,
            policy="Original activation/math, pessimistic full-row reserve headroom, unchanged logical unique hardcap",
            note="Separate unmeasured scope; measured v1 failed due backend resize. No savings/mesh authority until new actual capacity and fixed-coordinate quality proofs.")
        return result


class ObservedOriginalHeadroomScope(HeadroomFinalScope):
    def final(self,owner,progress_cb=None):
        check(owner is self.engine,"Headroom control received another engine")
        if owner.settings.final_voxel_m is None or owner._final_vbg is not None:
            return self.original_final(owner,progress_cb)
        plan = plan_final_headroom(owner,original_final=self.original_final,physical_budget=self.physical_budget)
        self.plans.append(plan)
        if self.on_plan is not None:
            self.on_plan(plan.report)
        value = self.original_final(owner,progress_cb)
        self.allocations.append({"requested_logical_blocks":plan.configured_limit,
            "allocated_physical_blocks":int(value.hashmap().capacity()),"actual_capacity":int(value.hashmap().capacity()),
            "final_capacity":int(value.hashmap().capacity()),"final_unique_blocks":int(value.hashmap().size()),
            "configured_logical_limit":plan.configured_limit})
        return value

    def report(self):
        result = super().report()
        result["control"] = "Matched headroom observation, unchanged original configured allocation and overflow behavior"
        return result
