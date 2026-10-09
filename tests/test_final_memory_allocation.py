"""Stdlib proposal contracts: ownership/control, not native numerical proof."""

import ast
import copy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "scanner_server" / "fusion_allocation.py").exists():
    ROOT = Path(__file__).resolve().parent / "proposed"


class Mask:
    def __init__(self, values): self.values = list(values)
    def logical_not(self): return Mask(not value for value in self.values)
    def __invert__(self): raise TypeError("Open3D Tensor does not support unary ~")
    def cpu(self): return self
    def numpy(self): return self
    def all(self): return all(self.values)


class Keys:
    def __init__(self, values): self.values = list(values)
    def __len__(self): return len(self.values)
    def __getitem__(self, selected):
        return Keys(value for value,keep in zip(self.values,selected.values) if keep)


class HashMap:
    def __init__(self, capacity):
        self.allocated = capacity
        self.keys, self.activation_inputs = [], []
        self.native_error = None
        self.force_growth = False
        self.omit_found_after_activation = False
    def size(self): return len(self.keys)
    def capacity(self): return self.allocated
    def find(self, keys):
        existing = set(self.keys)
        found = [value in existing for value in keys.values]
        if self.omit_found_after_activation and self.activation_inputs and found:
            found[0] = False
        return None, Mask(found)
    def activate(self, keys):
        rows = list(keys.values)
        self.activation_inputs.append(rows)
        if self.size()+len(rows) > self.allocated:
            self.allocated = max(self.size()+len(rows), 2*self.allocated)
        existing = set(self.keys)
        for value in rows:
            if value not in existing:
                self.keys.append(value)
                existing.add(value)
            if self.native_error is not None: raise self.native_error
        if self.force_growth: self.allocated += 1
        self.last_result = object()
        return self.last_result


CORE = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None))
helper_tree = ast.parse((ROOT / "scanner_server/fusion_allocation.py").read_text(encoding="utf-8"))
helper_scope = {"sys":sys, "o3c":CORE}
exec(compile(ast.Module(body=[node for node in helper_tree.body if isinstance(node,ast.FunctionDef)],
                       type_ignores=[]), "<proposed-fusion-allocation>", "exec"), helper_scope)
activate = helper_scope["activate_fusion_blocks"]


def policy(capacity=3, *, device="CPU:0", enabled=True, logical=10):
    return SimpleNamespace(_final_missing_only_activation=enabled, _final_allocated_blocks=capacity,
        _fusion_block_limit=logical, settings=SimpleNamespace(confidence_fusion=True), device=device)


class ActivationContracts(unittest.TestCase):
    def test_uncapped_live_and_other_stages_delegate_original_input_and_return(self):
        for enabled in (False,None):
            engine = policy(enabled=enabled)
            mapping = HashMap(3)
            blocks = Keys([1,2,3])
            result = activate(engine,mapping,blocks)
            self.assertIs(result,mapping.last_result)
            activate(engine,mapping,blocks)
            self.assertEqual(mapping.activation_inputs,[[1,2,3],[1,2,3]])
            self.assertEqual(mapping.capacity(),6)
        engine = SimpleNamespace()
        mapping = HashMap(1)
        self.assertIs(activate(engine,mapping,Keys([1])),mapping.last_result)

    def test_all_existing_final_keys_prevent_pessimistic_growth(self):
        mapping, engine = HashMap(3), policy()
        activate(engine,mapping,Keys([1,2,3]))
        for _ in range(4): activate(engine,mapping,Keys([3,1,2]))
        self.assertEqual(mapping.activation_inputs,[[1,2,3]])
        self.assertEqual((mapping.capacity(),mapping.size()),(3,3))

    def test_partial_new_keys_keep_original_order_and_full_input_unchanged(self):
        mapping, engine = HashMap(4), policy(4)
        activate(engine,mapping,Keys([1,2]))
        blocks = Keys([2,4,1,3])
        activate(engine,mapping,blocks)
        self.assertEqual(mapping.activation_inputs,[[1,2],[4,3]])
        self.assertEqual(blocks.values,[2,4,1,3])
        self.assertEqual(mapping.keys,[1,2,4,3])

    def test_empty_final_frustum_does_not_activate(self):
        mapping = HashMap(1)
        activate(policy(1),mapping,Keys([]))
        self.assertEqual((mapping.capacity(),mapping.size()),(1,0))
        self.assertEqual(mapping.activation_inputs,[])

    def test_wrong_policy_capacity_and_budget_stop_before_mutation(self):
        configurations = [policy(0),policy(4,logical=3),policy(True),policy()]
        configurations[-1].settings.confidence_fusion = False
        for engine in configurations:
            with self.subTest(engine=engine):
                mapping = HashMap(3)
                with self.assertRaises(ValueError): activate(engine,mapping,Keys([1]))
                self.assertEqual(mapping.activation_inputs,[])
        mapping = HashMap(3)
        mapping.allocated = 4
        with self.assertRaises(ValueError): activate(policy(),mapping,Keys([1]))
        self.assertEqual(mapping.activation_inputs,[])
        mapping = HashMap(3)
        with self.assertRaises(ValueError): activate(policy(),mapping,Keys([1,2,3,4]))
        self.assertEqual(mapping.activation_inputs,[])

    def test_unexpected_native_growth_rejects_private_candidate(self):
        mapping = HashMap(3)
        mapping.force_growth = True
        with self.assertRaisesRegex(ValueError,"capacity or key count"):
            activate(policy(),mapping,Keys([1]))

    def test_full_original_frustum_membership_is_required_even_when_count_matches(self):
        mapping = HashMap(3)
        mapping.omit_found_after_activation = True
        with self.assertRaisesRegex(ValueError,"omitted original frustum keys"):
            activate(policy(),mapping,Keys([1,2,3]))
        self.assertEqual((mapping.capacity(),mapping.size()),(3,3))

    def test_cuda_retains_primary_native_error_and_selected_device_sync(self):
        for primary in (None,RuntimeError("Partial native candidate update")):
            mapping = HashMap(3)
            mapping.native_error = primary
            cleanup = RuntimeError("Selected-device synchronization failed")
            with patch.object(CORE.cuda,"synchronize",side_effect=cleanup) as sync:
                with self.assertRaises(RuntimeError) as caught:
                    activate(policy(device="CUDA:0"),mapping,Keys([1,2]))
                self.assertIs(caught.exception,primary if primary is not None else cleanup)
                sync.assert_called_once_with("CUDA:0")
            if primary is not None:
                self.assertTrue(any("synchronization" in note for note in primary.__notes__))

    def test_cuda_success_and_all_existing_still_finish_selected_work(self):
        mapping = HashMap(3)
        with patch.object(CORE.cuda,"synchronize") as sync:
            activate(policy(device="CUDA:0"),mapping,Keys([1,2]))
            activate(policy(device="CUDA:0"),mapping,Keys([2,1]))
            self.assertEqual(sync.call_count,2)
            self.assertTrue(all(call.args == ("CUDA:0",) for call in sync.call_args_list))

    def test_python310_cleanup_preserves_primary_without_exception_notes_api(self):
        class LegacyError(RuntimeError):
            add_note = None
        primary = LegacyError("Partial native candidate update")
        cleanup = RuntimeError("Selected-device synchronization failed")
        mapping = HashMap(3)
        mapping.native_error = primary
        # Model the Python 3.10 sys surface and exception API explicitly.
        legacy_sys = SimpleNamespace(exc_info=sys.exc_info)
        with patch.dict(helper_scope, sys=legacy_sys), patch.object(CORE.cuda,"synchronize",side_effect=cleanup) as sync:
            with self.assertRaises(LegacyError) as caught:
                activate(policy(device="CUDA:0"),mapping,Keys([1,2]))
            self.assertIs(caught.exception,primary)
            self.assertIs(caught.exception.__cause__,cleanup)
            sync.assert_called_once_with("CUDA:0")


@dataclass(frozen=True)
class Settings:
    voxel_m: float = .01
    final_voxel_m: float | None = .005
    final_block_count: int = 10
    confidence_fusion: bool = True


class Grid:
    def __init__(self, capacity):
        self.mapping = HashMap(capacity)
        self.updates = {}
    def hashmap(self): return self.mapping


class Engine:
    def __init__(self, batches=((1,2),(2,3),(1,3)), *, limit=10, weighted=True, required=None):
        self.settings = Settings(final_block_count=limit,confidence_fusion=weighted)
        self.voxel_size, self.sdf_trunc, self.device = .01,.08,"CPU:0"
        self.vbg, self.mesh, self.point_cloud, self.model_pcd = (object() for _ in range(4))
        self._final_vbg = None
        self.raw_frames = [(index,Keys(batch)) for index,batch in enumerate(batches)]
        self.poses = [(index,object()) for index in range(len(batches))]
        self.required = len(set(key for batch in batches for key in batch)) if required is None else required
        self.allocations, self.plans = [],[]
        self.stage_totals_ms, self.final_reconstruction = {},{"previous":True}
        self.initial_capacity_offset = 0
        self.native_error = None
    def _required_fusion_blocks(self,poses,progress_cb=None,*,stage):
        self.plans.append((self.voxel_size,self.sdf_trunc,stage,poses))
        return self.required
    def _create_vbg(self,block_count):
        self.allocations.append(block_count)
        value = Grid(block_count+self.initial_capacity_offset)
        value.mapping.native_error = self.native_error
        return value
    def _prepare_input(self,rgb,depth,settings): return rgb,depth
    def _integrate_vbg(self,rgb,depth,extrinsic):
        # The original caller's complete post-activation find and every original
        # key's update remain observable in this artificial ownership model.
        if self.settings.confidence_fusion:
            activate(self,self.vbg.hashmap(),depth)
        else:
            self.vbg.hashmap().activate(depth)
        _,found = self.vbg.hashmap().find(depth)
        if not all(found.values): raise RuntimeError("Original complete frustum find failed")
        for key in depth.values: self.vbg.updates[key] = self.vbg.updates.get(key,0)+1


engine_tree = ast.parse((ROOT / "scanner_server/engine.py").read_text(encoding="utf-8"))
cls = next(node for node in engine_tree.body if isinstance(node,ast.ClassDef) and node.name == "ScanEngine")
method = next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name == "_final_volume")
engine_scope = {"copy":copy,"replace":replace,"time":SimpleNamespace(monotonic=lambda:1.),
                "np":SimpleNamespace(linalg=SimpleNamespace(inv=lambda pose:pose)),
                # Inject a successful memory plan; real memory planning and
                # native integration are tested separately.
                "plan_fusion":lambda device,blocks,voxel: {
                    "allocation":"automatic", "required_blocks":blocks,
                    "allocated_blocks":max(1,(blocks*102+99)//100),
                    "attribute_budget_mib":max(1,(blocks*102+99)//100)*4096*20/2**20}}
exec(compile(ast.Module(body=[method],type_ignores=[]),"<proposed-final-volume>","exec"),engine_scope)
Engine._final_volume = engine_scope["_final_volume"]


def owners(engine):
    return engine.vbg,engine.mesh,engine.point_cloud,engine.model_pcd,engine._final_vbg,engine.poses,engine.raw_frames


class FinalContracts(unittest.TestCase):
    def test_weighted_final_exact_size_original_math_coverage_and_owner_rollback_boundary(self):
        engine = Engine()
        before = owners(engine)
        value = engine._final_volume()
        self.assertEqual(engine.allocations,[4])
        self.assertEqual(value.mapping.activation_inputs,[[1,2],[3]])
        self.assertEqual(value.updates,{1:2,2:2,3:2})
        self.assertEqual(owners(engine),before)
        self.assertFalse(hasattr(engine,"_final_missing_only_activation"))
        self.assertEqual(engine.plans[0][:3],(.005,.08,"final_reintegration"))
        self.assertIs(engine.plans[0][3],engine.poses)
        self.assertEqual((engine.voxel_size,engine.sdf_trunc),(.01,.08))
        self.assertEqual(engine.final_reconstruction["block_limit"],4)
        self.assertEqual(engine.final_reconstruction["allocated_blocks"],4)
        self.assertFalse(engine.final_reconstruction["applied"])

    def test_c7_legacy_limit_does_not_reject_automatic_allocation(self):
        engine = Engine(batches=(tuple(range(13302)),),limit=10000)
        before = owners(engine)
        value = engine._final_volume()
        self.assertEqual(engine.allocations,[13569])
        self.assertEqual(value.mapping.size(),13302)
        self.assertEqual(engine.final_reconstruction["allocation"],"automatic")
        self.assertEqual(owners(engine),before)

    def test_unweighted_plans_automatically_and_reports_actual_native_growth(self):
        engine = Engine(batches=((1,2,3),(1,2,3)),limit=3,weighted=False)
        before = owners(engine)
        value = engine._final_volume()
        self.assertEqual(len(engine.plans),1)
        self.assertEqual(engine.allocations,[4])
        self.assertEqual(value.mapping.activation_inputs,[[1,2,3],[1,2,3]])
        self.assertEqual(value.mapping.capacity(),8)
        self.assertEqual(engine.final_reconstruction["required_blocks"],3)
        self.assertEqual(engine.final_reconstruction["allocated_blocks"],8)
        self.assertEqual(engine.final_reconstruction["block_limit"],4)
        self.assertEqual(engine.final_reconstruction["allocation_strategy"],"automatic native activation")
        self.assertGreater(engine.final_reconstruction["attribute_budget_mib"],engine.final_reconstruction["planned_attribute_budget_mib"])
        self.assertEqual(owners(engine),before)

    def test_memory_preflight_failure_preserves_owners_before_candidate_allocation(self):
        engine = Engine()
        before, report = owners(engine),engine.final_reconstruction
        def unavailable(*args): raise ValueError("Not enough RAM")
        with patch.dict(engine_scope,plan_fusion=unavailable):
            with self.assertRaisesRegex(ValueError,"Not enough RAM"):
                engine._final_volume()
        self.assertEqual(engine.allocations,[])
        self.assertEqual(owners(engine),before)
        self.assertIs(engine.final_reconstruction,report)

    def test_disabled_and_cached_final_preserve_original_early_return(self):
        for disabled in (True,False):
            engine = Engine()
            if disabled: engine.settings = replace(engine.settings,final_voxel_m=None)
            else: engine._final_vbg = object()
            expected = engine.vbg if disabled else engine._final_vbg
            self.assertIs(engine._final_volume(),expected)
            self.assertEqual((engine.allocations,engine.plans),([],[]))

    def test_empty_union_preserves_minimum_native_capacity(self):
        engine = Engine(batches=())
        value = engine._final_volume()
        self.assertEqual(engine.allocations,[1])
        self.assertEqual(value.mapping.size(),0)
        self.assertEqual(engine.final_reconstruction["required_blocks"],0)

    def test_initial_capacity_fault_and_final_unique_count_mismatch_never_commit_live(self):
        for initial_offset,required in ((1,3),(0,4)):
            engine = Engine(required=required)
            engine.initial_capacity_offset = initial_offset
            before, report = owners(engine),engine.final_reconstruction
            with self.assertRaises(ValueError): engine._final_volume()
            self.assertEqual(owners(engine),before)
            self.assertIs(engine.final_reconstruction,report)
            self.assertEqual(engine.stage_totals_ms,{})

    def test_partial_native_candidate_fault_and_progress_fault_leave_previous_geometry(self):
        for native in (True,False):
            engine = Engine()
            primary = RuntimeError("Injected private Final failure")
            engine.native_error = primary if native else None
            before, report = owners(engine),engine.final_reconstruction
            def progress(*args): raise primary
            with self.assertRaises(RuntimeError) as caught:
                engine._final_volume(None if native else progress)
            self.assertIs(caught.exception,primary)
            self.assertEqual(owners(engine),before)
            self.assertIs(engine.final_reconstruction,report)
            self.assertEqual(engine.stage_totals_ms,{})


class OriginalMathSeams(unittest.TestCase):
    def test_three_voxel_update_bodies_are_unchanged_after_reversing_only_activation_call(self):
        expected = {
            "weighted_fusion.py":{"_integrate_cpu":"c461369a7edef7d08a2a6dc352f06794cade950a30d11ffc1b33ac8d43456593",
                "_integrate_tensor":"269d9fbc6d88aa6d56ab501a9e0a62642d98083387074b5c68344bff1cfb78c9"},
            "cuda_fusion.py":{"_integrate_validated":"58e6594564719729a7c1f9a1d923823f7377ecc3fdf7ce50b3dbd85b6a287785"}}
        for name,functions in expected.items():
            tree = ast.parse((ROOT / "scanner_server" / name).read_text(encoding="utf-8"))
            for function,digest in functions.items():
                node = next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name == function)
                seam = [item for item in node.body if isinstance(item,ast.Expr)
                    and isinstance(item.value,ast.Call) and isinstance(item.value.func,ast.Name)
                    and item.value.func.id == "activate_fusion_blocks"]
                self.assertEqual(len(seam),1)
                self.assertEqual(ast.unparse(seam[0].value),"activate_fusion_blocks(engine, hashmap, blocks)")
                seam[0].value = ast.parse("hashmap.activate(blocks)").body[0].value
                # Baseline digests use Python 3.12's AST representation.
                # 3.13 omits empty fields by default; 3.10/3.11 do not yet
                # expose type_params. Normalize metadata, preserving the
                # original numerical-body digests across supported Pythons.
                for item in ast.walk(node):
                    if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and "type_params" not in item._fields:
                        item._fields = (*item._fields,"type_params")
                        item.type_params = []
                options = {"show_empty":True} if sys.version_info >= (3,13) else {}
                normalized = ast.dump(node,include_attributes=False,**options)
                self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(),digest)


if __name__ == "__main__": unittest.main()
