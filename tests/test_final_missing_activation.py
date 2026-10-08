"""Missing-key causal activation and private Final contracts; stdlib only.

Artificial tensors model ownership/control, not Open3D numerical authority.
The separately pinned actual three caller bodies are inspected without imports.
"""

import ast
import copy
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from tests.final_source_fixture import baseline_sources
from unittest.mock import patch

from scripts.research import final_missing_activation as missing
from scripts.research import final_allocation_scope as original
from tests.test_final_allocation_contract import Engine as BaseEngine, Settings as BaseSettings


def key(value):
    return (value, -value, 1)


class Array(list):
    def tolist(self):
        return copy.deepcopy(list(self))


class Tensor:
    def __init__(self, values, *, dtype="Int32", device="CPU:0", shape=None, contiguous=True):
        self.values = copy.deepcopy(list(values))
        self.dtype, self.device, self.contiguous = dtype, device, contiguous
        self.shape = shape if shape is not None else (
            (len(values), len(values[0])) if values and isinstance(values[0], (tuple, list))
            else (len(values),))

    def is_contiguous(self):
        return self.contiguous

    def cpu(self):
        return Tensor(self.values, dtype=self.dtype, shape=self.shape, contiguous=self.contiguous)

    def numpy(self):
        return Array(self.values)

    def to(self, dtype):
        return Tensor(self.values, dtype=dtype, device=self.device, shape=self.shape)

    def __getitem__(self, index):
        if index.dtype == "Bool":
            result = [row for row, keep in zip(self.values, index.values) if keep]
        else:
            result = [self.values[i] for i in index.values]
        return Tensor(result, dtype=self.dtype, device=self.device,
                      shape=(len(result), 3) if len(self.shape) == 2 else (len(result),))


CORE = SimpleNamespace(Tensor=Tensor, int32="Int32", int64="Int64", bool="Bool",
                       cuda=SimpleNamespace(synchronize=lambda device: None))


class HashMap:
    def __init__(self, capacity, device="CPU:0"):
        self.allocated, self.device = capacity, device
        self.keys, self.activate_inputs, self.find_inputs = [], [], []
        self.activate_fault = self.find_fault = None
        self.omit = self.grow = self.bad_ids = self.bad_masks = False

    def capacity(self):
        return self.allocated

    def size(self):
        return len(self.keys)

    def activate(self, values):
        rows = list(map(tuple, values.values))
        self.activate_inputs.append(rows)
        if self.activate_fault is not None:
            # Deliberately partial native mutation: private candidate must fail
            # and never be reused, rather than retrying or erasing Live data.
            if rows and rows[0] not in self.keys:
                self.keys.append(rows[0])
            raise self.activate_fault
        if self.size()+len(rows) > self.allocated:
            self.allocated = max(self.size()+len(rows), 2*self.allocated)
        new = [row not in self.keys for row in rows]
        for row in (rows[:-1] if self.omit else rows):
            if row not in self.keys:
                self.keys.append(row)
        if self.grow:
            self.allocated += 1
        return self.find(values)[0], Tensor(new, dtype=CORE.bool, device=self.device)

    def find(self, values):
        if self.find_fault is not None:
            raise self.find_fault
        rows = list(map(tuple, values.values))
        self.find_inputs.append(rows)
        found = [row in self.keys for row in rows]
        ids = [self.keys.index(row) if present else 0 for row, present in zip(rows, found)]
        if self.bad_ids:
            ids = [self.allocated]*len(rows)
        indices = Tensor(ids, device=self.device)
        mask = Tensor(found, dtype="Int32" if self.bad_masks else CORE.bool, device=self.device)
        return indices, mask

    def active_buf_indices(self):
        return Tensor(list(range(self.size())), device=self.device)

    def key_tensor(self):
        return Tensor(self.keys, shape=(self.size(), 3), device=self.device)

    def sentinel(self):
        return self


class Grid:
    def __init__(self, capacity, device="CPU:0"):
        self.map = HashMap(capacity, device)
        self.updates = {}
        self.attributes = {name: object() for name in ("tsdf", "weight", "color")}

    def hashmap(self):
        return self.map

    def attribute(self, name):
        return self.attributes[name]

    def extract_triangle_mesh(self, *args, **kwargs):
        return self.attributes["tsdf"]

    def voxel_coordinates_and_flattened_indices(self, value):
        return value, value


class Preparation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass(frozen=True)
class Settings(BaseSettings):
    confidence_fusion: bool = True


class Engine(BaseEngine):
    def __init__(self, batches, limit=10):
        super().__init__(len(set(value for batch in batches for value in batch)))
        self.settings = Settings(final_block_count=limit)
        self.device = "CPU:0"
        self.batches = batches
        self.raw_frames = [(f"rgb{i}", f"depth{i}") for i in range(len(batches))]
        self.poses = self.poses[:len(batches)]
        self.vbg = Grid(limit)
        self.create_native_fault = None

    def _create_vbg(self, block_count=None):
        self.events.append(("allocation", block_count, self.voxel_size, self.sdf_trunc))
        if self.create_fault is not None and block_count != 1:
            raise self.create_fault
        value = Grid(block_count+self.capacity_offset if block_count != 1 else 1)
        value.map.activate_fault = self.create_native_fault
        return value

    def _integrate_vbg(self, rgb, depth, pose):
        rows = self.batches[int(depth[5:])]
        blocks = Tensor(rows, shape=(len(rows), 3))
        mapping = self.vbg.hashmap()
        mapping.activate(blocks)
        buffers, found = mapping.find(blocks)
        if not all(found.values):
            raise RuntimeError("Artificial original full-frustum find lost keys")
        for row in rows:
            self.vbg.updates[row] = self.vbg.updates.get(row, 0)+1
        self.helper.calls += 1
        self.backend["integrations"] += 1


class MissingHashMapContracts(unittest.TestCase):
    def setUp(self):
        # Artificial hashmap tests exercise the contract interface; acceptance
        # of actual source is checked separately by WeightedSourceContracts.
        self.contract = object.__new__(missing.WeightedActivationSourceContract)
        self.caller = patch.object(self.contract, "caller")
        self.caller.start()
        self.addCleanup(self.caller.stop)

    def adapter(self, capacity=3, *, device="CPU:0", configuration=None, native=None):
        native = HashMap(capacity, device) if native is None else native
        return missing.MissingOnlyHashMap(native, core=CORE, device=device,
            physical_capacity=capacity, logical_limit=max(10, capacity), source_contract=self.contract,
            configuration=(lambda: (True, device)) if configuration is None else configuration)

    def blocks(self, values, **kwargs):
        return Tensor(values, shape=(len(values), 3), **kwargs)

    def test_pessimistic_growth_causal_control_and_missing_only_capacity(self):
        native, adapter = HashMap(3), self.adapter()
        batches = [[key(0), key(1), key(2)], [key(2), key(0)]]
        for values in batches:
            native.activate(self.blocks(values))
            adapter.activate(self.blocks(values))
        self.assertEqual(native.capacity(), 6)
        self.assertEqual(adapter.capacity(), 3)
        self.assertEqual(set(native.keys), set(adapter.native.keys))
        self.assertEqual(adapter.native.activate_inputs, [batches[0]])
        proof = adapter.verify_final(3)
        self.assertTrue(proof["exact_key_union"])
        self.assertEqual(proof["expected_union_sha256"], proof["actual_union_sha256"])

    def test_partial_new_rows_preserve_order_full_indices_and_insert_masks(self):
        adapter = self.adapter(4)
        first = [key(1), key(2)]
        ids, masks = adapter.activate(self.blocks(first))
        self.assertEqual(masks.values, [True, True])
        second = [key(2), key(3), key(1), key(4)]
        ids, masks = adapter.activate(self.blocks(second))
        self.assertEqual(adapter.native.activate_inputs, [first, [key(3), key(4)]])
        self.assertEqual(ids.values, [1, 2, 0, 3])
        self.assertEqual(masks.values, [False, True, False, True])
        self.assertEqual(ids.dtype, CORE.int32)
        self.assertEqual(masks.dtype, CORE.bool)
        self.assertEqual(adapter.native.find_inputs[-1], second)
        self.assertEqual(adapter.verify_final(4)["final_capacity"], 4)

    def test_empty_and_all_present_never_call_native_activate(self):
        adapter = self.adapter()
        ids, masks = adapter.activate(self.blocks([]))
        self.assertEqual((ids.values, masks.values), ([], []))
        self.assertEqual(adapter.native.activate_inputs, [])
        adapter.activate(self.blocks([key(1)]))
        for _ in range(3):
            ids, masks = adapter.activate(self.blocks([key(1)]))
            self.assertEqual((ids.values, masks.values), ([0], [False]))
        self.assertEqual(adapter.native.activate_inputs, [[key(1)]])
        self.assertEqual(adapter.report()["existing_rows"], 3)

    def test_reject_duplicate_malformed_contiguity_dtype_and_device_before_mutation(self):
        cases = [self.blocks([key(1), key(1)]), Tensor([[1, 2]], shape=(1, 2)),
                 self.blocks([key(1)], dtype="Int64"), self.blocks([key(1)], device="CUDA:0"),
                 self.blocks([key(1)], contiguous=False), self.blocks([(True, 0, 0)]),
                 self.blocks([(2**31, 0, 0)]), Tensor([key(1)], shape=(50001, 3))]
        for value in cases:
            with self.subTest(value=value.shape):
                adapter = self.adapter()
                with self.assertRaises(missing.MissingActivationContractError):
                    adapter.activate(value)
                self.assertEqual(adapter.native.activate_inputs, [])
                self.assertEqual(adapter.native.size(), 0)
                self.assertIsNotNone(adapter.fault)

    def test_budget_enforced_before_new_native_activation(self):
        adapter = self.adapter(2)
        adapter.activate(self.blocks([key(0), key(1)]))
        with self.assertRaisesRegex(missing.MissingActivationContractError, "budget"):
            adapter.activate(self.blocks([key(2)]))
        self.assertEqual(adapter.native.activate_inputs, [[key(0), key(1)]])
        self.assertEqual(adapter.native.size(), 2)

    def test_native_fault_latches_original_cause_without_retry(self):
        adapter = self.adapter()
        primary = RuntimeError("Native partial mutation")
        adapter.native.activate_fault = primary
        with self.assertRaises(RuntimeError) as caught:
            adapter.activate(self.blocks([key(0), key(1)]))
        self.assertIs(caught.exception, primary)
        self.assertEqual(adapter.native.size(), 1)
        for operation in (lambda: adapter.activate(self.blocks([key(1)])), lambda: adapter.capacity(),
                          lambda: adapter.verify_final(1)):
            with self.assertRaises(RuntimeError) as caught:
                operation()
            self.assertIs(caught.exception, primary)
        self.assertEqual(len(adapter.native.activate_inputs), 1)

    def test_wrong_find_masks_indices_omitted_keys_and_capacity_growth_fail_closed(self):
        for attribute in ("bad_masks", "bad_ids", "omit", "grow"):
            with self.subTest(attribute=attribute):
                adapter = self.adapter()
                if attribute == "bad_ids":
                    adapter.activate(self.blocks([key(0)]))
                setattr(adapter.native, attribute, True)
                with self.assertRaises(missing.MissingActivationContractError):
                    adapter.activate(self.blocks([key(0), key(1)]))
                self.assertFalse(adapter.calls[-1]["complete"])
                self.assertIsNotNone(adapter.fault)

    def test_policy_mutation_and_nonempty_candidate_are_rejected(self):
        cfg = [True, "CPU:0"]
        adapter = self.adapter(configuration=lambda: tuple(cfg))
        cfg[0] = False
        with self.assertRaisesRegex(missing.MissingActivationContractError, "policy"):
            adapter.activate(self.blocks([key(0)]))
        self.assertEqual(adapter.native.activate_inputs, [])
        native = HashMap(3)
        native.activate(self.blocks([key(0)]))
        with self.assertRaisesRegex(missing.MissingActivationContractError, "empty"):
            self.adapter(native=native)

    def test_selected_device_and_delegated_identity(self):
        adapter = self.adapter(device="CUDA:0")
        with patch.object(CORE.cuda, "synchronize") as synchronize:
            ids, masks = adapter.activate(self.blocks([key(0)], device="CUDA:0"))
            synchronize.assert_called_once_with("CUDA:0")
        self.assertEqual((ids.device, masks.device), ("CUDA:0", "CUDA:0"))
        self.assertIs(adapter.sentinel(), adapter.native)
        self.assertEqual(adapter.verify_final(1)["final_blocks"], 1)

    def test_cuda_synchronization_fault_preserves_primary_and_quarantines(self):
        for primary in (None, RuntimeError("Original native activation failure")):
            with self.subTest(primary=primary):
                adapter = self.adapter(device="CUDA:0")
                adapter.native.activate_fault = primary
                secondary = RuntimeError("Selected-device synchronization failed")
                with patch.object(CORE.cuda, "synchronize", side_effect=secondary) as synchronize:
                    with self.assertRaises(RuntimeError) as caught:
                        adapter.activate(self.blocks([key(0)], device="CUDA:0"))
                    self.assertIs(caught.exception, secondary if primary is None else primary)
                    synchronize.assert_called_once_with("CUDA:0")
                self.assertFalse(adapter.calls[-1]["complete"])
                self.assertIs(adapter.fault, caught.exception)
                if primary is not None:
                    self.assertTrue(any("synchronization" in note for note in primary.__notes__))

    def test_final_union_checks_actual_keys_not_only_count(self):
        adapter = self.adapter()
        adapter.activate(self.blocks([key(0)]))
        adapter.native.keys[0] = key(2)
        with self.assertRaisesRegex(missing.MissingActivationContractError, "active keys"):
            adapter.verify_final(1)

    def test_input_replacement_during_native_call_is_detected(self):
        adapter = self.adapter()
        value = self.blocks([key(0)])
        original_activate = adapter.native.activate
        def mutate(missing_keys):
            result = original_activate(missing_keys)
            value.values[0] = key(1)
            return result
        adapter.native.activate = mutate
        with self.assertRaises(missing.MissingActivationContractError):
            adapter.activate(value)


class MissingFinalContracts(unittest.TestCase):
    def setUp(self):
        fixture = baseline_sources()
        fixture.__enter__()
        self.addCleanup(fixture.__exit__, None, None, None)
        self.guards = [patch.object(original, "validate_original_functions", return_value={}),
                       patch.object(original, "private_depth_preparation", side_effect=lambda fn: fn),
                       patch.object(missing, "private_prepare_input", side_effect=lambda owner: owner._prepare_input.__func__),
                       patch.object(missing.WeightedActivationSourceContract, "caller"),
                       patch.dict(sys.modules, {"open3d": SimpleNamespace(core=CORE)})]
        for guard in self.guards:
            guard.start()
            self.addCleanup(guard.stop)

    def scope(self, engine):
        return missing.MissingKeyFinalScope(engine, original_final=BaseEngine._final_volume)

    def owners(self, engine):
        return (engine.vbg, engine.mesh, engine.pointcloud, engine.model, engine._final_vbg)

    def test_fit_retains_original_full_updates_and_returns_native_grid(self):
        engine = Engine([[key(0), key(1), key(2)], [key(0), key(1)]], limit=10)
        before = self.owners(engine)
        with self.scope(engine) as scope:
            final = engine._final_volume()
        self.assertIsInstance(final, Grid)
        self.assertNotIsInstance(final, missing.MissingOnlyGrid)
        self.assertEqual(final.hashmap().capacity(), 3)
        self.assertEqual(final.updates, {key(0): 2, key(1): 2, key(2): 1})
        self.assertEqual(final.hashmap().activate_inputs, [[key(0), key(1), key(2)]])
        self.assertEqual(self.owners(engine), before)
        self.assertEqual(engine.vbg.hashmap().size(), 0)
        self.assertEqual(engine.backend, {"integrations": 0, "planner_visits": 0})
        self.assertEqual(scope.private_telemetry[0]["backend"], {"integrations": 2, "planner_visits": 0})
        self.assertEqual((engine.voxel_size, engine.sdf_trunc), (.01, .08))
        self.assertEqual(engine.final_reconstruction["allocator_scope"], missing.KIND)
        self.assertTrue(scope.restored)
        self.assertTrue(scope.report()["final_key_proofs"][0]["exact_key_union"])
        self.assertFalse(scope.report()["geometry_proven"])

    def test_unweighted_policy_refused_before_planning_or_allocation(self):
        engine = Engine([[key(0)]])
        engine.settings = replace(engine.settings, confidence_fusion=False)
        with self.assertRaisesRegex(missing.MissingActivationContractError, "weighted"):
            with self.scope(engine):
                engine._final_volume()
        self.assertEqual(engine.events, [])
        self.assertEqual(engine.helper.calls, 0)

    def test_union_budget_overflow_refused_before_candidate_allocation(self):
        engine = Engine([[key(0), key(1)], [key(2)]], limit=2)
        before = self.owners(engine)
        with self.scope(engine) as scope:
            with self.assertRaises(original.FinalCapacityError):
                engine._final_volume()
        self.assertEqual(scope.allocations, [])
        self.assertEqual(engine.events, [])
        self.assertEqual(self.owners(engine), before)

    def test_native_partial_failure_never_commits_or_retries_and_scope_restores(self):
        engine = Engine([[key(0), key(1)]])
        before = self.owners(engine)
        primary = RuntimeError("Native candidate partial activation")
        engine.create_native_fault = primary
        with self.scope(engine) as scope:
            with self.assertRaises(RuntimeError) as caught:
                engine._final_volume()
            self.assertIs(caught.exception, primary)
            with self.assertRaisesRegex(missing.MissingActivationContractError, "one Final"):
                engine._final_volume()
        self.assertEqual(self.owners(engine), before)
        self.assertEqual(engine.final_reconstruction, {"original": True})
        self.assertEqual(engine.helper.calls, 0)
        self.assertTrue(scope.restored)
        self.assertEqual(len(scope.grids[0].volume.hashmap().activate_inputs), 1)

    def test_grid_delegation_and_unweighted_native_bypass_failure(self):
        engine = Engine([[key(0)]])
        with self.scope(engine) as scope:
            final = engine._final_volume()
            adapter = scope.grids[0]
            for name in ("tsdf", "weight", "color"):
                self.assertIs(adapter.attribute(name), final.attribute(name))
            self.assertIs(adapter.extract_triangle_mesh(), final.extract_triangle_mesh())
            marker = object()
            self.assertEqual(adapter.voxel_coordinates_and_flattened_indices(marker), (marker, marker))
            with self.assertRaisesRegex(missing.MissingActivationContractError, "bypasses"):
                adapter.integrate(object())
            with self.assertRaises(missing.MissingActivationContractError):
                adapter.attribute("tsdf")

    def test_empty_union_preserves_minimum_allocation_and_no_activation(self):
        engine = Engine([[]])
        with self.scope(engine) as scope:
            final = engine._final_volume()
        self.assertEqual(final.hashmap().capacity(), 1)
        self.assertEqual(final.hashmap().activate_inputs, [])
        self.assertEqual(scope.final_key_proofs[0]["required_blocks"], 0)

    def test_private_adapter_state_preserves_backend_aliases_and_native_owner(self):
        engine = Engine([[key(0)]])
        backend = {"cuda_input": {"calls": 0}}
        native = SimpleNamespace(lookup=object(), mode="unchanged")
        engine.backend = backend
        engine._input_preparation = Preparation(backend=backend, status=backend["cuda_input"], _gpu=native)
        engine._confidence_preparation = Preparation(backend=backend, status={"calls": 0}, _gpu=None)
        before = missing.preparation_owner_evidence(engine)
        private = missing.private_final_owner(engine)
        self.assertIsNot(private.backend, backend)
        self.assertIsNot(private._input_preparation, engine._input_preparation)
        self.assertIs(private._input_preparation.backend, private.backend)
        self.assertIs(private._input_preparation.status, private.backend["cuda_input"])
        self.assertIs(private._input_preparation._gpu, native)
        private._input_preparation.status["calls"] += 1
        private._confidence_preparation.status["calls"] += 1
        private.raw_frames.append(("private", "only"))
        self.assertEqual(engine._input_preparation.status["calls"], 0)
        self.assertEqual(engine._confidence_preparation.status["calls"], 0)
        self.assertEqual(len(engine.raw_frames), 1)
        self.assertEqual(missing.preparation_owner_evidence(engine), before)

    def test_shared_native_configuration_mutation_cannot_commit_final(self):
        engine = Engine([[key(0)]])
        native = SimpleNamespace(lookup=object(), mode="original")
        engine._input_preparation = Preparation(backend=engine.backend, status={}, _gpu=native)
        before = self.owners(engine)
        scope = self.scope(engine)
        original_final = scope.original_final
        def mutate(owner, callback=None):
            result = original_final(owner, callback)
            native.mode = "changed"
            return result
        scope.original_final = mutate
        with scope:
            with self.assertRaises(missing.MissingActivationContractError):
                engine._final_volume()
        self.assertEqual(self.owners(engine), before)
        self.assertEqual(engine.final_reconstruction, {"original": True})
        self.assertFalse(scope.preparation_observations[0]["data_owners_and_configuration_unchanged"])
        self.assertFalse(scope.preparation_observations[0]["native_array_bit_rollback_proven"])

    def test_native_modules_with_shape_functions_are_opaque_configuration_owners(self):
        engine = Engine([[key(0)]])
        cp, np = ModuleType("cupy"), ModuleType("numpy")
        def shape_function(value):
            raise AssertionError("Observer must never invoke module shape functions")
        cp.shape = np.shape = shape_function
        native = SimpleNamespace(cp=cp, np=np, kernel=shape_function, mode="unchanged")
        engine._input_preparation = Preparation(backend=engine.backend, status={}, _gpu=native)
        before = missing.preparation_owner_evidence(engine)
        states = before["_input_preparation"]["gpu_state"]["dict"]
        self.assertEqual(states["cp"], {"owner": id(cp), "type": "builtins.module", "module_name": "cupy"})
        self.assertEqual(states["np"]["module_name"], "numpy")
        self.assertEqual(states["kernel"]["owner"], id(shape_function))
        self.assertNotIn("shape", states["cp"])
        with self.scope(engine) as scope:
            result = engine._final_volume()
        self.assertEqual(result.hashmap().size(), 1)
        self.assertEqual(missing.preparation_owner_evidence(engine), before)
        self.assertTrue(scope.preparation_observations[0]["data_owners_and_configuration_unchanged"])
        self.assertFalse(scope.preparation_observations[0]["native_array_bit_rollback_proven"])

    def test_array_shape_metadata_is_retained_and_shape_methods_not_iterated(self):
        engine = Engine([[key(0)]])
        def shape_method():
            raise AssertionError("Opaque shape methods are not arrays")
        array = SimpleNamespace(shape=(2, 3), dtype="<f4", device="CUDA:0", nbytes=24)
        opaque = SimpleNamespace(shape=shape_method, dtype="opaque")
        native = SimpleNamespace(array=array, opaque=opaque)
        engine._input_preparation = Preparation(backend=engine.backend, status={}, _gpu=native)
        before = missing.preparation_owner_evidence(engine)
        states = before["_input_preparation"]["gpu_state"]["dict"]
        self.assertEqual(states["array"]["shape"], [2, 3])
        self.assertEqual(states["array"]["dtype"], "<f4")
        self.assertNotIn("shape", states["opaque"])
        array.shape = (3, 2)
        self.assertNotEqual(missing.preparation_owner_evidence(engine), before)

    def test_replaced_module_owner_with_same_name_is_detected(self):
        engine = Engine([[key(0)]])
        native = SimpleNamespace(cp=ModuleType("cupy"))
        engine._input_preparation = Preparation(backend=engine.backend, status={}, _gpu=native)
        before = missing.preparation_owner_evidence(engine)
        native.cp = ModuleType("cupy")
        self.assertNotEqual(missing.preparation_owner_evidence(engine), before)

    def test_observation_failure_preserves_original_activation_fault(self):
        engine = Engine([[key(0)]])
        primary = RuntimeError("Native activation failed")
        engine.create_native_fault = primary
        scope = self.scope(engine)
        original_evidence = missing.preparation_owner_evidence
        count = [0]
        def broken(owner):
            count[0] += 1
            if count[0] > 1:
                raise RuntimeError("Observer failed")
            return original_evidence(owner)
        with patch.object(missing, "preparation_owner_evidence", side_effect=broken):
            with scope:
                with self.assertRaises(RuntimeError) as caught:
                    engine._final_volume()
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("observation" in note for note in primary.__notes__))


class WeightedSourceContracts(unittest.TestCase):
    def test_real_pinned_source_guard_and_unused_tuple_contract(self):
        try:
            current = missing.WeightedActivationSourceContract()
        except missing.MissingActivationContractError as error:
            self.assertIn("Original weighted activation caller changed", str(error))
        else:
            self.assertEqual(len(current.caller_signatures), 3)
            self.assertEqual(current.unchanged(), current.source_sha256)
        # Preserved source acceptance is an explicit artificial source fixture;
        # it cannot authorize changed current production callers.
        with baseline_sources():
            contract = missing.WeightedActivationSourceContract()
            self.assertEqual(len(contract.caller_signatures), 3)
            self.assertEqual(contract.unchanged(), contract.source_sha256)

    def test_unreviewed_caller_refused_without_native_mutation(self):
        with baseline_sources():
            contract = missing.WeightedActivationSourceContract()
            native = HashMap(1)
            adapter = missing.MissingOnlyHashMap(native, core=CORE, device="CPU:0", physical_capacity=1,
                logical_limit=1, source_contract=contract, configuration=lambda: (True, "CPU:0"))
            with self.assertRaisesRegex(missing.MissingActivationContractError, "caller"):
                adapter.activate(Tensor([key(0)]))
            self.assertEqual(native.activate_inputs, [])

    def test_source_change_refused_without_numerical_import(self):
        real = Path.read_bytes
        with patch.object(Path, "read_bytes", lambda path: real(path)+b"\n" if path.name == "final_allocation_scope.py" else real(path)):
            with self.assertRaisesRegex(missing.MissingActivationContractError, "helper changed"):
                missing.WeightedActivationSourceContract()

    def test_activation_result_consumption_is_not_silently_authorized(self):
        with baseline_sources():
            real = Path.read_text
            def altered(path, *args, **kwargs):
                text = real(path, *args, **kwargs)
                return text.replace("    hashmap.activate(blocks)", "    ignored = hashmap.activate(blocks)") if path.name == "weighted_fusion.py" else text
            with patch.object(Path, "read_text", altered):
                with self.assertRaises(missing.MissingActivationContractError):
                    missing.WeightedActivationSourceContract()


if __name__ == "__main__":
    unittest.main()
