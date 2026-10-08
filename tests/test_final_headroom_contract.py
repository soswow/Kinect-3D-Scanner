"""Stdlib causal reserve, logical/physical budget and transaction contracts."""

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import final_allocation_headroom as allocation
from scripts.research import final_allocation_scope as original
from tests.test_final_allocation_contract import Engine as OriginalFakeEngine, Volume, prepare_metric_depth


def key(index):
    return (index,0,0)


class Coordinates(list):
    dtype = SimpleNamespace(kind="i",itemsize=4)
    ndim = 2
    @property
    def shape(self):
        return (len(self),3)


class Tensor:
    def __init__(self,rows):
        self.rows = Coordinates(rows)
    def cpu(self):
        return self
    def numpy(self):
        return self.rows


class ReserveVolume(Volume):
    def __init__(self,capacity):
        super().__init__(capacity)
        self.keys,self.updates = set(),{}
    def size(self):
        return len(self.keys)
    def compute_unique_block_coordinates(self,rows):
        return Tensor(rows)
    def activate(self,rows):
        pessimistic = len(self.keys)+len(rows)
        if pessimistic > self.allocated:
            self.allocated = max(pessimistic,self.allocated*2)
        self.keys.update(rows)
        for value in rows:
            self.updates[value] = self.updates.get(value,0)+1


class Engine(OriginalFakeEngine):
    def __init__(self,batches,logical_limit=3):
        super().__init__()
        self.batches = batches
        self.settings = replace(self.settings,final_block_count=logical_limit)
        self._fusion_block_limit = logical_limit
        self.raw_frames = [(f"rgb{i}",f"depth{i}") for i in range(len(batches))]
        self.poses = [(i,SimplePose(i)) for i in range(len(batches))]
        self.vbg = ReserveVolume(logical_limit)
        self.unexpected_growth = self.omit_key = False
    def _create_vbg(self,block_count=None):
        self.events.append(("allocation",block_count,self.voxel_size,self.sdf_trunc))
        return ReserveVolume(block_count+self.capacity_offset if block_count != 1 else 1)
    def _required_fusion_blocks(self,proposals,progress_cb=None,stage="original"):
        scratch = self._create_vbg(block_count=1)
        blocks = set()
        for index,pose in proposals:
            prepare_metric_depth(self.raw_frames[index][1],self.settings)
            coordinates = scratch.compute_unique_block_coordinates(self.batches[index]).cpu().numpy()
            blocks.update(map(tuple,coordinates))
            self.backend["planner_visits"] += 1
            pose.values.append("private planner write")
        return len(blocks)
    def _integrate_vbg(self,rgb,depth,pose):
        index = int(depth[5:])
        rows = self.batches[index]
        added = len(set(rows)-self.vbg.keys)
        if self.vbg.size()+added > self._fusion_block_limit:
            raise ValueError("Original logical unique cap exceeded")
        self.events.append(("integrate",self._fusion_block_limit,self.sdf_trunc,self.voxel_size))
        self.vbg.activate(rows[:-1] if self.omit_key else rows)
        if self.unexpected_growth:
            self.vbg.allocated += 1
        self.helper.calls += 1
        self.backend["integrations"] += 1


class SimplePose:
    def __init__(self,index):
        self.values = [f"unrounded pose {index}"]
    def copy(self):
        return copy.deepcopy(self)


class ReserveBoundContracts(unittest.TestCase):
    def test_existing_keys_pessimistic_resize_reproduced_and_bound_prevents_it(self):
        batches = [[key(0),key(1),key(2)],[key(0),key(1)]]
        recorder = allocation.FrustumReserveRecorder([0,1])
        for rows in batches:
            recorder.observe(rows)
        physical = recorder.finish(3)
        self.assertEqual(physical,5)
        exact_only,headroom = ReserveVolume(3),ReserveVolume(physical)
        for rows in batches:
            exact_only.activate(rows)
            headroom.activate(rows)
        self.assertEqual(exact_only.capacity(),6)
        self.assertEqual(headroom.capacity(),5)
        self.assertEqual(exact_only.keys,headroom.keys)
        self.assertEqual(exact_only.updates,headroom.updates)

    def test_tight_prefix_bound_smaller_than_final_unique_plus_maximum_view(self):
        recorder = allocation.FrustumReserveRecorder([4,9])
        recorder.observe([key(0),key(1)])
        recorder.observe([key(2)])
        self.assertEqual(recorder.finish(3),3)
        self.assertEqual(recorder.maximum_frustum,2)
        self.assertLess(3,len(recorder.blocks)+recorder.maximum_frustum)
        self.assertEqual([row["frame_index"] for row in recorder.rows],[4,9])

    def test_empty_accepted_set_has_capacity_one_without_false_keys(self):
        recorder = allocation.FrustumReserveRecorder([])
        self.assertEqual(recorder.finish(0),1)
        self.assertEqual(recorder.blocks,set())

    def test_duplicate_fractional_out_of_range_extra_and_missing_rows_fail(self):
        for rows in ([key(1),key(1)],[(1.5,0,0)],[(2**31,0,0)],[(True,0,0)]):
            with self.subTest(rows=rows),self.assertRaises(original.FinalAllocationContractError):
                allocation.FrustumReserveRecorder([0]).observe(rows)
        recorder = allocation.FrustumReserveRecorder([0])
        with self.assertRaises(original.FinalAllocationContractError):
            recorder.finish(0)
        recorder.observe([key(0)])
        with self.assertRaises(original.FinalAllocationContractError):
            recorder.observe([key(1)])
        with self.assertRaises(original.FinalAllocationContractError):
            recorder.finish(2)

    def test_coordinate_transport_returns_identical_original_array(self):
        tensor = Tensor([key(0)])
        recorder = allocation.FrustumReserveRecorder([0])
        observed = allocation.ObservedCoordinateTensor(tensor,recorder).cpu().numpy()
        self.assertIs(observed,tensor.rows)
        self.assertEqual(recorder.finish(1),1)


class ScopeContracts(unittest.TestCase):
    def setUp(self):
        self.guard = patch.object(original,"validate_original_functions",return_value={})
        self.depth = patch.object(original,"private_depth_preparation",side_effect=lambda fn:fn)
        self.guard.start()
        self.depth.start()
        self.addCleanup(self.guard.stop)
        self.addCleanup(self.depth.stop)

    def state(self,engine):
        return (engine.vbg,engine.mesh,engine.pointcloud,engine.model,
            [[index,copy.deepcopy(pose.values)] for index,pose in engine.poses])

    def test_observed_original_planner_private_state_exact_union_and_headroom(self):
        engine = Engine([[key(0),key(1),key(2)],[key(0),key(1)]])
        before = self.state(engine)
        plan = allocation.plan_final_headroom(engine,original_final=Engine._final_volume,physical_budget=5)
        self.assertEqual((plan.required_blocks,plan.physical_blocks),(3,5))
        self.assertEqual(engine.backend["planner_visits"],0)
        self.assertEqual(engine.events,[])
        self.assertEqual(self.state(engine),before)
        self.assertEqual((plan.final_voxel_m,plan.sdf_trunc_m),(.005,.08))
        self.assertEqual([row["reserve_bound"] for row in plan.report["rows"]],[3,5])

    def test_original_integration_unchanged_logical_cap_helper_and_geometry_owners(self):
        engine = Engine([[key(0),key(1),key(2)],[key(0),key(1)]])
        before = self.state(engine)
        with allocation.HeadroomFinalScope(engine,original_final=Engine._final_volume,physical_budget=5) as scope:
            value = engine._final_volume()
        self.assertEqual((value.size(),value.capacity()),(3,5))
        self.assertEqual(engine.helper.calls,2)
        self.assertEqual(engine.backend["integrations"],2)
        self.assertEqual(self.state(engine),before)
        self.assertEqual([row for row in engine.events if row[0] == "integrate"],[("integrate",3,.08,.005)]*2)
        self.assertNotIn("_final_volume",engine.__dict__)
        self.assertEqual(scope.report()["kind"],allocation.KIND)
        self.assertEqual(scope.report()["actual_allocated_attribute_bytes"],[5*original.ATTRIBUTE_BYTES_PER_BLOCK])

    def test_logical_overflow_early_no_candidate_even_with_sufficient_physical_budget(self):
        engine = Engine([[key(0),key(1),key(2),key(3)]],logical_limit=3)
        before = self.state(engine)
        with self.assertRaises(original.FinalCapacityError) as caught:
            with allocation.HeadroomFinalScope(engine,original_final=Engine._final_volume,physical_budget=20):
                engine._final_volume()
        self.assertEqual(caught.exception.plan.required_blocks,4)
        self.assertEqual(engine.helper.calls,0)
        self.assertEqual(engine.events,[])
        self.assertEqual(self.state(engine),before)

    def test_physical_budget_distinct_early_failure_preserves_logical_configuration(self):
        engine = Engine([[key(0),key(1),key(2)],[key(0),key(1)]])
        before = self.state(engine)
        with self.assertRaises(allocation.PhysicalFinalCapacityError) as caught:
            with allocation.HeadroomFinalScope(engine,original_final=Engine._final_volume,physical_budget=4):
                engine._final_volume()
        self.assertEqual(caught.exception.plan.physical_blocks,5)
        self.assertTrue(caught.exception.plan.capacity_fits)
        self.assertEqual(engine.settings.final_block_count,3)
        self.assertEqual(engine.helper.calls,0)
        self.assertEqual(self.state(engine),before)

    def test_control_retains_original_backend_resize_and_error_path(self):
        engine = Engine([[key(0),key(1),key(2)],[key(0),key(1)]])
        with allocation.ObservedOriginalHeadroomScope(engine,original_final=Engine._final_volume,physical_budget=4) as scope:
            value = engine._final_volume()
        self.assertEqual(value.capacity(),6)
        self.assertFalse(scope.plans[0].physical_fits)
        self.assertEqual(scope.allocations[0]["requested_logical_blocks"],3)

    def test_unexpected_backend_growth_after_write_fails_without_geometry_commit(self):
        engine = Engine([[key(0),key(1),key(2)],[key(0),key(1)]])
        engine.unexpected_growth = True
        before = self.state(engine)
        with self.assertRaises(original.FinalAllocationContractError):
            with allocation.HeadroomFinalScope(engine,original_final=Engine._final_volume,physical_budget=5) as scope:
                engine._final_volume()
        self.assertGreater(scope.allocations[0]["final_capacity"],5)
        self.assertEqual(self.state(engine),before)
        self.assertEqual(engine.final_reconstruction,{"original":True})

    def test_initial_capacity_mismatch_rejected_before_fusion(self):
        engine = Engine([[key(0),key(1),key(2)]])
        engine.capacity_offset = 1
        with self.assertRaises(original.FinalAllocationContractError):
            with allocation.HeadroomFinalScope(engine,original_final=Engine._final_volume,physical_budget=5):
                engine._final_volume()
        self.assertEqual(engine.helper.calls,0)

    def test_integrated_unique_key_mismatch_rejected_before_commit(self):
        engine = Engine([[key(0),key(1),key(2)]])
        engine.omit_key = True
        with self.assertRaises(original.FinalAllocationContractError):
            with allocation.HeadroomFinalScope(engine,original_final=Engine._final_volume,physical_budget=5):
                engine._final_volume()
        self.assertEqual(engine.final_reconstruction,{"original":True})

    def test_physical_budget_boolean_zero_and_unbounded_rejected(self):
        for value in (True,0,100001):
            engine = Engine([[key(0)]])
            with self.subTest(value=value),self.assertRaises(original.FinalAllocationContractError):
                allocation.plan_final_headroom(engine,original_final=Engine._final_volume,physical_budget=value)
            self.assertEqual(engine.events,[])


if __name__ == "__main__":
    unittest.main()
