"""Stdlib-only current raw provenance and original allocation-boundary tests."""
import copy
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import probe_final_budget as probe


def profile():
    return {"schema_version":1,"finish_requested":True,"mesh_built":True,"pose_seeds_used":False,
        "input_changed_during_profile":False,"source_changed_during_profile":False,"source_sha256":"a"*64,
        "input_sha256":"b"*64,"experimental_visual_fallback":False,"live_s":1.,"frames":2,
        "selected_indices":[0,1],"live_diagnostics":[{},{}],"settings":{"confidence_fusion":True,
            "final_voxel_m":.005,"final_block_count":20000},"settings_overrides":{"final_block_count":20000},
        "final_reconstruction":{"applied":True,"voxel_m":.005,"allocation_strategy":"exact missing-key activation",
            "required_blocks":13302,"blocks":13302,"initial_block_capacity":13302,"allocated_blocks":13302},
        "native_mode":"on","native_extension":{"changed_during_profile":False,"sha256":"c"*64},
        "poses":[{"index":0,"camera_to_world":[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]}],
        "accepted_indices":[0],"accepted":1,"thread_policy":{"open3d_threads":20,"opencv_threads":20},"omp_threads":"8"}


def preflight(value):
    return probe.validate_profile(value,current_source="a"*64,input_digest="b"*64,test_limit=10000,expected_required=13302)


class FakeEngine:
    def __init__(self):
        self.settings=SimpleNamespace(confidence_fusion=True,final_block_count=10000)
        self.voxel_size=.01;self.poses=[(0,[[1.]])];self.actual_required=13302
        self.native_creations=[];self.candidate_attempt=False;self.planner_count=0
    def _create_vbg(self,block_count=None):
        self.native_creations.append(block_count)
        return SimpleNamespace(hashmap=lambda:SimpleNamespace(capacity=lambda:block_count))
    def _required_fusion_blocks(self,proposals,progress_cb=None,*,stage="fragment_reconnection"):
        self.planner_count+=1
        self._create_vbg(block_count=1)
        return self.actual_required
    def _integrate_vbg(self,*args):raise AssertionError("Original fusion must never be reached")
    def process_frames(self,*args):raise AssertionError("Original tracking must never be reached")
    def build_mesh(self,*args):raise AssertionError("Original mesh must never be reached")
    def _final_volume(self):
        candidate=copy.copy(self);candidate.voxel_size=.005;candidate._fusion_block_limit=self.settings.final_block_count
        if self.candidate_attempt:candidate._create_vbg(block_count=10000)
        required=candidate._required_fusion_blocks(self.poses,stage="final_capacity_preflight")
        if required > candidate._fusion_block_limit:
            raise ValueError(f"Final 0.005 m model needs {required} blocks; increase final_block_count from {candidate._fusion_block_limit} "
                f"to at least {required}, or choose a coarser final voxel. No Final fusion candidate was allocated.")


class ProvenanceTests(unittest.TestCase):
    def test_current_completed_full_raw_only(self):
        self.assertEqual(preflight(profile())["required_blocks"],13302)
    def test_old_core_wrong_raw_changed_or_unfinished_rejected(self):
        for key,value in (("source_sha256","old"),("input_sha256","old"),("source_changed_during_profile",True),
            ("input_changed_during_profile",True),("mesh_built",False),("finish_requested",False),("pose_seeds_used",True)):
            item=profile();item[key]=value
            with self.subTest(key=key),self.assertRaises(probe.BudgetProbeFault):preflight(item)
    def test_checkpoints_research_archived_policy_or_no_live_replay_rejected(self):
        for key,value in (("checkpoint",{"sha256":"old"}),("research_finish",{"mode":"native"}),
            ("experimental_visual_fallback",True),("live_s",None),("live_s",0.)):
            item=profile();item[key]=value
            with self.subTest(key=key),self.assertRaises(probe.BudgetProbeFault):preflight(item)
    def test_partial_shuffled_bool_or_missing_raw_selection_rejected(self):
        for key,value in (("selected_indices",[0]),("selected_indices",[1,0]),("selected_indices",[False,1]),
            ("frames",True),("live_diagnostics",[{}])):
            item=profile();item[key]=value
            with self.subTest(key=key),self.assertRaises(probe.BudgetProbeFault):preflight(item)
    def test_known_current_final_algorithm_required(self):
        for key,value in (("required_blocks",13303),("required_blocks",13302.0),("allocated_blocks",20000),
            ("applied",False),("allocation_strategy","configured native activation"),("voxel_m",.01)):
            item=profile();item["final_reconstruction"][key]=value
            with self.subTest(key=key),self.assertRaises(probe.BudgetProbeFault):preflight(item)
        item=profile();item["settings"]["confidence_fusion"]=False
        with self.assertRaises(probe.BudgetProbeFault):preflight(item)
    def test_only_test_budget_is_allowed_to_change(self):
        item=profile();item["settings_overrides"]["voxel_m"]=.005
        with self.assertRaises(probe.BudgetProbeFault):preflight(item)
        with self.assertRaises(probe.BudgetProbeFault):probe.validate_profile(profile(),current_source="a"*64,
            input_digest="b"*64,test_limit=20000,expected_required=13302)
    def test_original_native_and_actual_threads_required(self):
        for key,value in (("native_mode","off"),("native_extension",{"changed_during_profile":True,"sha256":"c"*64}),
            ("thread_policy",{"open3d_threads":True,"opencv_threads":20})):
            item=profile();item[key]=value
            with self.subTest(key=key),self.assertRaises(probe.BudgetProbeFault):preflight(item)
    def test_final_pose_order_membership_and_finite_precision(self):
        item=profile();item["poses"][0]["index"]=1
        with self.assertRaises(probe.BudgetProbeFault):preflight(item)
        for value in (float("nan"),float("inf"),True,"1"):
            item=profile();item["poses"][0]["camera_to_world"][0][0]=value
            with self.subTest(value=value),self.assertRaises(probe.BudgetProbeFault):preflight(item)
    def test_no_allocation_flag_stops_before_native_imports(self):
        with self.assertRaisesRegex(probe.BudgetProbeFault,"allocated"):
            probe.run(SimpleNamespace(run_allocated=False))
    def test_original_serializer_has_no_rounding_or_reconstruction_seed(self):
        contract=probe.pose_serializer_contract()
        self.assertTrue(contract["original_p_tolist_expression"])
        self.assertFalse(contract["loaded_code_checked"])
    def test_loaded_serializer_does_not_inherit_helpers_future_flags(self):
        tree=ast.parse((probe.ROOT/"scanner_server/engine.py").read_text(encoding="utf-8"))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "ScanEngine")
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "reconstruction_report")
        namespace={}
        exec(compile(ast.Module(body=[method],type_ignores=[]),"current-engine-serializer","exec",dont_inherit=True),namespace)
        owner=SimpleNamespace(reconstruction_report=namespace["reconstruction_report"])
        self.assertTrue(probe.pose_serializer_contract(owner)["loaded_code_checked"])
        original=owner.reconstruction_report
        owner.reconstruction_report=lambda self:None
        with self.assertRaisesRegex(probe.BudgetProbeFault,"Loaded"):probe.pose_serializer_contract(owner)
        self.assertIsNot(owner.reconstruction_report,original)


class SettingsReconstructionTests(unittest.TestCase):
    def actual_settings(self):
        # Actual ScanSettings.to_dict camera distortion remains a tuple, while
        # its persisted profile is the lossless JSON list of these same doubles.
        return {"camera":{"distortion":(-0.13439164241585547,0.5252875235942552,
            0.0031447382908560296,0.0010622336987000532,-0.6435102692824457)},
            "roi":(0,0,640,480),"confidence_fusion":True,"final_block_count":20000,
            "final_voxel_m":.005,"signed_zero":-0.0}

    def test_tuple_camera_and_roi_match_persisted_json_without_input_mutation(self):
        actual=self.actual_settings();before=copy.deepcopy(actual)
        saved=json.loads(json.dumps(actual,allow_nan=False))
        self.assertNotEqual(actual,saved)
        result=probe.validate_reconstructed_settings(actual,saved)
        self.assertEqual(result["sha256"],probe.canonical(saved))
        self.assertEqual(result["numeric_tolerance"],0)
        self.assertEqual(actual,before)
        self.assertIsInstance(actual["camera"]["distortion"],tuple)

    def test_changed_coefficient_order_roi_keys_or_one_ulp_rejected(self):
        actual=self.actual_settings()
        for mutate in (lambda saved:saved["camera"]["distortion"].reverse(),
            lambda saved:saved["roi"].__setitem__(2,639),
            lambda saved:saved.pop("roi"),
            lambda saved:saved["camera"]["distortion"].__setitem__(0,
                math.nextafter(saved["camera"]["distortion"][0],math.inf))):
            saved=json.loads(json.dumps(actual,allow_nan=False));mutate(saved)
            with self.assertRaisesRegex(probe.BudgetProbeFault,"reconstruct"):
                probe.validate_reconstructed_settings(actual,saved)

    def test_bool_integer_float_and_signed_zero_remain_distinct(self):
        actual=self.actual_settings()
        for key,value in (("confidence_fusion",1),("final_block_count",20000.0),("signed_zero",0.0)):
            saved=json.loads(json.dumps(actual,allow_nan=False));saved[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(probe.BudgetProbeFault,"reconstruct"):
                probe.validate_reconstructed_settings(actual,saved)

    def test_nonfinite_or_unserializable_settings_rejected(self):
        for value in (float("nan"),float("inf"),object()):
            saved=self.actual_settings();saved["final_voxel_m"]=value
            with self.subTest(value=type(value).__name__),self.assertRaisesRegex(probe.BudgetProbeFault,"finite JSON"):
                probe.validate_reconstructed_settings(self.actual_settings(),saved)


class BoundaryTests(unittest.TestCase):
    def boundary(self,engine):
        return probe.AllocationBoundary(engine,FakeEngine,logical_limit=10000,required_blocks=13302,
            pose_digest=lambda poses:probe.canonical(poses))
    def test_only_original_scratch_and_single_planner_original_error_pass(self):
        engine=FakeEngine();original={name:getattr(FakeEngine,name) for name in
            ("_create_vbg","_required_fusion_blocks","_integrate_vbg","process_frames","build_mesh")}
        boundary=self.boundary(engine)
        with boundary:
            with self.assertRaises(ValueError) as error:engine._final_volume()
        result=boundary.close(error.exception)
        self.assertFalse(result["candidate_allocated"])
        self.assertEqual(engine.native_creations,[1])
        self.assertTrue(all(getattr(FakeEngine,name) is value for name,value in original.items()))
    def test_candidate_is_stopped_before_native_allocation(self):
        engine=FakeEngine();engine.candidate_attempt=True
        with self.boundary(engine),self.assertRaisesRegex(probe.BudgetProbeFault,"candidate"):
            engine._final_volume()
        self.assertEqual(engine.native_creations,[])
    def test_planner_wrong_count_or_changed_pose_latches(self):
        engine=FakeEngine();engine.actual_required=13303
        with self.boundary(engine),self.assertRaisesRegex(probe.BudgetProbeFault,"count"):
            engine._final_volume()
        engine=FakeEngine();boundary=self.boundary(engine)
        with boundary:
            engine.poses[0][1][0][0]=2.
            with self.assertRaisesRegex(probe.BudgetProbeFault,"inputs"):
                engine._final_volume()
    def test_extra_scratch_and_all_tracking_fusion_entrypoints_are_forbidden(self):
        for name in ("_integrate_vbg","process_frames","build_mesh"):
            engine=FakeEngine()
            with self.subTest(name=name),self.boundary(engine),self.assertRaises(probe.BudgetProbeFault):getattr(engine,name)()
        engine=FakeEngine();engine.voxel_size=.005;engine._fusion_block_limit=10000
        with self.boundary(engine):
            engine._create_vbg(block_count=1)
            with self.assertRaises(probe.BudgetProbeFault):engine._create_vbg(block_count=1)
        self.assertEqual(engine.native_creations,[1])
    def test_wrong_native_scratch_capacity_stops(self):
        engine=FakeEngine()
        def wrong(owner,block_count=None):return SimpleNamespace(hashmap=lambda:SimpleNamespace(capacity=lambda:2))
        with patch.object(FakeEngine,"_create_vbg",wrong),self.boundary(engine),self.assertRaisesRegex(probe.BudgetProbeFault,"capacity"):
            engine._final_volume()
    def test_arbitrary_valueerror_without_original_completed_plan_cannot_pass(self):
        boundary=self.boundary(FakeEngine())
        with self.assertRaises(probe.BudgetProbeFault):boundary.close(ValueError("Unrelated native failure"))
    def test_native_failure_is_original_and_all_hooks_restore(self):
        engine=FakeEngine();primary=OSError("Native scratch failure");original=FakeEngine._required_fusion_blocks
        def failed(*args,**kwargs):raise primary
        with patch.object(FakeEngine,"_create_vbg",failed):
            with self.assertRaises(OSError) as result:
                with self.boundary(engine):engine._final_volume()
            self.assertIs(result.exception,primary)
            self.assertIs(FakeEngine._required_fusion_blocks,original)
    def test_restoration_attempts_every_hook_independently(self):
        class Owner:
            def __setattr__(self,name,value):
                if name == "fail":raise OSError("Restore failure")
                object.__setattr__(self,name,value)
        owner=Owner();original=object()
        errors=probe.restore_hooks([(owner,"healthy",original),(owner,"fail",original)])
        self.assertEqual(len(errors),1)
        self.assertIs(owner.healthy,original)


if __name__ == "__main__":unittest.main()
