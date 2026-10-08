"""Global marker source seam, frame direction and hard-failure contracts.

All matrices/fixtures below are artificial stdlib values. Original numerical
pose/NN authority must still be established in a separately allocated replay.
"""

import ast
import copy
import hashlib
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import global_marker_proposal_scope as marker


def module_fixture():
    module = ModuleType("scanner_server.fragments")
    module.MAX_PAIRS, module.MAX_FRAGMENTS, module.MAX_FRAGMENT_VIEWS = 256, 32, 16
    exec("def propose_fragment_poses(engine, progress_cb=None):\n return ['original']\n", module.__dict__)
    return module


def fake_enhanced(original):
    code = compile("def propose_fragment_poses(engine, progress_cb=None):\n"
                   " proposals=['original-top4']\n"
                   " for seed in (10,20):\n"
                   "  proposals.append(_global_seed(engine.source,engine.target,seed))\n"
                   " proposals.extend(_offline_global_marker_seed_provider(engine.source,engine.target,engine.camera))\n"
                   " return [_verify_bridge(value) for value in proposals]\n", "artificial", "exec")
    function = next(value for value in code.co_consts if hasattr(value,"co_name"))
    return FunctionType(function, original.__globals__, "propose_fragment_poses", (None,)), {"artificial":True}


class ScopeContracts(unittest.TestCase):
    def setUp(self):
        self.module = module_fixture()
        self.provider = SimpleNamespace(global_seeds=lambda *args:["marker"],
            validate_seed=lambda value:None, seed_hash=lambda value:value, report=lambda:{"artificial":True})
        self.build = patch.object(marker,"enhanced_function",side_effect=fake_enhanced)
        self.build.start()
        self.addCleanup(self.build.stop)
        self.engine = SimpleNamespace(source=SimpleNamespace(index=1),target=SimpleNamespace(index=2),camera=object())

    def test_original_seeds_remain_and_later_original_observer_hooks_are_resolved(self):
        original = self.module.propose_fragment_poses
        global_calls, verification = [],[]
        self.module._global_seed = lambda source,target,seed: global_calls.append(seed) or ("ransac",seed)
        self.module._verify_bridge = lambda pose: verification.append(pose) or pose
        with marker.GlobalMarkerProposalScope(self.module,self.provider) as scope:
            # Mimic observers/thread wrappers installed after the scope.
            self.module._global_seed = lambda source,target,seed: global_calls.append(seed) or ("wrapped-ransac",seed)
            result = self.module.propose_fragment_poses(self.engine)
        self.assertEqual(global_calls,[10,20])
        self.assertEqual(result,["original-top4",("wrapped-ransac",10),("wrapped-ransac",20),"marker"])
        self.assertEqual(verification,result)
        self.assertIs(self.module.propose_fragment_poses,original)
        self.assertNotIn(marker.PROVIDER_SYMBOL,self.module.__dict__)
        self.assertTrue(scope.restored)

    def test_provider_or_trace_fault_is_latched_after_core_masks_it(self):
        for failing_trace in (False,True):
            with self.subTest(trace=failing_trace):
                self.provider.global_seeds = lambda *args:["marker"]
                trace = None
                if failing_trace:
                    def trace(_):
                        raise RuntimeError("trace failure")
                else:
                    self.provider.global_seeds = lambda *args: (_ for _ in ()).throw(RuntimeError("provider failure"))
                scope = marker.GlobalMarkerProposalScope(self.module,self.provider,trace=trace)
                with self.assertRaises(marker.GlobalMarkerFailure) as caught:
                    with scope:
                        try:
                            scope.call(self.engine.source,self.engine.target,self.engine.camera)
                        except BaseException:
                            raise RuntimeError("cleanup masked the research fault")
                self.assertIs(caught.exception,scope.failure)
                self.assertTrue(scope.restored)

    def test_more_than_four_or_malformed_additions_fail_before_graph_verification(self):
        for values in (list(range(5)),(1,2)):
            self.provider.global_seeds = lambda *args:values
            with self.assertRaises(marker.GlobalMarkerFailure):
                with marker.GlobalMarkerProposalScope(self.module,self.provider) as scope:
                    scope.call(self.engine.source,self.engine.target,self.engine.camera)
            self.assertTrue(scope.restored)

    def test_wrong_entry_order_does_not_clobber_an_existing_original_observer(self):
        scope = marker.GlobalMarkerProposalScope(self.module,self.provider)
        observer = lambda *args:"existing observer"
        self.module.propose_fragment_poses = observer
        with self.assertRaises(marker.GlobalMarkerFailure):
            scope.__enter__()
        self.assertIs(self.module.propose_fragment_poses,observer)
        self.assertNotIn(marker.PROVIDER_SYMBOL,self.module.__dict__)

    def test_existing_global_symbol_is_restored_without_class_or_local_patches(self):
        previous = object()
        self.module.__dict__[marker.PROVIDER_SYMBOL] = previous
        original_local = object()
        self.module._local_match = original_local
        with marker.GlobalMarkerProposalScope(self.module,self.provider) as scope:
            self.assertIs(self.module._local_match,original_local)
        self.assertIs(self.module.__dict__[marker.PROVIDER_SYMBOL],previous)
        with self.assertRaises(marker.GlobalMarkerFailure):
            scope.__enter__()

    def test_changed_original_pair_budget_cannot_enter_marker_scope(self):
        self.module.MAX_PAIRS = 257
        with self.assertRaises(marker.GlobalMarkerFailure):
            marker.GlobalMarkerProposalScope(self.module,self.provider).__enter__()
        self.assertNotIn(marker.PROVIDER_SYMBOL,self.module.__dict__)

    def test_provider_cannot_change_original_pair_budget_while_adding_seeds(self):
        def mutate(*args):
            self.module.MAX_PAIRS = 300
            return ["marker"]
        self.provider.global_seeds = mutate
        with self.assertRaises(marker.GlobalMarkerFailure):
            with marker.GlobalMarkerProposalScope(self.module,self.provider) as scope:
                scope.call(self.engine.source,self.engine.target,self.engine.camera)
        self.assertTrue(scope.restored)

    def test_cleanup_fault_is_hard_even_with_an_ordinary_body_failure(self):
        original = self.module.propose_fragment_poses
        class RestoreFault(ModuleType):
            def __setattr__(owner,name,value):
                if name == "propose_fragment_poses" and value is original and owner.__dict__.get("fail_restore"):
                    raise RuntimeError("hook restore failed")
                super().__setattr__(name,value)
        self.module.__class__ = RestoreFault
        scope = marker.GlobalMarkerProposalScope(self.module,self.provider)
        primary = RuntimeError("ordinary body exception")
        with self.assertRaises(marker.GlobalMarkerFailure) as caught:
            with scope:
                self.module.fail_restore = True
                raise primary
        self.assertIs(caught.exception,scope.failure)
        self.assertIs(caught.exception.__cause__,primary)
        self.assertFalse(scope.restored)
        self.assertNotIn(marker.PROVIDER_SYMBOL,self.module.__dict__)
        self.assertTrue(any("hook restore failed" in note for note in caught.exception.__notes__))


class SourceContracts(unittest.TestCase):
    def loaded_original_without_imports(self,source):
        module = ModuleType("scanner_server.fragments")
        compiled = compile(source,str(marker.ROOT/"scanner_server/fragments.py"),"exec",dont_inherit=True)
        code = next(value for value in compiled.co_consts if hasattr(value,"co_name") and value.co_name == "propose_fragment_poses")
        return FunctionType(code,module.__dict__,"propose_fragment_poses",(None,))

    def test_pinned_entire_original_body_and_shared_live_globals(self):
        source = (marker.ROOT/"scanner_server/fragments.py").read_text(encoding="utf-8")
        original = self.loaded_original_without_imports(source)
        enhanced,binding = marker.enhanced_function(original)
        self.assertIs(enhanced.__globals__,original.__globals__)
        self.assertEqual(binding["original_ast_sha256"],marker.ORIGINAL_AST_SHA256)
        self.assertNotEqual(binding["enhanced_ast_sha256"],marker.ORIGINAL_AST_SHA256)
        self.assertTrue(binding["removal_reconstructs_entire_original"])
        self.assertIn(marker.PROVIDER_SYMBOL,enhanced.__code__.co_names)
        self.assertEqual(enhanced.__defaults__,original.__defaults__)
        self.assertNotIn("numpy",sys.modules)

    def test_changed_verification_or_ranking_body_is_rejected_even_with_matching_loaded_code(self):
        source = (marker.ROOT/"scanner_server/fragments.py").read_text(encoding="utf-8")
        changed = source.replace('while pending and report["tested_pairs"] < MAX_PAIRS:',
                                 'while pending and report["tested_pairs"] < 1:')
        self.assertNotEqual(changed,source)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"fragments.py"
            path.write_text(changed,encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"entire fragment proposal body"):
                marker.enhanced_function(self.loaded_original_without_imports(changed),path)

    def test_loaded_other_function_is_rejected_even_with_original_source(self):
        other = module_fixture().propose_fragment_poses
        with self.assertRaisesRegex(ValueError,"loaded unwrapped original"):
            marker.enhanced_function(other)


class Matrix:
    """Exact artificial rigid frame composition, using only Python doubles."""
    shape = (4,4)
    def __init__(self,values):
        self.values = tuple(tuple(row) for row in values)
    def __matmul__(self,other):
        return Matrix([[sum(self.values[i][k]*other.values[k][j] for k in range(4)) for j in range(4)] for i in range(4)])
    def inverse(self):
        rotation = [[self.values[j][i] for j in range(3)] for i in range(3)]
        return Matrix([rotation[i]+[-sum(rotation[i][j]*self.values[j][3] for j in range(3))] for i in range(3)]+[[0.,0.,0.,1.]])
    def bytes(self):
        return struct.pack("<16d",*(value for row in self.values for value in row))


def translation(x,y=0):
    return Matrix([[1.,0.,0.,x],[0.,1.,0.,y],[0.,0.,1.,0.],[0.,0.,0.,1.]])


class ProviderContracts(unittest.TestCase):
    def setUp(self):
        self.camera = object()
        self.calls = []
        finite = lambda pose:SimpleNamespace(all=lambda:all(math.isfinite(value) for row in pose.values for value in row))
        self.native = SimpleNamespace(np=SimpleNamespace(isfinite=finite,linalg=SimpleNamespace(inv=lambda pose:pose.inverse())),
            engine=SimpleNamespace(settings=SimpleNamespace(camera=self.camera)),
            _features=lambda index:(None,{i:i for i in range(50)}),
            features_hash=lambda features:copy.deepcopy(features),
            array_hash=lambda matrix:{"sha256":hashlib.sha256(matrix.bytes()).hexdigest()},report=lambda:{})
        self.fragments = SimpleNamespace(_rigid=lambda pose:True,
            _disagrees=lambda left,right,distance,angle:left.bytes()!=right.bytes())
        self.source = SimpleNamespace(index=0,keys=[SimpleNamespace(index=1,pose=translation(10),features={"original":1})])
        self.target = SimpleNamespace(index=1,keys=[SimpleNamespace(index=2,pose=translation(20,5),features={"original":2})])
        self.native.proposal = lambda source,target,camera:self.calls.append((source.index,target.index)) or translation(3,2)

    def test_seed_maps_source_fragment_into_target_fragment_using_original_camera_poses(self):
        # Rotation makes composition order matter, beyond translation signs.
        self.target.keys[0].pose = Matrix([[0.,-1.,0.,20.],[1.,0.,0.,5.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        before = [view.pose.bytes() for view in self.source.keys+self.target.keys]
        provider = marker.OriginalMarkerGlobalProvider(self.native,self.fragments)
        seeds = provider.global_seeds(self.source,self.target,self.camera)
        expected = self.target.keys[0].pose @ translation(3,2) @ self.source.keys[0].pose.inverse()
        self.assertEqual(seeds[0].bytes(),expected.bytes())
        self.assertEqual([view.pose.bytes() for view in self.source.keys+self.target.keys],before)
        self.assertEqual(provider.rows[0]["camera_pairs"][0]["common_identity_corners"],50)

    def test_insufficient_original_corner_support_never_calls_marker_pnp(self):
        self.native._features = lambda index:(None,{i:i for i in range(39)})
        provider = marker.OriginalMarkerGlobalProvider(self.native,self.fragments)
        self.assertEqual(provider.global_seeds(self.source,self.target,self.camera),[])
        self.assertEqual(self.calls,[])

    def test_additions_are_ranked_by_actual_shared_corners_and_bounded_to_four(self):
        counts = {1:70,2:60,3:55,11:65,12:50,13:45}
        self.source.keys = [SimpleNamespace(index=i,pose=translation(i),features={"i":i}) for i in (1,2,3)]
        self.target.keys = [SimpleNamespace(index=i,pose=translation(i),features={"i":i}) for i in (11,12,13)]
        self.native._features = lambda index:(None,{i:i for i in range(counts[index])})
        self.native.proposal = lambda a,b,camera:self.calls.append((a.index,b.index)) or translation(a.index*100+b.index)
        provider = marker.OriginalMarkerGlobalProvider(self.native,self.fragments)
        self.assertEqual(len(provider.global_seeds(self.source,self.target,self.camera)),4)
        self.assertEqual(self.calls,[(1,11),(2,11),(3,11),(1,12)])

    def test_wrong_camera_oversized_keys_or_mutated_original_witness_is_fatal(self):
        provider = marker.OriginalMarkerGlobalProvider(self.native,self.fragments)
        with self.assertRaisesRegex(ValueError,"calibration"):
            provider.global_seeds(self.source,self.target,object())
        oversized = SimpleNamespace(index=0,keys=self.source.keys*19)
        with self.assertRaisesRegex(ValueError,"preparation bound"):
            provider.global_seeds(oversized,self.target,self.camera)
        def mutation(a,b,camera):
            a.features["original"] = "changed"
            return translation(3)
        self.native.proposal = mutation
        with self.assertRaisesRegex(ValueError,"SIFT witness bytes"):
            provider.global_seeds(self.source,self.target,self.camera)

    def test_duplicate_markers_do_not_bypass_original_seed_uniqueness(self):
        self.source.keys += [SimpleNamespace(index=3,pose=translation(10),features={"i":3})]
        checks = []
        self.fragments._disagrees = lambda a,b,distance,angle:checks.append((distance,angle)) or a.bytes()!=b.bytes()
        provider = marker.OriginalMarkerGlobalProvider(self.native,self.fragments)
        self.assertEqual(len(provider.global_seeds(self.source,self.target,self.camera)),1)
        self.assertEqual(checks,[(.01,1)])


if __name__ == "__main__":
    unittest.main()
