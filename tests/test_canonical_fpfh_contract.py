"""Stdlib-only boundary, ownership, source and hard-failure contracts."""

import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings
import zipfile

from scripts.research import canonical_fpfh_proposals as helper


class Fragment:
    def __init__(self,index,points):
        self.index,self.points = index,points
        self.train,self.heldout,self.coarse,self.fpfh = object(),object(),object(),object()


class Builder:
    def __init__(self):
        self.prepared,self.cleared = [],False
        self.action = None

    def prepare(self,fragment):
        if self.action:
            self.action(fragment)
        private = copy.copy(fragment)
        private.points = helper.quantize_rows_reference(fragment.points)
        private.coarse,private.fpfh = object(),object()
        self.prepared.append((fragment,private))
        return private

    def snapshot(self,fragment):
        return [copy.deepcopy(fragment.points),id(fragment.train),id(fragment.heldout),id(fragment.coarse),id(fragment.fpfh)]

    def clear(self):
        self.cleared = True

    def report(self):
        return {"cleared":self.cleared}


def scope_setup():
    module,builder = types.SimpleNamespace(),Builder()
    observed = []
    output = object()
    def original(source,target,seed):
        observed.append((source,target,seed))
        return output
    module._global_seed = original
    source,target = Fragment(0,[[1.0000001,0.0,-0.0]]),Fragment(2,[[2.0000001,1.0,-0.0]])
    scope = helper.CanonicalGlobalSeedScope(module,builder,binding_factory=lambda _: {"original":True})
    return module,builder,observed,output,original,source,target,scope


def npy_bytes(shape,values,dtype="<f8"):
    header = repr({"descr":dtype,"fortran_order":False,"shape":shape}).encode("latin1")
    header += b" "*((64-(10+len(header)+1)%64)%64)+b"\n"
    return b"\x93NUMPY\x01\x00"+len(header).to_bytes(2,"little")+header+struct.pack("<"+"d"*len(values),*values)


def diagnostic_fixture(folder, *, dtype="<f8", duplicate=False):
    folder = Path(folder)
    snapshot = folder/"input.npz"
    inputs = {}
    with zipfile.ZipFile(snapshot,"w") as archive:
        for side in ("source","target"):
            for part in ("train_points","train_normals","coarse_points","coarse_normals","fpfh"):
                name = side+"_"+part
                shape = (33,1) if part == "fpfh" else (1,3)
                values = [0.0]*math.prod(shape)
                key = "row0_"+name
                archive.writestr(key+".npy",npy_bytes(shape,values,dtype))
                inputs[name] = {"dtype":dtype,"shape":list(shape),"snapshot_key":key,
                    "sha256":hashlib.sha256(struct.pack("<"+"d"*len(values),*values)).hexdigest()}
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",UserWarning)
                archive.writestr("row0_source_train_points.npy",npy_bytes((1,3),[0.,0.,0.]))
    report = {"kind":"offline-finish-global-proposal-input-diagnostic-v2","status":"complete",
        "supervisor_restored":True,"performance_authority":False,"artifacts_sha256":{"producer":"a"*64},
        "rows":[{"invocation":0,"source":0,"target":2,"original_seed":2,"complete":True,"inputs":inputs}],
        "snapshot":{"path":str(snapshot),"sha256":helper.file_hash(snapshot)}}
    path = folder/"report.json"
    path.write_text(json.dumps(report),encoding="utf-8")
    return path,report,snapshot


class QuantizationContracts(unittest.TestCase):
    def test_negative_zero_normalized_and_inputs_preserved(self):
        rows = [[-0.0,0.0,-.1e-6],[1.0000001,-1.0000001,0.0]]
        before = copy.deepcopy(rows)
        result = helper.quantize_rows_reference(rows)
        self.assertEqual(rows,before)
        self.assertEqual(struct.pack("<d",result[0][0]),b"\0"*8)
        self.assertEqual(struct.pack("<d",result[0][2]),b"\0"*8)
        self.assertEqual(result[1],[1.0,-1.0,0.0])

    def test_ties_to_even_reference_both_signs(self):
        for multiplier,expected in ((.5,0.),(1.5,2.),(2.5,2.),(-.5,0.),(-1.5,-2.),(-2.5,-2.)):
            actual,_ = helper.quantize_scalar(multiplier*helper.STEP_M,reject_halfway=False)
            self.assertEqual(actual,expected*helper.STEP_M)
            with self.assertRaises(helper.CanonicalProposalError):
                helper.quantize_scalar(multiplier*helper.STEP_M)

    def test_halfway_neighbors_rejected_without_silent_rounding(self):
        midpoint = 1.5*helper.STEP_M
        for value in (math.nextafter(midpoint,-math.inf),midpoint,math.nextafter(midpoint,math.inf)):
            with self.assertRaises(helper.CanonicalProposalError):
                helper.quantize_scalar(value)
        for value in ((1.5-1e-7)*helper.STEP_M,(1.5+1e-7)*helper.STEP_M):
            helper.quantize_scalar(value)

    def test_tiny_observed_scale_drift_canonicalizes_when_away_from_boundary(self):
        left = 1.23456712
        right = left+3.2e-15
        self.assertEqual(helper.quantize_scalar(left)[0],helper.quantize_scalar(right)[0])

    def test_row_order_duplicates_and_coordinate_error_bound(self):
        rows = [[2.0000001,1.0,0.0],[1.0000001,2.0,0.0],[2.0000001,1.0,0.0]]
        result = helper.quantize_rows_reference(rows)
        self.assertEqual(result,[[2.,1.,0.],[1.,2.,0.],[2.,1.,0.]])
        self.assertLessEqual(max(abs(a-b) for x,y in zip(rows,result) for a,b in zip(x,y)),.5e-6)

    def test_nonfinite_oversized_non_fp64_and_bad_shape_fail_closed(self):
        for value in (math.nan,math.inf,-math.inf,1000.1,True,1):
            with self.assertRaises(helper.CanonicalProposalError):
                helper.quantize_scalar(value)
        for rows in ([],[[0.,0.]],[[0.,0.,0.]]*(helper.MAX_POINTS+1)):
            with self.assertRaises(helper.CanonicalProposalError):
                helper.quantize_rows_reference(rows)

    def test_explicit_normal_sign_is_invariant_to_whole_sign_with_axis_ties(self):
        for vector in ([.8,-.5,-.3],[-.5,.5,0.],[-0.,0.,-1.]):
            self.assertEqual(helper.orient_normal_reference(vector),helper.orient_normal_reference([-x for x in vector]))
        self.assertEqual(helper.orient_normal_reference([-.5,.5,0.]),[.5,-.5,0.])
        self.assertEqual(helper.policy()["normal_policy"],"fresh")
        self.assertEqual(helper.policy("largest-component-positive")["normal_policy"],"largest-component-positive")
        with self.assertRaises(helper.CanonicalProposalError):
            helper.orient_normal_reference([0.,0.,0.])


class ScopeContracts(unittest.TestCase):
    def test_private_proposal_only_seed_and_identical_return_original_verification_owners(self):
        module,builder,seen,output,original,a,b,scope = scope_setup()
        with scope:
            result = module._global_seed(a,b,10002)
            self.assertIs(result,output)
            source,target,seed = seen[0]
            self.assertEqual(seed,10002)
            self.assertIsNot(source,a)
            self.assertIsNot(source.coarse,a.coarse)
            self.assertIsNot(source.fpfh,a.fpfh)
            self.assertIs(source.train,a.train)
            self.assertIs(source.heldout,a.heldout)
            self.assertEqual(source.points,[[1.,0.,0.]])
            self.assertEqual(a.points,[[1.0000001,0.,-0.]])
        self.assertIs(module._global_seed,original)
        self.assertTrue(builder.cleared)
        self.assertTrue(scope.report()["hooks_restored"])
        self.assertFalse(scope.report()["geometry_quality_proven"])

    def test_preprocessing_failure_latches_even_if_core_masked_it(self):
        module,builder,_,_,original,a,b,scope = scope_setup()
        builder.action = lambda _: (_ for _ in ()).throw(ValueError("unsupported boundary"))
        with self.assertRaises(helper.CanonicalProposalFailure):
            with scope:
                try:
                    module._global_seed(a,b,2)
                except BaseException:
                    pass
        self.assertIs(module._global_seed,original)
        self.assertTrue(builder.cleared)

    def test_input_mutation_detected_and_scope_restored(self):
        module,builder,_,_,_,a,b,_ = scope_setup()
        original_points = a.points
        def original(*_):
            original_points[0][0] += .01
            return object()
        module._global_seed = original
        scope = helper.CanonicalGlobalSeedScope(module,builder,binding_factory=lambda _: {})
        with self.assertRaises(helper.CanonicalProposalFailure):
            with scope:
                module._global_seed(a,b,2)
        self.assertIs(module._global_seed,original)

    def test_cached_private_ransac_mutation_detected(self):
        module,builder,_,_,_,a,b,_ = scope_setup()
        def original(source,*_):
            source.points[0][0] += .01
        module._global_seed = original
        scope = helper.CanonicalGlobalSeedScope(module,builder,binding_factory=lambda _: {})
        with self.assertRaises(helper.CanonicalProposalFailure):
            with scope:
                module._global_seed(a,b,2)

    def test_wrong_entry_order_does_not_clobber_installed_observer(self):
        module,_,_,_,_,_,_,scope = scope_setup()
        observer = lambda *args: None
        module._global_seed = observer
        with self.assertRaises(helper.CanonicalProposalFailure):
            with scope:
                pass
        self.assertIs(module._global_seed,observer)

    def test_restore_fault_hard_latched_with_ordinary_primary_and_other_cleanup_attempted(self):
        module,builder,_,_,original,_,_,scope = scope_setup()
        class BrokenModule:
            def __init__(self):
                self._global_seed = original
                self.armed = False
            def __setattr__(self,name,value):
                if name == "_global_seed" and getattr(self,"armed",False) and value is original:
                    raise RuntimeError("injected hook restore failure")
                object.__setattr__(self,name,value)
        module = BrokenModule()
        scope.module = module
        primary = ValueError("body failed")
        with self.assertRaises(helper.CanonicalProposalFailure) as caught:
            with scope:
                module.armed = True
                raise primary
        self.assertIs(caught.exception.__cause__,primary)
        self.assertTrue(builder.cleared)
        self.assertFalse(scope.restored)

    def test_seed_boolean_rejected_before_original_and_no_scope_reuse(self):
        module,_,seen,_,_,a,b,scope = scope_setup()
        with self.assertRaises(helper.CanonicalProposalFailure):
            with scope:
                module._global_seed(a,b,True)
        self.assertEqual(seen,[])
        with self.assertRaises(helper.CanonicalProposalFailure):
            scope.__enter__()

    def test_trace_failure_is_hard_latched(self):
        module,_,_,_,_,a,b,scope = scope_setup()
        scope.trace = lambda _: (_ for _ in ()).throw(RuntimeError("lost proof trace"))
        with self.assertRaises(helper.CanonicalProposalFailure):
            with scope:
                module._global_seed(a,b,2)


class FreshPreparationContracts(unittest.TestCase):
    def stand_ins(self, *, entry_threads=1, normal_failure=None, restore_failure=False):
        class Values:
            def __init__(self,values,shape):
                self.values,self.shape,self.dtype = values,shape,"float64"
            def tobytes(self,order="C"):
                return struct.pack("<"+"d"*len(self.values),*self.values)
            def __len__(self):
                return self.shape[0]
        class Norms:
            def __gt__(self,value):
                return [True]
        calls = []
        class Cloud:
            def __init__(self):
                self.points,self.normals = Values([], (0,3)),Values([], (0,3))
            def has_normals(self):
                return len(self.normals) > 0
            def estimate_normals(self,settings):
                calls.append(("normals",settings,self.has_normals()))
                if normal_failure:
                    raise normal_failure
                self.normals = Values([0.,0.,1.],(1,3))
        threads = [entry_threads]
        def set_threads(value):
            calls.append(("threads",value))
            if restore_failure and len([x for x in calls if x[0] == "threads"]) >= 2:
                raise RuntimeError("injected restoration failure")
            threads[0] = value
        def fpfh(cloud,settings):
            calls.append(("fpfh",settings))
            return types.SimpleNamespace(data=Values([0.]*33,(33,1)))
        np = types.SimpleNamespace(float64="float64",asarray=lambda value:value,
            isfinite=lambda value:types.SimpleNamespace(all=lambda:all(math.isfinite(x) for x in value.values)),
            linalg=types.SimpleNamespace(norm=lambda value,axis:Norms()),all=all)
        o3d = types.SimpleNamespace(geometry=types.SimpleNamespace(PointCloud=Cloud,KDTreeSearchParamHybrid=lambda **kw:kw),
            utility=types.SimpleNamespace(Vector3dVector=lambda value:value,get_max_threads=lambda:threads[0],set_max_threads=set_threads),
            pipelines=types.SimpleNamespace(registration=types.SimpleNamespace(compute_fpfh_feature=fpfh)))
        points,rounded = Values([1.0000001,0.,0.],(1,3)),Values([1.,0.,0.],(1,3))
        fake_quantize = lambda *args:(rounded,{"canonical_sha256":helper.array_hash(rounded),"points":1})
        return np,o3d,points,calls,threads,fake_quantize

    def test_fresh_cloud_original_searches_and_ransac_thread_context_restored(self):
        np,o3d,points,calls,threads,quantize = self.stand_ins()
        builder = helper.FreshCanonicalFeatures(np,o3d)
        with patch.object(helper,"canonical_points",quantize):
            cloud,features,record = builder.build(points)
        self.assertEqual(calls,[("threads",20),("normals",{"radius":.08,"max_nn":30},False),
                              ("fpfh",{"radius":.2,"max_nn":100}),("threads",1)])
        self.assertEqual(threads,[1])
        self.assertEqual(record["entry_threads"],1)
        self.assertEqual(record["restored_threads"],1)
        self.assertEqual(helper.array_hash(points),helper.array_hash(self.stand_ins()[2]))

    def test_normal_estimation_fault_still_restores_threads(self):
        primary = ValueError("original feature estimate failed")
        np,o3d,points,calls,threads,quantize = self.stand_ins(normal_failure=primary)
        with patch.object(helper,"canonical_points",quantize):
            with self.assertRaises(ValueError) as caught:
                helper.FreshCanonicalFeatures(np,o3d).build(points)
        self.assertIs(caught.exception,primary)
        self.assertEqual(threads,[1])
        self.assertNotIn("fpfh",[row[0] for row in calls])

    def test_primary_feature_failure_preserved_when_thread_cleanup_also_fails(self):
        primary = ValueError("original feature estimate failed")
        np,o3d,points,_,_,quantize = self.stand_ins(normal_failure=primary,restore_failure=True)
        with patch.object(helper,"canonical_points",quantize):
            with self.assertRaises(ValueError) as caught:
                helper.FreshCanonicalFeatures(np,o3d).build(points)
        self.assertIs(caught.exception,primary)
        self.assertIn("restoration",str(primary.__notes__))

    def test_policy_mutation_rejected_before_native_calls(self):
        np,o3d,points,calls,_,quantize = self.stand_ins()
        builder = helper.FreshCanonicalFeatures(np,o3d)
        builder.policy["normal_policy"] = "largest-component-positive"
        with patch.object(helper,"canonical_points",quantize):
            with self.assertRaises(helper.CanonicalProposalError):
                builder.build(points)
        self.assertEqual(calls,[])


class SourceAndManifestContracts(unittest.TestCase):
    def test_actual_pinned_original_whole_function_without_native_imports(self):
        path = helper.ROOT/"scanner_server/fragments.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        code = next(value for value in compile(tree,str(path),"exec").co_consts
            if isinstance(value,types.CodeType) and value.co_name == "_global_seed")
        original = types.FunctionType(code,{"__name__":"scanner_server.fragments"})
        self.assertEqual(helper.original_binding(original)["global_seed_ast_sha256"],helper.ORIGINAL_GLOBAL_AST_SHA256)
        altered = types.FunctionType(compile("def _global_seed(a,b,c): return None","changed","exec").co_consts[0],
            {"__name__":"scanner_server.fragments"})
        with self.assertRaises(helper.CanonicalProposalError):
            helper.original_binding(altered)

    def test_snapshot_headers_closed_hash_and_scope_preflight_stdlib(self):
        with tempfile.TemporaryDirectory() as folder:
            path,report,snapshot = diagnostic_fixture(folder)
            result,binding,descriptors = helper.load_input_manifest(path)
            self.assertEqual(len(descriptors),10)
            self.assertEqual(binding["report_sha256"],helper.file_hash(path))
            helper.validate_pair(result,result)
            snapshot.write_bytes(snapshot.read_bytes()+b"mutated")
            with self.assertRaises(helper.CanonicalProposalError):
                helper.load_input_manifest(path)

    def test_object_npy_and_duplicate_payloads_rejected_before_numerical_import(self):
        for dtype,duplicate in (("|O",False),("<f8",True)):
            with self.subTest(dtype=dtype,duplicate=duplicate),tempfile.TemporaryDirectory() as folder:
                path,_,_ = diagnostic_fixture(folder,dtype=dtype,duplicate=duplicate)
                with self.assertRaises(helper.CanonicalProposalError):
                    helper.load_input_manifest(path)
        with self.assertRaises(helper.CanonicalProposalError):
            helper.snapshot_header(io.BytesIO(b"not_npy"))

    def test_paired_membership_seed_shapes_and_source_pins_not_relaxed(self):
        with tempfile.TemporaryDirectory() as folder:
            _,report,_ = diagnostic_fixture(folder)
            for alter in (lambda row:row["rows"][0].update(original_seed=3),
                          lambda row:row["rows"][0]["inputs"]["source_coarse_points"].update(shape=[2,3]),
                          lambda row:row["artifacts_sha256"].update(producer="b"*64)):
                changed = copy.deepcopy(report)
                alter(changed)
                with self.assertRaises(helper.CanonicalProposalError):
                    helper.validate_pair(report,changed)

    def test_relocated_cli_help_uses_stdlib_only_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            completed = subprocess.run([sys.executable,"-S",str(helper.ROOT/"scripts/research/canonical_fpfh_proposals.py"),"--help"],
                cwd=folder,text=True,capture_output=True,timeout=10)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertIn("--normal-policy",completed.stdout)
            self.assertEqual(list(Path(folder).iterdir()),[])


if __name__ == "__main__":
    unittest.main()
