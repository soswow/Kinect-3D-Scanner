"""Stdlib-only transaction and fresh-scope contracts; no GPU/native imports."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research import profile_gpu_icp_finish as producer


class Clock:
    def __init__(self):self.value=0
    def __call__(self):return self.value
    def add(self,value):self.value+=value


class FakeScope:
    def __init__(self,clock,*,latch=False,cleanup_fault=False,ordinary=None):
        self.clock=clock;self.complete=False;self.restored=False;self.closed=False
        self.failure=None;self.latch=latch;self.cleanup_fault=cleanup_fault;self.ordinary=ordinary
    def __enter__(self):self.clock.add(3);return self
    def build(self,engine,original,*args,**kwargs):
        self.clock.add(5)
        value=original(engine,**kwargs)
        if self.latch:self.failure=producer.FinishProfileFailure("stage observer failed")
        if self.ordinary:raise self.ordinary
        return value
    def finish(self):
        self.clock.add(7)
        if self.failure:raise self.failure
        self.complete=True
    def __exit__(self,kind,error,trace):
        self.clock.add(11);self.restored=True;self.closed=True
        if self.cleanup_fault:raise ValueError("scope cleanup")
    def report(self):
        return {"complete":self.complete,"restored":self.restored,"closed":self.closed,
            "failure":None if self.failure is None else str(self.failure),"cleanup_failures":[]}


class FakeProtocol:
    def __init__(self,clock,*,fault=False):self.clock=clock;self.closed=False;self.fault=fault;self.token=object()
    def validate_finish_audit(self,*args,**kwargs):self.clock.add(13);return self.token
    def close_permit(self,token,primary):
        assert token is self.token
        self.clock.add(17);self.closed=True
        if self.fault:raise ValueError("proof cleanup")
    def permit_report(self,token):
        return {"required":token is not None,"closed":self.closed,"failure":None,"cleanup_failures":[]}


class FinishProfileContracts(unittest.TestCase):
    def test_fixed_file_string_paths_have_real_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"resource";path.write_bytes(b"native identity")
            self.assertEqual(producer.file_hash(str(path)),producer.file_hash(path))
    def args(self,root,mode="native",**overrides):
        value={"run_allocated":True,"mode":mode,"final_block_count":None,
            "checkpoint":root/"live/checkpoint.json","checkpoint_directory":None,
            "finish_audit":None,"quality_proof":None,"output":root/"benchmark-output/result.json"}
        value.update(overrides);return argparse.Namespace(**value)
    def test_freshness_precedes_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root);args.output.parent.mkdir();args.output.write_text("preserve")
            with self.assertRaisesRegex(producer.FinishProfileFailure,"fresh outputs"):
                producer.preflight(args,root=root,idle=lambda:True)
            self.assertEqual(args.output.read_text(),"preserve")
    def test_geometry_and_trace_are_also_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root);args.output.parent.mkdir()
            for suffix in (".geometry.npz",".trace.jsonl"):
                path=args.output.with_suffix(suffix);path.write_text("existing")
                with self.assertRaises(producer.FinishProfileFailure):producer.preflight(args,root=root)
                path.unlink()
    def test_require_allocated_idle_server_and_private_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root)
            self.assertEqual(producer.preflight(args,root=root,idle=lambda:True),args.output)
            for change in ({"run_allocated":False},{"checkpoint":None},{"output":root/"public.json"}):
                with self.assertRaises(producer.FinishProfileFailure):producer.preflight(self.args(root,**change),root=root)
            with self.assertRaises(producer.FinishProfileFailure):producer.preflight(args,root=root,idle=lambda:False)
    def test_capture_cannot_materialize_or_reuse_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/"benchmark-output/checkpoint"
            args=self.args(root,"capture",checkpoint=None,checkpoint_directory=folder)
            producer.preflight(args,root=root)
            folder.mkdir(parents=True)
            with self.assertRaises(producer.FinishProfileFailure):producer.preflight(args,root=root)
            with self.assertRaises(producer.FinishProfileFailure):
                producer.preflight(self.args(root,"capture",checkpoint_directory=folder),root=root)
    def test_timing_only_has_audit_and_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=self.args(root,"timing",finish_audit=root/"audit.json",quality_proof=root/"quality.json")
            producer.preflight(args,root=root)
            for changes in ({"finish_audit":None},{"quality_proof":None},{"mode":"audit"}):
                altered=vars(args)|changes
                with self.assertRaises(producer.FinishProfileFailure):producer.preflight(argparse.Namespace(**altered),root=root)
    def test_no_selection_or_pose_seed_cli(self):
        tree=ast.parse(Path(producer.__file__).read_text(encoding="utf-8"))
        options={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr=="add_argument" and n.args and isinstance(n.args[0],ast.Constant)}
        self.assertFalse(options&{"--limit","--stride","--use-pose-seeds","--component-synthetic","--component-bridge"})
    def test_bundle_is_observed_and_not_disabled(self):
        tree=ast.parse(Path(producer.__file__).read_text(encoding="utf-8"))
        factories=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr=="GpuICPFinishScope"]
        self.assertEqual(len(factories),1)
        keyword=next(k for k in factories[0].keywords if k.arg=="bundle_module")
        self.assertIsInstance(keyword.value,ast.Name);self.assertEqual(keyword.value.id,"bundle_adjustment")
        overrides=[k for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id=="replace" for k in n.keywords]
        self.assertNotIn("bundle_adjustment",[k.arg for k in overrides])
    def test_finish_charges_permit_factories_scope_and_cleanup(self):
        clock=Clock();protocol=FakeProtocol(clock);scope=FakeScope(clock)
        def factory(token):self.assertIs(token,protocol.token);clock.add(19);return scope
        result=producer.execute_finish(object(),mode="timing",scope_factory=factory,
            original_build=lambda engine,**kw:(True,{"message":"mesh"}),protocol=protocol,binding={},scope_binding={},clock=clock)
        self.assertEqual(result["finish_s"],75)
        self.assertTrue(result["registration"]["closed"])
        self.assertTrue(protocol.closed)
    def test_native_has_no_permit_but_full_scope_cleanup(self):
        clock=Clock();protocol=FakeProtocol(clock);scope=FakeScope(clock)
        result=producer.execute_finish(object(),mode="native",scope_factory=lambda token:scope,
            original_build=lambda engine,**kw:(True,{}),protocol=protocol,binding={},scope_binding={},clock=clock)
        self.assertEqual(result["finish_s"],26);self.assertFalse(protocol.closed)
    def test_swallowed_engine_stage_fault_is_sticky(self):
        clock=Clock();scope=FakeScope(clock,latch=True);protocol=FakeProtocol(clock)
        def swallowed(engine,**kw):return True,{"message":"core retained poses"}
        with self.assertRaisesRegex(producer.FinishProfileFailure,"observer") as caught:
            producer.execute_finish(object(),mode="audit",scope_factory=lambda token:scope,original_build=swallowed,
                protocol=protocol,binding={},scope_binding={},clock=clock)
        self.assertTrue(scope.closed);self.assertTrue(scope.restored)
        self.assertEqual(caught.exception.finish_profile_record["finish_s"],26)
    def test_primary_survives_proof_cleanup_failure(self):
        clock=Clock();primary=producer.FinishProfileFailure("actual numerical failure")
        scope=FakeScope(clock,ordinary=primary);protocol=FakeProtocol(clock,fault=True)
        with self.assertRaises(producer.FinishProfileFailure) as caught:
            producer.execute_finish(object(),mode="timing",scope_factory=lambda token:scope,
                original_build=lambda engine,**kw:(True,{}),protocol=protocol,binding={},scope_binding={},clock=clock)
        self.assertIs(caught.exception,primary);self.assertTrue(protocol.closed)
        self.assertEqual(caught.exception.finish_profile_record["finish_cleanup_failures"][0]["action"],"owned Finish proof close")
    def test_incomplete_scope_report_is_refused(self):
        clock=Clock();scope=FakeScope(clock);scope.report=lambda:{"complete":True,"restored":False,"closed":True,"failure":None,"cleanup_failures":[]}
        with self.assertRaisesRegex(producer.FinishProfileFailure,"restored"):
            producer.execute_finish(object(),mode="native",scope_factory=lambda token:scope,
                original_build=lambda engine,**kw:(True,{}),protocol=FakeProtocol(clock),binding={},scope_binding={},clock=clock)
    def test_failed_build_not_mesh_authority(self):
        clock=Clock();scope=FakeScope(clock)
        with self.assertRaisesRegex(producer.FinishProfileFailure,"mesh build failed"):
            producer.execute_finish(object(),mode="native",scope_factory=lambda token:scope,
                original_build=lambda engine,**kw:(False,{"message":"memory cap"}),protocol=FakeProtocol(clock),binding={},scope_binding={},clock=clock)
    def test_failed_write_preserves_primary(self):
        primary=producer.FinishProfileFailure("CPU shadow failed")
        with patch.object(Path,"write_text",side_effect=OSError("disk")):
            with self.assertRaises(producer.FinishProfileFailure) as caught:producer.save_report(Path("unused"),{},primary)
        self.assertIs(caught.exception,primary);self.assertIsInstance(primary.__cause__,OSError)
    def test_independent_cleanup_attempts_preserve_primary(self):
        failures=[];primary=ValueError("original");calls=[]
        producer.clean("first",lambda:(_ for _ in ()).throw(OSError("first")),failures,primary)
        producer.clean("second",lambda:calls.append("second"),failures,primary)
        self.assertEqual(calls,["second"]);self.assertEqual(len(failures),1)
    def test_scope_identity_is_detached_and_rejects_seeded_live(self):
        base={"archive":{"sha256":"a"},"settings":{"voxel_m":.01}}
        manifest={"scope_base":base,"fresh_live":{"unprocessed_count":0,"pose_source":"fresh raw Live replay"},"logical_state_sha256":"b"}
        row=producer.scope_identity(manifest,Path("checkpoint"),"c",base)
        base["settings"]["voxel_m"]=1;self.assertEqual(row["settings"]["voxel_m"],.01)
        manifest["fresh_live"]["pose_source"]="archived";
        with self.assertRaises(producer.FinishProfileFailure):producer.scope_identity(manifest,Path("checkpoint"),"c",base)
    def test_loaded_code_and_defaults_mutation(self):
        module=ModuleType("fake_whole_finish_owner")
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"owner.py";source="def check(value=1):\n    return value\n";path.write_text(source)
            module.__file__=str(path);exec(compile(source,str(path),"exec",dont_inherit=True),module.__dict__)
            sys.modules[module.__name__]=module
            try:
                guard=producer.LoadedOwners((module,),(module,));guard.check()
                module.check.__defaults__=(2,)
                with self.assertRaises(producer.FinishProfileFailure):guard.check()
            finally:sys.modules.pop(module.__name__,None)
    def test_original_contextmanager_and_property_are_source_checked(self):
        module=ModuleType("fake_decorated_finish_owner")
        source="from contextlib import contextmanager\nclass Engine:\n    @contextmanager\n    def stage(self):\n        yield 1\n    @property\n    def pending(self):\n        return 0\n"
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"owner.py";path.write_text(source);module.__file__=str(path)
            exec(compile(source,str(path),"exec",dont_inherit=True),module.__dict__);sys.modules[module.__name__]=module
            try:
                guard=producer.LoadedOwners((module,),(module,));guard.check()
                self.assertTrue(any(name.endswith("contextlib.py") for name in guard.used_sources))
                module.Engine.stage.__wrapped__=lambda self:None
                with self.assertRaisesRegex(producer.FinishProfileFailure,"decorator"):guard.check()
            finally:sys.modules.pop(module.__name__,None)
    def test_cold_modified_original_class_body_is_refused(self):
        module=ModuleType("fake_mutated_finish_owner");source="class Engine:\n    def build(self):\n        return True\n"
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"owner.py";path.write_text(source);module.__file__=str(path)
            exec(compile(source,str(path),"exec",dont_inherit=True),module.__dict__);sys.modules[module.__name__]=module
            try:
                module.Engine.build.__code__=(lambda self:False).__code__
                with self.assertRaisesRegex(producer.FinishProfileFailure,"differs"):producer.LoadedOwners((module,),(module,))
            finally:sys.modules.pop(module.__name__,None)
    def test_actual_bundle_dataclass_generated_methods_need_no_native_import(self):
        import builtins
        module=ModuleType("fake_actual_bundle_source")
        path=ROOT/"scanner_server/bundle_adjustment.py";module.__file__=str(path)
        native=SimpleNamespace(ndarray=object,prepare_rgbd=None,RGB_DEPTH_ASSISTANCE_LIMIT_MS=0,
            Features=object,correspondences=None,extract_features=None,propose_transform=None)
        imported=[]
        def stub_import(name,globals=None,locals=None,fromlist=(),level=0):
            if name in ("dataclasses","time"):return builtins.__import__(name,globals,locals,fromlist,level)
            imported.append(name);return native
        module.__dict__["__builtins__"]=dict(vars(builtins),__import__=stub_import)
        sys.modules[module.__name__]=module
        try:
            exec(compile(path.read_text(encoding="utf-8"),str(path),"exec",dont_inherit=True),module.__dict__)
            guard=producer.LoadedOwners((module,),(module,));guard.check()
            self.assertTrue(callable(module.BundleProblem.__init__))
            self.assertEqual(module.BundleProblem.__dataclass_fields__.keys(),{"poses","landmarks","cameras","tracks","pixels","depths"})
            self.assertIn("numpy",imported)
            module._notify=lambda *args:None
            with self.assertRaises(producer.FinishProfileFailure):guard.check()
        finally:sys.modules.pop(module.__name__,None)
    def test_declared_observer_slot_keeps_body_and_imported_alias_checks(self):
        module=ModuleType("fake_observed_bundle_owner")
        source="dependency=object()\ndef gate(value):\n    return dependency\n"
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"owner.py";path.write_text(source);module.__file__=str(path)
            exec(compile(source,str(path),"exec",dont_inherit=True),module.__dict__);sys.modules[module.__name__]=module
            try:
                original=module.gate
                guard=producer.LoadedOwners((module,),(module,),observed_slots=((module,"gate"),))
                module.gate=lambda value:original(value)
                guard.check() # The actual observer wrapper belongs to the scope.
                module.dependency=object()
                with self.assertRaisesRegex(producer.FinishProfileFailure,"helper owner"):guard.check()
            finally:sys.modules.pop(module.__name__,None)
    def test_terminal_worker_is_after_closed_save_and_has_no_later_work(self):
        tree=ast.parse(Path(producer.__file__).read_text(encoding="utf-8"))
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        self.assertIsInstance(main.body[-1],ast.Expr)
        self.assertEqual(main.body[-1].value.func.id,"finish_cuda_worker")
        self.assertTrue(any(isinstance(n,ast.Try) and any(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call)
            and getattr(x.value.func,"id",None)=="save_report" for x in n.finalbody) for n in main.body))


if __name__=="__main__":unittest.main()
