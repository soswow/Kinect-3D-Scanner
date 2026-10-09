"""Stdlib-only candidate controller preflight, ownership and transaction tests."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research import profile_gpu_icp_finish_candidate as controller


class Clock:
    def __init__(self): self.value = 0
    def __call__(self): return self.value
    def add(self, value): self.value += value


def engine():
    final = {"applied": True, "voxel_m": .005, "allocation": "automatic",
        "allocation_strategy": "exact missing-key activation", "allocated_blocks": 3395,
        "requested_block_capacity": 3395, "initial_block_capacity": 3395,
        "required_blocks": 3328, "blocks": 3328}
    grid = SimpleNamespace(hashmap=lambda:SimpleNamespace(capacity=lambda:3395,size=lambda:3328))
    return SimpleNamespace(unprocessed_count=0, mesh=object(), _final_vbg=grid,
        settings=SimpleNamespace(confidence_fusion=True,final_voxel_m=.005,final_block_count=1),
        final_reconstruction=final)


class Context:
    def __init__(self, clock, *, fault=None, cleanup=None, report_fault=None):
        self.clock=clock;self.fault=fault;self.cleanup=cleanup;self.report_fault=report_fault
        self.closed=False;self.complete=False;self.builds=0
    def __enter__(self): self.clock.add(3);return self
    def build(self, owner, original_build, **kwargs):
        self.builds+=1
        if self.fault:raise self.fault
        return original_build(owner,**kwargs)
    def finish(self):self.clock.add(7);self.complete=True
    def __exit__(self,kind,error,trace):
        self.clock.add(11);self.closed=True
        if self.cleanup:raise self.cleanup
    def report(self):
        if self.report_fault:raise self.report_fault
        return {"complete":self.complete,"restored":True,"closed":self.closed,
            "failure":None,"cleanup_failures":[],"calls":[],"events":[],"graphs":[],
            "final_pose_inventory":[],"workspace":None,"cache_receipts":[]}
    def close(self,primary=None):self.closed=True


class Qualification:
    def __init__(self,clock):self.clock=clock;self.calls=[];self.token=object();self.path=None
    def qualify_candidate_method(self,path,binding,configuration=None):
        self.path=path;self.hash=controller.sha(path)
        self.clock.add(13);self.calls.append((path,binding,configuration));return self.token
    def qualification_receipt(self,token,close=False):
        assert token is self.token
        return {"required":True,"closed":close,"failure":None,"report_path":str(self.path),
            "report_sha256":self.hash,"report_sha256_after":controller.sha(self.path),
            "historical_trajectory_authority":False,"general_domain_authority":False}


class CandidateProfileContracts(unittest.TestCase):
    def args(self, root, mode="audit", **changes):
        values={"mode":mode,"run_allocated":True,"session":root/"raw.zip",
            "checkpoint":root/"live/checkpoint.json","checkpoint_directory":None,
            "output":root/"benchmark-output/result.json","candidate_proof":None,"final_block_count":None}
        values.update(changes);return argparse.Namespace(**values)
    def call(self, mode="audit", *, owner=None, clock=None, context=None, build=None,
             factory=None, completion=None, proof=None, qualification=None):
        owner=owner or engine();clock=clock or Clock();context=context or Context(clock)
        def factory_default(*args):clock.add(2);return context
        def build_default(value,**kwargs):clock.add(5);return True,{"success":True}
        def forbidden(*args,**kwargs):raise AssertionError("old permit/scope uncalled")
        with patch.object(controller.original,"source_hash",return_value="a"*64):
            result=controller.execute_candidate_finish(owner,mode=mode,scope_factory=forbidden,
                original_build=build or build_default,protocol=SimpleNamespace(validate_finish_audit=forbidden),
                binding={"source_sha256":"a"*64},scope_binding={},audit_path=proof,
                candidate_factory=factory or factory_default,qualification_module=qualification or Qualification(clock),
                configuration={"graph":True,"chunk_iterations":4},clock=clock,
                completion=completion or (lambda:clock.add(17)))
        return result,context,clock

    def test_own_mode_proof_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            controller.validate_args(self.args(root),root=root,idle=lambda:True)
            controller.validate_args(self.args(root,"measure",candidate_proof=root/"audit.json"),root=root)
            for args in (self.args(root,"measure"),self.args(root,"shadow",candidate_proof=root/"audit.json"),
                         self.args(root,"native",candidate_proof=root/"audit.json")):
                with self.assertRaises(controller.original.FinishProfileFailure):controller.validate_args(args,root=root)
    def test_fresh_allocated_idle_private_original_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for changes in ({"run_allocated":False},{"checkpoint":None},{"output":root/"public.json"}):
                with self.assertRaises(controller.original.FinishProfileFailure):
                    controller.validate_args(self.args(root,**changes),root=root)
            with self.assertRaises(controller.original.FinishProfileFailure):
                controller.validate_args(self.args(root),root=root,idle=lambda:False)
            args=self.args(root);args.output.parent.mkdir();args.output.write_text("preserved")
            with self.assertRaises(controller.original.FinishProfileFailure):controller.validate_args(args,root=root)
            self.assertEqual(args.output.read_text(),"preserved")
    def test_current_capture_requires_new_directory_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);live=root/"benchmark-output/live"
            args=self.args(root,"capture",checkpoint=None,checkpoint_directory=live)
            controller.validate_args(args,root=root)
            live.mkdir(parents=True)
            with self.assertRaises(controller.original.FinishProfileFailure):controller.validate_args(args,root=root)
    def test_main_inverse_exact_and_original_module_slots_untouched(self):
        held=(controller.original.main,controller.original.execute_finish,controller.original.CURRENT)
        changed=controller.derivation_contract();proof=controller.derivation_contract(changed)
        self.assertEqual(proof["inverse_final_call_count"],1)
        self.assertEqual(held,(controller.original.main,controller.original.execute_finish,controller.original.CURRENT))
        changed.body.insert(0,ast.Pass())
        with self.assertRaises(controller.original.FinishProfileFailure):controller.derivation_contract(changed)
    def test_own_checkpoint_family_does_not_repin_old_producer(self):
        self.assertNotEqual(controller.CHECKPOINT_FILES,controller.original.CHECKPOINT_FILES)
        self.assertIn("scripts/research/profile_gpu_icp_finish_candidate.py",controller.CHECKPOINT_FILES)
        self.assertNotIn("scripts/research/profile_gpu_icp_finish_candidate.py",controller.original.CHECKPOINT_FILES)
        self.assertIn("scripts/research/private_live_checkpoint.py",controller.CHECKPOINT_FILES)
    def test_portable_current_digest_is_private_and_not_a_historical_repin(self):
        original_digest=controller.original.CURRENT
        derived=controller.derive_main()
        self.assertEqual(derived.__globals__["CURRENT"],controller.original.source_hash())
        self.assertEqual(controller.original.CURRENT,original_digest)
        self.assertIs(derived.__globals__["CHECKPOINT_FILES"],controller.CHECKPOINT_FILES)
        self.assertTrue(set(controller.CANDIDATE_FILES) <= set(derived.__globals__["FAMILY_FILES"]))
        self.assertEqual(derived.__globals__["__name__"],controller.original.__name__)
    def test_late_derived_global_mutation_writes_failed_evidence(self):
        derived=controller.derive_main()
        save=derived.__globals__["save_report"]
        derived.__globals__["CURRENT"]="f"*64
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"failed.json"
            with self.assertRaisesRegex(controller.original.FinishProfileFailure,"Derived candidate"):
                save(path,{"mode":"audit","status":"passed","failure":None})
            self.assertIn('"status": "failed"',path.read_text(encoding="utf-8"))
    def test_measure_charges_qualification_constructor_build_exit_and_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            proof=Path(folder)/"audit.json";proof.write_text("closed method audit")
            clock=Clock();qual=Qualification(clock)
            result,context,_=self.call("measure",clock=clock,proof=proof,qualification=qual)
            self.assertEqual(result["finish_s"],13+2+3+5+7+11+17)
            self.assertEqual(context.builds,1)
            self.assertEqual(len(qual.calls),1)
            self.assertTrue(result["method_qualification"]["required"])
            self.assertEqual(result["method_qualification"]["report_sha256"],
                             result["method_qualification"]["report_sha256_after"])
            self.assertFalse(result["owned_proof"]["required"])
    def test_audit_and_shadow_do_not_consume_qualification(self):
        for mode in ("audit","shadow"):
            clock=Clock();qual=Qualification(clock)
            result,context,_=self.call(mode,clock=clock,qualification=qual)
            self.assertEqual(qual.calls,[])
            self.assertFalse(result["method_qualification"]["required"])
            self.assertTrue(result["registration"]["closed"])
    def test_native_is_original_direct_build_without_context(self):
        def forbidden(*args):raise AssertionError("native candidate context")
        result,context,_=self.call("native",factory=forbidden)
        self.assertEqual(context.builds,0)
        self.assertFalse(result["registration"]["registration_evidence_available"])
        self.assertEqual(result["finish_s"],5+17)
    def test_current_auto_capacity_is_not_legacy_settings_ceiling(self):
        owner=engine();self.assertEqual(owner.settings.final_block_count,1)
        controller.validate_final(owner,(True,{}))
        owner._final_vbg=SimpleNamespace(hashmap=lambda:SimpleNamespace(capacity=lambda:3396,size=lambda:3328))
        with self.assertRaises(controller.original.FinishProfileFailure):controller.validate_final(owner,(True,{}))
    def test_false_original_build_and_uncommitted_mesh_rejected(self):
        with self.assertRaises(controller.original.FinishProfileFailure) as caught:
            self.call(build=lambda *args,**kwargs:(False,{"message":"original stage"}))
        self.assertEqual(caught.exception.finish_profile_record["built"][1]["message"],"original stage")
        owner=engine();owner.mesh=None
        with self.assertRaises(controller.original.FinishProfileFailure):self.call(owner=owner)
    def test_primary_fault_survives_completion_and_report_faults(self):
        clock=Clock();primary=ValueError("actual candidate")
        context=Context(clock,fault=primary,report_fault=RuntimeError("late report"))
        def complete():raise RuntimeError("completion")
        with self.assertRaises(ValueError) as caught:
            self.call(clock=clock,context=context,completion=complete)
        self.assertIs(caught.exception,primary)
        self.assertEqual(len(primary.finish_profile_record["finish_cleanup_failures"]),2)
    def test_failed_enter_still_closes_retained_context(self):
        clock=Clock();fault=ValueError("partial entry")
        class EnterFailure(Context):
            def __enter__(self):raise fault
        context=EnterFailure(clock)
        with self.assertRaises(ValueError) as caught:self.call(clock=clock,context=context)
        self.assertIs(caught.exception,fault)
        self.assertTrue(context.closed)
        self.assertEqual(context.builds,0)
    def test_partial_retrieval_constructor_owner_remains_retained(self):
        fault=ValueError("partial retrieval")
        class Partial:
            def __init__(self,**kwargs):
                self.numeric_owner=object();raise fault
        module=ModuleType("scripts.research.cuda_device_flat_grid_registration")
        module.DeviceFlatGridICP=Partial
        with patch.dict(sys.modules,{module.__name__:module}):
            with self.assertRaises(ValueError) as caught:controller.make_retrieval()
        self.assertIs(caught.exception,fault)
        self.assertIsInstance(fault.candidate_retrieval,Partial)
        self.assertIsNotNone(fault.candidate_retrieval.numeric_owner)
    def test_method_audit_replacement_refuses_measure(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"audit.json";path.write_text("original")
            clock=Clock();context=Context(clock)
            def build(*args,**kwargs):path.write_text("changed");return True,{}
            with self.assertRaises(controller.original.FinishProfileFailure) as caught:
                self.call("measure",clock=clock,context=context,proof=path,build=build)
            self.assertNotEqual(caught.exception.finish_profile_record["method_qualification"]["report_sha256"],
                                caught.exception.finish_profile_record["method_qualification"]["report_sha256_after"])
    def test_incomplete_context_refuses_after_successful_build(self):
        clock=Clock();context=Context(clock)
        context.finish=lambda:None
        with self.assertRaises(controller.original.FinishProfileFailure):self.call(clock=clock,context=context)
    def test_bundle_slot_owners_remain_strict_without_gate_observers(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bundle.py";path.write_text("def propose_bundle_poses():\n    return 1\n")
            module=ModuleType("candidate_bundle_test");module.__file__=str(path)
            exec(compile(path.read_text(),str(path),"exec",dont_inherit=True),module.__dict__)
            with patch.dict(sys.modules,{module.__name__:module}):
                owners=controller.unobserved_loaded_owners((module,),(module,),
                    observed_slots=((module,"propose_bundle_poses"),))
                module.propose_bundle_poses=lambda:2
                with self.assertRaises(controller.original.FinishProfileFailure):owners.check()
    def test_current_allocator_imported_body_is_independently_owned(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"fusion_memory.py";path.write_text("def plan_fusion(device, blocks, voxel_m):\n    return blocks\n")
            module=ModuleType("scanner_server.fusion_memory");module.__file__=str(path)
            exec(compile(path.read_text(),str(path),"exec",dont_inherit=True),module.__dict__)
            with patch.dict(sys.modules,{module.__name__:module}):
                owners=controller.unobserved_loaded_owners((),())
                module.plan_fusion.__code__=(lambda device,blocks,voxel_m:blocks+1).__code__
                with self.assertRaises(controller.original.FinishProfileFailure):owners.check()
    def test_original_terminal_exit_stays_last(self):
        changed=controller.derivation_contract()
        self.assertEqual(changed.body[-1].value.func.id,"finish_cuda_worker")
    def test_cli_has_no_archived_pose_or_old_proof_options(self):
        tree=ast.parse(Path(controller.__file__).read_text(encoding="utf-8"))
        options={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=="add_argument"
            and n.args and isinstance(n.args[0],ast.Constant)}
        self.assertFalse(options & {"--finish-audit","--quality-proof","--use-pose-seeds","--limit","--stride"})
        self.assertTrue({"--candidate-proof","--checkpoint-directory","--mode"} <= options)


if __name__ == "__main__":
    unittest.main()
