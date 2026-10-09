"""Separate owned-proof workspace: exact held lifecycle and source regressions."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from scripts.research import device_loop_owned_workspace as w
from tests.test_device_loop_workspace import template,workspace as old_fixture


def fixture():
    old=old_fixture()
    owner=w.DeviceLoopWorkspace.__new__(w.DeviceLoopWorkspace)
    owner.__dict__.update(old.__dict__)
    owner.provenance={"policy":w.POLICY}
    owner.runtime_binding=lambda:owner.runtime
    return owner


class DerivationTests(unittest.TestCase):
    def test_exact_inverse_only_two_new_registered_protocol_imports(self):
        result=w.derivation_contract()
        self.assertEqual(result["inverse_protocol_imports"],2)
        self.assertEqual(result["held_workspace_sha256"],w.OLD_SHA256)
        self.assertTrue(result["held_guards_clones_and_lifecycle_exact"])
        self.assertEqual(len(w.NEW_FILES),6)
        self.assertEqual(len(w.OLD_PINS),8)
        self.assertEqual(len(set(w.FILES)),14)
        self.assertTrue(all(w.sha(w.ROOT/name)==digest for name,digest in w.OLD_PINS.items()))

    def test_any_lifecycle_or_budget_relaxation_is_refused(self):
        source=Path(w.__file__).read_text(encoding="utf-8")
        for old,new in (("self.max_jobs=max_jobs","self.max_jobs=4096"),
                ("lane.close()","pass"),("self.template.close()","pass"),
                ("self.active=None","self.active=lane"),
                ("verify_snapshot(self.template,self.expected)","pass")):
            self.assertIn(old,source)
            with self.assertRaises(w.WorkspaceError):w.derivation_contract(source.replace(old,new))

    def test_clone_native_owner_or_buffer_changes_are_refused(self):
        source=Path(w.__file__).read_text(encoding="utf-8")
        for old,new in (("lane.buffers={}","lane.buffers=template.buffers"),
                ("lane.kernels=dict(template.kernels)","lane.kernels=template.kernels"),
                ("    pristine(template)\n","    pass\n")):
            with self.assertRaises(w.WorkspaceError):w.derivation_contract(source.replace(old,new))

    def test_old_or_foreign_protocol_cannot_replace_new_registry(self):
        source=Path(w.__file__).read_text(encoding="utf-8")
        for name in ("device_loop_workspace_protocol","gpu_icp_device_loop_protocol"):
            with self.assertRaises(w.WorkspaceError):w.derivation_contract(source.replace("microbatch_bridge_owned_protocol as protocol",name+" as protocol",1))

    def test_held_file_mutation_refused_before_derivation(self):
        with patch.object(w,"sha",return_value="f"*64):
            with self.assertRaises(w.WorkspaceError):w.derivation_contract()

    def test_fresh_import_has_no_numerical_imports_or_gpu_setup(self):
        code="from scripts.research import device_loop_owned_workspace; import sys; assert not {'numpy','cupy','open3d','cv2'} & set(sys.modules)"
        result=subprocess.run([sys.executable,"-S","-c",code],cwd=w.ROOT,text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


class OwnedLifecycleTests(unittest.TestCase):
    def test_exact_twentyseven_fields_and_no_previous_input_state(self):
        t=template();lane=w.fresh_clone(t)
        self.assertEqual(set(lane.__dict__),w.FIELDS)
        self.assertEqual(len(w.FIELDS),27)
        self.assertFalse(hasattr(lane,"input_binding"))
        self.assertFalse(lane.started);self.assertIsNone(lane.graph)
        for name in w.IDENTITY_FIELDS:self.assertIs(getattr(lane,name),getattr(t,name))
        for name in ("buffers","grid_owners","provenance","statistics","kernels"):
            self.assertIsNot(getattr(lane,name),getattr(t,name))

    def test_sequential_audit_clones_have_private_phase_and_graph(self):
        owner=fixture();first=owner.new_lane();first.buffers["phase"]=object();first.graph=object();first.started=True
        owner.close_lane(first);second=owner.new_lane()
        self.assertIsNot(first,second);self.assertFalse(second.started);self.assertEqual(second.buffers,{})
        self.assertIsNone(second.graph);self.assertFalse(owner.template.started)
        owner.close_lane(second);owner.close();self.assertTrue(owner.closed)

    def test_no_second_lane_before_selected_completion(self):
        owner=fixture();lane=owner.new_lane()
        with self.assertRaises(w.WorkspaceError):owner.new_lane()
        self.assertIs(owner.active,lane);self.assertIsNotNone(owner.failure)
        owner.close();self.assertTrue(owner.closed)

    def test_sync_fault_retains_numeric_and_template_owners(self):
        owner=fixture();lane=owner.new_lane();lane.sync_fault=True
        with self.assertRaises(RuntimeError):owner.close_lane(lane)
        self.assertFalse(owner.closed);self.assertIs(owner.active,lane)
        self.assertFalse(owner.template.closed);self.assertEqual(owner.retained_failed_lanes,[lane])
        with self.assertRaises(w.WorkspaceError):owner.new_lane()
        lane.sync_fault=False;owner.close();self.assertTrue(owner.closed)
        self.assertIsNotNone(owner.failure)

    def test_primary_failure_identity_survives_cleanup_failure(self):
        owner=fixture();lane=owner.new_lane();lane.sync_fault=True;primary=ValueError("original math")
        with self.assertRaises(ValueError) as caught:owner.close_lane(lane,primary=primary)
        self.assertIs(caught.exception,primary);self.assertIs(owner.failure,primary)
        self.assertIsInstance(primary.__cause__,RuntimeError)

    def test_mutated_shared_kernel_refuses_success_but_cleanup_is_allowed(self):
        owner=fixture();lane=owner.new_lane();lane.kernels["kernel"]=object()
        with self.assertRaises(w.WorkspaceError):owner.close_lane(lane)
        self.assertIsNotNone(owner.failure);self.assertIs(owner.active,lane)
        owner.close();self.assertTrue(owner.closed)

    def test_bound_job_inventory_unchanged(self):
        owner=fixture();owner.max_jobs=1;lane=owner.new_lane();owner.close_lane(lane)
        with self.assertRaises(w.WorkspaceError):owner.new_lane()
        self.assertEqual(len(owner.jobs),1)


if __name__=="__main__":unittest.main()
