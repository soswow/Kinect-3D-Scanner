"""Stdlib-only exact-clone ownership and completion fault regressions."""
import copy
from pathlib import Path
from types import SimpleNamespace,FunctionType,ModuleType
import unittest
from unittest.mock import patch
from scripts.research import device_loop_workspace as w
from scripts.research import device_loop_workspace_protocol as p


class FakeLoop:
    def close(self):
        if getattr(self,"sync_fault",False):raise RuntimeError("completion")
        self.buffers.clear();self.grid_owners=[];self.graph=None;self.lease=None;self.closed=True
    def report(self):return {"closed":self.closed,"failure":None if self.failure is None else repr(self.failure)}


def template():
    t=FakeLoop()
    for name in w.IDENTITY_FIELDS:setattr(t,name,object())
    values={"device":0,"max_points":1_000_000,"max_scratch_bytes":256*1024**2,"max_total_bytes":512*1024**2,
        "audit_nearest":True,"audit_misses":True,"cuda_graph":True,"configuration":(0,1_000_000,256*1024**2,512*1024**2,True,True,True),
        "authorizer":None,"audit_observer":None,"buffers":{},"lease":None,"failure":None,"grid_owners":[],
        "graph":None,"graph_chunk":None,"pending_host_packets":None,"started":False,"closed":False,
        "provenance":{"nested":{"value":[1,2]}},"statistics":{"queries":0,"updates":0,"setup_s":.2,"graph_capture_s":0.},
        "kernels":{"kernel":object()}}
    for name,value in values.items():setattr(t,name,value)
    return t


def workspace():
    result=w.DeviceLoopWorkspace.__new__(w.DeviceLoopWorkspace)
    result.template=template();result.expected=w.snapshot(result.template)
    result.closed=False;result.failure=None;result.active=None;result.max_jobs=3
    result.jobs=[];result.retained_failed_lanes=[];result.provenance={"policy":w.POLICY}
    result.statistics={"cold_setup_wall_s":.3,"clone_guard_wall_s":0.,"lane_close_wall_s":0.,"workspace_close_wall_s":0.}
    result.guard=SimpleNamespace(check=lambda:None);result.file_pins={};result.runtime={"version":1}
    result.runtime_binding=lambda:result.runtime
    result.timing_permit=None
    return result


class CloneTests(unittest.TestCase):
    def test_exact_constructor_inventory_is_source_bound(self):
        value=w.source_contract()
        self.assertEqual(set(value["constructor_fields"]),w.FIELDS)
        self.assertEqual(value["held_loop_sha256"],w.LOOP_SHA256)
        self.assertTrue(value["registered_timing_supported"])

    def test_clone_shares_only_code_and_completed_stream(self):
        t=template();a=w.fresh_clone(t);b=w.fresh_clone(t)
        for name in w.IDENTITY_FIELDS:self.assertIs(getattr(a,name),getattr(t,name))
        self.assertIs(a.kernels["kernel"],t.kernels["kernel"])
        for name in ("buffers","grid_owners","provenance","statistics","kernels"):
            self.assertIsNot(getattr(a,name),getattr(t,name));self.assertIsNot(getattr(a,name),getattr(b,name))
        self.assertFalse(a.started);self.assertIsNone(a.graph);self.assertEqual(a.statistics["setup_s"],0.)
        a.provenance["nested"]["value"].append(3)
        self.assertEqual(t.provenance["nested"]["value"],[1,2])

    def test_started_or_extra_template_field_refused(self):
        for name,value in (("started",True),("input_binding",{}),("graph",object()),("failure",RuntimeError("math"))):
            t=template();setattr(t,name,value)
            with self.assertRaises(w.WorkspaceError):w.fresh_clone(t)

    def test_template_shared_owner_or_kernel_replacement_refused(self):
        for name in ("stream","module","np"):
            t=template();saved=w.snapshot(t);setattr(t,name,object())
            with self.assertRaises(w.WorkspaceError):w.verify_snapshot(t,saved)
        t=template();saved=w.snapshot(t);t.kernels["kernel"]=object()
        with self.assertRaises(w.WorkspaceError):w.verify_snapshot(t,saved)

    def test_mutated_or_replaced_template_container_refused(self):
        t=template();saved=w.snapshot(t);t.provenance["nested"]["value"].append(3)
        with self.assertRaises(w.WorkspaceError):w.verify_snapshot(t,saved)
        t=template();saved=w.snapshot(t);t.buffers={}
        with self.assertRaises(w.WorkspaceError):w.verify_snapshot(t,saved)

    def test_timing_clone_requires_separate_call_authorizer(self):
        t=template();t.audit_nearest=t.audit_misses=False;t.authorizer=lambda *_:None
        with self.assertRaises(w.WorkspaceError):w.fresh_clone(t)
        fn=lambda *_:True;lane=w.fresh_clone(t,timing_authorizer=fn)
        self.assertIs(lane.authorizer,fn)


class WorkspaceTests(unittest.TestCase):
    def test_one_lane_at_a_time_then_fresh_buffers_and_graph(self):
        owner=workspace();a=owner.new_lane();a.buffers["phase"]=object();a.graph=object()
        with self.assertRaises(w.WorkspaceError):owner.new_lane()
        self.assertIsNotNone(owner.failure)
        owner.close_lane(a)
        self.assertIsNone(owner.active)
        with self.assertRaises(w.WorkspaceError):owner.new_lane()

    def test_successive_lanes_never_reuse_previous_phase_or_input_owners(self):
        owner=workspace();a=owner.new_lane();owner.check_lane(a)
        a.buffers["phase"]=object();a.input_binding={"old":1};a.started=True
        owner.close_lane(a);b=owner.new_lane()
        self.assertIsNot(a,b);self.assertEqual(b.buffers,{})
        self.assertFalse(hasattr(b,"input_binding"));self.assertIsNone(b.graph)
        self.assertEqual(b.statistics["queries"],0);self.assertFalse(owner.template.started)
        owner.close_lane(b);owner.close();self.assertTrue(owner.closed)

    def test_selected_completion_failure_retains_lane_and_template(self):
        owner=workspace();lane=owner.new_lane();lane.sync_fault=True
        with self.assertRaises(RuntimeError):owner.close_lane(lane)
        self.assertIs(owner.active,lane);self.assertEqual(owner.retained_failed_lanes,[lane])
        self.assertFalse(owner.closed);self.assertFalse(owner.template.closed)
        with self.assertRaises(w.WorkspaceError):owner.new_lane()
        lane.sync_fault=False;owner.close()
        self.assertTrue(owner.closed);self.assertIsNotNone(owner.failure)

    def test_primary_math_fault_retained_when_completion_also_fails(self):
        owner=workspace();lane=owner.new_lane();lane.sync_fault=True
        primary=ValueError("math")
        with self.assertRaises(ValueError) as caught:owner.close_lane(lane,primary=primary)
        self.assertIs(caught.exception,primary);self.assertIs(owner.failure,primary)
        self.assertIsInstance(primary.__cause__,RuntimeError)
        self.assertIs(owner.active,lane)

    def test_lane_shared_kernel_mutation_is_sticky_then_cleanup_allowed(self):
        owner=workspace();lane=owner.new_lane();lane.kernels["kernel"]=object()
        with self.assertRaises(w.WorkspaceError):owner.close_lane(lane)
        self.assertIsNotNone(owner.failure);self.assertIs(owner.active,lane)
        owner.close();self.assertTrue(owner.closed)

    def test_foreign_lane_cannot_close_owned_lane(self):
        owner=workspace();a=owner.new_lane()
        with self.assertRaises(w.WorkspaceError):owner.close_lane(w.fresh_clone(owner.template))
        self.assertIs(owner.active,a);owner.close()

    def test_job_cap_prevents_unbounded_completed_inventory(self):
        owner=workspace();owner.max_jobs=1;a=owner.new_lane();owner.close_lane(a)
        with self.assertRaises(w.WorkspaceError):owner.new_lane()
        self.assertEqual(len(owner.jobs),1)

    def test_changed_constructor_runtime_refuses_new_job(self):
        owner=workspace();owner.runtime_binding=lambda:{"version":2}
        with self.assertRaises(w.WorkspaceError):owner.new_lane()

    def test_old_constructed_or_foreign_permits_refused(self):
        for value in (None,SimpleNamespace(report_sha256="a"*64),p.WorkspaceTimingPermit("missing","a"*64,"{}","{}","[]")):
            with self.assertRaises(ValueError):p.registered(value)

    def test_identical_code_with_foreign_globals_cannot_masquerade_as_original(self):
        module=ModuleType("workspace_fake_owner")
        exec("def original(): return 1",module.__dict__)
        guard=w.SourceOwnerGuard.__new__(w.SourceOwnerGuard);guard.module=module;guard.functions=[]
        original=module.original
        guard.add(module,"original",original,original.__code__)
        counterfeit=FunctionType(original.__code__,dict(module.__dict__))
        with self.assertRaises(w.WorkspaceError):guard.add(module,"original",counterfeit,original.__code__)


if __name__=="__main__":unittest.main()
