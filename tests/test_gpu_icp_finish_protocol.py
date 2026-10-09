"""Artificial complete-Finish receipts; no numerical or hardware evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.research import gpu_icp_finish_protocol as p
from scripts.research import device_loop_finish_workspace as workspace
from scripts.research import gpu_icp_finish_scope as injection
from tests.test_gpu_icp_device_loop_protocol import fixture as loop_fixture


def full(value):
    result=copy.deepcopy(value);result["nbytes"]=8*result["shape"][0]*result["shape"][1]
    return result


def fixture():
    old,binding=loop_fixture()
    contract={"policy":workspace.POLICY,"artifacts":{"scripts/fake-workspace.py":"a"*64}}
    proof={"policy":p.POLICY,"artifacts":{"scripts/fake-protocol.py":"b"*64}}
    scope_contract={"policy":injection.POLICY,"artifacts":{"scripts/fake-scope.py":"d"*64}}
    binding.update(workspace_source=contract,finish_proof_source=proof,finish_scope_source=scope_contract,
        configuration={"graph":True,"chunk_iterations":4})
    binding["artifacts_sha256"].update(dict(contract["artifacts"],**proof["artifacts"],**scope_contract["artifacts"]))
    common=old["rows"][0];consumed=common["input_binding"]
    common["source_binding"]["setup_reuse"]=contract
    full_input={"source":{k:full(consumed["source"]) for k in ("points","normals","colors")},
        "target":{k:full(consumed["normals"] if k=="normals" else consumed["target"]) for k in ("points","normals","colors")},
        "seed":full(consumed["seed"])}
    for query in common["query_trace"]:query["packet"]=full(query["packet"])
    payload={"pair_index":0,"pair":[0,2],"proposal_index":0,"caller":"_pair","gate_context":[],"input_binding":full_input}
    call={"event":"match","call_index":0,"payload":payload,"input_binding":full_input,"consumed_input_binding":consumed,
        "source_binding":common["source_binding"],"terminal":common["terminal"],"result":{"accepted":True},
        "native_shadow":common["native_shadow"],"loop_report":common["loop_report"],"query_trace":common["query_trace"],
        "query_trace_sha256":hashlib.sha256(p.values.canonical(common["query_trace"]).encode()).hexdigest(),
        "complete":True,"input_bytes_unchanged":True,"cleanup_failures":[],"failure":None}
    record={"complete":True,"restored":True,"closed":True,"failure":None,"cleanup_failures":[],"successful_builds":1,
        "final_pose_inventory":{"view_indices":[0,1]},"calls":[call],"pairs":[{"pair":[0,2],"verdict":{"accepted":True}}],
        "gates":[{"event":"gate","gate_index":0,"complete":True,"name":"_pair","result":True,"wall_s_inclusive":.1}],
        "graphs":[{"event":"graph_optimization","graph_index":0,"complete":True,"edges":[[0,1,False]]}],
        "events":[{"event":"match","index":0},{"event":"gate","index":0},{"event":"graph_optimization","index":0}]}
    owner={"policy":workspace.POLICY,"provenance":contract,"closed":True,"failure":None,"active_lane":False,
        "retained_failed_lanes":0,"template_started":False,"audit":True,
        "jobs":[{"index":0,"closed":True,"failure":None,"report":call["loop_report"]}]}
    scope={"checkpoint":{"sha256":"c"*64},"settings":{"voxel_m":.005}}
    report={"kind":p.KIND,"mode":"audit","status":"passed","binding":binding,"binding_after":binding,
        "scope_binding":scope,"failure":None,"cleanup_failures":[],"cleanup_passed":True,"registration":record,
        "final_pose_inventory":record["final_pose_inventory"],"calls":record["calls"],"workspace":owner,
        "cache_receipts":[{"closed":True,"failure":None,"active_streams":0}]}
    record["workspace"]=owner;record["cache_receipts"]=report["cache_receipts"]
    return report,binding,scope,contract,proof


class FakeLock:
    def __init__(self,path):
        self.path=str(path);self.handle=object();self.api=object();self.closed=False
        self.data=Path(path).read_bytes();self.reads=0;self.close_error=None
    def read(self):
        if self.closed:raise ValueError("closed proof")
        self.reads+=1;return self.data
    def close(self):
        if self.close_error is not None:raise self.close_error
        self.closed=True


class AuditContracts(unittest.TestCase):
    def check(self,report,binding,scope,contract,proof):
        with patch.object(p.values,"actual_resource_closure"),patch.object(workspace,"source_contract",return_value=contract),patch.object(p,"source_contract",return_value=proof),\
             patch.object(injection,"source_contract",return_value=binding["finish_scope_source"]):
            return p.validate_report(report,binding,scope)

    def test_complete_current_query_and_native_shadows(self):
        args=fixture();contract,calls=self.check(*args)
        self.assertEqual(len(calls),1);self.assertEqual(contract,args[3])

    def test_old_kind_changed_checkpoint_or_omitted_source_rejected(self):
        for mutate in (lambda r,b:r.update(kind="gpu-icp-complete-bridge-owned-proof-audit-v2"),
            lambda r,b:r.update(scope_binding={}),lambda r,b:b["artifacts_sha256"].pop("scripts/fake-workspace.py")):
            args=fixture();mutate(args[0],args[1])
            with self.assertRaises(ValueError):self.check(*args)

    def test_scope_code_family_must_be_current_and_complete(self):
        args=fixture();current=copy.deepcopy(args[1]["finish_scope_source"])
        for mutate in (lambda b:b["finish_scope_source"].update(policy="old-scope"),
            lambda b:b["artifacts_sha256"].pop("scripts/fake-scope.py")):
            changed=copy.deepcopy(args);mutate(changed[1])
            with patch.object(p.values,"actual_resource_closure"),patch.object(workspace,"source_contract",return_value=args[3]),\
                 patch.object(p,"source_contract",return_value=args[4]),patch.object(injection,"source_contract",return_value=current):
                with self.assertRaises(ValueError):p.validate_report(*changed[:3])

    def test_query_shadow_false_miss_or_native_id_failure_rejected(self):
        for mutate in (lambda c:c["query_trace"].pop(),lambda c:c["native_shadow"].update(correspondence_ids_equal=False),
            lambda c:c["loop_report"]["statistics"].update(audited_misses=0),lambda c:c["native_shadow"].update(transform_max_abs_delta=float("nan"))):
            args=fixture();mutate(args[0]["calls"][0])
            with self.assertRaises(ValueError):self.check(*args)

    def test_dynamic_call_order_job_report_and_cleanup_exact(self):
        for mutate in (lambda r:r["calls"][0].update(call_index=1),lambda r:r["workspace"]["jobs"][0].update(index=1),
            lambda r:r["workspace"].update(retained_failed_lanes=1),lambda r:r["cache_receipts"][0].update(closed=False),
            lambda r:r["calls"][0].update(cleanup_failures=["completion failed"])):
            args=fixture();mutate(args[0])
            with self.assertRaises(ValueError):self.check(*args)

    def test_current_cloud_color_seed_and_original_context_bound(self):
        for mutate in (lambda c:c["input_binding"]["seed"].update(sha256="f"*64),
            lambda c:c["payload"].update(pair=[1,2,3]),lambda c:c["payload"].update(caller=None)):
            args=fixture();mutate(args[0]["calls"][0])
            with self.assertRaises(ValueError):self.check(*args)

    def test_zero_gpu_scope_is_an_explicit_native_negative(self):
        args=fixture();r=args[0];r["calls"].clear();r["workspace"]=None;r["cache_receipts"]=[]
        r["registration"].update(workspace=None,cache_receipts=[])
        r["registration"]["events"].pop(0)
        _,calls=self.check(*args);self.assertEqual(calls,[])
        r["workspace"]={"closed":True}
        with self.assertRaises(ValueError):self.check(*args)

    def test_complete_event_suffix_cannot_be_omitted_or_duplicated(self):
        for mutate in (lambda r:r["registration"]["events"].pop(),
            lambda r:r["registration"]["events"].append(copy.deepcopy(r["registration"]["events"][0]))):
            args=fixture();mutate(args[0])
            with self.assertRaises(ValueError):self.check(*args)


class OwnedTimingContracts(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/"audit.json";self.qpath=Path(self.folder.name)/"quality.json"
        self.native=Path(self.folder.name)/"native.json";self.geometry=Path(self.folder.name)/"geometry.ply"
        self.geometry.write_bytes(b"geometry");gh=p.values.sha(self.geometry)
        self.report,self.binding,self.scope,self.contract,self.proof=fixture()
        self.path.write_text(json.dumps(self.report),encoding="utf-8");self.native.write_text("{}",encoding="utf-8")
        self.quality={"native_report":{"path":str(self.native),"sha256":p.values.sha(self.native)},
            "candidate_geometry":{"path":str(self.geometry),"sha256":gh,"sha256_after":gh},
            "reference_geometry":{"path":str(self.geometry),"sha256":gh,"sha256_after":gh}}
        self.qpath.write_text(json.dumps(self.quality),encoding="utf-8")
        call=self.report["calls"][0]
        refs=[{"payload":call["payload"],"input":p.values.normalized_input(call["consumed_input_binding"]),
            "source":p.values.normalized_source(call["source_binding"]),"terminal":call["terminal"],"result":call["result"]}]
        self.mocks=[patch.object(p,"WindowsProofLock",FakeLock),patch.object(p,"LoadedGuard",lambda *_:SimpleNamespace(check=lambda:None)),
            patch.object(p,"validate_report",return_value=(self.contract,refs)),patch.object(p,"validate_quality")]
        # Comparator import is injected only for this ownership test. Actual
        # quality evidence has an independent positive/negative suite below.
        from scripts.research.compare_resident_finishes import semantic_agreement
        self.comparator=SimpleNamespace(__file__=str(self.native),quality_record=lambda *_:{},semantic_agreement=semantic_agreement)
        self.mocks.append(patch.dict(sys.modules,{"scripts.research.compare_gpu_icp_finishes":self.comparator}))
        for mock in self.mocks:mock.start();self.addCleanup(mock.stop)
        self.token=p.validate_finish_audit(self.path,self.binding,self.scope,quality_path=self.qpath)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        state=p._REGISTRY[id(self.token)]
        if not state["closed"]:
            for lock in state["locks_owner"]:lock.close_error=None
            try:p.close_permit(self.token)
            except ValueError:
                if not state["closed"]:raise

    def timing(self):
        actual=copy.deepcopy(self.report["registration"]);call=actual["calls"][0]
        consumed=call["consumed_input_binding"]
        consumed["configuration"]["audit_nearest"]=consumed["configuration"]["audit_misses"]=False
        call["loop_report"]["statistics"].update(audited_hits=0,audited_misses=0,flagged_rows=0,packet_bytes=0)
        actual["workspace"]["audit"]=False
        return actual,call

    def finish_call(self,call):
        index=p.next_workspace_job(self.token,call["payload"])
        p.workspace_authorizer(self.token,index)(call["consumed_input_binding"],call["source_binding"])
        result=p.validate_finish_terminal(self.token,index,call["payload"],call["terminal"],call["loop_report"])
        p.validate_finish_event(self.token,call)
        for key in ("gates","graphs"):
            for row in self.report["registration"][key]:p.validate_finish_event(self.token,row)
        return result

    def test_global_schedule_start_terminal_complete_and_two_owned_reads(self):
        actual,call=self.timing();self.assertTrue(self.finish_call(call))
        self.assertTrue(p.validate_complete_finish(self.token,actual))
        for _ in range(20):p.registered(self.token)
        self.assertEqual(p._state(self.token)["reads"],[1,1])
        p.close_permit(self.token);self.assertEqual(p.permit_report(self.token)["held_handle_reads"],[2,2])

    def test_no_start_without_scheduled_call_or_second_start(self):
        _,call=self.timing()
        with self.assertRaises(ValueError):p.validate_start(self.token,0,call["consumed_input_binding"],call["source_binding"])
        p.next_workspace_job(self.token,call["payload"])
        p.validate_start(self.token,0,call["consumed_input_binding"],call["source_binding"])
        with self.assertRaises(ValueError):p.validate_start(self.token,0,call["consumed_input_binding"],call["source_binding"])

    def test_exact_geometry_seed_caller_context_checked_before_enqueue(self):
        _,call=self.timing()
        for mutate in (lambda v:v.update(caller="other"),lambda v:v.update(proposal_index=1),
            lambda v:v["input_binding"]["source"]["colors"].update(sha256="f"*64),
            lambda v:v["input_binding"]["seed"].update(sha256="f"*64)):
            changed=copy.deepcopy(call["payload"]);mutate(changed)
            with self.assertRaises(ValueError):p.next_workspace_job(self.token,changed)

    def test_terminal_damage_unclosed_lane_and_hidden_audit_rejected(self):
        _,call=self.timing();p.next_workspace_job(self.token,call["payload"])
        p.validate_start(self.token,0,call["consumed_input_binding"],call["source_binding"])
        wrong=copy.deepcopy(call["terminal"]);wrong["queries"]+=1
        with self.assertRaises(ValueError):p.validate_finish_terminal(self.token,0,call["payload"],wrong,call["loop_report"])
        for mutate in (lambda r:r.update(closed=False),lambda r:r["statistics"].update(audited_hits=1)):
            changed=copy.deepcopy(call["loop_report"]);mutate(changed)
            with self.assertRaises(ValueError):p.validate_finish_terminal(self.token,0,call["payload"],call["terminal"],changed)

    def test_original_gate_graph_and_missing_suffix_rejected(self):
        actual,call=self.timing()
        with self.assertRaises(ValueError):p.validate_complete_finish(self.token,actual)
        self.finish_call(call)
        for mutate in (lambda a:a["gates"][0].update(result=False),lambda a:a["graphs"].pop(),lambda a:a["calls"].pop(),
            lambda a:a["final_pose_inventory"].update(view_indices=[0])):
            wrong=copy.deepcopy(actual);mutate(wrong)
            with self.assertRaises(ValueError):p.validate_complete_finish(self.token,wrong)

    def test_final_completion_cannot_contradict_workspace_cache_or_jobs(self):
        actual,call=self.timing();self.finish_call(call)
        for mutate in (lambda r:r.update(closed=False),lambda r:r.update(restored=False),
            lambda r:r["workspace"].update(audit=True),lambda r:r["workspace"]["jobs"][0].update(index=1),
            lambda r:r["workspace"]["jobs"].clear(),lambda r:r["cache_receipts"][0].update(active_streams=1)):
            changed=copy.deepcopy(actual);mutate(changed)
            with self.assertRaises(ValueError):p.validate_complete_finish(self.token,changed)

    def test_foreign_constructed_modified_and_replaced_reference_owner_rejected(self):
        for token in (None,SimpleNamespace(policy=p.POLICY),p.FinishTimingPermit(**self.token.__dict__)):
            with self.assertRaises(ValueError):p.registered(token)
        state=p._state(self.token);old=state["refs"]
        state["refs"]=p.References(*old)
        with self.assertRaises(ValueError):p.registered(self.token)
        state["refs"]=old
        object.__setattr__(self.token,"policy","old")
        with self.assertRaises(ValueError):p.registered(self.token)
        object.__setattr__(self.token,"policy",p.POLICY)

    def test_failed_end_hash_closes_both_handles_preserving_primary(self):
        state=p._state(self.token);state["locks"][0].data=b"changed";primary=RuntimeError("original GPU fault")
        with self.assertRaises(RuntimeError) as caught:p.close_permit(self.token,primary=primary)
        self.assertIs(caught.exception,primary);self.assertIsInstance(primary.__cause__,ValueError)
        self.assertTrue(all(lock.closed for lock in state["locks_owner"]));self.assertIsNotNone(state["failure"])

    def test_first_handle_close_failure_does_not_skip_second(self):
        state=p._state(self.token);state["locks"][0].close_error=OSError("CloseHandle")
        with self.assertRaises(OSError):p.close_permit(self.token)
        self.assertFalse(state["closed"]);self.assertTrue(state["locks"][1].closed)

    def test_damaged_handle_metadata_uses_original_owner_for_cleanup(self):
        state=p._state(self.token);state["locks"][0].handle=object()
        with self.assertRaises(ValueError):p.close_permit(self.token)
        self.assertTrue(all(lock.closed for lock in state["locks_owner"]))

    def test_copied_values_detached_from_immutable_cached_references(self):
        original=p._call(self.token,0);original["payload"]["pair"][0]=99
        self.assertEqual(p._call(self.token,0)["payload"]["pair"],[0,2])
        with self.assertRaises(AttributeError):p._state(self.token)["refs"].calls=()


class LoadedOwnerTests(unittest.TestCase):
    def test_actual_cold_source_guard_and_in_place_code_default_change(self):
        guard=p.LoadedGuard((p,));guard.check();fn=p.current_workspace_job;code=fn.__code__
        try:
            fn.__code__=(lambda token:None).__code__
            with self.assertRaises(ValueError):guard.check()
        finally:fn.__code__=code
        defaults=fn.__defaults__
        try:
            fn.__defaults__=(None,)
            with self.assertRaises(ValueError):guard.check()
        finally:fn.__defaults__=defaults

    def test_fresh_import_without_numerical_modules(self):
        result=subprocess.run([sys.executable,"-S","-c",
            "from scripts.research import gpu_icp_finish_protocol; import sys; assert not {'numpy','cupy','open3d','cv2'} & set(sys.modules)"],
            cwd=p.ROOT,text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


class QualityCertificateTests(unittest.TestCase):
    def setUp(self):
        from tests.test_gpu_icp_finish_quality import report
        from scripts.research import compare_gpu_icp_finishes as comparator
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/"geometry.npz";self.path.write_bytes(b"closed geometry fixture")
        geometry={"path":str(self.path),"sha256":p.values.sha(self.path)}
        self.native,self.audit=report(),report("audit")
        for mode,r in (("native",self.native),("audit",self.audit)):
            r["kind"]=p.NATIVE_KIND if mode=="native" else p.KIND
            r["registration"].update(closed=True,successful_builds=1,final_pose_inventory=r["final_pose_inventory"])
            r["geometry"]=copy.deepcopy(geometry)
        computed=comparator.quality_record(self.native,self.audit)
        self.quality={"kind":p.QUALITY_KIND,"status":"passed","binding":self.audit["binding"],
            "scope_binding":self.audit["scope_binding"],"audit_report":{"sha256":"a"*64},
            "native_report":{"sha256":"b"*64},"comparator_source_sha256":p.values.sha(comparator.__file__),
            "candidate_geometry":dict(geometry,sha256_after=geometry["sha256"]),
            "reference_geometry":dict(geometry,sha256_after=geometry["sha256"]),
            "surface":{"surface_p95_m":.0001,"precision":1.,"completeness":1.,"threshold_m":.005,"samples":30000,
                "candidate_geometry_sha256":geometry["sha256"],"reference_geometry_sha256":geometry["sha256"]},
            "checks":dict(computed["checks"],surface_bounds=True),
            "metrics":dict(computed["metrics"],surface_p95_m=.0001,precision=1.,completeness=1.,threshold_m=.005,samples=30000)}

    def check(self):
        return p.validate_quality(self.quality,self.native,self.audit,self.audit["binding"],self.audit["scope_binding"],"a"*64,"b"*64)

    def test_complete_current_geometry_certificate_and_independent_pure_checks(self):
        self.assertTrue(all(self.check()["checks"].values()))

    def test_wrong_numeric_surface_and_missing_geometry_closure_rejected(self):
        good=copy.deepcopy(self.quality)
        for mutate in (lambda q:q["surface"].update(surface_p95_m=.0005000000000000001),
            lambda q:q["surface"].update(completeness=.9989),lambda q:q["surface"].update(samples=29999),
            lambda q:q["reference_geometry"].update(sha256_after="f"*64)):
            self.quality=copy.deepcopy(good);mutate(self.quality)
            with self.assertRaises(ValueError):self.check()

    def test_passed_flags_cannot_override_recomputed_memberships_or_shadow(self):
        graph=self.audit["registration"]["graphs"][0]
        graph["after"]["edges"][0]["target_node"]=2
        graph["after"]["nodes"].append(copy.deepcopy(graph["after"]["nodes"][0]))
        graph["context"]["node_fragment_indices"].append(3)
        with self.assertRaises(ValueError):self.check()

    def test_false_quality_metrics_changed_report_hash_and_wrong_kind_rejected(self):
        good=copy.deepcopy(self.quality)
        for mutate in (lambda q:q["metrics"].update(actual_gpu_calls=0),lambda q:q["audit_report"].update(sha256="f"*64),
            lambda q:q.update(kind="offline-field-finish-actual-input-conformance-v2")):
            self.quality=copy.deepcopy(good);mutate(self.quality)
            with self.assertRaises(ValueError):self.check()


if __name__=="__main__":unittest.main()
