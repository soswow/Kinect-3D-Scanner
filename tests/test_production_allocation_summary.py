"""Publication guards: observed capacity, physical closure, privacy and failures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research import summarize_production_allocation as s


def raw(current=False, required=3328, limit=10000):
    p = {"schema_version":1, "source_sha256":s.CURRENT if current else s.BASELINE, "session":"C:/private/archive.zip",
        "input_sha256":"a"*64, "source_changed_during_profile":False, "input_changed_during_profile":False,
        "pose_seeds_used":False, "finish_requested":True, "mesh_built":True, "experimental_visual_fallback":False,
        "frames":2, "selected_indices":[0,1], "live_diagnostics":[{},{}], "seed":0,
        "accepted_indices":[0,1], "accepted_indices_before_finish":[0,1], "accepted":2, "accepted_before_finish":2,
        "poses":[{"index":i,"camera_to_world":[[1.,0,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]]} for i in (0,1)],
        "settings":{"confidence_fusion":True,"final_voxel_m":.005,"final_block_count":limit,
            "sensor_calibration":{"camera_serial":"PRIVATE-SERIAL","source":"C:/Users/private/calibration"}},
        "settings_overrides":{},"original_settings_sha256":"b"*64,
        "versions":{"numpy":"test","open3d":"test","opencv":"test","python":"test"},
        "native_mode":"on", "initial_blocks":"2000", "thread_policy":{"opencv_threads":20,"open3d_threads":20},
        "omp_threads":"8", "pipeline_options":{"KINECT_LIVE_RECOVERY":"full","KINECT_CUDA_CONFIDENCE":"off"},
        "gpu_hardware":[{"gpu":"example"}], "native_extension":{"path":"C:/private/native.pyd","sha256":"c"*64,
            "changed_during_profile":False}, "live_s":1.,"finish_s":2.,"processing_s":3.,
        "geometry":{"artifact":"C:/private/geometry.npz","vertices":4,"triangles":2},
        "backend":{"requested":"cuda","device":"CUDA:0","tracking":"legacy","tracking_device":"CPU:0",
            "cuda_available":True,"fallback_reason":None,"native_kernels":{"requested":"on","active":True,"api_version":2},
            "geometric_verification":{"implementation":"legacy"},
            "confidence_cuda":{"requested":"fused","implementation":"fused","fallback_reason":None},
            "cuda_input":{"requested":"auto","implementation":"cuda","device":"CUDA:0","probe_passed":True,
                "gpu_batches":2,"cpu_batches":0,"fallback_batches":0},
            "descriptor_matching":{"requested":"cuda","implementation":"cuda","cuda_batches":2,"cpu_batches":0,"fallback_reason":None},
            "depth_confidence":{"requested":"off","implementation":"cpu","device":"CPU:0","cpu_calls":2,
                "gpu_calls":0,"fallback_calls":0}},
        "final_reconstruction":{"applied":True,"voxel_m":.005,"block_limit":limit,"blocks":required,
            "attribute_budget_mib":limit*.078125}}
    if current:
        p["final_reconstruction"].update(required_blocks=required, requested_block_capacity=required,
            allocated_blocks=required, initial_block_capacity=required, allocation_strategy="exact missing-key activation",
            configured_attribute_budget_mib=limit*.078125, attribute_budget_mib=required*.078125, planning_elapsed_ms=12.)
    return p


def current_closure(kind):
    return {"kind":kind,"status":"passed","source_sha256":s.CURRENT,"source_sha256_after":s.CURRENT,
        "artifacts_sha256":{"scanner_server/test.py":"d"*64},"artifacts_sha256_after":{"scanner_server/test.py":"d"*64},
        "cleanup_failures":[],"environment_restored":True}


def physical(profile):
    p = current_closure("production-weighted-final-missing-activation-physical-smoke-v1")
    runtime = {"open3d_threads":20, "versions":{"numpy":"test","open3d":"test"}, "gpu":profile["gpu_hardware"],
        "binaries":{"C:/private/_kinect_native.pyd":{"sha256":"c"*64}}, "environment":{"OMP_NUM_THREADS":"8"}}
    p.update(runtime_binding=runtime,runtime_binding_after=copy.deepcopy(runtime),performance_claim=False,whole_session_quality_proven=False,pairs=[])
    for name in ("cpu-original","cuda-tensor","cuda-fused"):
        snapshots={k:{"sha256":"e"*64,"dtype":"<f4","shape":[4,4096,c],"bytes":4*4096*c*4}
            for k,c in (("tsdf",1),("weight",1),("color",3))}
        ordinary = {"initial_capacity":4,"final_capacity":8,"final_blocks":4,
            "frames":[{"capacity":c,"active_keys":n} for c,n in zip([4,8,8],[3,4,4])],"attribute_snapshots":snapshots}
        candidate=copy.deepcopy(ordinary);candidate["final_capacity"]=4
        for frame in candidate["frames"]:frame["capacity"]=4
        p["pairs"].append({"backend":name,"device":"CPU:0" if name=="cpu-original" else "CUDA:0","complete":True,
            "inputs":{"depth":{"sha256":"f"*64}},"inputs_after":{"depth":{"sha256":"f"*64}},"inputs_unchanged":True,
            "fractional_confidence_pixels":9,"nonzero_weight_voxels":16384,"fractional_weight_voxels":224,
            "ordinary":ordinary,"candidate":candidate,"comparison":{k:{"elements":e,"bit_mismatches":0}
                for k,e in (("tsdf",16384),("weight",16384),("color",49152))}})
    return p


def quality(old_digest, new_digest):
    return {"baseline_report_sha256":old_digest,"candidate_report_sha256":new_digest,"same_mesh_success":True,
        "same_accepted_indices":True,"common_poses":2,"preview_resolution_comparison":False,
        "max_pose_translation_delta_m":1e-14,"max_pose_rotation_delta_deg":.000001,
        "symmetric_vertex_distance_m":{"max":0.},"triangle_surface_metrics_vs_cpu":{"threshold_m":.005,
            "samples_per_surface":30000,"precision":1.,"completeness":1.,"surface_p95_m":1e-7,"surface_rmse_m":1e-8,
            "alignment":"fixed input coordinate frame; no scale or trajectory fitting"}}


class FakeReports:
    def __init__(self, values):self.values=values;self.files={};self.root=Path("C:/fixture")
    def read(self,name):
        s.relative(name);value=self.values[name]
        self.files[name]={"path":"benchmark-output/field-cuda-study/"+name,"sha256":s.canonical(value),
            "reported_status":value.get("status","profile"),"bytes":1}
        return value
    def ref(self,name):self.read(name);return dict(self.files[name])
    def matches_path(self,name,path):s.require(path==str(self.root/name),"wrong path")
    def attachment(self,name,digest):s.relative(name);return {"path":name,"sha256":s.sha256(digest)}
    def close(self):pass


def worker_record(reports,name,p,current):
    return {"profile" if current else "report":str(reports.root/name),
        "profile_sha256" if current else "report_sha256":reports.ref(name)["sha256"],"input_sha256":p["input_sha256"],
        "status":"complete","exit_code":0,"child_wait_completed":True,"owned_tree_closed":True,
        "process_cleanup_failures":[],"process_ownership":"windows-kill-on-close-job","child_pid":123,"mesh_built":True,
        "accepted_before_finish":p["accepted_before_finish"],"accepted_after_finish" if current else "accepted":p["accepted"],
        "live_s":p["live_s"],"finish_s":p["finish_s"],"log":str(reports.root/(name.removesuffix('.json')+'.log')),
        "log_sha256":"1"*64,"geometry_sha256":"2"*64}


def complete_fixture():
    values={};reports=FakeReports(values)
    observations={"kind":"root-completed-exec-observations-v1","status":"complete","observations":[]}
    quality_exec={"kind":"production-allocation-fixed-surface-quality-execution-v1","status":"complete",
        "source_sha256":s.CURRENT,"source_sha256_after":s.CURRENT,"inputs_sha256":{},"artifacts_sha256":{
            name:"d"*64 for name in ("scripts/compare_session_profiles.py","scripts/profile_session.py","shared/surface_metrics.py")},
        "runs":[]}
    def observe(name):
        observations["observations"].append({"report":{"path":name,"sha256":reports.ref(name)["sha256"]},
            "exit_code":0,"tool_completed":True,"pid_recorded":False,"tool_chunk_id":"example",
            "executable":"C:/private/python.exe","argv":["example"]})
    for label,(old_name,new_name,old_exec,new_exec) in s.PAIRS.items():
        required,limit=(13302,20000) if label=="chest-7" else (3328,10000)
        old,new=raw(required=required,limit=limit),raw(True,required,limit)
        old["session"]=new["session"]=label+".zip"
        old["geometry"]["artifact"]=str(reports.root/(old_name.removesuffix('.json')+'.geometry.npz'))
        new["geometry"]["artifact"]=str(reports.root/(new_name.removesuffix('.json')+'.geometry.npz'))
        values[old_name],values[new_name]=old,new
        for name,p,current,execution_name in ((old_name,old,False,old_exec),(new_name,new,True,new_exec)):
            if execution_name not in values:
                values[execution_name]={"kind":"raw-field-existing-recovery-policy-experiment-v1","status":"complete",
                    "source_sha256":s.CURRENT if current else s.BASELINE,"artifacts_sha256":{"scripts/test.py":"d"*64},
                    "expected_thread_policy":new["thread_policy"],"native_extension":new["native_extension"],"runs":[]}
            values[execution_name]["runs"].append(worker_record(reports,name,p,current))
        qname="production-memory-quality-v2/"+label+".json"
        values[qname]=quality(reports.ref(old_name)["sha256"],reports.ref(new_name)["sha256"]);observe(qname)
        quality_exec["runs"].append({"label":label,"baseline":str(reports.root/old_name),"candidate":str(reports.root/new_name),
            "quality":str(reports.root/qname),"log":str(reports.root/(qname.removesuffix('.json')+'.log')),
            "exit_code":0,"child_wait_completed":True,"quality_sha256":reports.ref(qname)["sha256"],"log_sha256":"1"*64,
            "baseline_report_sha256":reports.ref(old_name)["sha256"],"candidate_report_sha256":reports.ref(new_name)["sha256"],
            "baseline_geometry_sha256":"2"*64,"candidate_geometry_sha256":"2"*64,
            "mesh_arrays_equal":{"points":False,"faces":False}})
        inputs=quality_exec["inputs_sha256"]
        for report_name in (old_name,new_name):
            inputs[str((reports.root/report_name).resolve())]=reports.ref(report_name)["sha256"]
            inputs[str((reports.root/(report_name.removesuffix('.json')+'.geometry.npz')).resolve())]="2"*64
        inputs[str(Path("C:/private/raw")/new["session"])]=new["input_sha256"]
        inputs[str(Path(new["native_extension"]["path"]).resolve())]=new["native_extension"]["sha256"]
    name="production-allocation-smoke-v2/report.json"
    values[name]=physical(values[s.PAIRS["chest-5"][1]]);observe(name)
    name="production-final-budget-v2/report.json";p=values[s.PAIRS["chest-7"][1]]
    probe=current_closure("current-full-raw-final-budget-boundary-probe-v2")
    inputs={"profile":reports.ref(s.PAIRS["chest-7"][1])["sha256"],"raw_zip":p["input_sha256"]}
    probe.update(inputs_sha256=inputs,inputs_sha256_after=copy.deepcopy(inputs),geometry_authority=False,
        registration_authority=False,performance_claim=False,owner_state_unchanged=True,native_extension_sha256="c"*64,
        native_extension_sha256_after="c"*64,actual_native_sha256="c"*64,gpu_hardware=p["gpu_hardware"],
        gpu_hardware_after=p["gpu_hardware"],actual_versions=p["versions"],actual_device="CUDA:0",
        actual_threads={"open3d":20,"opencv":20,"omp":"8"},
        preflight={"frames":2,"final_views":2,"required_blocks":13302,"settings_sha256":s.canonical(p["settings"])},
        settings_reconstruction={"sha256":s.canonical(p["settings"]),"numeric_tolerance":0},
        pose_serializer={"original_p_tolist_expression":True,"loaded_code_checked":True},
        boundary={"candidate_allocated":False,"tracking_or_fusion_called":False,"hooks_restored":True,
            "required_blocks":13302,"logical_limit":10000,"original_error_type":"ValueError",
            "create_calls":[{"requested_blocks":1,"actual_capacity":1,"complete":True}],
            "planner_calls":[{"stage":"final_capacity_preflight","required_blocks":13302,"views":2,"complete":True}]})
    values[name]=probe;observe(name)
    quality_exec["inputs_sha256_after"]=copy.deepcopy(quality_exec["inputs_sha256"])
    quality_exec["artifacts_sha256_after"]=copy.deepcopy(quality_exec["artifacts_sha256"])
    values["production-memory-quality-v2/execution.json"]=quality_exec
    observations["geometry_execution"]={"path":"production-memory-quality-v2/execution.json",
        "sha256":reports.ref("production-memory-quality-v2/execution.json")["sha256"]}
    values["root-completed-exec-observations-v1.json"]=observations
    for name in ("production-allocation-smoke-v1/report.json","production-final-budget-v1/report.json"):
        values[name]={"status":"failed","failure":{"type":"TypeError","message":"C:/private/secret"}}
    return reports


class ProductionPublicationTests(unittest.TestCase):
    def test_full_publishable_summary_never_authorizes_math_or_speed(self):
        result=s.summarize(complete_fixture(),current_source=s.CURRENT)
        self.assertFalse(result["performance_authority"])
        self.assertFalse(result["geometry_authority"])
        self.assertFalse(result["causal_speed_claim"])
        self.assertEqual(len(result["sessions"]),3)
        self.assertEqual(result["sessions"][0]["allocation"]["historical_actual_block_capacity"],None)
        text=json.dumps(result)
        for private in ("PRIVATE-SERIAL","C:/private","camera_to_world","sensor_calibration"):
            self.assertNotIn(private,text)
        self.assertEqual([x["reported_status"] for x in result["preserved_failures"]],["failed","failed"])
        self.assertEqual(result["sessions"][0]["geometry"]["physical_input_closure"]["ordered_mesh_arrays_equal"],
            {"points":False,"faces":False})

    def test_wrong_core_never_closes(self):
        with self.assertRaises(ValueError):s.summarize(complete_fixture(),current_source=s.BASELINE)

    def test_scope_source_input_runtime_settings_and_coverage_drift(self):
        for key,value in (("source_sha256",s.BASELINE),("input_sha256","9"*64),("settings",{}),
            ("thread_policy",{}),("accepted_indices",[0]),("selected_indices",[1,0]),("versions",{}),
            ("pipeline_options",{}),("gpu_hardware",[])):
            with self.subTest(key=key):
                a,b=raw(),raw(True);b[key]=value
                with self.assertRaises((ValueError,KeyError)):s.pair_scope(a,b)

    def test_partial_seeded_or_checkpoint_replay_rejected(self):
        for key,value in (("pose_seeds_used",True),("checkpoint",{"sha256":"a"*64}),("mesh_built",False),
            ("source_changed_during_profile",True),("input_changed_during_profile",True),("finish_requested",False)):
            a,b=raw(),raw(True);b[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.pair_scope(a,b)

    def test_live_coverage_scalar_staleness_or_duplicate_rows_rejected(self):
        for key,value in (("accepted_before_finish",1),("accepted_indices_before_finish",[0,0])):
            a,b=raw(),raw(True);a[key]=b[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.pair_scope(a,b)

    def test_confidence_algorithm_not_disabled(self):
        a,b=raw(),raw(True);a["settings"]["confidence_fusion"]=b["settings"]["confidence_fusion"]=False
        with self.assertRaises(ValueError):s.pair_scope(a,b)

    def test_actual_backend_rejected_despite_requested_cuda(self):
        for key,value in (("implementation","cpu"),("probe_passed",False),("fallback_batches",1),("gpu_batches",True)):
            p=raw(True);p["backend"]["cuda_input"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.actual_cuda(p)

    def test_capacity_growth_boolean_counts_or_bad_floor_rejected(self):
        for key,value in (("allocated_blocks",6656),("initial_block_capacity",True),("required_blocks",3330),
            ("attribute_budget_mib",781.25),("configured_attribute_budget_mib",260.)):
            a,b=raw(),raw(True);b["final_reconstruction"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.allocation(a,b)

    def test_old_capacity_never_inferred_or_silently_dropped(self):
        a,b=raw(),raw(True);a["final_reconstruction"]["allocated_blocks"]=10000
        with self.assertRaises(ValueError):s.allocation(a,b)

    def test_quality_ref_pose_and_physical_surface_failures(self):
        for target,key,value in ((None,"baseline_report_sha256","1"*64),(None,"same_accepted_indices",False),
            (None,"common_poses",1),(None,"max_pose_translation_delta_m",.000501),
            (None,"max_pose_rotation_delta_deg",.100001),("triangle_surface_metrics_vs_cpu","precision",.998),
            ("triangle_surface_metrics_vs_cpu","samples_per_surface",29999),
            ("triangle_surface_metrics_vs_cpu","surface_p95_m",.000501),
            ("triangle_surface_metrics_vs_cpu","alignment","fitted")):
            r=complete_fixture();name="production-memory-quality-v2/chest-5.json"
            obj=r.values[name] if target is None else r.values[name][target];obj[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                s.quality(r,name,s.PAIRS["chest-5"][0],s.PAIRS["chest-5"][1],raw(),raw(True))

    def test_current_worker_exit_wait_tree_cleanup_and_report_hash(self):
        old_name,new_name,old_exec,new_exec=s.PAIRS["chest-5"]
        for key,value in (("exit_code",1),("child_wait_completed",False),("owned_tree_closed",False),
            ("process_cleanup_failures",[{}]),("profile_sha256","1"*64),("process_ownership","unowned"),("child_pid",True),
            ("accepted_before_finish",1),("accepted_after_finish",1)):
            r=complete_fixture();r.values[new_exec]["runs"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                s.worker(r,new_exec,new_name,r.values[new_name],current=True)

    def test_historical_worker_does_not_require_unrecorded_tree(self):
        r=complete_fixture();old_name,_,old_exec,_=s.PAIRS["chest-5"]
        row=r.values[old_exec]["runs"][0]
        del row["owned_tree_closed"],row["process_cleanup_failures"],row["process_ownership"]
        result=s.worker(r,old_exec,old_name,r.values[old_name],current=False)
        self.assertIsNone(result["owned_tree_closed"])

    def test_quality_input_manifest_source_payload_refs_and_exit(self):
        ename="production-memory-quality-v2/execution.json";qname="production-memory-quality-v2/chest-5.json"
        old_name,new_name,*_=s.PAIRS["chest-5"]
        for mutation in (lambda v:v.update(source_sha256_after=s.BASELINE),lambda v:v.update(inputs_sha256_after={}),
                lambda v:v["runs"][0].update(baseline_geometry_sha256="3"*64),
                lambda v:v["runs"][0].update(candidate_report_sha256="3"*64),
                lambda v:v["runs"][0].update(exit_code=1),lambda v:v.update(artifacts_sha256_after={})):
            r=complete_fixture();mutation(r.values[ename])
            with self.assertRaises(ValueError):s.quality_execution(r,ename,"root-completed-exec-observations-v1.json",
                "chest-5",qname,old_name,new_name,r.values[old_name],r.values[new_name])

    def test_false_ordered_mesh_equality_is_diagnostic_not_failure(self):
        r=complete_fixture();a,b,*_=s.PAIRS["chest-5"]
        result=s.quality_execution(r,"production-memory-quality-v2/execution.json","root-completed-exec-observations-v1.json",
            "chest-5","production-memory-quality-v2/chest-5.json",a,b,r.values[a],r.values[b])
        self.assertFalse(result["ordered_mesh_array_identity_required"])
        self.assertFalse(result["ordered_mesh_arrays_equal"]["faces"])

    def test_actual_raw_basename_scope_requires_unique_controller_name_and_sha(self):
        ename="production-memory-quality-v2/execution.json";a,b,*_=s.PAIRS["chest-5"]
        for mutation in (lambda d:d.update({str(Path('C:/other/chest-5.zip')):'a'*64}),
            lambda d:d.update({str(Path('C:/private/raw/chest-5.zip')):'9'*64})):
            r=complete_fixture();mutation(r.values[ename]["inputs_sha256"])
            r.values[ename]["inputs_sha256_after"]=copy.deepcopy(r.values[ename]["inputs_sha256"])
            with self.assertRaises(ValueError):s.quality_execution(r,ename,"root-completed-exec-observations-v1.json",
                "chest-5","production-memory-quality-v2/chest-5.json",a,b,r.values[a],r.values[b])

    def test_smoke_missing_backend_zero_work_growth_or_changed_bits(self):
        for field,value in (("complete",False),("inputs_unchanged",False),("fractional_weight_voxels",0)):
            r=complete_fixture();name="production-allocation-smoke-v2/report.json"
            r.values[name]["pairs"][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):s.smoke(r,name,raw(True))
        for alteration in (lambda v:v["pairs"].pop(), lambda v:v["pairs"][0]["comparison"]["weight"].update(bit_mismatches=1),
            lambda v:v["pairs"][0]["candidate"].update(final_capacity=8),
            lambda v:v.update(runtime_binding_after={})):
            r=complete_fixture();name="production-allocation-smoke-v2/report.json";alteration(r.values[name])
            with self.assertRaises(ValueError):s.smoke(r,name,raw(True))

    def test_smoke_source_and_cleanup_closure(self):
        for key,value in (("source_sha256_after",s.BASELINE),("artifacts_sha256_after",{}),("environment_restored",False),
            ("cleanup_failures",[{}])):
            r=complete_fixture();name="production-allocation-smoke-v2/report.json";r.values[name][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.smoke(r,name,raw(True))

    def test_probe_old_kind_changed_native_owner_or_candidate_rejected(self):
        for key,value in (("kind","current-full-raw-final-budget-boundary-probe-v1"),("status","failed"),
            ("native_extension_sha256_after","1"*64),("owner_state_unchanged",False),("actual_device","CPU:0")):
            r=complete_fixture();name="production-final-budget-v2/report.json";r.values[name][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                s.budget_probe(r,name,s.PAIRS["chest-7"][1],r.values[s.PAIRS["chest-7"][1]])
        r=complete_fixture();name="production-final-budget-v2/report.json"
        r.values[name]["boundary"]["candidate_allocated"]=True
        with self.assertRaises(ValueError):s.budget_probe(r,name,s.PAIRS["chest-7"][1],r.values[s.PAIRS["chest-7"][1]])

    def test_observed_tool_completion_not_invented_pid_or_tree(self):
        for key,value in (("exit_code",1),("tool_completed",False),("pid_recorded",True)):
            r=complete_fixture();r.values["root-completed-exec-observations-v1.json"]["observations"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.observed_completion(r,
                "root-completed-exec-observations-v1.json","production-memory-quality-v2/chest-5.json")

    def test_old_failure_must_remain_failed(self):
        r=complete_fixture();name="production-final-budget-v1/report.json";r.values[name]["status"]="passed"
        with self.assertRaises(ValueError):s.failure(r,name)

    def test_relative_paths_refuse_windows_posix_and_traversal(self):
        for path in ("/tmp/report.json","C:/private/report.json","../report.json","x/../../report.json","\\\\host\\secret"):
            with self.subTest(path=path),self.assertRaises(ValueError):s.relative(path)

    def test_historical_windows_artifact_spelling_is_preserved_and_bounded(self):
        inventory={"scripts\\research\\benchmark_field_sessions.py":"d"*64}
        self.assertIs(s.artifact_map(inventory),inventory)
        for invalid in ({"C:\\private\\report.py":"d"*64},{"..\\escape.py":"d"*64},
            {"scripts/x.py":"d"*64,"scripts\\x.py":"d"*64}):
            with self.assertRaises(ValueError):s.artifact_map(invalid)

    def test_real_report_after_read_mutation_and_payload_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"report.json";path.write_text('{}',encoding='utf-8')
            r=s.Reports(folder);r.read("report.json");path.write_text('{"changed":true}',encoding='utf-8')
            with self.assertRaises(ValueError):r.close()
            payload=Path(folder)/"payload.npz";payload.write_bytes(b'original')
            with self.assertRaises(ValueError):r.attachment("payload.npz",s.digest(b'wrong'))


if __name__ == "__main__":
    unittest.main()
