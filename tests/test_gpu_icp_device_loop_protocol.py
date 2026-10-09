"""Stdlib-only own-loop authority, exact scope and lifetime regressions."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from scripts.research import gpu_icp_device_loop_protocol as p
from scripts.research import gpu_icp_device_loop_experiment as driver


def desc(shape, dtype="<f8", value="a"):
    return {"shape":shape,"dtype":dtype,"sha256":hashlib.sha256(value.encode()).hexdigest()}


def fixture():
    consumed = {"source":desc([4,3]),"target":desc([5,3]),"normals":desc([5,3],value="normal"),
        "seed":desc([4,4]),"configuration":{"device":"CUDA:0","max_points":1_000_000,
        "max_scratch_bytes":256*1024**2,"max_total_bytes":512*1024**2,"stages":[[.12,40],[.06,30],[.03,20]],
        "cuda_graph":True,"chunk_iterations":4,"audit_nearest":True,"audit_misses":True}}
    method = {"policy":p.POLICY,"original_math_guard_inverse":True,"stages":[[.12,40],[.06,30],[.03,20]],
        "graph_option_supported":True,"options":["--std=c++11","--fmad=false"],
        "artifacts":{"scripts/fake.cu":"a"*64},"generated_sha256":"b"*64}
    code = dict(method,device="CUDA:0",stream_ptr=42,cuda_graph=True,cupy="13",numpy="2",driver_version=12,runtime_version=12)
    terminal = {"pose":desc([4,4]),"correspondences":desc([2,2],"<i4"),
        "fitness":.5,"inlier_rmse":.001,"queries":6,"updates":3}
    stats = {"queries":6,"updates":3,"query_rows":24,"direct_hits":12,"direct_misses":12,
        "audited_hits":12,"audited_misses":12,"cpu_ambiguity_rows":0,"solve_blocks":0}
    trace = [{"stage":index//2,"query_index":index,"radius":(.12,.06,.03)[index//2],
        "target_sha256":consumed["target"]["sha256"],"packet":desc([4,8]),"corrected_ids_sha256":"c"*64}
        for index in range(6)]
    row = {"input_binding":consumed,"source_binding":code,"terminal":terminal,
        "loop_report":{"input_binding":consumed,"provenance":code,"closed":True,"failure":None,"statistics":stats},
        "native_shadow":{"passed":True,"correspondence_ids_equal":True,"transform_max_abs_delta":1e-15,
            "fitness_abs_delta":0.,"rmse_abs_delta":1e-17},
        "query_trace":trace,"query_trace_sha256":hashlib.sha256(p.canonical(trace).encode()).hexdigest()}
    binding = {"source_sha256":p.CURRENT,"method_source":method,"artifacts_sha256":method["artifacts"],
        "runtime":{},"fixed_files":{}}
    report = {"kind":p.KIND,"status":"passed","mode":"audit","binding":binding,"binding_after":binding,
        "failure":None,"cleanup_failures":[],"cleanup_passed":True,"input_bytes_unchanged":True,
        "loaded_owners_unchanged":True,"performance_attribution_valid":False,"whole_finish_authority":False,"rows":[row]}
    return report,binding


class OwnAuditTests(unittest.TestCase):
    def check(self, report, binding):
        with patch.object(p,"actual_resource_closure"):
            return p.validate_audit_report(report,binding)

    def test_exact_audited_scope_passes(self):
        report,binding=fixture()
        self.assertEqual(len(self.check(report,binding)),1)

    def test_old_or_microbatch_kinds_refused(self):
        for kind in ("actual-input-exhaustive-device-loop-graph-probe-v2","gpu-icp-seed-microbatch-audit-v1"):
            report,binding=fixture();report["kind"]=kind
            with self.assertRaises(ValueError): self.check(report,binding)

    def test_source_and_cleanup_closure_are_required(self):
        for key,value in (("cleanup_passed",False),("loaded_owners_unchanged",False),("input_bytes_unchanged",False),
                ("cleanup_failures",["sync"]),("failure","bad")):
            report,binding=fixture();report[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.check(report,binding)

    def test_before_after_resource_change_refused(self):
        report,binding=fixture();report["binding_after"]=dict(binding,source_sha256="f"*64)
        with self.assertRaises(ValueError): self.check(report,binding)

    def test_audit_cannot_claim_timing_or_finish(self):
        for key in ("performance_attribution_valid","whole_finish_authority"):
            report,binding=fixture();report[key]=True
            with self.assertRaises(ValueError): self.check(report,binding)

    def test_partial_hit_miss_shadow_refused(self):
        for key,value in (("audited_hits",11),("audited_misses",11),("query_rows",23),("solve_blocks",1)):
            report,binding=fixture();report["rows"][0]["loop_report"]["statistics"][key]=value
            with self.assertRaises(ValueError): self.check(report,binding)

    def test_native_id_or_numeric_shadow_refused(self):
        for key,value in (("correspondence_ids_equal",False),("passed",False),("transform_max_abs_delta",1e-7),
                ("rmse_abs_delta",float("nan"))):
            report,binding=fixture();report["rows"][0]["native_shadow"][key]=value
            with self.assertRaises(ValueError): self.check(report,binding)

    def test_closed_lane_inputs_and_shader_must_equal_recorded_actual(self):
        for field in ("input_binding","provenance"):
            report,binding=fixture();report["rows"][0]["loop_report"][field]={}
            with self.assertRaises((ValueError,KeyError)): self.check(report,binding)

    def test_missing_query_or_foreign_target_refused(self):
        report,binding=fixture();report["rows"][0]["query_trace"].pop()
        with self.assertRaises(ValueError): self.check(report,binding)
        report,binding=fixture();report["rows"][0]["query_trace"][0]["target_sha256"]="e"*64
        with self.assertRaises(ValueError): self.check(report,binding)

    def test_changed_query_payload_hash_refused(self):
        report,binding=fixture();report["rows"][0]["query_trace"][0]["packet"]["sha256"]="e"*64
        with self.assertRaises(ValueError): self.check(report,binding)

    def test_initial_evaluation_per_scale_cannot_be_omitted(self):
        report,binding=fixture();report["rows"][0]["terminal"]["updates"]=4
        report["rows"][0]["loop_report"]["statistics"]["updates"]=4
        with self.assertRaises(ValueError):self.check(report,binding)

    def test_reordered_or_missing_scale_refused_even_with_closed_trace_hash(self):
        for stages in ([1,0,0,1,2,2],[0,0,1,1,1,1]):
            report,binding=fixture();row=report["rows"][0]
            for query,stage in zip(row["query_trace"],stages):
                query["stage"]=stage;query["radius"]=(.12,.06,.03)[stage]
            row["query_trace_sha256"]=hashlib.sha256(p.canonical(row["query_trace"]).encode()).hexdigest()
            with self.assertRaises(ValueError):self.check(report,binding)

    def test_coarse_stage_cannot_exceed_original_limit(self):
        report,binding=fixture();row=report["rows"][0];template=row["query_trace"][0]
        stages=[0]*42+[1]*2+[2]*2
        row["query_trace"]=[dict(copy.deepcopy(template),stage=s,radius=(.12,.06,.03)[s],query_index=i) for i,s in enumerate(stages)]
        row["query_trace_sha256"]=hashlib.sha256(p.canonical(row["query_trace"]).encode()).hexdigest()
        row["terminal"].update(queries=46,updates=43)
        row["loop_report"]["statistics"].update(queries=46,updates=43,query_rows=184,
            direct_hits=92,audited_hits=92,direct_misses=92,audited_misses=92)
        with self.assertRaises(ValueError):self.check(report,binding)

    def test_duplicate_scope_not_missing_configuration_substitute(self):
        report,binding=fixture();report["rows"].append(copy.deepcopy(report["rows"][0]))
        with self.assertRaises(ValueError): self.check(report,binding)

    def test_dtype_bool_budget_or_nonfinite_terminal_refused(self):
        report,binding=fixture();report["rows"][0]["input_binding"]["seed"]["dtype"]="<f4"
        with self.assertRaises(ValueError): self.check(report,binding)
        report,binding=fixture();report["rows"][0]["input_binding"]["configuration"]["max_points"]=True
        with self.assertRaises(ValueError): self.check(report,binding)
        report,binding=fixture();report["rows"][0]["terminal"]["fitness"]=float("inf")
        with self.assertRaises(ValueError): self.check(report,binding)


class PermitTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/"audit.json"
        self.report,self.binding=fixture()
        self.path.write_text(json.dumps(self.report),encoding="utf-8")
        with patch.object(p,"actual_resource_closure"):
            self.token=p.validate_device_loop_audit(self.path,self.binding)
        self.inputs=copy.deepcopy(self.report["rows"][0]["input_binding"])
        self.inputs["configuration"]["audit_nearest"]=self.inputs["configuration"]["audit_misses"]=False
        self.source=copy.deepcopy(self.report["rows"][0]["source_binding"])
        self.source["stream_ptr"]=999

    def test_exact_own_scope_and_terminal_pass(self):
        self.assertIs(p.validate_device_loop_start(self.token,self.inputs,self.source,self.binding),self.token)
        self.assertTrue(p.validate_device_loop_terminal(self.token,self.inputs,self.source,
            self.report["rows"][0]["terminal"],self.binding))

    def test_constructed_or_foreign_tokens_refused(self):
        fake=p.DeviceLoopTimingPermit(**self.token.__dict__)
        for token in (fake,SimpleNamespace(**self.token.__dict__),None):
            with self.assertRaises(ValueError): p.validate_device_loop_start(token,self.inputs,self.source,self.binding)

    def test_closed_audit_rewrite_refused(self):
        self.path.write_text("{}",encoding="utf-8")
        with self.assertRaises(ValueError): p.validate_device_loop_start(self.token,self.inputs,self.source,self.binding)

    def test_report_change_during_validation_refused(self):
        def altered(_): self.path.write_text("{}",encoding="utf-8")
        with patch.object(p,"actual_resource_closure",side_effect=altered),self.assertRaises(ValueError):
            p.validate_device_loop_audit(self.path,self.binding)

    def test_seed_graph_chunk_dtype_or_runtime_changes_refused(self):
        for key,value in (("cuda_graph",False),("chunk_iterations",1),("max_total_bytes",1024)):
            inputs=copy.deepcopy(self.inputs);inputs["configuration"][key]=value
            with self.assertRaises(ValueError): p.validate_device_loop_start(self.token,inputs,self.source,self.binding)
        inputs=copy.deepcopy(self.inputs);inputs["seed"]["sha256"]="e"*64
        with self.assertRaises(ValueError): p.validate_device_loop_start(self.token,inputs,self.source,self.binding)
        code=dict(self.source,driver_version=13)
        with self.assertRaises(ValueError): p.validate_device_loop_start(self.token,self.inputs,code,self.binding)

    def test_timing_cannot_retain_full_observational_audit(self):
        with self.assertRaises(ValueError): p.validate_device_loop_start(self.token,
            self.report["rows"][0]["input_binding"],self.source,self.binding)

    def test_terminal_change_any_output_or_counter_refused(self):
        for key,value in (("fitness",.5000000000000001),("inlier_rmse",.002),("queries",7),("updates",4)):
            output=copy.deepcopy(self.report["rows"][0]["terminal"]);output[key]=value
            with self.assertRaises(ValueError): p.validate_device_loop_terminal(self.token,self.inputs,self.source,output,self.binding)
        output=copy.deepcopy(self.report["rows"][0]["terminal"]);output["correspondences"]["sha256"]="f"*64
        with self.assertRaises(ValueError): p.validate_device_loop_terminal(self.token,self.inputs,self.source,output,self.binding)

    def test_original_ambiguity_is_counted_but_full_audits_not_timed(self):
        stats={"queries":3,"query_rows":12,"direct_hits":5,"direct_misses":6,"cpu_ambiguity_rows":1,
            "audited_hits":0,"audited_misses":0,"flagged_rows":1,"packet_bytes":64,"solve_blocks":0}
        p.validate_timing_accounting(stats,4)
        for key,value in (("audited_hits",5),("packet_bytes",0),("flagged_rows",2),("solve_blocks",1),("query_rows",11)):
            with self.assertRaises(ValueError):p.validate_timing_accounting(dict(stats,**{key:value}),4)


class ResourcesAndLifetimeTests(unittest.TestCase):
    def test_actual_binary_and_method_bytes_cannot_be_omitted_or_changed(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(p,"ROOT",Path(folder)):
            root=Path(folder);(root/"scripts").mkdir();path=root/"scripts/fake.cu";path.write_bytes(b"code")
            binary=root/"binary";binary.write_bytes(b"native")
            _,binding=fixture();binding["method_source"]["artifacts"]={"scripts/fake.cu":p.sha(path)}
            binding["artifacts_sha256"]={"scripts/fake.cu":p.sha(path)}
            binding["fixed_files"]={str(binary):p.sha(binary)}
            binding["runtime"]={"device":"CUDA:0","open3d":"0.20.0",
                "thread_policy":{"open3d":20,"opencv":20,"omp":"8"},
                "binaries":{name:{"path":str(binary),"sha256":p.sha(binary)} for name in ("open3d","numpy","cupy","native")}}
            p.actual_resource_closure(binding)
            altered=copy.deepcopy(binding);altered["fixed_files"]={str(path):p.sha(path)}
            with self.assertRaises(ValueError):p.actual_resource_closure(altered)
            altered=copy.deepcopy(binding);altered["artifacts_sha256"]={"scripts/other.cu":"f"*64}
            with self.assertRaises(ValueError):p.actual_resource_closure(altered)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):p.actual_resource_closure(binding)

    def test_failed_lane_completion_retains_native_cache(self):
        fault=RuntimeError("completion")
        calls=[]
        loop=SimpleNamespace(close=lambda:(_ for _ in ()).throw(fault))
        cache=SimpleNamespace(close=lambda:calls.append("released"))
        with self.assertRaises(RuntimeError) as caught:driver.close_lane(loop,cache)
        self.assertIs(caught.exception,fault);self.assertEqual(calls,[])

    def test_primary_math_failure_preserved_with_cleanup_cause(self):
        primary=ValueError("math");cleanup=RuntimeError("sync")
        loop=SimpleNamespace(close=lambda:(_ for _ in ()).throw(cleanup))
        with self.assertRaises(ValueError) as caught:driver.close_lane(loop,None,primary=primary)
        self.assertIs(caught.exception,primary);self.assertIs(primary.__cause__,cleanup)

    def test_completed_lane_precedes_cache_release(self):
        calls=[]
        driver.close_lane(SimpleNamespace(close=lambda:calls.append("lane")),SimpleNamespace(close=lambda:calls.append("cache")))
        self.assertEqual(calls,["lane","cache"])

    def test_loaded_code_default_or_function_replacement_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"fake.py"
            text="def function(value=1): return value\n"+"\n".join(name+"=object()" for name in
                ("DeviceLoopICP","STAGES","POLICY","MAX_POINTS","CONTROL","RESIDENT","FLAT","CLASSIFIER","LDLT"))
            path.write_text(text,encoding="utf-8")
            module=ModuleType("device_loop_mock");module.__file__=str(path)
            exec(compile(text,str(path),"exec"),module.__dict__)
            with patch.dict(sys.modules,{module.__name__:module}):
                guard=driver.LoadedCodeGuard(module);self.assertTrue(guard.check())
                original=module.function
                module.function.__defaults__=(2,)
                with self.assertRaises(ValueError):guard.check()
                module.function.__defaults__=(1,)
                module.function=lambda:None
                with self.assertRaises(ValueError):guard.check()
                module.function=original;module.function.__code__=(lambda x=1:x+1).__code__
                with self.assertRaises(ValueError):guard.check()

    def test_other_cwd_cli_help_has_no_numeric_import_or_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,"-S",str(driver.ROOT/OWN_DRIVER),"--help"],cwd=folder,
                capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("usage:",result.stdout)
            self.assertEqual(list(Path(folder).iterdir()),[])


OWN_DRIVER="scripts/research/gpu_icp_device_loop_experiment.py"
if __name__=="__main__": unittest.main()
