"""Measured-Live checkpoint boundaries using only stdlib and artificial bytes."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.research import validate_checkpoint_finish_proof as proof
from scripts.research.compare_checkpoint_resident_finishes import compare_gates
from scripts.research.private_live_checkpoint import PreparedInputRecorder, CheckpointCaptureFailure, semantic_decisions
from scripts.research.validate_device_flat_grid_proof import DeviceFlatGridProofAuthority
from tests.test_finish_resident_contract import payload,good_comparison


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Fixture:
    """A complete tiny logical checkpoint with real non-executable NPY files."""
    def __init__(self,root):
        self.root = root
        self.folder = root/"checkpoint"
        self.folder.mkdir()
        self.nodes,self.payloads = [],[]
        self.runtime = {"source_sha256":"a"*64}
        self.artifacts = {}
        for name in proof.FINISH_ARTIFACTS:
            path = root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(name.encode())
            self.artifacts[name] = sha(name.encode())
        archive = root/"raw.zip"
        with zipfile.ZipFile(archive,"w") as data:
            data.writestr("manifest.json",json.dumps({"frames":[{}]}))
        self.scope = {"archive":{"path":str(archive),"sha256":proof.file_hash(archive)},
            "selected_indices":[0],"seed":0,"settings":{"camera":{},"sensor_calibration":{},"voxel_m":.01,
                "final_voxel_m":.005,"final_block_count":10000},"pipeline_options":{
                    "research_final_preplan":False,"research_ransac_threads":1},
            "thread_policy":dict(proof.original.THREAD_POLICY),"runtime_binding":self.runtime}
        pixels = self.ndarray("|u1",[2],b"\x01\x02")
        pair = self.node(type="tuple",items=[pixels,pixels])
        raw = self.node(type="list",items=[pair])
        pose = self.node(type="tuple",items=[{"literal":0},self.ndarray("<f8",[4,4],struct.pack("<16d",*([1.]+[0.]*15)))])
        poses = self.node(type="list",items=[pose])
        diag = self.node(type="list",items=[self.mapping({"index":{"literal":0},"success":{"literal":True}})])
        prepared = self.node(type="dict",items=[[{"literal":0},pair]])
        self.storage = pixels
        volume = self.volume()
        self.engine = {name:{"literal":None} for name in
            ("fusion_failure","input_failure","_final_vbg","mesh","point_cloud","_cuda_rgbd_odometry")}
        self.engine.update(_pose_seeds_only={"literal":False},_stage_children=self.node(type="list",items=[]),
            raw_frames=raw,poses=poses,diagnostics=diag,_processed_count={"literal":1},frame_count={"literal":1},vbg=volume)
        engine = self.mapping(self.engine)
        internal = self.node(type="tuple",items=[{"literal":0} for _ in range(624)]+[{"literal":624}])
        python = self.node(type="tuple",items=[{"literal":3},internal,{"literal":None}])
        numpy = self.node(type="tuple",items=[{"literal":"MT19937"},self.ndarray("<u4",[624],b"\0"*(624*4)),
            {"literal":624},{"literal":0},{"float_hex":0.0.hex()}])
        rng = self.mapping({"python":python,"numpy":numpy})
        graph = {"root":self.mapping({"engine":engine,"prepared_inputs":prepared,"rng_state":rng}),"nodes":self.nodes}
        engine_path = root/"scanner_server/engine.py"
        engine_path.parent.mkdir(parents=True,exist_ok=True)
        engine_path.write_text("class ScanEngine:\n    def reset(self):\n"+"".join("        self."+key+" = None\n" for key in self.engine),encoding="utf-8")
        self.manifest = {"kind":proof.CHECKPOINT_KIND,"status":"complete","source_sha256":"a"*64,
            "runtime_binding":self.runtime,"scope_base":self.scope,"producer_artifacts_sha256":self.artifacts,
            "fresh_live":{"accepted_indices":[0],"unprocessed_count":0,"pose_source":"fresh raw Live replay",
                **{key:"b"*64 for key in ("poses_sha256","semantic_decisions_sha256","raw_diagnostics_sha256",
                    "raw_frames_sha256","prepared_inputs_sha256")}},"engine_attribute_inventory":sorted(self.engine),
            "graph":graph,"payloads":self.payloads,"logical_state_sha256":"",
            "capture_report":{"observation_healthy":True,"prepared_actual_output_count":1,"stored_count":1,
                "accepted_indices":[0],"source_sha256":"a"*64,"archive_sha256":self.scope["archive"]["sha256"]}}
        self.path = self.folder/"checkpoint.json"
        self.save()

    def node(self,**node):
        self.nodes.append(node)
        return {"ref":len(self.nodes)-1}

    def mapping(self,values):
        return self.node(type="dict",items=[[{"literal":key},value] for key,value in values.items()])

    def payload(self,dtype,shape,data):
        name = f"array-{len(self.payloads):06d}.npy"
        header = repr({"descr":dtype,"fortran_order":False,"shape":tuple(shape)})+"\n"
        blob = b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header.encode()+data
        (self.folder/name).write_bytes(blob)
        item = {"dtype":dtype,"shape":shape,"payload_sha256":sha(data),"nbytes":len(data),"path":name,"sha256":sha(blob)}
        self.payloads.append(item)
        return item

    def ndarray(self,dtype,shape,data):
        stride,values = int(dtype[-1]),[]
        for n in reversed(shape):
            values.insert(0,stride)
            stride *= n
        return self.node(type="ndarray",array=self.payload(dtype,shape,data),strides=values,writable=True)

    def volume(self):
        keys = self.payload("<i4",[1,3],b"\0"*12)
        attributes = {}
        for name,channels in (("tsdf",1),("weight",1),("color",3)):
            data = b"\0"*(16**3*channels*4)
            attributes[name] = {"channels":channels,"payload_sha256":sha(data),
                "chunks":[self.payload("<f4",[1,16,16,16,channels],data)]}
        descriptor = {key:keys[key] for key in ("dtype","shape","payload_sha256","nbytes")}
        return self.node(type="vbg",capacity=10000,size=1,resolution=16,voxel_size_hex=.01.hex(),device="CUDA:0",
            keys=keys,attributes=attributes,logical_sha256=proof.canonical_hash({"keys":descriptor,
                "attributes":{k:v["payload_sha256"] for k,v in attributes.items()},"capacity":10000}))

    def save(self):
        self.manifest["logical_state_sha256"] = proof.canonical_hash(proof.logical_graph(self.manifest["graph"]))
        self.path.write_text(json.dumps(self.manifest),encoding="utf-8")

    def validate(self):
        with patch.object(proof,"ROOT",self.root),patch.object(proof,"current_core_hash",return_value="a"*64):
            return proof.validate_checkpoint_manifest(self.path,self.runtime,self.artifacts,self.scope)


class CheckpointContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def reject(self):
        self.fixture.save()
        with self.assertRaises(proof.GridProofError):
            self.fixture.validate()

    def test_closed_numeric_checkpoint_and_defensive_authority(self):
        authority = self.fixture.validate()
        self.assertEqual(authority.record["sha256"],proof.file_hash(self.fixture.path))
        changed = authority.manifest
        changed["fresh_live"]["poses_sha256"] = "c"*64
        self.assertNotEqual(changed,authority.manifest)

    def test_payload_path_escape_even_with_valid_file_hash(self):
        self.fixture.payloads[0]["path"] = "../outside.npy"
        self.reject()

    def test_duplicate_payload_and_orphan_graph_rejected(self):
        self.fixture.payloads.append(copy.deepcopy(self.fixture.payloads[0]))
        self.reject()

    def test_orphan_node_rejected(self):
        self.fixture.node(type="device",value="CPU:0")
        self.reject()

    def test_undeclared_graph_payload_rejected(self):
        self.fixture.payloads.pop()
        self.reject()

    def test_npy_object_dtype_and_header_extent_rejected(self):
        self.fixture.payloads[0]["dtype"] = "|O8"
        self.reject()

    def test_npy_file_changed_rejected(self):
        path = self.fixture.folder/self.fixture.payloads[0]["path"]
        path.write_bytes(path.read_bytes()+b"trailing")
        self.reject()

    def test_graph_descriptor_vs_payload_table_rejected(self):
        self.fixture.nodes[0]["array"] = dict(self.fixture.payloads[0],payload_sha256="c"*64)
        self.reject()

    def test_invalid_reference_unknown_node_and_constructor_rejected(self):
        for mutation in ({"type":"executable"},{"type":"registered","name":"untrusted.Class","state":{"ref":0}}):
            with self.subTest(mutation=mutation):
                old = self.fixture.nodes[0]
                self.fixture.nodes[0] = mutation
                self.reject()
                self.fixture.nodes[0] = old

    def test_unknown_engine_attribute_rejected(self):
        graph = self.fixture.manifest["graph"]
        engine = proof.dictionary(graph,graph["root"])["engine"]
        self.fixture.nodes[engine["ref"]]["items"].append([{"literal":"unknown_state"},{"literal":3}])
        self.fixture.manifest["engine_attribute_inventory"].append("unknown_state")
        self.fixture.manifest["engine_attribute_inventory"].sort()
        self.reject()

    def test_transient_failure_and_archived_seed_rejected(self):
        graph = self.fixture.manifest["graph"]
        engine = self.fixture.nodes[proof.dictionary(graph,graph["root"])["engine"]["ref"]]
        for name,value in (("fusion_failure","failure"),("_pose_seeds_only",True),("_processed_count",0)):
            pair = next(x for x in engine["items"] if x[0]["literal"] == name)
            old = pair[1]
            pair[1] = {"literal":value}
            self.reject()
            pair[1] = old

    def test_observer_partial_capture_rejected(self):
        self.fixture.manifest["capture_report"]["observation_healthy"] = False
        self.reject()

    def test_vbg_capacity_chunk_and_bit_digest_rejected(self):
        volume = next(n for n in self.fixture.nodes if n["type"] == "vbg")
        for key,value in (("capacity",0),("resolution",8),("logical_sha256","c"*64)):
            old = volume[key]
            volume[key] = value
            self.reject()
            volume[key] = old

    def test_storage_alias_offsets_and_strides_preserved(self):
        owner = self.fixture.storage
        view = self.fixture.node(type="ndarray",array=self.fixture.payload("|u1",[1],b"\x02"),
            strides=[1],writable=False,storage=owner,storage_offset=1)
        engine_ref = proof.dictionary(self.fixture.manifest["graph"],self.fixture.manifest["graph"]["root"])["engine"]
        engine = self.fixture.nodes[engine_ref["ref"]]
        # Reuse a source-known cached field and include it in the source fixture.
        engine["items"].append([{"literal":"model_pcd"},view])
        self.fixture.manifest["engine_attribute_inventory"].append("model_pcd")
        self.fixture.manifest["engine_attribute_inventory"].sort()
        with (self.fixture.root/"scanner_server/engine.py").open("a") as source:
            source.write("        self.model_pcd = None\n")
        self.fixture.save()
        self.fixture.validate()
        self.fixture.nodes[view["ref"]]["storage_offset"] = 2
        self.reject()

    def test_rng_missing_or_wrong_policy_rejected(self):
        rng_ref = proof.dictionary(self.fixture.manifest["graph"],self.fixture.manifest["graph"]["root"])["rng_state"]
        numpy_ref = proof.dictionary(self.fixture.manifest["graph"],rng_ref)["numpy"]
        self.fixture.nodes[numpy_ref["ref"]]["items"][0] = {"literal":"PCG64"}
        self.reject()

    def test_materialization_requires_all_actual_volume_and_prepared_bits(self):
        authority = self.fixture.validate()
        live = authority.manifest["fresh_live"]
        volume = next(n for n in self.fixture.nodes if n["type"] == "vbg")
        check = {"passed":True,"checkpoint_sha256":authority.sha256,"logical_state_equal":True,
            "logical_state_sha256":authority.logical_state_sha256,"prepared_input_parity_passed":True,"prepared_input_checks":1,
            **{k:live[k] for k in ("poses_sha256","raw_diagnostics_sha256","raw_frames_sha256","prepared_inputs_sha256")},
            "volume_checks":[{"passed":True,"capacity":10000,"size":1,"keys_sha256":volume["keys"]["payload_sha256"],
                "attributes_sha256":{k:v["payload_sha256"] for k,v in volume["attributes"].items()}}]}
        report = {"checkpoint":authority.record,"checkpoint_after":authority.record,"checkpoint_scope_base":self.fixture.scope,
            "checkpoint_verification":check}
        proof.validate_materialization(report,authority)
        check["volume_checks"][0]["capacity"] = 1
        with self.assertRaises(proof.GridProofError):
            proof.validate_materialization(report,authority)

    def test_scope_must_match_raw_semantic_and_pose_checkpoint_exactly(self):
        authority = self.fixture.validate()
        live = authority.manifest["fresh_live"]
        scope = dict(self.fixture.scope,checkpoint=authority.record,live={"accepted_indices":[0],
            "decisions_sha256":live["semantic_decisions_sha256"],**{k:live[k] for k in
            ("poses_sha256","raw_diagnostics_sha256","raw_frames_sha256","prepared_inputs_sha256","unprocessed_count","pose_source")}})
        proof.validate_checkpoint_scope(scope,authority,{})
        scope["live"]["poses_sha256"] = "c"*64
        with self.assertRaises(proof.GridProofError):
            proof.validate_checkpoint_scope(scope,authority,{})

    def test_success_duration_semantics_preserve_raw_failure_and_numeric_evidence(self):
        row = {"index":0,"success":True,"method":"ICP","message":"Frame 1 (fitness=0.800, RMSE=0.001, 100ms)"}
        other = dict(row,message="Frame 1 (fitness=0.800, RMSE=0.001, 200ms)")
        self.assertEqual(semantic_decisions([row]),semantic_decisions([other]))
        self.assertNotEqual(proof.canonical_hash(row),proof.canonical_hash(other))
        self.assertNotEqual(semantic_decisions([row]),semantic_decisions([dict(other,message=other["message"].replace("0.800","0.700"))]))
        self.assertNotEqual(semantic_decisions([dict(row,success=False)]),semantic_decisions([dict(other,success=False)]))

    def test_only_original_bundle_elapsed_field_is_nonsemantic(self):
        native = [{"gate_index":0,"function":"bundle_adjustment.propose_bundle_poses",
            "result":{"elapsed_ms":1.,"time_budget_s":4.,"applied":False,"reason":"unsupported"}}]
        audit = copy.deepcopy(native)
        audit[0]["result"]["elapsed_ms"] = 9.
        self.assertTrue(compare_gates(native,audit)[0])
        audit[0]["result"]["time_budget_s"] = 5.
        self.assertFalse(compare_gates(native,audit)[0])
        audit = copy.deepcopy(native)
        audit[0]["function"] = native[0]["function"] = "other.function"
        audit[0]["result"]["elapsed_ms"] = 9.
        self.assertFalse(compare_gates(native,audit)[0])

    def test_observer_latches_masked_error_and_restores_original(self):
        rgb,raw = object(),object()
        original = lambda *args,**kwargs: (object(),object())
        engine = SimpleNamespace(_prepare_input=original,settings="expected",raw_frames=[(rgb,raw)])
        recorder = PreparedInputRecorder(engine)
        with self.assertRaises(CheckpointCaptureFailure) as caught:
            with recorder:
                try:
                    engine._prepare_input(rgb,raw,"wrong")
                except BaseException:
                    raise RuntimeError("cleanup replaced the typed capture fault")
        self.assertIs(caught.exception,recorder.failure)
        self.assertIs(engine._prepare_input,original)
        self.assertTrue(any("cleanup" in note for note in caught.exception.__notes__))

    def test_observer_refuses_retry_after_swallowed_failure(self):
        original = lambda *args,**kwargs: (object(),object())
        engine = SimpleNamespace(_prepare_input=original,settings=1,raw_frames=[])
        recorder = PreparedInputRecorder(engine)
        recorder.__enter__()
        try:
            with self.assertRaises(CheckpointCaptureFailure):
                engine._prepare_input(object(),object())
            with self.assertRaises(CheckpointCaptureFailure) as retry:
                engine._prepare_input(object(),object())
            self.assertIs(retry.exception,recorder.failure)
        finally:
            with self.assertRaises(CheckpointCaptureFailure):
                recorder.__exit__(None,None,None)
        self.assertIs(engine._prepare_input,original)

    def test_changed_repeated_actual_output_is_fatal_with_no_mutated_return(self):
        rgb,raw = object(),object()
        outputs = ([1],[2])
        original = lambda *args,**kwargs: outputs
        engine = SimpleNamespace(_prepare_input=original,settings=1,raw_frames=[(rgb,raw)])
        fake_numpy = SimpleNamespace(array=lambda value,**kwargs:list(value),array_equal=lambda a,b:a == b)
        recorder = PreparedInputRecorder(engine)
        with patch.dict(sys.modules,{"numpy":fake_numpy}):
            with self.assertRaises(CheckpointCaptureFailure):
                with recorder:
                    actual = engine._prepare_input(rgb,raw)
                    self.assertIs(actual[0],outputs[0])
                    self.assertIs(actual[1],outputs[1])
                    outputs[0][0] = 3
                    engine._prepare_input(rgb,raw)
        self.assertEqual(recorder.inputs[0],([1],[2]))
        self.assertIs(engine._prepare_input,original)

    def test_modules_and_relocated_help_need_no_numerical_packages(self):
        root = Path(__file__).resolve().parents[1]
        command = "import sys; import scripts.research.validate_checkpoint_finish_proof; import scripts.research.compare_checkpoint_resident_finishes; assert not any(x in sys.modules for x in ('numpy','open3d','cupy','scanner_server.engine'))"
        result = subprocess.run([sys.executable,"-S","-c",command],cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for name in ("compare_checkpoint_resident_finishes.py","profile_checkpoint_resident_finish.py"):
            result = subprocess.run([sys.executable,"-S",str(root/"scripts/research"/name),"--help"],
                cwd=self.temp.name,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("usage:",result.stdout)


class CheckpointFinishEnvelopeContracts(unittest.TestCase):
    """New envelopes are real byte files; native math compatibility is mocked."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))
        f = self.fixture
        f.runtime["artifacts_sha256"] = {}
        f.save()
        self.checkpoint = f.validate()
        live = f.manifest["fresh_live"]
        self.scope = dict(f.scope,checkpoint=self.checkpoint.record,live={"accepted_indices":[0],
            "decisions_sha256":live["semantic_decisions_sha256"],**{k:live[k] for k in
            ("poses_sha256","raw_diagnostics_sha256","raw_frames_sha256","prepared_inputs_sha256","unprocessed_count","pose_source")}})
        self.component_proof = {name:self.write(name+".json",{"artificial":True}) for name in ("synthetic","bridge")}
        self.component = DeviceFlatGridProofAuthority(proof.canonical_hash(f.runtime),"1"*64,
            self.component_proof["synthetic"]["sha256"],self.component_proof["bridge"]["sha256"],
            frozenset({"0"*64}),(),json.dumps(f.runtime))
        self.call = payload()
        signature = proof.original.call_signature(self.call)
        counts = {key:0 for key in proof.original.QUERY_COUNTERS}
        counts.update(query_rows=10,direct_gpu_hits=6,declared_gpu_misses=4,audited_hits=6,audited_misses=4,
            device_calls=1,device_flagged_rows=10,device_query_download_rows=10,device_query_download_bytes=640)
        self.match = {"event":"match","complete":True,"call_index":0,"call_inputs":self.call,"call_signature":signature,
            "inputs_after_resident":signature,"inputs_before_shadow":signature,"inputs_after_shadow":signature,
            "cpu_shadow":good_comparison(),"resident_full_call_fallbacks":0,"query_statistics_delta":counts}
        self.gate = {"event":"gate","complete":True,"gate_index":0,"function":"verify","result":{"applied":True}}
        self.candidate_trace = self.trace("candidate.jsonl",[self.match,self.gate])
        self.native_trace = self.trace("native.jsonl",[self.match,self.gate])
        volume = next(n for n in f.nodes if n["type"] == "vbg")
        self.verification = {"passed":True,"checkpoint_sha256":self.checkpoint.sha256,"logical_state_equal":True,
            "logical_state_sha256":self.checkpoint.logical_state_sha256,"prepared_input_parity_passed":True,"prepared_input_checks":1,
            **{k:live[k] for k in ("poses_sha256","raw_diagnostics_sha256","raw_frames_sha256","prepared_inputs_sha256")},
            "volume_checks":[{"passed":True,"capacity":10000,"size":1,"keys_sha256":volume["keys"]["payload_sha256"],
                "attributes_sha256":{k:v["payload_sha256"] for k,v in volume["attributes"].items()}}]}
        registration = {"complete":True,"restored":True,"calls":1,"gate_calls":1,"input_immutability_passed":True,
            "call_signatures":[signature],"target_digests":["a"*64],"cpu_shadow_calls":1,"cpu_shadow_failures":0,
            "full_cpu_fallback_calls":0,"retrieval_statistics":counts,"resident":{"statistics":{"calls":1}}}
        self.candidate,self.native = (self.profile(name) for name in ("candidate","native"))
        self.audit = self.sidecar("audit",self.candidate,self.candidate_trace,registration)
        self.native_sidecar = self.sidecar("native",self.native,self.native_trace,
            dict(registration,cpu_shadow_calls=0,resident=None,retrieval_statistics={}))
        self.native_record = self.write("native.resident.json",self.native_sidecar)
        self.audit_record = self.write("audit.resident.json",self.audit)
        self.quality = {"kind":proof.QUALITY_KIND,"status":"passed","audit_report_sha256":self.audit_record["sha256"],
            "audit":self.audit_record,"native_sidecar":self.native_record,"candidate":self.candidate,"native":self.native,
            "candidate_trace":self.candidate_trace,"native_trace":self.native_trace,"checkpoint":self.checkpoint.record,
            "scope_binding_sha256":proof.canonical_hash(self.scope),"live_pose_signatures_equal":True,"live_decision_signatures_equal":True,
            "same_ordered_gate_decisions":True,"ordered_gate_evidence_passed":True,
            "comparison_artifact_sha256":f.artifacts["scripts/research/compare_checkpoint_resident_finishes.py"],
            "comparison":{"candidate_report_sha256":self.candidate["sha256"],"baseline_report_sha256":self.native["sha256"],
                "same_mesh_success":True,"same_accepted_indices":True,"triangle_surface_metrics_vs_cpu":{
                    "threshold_m":.005,"samples_per_surface":30000,"alignment":"fixed input coordinate frame; no scale or trajectory fitting",
                    "surface_p95_m":.0001,"precision":1.,"completeness":1.}}}
        self.quality_record = self.write("quality.json",self.quality)

    def write(self,name,value):
        path = self.fixture.root/name
        path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8")
        return {"path":str(path),"sha256":proof.file_hash(path)}

    def trace(self,name,rows):
        path = self.fixture.root/name
        path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
        return {"path":str(path),"sha256":proof.file_hash(path),"rows":len(rows)}

    def profile(self,name):
        geometry = self.fixture.root/(name+".npz")
        geometry.write_bytes(b"artificial geometry")
        profile = {"input_sha256":self.scope["archive"]["sha256"],"source_sha256":"a"*64,"frames":1,
            "input_changed_during_profile":False,"source_changed_during_profile":False,"finish_requested":True,
            "mesh_built":True,"pose_seeds_used":False,"selected_indices":[0],"seed":0,"settings":self.scope["settings"],
            "pipeline_options":{},"research_pipeline_options":self.scope["pipeline_options"],"accepted_indices_before_finish":[0],"accepted_indices":[0],
            "final_reconstruction":{"voxel_m":.005},"geometry":{"vertices":3,"triangles":1,"artifact":str(geometry)}}
        record = self.write(name+".json",profile)
        return dict(record,geometry_path=str(geometry),geometry_sha256=proof.file_hash(geometry))

    def sidecar(self,mode,profile,trace,registration):
        f = self.fixture
        return {"kind":proof.KIND,"mode":mode,"status":"complete","performance_attribution_valid":mode != "audit",
            "runner_hooks_restored":True,"source_sha256":"a"*64,"source_sha256_after":"a"*64,
            "archive_sha256":f.scope["archive"]["sha256"],"archive_sha256_after":f.scope["archive"]["sha256"],
            "runtime_binding":f.runtime,"runtime_binding_after":f.runtime,"artifacts_sha256":f.artifacts,"artifacts_sha256_after":f.artifacts,
            "component_proof":self.component_proof,"component_proof_after":self.component_proof,"scope_binding":self.scope,
            "scope_binding_sha256":proof.canonical_hash(self.scope),"checkpoint":self.checkpoint.record,"checkpoint_after":self.checkpoint.record,
            "checkpoint_scope_base":f.scope,"checkpoint_verification":self.verification,"registration":registration,
            "trace":trace,"profile":profile,"cpu_query_auditor":{"mode":"scalar"},"research_ransac_threads":1,
            "ransac_proposals":{"policy":proof.RANSAC_POLICY,"requested_threads":1,"outside_threads":20,
                "calls":[],"call_count":0,"failure":None,"hooks_restored":True}}

    def rewrite(self):
        self.audit_record = self.write("audit.resident.json",self.audit)
        self.quality.update(audit=self.audit_record,audit_report_sha256=self.audit_record["sha256"])
        self.quality_record = self.write("quality.json",self.quality)

    def validate(self):
        with patch.object(proof,"ROOT",self.fixture.root),patch.object(proof,"current_core_hash",return_value="a"*64),\
             patch.object(proof,"validate_configuration",return_value={}),patch.object(proof,"validate_current_artifacts"),\
             patch.object(proof,"resident_report_checks",return_value={}):
            return proof.validate_checkpoint_finish_proof(self.audit_record["path"],self.component,self.scope,
                self.fixture.artifacts,quality_path=self.quality_record["path"],checkpoint_authority=self.checkpoint)

    def test_new_envelope_mints_contained_exact_original_type_with_fresh_targets(self):
        token = self.validate()
        self.assertIs(type(token),proof.CheckpointFinishProofAuthority)
        self.assertIs(type(token.registration_authority),proof.original.FinishResidentProofAuthority)
        self.assertEqual(token.registration_authority.target_digests,frozenset({"a"*64}))
        self.assertEqual(token.registration_authority.expected_call_signatures,(self.match["call_signature"],))
        proof.original.validate_expected_call(token.registration_authority,0,self.call)
        with self.assertRaises(proof.GridProofError):
            proof.original.validate_expected_call(token,0,self.call)

    def test_old_kind_incomplete_restore_or_checkpoint_drift_cannot_mint(self):
        original = copy.deepcopy(self.audit)
        for mutate in (lambda r:r.update(kind=proof.original.KIND),
                       lambda r:r["checkpoint_verification"].update(prepared_input_checks=0),
                       lambda r:r["checkpoint_after"].update(sha256="c"*64),
                       lambda r:r["registration"].update(cpu_shadow_calls=0),
                       lambda r:r["registration"].update(restored=False)):
            self.audit = copy.deepcopy(original)
            mutate(self.audit)
            self.rewrite()
            with self.assertRaises(proof.GridProofError):
                self.validate()

    def test_separate_native_same_checkpoint_gate_evidence_is_checked(self):
        native = copy.deepcopy(self.native_sidecar)
        native["scope_binding"]["live"]["poses_sha256"] = "c"*64
        native["scope_binding_sha256"] = proof.canonical_hash(native["scope_binding"])
        self.quality["native_sidecar"] = self.write("native.resident.json",native)
        self.quality_record = self.write("quality.json",self.quality)
        with self.assertRaises(proof.GridProofError):
            self.validate()

    def test_ordered_original_gate_change_fails_even_if_quality_claims_pass(self):
        gate = dict(self.gate,result={"applied":False})
        trace = self.trace("native.jsonl",[self.match,gate])
        self.native_sidecar["trace"] = trace
        self.quality["native_sidecar"] = self.write("native.resident.json",self.native_sidecar)
        self.quality["native_trace"] = trace
        self.quality_record = self.write("quality.json",self.quality)
        with self.assertRaises(proof.GridProofError):
            self.validate()

    def test_unvalidated_bulk_or_relaxed_surface_cannot_mint(self):
        self.audit["cpu_query_auditor"] = {"mode":"bulk-shadow"}
        self.rewrite()
        with self.assertRaises(proof.GridProofError):
            self.validate()
        self.audit["cpu_query_auditor"] = {"mode":"scalar"}
        for key,value in (("precision",.9989),("surface_p95_m",.0005001),("alignment","best fit")):
            self.quality["comparison"]["triangle_surface_metrics_vs_cpu"][key] = value
            self.rewrite()
            with self.assertRaises(proof.GridProofError):
                self.validate()

    def test_changed_closed_native_geometry_fails(self):
        Path(self.native["geometry_path"]).write_bytes(b"changed")
        with self.assertRaises(proof.GridProofError):
            self.validate()

    def test_bulk_compatibility_never_replaces_full_field_shadow_or_target_scope(self):
        # This tests only CPU-auditor composition; native compatibility is the
        # separate complete-old proof validator's responsibility.
        from scripts.research.validate_bulk_resident_audit import BulkResidentAuditAuthority
        runtime = {key:{"artificial":key} for key in ("source_sha256","component_source_sha256","domain",
            "thread_policy","resident_math","cache_policy","gpu","versions")}
        runtime["resident_configuration"] = {"device":"CUDA:0"}
        dual = self.write("dual.json",{"artificial_complete_dual":True})
        api = self.write("api.json",{"artificial_api":True})
        token = BulkResidentAuditAuthority(dual["path"],dual["sha256"],api["path"],api["sha256"],
            json.dumps(runtime),"1"*64,tuple(self.fixture.artifacts.items()))
        audit = {"cpu_query_auditor":{"mode":"bulk-shadow","policy":proof.BULK_POLICY,"proof":dual,
            "api_proof":api,"native_runtime":{"artificial":True},"chunk_rows":65536},
            "referenced_bulk_audit_proof":dual,"referenced_bulk_audit_proof_after":dual,
            "bulk_runtime_binding":runtime,"bulk_runtime_binding_after":runtime,"registration":{"retrieval_statistics":{}}}
        stats = dict(self.match["query_statistics_delta"],bulk_shadow_queries=8,bulk_shadow_hits=5,
            bulk_shadow_misses=3,bulk_domain_scalar_queries=2,dual_scalar_bulk_queries=0,dual_id_mismatches=0,
            dual_metric_bit_mismatches=0)
        audit["registration"]["retrieval_statistics"] = stats
        match = dict(self.match,query_statistics_delta=stats)
        with patch.object(proof,"validate_authority_for_bulk_shadow",return_value=True):
            proof.validate_field_cpu_auditor(audit,[match],token,runtime,self.fixture.artifacts,{})
            # Omitted CPU shadows and changed runtime metadata fail rather than
            # turning a legacy target token into new field acceptance authority.
            stats["bulk_domain_scalar_queries"] = 1
            with self.assertRaises(proof.GridProofError):
                proof.validate_field_cpu_auditor(audit,[match],token,runtime,self.fixture.artifacts,{})


if __name__ == "__main__":
    unittest.main()
