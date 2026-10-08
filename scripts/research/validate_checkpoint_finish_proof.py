"""Non-executable, closed measured-Live checkpoint and Finish proof boundaries.

This module imports no numerical runtime. A checkpoint is private research
state, not a session export or permission to use archived poses. Timing needs
a new complete audit and independent native Finish from the same checkpoint.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ast
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct

from scripts.research import validate_finish_resident_proof as original
from scripts.research.validate_bulk_resident_audit import (
    BULK_ARTIFACTS, BulkResidentAuditAuthority, POLICY as BULK_POLICY,
    validate_authority_for_bulk_shadow)
from scripts.research.validate_device_flat_grid_proof import (
    DeviceFlatGridProofAuthority, digest, normalized_artifacts, current_core_hash,
    validate_configuration, validate_current_artifacts, resident_report_checks,
    device_audit_coverage)
from scripts.research.archive.validate_uniform_grid_proof import (
    GridProofError, canonical_hash, file_hash, positive, require, zero)

CHECKPOINT_KIND = "private-fresh-raw-live-checkpoint-v2"
KIND = "offline-full-finish-device-flat-resident-checkpoint-proposal-threads-v3"
QUALITY_KIND = "offline-full-finish-device-flat-checkpoint-proposal-threads-quality-v3"
RANSAC_POLICY = "original-fragment-global-seed-scoped-open3d-threads-v1"
CHECKPOINT_ARTIFACTS = ("scripts/research/private_live_checkpoint.py",
    "scripts/research/profile_checkpoint_resident_finish.py",
    "scripts/research/validate_checkpoint_finish_proof.py",
    "scripts/research/compare_checkpoint_resident_finishes.py")
FINISH_ARTIFACTS = original.FINISH_ARTIFACTS + CHECKPOINT_ARTIFACTS + BULK_ARTIFACTS
MAX_ARRAY_BYTES = 2 * 1024**3
MAX_PAYLOAD_BYTES = 8 * 1024**3
REGISTERED = {"shared.config.ScanPreset", "shared.settings.ScanSettings",
    "shared.settings.CameraCalibration", "shared.sensor_calibration.SensorCalibration",
    "scanner_server.appearance.Features", "scanner_server.tracking_cache.TargetPyramid",
    "scanner_server.tracking_cache.SourcePyramid"}
PREPARATIONS = {"scanner_server.cuda_input.InputPreparation",
    "scanner_server.cuda_confidence.ConfidencePreparation"}
NODE_KEYS = {
    "dict": {"type", "items"}, "ordered_dict": {"type", "items"},
    "list": {"type", "items"}, "tuple": {"type", "items"},
    "set": {"type", "items"}, "frozenset": {"type", "items"},
    "ndarray": {"type", "array", "strides", "writable"},
    "numpy_scalar": {"type", "array"}, "cupy": {"type", "device", "array"},
    "device": {"type", "value"}, "tensor": {"type", "device", "array"},
    "intrinsic": {"type", "width", "height", "matrix"},
    "cloud": {"type", "attributes"}, "image": {"type", "array"},
    "rgbd": {"type", "color", "depth"}, "fpfh": {"type", "data"},
    "tensor_cloud": {"type", "device", "attributes"},
    "registered": {"type", "name", "state"},
    "preparation": {"type", "name", "state"},
    "native_rgbd": {"type", "device", "state"}, "matcher": {"type", "device", "bank"},
    "vbg": {"type", "capacity", "size", "resolution", "voxel_size_hex", "device",
            "keys", "attributes", "logical_sha256"}}


@dataclass(frozen=True)
class CheckpointAuthority:
    path: str
    sha256: str
    logical_state_sha256: str
    manifest_json: str
    closed_files: tuple

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    @property
    def record(self):
        return {"path": self.path, "sha256": self.sha256,
                "logical_state_sha256": self.logical_state_sha256}


@dataclass(frozen=True)
class CheckpointFinishProofAuthority:
    # Composition is deliberate: original dispatch requires the exact v1 type.
    registration_authority: original.FinishResidentProofAuthority
    checkpoint_authority: CheckpointAuthority
    checkpoint_audit_sha256: str
    checkpoint_quality_sha256: str


def logical_graph(value):
    if isinstance(value, dict):
        return {key: logical_graph(item) for key, item in value.items() if key not in ("path", "sha256")}
    if isinstance(value, list):
        return [logical_graph(item) for item in value]
    return value


def array_descriptor(item):
    require(isinstance(item, dict) and set(item) ==
            {"dtype", "shape", "payload_sha256", "nbytes", "path", "sha256"},
            "Incomplete checkpoint array descriptor")
    match = re.fullmatch(r"[<>=|]([biuf])([1248])", item["dtype"])
    require(match is not None and (match[1] != "b" or match[2] == "1"),
            "Checkpoint contains an unsupported or executable dtype")
    shape = item["shape"]
    require(isinstance(shape, list) and len(shape) <= 8 and
            all(type(n) is int and 0 <= n <= MAX_ARRAY_BYTES for n in shape), "Malformed array shape")
    require(type(item["nbytes"]) is int and item["nbytes"] == math.prod(shape) * int(match[2])
            and 0 <= item["nbytes"] <= MAX_ARRAY_BYTES, "Array bytes exceed shape or checkpoint bound")
    require(digest(item["sha256"]) and digest(item["payload_sha256"]), "Array fingerprint absent")
    require(isinstance(item["path"], str) and re.fullmatch(r"array-[0-9]{6}\.npy", item["path"]),
            "Checkpoint payload must be a unique contained numeric NPY file")
    return int(match[2])


def npy_header(path, item):
    """Inspect the non-executable NPY header without NumPy or pickle."""
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "Invalid checkpoint NPY magic")
        version = stream.read(2)
        require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Unsupported NPY version")
        count = 2 if version == b"\x01\x00" else 4
        raw = stream.read(count)
        require(len(raw) == count, "Truncated NPY length")
        size = int.from_bytes(raw, "little")
        require(0 < size <= 65536, "Unbounded NPY header")
        header = ast.literal_eval(stream.read(size).decode("utf-8" if version == b"\x03\x00" else "latin1"))
        require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}
                and header["descr"] == item["dtype"] and type(header["fortran_order"]) is bool
                and type(header["shape"]) is tuple and list(header["shape"]) == item["shape"],
                "NPY header disagrees with closed numeric descriptor")
        offset = stream.tell()
        require(path.stat().st_size == offset + item["nbytes"], "Truncated/trailing NPY data")
        if not header["fortran_order"] or len(item["shape"]) <= 1:
            value = hashlib.sha256()
            for block in iter(lambda: stream.read(1024**2), b""):
                value.update(block)
            require(value.hexdigest() == item["payload_sha256"], "NPY logical payload bits changed")
        # The original codec verifies C-order payload bits of Fortran arrays on
        # materialization; file SHA and header bind them before any native load.
        return offset, header["fortran_order"]


def validate_graph(manifest, folder, tracked):
    graph = manifest["graph"]
    require(isinstance(graph, dict) and set(graph) == {"root", "nodes"}, "Malformed checkpoint graph")
    nodes = graph["nodes"]
    require(isinstance(nodes, list) and 0 < len(nodes) <= 100000, "Unbounded checkpoint node graph")
    table, offsets = {}, {}
    require(isinstance(manifest["payloads"], list) and manifest["payloads"], "Checkpoint payloads absent")
    total = 0
    for item in manifest["payloads"]:
        array_descriptor(item)
        name = item["path"]
        require(name not in table, "Aliased duplicate payload path")
        path = (folder / name).resolve(strict=True)
        require(path.parent == folder and not (folder/name).is_symlink(), "Payload escaped checkpoint folder")
        require(file_hash(path) == item["sha256"], "Closed NPY file changed")
        tracked[path] = item["sha256"]
        offsets[name] = npy_header(path, item)
        table[name] = item
        total += item["nbytes"]
    require(total <= MAX_PAYLOAD_BYTES, "Checkpoint total payload bound exceeded")
    used, seen = set(), set()

    def array(item):
        array_descriptor(item)
        require(table.get(item["path"]) == item, "Graph array is absent/different in payload table")
        used.add(item["path"])

    def reference(item):
        require(isinstance(item, dict) and len(item) == 1, "Malformed graph reference/literal")
        if "literal" in item:
            require(item["literal"] is None or type(item["literal"]) in (str, bool, int), "Executable graph literal")
            return
        if "float_hex" in item:
            require(isinstance(item["float_hex"], str) and math.isfinite(float.fromhex(item["float_hex"])),
                    "Nonfinite graph scalar")
            return
        index = item.get("ref")
        require(type(index) is int and 0 <= index < len(nodes), "Invalid graph reference")
        if index in seen:
            return
        seen.add(index)
        node = nodes[index]
        require(isinstance(node, dict) and node.get("type") in NODE_KEYS, "Unknown checkpoint node type")
        kind = node["type"]
        expected = NODE_KEYS[kind]
        require(set(node) == expected or (kind == "ndarray" and set(node) == expected | {"storage","storage_offset"}),
                "Incomplete/unknown checkpoint node fields")
        if kind in ("dict", "ordered_dict"):
            require(isinstance(node["items"], list), "Malformed dictionary graph")
            for pair in node["items"]:
                require(isinstance(pair, list) and len(pair) == 2, "Malformed dictionary pair")
                reference(pair[0]); reference(pair[1])
        elif kind in ("list", "tuple", "set", "frozenset"):
            require(isinstance(node["items"], list), "Malformed sequence graph")
            for child in node["items"]:
                reference(child)
        elif kind in ("registered", "preparation", "native_rgbd"):
            if kind != "native_rgbd":
                require(node["name"] in (REGISTERED if kind == "registered" else PREPARATIONS), "Unknown constructor name")
            reference(node["state"])
        elif kind == "matcher":
            reference(node["bank"])
        elif kind == "rgbd":
            reference(node["color"]); reference(node["depth"])
        elif kind in ("cloud", "tensor_cloud"):
            require(isinstance(node["attributes"], dict), "Malformed cloud attributes")
            if kind == "cloud":
                require(set(node["attributes"]) == {"points", "normals", "colors", "covariances"}, "Cached CPU cloud attributes incomplete")
            for value in node["attributes"].values():
                (array if kind == "cloud" else reference)(value)
        elif kind == "vbg":
            validate_volume(node, array, folder, offsets)
        elif kind != "device":
            array(node["matrix"] if kind == "intrinsic" else node["data"] if kind == "fpfh" else node["array"])
            if kind == "ndarray":
                shape, strides = node["array"]["shape"], node["strides"]
                require(isinstance(strides, list) and len(strides) == len(shape)
                        and all(type(n) is int for n in strides) and type(node["writable"]) is bool,
                        "Malformed original ndarray layout/flags")
                span = array_descriptor(node["array"]) + sum((n-1)*abs(step) for n, step in zip(shape,strides)) if math.prod(shape) else 0
                require(span <= MAX_ARRAY_BYTES, "Original stride span exceeds owned-memory bound")
                if "storage" in node:
                    reference(node["storage"])
                    owner = nodes[node["storage"]["ref"]]
                    require(owner["type"] == "ndarray" and "storage" not in owner,
                            "Array backing storage must be its original contiguous owner")
                    count = owner["array"]["nbytes"]
                    offset = node["storage_offset"]
                    low = sum((n-1)*step for n,step in zip(shape,strides) if step < 0)
                    high = sum((n-1)*step for n,step in zip(shape,strides) if step > 0)
                    require(type(offset) is int and 0 <= offset <= count and
                            (not math.prod(shape) or (offset+low >= 0 and offset+high+int(node["array"]["dtype"][-1]) <= count)),
                            "Array view escapes captured original backing storage")
        if "device" in node or kind == "device":
            value = node.get("device", node.get("value"))
            require((type(value) is int and value == 0) or value in ("CUDA:0", "CPU:0"), "Unexpected checkpoint device")

    reference(graph["root"])
    require(seen == set(range(len(nodes))) and used == set(table), "Orphan graph nodes or undeclared/unused payloads")
    return nodes


def validate_volume(node, array, folder, offsets):
    size, capacity = node["size"], node["capacity"]
    require(type(size) is int and type(capacity) is int and 0 <= size <= capacity <= 50000
            and capacity > 0 and node["resolution"] == 16, "Invalid logical VBG capacity/layout")
    voxel = float.fromhex(node["voxel_size_hex"])
    require(math.isfinite(voxel) and voxel > 0, "Invalid VBG voxel size")
    array(node["keys"])
    require(node["keys"]["dtype"] == "<i4" and node["keys"]["shape"] == [size,3], "VBG original key descriptor differs")
    offset, fortran = offsets[node["keys"]["path"]]
    require(not fortran, "Canonical VBG keys must be C-order")
    with (folder/node["keys"]["path"]).open("rb") as stream:
        stream.seek(offset)
        keys = list(struct.iter_unpack("<iii", stream.read()))
    require(keys == sorted(set(keys)), "Logical VBG keys are duplicated or not canonical")
    require(set(node["attributes"]) == {"tsdf", "weight", "color"}, "Logical VBG attributes incomplete")
    for name, channels in (("tsdf",1),("weight",1),("color",3)):
        attr = node["attributes"][name]
        require(set(attr) == {"channels", "payload_sha256", "chunks"} and attr["channels"] == channels
                and digest(attr["payload_sha256"]) and isinstance(attr["chunks"],list), "Malformed VBG attribute chunks")
        count, bits = 0, hashlib.sha256()
        for chunk in attr["chunks"]:
            array(chunk)
            require(chunk["dtype"] == "<f4" and len(chunk["shape"]) == 5
                    and chunk["shape"][1:] == [16,16,16,channels]
                    and chunk["shape"][0] == min(128,size-count), "VBG chunk coverage/order/layout differs")
            offset, fortran = offsets[chunk["path"]]
            require(not fortran, "Active logical VBG chunks must be C-order")
            with (folder/chunk["path"]).open("rb") as stream:
                stream.seek(offset)
                for block in iter(lambda: stream.read(1024**2), b""):
                    bits.update(block)
            count += chunk["shape"][0]
        require(count == size and bits.hexdigest() == attr["payload_sha256"], "Active VBG bits/chunk coverage changed")
    descriptor = {k: node["keys"][k] for k in ("dtype","shape","payload_sha256","nbytes")}
    require(node["logical_sha256"] == canonical_hash({"keys":descriptor,
        "attributes":{k:v["payload_sha256"] for k,v in node["attributes"].items()},"capacity":capacity}), "VBG logical fingerprint differs")


def dictionary(graph, ref):
    require(set(ref) == {"ref"}, "Require an original dictionary node")
    node = graph["nodes"][ref["ref"]]
    require(node["type"] in ("dict", "ordered_dict"), "Require an original dictionary node")
    result = {}
    for key,value in node["items"]:
        require(set(key) == {"literal"} and type(key["literal"]) is str and key["literal"] not in result,
                "Duplicate/non-string engine state key")
        result[key["literal"]] = value
    return result


def validate_rng(graph, reference):
    states = dictionary(graph,reference)
    require(set(states) == {"python","numpy"}, "Original RNG state absent")
    def sequence(ref,count):
        require(set(ref) == {"ref"}, "RNG tuple absent")
        node = graph["nodes"][ref["ref"]]
        require(node["type"] == "tuple" and len(node["items"]) == count, "Original RNG tuple shape differs")
        return node["items"]
    def integer(ref,low,high):
        require(set(ref) == {"literal"} and type(ref["literal"]) is int and low <= ref["literal"] <= high,
                "Invalid original RNG integer state")
    python = sequence(states["python"],3)
    integer(python[0],3,3)
    internal = sequence(python[1],625)
    for ref in internal[:-1]:
        integer(ref,0,2**32-1)
    integer(internal[-1],0,624)
    require(python[2] == {"literal":None} or (set(python[2]) == {"float_hex"}
            and math.isfinite(float.fromhex(python[2]["float_hex"]))), "Invalid Python Gaussian cache")
    numpy = sequence(states["numpy"],5)
    require(numpy[0] == {"literal":"MT19937"} and set(numpy[1]) == {"ref"}, "Original NumPy RNG policy changed")
    array = graph["nodes"][numpy[1]["ref"]]
    require(array["type"] == "ndarray" and array["array"]["dtype"] == "<u4" and array["array"]["shape"] == [624],
            "Incomplete NumPy RNG state bits")
    integer(numpy[2],0,624); integer(numpy[3],0,1)
    require(set(numpy[4]) == {"float_hex"} and math.isfinite(float.fromhex(numpy[4]["float_hex"])), "Invalid NumPy Gaussian cache")


def validate_checkpoint_manifest(path, expected_binding, expected_producer_artifacts, expected_scope_base=None):
    """Validate contained numeric graph/file/state before native materialization."""
    try:
        path = Path(path).resolve(strict=True)
        before = file_hash(path)
        tracked = {path:before}
        manifest = original.read_json(path)
        artifacts = normalized_artifacts(expected_producer_artifacts)
        require(set(FINISH_ARTIFACTS).issubset(artifacts), "Complete checkpoint and original Finish producer pins absent")
        require(manifest.get("kind") == CHECKPOINT_KIND and manifest.get("status") == "complete"
                and manifest.get("runtime_binding") == expected_binding
                and manifest.get("source_sha256") == expected_binding["source_sha256"] == current_core_hash()
                and manifest.get("producer_artifacts_sha256") == artifacts,
                "Checkpoint source/runtime/producer differs or capture is incomplete")
        if expected_scope_base is not None:
            require(manifest["scope_base"] == expected_scope_base, "Checkpoint raw/settings/selection scope changed")
        validate_proposal_policy(manifest["scope_base"])
        for name,value in artifacts.items():
            require(file_hash(ROOT/name) == value, "Current checkpoint producer source changed")
        live = manifest["fresh_live"]
        require(set(live) == {"accepted_indices","poses_sha256","semantic_decisions_sha256","raw_diagnostics_sha256",
                "raw_frames_sha256","prepared_inputs_sha256","unprocessed_count","pose_source"}
                and live["pose_source"] == "fresh raw Live replay" and zero(live["unprocessed_count"])
                and all(digest(live[k]) for k in live if k.endswith("sha256")), "Incomplete or archived measured Live state")
        selected = manifest["scope_base"]["selected_indices"]
        accepted = live["accepted_indices"]
        require(isinstance(selected,list) and selected == list(range(len(selected))) and 0 < len(selected) <= 3000
                and all(type(n) is int for n in selected), "Checkpoint omitted/reordered original raw inputs")
        require(isinstance(accepted,list) and accepted and accepted == sorted(set(accepted))
                and all(type(n) is int and n in selected for n in accepted), "Checkpoint Live acceptance incomplete")
        scope = dict(manifest["scope_base"],live={"accepted_indices":accepted,
            "unprocessed_count":0,"pose_source":"fresh raw Live replay",
            "poses_sha256":live["poses_sha256"],"decisions_sha256":live["semantic_decisions_sha256"]})
        original.validate_scope(scope,expected_binding,tracked)
        require(manifest["logical_state_sha256"] == canonical_hash(logical_graph(manifest["graph"])), "Logical state/aliases changed")
        validate_graph(manifest,path.parent,tracked)
        root = dictionary(manifest["graph"],manifest["graph"]["root"])
        require(set(root) == {"engine","prepared_inputs","rng_state"}, "Unknown/incomplete checkpoint root state")
        validate_rng(manifest["graph"],root["rng_state"])
        engine = dictionary(manifest["graph"],root["engine"])
        require({"raw_frames","poses","diagnostics","_processed_count","frame_count","vbg"} <= set(engine),
                "Core stored/accepted/volume state absent from checkpoint")
        def sequence(ref):
            require(set(ref) == {"ref"}, "Original sequence node absent")
            node = manifest["graph"]["nodes"][ref["ref"]]
            require(node["type"] == "list", "Original stored state must be a list")
            return node["items"]
        require(len(sequence(engine["raw_frames"])) == len(selected)
                and len(sequence(engine["poses"])) == len(accepted)
                and len(sequence(engine["diagnostics"])) == len(selected)
                and engine["_processed_count"] == {"literal":len(selected)}
                and engine["frame_count"] == {"literal":len(accepted)}, "Stored/processed/accepted state coverage differs")
        capture = manifest["capture_report"]
        require(capture.get("observation_healthy") is True and capture.get("prepared_actual_output_count") == len(selected)
                and capture.get("stored_count") == len(selected) and capture.get("accepted_indices") == accepted
                and capture.get("source_sha256") == expected_binding["source_sha256"]
                and capture.get("archive_sha256") == manifest["scope_base"]["archive"]["sha256"],
                "Raw Live output observer failed or omitted actual prepared inputs")
        names = manifest["engine_attribute_inventory"]
        cls = next(n for n in ast.parse((ROOT/"scanner_server/engine.py").read_text(encoding="utf-8")).body
                   if isinstance(n,ast.ClassDef) and n.name == "ScanEngine")
        allowed = {n.attr for n in ast.walk(cls) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)
                   and n.value.id == "self" and isinstance(n.ctx,ast.Store)}
        require(names == sorted(engine) and set(engine) <= allowed, "Unknown/missing source-bound engine inventory")
        for name in ("fusion_failure","input_failure","_final_vbg","mesh","point_cloud","_cuda_rgbd_odometry"):
            require(engine.get(name) == {"literal":None}, "Failed or transient post-Finish state cannot be checkpointed")
        for name in ("_pose_seeds_only","_icp_source_pyramid"):
            if name in engine:
                require(engine[name] in ({"literal":False},{"literal":None}), "Archived pose/transient ICP state present")
        for name in ("_stage_children",):
            node = manifest["graph"]["nodes"][engine[name]["ref"]]
            require(node["type"] == "list" and node["items"] == [], "Active reconstruction stage cannot be serialized")
        prepared = manifest["graph"]["nodes"][root["prepared_inputs"]["ref"]]
        require(prepared["type"] == "dict" and len(prepared["items"]) == len(selected) and
                {pair[0].get("literal") for pair in prepared["items"]} == set(selected) and
                all(set(pair[0]) == {"literal"} and type(pair[0]["literal"]) is int for pair in prepared["items"]),
                "Prepared-input capture omitted stored observations")
        require(all(file_hash(p) == value for p,value in tracked.items()), "Checkpoint/payload changed during validation")
        return CheckpointAuthority(str(path),before,manifest["logical_state_sha256"],
            json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False),tuple(sorted((str(p),v) for p,v in tracked.items())))
    except GridProofError:
        raise
    except (KeyError,TypeError,ValueError,AttributeError,IndexError,OSError,SyntaxError,RecursionError) as error:
        raise GridProofError(f"Incomplete/malformed checkpoint: {error}") from error


def validate_checkpoint_scope(scope, authority, tracked):
    require(type(authority) is CheckpointAuthority, "Require the new closed checkpoint authority")
    manifest = authority.manifest
    live = manifest["fresh_live"]
    expected_live = {"accepted_indices":live["accepted_indices"],"decisions_sha256":live["semantic_decisions_sha256"],
        **{key:live[key] for key in ("poses_sha256","raw_diagnostics_sha256","raw_frames_sha256",
            "prepared_inputs_sha256","unprocessed_count","pose_source")}}
    require(scope == dict(manifest["scope_base"],live=expected_live,checkpoint=authority.record),
            "Finish does not use the exact captured raw/settings/Live/RNG checkpoint scope")
    original.validate_scope(scope,manifest["runtime_binding"],tracked)
    for path,value in authority.closed_files:
        path = Path(path)
        require(file_hash(path) == value, "Closed checkpoint payload changed")
        tracked[path] = value


def validate_materialization(report, authority):
    manifest = authority.manifest
    live, check = manifest["fresh_live"], report["checkpoint_verification"]
    require(report.get("checkpoint") == report.get("checkpoint_after") == authority.record
            and report.get("checkpoint_scope_base") == manifest["scope_base"], "Checkpoint before/after scope changed")
    require(check.get("passed") is True and check.get("logical_state_equal") is True
            and check.get("checkpoint_sha256") == authority.sha256
            and check.get("logical_state_sha256") == authority.logical_state_sha256
            and check.get("prepared_input_parity_passed") is True
            and check.get("prepared_input_checks") == len(manifest["scope_base"]["selected_indices"]),
            "Native materialization did not prove complete state and actual prepared input parity")
    require(all(check.get(key) == live[key] for key in
        ("poses_sha256","raw_diagnostics_sha256","raw_frames_sha256","prepared_inputs_sha256")),
        "Materialized raw, prepared inputs, diagnostics or unrounded poses changed")
    expected = [node for node in manifest["graph"]["nodes"] if node["type"] == "vbg"]
    checks = check.get("volume_checks")
    require(isinstance(checks,list) and len(checks) == len(expected), "Logical VBG verification coverage absent")
    for node, actual in zip(expected,checks):
        require(actual.get("passed") is True and actual.get("capacity") == node["capacity"]
                and actual.get("size") == node["size"] and actual.get("keys_sha256") == node["keys"]["payload_sha256"]
                and actual.get("attributes_sha256") == {k:v["payload_sha256"] for k,v in node["attributes"].items()},
                "Actual restored VBG capacity/keys/attribute bits differ")


def validate_proposal_policy(scope):
    """Explicit proposal-only thread experiment; original ICP policy stays fixed."""
    pipeline = scope.get("pipeline_options", {})
    threads = pipeline.get("research_ransac_threads")
    require(type(threads) is int and threads in (1,20)
            and type(pipeline.get("research_final_preplan")) is bool
            and scope.get("thread_policy") == original.THREAD_POLICY,
            "Explicit RANSAC 1|20 policy, matched planner and original ICP20/CV20/OMP8 are required")
    return threads


def validate_proposal_report(report, gates):
    threads = validate_proposal_policy(report["scope_binding"])
    require(type(report.get("research_ransac_threads")) is int
            and report["research_ransac_threads"] == threads,
            "Reported proposal threads differ from the exact checkpoint scope")
    proposals = report.get("ransac_proposals")
    require(isinstance(proposals,dict) and proposals.get("policy") == RANSAC_POLICY
            and type(proposals.get("requested_threads")) is int and proposals["requested_threads"] == threads
            and proposals.get("outside_threads") == 20 and proposals.get("hooks_restored") is True
            and not proposals.get("failure"), "Original proposal-only thread scope did not restore cleanly")
    rows = proposals.get("calls")
    original_gates = sorted((row for row in gates if row["function"] == "fragments._global_seed"),
                            key=lambda row: row["gate_index"])
    require(isinstance(rows,list) and type(proposals.get("call_count")) is int
            and proposals["call_count"] == len(rows) == len(original_gates),
            "Original global proposal thread/gate coverage is incomplete")
    selected = report["scope_binding"]["selected_indices"]
    for index,(row,gate) in enumerate(zip(rows,original_gates)):
        require(isinstance(row,dict) and type(row.get("invocation")) is int and row["invocation"] == index
                and row.get("complete") is True and row.get("restored_threads") == 20
                and type(row.get("requested_threads")) is int and row["requested_threads"] == threads
                and type(row.get("original_seed")) is int
                and all(type(row.get(key)) is int and row[key] in selected for key in ("source","target"))
                and type(row.get("proposal_present")) is bool and row["proposal_present"] is (gate["result"] is not None)
                and original.finite_nonnegative(row.get("wall_s")) and not row.get("original_exception"),
                "Original proposal inputs/outcome, thread restoration or timing evidence changed")


def closed_sidecar(path, mode, authority, artifacts, component_proof, tracked):
    """Explicit proposal-thread native/audit closure for the comparator."""
    path = Path(path).resolve(strict=True)
    tracked[path] = file_hash(path)
    report = original.read_json(path)
    runtime = authority.manifest["runtime_binding"]
    scope = report["scope_binding"]
    require(report.get("kind") == KIND and report.get("mode") == mode and report.get("status") == "complete"
            and report.get("performance_attribution_valid") is (mode != "audit")
            and report.get("runner_hooks_restored") is True and not report.get("failure") and not report.get("cleanup_failures"),
            "New checkpoint Finish did not complete and restore cleanly")
    require(report.get("source_sha256") == report.get("source_sha256_after") == runtime["source_sha256"]
            and report.get("archive_sha256") == report.get("archive_sha256_after") == scope["archive"]["sha256"]
            and report.get("runtime_binding") == report.get("runtime_binding_after") == runtime
            and report.get("artifacts_sha256") == report.get("artifacts_sha256_after") == artifacts
            and report.get("component_proof") == report.get("component_proof_after") == component_proof
            and report.get("scope_binding_sha256") == canonical_hash(scope), "Finish provenance changed before closure")
    validate_checkpoint_scope(scope,authority,tracked)
    validate_materialization(report,authority)
    for record in component_proof.values():
        original.closed_file(record,tracked)
    trace_path = original.closed_file(report["trace"],tracked)
    rows = [original.strict_json(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(rows and report["trace"].get("rows") == len(rows) and all(isinstance(row,dict)
            and row.get("event") in ("match","gate") and row.get("complete") is True for row in rows), "Trace incomplete or unknown events")
    matches, gates = ([row for row in rows if row["event"] == event] for event in ("match","gate"))
    require(0 < len(matches) <= 100000 and [r.get("call_index") for r in matches] == list(range(len(matches)))
            and all(type(r["call_index"]) is int for r in matches), "Original match order/count incomplete")
    require(gates and sorted(r.get("gate_index") for r in gates) == list(range(len(gates)))
            and all(type(r["gate_index"]) is int and isinstance(r.get("function"),str) for r in gates), "Original gate order/count incomplete")
    validate_proposal_report(report,gates)
    registration = report["registration"]
    require(registration.get("complete") is True and registration.get("restored") is True
            and registration.get("input_immutability_passed") is True and registration.get("calls") == len(matches)
            and registration.get("gate_calls") == len(gates) and zero(registration.get("full_cpu_fallback_calls"))
            and zero(registration.get("cpu_shadow_failures")) and not registration.get("failure") and not registration.get("cleanup_failures"),
            "Registration/gate scope did not close without fallback or errors")
    signatures = tuple(original.call_signature(r["call_inputs"]) for r in matches)
    require([r.get("call_signature") for r in matches] == list(signatures)
            and registration.get("call_signatures") == list(signatures), "Call signatures differ from exact original inputs")
    targets = frozenset(r["call_inputs"]["target_points"]["sha256"] for r in matches)
    require(isinstance(registration.get("target_digests"),list)
            and len(registration["target_digests"]) == len(targets) and set(registration["target_digests"]) == targets,
            "New field target membership incomplete")
    if mode == "native":
        require(report.get("cpu_query_auditor") == {"mode":"scalar"}, "Native reference must keep original scalar policy")
        require(zero(registration.get("cpu_shadow_calls")) and registration.get("resident") is None
                and registration.get("retrieval_statistics") == {}, "Native control was replaced by the resident solver")
        for row,signature in zip(matches,signatures):
            require(row.get("inputs_after_resident") == signature, "Original CPU mutated its exact call inputs")
    elif mode == "audit":
        require(registration.get("cpu_shadow_calls") == len(matches), "Not every resident call has a complete original CPU shadow")
        totals = {key:0 for key in original.QUERY_COUNTERS}
        for row in matches:
            original.validate_shadow(row)
            stats = row["query_statistics_delta"]
            require(all(type(stats.get(key)) is int and stats[key] >= 0 for key in totals), "Incomplete field query accounting")
            require(stats["direct_gpu_hits"] == stats["audited_hits"] and stats["declared_gpu_misses"] == stats["audited_misses"]
                    and stats["query_rows"] == stats["device_flagged_rows"] == stats["device_query_download_rows"]
                    and stats["device_query_download_bytes"] == 64*stats["query_rows"]
                    and stats["query_rows"] == stats["direct_gpu_hits"]+stats["declared_gpu_misses"]+stats["exact_cpu_queries"]
                    and all(zero(stats[key]) for key in ("audit_index_mismatches","audit_false_misses","device_malformed_results")),
                    "Every field query must retain complete original CPU hit/miss/ambiguity authority")
            for key in totals:
                totals[key] += stats[key]
        require(all(registration["retrieval_statistics"].get(k) == value for k,value in totals.items()), "Full query totals incomplete")
        device_audit_coverage(registration["retrieval_statistics"],complete_trajectory=True)
        component_artifacts = validate_configuration(runtime)
        resident_report_checks(registration["resident"],runtime,component_artifacts)
        require(registration["resident"]["statistics"].get("calls") == len(matches), "Resident call totals incomplete")
    original.validate_profile(report["profile"],scope,tracked)
    return report, matches, sorted(gates,key=lambda r:r["gate_index"])


def validate_checkpoint_quality(path, audit_hash, audit, authority, artifacts, component_proof, tracked):
    from scripts.research.compare_resident_finishes import semantic_agreement
    path = Path(path).resolve(strict=True)
    quality_hash = file_hash(path)
    tracked[path] = quality_hash
    quality = original.read_json(path)
    require(quality.get("kind") == QUALITY_KIND and quality.get("status") == "passed"
            and quality.get("audit_report_sha256") == audit_hash and quality.get("checkpoint") == authority.record
            and quality.get("scope_binding_sha256") == audit["scope_binding_sha256"], "Closed v2 quality is stale or belongs to another checkpoint")
    require(quality.get("candidate") == audit["profile"] and quality.get("candidate_trace") == audit["trace"], "Quality candidate outputs differ")
    original.closed_file(quality["audit"],tracked)
    require(quality["audit"]["sha256"] == audit_hash, "Quality audit closure changed")
    native_path = original.closed_file(quality["native_sidecar"],tracked)
    native,_,native_gates = closed_sidecar(native_path,"native",authority,artifacts,component_proof,tracked)
    require(native["scope_binding"] == audit["scope_binding"] and native["profile"] == quality.get("native")
            and native["trace"] == quality.get("native_trace") and quality.get("live_pose_signatures_equal") is True
            and quality.get("live_decision_signatures_equal") is True,
            "Native and audited Finish must share the exact measured Live checkpoint")
    require(Path(quality["native"]["path"]).resolve() != Path(quality["candidate"]["path"]).resolve(), "Candidate cannot be its own native reference")
    audit_rows = [original.strict_json(line) for line in Path(audit["trace"]["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_gates = sorted((r for r in audit_rows if r["event"] == "gate"),key=lambda r:r["gate_index"])
    require(len(native_gates) == len(audit_gates) and all(
        a["gate_index"] == b["gate_index"] and a["function"] == b["function"]
        and semantic_agreement(a["result"],b["result"],a["function"]+"/result") for a,b in zip(native_gates,audit_gates)),
        "Original ordered witnesses/poses/information/gate outcomes changed")
    require(quality.get("same_ordered_gate_decisions") is True and quality.get("ordered_gate_evidence_passed") is True,
            "Original gate quality not proved")
    comparison = quality["comparison"]
    require(comparison.get("candidate_report_sha256") == audit["profile"]["sha256"]
            and comparison.get("baseline_report_sha256") == native["profile"]["sha256"]
            and comparison.get("same_mesh_success") is True and comparison.get("same_accepted_indices") is True,
            "Stale or changed final mesh acceptance")
    candidate = original.read_json(audit["profile"]["path"])
    baseline = original.read_json(native["profile"]["path"])
    require(candidate.get("accepted_indices") == baseline.get("accepted_indices"), "Final accepted views changed")
    surface = comparison["triangle_surface_metrics_vs_cpu"]
    require(surface.get("threshold_m") == .005 and surface.get("samples_per_surface") == 30000
            and surface.get("alignment") == "fixed input coordinate frame; no scale or trajectory fitting"
            and original.finite_nonnegative(surface.get("surface_p95_m")) and surface["surface_p95_m"] <= .0005
            and all(original.finite_nonnegative(surface.get(k)) and .999 <= surface[k] <= 1 for k in ("precision","completeness")),
            "Fixed 0.5mm p95 /99.9% within5mm whole-surface gate failed")
    require(quality.get("comparison_artifact_sha256") == artifacts["scripts/research/compare_checkpoint_resident_finishes.py"],
            "Quality producer differs from pinned comparator")
    return quality_hash


def validate_checkpoint_finish_proof(audit_path, component_authority, expected_scope_binding,
        expected_artifacts, *, quality_path, checkpoint_authority, bulk_authority=None):
    """Only a new exact checkpoint audit plus separate quality mints timing scope."""
    try:
        require(type(component_authority) is DeviceFlatGridProofAuthority and type(checkpoint_authority) is CheckpointAuthority,
                "Require new component and measured-Live checkpoint authorities")
        runtime = component_authority.runtime_binding
        artifacts = normalized_artifacts(expected_artifacts)
        require(set(FINISH_ARTIFACTS).issubset(artifacts), "New and unchanged Finish dependency pins absent")
        require(checkpoint_authority.manifest["runtime_binding"] == runtime
                and checkpoint_authority.manifest["producer_artifacts_sha256"] == artifacts,
                "Checkpoint component/runtime/dependency pins differ")
        fresh = validate_checkpoint_manifest(checkpoint_authority.path,runtime,artifacts,
            checkpoint_authority.manifest["scope_base"])
        require(fresh == checkpoint_authority, "Checkpoint authority is stale")
        component_artifacts = validate_configuration(runtime)
        require(component_authority.bindings_sha256 == canonical_hash(runtime)
                and dict(component_authority.artifact_sha256) == runtime["artifacts_sha256"], "Component authority binding differs")
        validate_current_artifacts(runtime,component_artifacts)
        audit_path = Path(audit_path).resolve(strict=True)
        tracked = {}
        audit_hash = file_hash(audit_path)
        audit = original.read_json(audit_path)
        proof = audit["component_proof"]
        require(proof["synthetic"]["sha256"] == component_authority.synthetic_report_sha256
                and proof["bridge"]["sha256"] == component_authority.bridge_report_sha256
                and Path(proof["synthetic"]["path"]).resolve() != Path(proof["bridge"]["path"]).resolve(),
                "Separate current component proofs differ")
        audit,matches,_ = closed_sidecar(audit_path,"audit",checkpoint_authority,artifacts,proof,tracked)
        require(audit["scope_binding"] == expected_scope_binding, "Requested timing differs from audited exact checkpoint scope")
        # Bulk compatibility is a separate CPU-shadow proof, never a substitute
        # for any new field query, complete CPU result or original Finish gate.
        validate_field_cpu_auditor(audit,matches,bulk_authority,runtime,artifacts,tracked)
        quality_hash = validate_checkpoint_quality(quality_path,audit_hash,audit,checkpoint_authority,artifacts,proof,tracked)
        validate_current_artifacts(runtime,component_artifacts)
        require(all(file_hash(ROOT/name) == value for name,value in artifacts.items())
                and all(file_hash(path) == value for path,value in tracked.items()), "Proof/raw/payload/helpers changed during validation")
        token = dict(component_authority.__dict__,target_digests=frozenset(audit["registration"]["target_digests"]))
        registration = original.FinishResidentProofAuthority(**token,
            finish_scope_json=json.dumps(expected_scope_binding,sort_keys=True,separators=(",",":"),allow_nan=False),
            expected_call_signatures=tuple(r["call_signature"] for r in matches),finish_audit_report_sha256=audit_hash,
            quality_proof_sha256=quality_hash,full_finish_artifact_sha256=tuple(sorted(artifacts.items())))
        return CheckpointFinishProofAuthority(registration,checkpoint_authority,audit_hash,quality_hash)
    except GridProofError:
        raise
    except (KeyError,TypeError,ValueError,AttributeError,IndexError,OSError) as error:
        raise GridProofError(f"Incomplete/malformed/stale checkpoint Finish proof: {error}") from error


def validate_field_cpu_auditor(audit,matches,bulk_authority,runtime,artifacts,tracked):
    record = audit.get("cpu_query_auditor")
    if record == {"mode":"scalar"}:
        require(bulk_authority is None, "Scalar field audit cannot silently consume bulk compatibility authority")
        return
    require(type(bulk_authority) is BulkResidentAuditAuthority and isinstance(record,dict)
            and set(record) == {"mode","policy","proof","api_proof","native_runtime","chunk_rows"}
            and record["mode"] == "bulk-shadow" and record["policy"] == BULK_POLICY and record["chunk_rows"] == 65536,
            "New bulk field audit lacks its distinct complete dual CPU-auditor authority")
    require(record["proof"] == {"path":bulk_authority.dual_report_path,"sha256":bulk_authority.dual_report_sha256}
            and record["api_proof"] == {"path":bulk_authority.api_report_path,"sha256":bulk_authority.api_report_sha256},
            "Field bulk auditor uses different closed compatibility evidence")
    for name in ("proof","api_proof"):
        original.closed_file(record[name],tracked)
    validate_authority_for_bulk_shadow(bulk_authority,device=runtime["resident_configuration"]["device"],
        chunk_rows=record["chunk_rows"],native_runtime=record["native_runtime"])
    bulk_runtime = bulk_authority.runtime_binding
    require(audit.get("referenced_bulk_audit_proof") == audit.get("referenced_bulk_audit_proof_after") == record["proof"]
            and audit.get("bulk_runtime_binding") == audit.get("bulk_runtime_binding_after") == bulk_runtime,
            "Field CPU auditor runtime or closed compatibility proof changed")
    require(all(bulk_runtime[key] == runtime[key] for key in ("source_sha256","component_source_sha256",
        "domain","thread_policy","resident_configuration","resident_math","cache_policy","gpu","versions")),
        "Bulk compatibility uses a different actual device/math/runtime policy")
    require(all(dict(bulk_authority.artifact_sha256).get(name) == artifacts[name] for name in BULK_ARTIFACTS),
        "New field CPU-shadow adapter differs from the validated compatibility evidence")
    totals = {key:0 for key in ("bulk_shadow_queries","bulk_shadow_hits","bulk_shadow_misses",
        "dual_scalar_bulk_queries","dual_id_mismatches","dual_metric_bit_mismatches","bulk_domain_scalar_queries")}
    for row in matches:
        stats = row["query_statistics_delta"]
        require(all(type(stats.get(k)) is int and stats[k] >= 0 for k in totals), "Field bulk accounting absent")
        require(stats["bulk_shadow_queries"] == stats["bulk_shadow_hits"]+stats["bulk_shadow_misses"]
                and stats["bulk_shadow_queries"]+stats["bulk_domain_scalar_queries"] == stats["audited_hits"]+stats["audited_misses"]
                and all(zero(stats[k]) for k in ("dual_scalar_bulk_queries","dual_id_mismatches","dual_metric_bit_mismatches")),
                "Field hit/miss CPU shadows were omitted or the compatibility audit mode silently changed")
        for key in totals:
            totals[key] += stats[key]
    require(positive(totals["bulk_shadow_queries"]) and positive(totals["bulk_shadow_hits"])
            and positive(totals["bulk_shadow_misses"]) and
            all(audit["registration"]["retrieval_statistics"].get(k) == value for k,value in totals.items()),
            "Actual field bulk positive hit/miss coverage and complete accounting absent")
