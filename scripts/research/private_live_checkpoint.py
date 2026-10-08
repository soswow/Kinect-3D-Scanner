"""Private checkpoints of freshly measured raw Live; never exported-pose seeds.

The format is a closed JSON graph and bounded NPY payloads, not executable
pickle. It retains engine object aliases, original CPU point/feature order,
array strides/flags, preparation and descriptor caches, all transactional
scalars, and logical VBG key/attribute bits plus measured capacity. VBG buffer
indices are implementation details and may change on materialization; original
cached CPU geometry is restored directly rather than extracted or re-fused.
Unknown engine state fails closed. Importing this module loads no native code.

Open3D v0.20.0 Save/Load does not retain hashmap capacity (Load allocates the
number of active keys), so we explicitly activate into the recorded capacity:
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/t/geometry/VoxelBlockGrid.cpp
https://www.open3d.org/docs/release/tutorial/core/hashmap.html
Native transient frustum scratch and global allocator/kernel pools are recreated,
not serialized. Their measured reservations must be reported separately from
the logical engine-owned state and capacity checks below.
"""

from __future__ import annotations
from collections import OrderedDict
import ast
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash
from scripts.research.archive.validate_uniform_grid_proof import canonical_hash

KIND = "private-fresh-raw-live-checkpoint-v2"
MAX_NODES = 100000
MAX_ARRAY_BYTES = 2 * 1024**3
VBG_CHUNK_BLOCKS = 128


class CheckpointError(RuntimeError):
    pass


class CheckpointCaptureFailure(BaseException):
    """Observation/serialization faults cannot become rejected Live frames."""
    pass


def semantic_decisions(diagnostics):
    """Remove only known success-message durations; preserve raw data elsewhere."""
    result = []
    for row in diagnostics:
        item = {key: row[key] for key in ("index", "success", "method", "message") if key in row}
        if item.get("success") and isinstance(item.get("message"), str):
            message = item["message"]
            if re.fullmatch(r"Reference frame \([0-9]+ms\)", message):
                message = "Reference frame"
            elif re.fullmatch(r"Frame [0-9]+(?: via global)? \(fitness=[0-9.]+(?:, RMSE=[0-9.]+)?, [0-9]+ms\)", message):
                message = re.sub(r", [0-9]+ms(?=\)$)", "", message)
            item["message"] = message
        result.append(item)
    return result


def array_metadata(np, value):
    array = np.asarray(value)
    if array.dtype.hasobject or array.nbytes > MAX_ARRAY_BYTES or array.dtype.kind not in "biuf":
        raise CheckpointError("Unsupported or oversized numeric checkpoint array")
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "payload_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            "nbytes": array.nbytes}


def pose_sha256(np, poses):
    digest = hashlib.sha256()
    for index, pose in poses:
        array = np.asarray(pose)
        if array.dtype != np.dtype("<f8") or array.shape != (4, 4) or not np.isfinite(array).all():
            raise CheckpointError("Require original finite unrounded FP64 Live poses")
        digest.update(struct.pack("<Q", index))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def inputs_sha256(np, inputs):
    """Bind original indexed arrays, rather than file headers or addresses."""
    pairs = inputs.items() if isinstance(inputs, dict) else enumerate(inputs)
    return canonical_hash([{"index": index, "arrays": [array_metadata(np, value) for value in pair]}
                           for index, pair in sorted(pairs)])


def inventory(engine):
    module = ast.parse((ROOT/"scanner_server/engine.py").read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ScanEngine")
    allowed = {node.attr for node in ast.walk(cls) if isinstance(node, ast.Attribute)
               and isinstance(node.value, ast.Name) and node.value.id == "self" and isinstance(node.ctx, ast.Store)}
    if set(engine.__dict__) - allowed:
        raise CheckpointError("Unknown source-bound engine state attributes: "+str(sorted(set(engine.__dict__)-allowed)))
    if (engine.unprocessed_count or getattr(engine, "_pose_seeds_only", False)
            or engine.fusion_failure is not None or engine.input_failure is not None
            or engine._stage_children or engine._final_vbg is not None
            or getattr(engine, "_icp_source_pyramid", None) is not None):
        raise CheckpointError("Require healthy completed fresh Live before any Finish/transient registration")
    if engine.mesh is not None or engine.point_cloud is not None:
        raise CheckpointError("This scope captures fresh raw Live without an intervening mesh extraction")
    if engine._cuda_rgbd_odometry is not None:
        raise CheckpointError("Checkpoint v2 requires the original projective-odometry-off scope")
    return sorted(engine.__dict__)


class PreparedInputRecorder:
    """Observe original host preparation outputs; never change calls/counters."""
    def __init__(self, engine):
        self.engine, self.original, self.inputs = engine, engine._prepare_input, {}
        self.failure = None
        self.had_instance_method = "_prepare_input" in engine.__dict__
        self.original_instance_method = engine.__dict__.get("_prepare_input")

    def __enter__(self):
        engine = self.engine

        def prepare(rgb, raw, settings=None, *, cpu_prepare=None):
            if self.failure is not None:
                raise self.failure
            color, depth = self.original(rgb, raw, settings, cpu_prepare=cpu_prepare)
            try:
                actual_settings = engine.settings if settings is None else settings
                if actual_settings != engine.settings:
                    raise CheckpointError("Unexpected preparation settings during raw Live capture")
                matches = [index for index, pair in enumerate(engine.raw_frames) if pair[0] is rgb and pair[1] is raw]
                if len(matches) != 1:
                    raise CheckpointError("Prepared capture is not an original stored raw observation")
                import numpy as np
                index = matches[0]
                if index not in self.inputs:
                    self.inputs[index] = (np.array(color, copy=True, order="C"), np.array(depth, copy=True, order="C"))
                elif not (np.array_equal(color, self.inputs[index][0]) and np.array_equal(depth, self.inputs[index][1])):
                    raise CheckpointError("Repeated original preparation changed measured RGB/depth")
            except BaseException as error:
                self.failure = CheckpointCaptureFailure(f"{type(error).__name__}: {error}")
                raise self.failure from error
            return color, depth

        engine._prepare_input = prepare
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if self.had_instance_method:
                self.engine._prepare_input = self.original_instance_method
            else:
                del self.engine.__dict__["_prepare_input"]
        except BaseException as secondary:
            primary = self.failure if self.failure is not None else error
            if primary is not None:
                primary.add_note(f"Capture observer restore also failed: {secondary}")
                raise primary
            raise
        if self.failure is not None:
            if error is not None and error is not self.failure:
                self.failure.add_note(f"Capture body/cleanup also failed: {error}")
            raise self.failure


class GraphCodec:
    def __init__(self, folder, *, writing, nodes=None, payloads=None):
        import numpy as np
        import open3d as o3d
        import cupy as cp
        from shared.config import ScanPreset
        from shared.settings import ScanSettings, CameraCalibration
        from shared.sensor_calibration import SensorCalibration
        from scanner_server.appearance import Features
        from scanner_server.tracking_cache import TargetPyramid, SourcePyramid
        self.np, self.o3d, self.cp = np, o3d, cp
        self.folder, self.writing = Path(folder), writing
        self.nodes, self.payloads = nodes if nodes is not None else [], payloads if payloads is not None else []
        self.seen, self.owners, self.memo = {}, [], {}
        self.classes = {cls.__module__+"."+cls.__name__: cls for cls in
                        (ScanPreset, ScanSettings, CameraCalibration, SensorCalibration, Features, TargetPyramid, SourcePyramid)}
        self.volume_checks = []

    def array(self, value):
        np = self.np
        metadata = array_metadata(np, value)
        if not self.writing:
            raise CheckpointError("Read codec cannot write payloads")
        name = f"array-{len(self.payloads):06d}.npy"
        path = self.folder/name
        np.save(path, np.asarray(value), allow_pickle=False)
        item = dict(metadata, path=name, sha256=file_hash(path))
        self.payloads.append(item)
        return item

    def read_array(self, item):
        path = (self.folder/item["path"]).resolve()
        if path.parent != self.folder.resolve() or file_hash(path) != item["sha256"]:
            raise CheckpointError("Payload path escaped or closed bytes changed")
        array = self.np.load(path, allow_pickle=False)
        if array_metadata(self.np, array) != {k: item[k] for k in ("dtype", "shape", "payload_sha256", "nbytes")}:
            raise CheckpointError("NPY payload descriptor/bytes changed")
        return array

    def encode(self, value):
        np, o3d, cp = self.np, self.o3d, self.cp
        if value is None or type(value) in (str, bool, int):
            return {"literal": value}
        if type(value) is float:
            if not math.isfinite(value):
                raise CheckpointError("Nonfinite scalar engine state")
            return {"float_hex": value.hex()}
        key = id(value)
        if key in self.seen:
            return {"ref": self.seen[key]}
        if len(self.nodes) >= MAX_NODES:
            raise CheckpointError("Checkpoint object graph exceeds bound")
        index = len(self.nodes)
        self.seen[key] = index
        self.owners.append(value)  # prevent temporary wrapper/view ID reuse
        node = {}
        self.nodes.append(node)
        if isinstance(value, np.ndarray):
            node.update(type="ndarray", array=self.array(value), strides=list(value.strides), writable=bool(value.flags.writeable))
            if isinstance(value.base, np.ndarray):
                base = value.base
                while isinstance(base.base, np.ndarray):
                    base = base.base
                if not (base.flags.c_contiguous or base.flags.f_contiguous):
                    raise CheckpointError("Unsupported noncontiguous ndarray storage owner")
                node.update(storage=self.encode(base), storage_offset=int(value.__array_interface__["data"][0]
                    - base.__array_interface__["data"][0]))
        elif isinstance(value, np.generic):
            node.update(type="numpy_scalar", array=self.array(np.asarray(value)))
        elif isinstance(value, cp.ndarray):
            if not value.flags.c_contiguous:
                raise CheckpointError("Noncontiguous resident array ownership is outside this checkpoint scope")
            with cp.cuda.Device(value.device.id), cp.cuda.Stream.null:
                cp.cuda.Stream.null.synchronize()
                node.update(type="cupy", device=value.device.id, array=self.array(cp.asnumpy(value)))
        elif isinstance(value, (dict, OrderedDict)):
            node.update(type="ordered_dict" if isinstance(value, OrderedDict) else "dict",
                        items=[[self.encode(k), self.encode(v)] for k, v in value.items()])
        elif isinstance(value, (list, tuple, set, frozenset)):
            values = value
            if isinstance(value, (set, frozenset)):
                if any(type(item) not in (int, str) for item in value):
                    raise CheckpointError("Only original scalar frame-ID sets have a stable private order")
                values = sorted(value, key=lambda item: (type(item).__name__, item))
            node.update(type=type(value).__name__, items=[self.encode(v) for v in values])
        elif isinstance(value, o3d.core.Device):
            node.update(type="device", value=str(value))
        elif isinstance(value, o3d.core.Tensor):
            if not value.is_contiguous():
                raise CheckpointError("Noncontiguous native tensor ownership is outside this checkpoint scope")
            node.update(type="tensor", device=str(value.device), array=self.array(value.cpu().numpy()))
        elif isinstance(value, o3d.camera.PinholeCameraIntrinsic):
            node.update(type="intrinsic", width=value.width, height=value.height, matrix=self.array(value.intrinsic_matrix))
        elif isinstance(value, o3d.geometry.PointCloud):
            node.update(type="cloud", attributes={name: self.array(np.asarray(getattr(value, name)))
                        for name in ("points", "normals", "colors", "covariances")})
        elif isinstance(value, o3d.geometry.Image):
            node.update(type="image", array=self.array(np.asarray(value)))
        elif isinstance(value, o3d.geometry.RGBDImage):
            node.update(type="rgbd", color=self.encode(value.color), depth=self.encode(value.depth))
        elif isinstance(value, o3d.pipelines.registration.Feature):
            node.update(type="fpfh", data=self.array(np.asarray(value.data)))
        elif isinstance(value, o3d.t.geometry.PointCloud):
            node.update(type="tensor_cloud", device=str(value.device),
                        attributes={name: self.encode(value.point[name]) for name in sorted(value.point)})
        elif isinstance(value, o3d.t.geometry.VoxelBlockGrid):
            self._encode_volume(node, value)
        else:
            name = type(value).__module__+"."+type(value).__name__
            if name in self.classes:
                state = {field.name: getattr(value, field.name) for field in fields(value)} if is_dataclass(value) else value.__dict__
                node.update(type="registered", name=name, state=self.encode(state))
            elif name in ("scanner_server.cuda_input.InputPreparation", "scanner_server.cuda_confidence.ConfidencePreparation"):
                state = {k: v for k, v in value.__dict__.items() if k != "_factory"}
                factory = value._factory
                if not hasattr(factory, "__code__") or factory.__qualname__ != type(value).__name__+".__init__.<locals>.<lambda>":
                    raise CheckpointError("Custom preparation factory cannot be checkpointed")
                node.update(type="preparation", name=name, state=self.encode(state))
            elif name == "scanner_server.cuda_input.NativeRgbd":
                keys = ("settings", "shape", "count", "interpolation", "launch", "source", "lut", "rays", "transform", "camera")
                node.update(type="native_rgbd", device=value.cuda_device.id,
                            state=self.encode({k: getattr(value, k) for k in keys}))
            elif name == "scanner_server.cuda_matching.Matcher":
                node.update(type="matcher", device=value.device_id,
                            bank=self.encode([(entry[0], entry[1]) for entry in value.cache.values()]))
            else:
                raise CheckpointError(f"Unsupported exact engine state type: {name}")
        return {"ref": index}

    def _encode_volume(self, node, volume):
        np, core = self.np, self.o3d.core
        hashmap = volume.hashmap()
        capacity, size = int(hashmap.capacity()), int(hashmap.size())
        if not 0 <= size <= capacity <= 50000:
            raise CheckpointError("VBG logical capacity exceeds declared private bound")
        indices = hashmap.active_buf_indices().to(core.int64)
        keys = hashmap.key_tensor()[indices].cpu().numpy()
        if keys.dtype != np.int32 or keys.shape != (size, 3) or len(np.unique(keys, axis=0)) != size:
            raise CheckpointError("Malformed measured logical VBG keys")
        order = np.lexsort(keys.T[::-1]) if size else np.empty(0, np.int64)
        original_indices = indices.cpu().numpy()[order]
        attrs = {}
        for name, channels in (("tsdf", 1), ("weight", 1), ("color", 3)):
            attribute = volume.attribute(name)
            if attribute.dtype != core.float32 or list(attribute.shape) != [capacity, 16, 16, 16, channels]:
                raise CheckpointError("Measured VBG layout differs from original engine contract")
            digest, chunks = hashlib.sha256(), []
            for start in range(0, size, VBG_CHUNK_BLOCKS):
                subset = core.Tensor(original_indices[start:start+VBG_CHUNK_BLOCKS], dtype=core.int64, device=attribute.device)
                host = attribute[subset].cpu().numpy()
                digest.update(np.ascontiguousarray(host).tobytes())
                chunks.append(self.array(host))
            attrs[name] = {"channels": channels, "payload_sha256": digest.hexdigest(), "chunks": chunks}
        # Engine's float setting is captured separately; VBG stores float32.
        node.update(type="vbg", capacity=capacity, size=size, resolution=16,
                    voxel_size_hex=float(np.float32(self.engine_voxel)).hex(), device=str(hashmap.device),
                    keys=self.array(keys[order]), attributes=attrs,
                    logical_sha256=canonical_hash({"keys": array_metadata(np, keys[order]),
                        "attributes": {k: v["payload_sha256"] for k, v in attrs.items()}, "capacity": capacity}))

    def decode(self, item):
        if "literal" in item:
            return item["literal"]
        if "float_hex" in item:
            return float.fromhex(item["float_hex"])
        index = item["ref"]
        if type(index) is not int or not 0 <= index < len(self.nodes):
            raise CheckpointError("Invalid graph reference")
        if index in self.memo:
            return self.memo[index]
        np, o3d, cp, node = self.np, self.o3d, self.cp, self.nodes[index]
        kind = node["type"]
        if kind in ("dict", "ordered_dict", "list", "set"):
            value = {"dict": dict, "ordered_dict": OrderedDict, "list": list, "set": set}[kind]()
            self.memo[index] = value
            if kind in ("dict", "ordered_dict"):
                for key, data in node["items"]:
                    value[self.decode(key)] = self.decode(data)
            else:
                for data in node["items"]:
                    (value.add if kind == "set" else value.append)(self.decode(data))
            return value
        if kind in ("tuple", "frozenset"):
            value = (tuple if kind == "tuple" else frozenset)(self.decode(v) for v in node["items"])
        elif kind == "ndarray":
            saved = self.read_array(node["array"])
            shape, strides = tuple(saved.shape), tuple(node["strides"])
            span = saved.dtype.itemsize + sum((n-1)*abs(step) for n, step in zip(shape, strides)) if saved.size else 0
            if span > MAX_ARRAY_BYTES or len(shape) != len(strides):
                raise CheckpointError("Array stride span exceeds private bound")
            if "storage" in node:
                base = self.decode(node["storage"])
                value = np.ndarray(shape, dtype=saved.dtype, buffer=base,
                    offset=node["storage_offset"], strides=strides)
                if array_metadata(np, value) != array_metadata(np, saved):
                    raise CheckpointError("Array shared storage view changed captured bits")
            elif saved.size:
                offset = sum((n-1)*(-step) for n, step in zip(shape, strides) if step < 0)
                value = np.ndarray(shape, dtype=saved.dtype, buffer=bytearray(span), offset=offset, strides=strides)
                value[...] = saved
            else:
                value = np.ndarray(shape, dtype=saved.dtype, buffer=bytearray(saved.dtype.itemsize), strides=strides)
            value.flags.writeable = node["writable"]
        elif kind == "numpy_scalar":
            value = self.read_array(node["array"])[()]
        elif kind == "cupy":
            with cp.cuda.Device(node["device"]), cp.cuda.Stream.null:
                value = cp.asarray(self.read_array(node["array"]))
                cp.cuda.Stream.null.synchronize()
        elif kind == "device":
            value = o3d.core.Device(node["value"])
        elif kind == "tensor":
            value = o3d.core.Tensor(self.read_array(node["array"])).to(o3d.core.Device(node["device"]))
        elif kind == "intrinsic":
            matrix = self.read_array(node["matrix"])
            value = o3d.camera.PinholeCameraIntrinsic(node["width"], node["height"], matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2])
            value.intrinsic_matrix = matrix
        elif kind == "cloud":
            value = o3d.geometry.PointCloud()
            for name, data in node["attributes"].items():
                array = self.read_array(data)
                constructor = o3d.utility.Matrix3dVector if name == "covariances" else o3d.utility.Vector3dVector
                setattr(value, name, constructor(array))
        elif kind == "image":
            value = o3d.geometry.Image(np.ascontiguousarray(self.read_array(node["array"])))
        elif kind == "rgbd":
            value = o3d.geometry.RGBDImage()
            value.color, value.depth = self.decode(node["color"]), self.decode(node["depth"])
        elif kind == "fpfh":
            value = o3d.pipelines.registration.Feature()
            value.data = self.read_array(node["data"])
        elif kind == "tensor_cloud":
            value = o3d.t.geometry.PointCloud(o3d.core.Device(node["device"]))
            for name, tensor in node["attributes"].items():
                value.point[name] = self.decode(tensor)
        elif kind == "registered":
            if node["name"] not in self.classes:
                raise CheckpointError("Unregistered class in private manifest")
            value = self.classes[node["name"]].__new__(self.classes[node["name"]])
            self.memo[index] = value
            for key, data in self.decode(node["state"]).items():
                object.__setattr__(value, key, data)
        elif kind == "preparation":
            from scanner_server.cuda_input import InputPreparation
            from scanner_server.cuda_confidence import ConfidencePreparation
            classes = {"scanner_server.cuda_input.InputPreparation": InputPreparation,
                       "scanner_server.cuda_confidence.ConfidencePreparation": ConfidencePreparation}
            if node["name"] not in classes:
                raise CheckpointError("Unregistered preparation class in private manifest")
            state = self.decode(node["state"])
            value = classes[node["name"]](state["device"], {})
            value.__dict__.update(state)
        elif kind == "native_rgbd":
            from scanner_server.cuda_input import NativeRgbd, CUDA_SOURCE
            value = NativeRgbd.__new__(NativeRgbd)
            value.cp, value.cuda_device = cp, cp.cuda.Device(node["device"])
            value.__dict__.update(self.decode(node["state"]))
            with value.cuda_device, cp.cuda.Stream.null:
                module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11", "--fmad=false"))
                for attr, entry in (("rectify", "rectify_clip"), ("reject", "reject_isolated"),
                                    ("project", "project_occlusion"), ("sample", "sample_color")):
                    kernel = module.get_function(entry)
                    kernel.compile()
                    setattr(value, attr, kernel)
                cp.cuda.Stream.null.synchronize()
        elif kind == "matcher":
            from scanner_server.cuda_matching import Matcher
            value = Matcher(node["device"])
            for feature, array in self.decode(node["bank"]):
                value.cache[id(feature)] = (feature, array)
        elif kind == "vbg":
            value = self._decode_volume(node)
        else:
            raise CheckpointError(f"Unsupported checkpoint graph node: {kind}")
        self.memo[index] = value
        return value

    def _decode_volume(self, node):
        np, core = self.np, self.o3d.core
        keys = self.read_array(node["keys"])
        volume = self.o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(core.float32, core.float32, core.float32), attr_channels=((1,), (1,), (3,)),
            voxel_size=float.fromhex(node["voxel_size_hex"]), block_resolution=16,
            block_count=node["capacity"], device=core.Device(node["device"]))
        hashmap = volume.hashmap()
        key_tensor = core.Tensor(keys, dtype=core.int32, device=core.Device(node["device"]))
        if len(keys):
            indices, inserted = hashmap.activate(key_tensor)
            if not bool(inserted.cpu().numpy().all()):
                raise CheckpointError("Fresh VBG activation omitted recorded logical keys")
            indices, found = hashmap.find(key_tensor)
            if not bool(found.cpu().numpy().all()):
                raise CheckpointError("Restored VBG omitted recorded logical keys")
            indices = indices.to(core.int64)
            for name, item in node["attributes"].items():
                digest, offset = hashlib.sha256(), 0
                for chunk in item["chunks"]:
                    array = self.read_array(chunk)
                    volume.attribute(name)[indices[offset:offset+len(array)]] = core.Tensor(array).to(core.Device(node["device"]))
                    offset += len(array)
                    digest.update(np.ascontiguousarray(array).tobytes())
                if offset != len(keys) or digest.hexdigest() != item["payload_sha256"]:
                    raise CheckpointError("Restored VBG chunk coverage changed")
        if str(volume.hashmap().device).startswith("CUDA:"):
            core.cuda.synchronize()
        check = self.volume_descriptor(volume, node)
        if not check["passed"]:
            raise CheckpointError("Restored VBG capacity/key/attribute bit verification failed")
        self.volume_checks.append(check)
        return volume

    def volume_descriptor(self, volume, expected):
        np, core = self.np, self.o3d.core
        hashmap = volume.hashmap()
        indices = hashmap.active_buf_indices().to(core.int64)
        keys = hashmap.key_tensor()[indices].cpu().numpy()
        order = np.lexsort(keys.T[::-1]) if len(keys) else np.empty(0, np.int64)
        indices = core.Tensor(indices.cpu().numpy()[order], dtype=core.int64, device=hashmap.device)
        attrs = {}
        for name in expected["attributes"]:
            digest = hashlib.sha256()
            for start in range(0, len(keys), VBG_CHUNK_BLOCKS):
                digest.update(np.ascontiguousarray(volume.attribute(name)[indices[start:start+VBG_CHUNK_BLOCKS]].cpu().numpy()).tobytes())
            attrs[name] = digest.hexdigest()
        return {"passed": int(hashmap.capacity()) == expected["capacity"] and int(hashmap.size()) == expected["size"]
                and array_metadata(np, keys[order])["payload_sha256"] == expected["keys"]["payload_sha256"]
                and all(attrs[k] == v["payload_sha256"] for k, v in expected["attributes"].items()),
                "capacity": int(hashmap.capacity()), "size": int(hashmap.size()), "attributes_sha256": attrs,
                "keys_sha256": array_metadata(np, keys[order])["payload_sha256"]}


def logical_graph(graph):
    """Exclude file placement/NPY headers, retain payload bits/layout/alias graph."""
    if isinstance(graph, dict):
        return {key: logical_graph(value) for key, value in graph.items() if key not in ("path", "sha256")}
    if isinstance(graph, list):
        return [logical_graph(value) for value in graph]
    return graph


def export_checkpoint(engine, prepared_inputs, folder, *, runtime_binding, scope_base, artifacts, capture_report):
    started = time.perf_counter()
    folder = Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise CheckpointError("Require a fresh empty private checkpoint directory")
    folder.mkdir(parents=True, exist_ok=True)
    names = inventory(engine)
    codec = GraphCodec(folder, writing=True)
    codec.engine_voxel = engine.voxel_size
    if set(prepared_inputs) != set(range(engine.stored_count)):
        raise CheckpointError("Every original stored observation needs its actual prepared-input capture")
    rng_state = {"python": random.getstate(), "numpy": codec.np.random.get_state()}
    root = codec.encode({"engine": engine.__dict__, "prepared_inputs": prepared_inputs, "rng_state": rng_state})
    graph = {"root": root, "nodes": codec.nodes}
    state = {"accepted_indices": [index for index, _ in engine.poses],
        "poses_sha256": pose_sha256(codec.np, engine.poses),
        "semantic_decisions_sha256": canonical_hash(semantic_decisions(engine.diagnostics)),
        "raw_diagnostics_sha256": canonical_hash(engine.diagnostics),
        "raw_frames_sha256": inputs_sha256(codec.np, engine.raw_frames),
        "prepared_inputs_sha256": inputs_sha256(codec.np, prepared_inputs),
        "unprocessed_count": 0, "pose_source": "fresh raw Live replay"}
    manifest = {"kind": KIND, "status": "complete", "source_sha256": source_hash(),
        "runtime_binding": runtime_binding, "scope_base": scope_base, "producer_artifacts_sha256": artifacts,
        "fresh_live": state, "engine_attribute_inventory": names, "graph": graph,
        "payloads": codec.payloads, "logical_state_sha256": canonical_hash(logical_graph(graph)),
        "capture_report": capture_report, "snapshot_wall_s": time.perf_counter()-started,
        "rng_scope": "Python and NumPy global state retained; original fragment RANSAC reseeds each proposal. Official OpenCV5 legacy solvePnPRansac EPNP source constructs local RNG(uint64(-1)) per call; this is source evidence, not binary attestation. OpenCV global RNG is not exposed/serialized; native binary/configuration plus actual same-checkpoint ordered proposal/gate checks remain required.",
        "serialization_scope": "Logical engine state and owned payloads/capacity; native transient frustum scratch/global allocator/kernel pools recreated and reported separately. Buffer indices may change, cached CPU clouds/feature order and every logical active voxel bit retained. No archived Live seeds or re-fusion."}
    if any(file_hash(ROOT/name) != value for name, value in artifacts.items()):
        raise CheckpointError("Checkpoint producer/source helpers changed during capture")
    path = folder/"checkpoint.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return path, manifest


def validate_checkpoint_files(path, expected_binding, expected_artifacts):
    path = Path(path).resolve(strict=True)
    before = file_hash(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("kind") != KIND or manifest.get("status") != "complete"
            or manifest.get("source_sha256") != source_hash()
            or manifest.get("runtime_binding") != expected_binding
            or manifest.get("producer_artifacts_sha256") != expected_artifacts
            or manifest.get("logical_state_sha256") != canonical_hash(logical_graph(manifest["graph"]))):
        raise CheckpointError("Private fresh Live checkpoint runtime/source/graph scope changed")
    for name, value in expected_artifacts.items():
        if file_hash(ROOT/name) != value:
            raise CheckpointError("Current checkpoint producer/helper bytes changed")
    declared = set()
    for item in manifest["payloads"]:
        payload = (path.parent/item["path"]).resolve()
        if payload.parent != path.parent or payload in declared or file_hash(payload) != item["sha256"]:
            raise CheckpointError("Checkpoint payload path/bytes changed or aliased")
        declared.add(payload)
    if file_hash(path) != before:
        raise CheckpointError("Checkpoint manifest changed during validation")
    return manifest, before


def materialize_checkpoint(path, *, expected_binding, expected_artifacts):
    started = time.perf_counter()
    manifest, digest = validate_checkpoint_files(path, expected_binding, expected_artifacts)
    codec = GraphCodec(Path(path).parent, writing=False, nodes=manifest["graph"]["nodes"], payloads=manifest["payloads"])
    restored = codec.decode(manifest["graph"]["root"])
    from scanner_server.engine import ScanEngine
    engine = ScanEngine.__new__(ScanEngine)
    engine.__dict__.update(restored["engine"])
    random.setstate(restored["rng_state"]["python"])
    codec.np.random.set_state(restored["rng_state"]["numpy"])
    if inventory(engine) != manifest["engine_attribute_inventory"]:
        raise CheckpointError("Restored engine transactional/attribute inventory differs")
    live = manifest["fresh_live"]
    if (pose_sha256(codec.np, engine.poses) != live["poses_sha256"]
            or canonical_hash(engine.diagnostics) != live["raw_diagnostics_sha256"]
            or canonical_hash(semantic_decisions(engine.diagnostics)) != live["semantic_decisions_sha256"]
            or [i for i, _ in engine.poses] != live["accepted_indices"]
            or inputs_sha256(codec.np, engine.raw_frames) != live["raw_frames_sha256"]
            or inputs_sha256(codec.np, restored["prepared_inputs"]) != live["prepared_inputs_sha256"]):
        raise CheckpointError("Restored freshly measured Live poses/decisions changed")
    # Re-encode a validation-only graph: arrays are compared directly with the
    # captured descriptors via a writing sink, not another expensive snapshot.
    checker = GraphCodec(Path(path).parent, writing=True)
    checker.engine_voxel = engine.voxel_size
    checker.array = lambda value: array_metadata(checker.np, value)
    check_root = checker.encode({"engine": engine.__dict__, "prepared_inputs": restored["prepared_inputs"],
                                "rng_state": {"python": random.getstate(), "numpy": codec.np.random.get_state()}})
    actual_graph = {"root": check_root, "nodes": checker.nodes}
    if canonical_hash(logical_graph(actual_graph)) != manifest["logical_state_sha256"]:
        raise CheckpointError("Restored complete engine/cache/alias/array graph changed")
    prepared_checks = 0
    helper = engine._input_preparation
    before_helper = dict(helper.status)
    for index, pair in restored["prepared_inputs"].items():
        # Use its original checked resident constants without advancing capture
        # diagnostics/counters; CPU preparation fallback stays original too.
        if helper._gpu is not None:
            actual = helper._gpu.prepare_host(*engine.raw_frames[index])
        else:
            from shared.calibration import prepare_rgbd
            actual = prepare_rgbd(*engine.raw_frames[index], engine.settings)
        if any(array_metadata(codec.np, a) != array_metadata(codec.np, b) for a, b in zip(actual, pair)):
            raise CheckpointError("Restored actual prepared RGB/depth input parity failed")
        prepared_checks += 1
    if helper.status != before_helper or prepared_checks != engine.stored_count:
        raise CheckpointError("Preparation verification mutated captured diagnostics or omitted input")
    validate_checkpoint_files(path, expected_binding, expected_artifacts)
    return engine, {"passed": True, "checkpoint_sha256": digest,
        "logical_state_equal": True, "logical_state_sha256": manifest["logical_state_sha256"],
        "raw_diagnostics_sha256": live["raw_diagnostics_sha256"], "poses_sha256": live["poses_sha256"],
        "raw_frames_sha256": live["raw_frames_sha256"], "prepared_inputs_sha256": live["prepared_inputs_sha256"],
        "volume_checks": codec.volume_checks, "prepared_input_checks": prepared_checks,
        "prepared_input_parity_passed": True, "wall_s": time.perf_counter()-started,
        "memory_scope": "Same recorded logical VBG capacity and all serialized engine/cache-owned buffers; native transient scratch and global pool reservations observed separately"}
