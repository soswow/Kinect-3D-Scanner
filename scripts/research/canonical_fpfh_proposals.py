"""Isolated proposal-only micrometre quantization and fresh FPFH experiment.

Imports are stdlib-only. The paired diagnostic CLI requires an allocated CPU
slot before importing NumPy/Open3D; it never runs fusion or acceptance gates.
The optional scope replaces only fragment _global_seed arguments. Original
RANSAC and every original verification body remain authoritative.
"""

from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import types
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POLICY = "proposal-only-fp64-micrometre-fresh-fpfh-v1"
KIND = "paired-proposal-only-canonical-fpfh-diagnostic-v1"
STEP_M = 1e-6
MAX_POINTS = 12000
MAX_COORDINATE_M = 1000.0
MAX_FRAGMENT_CACHE = 32
MAX_SNAPSHOT_BYTES = 128 * 1024**2
MAX_CALLS = 64
MAX_SCOPE_CALLS = 512  # Original MAX_PAIRS256, two original RANSAC calls per pair.
HALFWAY_GUARD_ULPS = 64
ORIGINAL_GLOBAL_AST_SHA256 = "c9accdf2d67172cbf812508d90a90ac7cf520480f5e809ac5ec3d43ca31afa14"
NORMAL_POLICIES = ("fresh", "largest-component-positive")
ARTIFACTS = ("scripts/research/canonical_fpfh_proposals.py", "scanner_server/fragments.py")


class CanonicalProposalError(RuntimeError):
    pass


class CanonicalProposalFailure(BaseException):
    """Research preprocessing/transport faults must escape ordinary rejection."""


def require(condition, message):
    if not condition:
        raise CanonicalProposalError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def policy(normal_policy="fresh"):
    require(normal_policy in NORMAL_POLICIES, "Unknown proposal normal policy")
    return {"name": POLICY, "step_m": STEP_M, "rounding": "FP64 rint(x/step)*step, ties to even",
        "halfway_guard_ulps": HALFWAY_GUARD_ULPS, "halfway_policy": "reject near-halfway coordinates before proposal work",
        "zero_policy": "normalize both signs of zero to positive zero", "order": "retain original row order and duplicates",
        "normal_policy": normal_policy, "normal_search": {"radius_m": .08, "max_nn": 30},
        "fpfh_search": {"radius_m": .2, "max_nn": 100}, "features_open3d_threads": 20,
        "max_points": MAX_POINTS, "max_coordinate_m": MAX_COORDINATE_M,
        "scope": "Private proposal coarse cloud/features only; original train/heldout/ICP/fusion/witness/acceptance data unchanged"}


def quantize_scalar(value, *, reject_halfway=True):
    """Stdlib FP64 reference; default rejects unstable quantization boundaries."""
    require(type(value) is float and math.isfinite(value) and abs(value) <= MAX_COORDINATE_M,
            "Require finite bounded original FP64 coordinates")
    scaled = value / STEP_M
    halfway_distance = abs(scaled - (math.floor(scaled) + .5))
    guard = HALFWAY_GUARD_ULPS * max(math.ulp(scaled), sys.float_info.epsilon)
    if reject_halfway:
        require(halfway_distance > guard, "Coordinate lies in the guarded micrometre halfway band")
    result = float(round(scaled)) * STEP_M
    if result == 0.0:
        result = 0.0
    bound = STEP_M / 2 + 2 * max(math.ulp(value), math.ulp(result))
    require(math.isfinite(result) and abs(result-value) <= bound, "Quantization exceeded its coordinate error bound")
    return result, {"halfway_distance_units": halfway_distance, "halfway_guard_units": guard,
                    "max_coordinate_delta_m": abs(result-value)}


def quantize_rows_reference(rows, *, reject_halfway=True):
    require(isinstance(rows, (list, tuple)) and 1 <= len(rows) <= MAX_POINTS, "Require a bounded nonempty N x 3 cloud")
    output = []
    for row in rows:
        require(isinstance(row, (list, tuple)) and len(row) == 3, "Require N x 3 original coordinates")
        output.append([quantize_scalar(value, reject_halfway=reject_halfway)[0] for value in row])
    return output


def orient_normal_reference(values):
    require(len(values) == 3 and all(type(x) is float and math.isfinite(x) for x in values), "Invalid fresh normal")
    axis = max(range(3), key=lambda index: abs(values[index]))  # exact ties: X, then Y, then Z
    require(abs(values[axis]) > 0, "Fresh normal has zero length")
    sign = -1.0 if values[axis] < 0 else 1.0
    return [0.0 if value == 0 else sign * value for value in values]


def canonical_points(np, points):
    require(points.dtype == np.float64 and points.ndim == 2 and points.shape[1] == 3
            and 1 <= len(points) <= MAX_POINTS and points.flags.c_contiguous,
            "Require bounded original contiguous FP64 N x 3 proposal points")
    require(np.isfinite(points).all() and float(np.max(np.abs(points))) <= MAX_COORDINATE_M,
            "Unsupported original proposal point coordinates")
    before = array_hash(points)
    scaled = points / STEP_M
    halfway = np.abs(scaled - (np.floor(scaled) + .5))
    guard = HALFWAY_GUARD_ULPS * np.maximum(np.abs(np.spacing(scaled)), np.finfo(np.float64).eps)
    guarded = halfway <= guard
    require(not bool(np.any(guarded)), "Coordinate lies in the guarded micrometre halfway band")
    rounded = np.rint(scaled) * STEP_M
    rounded[rounded == 0] = 0.0
    delta = np.abs(rounded-points)
    limit = STEP_M / 2 + 2 * np.maximum(np.abs(np.spacing(points)), np.abs(np.spacing(rounded)))
    require(np.isfinite(rounded).all() and bool(np.all(delta <= limit)), "Quantization exceeded coordinate error bound")
    require(array_hash(points) == before and not np.shares_memory(points, rounded), "Proposal quantization changed/shared original points")
    return rounded, {"input_sha256": before, "canonical_sha256": array_hash(rounded), "points": len(points),
        "maximum_coordinate_delta_m": float(np.max(delta)), "minimum_halfway_distance_units": float(np.min(halfway)),
        "minimum_halfway_margin_units": float(np.min(halfway-guard)), "guarded_coordinates": 0,
        "input_duplicate_rows": len(points)-len(np.unique(points, axis=0)),
        "canonical_duplicate_rows": len(points)-len(np.unique(rounded, axis=0))}


def code_signature(code):
    return (code.co_code, code.co_names, code.co_varnames, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_flags,
        tuple(code_signature(value) if isinstance(value, types.CodeType) else value for value in code.co_consts))


def original_binding(original):
    path = ROOT/"scanner_server/fragments.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_global_seed"]
    require(len(matches) == 1, "Missing unique original global RANSAC function")
    node = matches[0]
    compiled = compile(tree, str(path), "exec", dont_inherit=True)
    expected = next(value for value in compiled.co_consts if isinstance(value, types.CodeType) and value.co_name == node.name)
    require(isinstance(original, types.FunctionType) and original.__module__ == "scanner_server.fragments"
            and original.__closure__ is None and code_signature(original.__code__) == code_signature(expected),
            "Loaded global RANSAC differs from current original source; enter canonical scope before observers")
    # Explicit numeric/checker contract, in addition to loaded source identity.
    text = ast.dump(node, include_attributes=False)
    ast_hash = hashlib.sha256(text.encode()).hexdigest()
    require(ast_hash == ORIGINAL_GLOBAL_AST_SHA256 and "registration_ransac_based_on_feature_matching" in text and
            "RANSACConvergenceCriteria" in text and "value=12000" in text and "value=0.999" in text
            and "CorrespondenceCheckerBasedOnEdgeLength" in text and "value=0.9" in text,
            "Original RANSAC/checker budget contract changed")
    return {"fragments_sha256": file_hash(path),
        "global_seed_ast_sha256": ast_hash,
        "unchanged_original_ransac": True}


class FreshCanonicalFeatures:
    """Bounded private clouds; caller supplies already loaded numerical modules."""
    def __init__(self, np, o3d, *, normal_policy="fresh", max_cache=MAX_FRAGMENT_CACHE):
        require(type(max_cache) is int and 1 <= max_cache <= MAX_FRAGMENT_CACHE, "Unsupported private proposal cache cap")
        self.np, self.o3d = np, o3d
        self.policy, self.max_cache = policy(normal_policy), max_cache
        self.configuration_json = json.dumps({"policy":self.policy,"max_cache":max_cache},sort_keys=True,allow_nan=False)
        self.cache, self.rows = OrderedDict(), []

    def check_configuration(self):
        require(json.dumps({"policy":self.policy,"max_cache":self.max_cache},sort_keys=True,allow_nan=False)
                == self.configuration_json,"Canonical proposal policy/cache configuration changed during experiment")

    def snapshot(self, fragment):
        np = self.np
        return {"owners": {name:id(getattr(fragment,name)) for name in ("train","heldout","coarse","fpfh")},
            "arrays": {name: {"shape": list(np.asarray(value).shape), "sha256": array_hash(np.asarray(value))}
            for name,value in (("train_points",fragment.train.points),("train_normals",fragment.train.normals),
                ("heldout_points",fragment.heldout.points),("coarse_points",fragment.coarse.points),
                ("coarse_normals",fragment.coarse.normals),("fpfh",fragment.fpfh.data))}}

    def prepare(self, fragment):
        self.check_configuration()
        points = self.np.asarray(fragment.coarse.points)
        key = (id(fragment), array_hash(points))
        if key in self.cache:
            record, owner, cloud, features = self.cache.pop(key)
            require(owner is fragment, "Private fragment identity changed")
            require(array_hash(self.np.asarray(cloud.points)) == record["canonical_sha256"]
                    and array_hash(self.np.asarray(cloud.normals)) == record["normals_sha256"]
                    and array_hash(self.np.asarray(features.data)) == record["fpfh_sha256"],
                    "Cached canonical proposal feature bytes changed")
            self.cache[key] = (record, owner, cloud, features)
        else:
            cloud, features, record = self.build(points)
            record["fragment_index"] = fragment.index
            self.rows.append(record)
            self.cache[key] = (record, fragment, cloud, features)
            while len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        result = copy.copy(fragment)
        result.coarse, result.fpfh = cloud, features
        return result

    def build(self, points):
        self.check_configuration()
        np,o3d = self.np,self.o3d
        original_hash = array_hash(points)
        started = time.perf_counter()
        quantized, record = canonical_points(np, points)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(quantized)
        require(not cloud.has_normals(), "Private proposal cloud inherited normals")
        entry_threads = o3d.utility.get_max_threads()
        require(type(entry_threads) is int and entry_threads in (1,20), "Unsupported proposal thread context")
        primary = None
        try:
            o3d.utility.set_max_threads(20)
            require(o3d.utility.get_max_threads() == 20, "Could not install original feature preparation threads")
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.08,max_nn=30))
            normals = np.asarray(cloud.normals)
            require(normals.shape == quantized.shape and normals.dtype == np.float64 and np.isfinite(normals).all()
                    and bool(np.all(np.linalg.norm(normals,axis=1)>0)), "Invalid freshly estimated proposal normals")
            if self.policy["normal_policy"] == "largest-component-positive":
                axes = np.argmax(np.abs(normals),axis=1)
                signs = np.where(normals[np.arange(len(normals)),axes] < 0,-1.0,1.0)
                oriented = normals * signs[:,None]
                oriented[oriented == 0] = 0.0
                cloud.normals = o3d.utility.Vector3dVector(oriented)
            features = o3d.pipelines.registration.compute_fpfh_feature(cloud,
                o3d.geometry.KDTreeSearchParamHybrid(radius=.2,max_nn=100))
            require(np.asarray(features.data).shape == (33,len(points)) and np.isfinite(np.asarray(features.data)).all(),
                    "Invalid fresh proposal FPFH")
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                o3d.utility.set_max_threads(entry_threads)
                require(o3d.utility.get_max_threads() == entry_threads, "Feature preparation thread context failed to restore")
            except BaseException as cleanup:
                if primary is not None:
                    primary.add_note(f"Canonical FPFH thread restoration also failed: {cleanup}")
                else:
                    raise
        require(array_hash(points) == original_hash and array_hash(np.asarray(cloud.points)) == record["canonical_sha256"],
                "Fresh proposal feature preparation changed original/canonical points")
        record.update(normal_policy=self.policy["normal_policy"],normals_sha256=array_hash(np.asarray(cloud.normals)),
            fpfh_sha256=array_hash(np.asarray(features.data)),preparation_wall_s=time.perf_counter()-started,
            entry_threads=entry_threads,restored_threads=entry_threads)
        return cloud,features,record

    def clear(self):
        self.cache.clear()

    def report(self):
        return {"policy":copy.deepcopy(self.policy),"prepared":copy.deepcopy(self.rows),
            "cached_fragments":len(self.cache),"max_cache":self.max_cache,
            "maximum_retained_numeric_array_bytes":self.max_cache*MAX_POINTS*(3+3+33)*8,
            "memory_scope":"Private XYZ/normals/FPFH logical bytes only; native trees, allocator pools, metadata and temporary feature workspace additional"}


class CanonicalGlobalSeedScope:
    """Enter before original gate observers/RANSAC thread scope; no graph patch."""
    def __init__(self, module, builder, *, trace=None, binding_factory=original_binding):
        self.module,self.builder,self.trace = module,builder,trace
        self.original,self.binding_factory = module._global_seed,binding_factory
        self.failure,self.restored,self.entered = None,False,False
        self.rows,self.cleanup_failures,self.binding = [],[],None

    def fail(self,message,cause=None):
        if self.failure is None:
            self.failure = CanonicalProposalFailure(message)
            if cause is not None:
                self.failure.__cause__ = cause
        raise self.failure

    def call(self,source,target,seed):
        if self.failure is not None:
            raise self.failure
        row = {"source":source.index,"target":target.index,"seed":seed,"complete":False}
        self.rows.append(row)
        try:
            require(len(self.rows) <= MAX_SCOPE_CALLS,"Original bounded global proposal call budget exceeded")
            require(type(seed) is int, "Original RANSAC seed must remain an exact integer")
            before = [self.builder.snapshot(fragment) for fragment in (source,target)]
            private = [self.builder.prepare(fragment) for fragment in (source,target)]
            private_before = [self.builder.snapshot(fragment) for fragment in private]
            result = self.original(*private,seed)
            require([self.builder.snapshot(fragment) for fragment in (source,target)] == before,
                    "Canonical proposal path changed original verification inputs")
            require([self.builder.snapshot(fragment) for fragment in private] == private_before,
                    "Original RANSAC changed cached private proposal input bytes")
            row.update(complete=True,original_inputs_unchanged=True)
            if self.trace is not None:
                self.trace(row)
            return result
        except BaseException as error:
            row["failure"] = {"type":type(error).__name__,"message":str(error)}
            self.fail("Canonical proposal preprocessing/original RANSAC/evidence failed",error)

    def restore(self):
        errors = []
        for action in (lambda:setattr(self.module,"_global_seed",self.entry_function),self.builder.clear):
            try:
                action()
            except BaseException as error:
                errors.append(error)
                self.cleanup_failures.append({"type":type(error).__name__,"message":str(error)})
        self.restored = not errors
        return errors

    def __enter__(self):
        if self.entered:
            self.fail("Canonical global seed scope cannot be reused")
        self.entered = True
        self.entry_function = self.module._global_seed
        try:
            require(self.entry_function is self.original, "Canonical scope must precede other seed observers")
            self.binding = self.binding_factory(self.original)
            self.module._global_seed = self.call
        except BaseException as error:
            for secondary in self.restore():
                error.add_note(f"Canonical partial-entry cleanup also failed: {secondary}")
            self.fail("Canonical proposal source guard/installation failed",error)
        return self

    def __exit__(self,kind,error,traceback):
        errors = self.restore()
        if self.failure is not None:
            for secondary in errors:
                self.failure.add_note(f"Canonical scope cleanup also failed: {secondary}")
            raise self.failure
        if errors:
            self.fail("Canonical scope restoration failed",error if error is not None else errors[0])
        return False

    def report(self):
        return {"policy":self.builder.report(),"source_binding":self.binding,"calls":self.rows,
            "hooks_restored":self.restored,"cleanup_failures":self.cleanup_failures,
            "failure":None if self.failure is None else {"type":type(self.failure).__name__,"message":str(self.failure)},
            "geometry_quality_proven":False,"performance_authority":False}


def snapshot_header(stream):
    require(stream.read(6) == b"\x93NUMPY", "Snapshot member is not NPY")
    version = stream.read(2)
    require(version in (b"\x01\x00",b"\x02\x00"), "Unsupported NPY header version")
    size_bytes = stream.read(2 if version[0] == 1 else 4)
    require(len(size_bytes) == (2 if version[0] == 1 else 4), "Truncated NPY header size")
    length = int.from_bytes(size_bytes,"little")
    require(1 <= length <= 65536, "Unbounded NPY header")
    header = ast.literal_eval(stream.read(length).decode("latin1"))
    require(isinstance(header,dict) and set(header) == {"descr","fortran_order","shape"}, "Unexpected NPY descriptor")
    shape = header["shape"]
    require(header["descr"] == "<f8" and header["fortran_order"] is False
            and isinstance(shape,tuple) and len(shape) == 2 and all(type(x) is int and x > 0 for x in shape),
            "Require original numeric contiguous little-endian FP64 snapshots")
    return {"dtype":header["descr"],"shape":list(shape)},6+2+len(size_bytes)+length


def load_input_manifest(path):
    path = Path(path).resolve()
    report_bytes = path.read_bytes()
    data = json.loads(report_bytes)
    require(data.get("kind") == "offline-finish-global-proposal-input-diagnostic-v2" and data.get("status") == "complete"
            and data.get("supervisor_restored") is True and data.get("performance_authority") is False,
            "Require a closed original proposal input diagnostic, without timing authority")
    rows = data.get("rows")
    require(isinstance(rows,list) and 1 <= len(rows) <= MAX_CALLS and all(row.get("complete") is True for row in rows),
            "Require bounded complete original proposal rows")
    names = {f"{side}_{part}" for side in ("source","target") for part in
             ("train_points","train_normals","coarse_points","coarse_normals","fpfh")}
    descriptors = {}
    for index,row in enumerate(rows):
        require(row.get("invocation") == index and all(type(row.get(key)) is int for key in ("source","target","original_seed"))
                and set(row.get("inputs",{})) == names,"Malformed ordered original proposal descriptor")
        for name,descriptor in row["inputs"].items():
            require(set(descriptor) == {"dtype","shape","sha256","snapshot_key"} and descriptor["dtype"] == "<f8"
                    and isinstance(descriptor["snapshot_key"],str) and descriptor["snapshot_key"].isidentifier()
                    and descriptor["snapshot_key"] not in descriptors,"Invalid/reused snapshot descriptor")
            shape = descriptor["shape"]
            require(isinstance(shape,list) and len(shape) == 2 and all(type(x) is int for x in shape)
                    and (shape[0] == 33 and 1 <= shape[1] <= MAX_POINTS if name.endswith("fpfh")
                         else 1 <= shape[0] <= MAX_POINTS and shape[1] == 3),"Unsupported snapshot array shape")
            require(isinstance(descriptor["sha256"],str) and len(descriptor["sha256"]) == 64,"Missing snapshot payload hash")
            descriptors[descriptor["snapshot_key"]] = descriptor
    snapshot = Path(data["snapshot"]["path"]).resolve()
    require(file_hash(snapshot) == data["snapshot"]["sha256"], "Captured NPZ differs from closed diagnostic")
    with zipfile.ZipFile(snapshot) as archive:
        members = archive.infolist()
        require(len(members) == len(descriptors) and len({item.filename for item in members}) == len(members)
                and sum(item.file_size for item in members) <= MAX_SNAPSHOT_BYTES,"Unbounded/duplicate NPZ payloads")
        require({item.filename for item in members} == {key+".npy" for key in descriptors},"Snapshot NPZ member scope differs")
        for item in members:
            require(not item.flag_bits & 1,"Encrypted snapshot member unsupported")
            with archive.open(item) as stream:
                header,size = snapshot_header(stream)
                expected = descriptors[item.filename[:-4]]
                require(header == {"dtype":expected["dtype"],"shape":expected["shape"]}
                        and item.file_size == size+math.prod(header["shape"])*8,"NPY header/size differs from closed metadata")
    return data,{"report_path":str(path),"report_sha256":hashlib.sha256(report_bytes).hexdigest(),"snapshot_path":str(snapshot),
                "snapshot_sha256":file_hash(snapshot)},descriptors


def validate_pair(left,right):
    require(left["artifacts_sha256"] == right["artifacts_sha256"],"Paired original diagnostics used different source policies")
    require([(row["source"],row["target"],row["original_seed"]) for row in left["rows"]] ==
            [(row["source"],row["target"],row["original_seed"]) for row in right["rows"]],
            "Paired original proposal membership/order/seeds differ")
    for a,b in zip(left["rows"],right["rows"]):
        require(all(a["inputs"][name]["shape"] == b["inputs"][name]["shape"] for name in a["inputs"]),
                "Paired point/normal/feature shape/order scope differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left",type=Path,required=True)
    parser.add_argument("--right",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--normal-policy",choices=NORMAL_POLICIES,default="fresh")
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--run-allocated",action="store_true")
    args = parser.parse_args()
    require(args.run_allocated and type(args.repeats) is int and 1 <= args.repeats <= 3,
            "Require allocated CPU/native slot and bounded repeats")
    output = args.output.resolve()
    output.relative_to(ROOT/"benchmark-output")
    require(not output.exists(),"Require fresh diagnostic output; preserve measured artifacts")
    pins = {name:file_hash(ROOT/name) for name in ARTIFACTS}
    report = {"kind":KIND,"status":"running","policy":policy(args.normal_policy),"artifacts_sha256":pins,
        "performance_authority":False,"geometry_quality_proven":False,"original_ransac_exercised":False,
        "scope":"Paired captured coarse inputs only; no Finish, RANSAC, GPU, graph or surface acceptance claim","rows":[]}
    primary = None
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    try:
        left,lbind,ldesc = load_input_manifest(args.left)
        right,rbind,rdesc = load_input_manifest(args.right)
        validate_pair(left,right)
        report["inputs"] = {"left":lbind,"right":rbind}
        save()
        import os
        os.environ.setdefault("OMP_NUM_THREADS","8")
        require(os.environ["OMP_NUM_THREADS"] == "8","Require original OMP8 feature context")
        import numpy as np
        import open3d as o3d
        o3d.utility.set_max_threads(20)
        report["runtime"] = {"numpy":np.__version__,"open3d":o3d.__version__,"numpy_module_sha256":file_hash(np.__file__),
            "open3d_module_sha256":file_hash(o3d.__file__),"open3d_threads":o3d.utility.get_max_threads(),"omp":"8"}
        report["runtime"]["loaded_native_modules"] = {name:{"path":str(Path(module.__file__).resolve()),
            "sha256":file_hash(module.__file__)} for name,module in list(sys.modules.items())
            if name in ("open3d.cpu.pybind","open3d.cuda.pybind","numpy._core._multiarray_umath","numpy.core._multiarray_umath")
            and getattr(module,"__file__",None)}
        loaded = []
        for binding,descriptors in ((lbind,ldesc),(rbind,rdesc)):
            with np.load(binding["snapshot_path"],allow_pickle=False) as archive:
                arrays = {key:archive[key] for key in descriptors}
            require(all(arrays[key].dtype == np.float64 and list(arrays[key].shape) == desc["shape"]
                and np.isfinite(arrays[key]).all() and array_hash(arrays[key]) == desc["sha256"]
                for key,desc in descriptors.items()),"Original snapshot payload hash/domain differs")
            loaded.append(arrays)
        builder = FreshCanonicalFeatures(np,o3d,normal_policy=args.normal_policy)
        all_equal = True
        for index,(a,b) in enumerate(zip(left["rows"],right["rows"])):
            for side in ("source","target"):
                arrays = [loaded[0][a["inputs"][side+"_coarse_points"]["snapshot_key"]],
                          loaded[1][b["inputs"][side+"_coarse_points"]["snapshot_key"]]]
                hashes = []
                metadata = []
                for repeat in range(args.repeats):
                    for which,points in enumerate(arrays):
                        cloud,features,record = builder.build(points)
                        hashes.append((record["canonical_sha256"],record["normals_sha256"],record["fpfh_sha256"]))
                        metadata.append({"side":("left","right")[which],"repeat":repeat,**record})
                flags = {name:len({values[position] for values in hashes}) == 1
                    for position,name in enumerate(("canonical_points_exact","fresh_normals_exact","fresh_fpfh_exact"))}
                all_equal = all_equal and all(flags.values())
                report["rows"].append({"invocation":index,"fragment_side":side,"fragment_index":a[side],
                    "original_points_exact":array_hash(arrays[0]) == array_hash(arrays[1]),
                    "maximum_original_point_delta_m":float(np.max(np.abs(arrays[0]-arrays[1]))),
                    "row_order_preserved":True,**flags,"builds":metadata})
                save()
        require(all_equal,"Canonical paired/repeated point, normal or FPFH bytes differ")
        require(all(array_hash(loaded[which][key]) == desc["sha256"]
            for which,descriptors in enumerate((ldesc,rdesc)) for key,desc in descriptors.items()),
            "Original captured arrays changed during feature experiment")
        report.update(status="passed",input_arrays_unchanged=True,all_paired_repeated_bytes_equal=True)
    except BaseException as error:
        primary = error
        report.update(status="failed",failure={"type":type(error).__name__,"message":str(error)})
        raise
    finally:
        try:
            report["artifacts_sha256_after"] = {name:file_hash(ROOT/name) for name in ARTIFACTS}
            require(report["artifacts_sha256_after"] == pins,"Proposal experiment sources changed")
            for binding in report.get("inputs",{}).values():
                require(file_hash(binding["report_path"]) == binding["report_sha256"] and
                        file_hash(binding["snapshot_path"]) == binding["snapshot_sha256"],"Captured input artifacts changed")
            report["input_files_unchanged"] = True
        except BaseException as cleanup:
            report.update(status="failed",closure_failure={"type":type(cleanup).__name__,"message":str(cleanup)})
            if primary is not None:
                primary.add_note(f"Canonical FPFH provenance closure also failed: {cleanup}")
            else:
                primary = cleanup
        try:
            save()
        except BaseException as write_error:
            if primary is not None:
                primary.add_note(f"Canonical FPFH report write also failed: {write_error}")
                raise primary from write_error
            raise
        if primary is not None and sys.exception() is None:
            raise primary


if __name__ == "__main__":
    main()
