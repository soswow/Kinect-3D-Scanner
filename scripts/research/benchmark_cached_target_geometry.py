"""Root-allocated cached-target lookup parity and separately scoped timings.

Search-only ID/d2 helper versus original EvaluateRegistration (which also
transforms, rebuilds a tree and reduces metrics). No Finish-query/gate or scanner
speed authority. Numerical imports and DLL loading occur only after preflight
and explicit --run-allocated. Optional mesh vertices are field-derived data only.
"""
from __future__ import annotations
import argparse
import ast
import copy
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from types import CodeType, FunctionType
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.research import cached_target_geometry_native as helper

KIND = "cached-target-geometry-original-api-parity-benchmark-v1"
OWN_FILES = ("scripts/research/benchmark_cached_target_geometry.py", "tests/test_cached_target_geometry_benchmark.py")
FIELD_KIND = "gpu-icp-whole-finish-candidate-native-v1"
MAX_JSON_BYTES = 64 * 1024 * 1024
IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


def require(ok, message):
    if not ok: raise ValueError(message)


def read_json(path):
    path = Path(path).resolve(strict=True)
    require(path.stat().st_size <= MAX_JSON_BYTES, "Bounded JSON receipt required")
    def invalid(value): raise ValueError("Nonfinite JSON value: " + value)
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    json.dumps(value, allow_nan=False)  # Also reject numeric overflow such as 1e400.
    return value


def point_header(path):
    """Inspect only a bounded numeric NPY header; no NumPy load/decompression."""
    path = Path(path).resolve(strict=True)
    require(path.stat().st_size <= 256 * 1024 * 1024, "Bounded field NPZ required")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) <= {"points.npy", "faces.npy", "colors.npy"}
                and "points.npy" in names, "Original geometry NPZ members required")
        item = archive.getinfo("points.npy")
        require(item.file_size <= 128 * 1024 * 1024, "Bounded numeric geometry points required")
        with archive.open(item) as stream:
            require(stream.read(6) == b"\x93NUMPY", "Missing NPY magic")
            version = tuple(stream.read(2)); require(version in ((1, 0), (2, 0), (3, 0)), "Unsupported NPY version")
            size_bytes = 2 if version == (1, 0) else 4
            raw_size = stream.read(size_bytes)
            require(len(raw_size) == size_bytes, "Truncated NPY header")
            size = int.from_bytes(raw_size, "little"); require(0 < size <= 4096, "Bounded NPY header required")
            raw = stream.read(size); require(len(raw) == size, "Truncated NPY header")
            header = ast.literal_eval(raw.decode("utf-8" if version == (3, 0) else "latin1"))
        require(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}, "Numeric NPY descriptor required")
        shape = header["shape"]
        require(header["descr"] == "<f8" and type(header["fortran_order"]) is bool and type(shape) is tuple
                and len(shape) == 2 and all(type(n) is int for n in shape) and 0 < shape[0] <= 4_000_000
                and shape[1] == 3 and item.file_size == 8 + size_bytes + size + 24 * shape[0],
                "Require bounded original FP64 geometry points")
    return {"dtype": "<f8", "shape": list(shape), "fortran_order": header["fortran_order"]}


def preflight(args):
    require(args.run_allocated, "Root-exclusive allocation requires --run-allocated")
    args.output = Path(args.output).resolve()
    require(not args.output.exists(), "Refuse an existing output")
    args.library, args.build_receipt = Path(args.library).resolve(strict=True), Path(args.build_receipt).resolve(strict=True)
    receipt = read_json(args.build_receipt)
    require(receipt.get("kind") == helper.KIND and receipt.get("stage") == "built"
            and type(receipt.get("actual_exit_code")) is int and receipt["actual_exit_code"] == 0
            and receipt.get("source_contract") == helper.source_contract()
            and receipt.get("library_sha256") == helper.file_hash(args.library), "Current successful helper build receipt required")
    command = receipt.get("command", [])
    require(type(command) is list and all(type(arg) is str for arg in command)
            and [arg for arg in command if arg.lower().startswith("/fp:")] == ["/fp:strict"]
            and "/std:c++17" in command and "/DEIGEN_DONT_PARALLELIZE" in command
            and not any("nanoflann_first_match" in arg.lower() for arg in command), "Original pinned floating/tie build policy required")
    compiler = receipt.get("compiler", {})
    require(type(compiler) is dict and helper.file_hash(compiler["path"]) == compiler.get("sha256"), "Current compiler bytes differ")
    for name, pinned, license_name in (("eigen", helper.EIGEN_ARCHIVE_SHA256, "COPYING.MPL2"),
                                      ("nanoflann", helper.NANOFLANN_ARCHIVE_SHA256, "COPYING")):
        dependency = receipt.get("dependencies", {}).get(name, {})
        files, header_sha = dependency.get("files", []), dependency.get("headers_sha256")
        require(dependency.get("archive_sha256") == pinned and type(files) is list and all(type(item) is str for item in files)
                and len(files) == len(set(files)) and license_name in files
                and type(header_sha) is str and len(header_sha) == 64 and all(c in "0123456789abcdef" for c in header_sha),
                "Original dependency/license receipt required")
    fixed = {str(args.library): helper.file_hash(args.library), str(args.build_receipt): helper.file_hash(args.build_receipt),
             str(Path(compiler["path"]).resolve()): compiler["sha256"], str(Path(sys.executable).resolve()): helper.file_hash(sys.executable)}
    field = None
    require((args.field_geometry is None) == (args.field_report is None), "Provide field geometry and closed native report together")
    if args.field_geometry is not None:
        args.field_geometry, args.field_report = Path(args.field_geometry).resolve(strict=True), Path(args.field_report).resolve(strict=True)
        native = read_json(args.field_report); geometry = native.get("geometry", {})
        profile = native.get("profile", {})
        require(native.get("kind") == FIELD_KIND and native.get("mode") == "native" and native.get("status") == "passed"
                and native.get("failure") is None and native.get("cleanup_passed") is True and native.get("cleanup_failures") == []
                and type(native.get("binding")) is dict and native["binding"] == native.get("binding_after")
                and type(native.get("candidate_source")) is dict and native["candidate_source"] == native.get("candidate_source_after")
                and native.get("loaded_owners_unchanged") is True and profile.get("mesh_built") is True,
                "Closed original candidate-native mesh receipt required; no old proof accepted")
        require(Path(geometry.get("path", "")).resolve() == args.field_geometry
                and geometry.get("sha256") == helper.file_hash(args.field_geometry)
                and profile.get("geometry", {}).get("sha256") == geometry["sha256"], "Field geometry must equal the closed native export")
        field = {"report_path": str(args.field_report), "report_sha256": helper.file_hash(args.field_report),
                 "geometry_path": str(args.field_geometry), "geometry_sha256": geometry["sha256"],
                 "point_header": point_header(args.field_geometry), "actual_finish_queries": False}
        fixed.update({str(args.field_report): field["report_sha256"], str(args.field_geometry): field["geometry_sha256"]})
    return {"helper_source": helper.source_contract(), "recorded_build_dependencies": copy.deepcopy(receipt["dependencies"]),
        "dependency_assets_rehashed_at_runtime": False, "artifacts_sha256": {
        name: helper.file_hash(ROOT / name) for name in OWN_FILES + ("scripts/research/cached_target_geometry_native.md",)},
        "fixed_files_sha256": fixed, "field_data": field}


def check_fixed(binding):
    require(helper.source_contract() == binding["helper_source"], "Held helper source changed")
    for name, expected in binding["artifacts_sha256"].items():
        require(helper.file_hash(ROOT / name) == expected, "Benchmark source changed: " + name)
    for name, expected in binding["fixed_files_sha256"].items():
        require(helper.file_hash(name) == expected, "Fixed resource changed: " + name)


def code_key(code):
    return (code.co_code, tuple(code_key(c) if isinstance(c, CodeType) else c for c in code.co_consts),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_flags)


class LoadedOwners:
    def __init__(self, o3d):
        self.records = []
        for module in (helper, sys.modules[__name__]):
            cold = compile(Path(module.__file__).read_text(encoding="utf-8"), module.__file__, "exec", dont_inherit=True)
            for code in (value for value in cold.co_consts if isinstance(value, CodeType)):
                owner = getattr(module, code.co_name, None)
                slots = [(module, code.co_name, owner, code)]
                if isinstance(owner, type):
                    slots = [(owner, c.co_name, getattr(owner, c.co_name, None), c)
                             for c in code.co_consts if isinstance(c, CodeType)]
                for slot, name, function, expected in slots:
                    if not isinstance(function, FunctionType): continue
                    require(function.__globals__ is module.__dict__ and code_key(function.__code__) == code_key(expected), "Loaded Python body differs from source")
                    aliases = [(key, module.__dict__[key]) for key in function.__code__.co_names if key in module.__dict__]
                    self.records.append((slot, name, function, function.__code__, repr(function.__defaults__), repr(function.__kwdefaults__), module, aliases))
        self.native = [(o3d.geometry, "PointCloud", o3d.geometry.PointCloud),
                       (o3d.geometry, "KDTreeFlann", o3d.geometry.KDTreeFlann),
                       (o3d.utility, "Vector3dVector", o3d.utility.Vector3dVector),
                       (o3d.utility, "get_max_threads", o3d.utility.get_max_threads),
                       (o3d.utility, "set_max_threads", o3d.utility.set_max_threads),
                       (o3d.pipelines.registration, "evaluate_registration", o3d.pipelines.registration.evaluate_registration),
                       (o3d.geometry.PointCloud, "transform", o3d.geometry.PointCloud.transform),
                       (o3d.geometry.KDTreeFlann, "search_hybrid_vector_3d", o3d.geometry.KDTreeFlann.search_hybrid_vector_3d)]
        self.check()

    def check(self):
        for slot, name, function, code, defaults, kwdefaults, module, aliases in self.records:
            require(getattr(slot, name) is function and function.__code__ is code
                    and function.__globals__ is module.__dict__ and repr(function.__defaults__) == defaults
                    and repr(function.__kwdefaults__) == kwdefaults
                    and all(module.__dict__.get(key) is value for key, value in aliases), "Loaded helper/function owner changed")
        require(all(getattr(slot, name) is value for slot, name, value in self.native), "Original native callable owner changed")


def runtime_binding(np, o3d, backend):
    paths = {Path(sys.executable).resolve()}
    for name in {o3d.geometry.KDTreeFlann.__module__.split(".geometry")[0],
                 o3d.pipelines.registration.evaluate_registration.__module__.split(".pipelines")[0]}:
        module = sys.modules.get(name); require(module is not None and getattr(module, "__file__", None), "Cannot bind loaded Open3D backend")
        paths.add(Path(module.__file__).resolve())
    package = Path(o3d.__file__).resolve().parent
    for name in ("Open3D.dll", "tbb12.dll"):
        for folder in (package, package / "cpu", package / "cuda"):
            if (folder / name).is_file(): paths.add((folder / name).resolve())
    numpy_modules = [module for name, module in sys.modules.items() if name.startswith("numpy.")
                     and name.endswith("._multiarray_umath") and getattr(module, "__file__", None)]
    require(numpy_modules, "Cannot bind loaded NumPy metric binary")
    paths.update(Path(module.__file__).resolve() for module in numpy_modules)
    return {"python": sys.version, "numpy": np.__version__, "open3d": o3d.__version__,
        "binaries": {str(path): helper.file_hash(path) for path in sorted(paths)},
        "numpy_cpu_features": dict(getattr(numpy_modules[0], "__cpu_features__", {})),
        "numpy_configuration_sha256": hashlib.sha256(json.dumps(np.__config__.CONFIG, sort_keys=True, default=str).encode()).hexdigest(),
        "original_threads": o3d.utility.get_max_threads(), "omp": os.environ.get("OMP_NUM_THREADS"),
        "cached_native": backend.closure(), "current_fp_environment": int(backend.dll.ctgn_fp_environment())}


def synthetic_specs():
    axes = [( .25, 0., 0.), (-.25, 0., 0.), (0., .25, 0.), (0., -.25, 0.)]
    duplicates = [(0., -0., 0.), (-0., 0., -0.), (0., 0., 0.), (.25, 0., 0.)]
    for family, targets in (("ties", axes), ("duplicates_signed_zero", duplicates)):
        for index, order in enumerate(itertools.permutations(range(4))):
            yield {"name": f"{family}-{index}", "family": family, "targets": [targets[i] for i in order],
                   "queries": [(0., -0., 0.), (math.nextafter(0., 1.), 0., 0.), (.125, 0., 0.)], "radius": .5, "transform": None}
    for radius in (2 ** -20, .03, .12, .5, 1.):
        yield {"name": f"strict-radius-{radius}", "family": "strict_radius_adjacent",
            "targets": [(0., 0., 0.)], "queries": [(radius, 0., 0.), (math.nextafter(radius, 0.), 0., 0.),
                (math.nextafter(radius, math.inf), 0., 0.), (-radius, 0., 0.)], "radius": radius, "transform": None}
    many = axes * 8
    for offset in (0, 7, 15, 23, 31):
        yield {"name": f"split-node-ties-{offset}", "family": "ties_across_leaf15",
            "targets": many[offset:] + many[:offset], "queries": [(0., 0., 0.), (.125, 0., 0.), (-.125, 0., 0.)], "radius": .5, "transform": None}
    yield {"name": "finite-extreme", "family": "finite_domain_extreme", "targets": [(2 ** 20, 0., 0.), (-(2 ** 20), 0., 0.), (0., 0., 0.)],
           "queries": [(2 ** 20 - .125, 0., 0.), (-(2 ** 20) + .125, 0., 0.), (0., 2 ** 20, 0.)], "radius": .5, "transform": None}
    yield {"name": "subnormal", "family": "subnormal", "targets": [(0., 0., 0.), (math.nextafter(0., 1.), 0., 0.)],
           "queries": [(math.nextafter(0., -1.), 0., 0.), (0., -0., 0.)], "radius": 2 ** -20, "transform": None}
    for name, transform in (("rigid", [[0., -1., 0., .01], [1., 0., 0., -.02], [0., 0., 1., .03], [0., 0., 0., 1.]]),
                             ("homogeneous", [[1., 0., 0., .01], [0., 1., 0., 0.], [0., 0., 1., 0.], [.001, 0., 0., 1.]])):
        yield {"name": "original-transform-" + name, "family": "original_transform", "targets": axes,
               "queries": [(0., 0., 0.), (.1, .2, .3)], "radius": .5, "transform": transform}
    yield {"name": "empty-queries", "family": "empty_queries", "targets": axes, "queries": [], "radius": .03, "transform": None}


def bytes_hash(value): return hashlib.sha256(value).hexdigest()


def prepare_case(spec, np, o3d):
    started = time.perf_counter()
    targets = np.ascontiguousarray(spec["targets"], dtype="<f8").reshape(-1, 3)
    source_values = np.ascontiguousarray(spec["queries"], dtype="<f8").reshape(-1, 3)
    target = o3d.geometry.PointCloud(); target.points = o3d.utility.Vector3dVector(targets)
    source = o3d.geometry.PointCloud(); source.points = o3d.utility.Vector3dVector(source_values)
    transform = np.asarray(spec["transform"] or IDENTITY, dtype="<f8")
    transform_start = time.perf_counter(); moving = copy.deepcopy(source)
    if spec["transform"] is not None: moving.transform(transform)
    queries = np.ascontiguousarray(np.asarray(moving.points), dtype="<f8")
    target_bytes, source_bytes, query_bytes = targets.tobytes(), source_values.tobytes(), queries.tobytes()
    transformation_and_snapshot_s = time.perf_counter() - transform_start
    record = {"name": spec["name"], "family": spec["family"], "radius": spec["radius"],
        "target": helper.points_descriptor(target_bytes, len(targets)),
        "source": helper.points_descriptor(source_bytes, len(source_values), allow_empty=True),
        "queries": helper.points_descriptor(query_bytes, len(queries), allow_empty=True),
        "transform_sha256": bytes_hash(transform.tobytes()), "original_transform_called": spec["transform"] is not None,
        "original_transform_and_query_snapshot_s": transformation_and_snapshot_s,
        "input_preparation_wall_s": time.perf_counter() - started, "parity": []}
    return {"record": record, "target": target, "source": source, "moving": moving, "matrix": transform,
            "target_bytes": target_bytes, "source_bytes": source_bytes, "query_bytes": query_bytes}


def check_case(case, np):
    for cloud, expected in ((case["target"], case["target_bytes"]), (case["source"], case["source_bytes"]), (case["moving"], case["query_bytes"])):
        require(np.ascontiguousarray(np.asarray(cloud.points), dtype="<f8").tobytes() == expected, "Original cloud/query bytes changed")
    require(bytes_hash(case["matrix"].tobytes()) == case["record"]["transform_sha256"], "Original transform changed")


class OwnedTarget:
    def __init__(self, backend): self.backend = backend; self._backend = backend; self.handle = None; self.closed = False
    def create(self, points, rows):
        require(self.backend is self._backend, "Raw target backend owner changed")
        self.handle, owned = self._backend.create(points, rows, helper.reservation(rows))
        require(owned == helper.reservation(rows), "Native reservation receipt changed")
    def close(self):
        if self.handle is not None:
            self._backend.destroy(self.handle); self.handle = None
        self.closed = True


def close_one(owner, primary=None):
    try: owner.close()
    except BaseException as error:
        if primary is not None:
            if callable(getattr(primary, "add_note", None)): primary.add_note("Native target cleanup: " + repr(error))
            raise primary from error
        raise


def scalar_gold(case, np, o3d):
    started = time.perf_counter(); tree = o3d.geometry.KDTreeFlann(case["target"])
    build_s = time.perf_counter() - started; started = time.perf_counter()
    rows = case["record"]["queries"]["shape"][0]
    ids = np.full(rows, -1, dtype="<i8"); squared = np.full(rows, np.inf, dtype="<f8")
    for row, point in enumerate(np.asarray(case["moving"].points)):
        count, indices, distances = tree.search_hybrid_vector_3d(point, case["record"]["radius"], 1)
        require(count in (0, 1) and len(indices) == count and len(distances) == count, "Malformed original scalar result")
        if count: ids[row], squared[row] = int(indices[0]), float(distances[0])
    ids, squared = ids.tobytes(), squared.tobytes()
    helper.validate_output(ids, squared, rows, case["record"]["target"]["shape"][0], case["record"]["radius"])
    case["record"]["original_scalar_reference"] = {"ids_sha256": bytes_hash(ids), "squared_sha256": bytes_hash(squared),
        "hits": sum(i >= 0 for (i,) in struct.iter_unpack("<q", ids)), "query_rows": rows,
        "tree_build_s": build_s, "scalar_shadow_s": time.perf_counter() - started}
    case["record"]["original_scalar_reference"]["misses"] = rows - case["record"]["original_scalar_reference"]["hits"]
    case["gold"] = (ids, squared)


def exact_result(case, ids, squared):
    rows = case["record"]["queries"]["shape"][0]
    helper.validate_output(ids, squared, rows, case["record"]["target"]["shape"][0], case["record"]["radius"])
    require(ids == case["gold"][0], "Canonical original nearest IDs differ")
    require(squared == case["gold"][1], "Original nearest squared-distance bits differ")


def audit_case(case, backend, np, o3d, owners):
    scalar_gold(case, np, o3d)
    raw = OwnedTarget(backend); owners.append(raw); primary = None
    try:
        raw.create(case["target_bytes"], case["record"]["target"]["shape"][0])
        for threads in helper.QUERY_THREADS:
            result = backend.query(raw.handle, case["query_bytes"], case["record"]["queries"]["shape"][0], case["record"]["radius"], threads)
            exact_result(case, *result)
            cache = helper.CachedTargets(backend, query_threads=threads); owners.append(cache); cache_error = None
            try:
                wrapped = cache.query(case["target_bytes"], case["record"]["target"]["shape"][0], case["query_bytes"],
                                      case["record"]["queries"]["shape"][0], case["record"]["radius"])
                exact_result(case, wrapped["ids"], wrapped["squared"])
            except BaseException as error: cache_error = error; raise
            finally: close_one(cache, cache_error)
            require(cache.report()["closed"] and cache.report()["retained_targets"] == 0, "Temporary cache did not close")
            case["record"]["parity"].append({"threads": threads, "raw_ids_exact": True, "raw_squared_bits_exact": True,
                                           "wrapper_ids_exact": True, "wrapper_squared_bits_exact": True, "owners_closed": True})
    except BaseException as error: primary = error; raise
    finally: close_one(raw, primary)
    check_case(case, np); case["record"]["complete"] = True


def evaluation_record(case, result, np):
    rows, target_rows = case["record"]["queries"]["shape"][0], case["record"]["target"]["shape"][0]
    pairs = np.asarray(result.correspondence_set)
    require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in "iu", "Malformed original evaluation pairs")
    ids = np.full(rows, -1, dtype="<i8")
    require(len(pairs) <= rows and (not len(pairs) or (np.all((pairs[:, 0] >= 0) & (pairs[:, 0] < rows))
            and np.all((pairs[:, 1] >= 0) & (pairs[:, 1] < target_rows)) and len(np.unique(pairs[:, 0])) == len(pairs))), "Original pair bounds/uniqueness")
    if len(pairs): ids[pairs[:, 0]] = pairs[:, 1]
    require(ids.tobytes() == case["gold"][0], "Original EvaluateRegistration mapping differs from scalar reference")
    fitness, rmse = float(result.fitness), float(result.inlier_rmse)
    require(math.isfinite(fitness) and 0 <= fitness <= 1 and math.isfinite(rmse) and rmse >= 0, "Nonfinite original evaluation metrics")
    return {"fitness": fitness, "rmse": rmse, "canonical_ids_sha256": bytes_hash(ids.tobytes()),
            "raw_pairs_sha256": bytes_hash(pairs.tobytes()), "hits": len(pairs), "canonical_ids_exact": True,
            "helper_rmse_equivalence_claim": False}


def timed_lookup(case, backend, threads, repeats, owners, wrapped=False, destination=None):
    started = time.perf_counter()
    result = {"threads": threads, "warm_query_wall_s": [], "complete": False}
    if destination is not None: destination.append(result)
    owner = helper.CachedTargets(backend, query_threads=threads) if wrapped else OwnedTarget(backend)
    owners.append(owner); primary = None
    try:
        if not wrapped: owner.create(case["target_bytes"], case["record"]["target"]["shape"][0])
        def query():
            if wrapped:
                value = owner.query(case["target_bytes"], case["record"]["target"]["shape"][0], case["query_bytes"],
                                    case["record"]["queries"]["shape"][0], case["record"]["radius"])
                return value["ids"], value["squared"]
            return backend.query(owner.handle, case["query_bytes"], case["record"]["queries"]["shape"][0], case["record"]["radius"], threads)
        first = query(); result["cold_constructor_and_first_query_wall_s"] = time.perf_counter() - started
        exact_result(case, *first)
        for _ in range(repeats):
            before = time.perf_counter(); value = query(); elapsed = time.perf_counter() - before
            result["warm_query_wall_s"].append(elapsed); exact_result(case, *value)
    except BaseException as error: primary = error; raise
    finally:
        before = time.perf_counter()
        try: close_one(owner, primary)
        finally:
            result["close_wall_s"] = time.perf_counter() - before
            result["whole_trial_wall_s"] = time.perf_counter() - started
    require(owner.closed is True, "Timed target owner did not close")
    if wrapped: result["cache"] = owner.report()
    result["complete"] = True
    return result


def benchmark_case(case, backend, np, o3d, repeats, owners, destination):
    result = {"name": case["record"]["name"], "target": case["record"]["target"], "queries": case["record"]["queries"],
        "original_transform_and_query_snapshot_s": case["record"]["original_transform_and_query_snapshot_s"],
        "evaluate_registration": [], "raw_cached_native": [], "full_python_cache": [], "complete": False}
    destination.append(result)
    for _ in range(repeats):
        started = time.perf_counter()
        evaluated = o3d.pipelines.registration.evaluate_registration(case["source"], case["target"], case["record"]["radius"], case["matrix"])
        elapsed = time.perf_counter() - started
        result["evaluate_registration"].append({"wall_s": elapsed, **evaluation_record(case, evaluated, np)})
    for threads in helper.QUERY_THREADS:
        timed_lookup(case, backend, threads, repeats, owners, destination=result["raw_cached_native"])
        timed_lookup(case, backend, threads, repeats, owners, wrapped=True, destination=result["full_python_cache"])
    check_case(case, np); result["complete"] = True
    return result


def validate_parity(rows):
    require(type(rows) is list and rows, "Fresh parity rows required")
    names = [row.get("name") for row in rows]
    require(len(names) == len(set(names)), "Duplicate parity case")
    required_names = {spec["name"] for spec in synthetic_specs()} | {"synthetic-lattice"}
    require(required_names <= set(names), "Missing original boundary/tie/transform parity cases")
    for row in rows:
        require(row.get("complete") is True and type(row.get("parity")) is list
                and [value.get("threads") for value in row["parity"]] == list(helper.QUERY_THREADS),
                "Every declared thread count requires complete fresh parity")
        for value in row["parity"]:
            require(all(value.get(key) is True for key in ("raw_ids_exact", "raw_squared_bits_exact",
                "wrapper_ids_exact", "wrapper_squared_bits_exact", "owners_closed")), "Incomplete exact lookup parity")
        reference = row.get("original_scalar_reference", {})
        count = row["queries"]["shape"][0]
        require(reference.get("query_rows") == count and type(reference.get("hits")) is int
                and 0 <= reference["hits"] <= count, "Original scalar coverage receipt differs")


def finite_wall(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_timings(rows, repeats, expected_names):
    require(type(rows) is list and [row.get("name") for row in rows] == expected_names, "Timing cases/order differ")
    for row in rows:
        require(row.get("complete") is True and len(row.get("evaluate_registration", [])) == repeats,
                "Complete original evaluation repeats required")
        for value in row["evaluate_registration"]:
            require(finite_wall(value.get("wall_s")) and value.get("canonical_ids_exact") is True
                    and value.get("helper_rmse_equivalence_claim") is False, "Original evaluation scope/closure differs")
        for name in ("raw_cached_native", "full_python_cache"):
            values = row.get(name, [])
            require([value.get("threads") for value in values] == list(helper.QUERY_THREADS), "Timing thread coverage differs")
            for value in values:
                walls = value.get("warm_query_wall_s", [])
                require(value.get("complete") is True and len(walls) == repeats and all(finite_wall(wall) for wall in walls)
                        and all(finite_wall(value.get(key)) for key in ("cold_constructor_and_first_query_wall_s",
                            "close_wall_s", "whole_trial_wall_s")), "Incomplete/nonfinite lookup timing")
                require(value["whole_trial_wall_s"] + 1e-9 >= value["cold_constructor_and_first_query_wall_s"]
                        + sum(walls) + value["close_wall_s"], "Inclusive trial timer omits charged work")
                if name == "full_python_cache":
                    cache = value.get("cache", {})
                    require(cache.get("closed") is True and cache.get("failure") is None and cache.get("retained_targets") == 0
                            and cache.get("owned_reservation_bytes") == 0
                            and cache.get("statistics", {}).get("builds") == 1
                            and cache["statistics"].get("queries") == repeats + 1, "Timed bounded cache closure differs")


def cleanup_step(report, primary, name, action):
    """Attempt each independent close/receipt; keep the first actual failure."""
    try: return primary, action()
    except BaseException as error:
        report["cleanup_failures"].append({"scope": name, "type": type(error).__name__, "message": str(error)})
        if primary is not None and callable(getattr(primary, "add_note", None)):
            primary.add_note(name + ": " + repr(error))
        return primary if primary is not None else error, None


def lattice_spec():
    targets = [(x * .01, y * .01, z * .01) for x in range(32) for y in range(32) for z in range(16)]
    return {"name": "synthetic-lattice", "family": "synthetic_timing", "targets": targets,
            "queries": targets[::8], "radius": .03, "transform": [[1., 0., 0., .001], [0., 1., 0., -.002],
                [0., 0., 1., .003], [0., 0., 0., 1.]]}


def field_specs(path, np):
    with np.load(path, allow_pickle=False) as archive:
        original = archive["points"]
        require(original.dtype.str == "<f8" and original.ndim == 2 and original.shape[1] == 3, "Field geometry dtype/shape changed")
        indices = np.arange(0, len(original), max(1, math.ceil(len(original) / 50000)), dtype=np.int64)
        points = np.ascontiguousarray(original[indices])
    queries = points[::max(1, math.ceil(len(points) / 2048))]
    selection = {"original_vertices": len(original), "target_rows": len(points), "query_rows": len(queries),
                 "target_vertex_indices_sha256": bytes_hash(indices.tobytes()), "actual_finish_queries": False}
    for name, transform in (("identity", None), ("original-transform", [[1., 0., 0., .001], [0., 1., 0., -.002],
                            [0., 0., 1., .001], [0., 0., 0., 1.]])):
        yield {"name": "field-mesh-vertices-" + name, "family": "field_mesh_vertices_not_finish_queries",
               "targets": points, "queries": queries, "radius": .03, "transform": transform, "selection": selection}


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field-geometry", type=Path)
    parser.add_argument("--field-report", type=Path)
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--run-allocated", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    require(args.run_allocated, "Root-exclusive allocation requires --run-allocated")
    args.output = args.output.resolve(); require(not args.output.exists(), "Refuse an existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); report = {"kind": KIND, "status": "running", "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "failure": None, "cleanup_failures": [], "parity": [], "benchmarks": [], "whole_finish_authority": False,
        "scanner_speed_claim": False, "actual_finish_query_coverage": False,
        "scopes": {"evaluate_registration": "Original tree rebuild, source copy/transform, TBB search+metric reduction; same canonical IDs checked outside call timer",
            "raw_cached_native": "Immutable transformed bytes prepared separately; ABI target/query copies, fresh search, worker creation/join and output copies; no registration reducer",
            "full_python_cache": "Raw helper plus bounded cache constructor, scans/hashes, lookup, per-row output guards and metadata",
            "cold": "Constructor+first query; target close separately; whole trial includes repeated queries and exact-output checks",
            "query_preparation": "Original PointCloud copy/transform and byte snapshots recorded separately, excluded from warm lookup; not a complete evaluator substitute"}}
    owners, primary, o3d, np, saved_threads, binding, guard = [], None, None, None, None, None, None
    old_omp = os.environ.get("OMP_NUM_THREADS")
    def save(): args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        binding = preflight(args); report["binding"] = binding; save()
        os.environ["OMP_NUM_THREADS"] = "8"
        import numpy as np
        import open3d as o3d
        saved_threads = o3d.utility.get_max_threads(); o3d.utility.set_max_threads(20)
        require(o3d.utility.get_max_threads() == 20, "Original TBB20 policy not applied")
        before = time.perf_counter(); backend = helper.NativeLibrary(args.library, args.build_receipt)
        report["native_library_load_s"] = time.perf_counter() - before
        guard = LoadedOwners(o3d); report["runtime"] = runtime_binding(np, o3d, backend)
        specs = list(synthetic_specs()); timing_specs = [lattice_spec()]
        if args.field_geometry is not None: timing_specs.extend(field_specs(args.field_geometry, np))
        cases = []
        for index, spec in enumerate(specs + timing_specs):
            guard.check(); case = prepare_case(spec, np, o3d)
            if "selection" in spec: case["record"]["selection"] = spec["selection"]
            report["parity"].append(case["record"])
            audit_case(case, backend, np, o3d, owners); guard.check()
            if index >= len(specs): cases.append(case)
        validate_parity(report["parity"])
        report["fresh_parity_passed_before_timing"] = True; save()
        for case in cases:
            guard.check(); benchmark_case(case, backend, np, o3d, args.repeats, owners, report["benchmarks"]); guard.check(); save()
        validate_timings(report["benchmarks"], args.repeats, [case["record"]["name"] for case in cases])
        report["runtime_after"] = runtime_binding(np, o3d, backend)
        require(report["runtime_after"] == report["runtime"], "Native/runtime/floating/thread metadata changed")
        check_fixed(binding); report["binding_after"] = copy.deepcopy(binding)
    except BaseException as error:
        primary = error; report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for owner in reversed(owners):
            primary, _ = cleanup_step(report, primary, "target owner", lambda owner=owner: owner.close() if not owner.closed else None)
        for name, action in (("loaded owners", lambda: guard.check() if guard else None),
                             ("source/resource closure", lambda: check_fixed(binding) if binding else None),
                             ("threads", lambda: o3d.utility.set_max_threads(saved_threads) if saved_threads is not None else None)):
            primary, _ = cleanup_step(report, primary, name, action)
        def restore_environment():
            if old_omp is None: os.environ.pop("OMP_NUM_THREADS", None)
            else: os.environ["OMP_NUM_THREADS"] = old_omp
        primary, _ = cleanup_step(report, primary, "environment restore", restore_environment)
        for name, action in (("owners_closed", lambda: all(owner.closed for owner in owners)),
                             ("threads_restored", lambda: saved_threads is None or o3d.utility.get_max_threads() == saved_threads),
                             ("environment_restored", lambda: os.environ.get("OMP_NUM_THREADS") == old_omp)):
            primary, value = cleanup_step(report, primary, name, action); report[name] = value is True
        report["whole_experiment_wall_s"] = time.perf_counter() - started
        report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        passed = primary is None and not report["cleanup_failures"] and report["owners_closed"] and report["threads_restored"] and report["environment_restored"]
        report["status"] = "passed" if passed else "failed"
        if primary is not None and report["failure"] is None:
            report["failure"] = {"type": type(primary).__name__, "message": str(primary)}
        try: save()
        except BaseException as error:
            if primary is not None: raise primary from error
            raise
    if primary is not None: raise primary
    require(report["status"] == "passed", "Final cleanup/source/resource closure failed; private diagnostics saved")
    print(json.dumps({"status": report["status"], "parity_cases": len(report["parity"]), "timing_cases": len(report["benchmarks"]), "output": str(args.output)}))


if __name__ == "__main__": main()
