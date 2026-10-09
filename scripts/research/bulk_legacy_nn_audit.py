"""Prepare/crosscheck bulk original Open3D nearest-ID CPU shadows.

This module has no native imports at import time. Its CLI requires an allocated
hardware slot. Existing scalar auditors are unchanged. A passed bounded API
crosscheck is not a shader/ICP or whole-field trajectory proof; before replacing
field shadows, separately compare scalar and bulk on the complete old resident
trajectory and retain its own current-source proof.

Primary Open3D v0.20.0 references:
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/pipelines/registration/Registration.cpp
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/geometry/KDTreeFlann.cpp
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/pybind/geometry/kdtreeflann.cpp
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile


MAX_POINTS = 1_000_000
MAX_COORDINATE = 2.0 ** 20
MIN_RADIUS = 2.0 ** -20
MAX_RADIUS = 1.0


class UnsupportedBulkDomain(ValueError):
    """A caller must retain the complete scalar audit for this input."""


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_binding(np, array):
    array = np.asarray(array)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.nbytes,
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def _check_matrix(np, array, name):
    if (not isinstance(array, np.ndarray) or array.dtype != np.float64
            or array.ndim != 2 or array.shape[1] != 3 or len(array) > MAX_POINTS):
        raise UnsupportedBulkDomain(f"{name} requires original float64(N,3), at most {MAX_POINTS} rows")
    if not np.isfinite(array).all() or np.any(np.abs(array) > MAX_COORDINATE):
        raise UnsupportedBulkDomain(f"{name} must be finite and within absolute coordinate {MAX_COORDINATE}")


@dataclasses.dataclass(frozen=True)
class BulkResult:
    ids: object
    diagnostic_squared: object
    statistics: dict


def bulk_legacy_ids(np, o3d, queries, target, radius, *, expected_target_digest=None,
                    chunk_rows=65536):
    """Return original legacy IDs, mapped by query index, with bounded copies.

Only native correspondence IDs establish hits/misses. The FP64 diagnostic
metric never reclassifies a native result. Unsupported input raises rather than
silently omitting audit rows. Caller lifetime must prevent concurrent target or
query mutation. A mutable target's original cached-tree digest is mandatory in
the eventual integration; the standalone crosscheck supplies that digest too.

This function does not itself authorize a new pipeline. Preserve reason&7
scalar ambiguity/fallback handling and validate a distinct complete dual-shadow
trajectory proof before using bulk-only reason&24 field shadows.
"""
    began = time.perf_counter()
    _check_matrix(np, queries, "Queries")
    points = np.asarray(target.points)
    _check_matrix(np, points, "Target")
    if not len(points):
        raise UnsupportedBulkDomain("Empty targets require original scalar error handling")
    if (isinstance(radius, bool) or not isinstance(radius, (float, int))
            or not math.isfinite(radius) or not MIN_RADIUS <= radius <= MAX_RADIUS):
        raise UnsupportedBulkDomain("Bulk audit radius domain is 2^-20 .. 1 metres")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or not 1 <= chunk_rows <= 65536:
        raise ValueError("Require a bounded positive chunk_rows <= 65536")
    target_before = array_binding(np, points)
    if expected_target_digest is not None:
        expected = (expected_target_digest.hex() if isinstance(expected_target_digest, bytes)
                    else expected_target_digest)
        if target_before["sha256"] != expected:
            raise RuntimeError("Original cached CPU tree target digest disagrees with bulk target")
    query_before = array_binding(np, queries)
    ids = np.full(len(queries), -1, np.int32)
    diagnostic = np.full(len(queries), np.inf, np.float64)
    evaluate_s = transport_s = normalization_s = 0.0
    calls = peak_query_copy_bytes = 0
    identity = np.eye(4, dtype=np.float64)
    for start in range(0, len(queries), chunk_rows):
        stop = min(start + chunk_rows, len(queries))
        block = np.ascontiguousarray(queries[start:stop])
        copied_at = time.perf_counter()
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(block)
        copied = np.asarray(source.points)
        if (copied.shape != block.shape or copied.dtype != np.float64
                or not np.array_equal(copied.view(np.uint64), block.view(np.uint64))):
            raise RuntimeError("Bulk PointCloud upload altered original coordinate bytes")
        transport_s += time.perf_counter() - copied_at
        evaluated_at = time.perf_counter()
        result = o3d.pipelines.registration.evaluate_registration(source, target, float(radius), identity)
        evaluate_s += time.perf_counter() - evaluated_at
        calls += 1
        normalized_at = time.perf_counter()
        transform = np.asarray(result.transformation)
        if (transform.shape != (4, 4) or transform.dtype != np.float64
                or not np.array_equal(transform.view(np.uint64), identity.view(np.uint64))):
            raise RuntimeError("Bulk evaluator returned a different identity transform")
        correspondence = np.asarray(result.correspondence_set)
        if (correspondence.ndim != 2 or correspondence.shape[1] != 2
                or correspondence.dtype.kind not in "iu" or len(correspondence) > len(block)):
            raise RuntimeError("Malformed native bulk correspondence matrix")
        if len(correspondence):
            source_rows, target_rows = correspondence[:, 0], correspondence[:, 1]
            if (np.any(source_rows < 0) or np.any(source_rows >= len(block))
                    or np.any(target_rows < 0) or np.any(target_rows >= len(points))
                    or len(np.unique(source_rows)) != len(source_rows)):
                raise RuntimeError("Duplicate/out-of-range native bulk correspondence rows")
            mapped = start + source_rows.astype(np.int64)
            ids[mapped] = target_rows.astype(np.int32)
            # Separate subtraction/multiply/adds: no radius decision uses this.
            delta = block[source_rows] - points[target_rows]
            squares = delta * delta
            diagnostic[mapped] = (squares[:, 0] + squares[:, 1]) + squares[:, 2]
        peak_query_copy_bytes = max(peak_query_copy_bytes, len(block) * 192)
        normalization_s += time.perf_counter() - normalized_at
    # Re-read the public vector: checking only the initial NumPy view would
    # miss replacement of target.points with a different vector.
    if (array_binding(np, queries) != query_before
            or array_binding(np, np.asarray(target.points)) != target_before):
        raise RuntimeError("Original bulk-audit query/target arrays changed during evaluation")
    ids.flags.writeable = diagnostic.flags.writeable = False
    return BulkResult(ids, diagnostic, {
        "query_rows": len(queries), "target_rows": len(points), "evaluate_calls": calls,
        "evaluate_wall_s": evaluate_s, "pointcloud_copy_check_s": transport_s,
        "normalize_metric_s": normalization_s, "all_in_wall_s": time.perf_counter() - began,
        "helper_chunk_temporary_estimate_bytes": peak_query_copy_bytes,
        "whole_array_validation_hash_estimate_bytes": 32 * (len(queries) + len(points)),
        "memory_scope": "Estimates cover visible query copies, correspondence normalization, metric and full-array validation/hash buffers. Native KDTree/reduction allocator internals, resident input/output arrays and allocator pools are not a hard measured memory cap.",
        "native_bulk_correspondence_order": "Mapped by source index; original C++ order is not assumed.",
        "metric_scope": "Diagnostic original-double subtraction, multiply and left-associated sum; native IDs alone determine strict-radius acceptance.",
    })


def scalar_legacy_ids(np, tree, queries, radius):
    ids = np.full(len(queries), -1, np.int32)
    squared = np.full(len(queries), np.inf, np.float64)
    started = time.perf_counter()
    for i, query in enumerate(queries):
        count, found, distances = tree.search_hybrid_vector_3d(query, radius, 1)
        if count not in (0, 1) or len(found) != count or len(distances) != count:
            raise RuntimeError("Malformed original scalar SearchHybrid result")
        if count:
            ids[i] = found[0]
            squared[i] = distances[0]
    return ids, squared, time.perf_counter() - started


def runtime_binding(np, o3d):
    package = Path(o3d.__file__).resolve().parent
    selected_modules = {
        o3d.geometry.KDTreeFlann.__module__.split(".geometry")[0],
        o3d.pipelines.registration.evaluate_registration.__module__.split(".pipelines")[0],
    }
    binaries = {}
    for module_name in selected_modules:
        module = sys.modules.get(module_name)
        path = Path(module.__file__).resolve() if module is not None and getattr(module, "__file__", None) else None
        if path is None or not path.is_file():
            raise RuntimeError(f"Cannot bind selected native Open3D module: {module_name}")
        binaries[str(path)] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    for name in ("Open3D.dll", "libOpen3D.so", "libOpen3D.dylib", "tbb12.dll"):
        for directory in (package, package / "cpu", package / "cuda"):
            path = directory / name
            if path.is_file():
                binaries[str(path.resolve())] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    numpy_binaries = {}
    for name, module in list(sys.modules.items()):
        if name.startswith("numpy.") and name.endswith("._multiarray_umath"):
            filename = getattr(module, "__file__", None)
            if filename is not None:
                path = Path(filename).resolve()
                numpy_binaries[str(path)] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    if not numpy_binaries:
        raise RuntimeError("Cannot bind the actual loaded NumPy ufunc native module")
    return {"python": sys.version, "python_executable": sys.executable,
            "numpy": np.__version__, "numpy_file": str(Path(np.__file__).resolve()),
            "open3d": o3d.__version__, "selected_modules": sorted(selected_modules),
            "native_binaries": binaries,
            "numpy_native_binaries": numpy_binaries,
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TCM_ENABLE")},
            "thread_scope": "Existing Open3D TBB configuration; helper changes no global thread policy."}


def synthetic_cases(np):
    # Exhaust all target-order permutations of six exact tied axes and four
    # duplicated/signed-zero points. Match native policy, never lowest-ID policy.
    axes = np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                       [0, 0, 1], [0, 0, -1]], np.float64) * .25
    for i, order in enumerate(itertools.permutations(range(6))):
        yield f"axis-tie-permutation-{i}", "ties", axes[list(order)], np.asarray(
            [[0, 0, 0], [np.nextafter(0., 1.), 0, 0], [np.nextafter(0., -1.), 0, 0]], np.float64), .5
    duplicates = np.asarray([[0., -0., 0.], [-0., 0., -0.], [0., 0., 0.], [.25, 0., 0.]], np.float64)
    for i, order in enumerate(itertools.permutations(range(4))):
        yield f"duplicate-permutation-{i}", "duplicates_signed_zero", duplicates[list(order)], np.asarray(
            [[0., -0., 0.], [-0., 0., -0.], [.125, 0., 0.], [.25, 0., 0.]], np.float64), .5
    for radius in (MIN_RADIUS, .03, .06, .12, .5, 1.):
        for anchor in (0., 1., -1., 2. ** 18, -(2. ** 18)):
            center = np.asarray([anchor, 0., 0.], np.float64)
            points = np.asarray([center, center + [4 * radius, 0., 0.], center + [0., 4 * radius, 0.]], np.float64)
            boundary = anchor + radius
            queries = np.asarray([
                center, [np.nextafter(boundary, -np.inf), 0., 0.], [boundary, 0., 0.],
                [np.nextafter(boundary, np.inf), 0., 0.],
                [anchor, radius, 0.], [anchor, np.nextafter(radius, 0.), 0.],
                [anchor, np.nextafter(radius, np.inf), 0.],
                center + [radius / math.sqrt(2), radius / math.sqrt(2), 0.],
                center + [3 * radius, 3 * radius, 3 * radius],
            ], np.float64)
            for i, order in enumerate(itertools.permutations(range(3))):
                yield f"radius-{radius}-anchor-{anchor}-permutation-{i}", "strict_radius_boundary", points[list(order)], queries, radius
    rng = np.random.default_rng(613)
    for i in range(8):
        points = rng.uniform(-1, 1, (128, 3)).astype(np.float64)
        queries = np.r_[points[:64], rng.uniform(-1, 1, (192, 3))].astype(np.float64)
        yield f"random-{i}", "finite_random", points, queries, (.03, .06, .12, .5)[i % 4]


def checked_case(np, o3d, name, category, points, queries, radius, chunk_sizes, gold=None):
    before = {"points": array_binding(np, points), "queries": array_binding(np, queries)}
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(points)
    if array_binding(np, np.asarray(target.points)) != before["points"]:
        raise RuntimeError("Original target point upload changed coordinate bytes")
    tree = o3d.geometry.KDTreeFlann(target)
    expected, native_squared, scalar_s = scalar_legacy_ids(np, tree, queries, radius)
    if gold is not None and not np.array_equal(expected, gold):
        raise RuntimeError(f"Old retained CPU gold changed at {name}")
    rows = []
    for chunk_rows in chunk_sizes:
        actual = bulk_legacy_ids(np, o3d, queries, target, radius,
                                 expected_target_digest=before["points"]["sha256"], chunk_rows=chunk_rows)
        id_errors = int(np.count_nonzero(actual.ids != expected))
        metric_errors = int(np.count_nonzero(actual.diagnostic_squared.view(np.uint64) != native_squared.view(np.uint64)))
        row = {"chunk_rows": chunk_rows, "id_errors": id_errors,
               "diagnostic_metric_bit_errors": metric_errors, **actual.statistics}
        rows.append(row)
        if id_errors or metric_errors:
            raise RuntimeError(f"Scalar/bulk changed {id_errors} IDs or {metric_errors} metric bits at {name}; partial={json.dumps(row)}")
    if before != {"points": array_binding(np, points), "queries": array_binding(np, queries)}:
        raise RuntimeError(f"Original crosscheck data mutated: {name}")
    return {"name": name, "category": category, "radius_m": radius,
            "query_rows": len(queries), "target_rows": len(points), "bindings": before,
            "scalar_wall_s": scalar_s, "hits": int(np.count_nonzero(expected >= 0)),
            "misses": int(np.count_nonzero(expected < 0)), "bulk_runs": rows,
            "all_original_rows_checked": True, "arrays_unchanged": True}


def contract_checks(np):
    """Synthetic API faults, separate from actual native numerical evidence."""
    passed = []
    for mode in ("unordered", "duplicate", "source_oob", "target_oob", "float_ids",
                 "nonidentity", "copy_change", "query_mutation", "target_mutation",
                 "target_replacement", "wrong_digest"):
        queries = np.asarray([[0., -0., 0.], [.01, 0., 0.], [.02, 0., 0.]], np.float64)
        target = SimpleNamespace(points=np.asarray([[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.]], np.float64))

        def copy(value):
            output = value.copy()
            if mode == "copy_change":
                output[0, 1] = 0.
            return output

        def evaluate(source, cloud, radius, identity):
            correspondence = np.asarray([[2, 1], [0, 2]], np.int32)
            if mode == "duplicate":
                correspondence = np.asarray([[0, 1], [0, 2]], np.int32)
            if mode == "source_oob":
                correspondence[0, 0] = len(queries)
            if mode == "target_oob":
                correspondence[0, 1] = len(target.points)
            if mode == "float_ids":
                correspondence = correspondence.astype(np.float64)
            transform = identity.copy()
            if mode == "nonidentity":
                transform[0, 3] = 1.
            if mode == "query_mutation":
                queries[1, 0] += .01
            if mode == "target_mutation":
                target.points[1, 0] += .01
            if mode == "target_replacement":
                target.points = target.points.copy()
                target.points[1, 0] += .01
            return SimpleNamespace(transformation=transform, correspondence_set=correspondence)

        fake = SimpleNamespace(
            geometry=SimpleNamespace(PointCloud=lambda: SimpleNamespace()),
            utility=SimpleNamespace(Vector3dVector=copy),
            pipelines=SimpleNamespace(registration=SimpleNamespace(evaluate_registration=evaluate)))
        expected = "0" * 64 if mode == "wrong_digest" else None
        try:
            result = bulk_legacy_ids(np, fake, queries, target, .1, expected_target_digest=expected)
        except RuntimeError:
            if mode == "unordered":
                raise
            passed.append(mode + "_hard_failure")
        else:
            if mode != "unordered" or not np.array_equal(result.ids, [2, -1, 1]):
                raise RuntimeError(f"Bulk correspondence contract fault was masked: {mode}")
            if result.ids.flags.writeable or result.diagnostic_squared.flags.writeable:
                raise RuntimeError("Bulk outputs must be immutable")
            passed.append("unordered_mapping_and_immutable_results")
    return {"passed": True, "checks": passed,
            "scope": "Artificial normalization/lifetime failures; actual native scalar/bulk evidence follows separately."}


def load_trace_metadata(path):
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture_path = Path(manifest["capture_report"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "passed" or not manifest.get("all_final_capture_guards_passed")
            or not capture.get("all_nine_original_gates_passed")
            or manifest.get("capture_report_sha256") != file_hash(capture_path)
            or manifest.get("trace_sha256") != file_hash(path)):
        raise RuntimeError("Historical real trace is not bound to its closed original capture")
    for key, capture_key in (("batches", "trace_batches"), ("retained_array_bytes", "retained_array_bytes"),
                             ("producer_sha256", "script_sha256"), ("source_sha256", "source_sha256"),
                             ("fixture_binding", "fixture_binding"), ("input_binding", "input_binding"),
                             ("proof_report_hashes", "proof_report_hashes")):
        if manifest.get(key) != capture.get(capture_key):
            raise RuntimeError(f"Historical trace/capture binding mismatch: {key}")
    if not 1 <= len(manifest["batches"]) <= 12 or not 0 < manifest["retained_array_bytes"] <= 128 * 1024**2:
        raise RuntimeError("Trace exceeds its historical retained-array/case bound")
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist()) > 129 * 1024**2:
            raise RuntimeError("Trace decompressed extent exceeds retained-array inspection bound")
    return manifest, {"trace_path": str(path.resolve()), "trace_sha256": file_hash(path),
                      "manifest_path": str(manifest_path.resolve()), "manifest_sha256": file_hash(manifest_path),
                      "capture_path": str(capture_path.resolve()), "capture_sha256": file_hash(capture_path)}


def run(args, report):
    manifest, trace_before = load_trace_metadata(args.trace)
    report["trace_binding"] = trace_before
    reference_path = args.references
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    for source in references["sources"]:
        if source["sha256"] != file_hash(source["snapshot"]):
            raise RuntimeError("Primary Open3D source reference snapshot changed")
    report["source_reference"] = {"manifest": str(reference_path.resolve()),
                                  "sha256": file_hash(reference_path), "sources": references["sources"]}
    # Native imports happen only after the CLI's explicit allocation flag.
    import numpy as np
    import open3d as o3d

    before = runtime_binding(np, o3d)
    if not o3d.__version__.startswith("0.20.0"):
        raise RuntimeError("This source-referenced crosscheck requires installed Open3D0.20.0")
    report["runtime"] = before
    report["normalization_contracts"] = contract_checks(np)
    cases = report["synthetic_cases"] = []
    for name, category, points, queries, radius in synthetic_cases(np):
        cases.append(checked_case(np, o3d, name, category, np.ascontiguousarray(points),
                                  np.ascontiguousarray(queries), radius, [65536]))
    report["synthetic_summary"] = {
        "case_counts": dict(collections.Counter(x["category"] for x in cases)),
        "query_rows": sum(x["query_rows"] for x in cases),
        "hits": sum(x["hits"] for x in cases), "misses": sum(x["misses"] for x in cases),
        "all_ids_and_metric_bits_exact": True,
    }
    real = report["real_cases"] = []
    retained = 0
    with np.load(args.trace, allow_pickle=False) as data:
        for batch in manifest["batches"]:
            index = batch["index"]
            arrays = {key: data[f"batch{index}_{key}"] for key in batch["arrays"]}
            for key, array in arrays.items():
                if array_binding(np, array) != batch["arrays"][key]:
                    raise RuntimeError(f"Historical retained real array changed: {index}/{key}")
            retained += sum(x.nbytes for x in arrays.values())
            if batch["query_count"] > 2048 or arrays["queries"].shape != (batch["query_count"], 3):
                raise RuntimeError("Historical query extent changed")
            row = checked_case(np, o3d, f"real-batch-{index}", batch["scope"],
                               arrays["target_points"], arrays["queries"], batch["radius_m"],
                               [257, 4096, 65536], gold=arrays["gold_ids"])
            row["historical_branch"] = {key: batch.get(key) for key in
                                         ("pair_position", "proposal_index", "scope", "branch_path")}
            real.append(row)
    if retained != manifest["retained_array_bytes"]:
        raise RuntimeError("Historical retained array accounting changed")
    report["real_summary"] = {
        "cases": len(real), "query_rows": sum(x["query_rows"] for x in real),
        "hits": sum(x["hits"] for x in real), "misses": sum(x["misses"] for x in real),
        "all_retained_rows_scalar_checked": True, "stored_original_gold_equal": True,
        "all_ids_and_metric_bits_exact": True,
        "scalar_query_wall_s": sum(x["scalar_wall_s"] for x in real),
        "bulk_all_in_wall_s_by_chunk": {str(size): sum(
            row["all_in_wall_s"] for case in real for row in case["bulk_runs"] if row["chunk_rows"] == size)
            for size in (257, 4096, 65536)},
    }
    report["runtime_after"] = runtime_binding(np, o3d)
    _, report["trace_binding_after"] = load_trace_metadata(args.trace)
    if report["runtime_after"] != before or report["trace_binding_after"] != trace_before:
        raise RuntimeError("Runtime or historical trace changed during fresh crosscheck")
    if report["source_reference"]["sha256"] != file_hash(reference_path) or any(
            source["sha256"] != file_hash(source["snapshot"]) for source in references["sources"]):
        raise RuntimeError("Official source references changed during fresh crosscheck")
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=Path(__file__).resolve().parents[2] /
                        "benchmark-output/field-cuda-study/bulk-nn-audit/reference/source-reference.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_allocated:
        parser.error("Require an explicitly allocated hardware slot before native imports")
    if args.output.exists() or args.output.resolve() in (args.trace.resolve(), args.trace.with_suffix(".json").resolve()):
        parser.error("Require a fresh output that cannot overwrite trace inputs")
    report = {"kind": "scalar-vs-original-legacy-bulk-nearest-api-crosscheck", "status": "running",
              "script_sha256": file_hash(__file__), "native_imports_requested": True,
              "domain": {"original_dtype": "float64", "point_and_query_row_limit": MAX_POINTS,
                         "absolute_coordinate_limit": MAX_COORDINATE, "radius_m": [MIN_RADIUS, MAX_RADIUS]},
              "scope": "Exhaustive generated synthetic permutations and every retained old real trace row. Not every old resident iteration or any new field trajectory.",
              "new_field_bulk_only_shadow_authorized": False,
              "required_next_proof": "Distinct current-source complete old resident trajectory with dual scalar/bulk IDs, then complete new-field hit/miss shadows and original authority/quality gates.",
              "gpu_queries_run": False}
    start = time.perf_counter()
    primary = None
    try:
        run(args, report)
    except BaseException as exc:
        primary = exc
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc),
                                              "traceback": traceback.format_exc()})
    finally:
        report["elapsed_s_including_native_imports_hashes_and_checks"] = time.perf_counter() - start
        report["script_sha256_after"] = file_hash(__file__)
        if report["script_sha256_after"] != report["script_sha256"]:
            report["status"] = "failed"
            if primary is None:
                primary = RuntimeError("Bulk crosscheck script changed during run")
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        except BaseException as write_error:
            if primary is not None:
                primary.add_note(f"Final report write also failed: {write_error}")
                raise primary from write_error
            raise
    if primary is not None:
        raise primary
    print(json.dumps({"status": report["status"], "synthetic": report["synthetic_summary"],
                      "real": report["real_summary"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
