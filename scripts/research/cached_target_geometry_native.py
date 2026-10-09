"""Source-only, unqualified immutable-target KDTree research helper.

No numerical import, compilation, download, or DLL load occurs at import or CLI.
The API accepts original *already transformed* little-endian FP64 XYZ bytes.
It returns fresh source-ordered IDs/d2, not EvaluateRegistration's metric reducer.
NativeLibrary is an explicit future allocated native action, never auto-built.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import tarfile
import threading
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
KIND = "research-cached-target-original-query-index-v1"
ABI = 1
MAX_ROWS = 1_000_000
COORDINATE_LIMIT = 2 ** 20
NATIVE_MAX_BYTES = 512 * 1024 * 1024
QUERY_THREADS = (1, 2, 4, 8, 20)
EIGEN_COMMIT = "da7909592376c893dabbc4b6453a8ffe46b1eb8e"
EIGEN_ARCHIVE_SHA256 = "37f71e1d7c408e2cc29ef90dcda265e2de0ad5ed6249d7552f6c33897eafd674"
NANOFLANN_ARCHIVE_SHA256 = "89aecfef1a956ccba7e40f24561846d064f309bc547cc184af7f4426e42f8e65"
SOURCES = ("scripts/research/cached_target_geometry_native.py",
           "scripts/research/cached_target_geometry_native.cpp",
           "tests/test_cached_target_geometry_native.py")
PROVENANCE = {
    "open3d": {"version": "v0.20.0", "license": "MIT",
        "tree_source": "https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/geometry/KDTreeFlann.cpp",
        "tree_source_sha256": "13a726ef9e7a2174dc560e663a334555824e42ef4d9ab7b8969735aa9e5d3185",
        "layout_source": "https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/geometry/KDTreeFlann.h",
        "evaluation_source": "https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/pipelines/registration/Registration.cpp",
        "evaluation_source_sha256": "456dac720f38c0903d275c4d6ec7eb07e9cc73595dd55d9181cd4dc3f43b9f06"},
    "eigen": {"commit": EIGEN_COMMIT, "license": "MPL-2.0",
        "url": f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_COMMIT}/eigen-{EIGEN_COMMIT}.tar.gz",
        "archive_sha256": EIGEN_ARCHIVE_SHA256},
    "nanoflann": {"version": "v1.5.0", "license": "BSD",
        "url": "https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.0.tar.gz",
        "archive_sha256": NANOFLANN_ARCHIVE_SHA256},
    "tbb_reference_only": {"version": "v2021.12.0", "license": "Apache-2.0", "used_by_helper": False,
        "url": "https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.12.0.tar.gz",
        "archive_sha256": "c7bb7aa69c254d91b8f0041a71c5bcc3936acb64408a1719aec0b2b7639dd84f"},
}


class NativeGeometryError(RuntimeError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_contract():
    return {"kind": KIND, "abi": ABI,
        "artifacts_sha256": {name: file_hash(ROOT / name) for name in SOURCES},
        "tree": "KDTreeEigenMatrixAdaptor<const Eigen::MatrixXd,-1,metric_L2,false>; leaf15",
        "query": "original transformed FP64 bytes; knn1 then strict lower_bound(radius*radius)",
        "query_threads": list(QUERY_THREADS), "row_mapping": "disjoint original rows; no reduction",
        "domain": {"max_rows": MAX_ROWS, "coordinate_limit": COORDINATE_LIMIT, "radius": [2 ** -20, 1]},
        "native_reservation_cap_bytes": NATIVE_MAX_BYTES,
        "tie": "pinned nanoflann traversal; NANOFLANN_FIRST_MATCH forbidden",
        "transform_and_registration_reduction": "original caller; not implemented here",
        "qualified": False, "provenance": json.loads(json.dumps(PROVENANCE))}


def reservation(rows):
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError("Target count must be a positive bounded integer")
    return 320 * rows + 65536


def _blob_length(payload, rows, *, allow_empty=False):
    if (type(payload) is not bytes or type(rows) is not int or
            not (0 if allow_empty else 1) <= rows <= MAX_ROWS or len(payload) != 24 * rows):
        raise ValueError("Require exact immutable little-endian FP64 XYZ bytes/count")


def points_descriptor(payload, rows, *, allow_empty=False):
    _blob_length(payload, rows, allow_empty=allow_empty)
    for (value,) in struct.iter_unpack("<d", payload):
        if not math.isfinite(value) or abs(value) > COORDINATE_LIMIT:
            raise ValueError("Unsupported point coordinate domain")
    return {"dtype": "<f8", "shape": [rows, 3], "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def radius_value(value):
    if type(value) not in (float, int) or isinstance(value, bool):
        raise ValueError("Require finite bounded radius")
    value = float(value)
    if not math.isfinite(value) or not 2 ** -20 <= value <= 1:
        raise ValueError("Unsupported radius domain")
    return value


def validate_output(ids, squared, rows, target_rows, radius):
    """Malformed output is an error; misses never carry an accepted metric."""
    if type(ids) is not bytes or type(squared) is not bytes or len(ids) != 8 * rows or len(squared) != 8 * rows:
        raise NativeGeometryError("Malformed result byte lengths")
    value = radius_value(radius); radius2 = value * value
    for (index,), (d2,) in zip(struct.iter_unpack("<q", ids), struct.iter_unpack("<d", squared)):
        if index == -1:
            if d2 != math.inf:
                raise NativeGeometryError("Malformed miss metric")
        elif not (0 <= index < target_rows and math.isfinite(d2) and 0 <= d2 < radius2):
            raise NativeGeometryError("Malformed hit/radius result")
    return ids, squared


def _verify_archive(archive_path, root, expected_hash, selected):
    """Compare extracted headers/licenses against the exact archive; no extraction."""
    archive_path, root = Path(archive_path), Path(root).resolve()
    if file_hash(archive_path) != expected_hash:
        raise ValueError("Dependency archive SHA differs from Open3D's pin")
    digest, names = hashlib.sha256(), set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda value: value.name):
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 2:
                continue
            relative = PurePosixPath(*parts[1:])
            if not selected(relative):
                continue
            if relative.is_absolute() or ".." in relative.parts or member.size > 8 * 1024 * 1024:
                raise ValueError("Unsafe dependency member")
            path = root.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Dependency file escaped its root")
            expected = archive.extractfile(member).read()
            actual = path.read_bytes()
            if actual != expected:
                raise ValueError(f"Extracted dependency bytes changed: {relative}")
            digest.update(str(relative).encode("utf-8")); digest.update(actual)
            names.add(str(relative))
    return {"archive_sha256": expected_hash, "headers_sha256": digest.hexdigest(),
            "files": sorted(names)}


def build_recipe(*, eigen_root, eigen_archive, nanoflann_root, nanoflann_archive, compiler, output):
    """Verify source assets and emit a command; never execute it or write outputs."""
    eigen = _verify_archive(eigen_archive, eigen_root, EIGEN_ARCHIVE_SHA256,
        lambda path: path.parts[0] == "Eigen" or str(path) == "COPYING.MPL2")
    nano = _verify_archive(nanoflann_archive, nanoflann_root, NANOFLANN_ARCHIVE_SHA256,
        lambda path: str(path) in ("include/nanoflann.hpp", "COPYING"))
    if len(eigen["files"]) < 10 or "COPYING.MPL2" not in eigen["files"] or nano["files"] != ["COPYING", "include/nanoflann.hpp"]:
        raise ValueError("Incomplete dependency header/license inventory")
    compiler, output = Path(compiler).resolve(), Path(output).resolve()
    if not compiler.is_file() or compiler.name.lower() != "cl.exe" or output.suffix.lower() != ".dll":
        raise ValueError("Require explicit MSVC cl.exe and fresh .dll output")
    if output.exists() or output.with_suffix(".obj").exists() or output.with_suffix(".build.json").exists():
        raise ValueError("Build recipe must name fresh outputs")
    source = Path(__file__).with_suffix(".cpp").resolve()
    return {"kind": KIND, "stage": "verified recipe only; not built", "source_contract": source_contract(),
        "dependencies": {"eigen": eigen, "nanoflann": nano},
        "compiler": {"path": str(compiler), "sha256": file_hash(compiler)},
        "output": str(output), "command": [str(compiler), "/nologo", "/O2", "/EHsc", "/LD", "/std:c++17",
            "/fp:strict", "/DEIGEN_DONT_PARALLELIZE", f"/I{Path(eigen_root).resolve()}",
            f"/I{Path(nanoflann_root).resolve() / 'include'}", str(source),
            f"/Fo:{output.with_suffix('.obj')}", "/link", f"/OUT:{output}"],
        "qualification": "none; root-exclusive compilation and fresh original-API parity required"}


class NativeLibrary:
    """Explicit future DLL loading, after a recorded build receipt is supplied."""
    def __init__(self, library, manifest):
        if os.name != "nt" or sys.byteorder != "little" or struct.calcsize("P") != 8:
            raise ValueError("Declared native route is Windows x64 little-endian only")
        library, manifest = Path(library).resolve(), Path(manifest).resolve()
        receipt = json.loads(manifest.read_text(encoding="utf-8"))
        if (receipt.get("kind") != KIND or receipt.get("stage") != "built" or
                type(receipt.get("actual_exit_code")) is not int or receipt.get("actual_exit_code") != 0 or
                receipt.get("source_contract") != source_contract() or receipt.get("library_sha256") != file_hash(library)):
            raise ValueError("Missing/current build receipt or DLL SHA")
        command = receipt.get("command", [])
        if (not isinstance(command, list) or "/fp:strict" not in command or "/std:c++17" not in command or
                any("fast" in str(item).lower() or "nanoflann_first_match" in str(item).lower() for item in command)):
            raise ValueError("Undeclared floating compiler policy")
        dependencies = receipt.get("dependencies", {})
        for name, pinned, license_name in (("eigen", EIGEN_ARCHIVE_SHA256, "COPYING.MPL2"),
                                          ("nanoflann", NANOFLANN_ARCHIVE_SHA256, "COPYING")):
            node = dependencies.get(name, {})
            if node.get("archive_sha256") != pinned or license_name not in node.get("files", []):
                raise ValueError("Dependency/license receipt differs")
        # This line loads native code only when the explicitly invoked constructor runs.
        import ctypes
        self.ct = ctypes
        self.library_path, self.library_sha256 = library, file_hash(library)
        self.dll = ctypes.CDLL(str(library))
        u64, doubles = ctypes.c_uint64, ctypes.POINTER(ctypes.c_double)
        self.dll.ctgn_abi_version.argtypes = []; self.dll.ctgn_abi_version.restype = ctypes.c_uint32
        self.dll.ctgn_fp_environment.argtypes = []; self.dll.ctgn_fp_environment.restype = u64
        self.dll.ctgn_last_error.argtypes = []; self.dll.ctgn_last_error.restype = ctypes.c_char_p
        self.dll.ctgn_create.argtypes = [doubles, u64, u64, u64, ctypes.POINTER(u64), ctypes.POINTER(u64)]
        self.dll.ctgn_query.argtypes = [u64, doubles, u64, ctypes.c_double, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int64), doubles]
        self.dll.ctgn_destroy.argtypes = [u64]
        for name in ("ctgn_create", "ctgn_query", "ctgn_destroy"):
            getattr(self.dll, name).restype = ctypes.c_int
        if self.dll.ctgn_abi_version() != ABI:
            raise NativeGeometryError("Native ABI version differs")
        self.fp = int(self.dll.ctgn_fp_environment())

    def _check(self, status):
        if status:
            raise NativeGeometryError(f"Native status {status}: {self.dll.ctgn_last_error().decode('utf-8', 'replace')}")

    def create(self, points, rows, limit):
        _blob_length(points, rows)
        if type(limit) is not int or not reservation(rows) <= limit <= NATIVE_MAX_BYTES:
            raise ValueError("Unsupported native reservation before ABI allocation")
        ct = self.ct; values = (ct.c_double * (3 * rows)).from_buffer_copy(points)
        handle, owned = ct.c_uint64(), ct.c_uint64()
        self._check(self.dll.ctgn_create(values, rows, limit, self.fp, ct.byref(handle), ct.byref(owned)))
        return int(handle.value), int(owned.value)

    def query(self, handle, points, rows, radius, threads):
        _blob_length(points, rows, allow_empty=True)
        radius = radius_value(radius)
        if (type(handle) is not int or not 1 <= handle < 2 ** 64 or
                type(threads) is not int or threads not in QUERY_THREADS):
            raise ValueError("Unsupported native handle/thread count before ABI allocation")
        ct = self.ct; values = (ct.c_double * (3 * rows)).from_buffer_copy(points)
        ids, squared = (ct.c_int64 * rows)(), (ct.c_double * rows)()
        self._check(self.dll.ctgn_query(handle, values, rows, radius, threads, ids, squared))
        return bytes(ids), bytes(squared)

    def destroy(self, handle):
        if type(handle) is not int or not 1 <= handle < 2 ** 64:
            raise ValueError("Malformed native handle")
        self._check(self.dll.ctgn_destroy(handle))

    def closure(self):
        if file_hash(self.library_path) != self.library_sha256:
            raise NativeGeometryError("Loaded library bytes changed")
        return {"library_sha256": self.library_sha256, "fp_environment": self.fp}


class TargetEntry(NamedTuple):
    payload: bytes
    rows: int
    handle: int
    native_reservation: int


class CachedTargets:
    """Content-keyed bounded LRU, immutable target bytes, synchronous fresh queries.

    Backend injection is for stdlib contracts. It is not a proof/timing authority.
    Every public call scans original supplied target/query bytes; those costs must
    be charged. Source transforms, correspondence reductions and gates stay out.
    """
    def __init__(self, backend, *, max_targets=4, max_owned_bytes=256 * 1024 * 1024, query_threads=1):
        if (type(max_targets) is not int or not 1 <= max_targets <= 16 or
                type(max_owned_bytes) is not int or not 65536 < max_owned_bytes <= NATIVE_MAX_BYTES):
            raise ValueError("Unsupported cache cap")
        if type(query_threads) is not int or query_threads not in QUERY_THREADS:
            raise ValueError("Unsupported query thread count")
        self.backend, self._backend = backend, backend
        self.configuration = (max_targets, max_owned_bytes, query_threads)
        self._configuration = self.configuration
        self.entries = OrderedDict(); self.owned_bytes = 0
        self.failure = None; self.closed = False; self.lock = threading.RLock()
        self.statistics = {"builds": 0, "hits": 0, "evictions": 0, "queries": 0, "query_rows": 0}

    def _healthy(self):
        if self.closed or self.failure:
            raise NativeGeometryError("Cache is closed or faulted") from self.failure
        if self.backend is not self._backend or self.configuration != self._configuration:
            raise NativeGeometryError("Cache effective owner/configuration changed")

    def _drop(self, key):
        entry = self.entries[key]
        self._backend.destroy(entry.handle)
        self.owned_bytes -= entry.native_reservation + len(entry.payload)
        del self.entries[key]

    def query(self, target_points, target_rows, transformed_queries, query_rows, radius):
        with self.lock:
            try:
                self._healthy()
                target = points_descriptor(target_points, target_rows)
                queries = points_descriptor(transformed_queries, query_rows, allow_empty=True)
                radius = radius_value(radius)
                key = (target_rows, target["sha256"])
                entry = self.entries.get(key)
                if entry is not None and entry.payload != target_points:
                    raise NativeGeometryError("Target hash collision")
                if entry is None:
                    native = reservation(target_rows); charged = native + len(target_points)
                    if charged > self.configuration[1]:
                        raise NativeGeometryError("Target exceeds cache reservation before allocation")
                    while self.entries and (len(self.entries) >= self.configuration[0] or self.owned_bytes + charged > self.configuration[1]):
                        self._drop(next(iter(self.entries))); self.statistics["evictions"] += 1
                    handle, owned = self._backend.create(target_points, target_rows, native)
                    # Keep any allocated owner even if malformed metadata must fault.
                    entry = TargetEntry(target_points, target_rows, handle, native)
                    self.entries[key] = entry; self.owned_bytes += charged
                    if type(handle) is not int or handle <= 0 or owned != native:
                        raise NativeGeometryError("Malformed native construction receipt")
                    self.statistics["builds"] += 1
                else:
                    self.statistics["hits"] += 1
                self.entries.move_to_end(key)
                ids, d2 = self._backend.query(entry.handle, transformed_queries, query_rows, radius, self.configuration[2])
                validate_output(ids, d2, query_rows, target_rows, radius)
                self._healthy()
                self.statistics["queries"] += 1; self.statistics["query_rows"] += query_rows
                return {"ids": ids, "squared": d2, "target": target, "queries": queries,
                        "radius": radius, "qualified": False}
            except BaseException as error:
                if self.failure is None: self.failure = error
                raise

    def close(self, primary=None):
        """Independent destruction; failed owners remain retained for diagnosis/retry."""
        errors = []
        with self.lock:
            for key in list(self.entries):
                try: self._drop(key)
                except BaseException as error: errors.append(error)
            self.closed = not self.entries
            if errors:
                if self.failure is None: self.failure = errors[0]
                if primary is not None:
                    if callable(getattr(primary, "add_note", None)):
                        for error in errors: primary.add_note(f"Target cleanup: {error}")
                    raise primary from errors[0]
                raise errors[0]

    def report(self):
        return {"kind": KIND, "qualified": False, "configuration": {
            "max_targets": self._configuration[0], "max_owned_bytes": self._configuration[1],
            "query_threads": self._configuration[2]},
            "statistics": dict(self.statistics), "retained_targets": len(self.entries),
            "owned_reservation_bytes": self.owned_bytes, "closed": self.closed,
            "failure": None if self.failure is None else f"{type(self.failure).__name__}: {self.failure}",
            "memory_scope": "owned Python bytes + conservative native array/index/pool reservation; excludes CRT/stack/RSS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eigen-root", required=True)
    parser.add_argument("--eigen-archive", required=True)
    parser.add_argument("--nanoflann-root", required=True)
    parser.add_argument("--nanoflann-archive", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", required=True, help="Fresh future DLL path; this CLI does not build it")
    args = vars(parser.parse_args())
    print(json.dumps(build_recipe(**args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
