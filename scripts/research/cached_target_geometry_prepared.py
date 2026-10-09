"""Unqualified prepared immutable target lease; no native load at import.

The pinned native ABI still validates all query coordinates and each NN result.
Only redundant Python target scans/hashes and per-row output scans are omitted.
Original source transformation and registration metric reduction remain outside.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.research import cached_target_geometry_native as native

KIND = "cached-target-geometry-prepared-immutable-lease-v1"
FILES = ("scripts/research/cached_target_geometry_prepared.py",
         "scripts/research/benchmark_cached_target_geometry_prepared.py",
         "tests/test_cached_target_geometry_prepared.py")


def source_contract():
    return {"kind": KIND, "artifacts_sha256": {name: native.file_hash(ROOT / name) for name in FILES},
            "native_source": native.source_contract(),
            "target_validation": "one Python finite/domain scan+hash, retained immutable original bytes and native owned copy",
            "warm_query_validation": "immutable FP64 byte count/radius/thread/owner; native finite/domain+strict NN result guards",
            "python_output_scan": False, "metric_reducer": False, "qualified": False}


class Entry(NamedTuple):
    points: bytes
    rows: int
    descriptor_json: str
    handle: int
    native_bytes: int
    charged_bytes: int


def callable_state(method):
    function = getattr(method, "__func__", method)
    return (getattr(method, "__self__", None), function, id(getattr(function, "__code__", None)),
            repr(getattr(function, "__defaults__", None)), repr(getattr(function, "__kwdefaults__", None)))


class PreparedTarget:
    """One synchronous owned target; no approximate result or pose caching.

    Backend injection supports stdlib tests only. Actual use must load the pinned
    NativeLibrary and perform fresh scalar parity in the allocated driver.
    Failed native owners stay retained; close-only retry is permitted.
    """
    def __init__(self, backend, target_points, target_rows, *, query_threads=1,
                 max_owned_bytes=256 * 1024 * 1024):
        self.backend = self._backend = backend
        self.entry = self._entry = None
        self.failure = None; self.closed = False
        self.queries = self.query_rows = 0
        self.configuration = self._configuration = (query_threads, max_owned_bytes)
        self._methods = ()
        try:
            if (type(query_threads) is not int or query_threads not in native.QUERY_THREADS or
                    type(max_owned_bytes) is not int or not 65536 < max_owned_bytes <= native.NATIVE_MAX_BYTES):
                raise ValueError("Unsupported prepared lease configuration")
            descriptor = native.points_descriptor(target_points, target_rows)
            reservation = native.reservation(target_rows); charged = reservation + len(target_points)
            if charged > max_owned_bytes: raise ValueError("Prepared target exceeds owned byte cap before allocation")
            self._methods = tuple((name, callable_state(getattr(backend, name)))
                                  for name in ("create", "query", "destroy", "_check") if hasattr(backend, name))
            self._create, self._query, self._destroy = backend.create, backend.query, backend.destroy
            self._callbacks = (callable_state(self._create), callable_state(self._query), callable_state(self._destroy))
            handle, owned = self._create(target_points, target_rows, reservation)
            # Retain any allocated handle before rejecting contradictory metadata.
            self.entry = self._entry = Entry(target_points, target_rows, json.dumps(descriptor, sort_keys=True),
                                             handle, reservation, charged)
            if type(handle) is not int or not 1 <= handle < 2 ** 64 or owned != reservation:
                raise native.NativeGeometryError("Malformed prepared native construction receipt")
            self._healthy()
        except BaseException as error:
            self.failure = error
            try: error.prepared_target = self
            except BaseException: pass
            raise

    def _healthy(self):
        if self.closed or self.failure:
            raise native.NativeGeometryError("Prepared target is closed or faulted") from self.failure
        if (self.backend is not self._backend or self.entry is not self._entry or
                self.configuration != self._configuration or self._entry is None or
                any(callable_state(getattr(self._backend, name)) != state for name, state in self._methods) or
                tuple(callable_state(method) for method in (self._create, self._query, self._destroy)) != self._callbacks):
            raise native.NativeGeometryError("Prepared target owner/configuration/callable changed")

    def query(self, transformed_queries, query_rows, radius):
        try:
            self._healthy()
            native._blob_length(transformed_queries, query_rows, allow_empty=True)
            radius = native.radius_value(radius)
            ids, squared = self._query(self._entry.handle, transformed_queries, query_rows, radius, self._configuration[0])
            # The unchanged native query checks all coordinates/indices/d2/radius
            # before status0; require complete immutable transport, without a
            # redundant Python per-row loop. Driver exact gold checks are outside
            # the measured query clock on every returned result.
            if type(ids) is not bytes or type(squared) is not bytes or len(ids) != 8 * query_rows or len(squared) != 8 * query_rows:
                raise native.NativeGeometryError("Malformed prepared result transport")
            self._healthy()
            self.queries += 1; self.query_rows += query_rows
            return ids, squared
        except BaseException as error:
            if self.failure is None: self.failure = error
            raise

    def close(self, primary=None):
        try:
            if self._entry is not None:
                self._destroy(self._entry.handle)
                self.entry = self._entry = None
            self.closed = True
        except BaseException as error:
            if self.failure is None: self.failure = error
            if primary is not None:
                if callable(getattr(primary, "add_note", None)): primary.add_note("Prepared target close: " + repr(error))
                raise primary from error
            raise

    def report(self):
        return {"kind": KIND, "configuration": {"query_threads": self._configuration[0], "max_owned_bytes": self._configuration[1]},
                "target": None if self._entry is None else json.loads(self._entry.descriptor_json),
                "owned_reservation_bytes": 0 if self._entry is None else self._entry.charged_bytes,
                "queries": self.queries, "query_rows": self.query_rows, "closed": self.closed,
                "failure": None if self.failure is None else f"{type(self.failure).__name__}: {self.failure}",
                "python_output_scan": False, "native_query_domain_and_result_checks": True,
                "memory_scope": "retained target bytes + conservative native array/index/pool reservation; excludes Python metadata/CRT/stack/RSS",
                "qualified": False, "metric_reducer": False}
