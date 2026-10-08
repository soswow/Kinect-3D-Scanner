"""Exact original-double CPU corner minima with a candidate-seeded x strip.

Standalone unexecuted research. SciPy's batch KDTree supplies only a valid
candidate row, never final distances or tie authority. An outward interval
then retains every point that can beat/tie that candidate under the original
NumPy FP64 metric. Final IDs use the original row order. No native imports at
module import, production hook, approximate search or pose authority.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import time

MAX_TARGETS = 640 * 480
MAX_QUERIES = 250 * 4
MAX_COORDINATE = 2048.
MAX_SQUARED = 2 * (2 * MAX_COORDINATE) ** 2


def configuration(workers, max_strip_rows, audit):
    if (type(workers) is not int or not 1 <= workers <= 20
            or type(max_strip_rows) is not int or not 1 <= max_strip_rows <= MAX_TARGETS
            or type(audit) is not bool):
        raise ValueError("Require bounded workers/strip rows and explicit audit bool")
    return workers, max_strip_rows, audit


def outward_x_interval(query_x, candidate_squared):
    """Enclose every x whose original RN squared component can be <= d2.

For nonnegative components, RN(a+b)>=a,b. Thus a winning/tied point's
RN(dx*dx)<=candidate_squared. Exact dx*dx is below nextafter(d2,+inf).
The sqrt bound is checked as an exact rational, not assumed correctly rounded.
One further outward ULP encloses the exact subtraction underlying RN(dx), and
outward endpoints enclose qx +/- that bound. This includes squared underflow.
"""
    if (not isinstance(query_x, (int, float)) or isinstance(query_x, bool)
            or not math.isfinite(query_x) or abs(query_x) > MAX_COORDINATE
            or not isinstance(candidate_squared, (int, float)) or isinstance(candidate_squared, bool)
            or not math.isfinite(candidate_squared) or not 0 <= candidate_squared <= MAX_SQUARED
            or sys.float_info.radix != 2 or sys.float_info.mant_dig != 53):
        raise ValueError("Require finite declared-domain IEEE FP64 original metric")
    qx, squared = float(query_x), float(candidate_squared)
    squared_upper = math.nextafter(squared, math.inf)
    radius = math.sqrt(squared_upper)
    numerator, denominator = squared_upper.as_integer_ratio()
    for _ in range(8):
        rn, rd = radius.as_integer_ratio()
        if rn * rn * denominator >= numerator * rd * rd:
            break
        radius = math.nextafter(radius, math.inf)
    else:
        raise ArithmeticError("Cannot certify outward square-root bound")
    # If rounded |dx| is <= radius, the exact difference cannot reach the next
    # representable value above radius. This safely widens the entire binade.
    exact_difference_limit = math.nextafter(radius, math.inf)
    return (math.nextafter(qx - exact_difference_limit, -math.inf),
            math.nextafter(qx + exact_difference_limit, math.inf))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeCornerCPU:
    def __init__(self, *, workers=1, max_strip_rows=65536, audit=True):
        started = time.perf_counter()
        self.workers, self.max_strip_rows, self.audit = configuration(workers, max_strip_rows, audit)
        self._configuration = (self.workers, self.max_strip_rows, self.audit)
        import numpy as np
        import scipy
        from scipy.spatial import cKDTree, _ckdtree
        self.np, self.tree_type = np, cKDTree
        module = Path(_ckdtree.__file__).resolve()
        self.provenance = {"script_sha256": file_hash(__file__), "numpy": np.__version__,
            "scipy": scipy.__version__, "ckdtree_binary": {"path": str(module), "sha256": file_hash(module)},
            "setup_and_import_s": time.perf_counter()-started,
            "workers": workers, "max_strip_rows": max_strip_rows,
            "status": "source-prepared; requires fresh actual-query proof and comparison before any speed claim",
            "candidate_scope": "Any valid cKDTree row is an upper-bound seed. Its metric/tie/pruning policy does not authorize final results.",
            "rounding_scope": "Enclosure assumes IEEE round-to-nearest FP64 subtraction/multiply/add. This prototype grants no unaudited authority; fresh actual original-metric shadows must detect unsupported runtime behavior.",
            "tie_policy": "Minimum original NumPy double squared distance, then lowest original input row",
            "scope": "Exact full nearest ID and squared metric including outside three-pixel association radius. Per-view index, no retained multi-view cache or pose authority."}

    def _binding(self, array):
        np = self.np
        return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}

    def query(self, queries, targets):
        np = self.np
        if self._configuration != (self.workers, self.max_strip_rows, self.audit):
            raise RuntimeError("CPU corner resource/audit policy changed after construction")
        started = time.perf_counter()
        bindings = []
        for name, array, bound in (("queries", queries, MAX_QUERIES), ("targets", targets, MAX_TARGETS)):
            if (not isinstance(array, np.ndarray) or array.dtype != np.float64 or array.ndim != 2
                    or array.shape[1] != 2 or not 0 < len(array) <= bound
                    or not np.isfinite(array).all() or np.any(np.abs(array) > MAX_COORDINATE)):
                raise ValueError("Require original finite FP64 %s(N,2) within declared bounds" % name)
            bindings.append(self._binding(array))
        validation_s = time.perf_counter()-started
        index_started = time.perf_counter()
        # Private original-order immutable data gives the tree no mutable input
        # aliases. Sorting affects enumeration only, never original row identity.
        points = np.array(targets, dtype=np.float64, order="C", copy=True)
        query_data = np.array(queries, dtype=np.float64, order="C", copy=True)
        points.flags.writeable = query_data.flags.writeable = False
        order = np.argsort(points[:, 0], kind="stable")
        sorted_x = np.ascontiguousarray(points[order, 0])
        order.flags.writeable = sorted_x.flags.writeable = False
        tree = self.tree_type(points, leafsize=16, compact_nodes=True, balanced_tree=True, copy_data=False)
        if self._binding(np.asarray(tree.data)) != bindings[1]:
            raise RuntimeError("KDTree changed original target bytes/order")
        np.asarray(tree.data).flags.writeable = False
        build_s = time.perf_counter()-index_started
        candidate_started = time.perf_counter()
        _, seed = tree.query(query_data, k=1, p=2, eps=0., workers=self.workers)
        seed = np.asarray(seed)
        if (seed.shape != (len(query_data),) or seed.dtype.kind not in "iu"
                or np.any(seed < 0) or np.any(seed >= len(points))):
            raise RuntimeError("Malformed candidate row IDs")
        candidate_s = time.perf_counter()-candidate_started
        result_i, result_d = np.empty(len(queries), np.int32), np.empty(len(queries), np.float64)
        interval_s = rerank_s = 0.
        strip_rows = candidate_rows = brute_fallback_queries = 0
        maximum_strip_rows = 0
        for i, query in enumerate(query_data):
            interval_started = time.perf_counter()
            # Exactly the original NumPy expression. A seed need not be nearest;
            # any original point supplies a valid original-metric upper bound.
            seed_squared = float(np.sum((points[seed[i]:seed[i]+1]-query)**2, axis=1)[0])
            low, high = outward_x_interval(float(query[0]), seed_squared)
            begin = int(np.searchsorted(sorted_x, low, side="left"))
            end = int(np.searchsorted(sorted_x, high, side="right"))
            count = end-begin
            maximum_strip_rows = max(maximum_strip_rows, count)
            strip_rows += count
            if not count or not low <= points[seed[i], 0] <= high:
                raise RuntimeError("Certified candidate interval omitted its seed")
            interval_s += time.perf_counter()-interval_started
            rerank_started = time.perf_counter()
            if count > self.max_strip_rows:
                # Wide strips are exact original full-brute fallbacks, explicitly
                # measured. This bounds retained candidate-selection arrays.
                distances = np.sum((points-query)**2, axis=1)
                index = int(np.argmin(distances))
                minimum = distances[index]
                brute_fallback_queries += 1
                candidate_rows += len(points)
            else:
                rows = order[begin:end]
                distances = np.sum((points[rows]-query)**2, axis=1)
                minimum = np.min(distances)
                index = int(np.min(rows[distances == minimum]))
                candidate_rows += count
            result_i[i], result_d[i] = index, minimum
            rerank_s += time.perf_counter()-rerank_started
        final_started = time.perf_counter()
        if (np.any(result_i < 0) or np.any(result_i >= len(points))
                or not np.isfinite(result_d).all() or np.any(result_d < 0)
                or self._binding(queries) != bindings[0] or self._binding(targets) != bindings[1]
                or self._binding(np.asarray(tree.data)) != bindings[1]):
            raise RuntimeError("Malformed CPU minima or mutated original input/index")
        result_i.flags.writeable = result_d.flags.writeable = False
        final_s = time.perf_counter()-final_started
        all_in_s = time.perf_counter()-started
        shadow_s = 0.
        if self.audit:
            shadow_started = time.perf_counter()
            gold_i, gold_d = np.empty(len(queries), np.int32), np.empty(len(queries), np.float64)
            for i, query in enumerate(queries):
                distances = np.sum((targets-query)**2, axis=1)
                index = int(np.argmin(distances))
                gold_i[i], gold_d[i] = index, distances[index]
            if (np.any(gold_i != result_i)
                    or np.any(gold_d.view(np.uint64) != result_d.view(np.uint64))
                    or self._binding(queries) != bindings[0] or self._binding(targets) != bindings[1]):
                raise RuntimeError("Original full-brute IDs/metric bits disagree or audit inputs mutated")
            shadow_s = time.perf_counter()-shadow_started
        return result_i, result_d, {"query_rows": len(queries), "target_rows": len(targets),
            "query_binding": bindings[0], "target_binding": bindings[1],
            "cpu_map_all_in_s": all_in_s, "input_validation_hash_s": validation_s,
            "index_copy_sort_build_s": build_s, "batch_candidate_query_s": candidate_s,
            "certified_interval_search_s": interval_s, "original_metric_rerank_s": rerank_s,
            "final_validation_hash_s": final_s, "original_full_brute_shadow_s": shadow_s,
            "all_queries_shadowed": self.audit, "strip_rows": strip_rows, "reranked_rows": candidate_rows,
            "maximum_strip_rows": maximum_strip_rows, "full_brute_fallback_queries": brute_fallback_queries,
            "original_row_ties_preserved": True, "outside_radius_nearest_ids_retained": True,
            "retained_visible_array_bytes": points.nbytes+query_data.nbytes+order.nbytes+sorted_x.nbytes,
            "retained_state_after_return": "No per-view tree or point/index arrays retained by the helper",
            "memory_scope": "Point/query counts and candidate strips bounded. Native KDTree allocations and NumPy transient buffers remain separately measured RSS, not a claimed byte cap.",
            "scope": "CPU all-in includes per-view validation, immutable copies, sort/tree, query, exact rerank and final hashes; separate constructor imports/runtime hashing and original brute shadow excluded."}
