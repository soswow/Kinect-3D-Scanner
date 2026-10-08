"""Audit-only single-copy ICP experiment; never selected by the scanner.

This first causal baseline uses the existing selected device/null stream, not
a CUDA graph. It combines the original flat-NN counters and original normal
equation totals in one 320-byte host copy. Uniform kernel guards skip equations
when any original CPU-resolution flag or malformed result exists. CPU resolution
then precedes the unchanged original equation kernels and Eigen solve.

Every common-path result is also compared with a complete original CPU nearest
audit and the unchanged original equation kernels before the CPU solve consumes
it. This constructor is intentionally audit-only: old NN, component, field or
Finish proofs cannot authorize unaudited timing of the changed orchestration.
It changes no original transform, equation arithmetic/reduction, solve, pose
composition, stage order, strict-radius filter or convergence rule.

Numerical libraries are imported only during construction. Stdlib source and
counter contracts can be checked without importing NumPy, Open3D or CuPy.
"""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ORIGINAL_RESIDENT = ROOT / "scripts/research/archive/research_resident_icp.py"
POLICY = "original-flat-normal-single-copy-audit-v1"
COUNTER_WORDS, EQUATION_WORDS, WORD_BYTES = 10, 30, 8
COMBINED_BYTES = (COUNTER_WORDS + EQUATION_WORDS) * WORD_BYTES
MAX_POINTS = 1_000_000


def original_gpu_source(path=ORIGINAL_RESIDENT):
    """Read the source literal without executing even its stdlib imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "GPU_SOURCE"
                       for target in node.targets)]
    if len(matches) != 1:
        raise RuntimeError("Original resident GPU source literal is ambiguous")
    value = ast.literal_eval(matches[0].value)
    if not isinstance(value, str) or not value:
        raise RuntimeError("Original resident GPU source is not a literal string")
    return value


_EDITS = (
    ('void normal_partials(', 'void combined_guarded_normal_partials('),
    ('double radius_squared, double* partials) {',
     'double radius_squared, double* partials, const unsigned long long* control) {\n'
     '    if (control[0] || control[7]) return;'),
    ('void collapse_partials(', 'void combined_guarded_collapse_partials('),
    ('const double* partials, double* totals, int blocks) {',
     'const double* partials, double* totals, int blocks, const unsigned long long* control) {\n'
     '    if (control[0] || control[7]) {\n'
     '        if (threadIdx.x < 30) totals[threadIdx.x] = 0.;\n'
     '        return;\n'
     '    }'),
)


def guarded_gpu_source(original=None):
    """Add uniform control guards; exact inverse proves all math bytes unchanged."""
    value = original_gpu_source() if original is None else original
    if not isinstance(value, str):
        raise TypeError("Original math source must be text")
    for before, after in _EDITS:
        if value.count(before) != 1 or after in value:
            raise RuntimeError("Original kernel signature differs from the reviewed scope")
        value = value.replace(before, after)
    recovered = value
    for before, after in reversed(_EDITS):
        if recovered.count(after) != 1:
            raise RuntimeError("Guarded source is ambiguous")
        recovered = recovered.replace(after, before)
    if recovered != original:
        # A caller omitting original still needs comparison against its read literal.
        expected = original_gpu_source() if original is None else original
        if recovered != expected:
            raise RuntimeError("Combined guards changed original mathematical source bytes")
    return value


def source_contract():
    original = original_gpu_source()
    guarded = guarded_gpu_source(original)
    recovered = guarded
    for before, after in reversed(_EDITS):
        recovered = recovered.replace(after, before)
    if recovered != original:
        raise RuntimeError("Guarded kernel math cannot be recovered byte for byte")
    return {"policy": POLICY, "original_math_bytes_unchanged": True,
            "original_gpu_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "guarded_gpu_source_sha256": hashlib.sha256(guarded.encode()).hexdigest(),
            "options": ["--fmad=false"], "graph_capture": False,
            "stream_policy": "unchanged selected device and legacy null stream",
            "guard": "Uniform flagged-count or malformed-count check before any equation work; guarded collapse writes zero placeholders that are never solved"}


def validate_provisional_counters(values, count, target_count, *, direct_miss):
    """Validate the original classifier's unaudited counters without numeric imports."""
    if (type(count) is not int or type(target_count) is not int
            or not 0 < count <= MAX_POINTS or not 0 < target_count <= MAX_POINTS
            or type(direct_miss) is not bool or len(values) != COUNTER_WORDS
            or any(type(value) is not int or value < 0 or value >= 2**64 for value in values)):
        raise RuntimeError("Invalid bounded provisional classifier counters")
    flagged, hits, misses, fallback, unsupported, uncertain, visits, malformed, audit_hits, audit_misses = values
    if (malformed or flagged > count or flagged != fallback
            or hits + misses + fallback != count or unsupported > fallback
            or uncertain > fallback or visits > count * target_count
            or audit_hits or audit_misses or (not direct_miss and misses)):
        raise RuntimeError("Malformed/inconsistent provisional NN results; CPU recovery is prohibited")
    return {"flagged": flagged, "hits": hits, "misses": misses,
            "fallback": fallback, "unsupported": unsupported, "uncertain": uncertain,
            "candidate_visits": visits}


def scratch_forecast(count, target_count):
    """Bound arrays; allocator pools, host audit copies and Python objects are separate."""
    if (type(count) is not int or type(target_count) is not int
            or not 0 <= count <= MAX_POINTS or not 0 <= target_count <= MAX_POINTS):
        raise ValueError("Unsupported bounded source/target size")
    blocks = (count + 127) // 128
    resident = 2 * count * 24 + target_count * 24 + count * 4 + blocks * 30 * 8 + 30 * 8 + 16 * 8
    # Original audit query path remains at136N+80; persistent fast buffers add
    # ids4N, metric8N, reasons4N, flagged4N, and combined320B. The transient
    # raw40N is released after the primary copy and before original audit starts.
    persistent = 20 * count + COMBINED_BYTES
    provisional = persistent + 40 * count
    original_audit = 136 * count + 80
    peak_query = persistent + original_audit  # >= provisional for everyN.
    return {"resident_owned_bytes": resident, "persistent_fast_bytes": persistent,
            "provisional_query_bytes": provisional, "original_audit_query_bytes": original_audit,
            "peak_query_bytes": peak_query, "combined_array_bytes": resident + peak_query}


# This parent module imports stdlib only. Construction performs numerical imports.
from scripts.research.research_device_grid_resident_icp import DeviceGridResidentICP
from scripts.research.archive.research_resident_icp import ResidentFallback, BLOCK


class CombinedSyncResidentICP(DeviceGridResidentICP):
    """Fresh fully shadowed trajectory only; no unaudited constructor exists yet."""

    def __init__(self, retrieval, solve, **kwargs):
        if (getattr(retrieval, "audit_nearest", None) is not True
                or getattr(retrieval, "audit_misses", None) is not True
                or getattr(retrieval, "proof_authority", None) is not None
                or kwargs.get("gpu_timing", True) is not False):
            raise ValueError("Combined orchestration requires fresh full audits, no old timing authority, and GPU event timers disabled")
        self.contract = source_contract()
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._scratch = None
        super().__init__(retrieval, solve, **kwargs)
        self._configuration = self._current_configuration()
        self.combined_statistics = {key: 0 for key in (
            "iteration_calls", "primary_counter_term_syncs", "primary_download_bytes",
            "provisional_queries", "provisional_candidate_visits", "common_calls", "common_queries",
            "flagged_calls", "flagged_queries", "discarded_placeholder_calls", "malformed_results",
            "full_query_audited_calls", "full_query_audited_rows", "id_bit_comparison_rows",
            "metric_bit_comparison_rows", "filtered_bit_comparison_rows", "term_bit_comparison_calls",
            "equation_bit_mismatches", "nearest_bit_mismatches", "peak_combined_owned_query_bytes")}
        self.combined_statistics.update({key: 0. for key in (
            "primary_enqueue_s", "primary_copy_wait_s", "primary_normalize_s",
            "original_full_audit_s", "original_equation_shadow_s", "proof_comparison_s")})
        cp = self.cp
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            text = guarded_gpu_source()
            self.combined_partial_kernel = cp.RawKernel(text, "combined_guarded_normal_partials", options=("--fmad=false",))
            self.combined_collapse_kernel = cp.RawKernel(text, "combined_guarded_collapse_partials", options=("--fmad=false",))
            self.combined_partial_kernel.compile()
            self.combined_collapse_kernel.compile()

    def _current_configuration(self):
        retrieval = self.retrieval
        return (self.device_id, retrieval.audit_nearest, retrieval.audit_misses,
                retrieval.proof_authority, retrieval.miss_policy, retrieval.max_query_bytes,
                retrieval.max_clouds, retrieval.max_cache_bytes, self.max_points,
                self.max_scratch_bytes, self.gpu_timing, retrieval.kernel, retrieval.classify)

    def _check_configuration(self, *, check_source=False):
        if self._current_configuration() != self._configuration:
            raise RuntimeError("Combined audit/device/memory/kernel policy changed after construction")
        if check_source and source_contract() != self.contract:
            raise RuntimeError("Original normal math changed during combined orchestration proof")

    def _match_resident(self, source, target, initial):
        # Source parsing is a call-boundary audit, not iteration control work.
        self._check_configuration(check_source=True)
        forecast = scratch_forecast(len(source.points), len(target.points))
        if forecast["peak_query_bytes"] > self.retrieval.max_query_bytes:
            raise ResidentFallback("Combined provisional plus complete-audit query byte cap exceeded")
        if forecast["combined_array_bytes"] > self.max_scratch_bytes:
            raise ResidentFallback("Combined math plus provisional/full-audit scratch byte cap exceeded")
        self.combined_statistics["peak_combined_owned_query_bytes"] = max(
            self.combined_statistics["peak_combined_owned_query_bytes"], forecast["combined_array_bytes"])
        try:
            return super()._match_resident(source, target, initial)
        finally:
            # Every successful original match already copied final correspondences;
            # a failed async launch still needs completion before owned buffers leave.
            primary = sys.exception()
            try:
                self.cp.cuda.Stream.null.synchronize()
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(f"Combined scratch cleanup synchronization also failed: {cleanup_error}")
            finally:
                self._scratch = None

    def _buffers(self, count):
        if self._scratch is None:
            cp = self.cp
            self._scratch = {"count": count, "ids": cp.empty(count, cp.int32),
                "squared": cp.empty(count, cp.float64), "reasons": cp.empty(count, cp.uint32),
                "flagged": cp.empty(count, cp.uint32), "combined": cp.empty(40, cp.uint64)}
        if self._scratch["count"] != count:
            raise RuntimeError("Source query size changed within one inherited resident match")
        return self._scratch

    def _normalize_equations(self, values, count):
        np = self.np
        if values.shape != (30,) or values[29] or not np.isfinite(values).all():
            raise ResidentFallback("Invalid resident coordinates, correspondence IDs or equations")
        if values[28] < 0 or values[28] > count or int(values[28]) != values[28]:
            raise RuntimeError("Malformed original equation count")
        matrix = np.zeros((6, 6), dtype=np.float64)
        upper = np.triu_indices(6)
        matrix[upper] = values[:21]
        matrix[(upper[1], upper[0])] = values[:21]
        selected = int(values[28])
        return matrix, values[21:27], selected / count, (values[27] / selected)**.5 if selected else 0., selected

    def _iteration_data(self, moving, item, normals, filtered, radius, partials, totals):
        self._check_configuration()
        cp, np, stats = self.cp, self.np, self.combined_statistics
        count = len(moving)
        if (not isinstance(moving, cp.ndarray) or moving.dtype != cp.float64 or moving.shape != (count, 3)
                or not moving.flags.c_contiguous or moving.device.id != self.device_id
                or not 0 < count <= self.max_points or not math.isfinite(radius)
                or radius <= 0 or not math.isfinite(radius * radius)):
            raise RuntimeError("Unsupported combined query buffer/radius")
        buffers = self._buffers(count)
        combined = buffers["combined"]
        combined[...] = 0
        provisional_totals = combined.view(cp.float64)[10:]
        started = time.perf_counter()
        raw = self.retrieval._raw_device(moving, item, radius)
        self.retrieval.classify(((count + 127) // 128,), (128,),
            (raw, np.uint32(count), np.uint32(len(item["target"].points)), np.float64(radius * radius),
             np.int32(self.retrieval.miss_policy == "direct-miss-research-v1"), np.int32(0), np.int32(0),
             buffers["ids"], buffers["squared"], buffers["reasons"], buffers["flagged"], combined))
        blocks = len(partials)
        self.combined_partial_kernel((blocks,), (BLOCK,),
            (moving, item["data"], normals, buffers["ids"], filtered,
             np.int32(count), np.int32(len(item["data"])), np.float64(radius * radius), partials, combined))
        self.combined_collapse_kernel((1,), (32,), (partials, provisional_totals, np.int32(blocks), combined))
        stats["primary_enqueue_s"] += time.perf_counter() - started
        started = time.perf_counter()
        host_combined = cp.asnumpy(combined)
        stats["primary_copy_wait_s"] += time.perf_counter() - started
        stats["iteration_calls"] += 1
        stats["primary_counter_term_syncs"] += 1
        stats["primary_download_bytes"] += host_combined.nbytes
        del raw  # Complete stream copy above; do not retain40N across original audit.
        try:
            counters = validate_provisional_counters([int(value) for value in host_combined[:10]],
                count, len(item["target"].points), direct_miss=self.retrieval.miss_policy == "direct-miss-research-v1")
        except BaseException:
            stats["malformed_results"] += 1
            raise
        stats["provisional_queries"] += count
        stats["provisional_candidate_visits"] += counters["candidate_visits"]
        # This is the original complete CPU query audit, including original scalar
        # ambiguity resolution, BEFORE any original Eigen solve can be reached.
        started = time.perf_counter()
        nearest, distances = self.retrieval.nearest_device(moving, item, radius)
        audit_wall = time.perf_counter() - started
        stats["original_full_audit_s"] += audit_wall
        self.statistics["nn_wall_s"] += audit_wall
        stats["full_query_audited_calls"] += 1
        stats["full_query_audited_rows"] += count
        if (nearest.dtype != cp.int32 or nearest.shape != (count,)
                or nearest.device.id != self.device_id or not nearest.flags.c_contiguous):
            raise RuntimeError("Original CPU-audited nearest buffer has incompatible ownership")
        if counters["flagged"]:
            stats["flagged_calls"] += 1
            stats["flagged_queries"] += counters["flagged"]
            stats["discarded_placeholder_calls"] += 1
            # Neither stale filtered IDs nor zero guard placeholders are consumed.
            # CPU resolution has completed; delegate exact original equation math.
            return super()._equations(moving, item["data"], normals, nearest, filtered, radius, partials, totals)
        stats["common_calls"] += 1
        stats["common_queries"] += count
        started = time.perf_counter()
        values = host_combined.view(np.float64)[10:].copy()
        result = self._normalize_equations(values, count)
        stats["primary_normalize_s"] += time.perf_counter() - started
        started = time.perf_counter()
        fast_ids, gold_ids = cp.asnumpy(buffers["ids"]), cp.asnumpy(nearest)
        fast_d2, gold_d2 = cp.asnumpy(buffers["squared"]), cp.asnumpy(distances)
        fast_filtered = cp.asnumpy(filtered)
        id_same = np.array_equal(fast_ids.view(np.uint32), gold_ids.view(np.uint32))
        metric_same = np.array_equal(fast_d2.view(np.uint64), gold_d2.view(np.uint64))
        stats["id_bit_comparison_rows"] += count
        stats["metric_bit_comparison_rows"] += count
        if not id_same or not metric_same:
            stats["nearest_bit_mismatches"] += 1
            raise RuntimeError("Combined provisional NN differs from full original CPU query audit")
        stats["proof_comparison_s"] += time.perf_counter() - started
        started = time.perf_counter()
        super()._equations(moving, item["data"], normals, nearest, filtered, radius, partials, totals)
        stats["original_equation_shadow_s"] += time.perf_counter() - started
        started = time.perf_counter()
        gold_values = cp.asnumpy(totals)
        gold_filtered = cp.asnumpy(filtered)
        term_same = np.array_equal(values.view(np.uint64), gold_values.view(np.uint64))
        filter_same = np.array_equal(fast_filtered.view(np.uint32), gold_filtered.view(np.uint32))
        stats["term_bit_comparison_calls"] += 1
        stats["filtered_bit_comparison_rows"] += count
        stats["proof_comparison_s"] += time.perf_counter() - started
        if not term_same or not filter_same:
            stats["equation_bit_mismatches"] += 1
            raise RuntimeError("Guarded original equation terms/filter differ in numerical bits")
        return result

    def report(self):
        value = super().report()
        value.update(combined_sync={"policy": POLICY, "source_contract": dict(self.contract),
            "statistics": dict(self.combined_statistics), "audit_only": True,
            "new_timing_authority": False, "graph_capture": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256_before": self.source_sha256,
            "source_unchanged": hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == self.source_sha256,
            "timer_scope": "Primary copy wait includes queued original raw NN and guarded equations; it is not a removable memory-copy cost. Original NN/equation shadows and bit comparisons add intentional proof overhead.",
            "memory_scope": "One match-owned persistent20N+320B buffer plus original audit136N+80B; provisional raw40N released before audit. Original resident arrays, host proof copies and allocator pools are separately scoped."})
        value["performance_attribution_valid"] = False
        return value
