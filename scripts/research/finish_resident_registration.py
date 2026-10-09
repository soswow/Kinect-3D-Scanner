"""Offline-only scoped Finish injection; original graph gates remain in charge.

Importing this helper loads no numerical packages.  A research fault is a
latched BaseException because the production Finish pipeline deliberately
converts ordinary registration exceptions into retained-pose reports.
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.validate_finish_resident_proof import (
    STAGES, THREAD_POLICY, call_signature, validate_expected_call,
    validate_complete_calls, validate_shadow)


class FinishResearchFailure(BaseException):
    """A failed offline experiment must never produce an accepted mesh/report."""


def array_descriptor(value):
    import numpy as np
    array = np.asarray(value)
    if array.dtype != np.dtype("<f8") or array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Require finite original native FP64 arrays")
    return {"sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            "dtype": "<f8", "shape": list(array.shape)}


def capture_inputs(source, target, initial, site, gate_context, thread_policy):
    return {"source_points": array_descriptor(source.points),
            "source_normals": array_descriptor(source.normals),
            "target_points": array_descriptor(target.points),
            "target_normals": array_descriptor(target.normals),
            "seed": array_descriptor(initial), "stages": [list(item) for item in STAGES],
            "huber_m": .01, "relative_fitness": 1e-6, "relative_rmse": 1e-6,
            "thread_policy": dict(thread_policy), "site": site,
            "gate_context": list(gate_context)}


def summarize_result(result, source, target):
    import numpy as np
    pose = np.asarray(result.transformation)
    fitness, rmse = float(result.fitness), float(result.inlier_rmse)
    pairs = np.asarray(result.correspondence_set)
    if (pose.dtype != np.dtype("<f8") or pose.shape != (4, 4)
            or not np.isfinite(pose).all() or not math.isfinite(fitness)
            or not 0 <= fitness <= 1 or not math.isfinite(rmse) or rmse < 0
            or pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu"):
        raise ValueError("Malformed original registration result contract")
    if len(pairs) and (np.any(pairs < 0) or np.any(pairs[:, 0] >= len(source.points))
                      or np.any(pairs[:, 1] >= len(target.points))
                      or len(np.unique(pairs[:, 0])) != len(pairs)):
        raise ValueError("Correspondence IDs are outside original point order")
    expected_fitness = len(pairs) / len(source.points) if len(source.points) else 0.
    if abs(fitness - expected_fitness) > 1e-12:
        raise ValueError("Registration fitness disagrees with original source correspondence count")
    canonical = pairs[np.argsort(pairs[:, 0], kind="stable")] if len(pairs) else pairs
    return {"transformation": pose.tolist(), "fitness": fitness, "rmse": rmse,
            "correspondence_count": len(pairs), "correspondence_sha256":
            hashlib.sha256(np.ascontiguousarray(pairs, dtype="<i8").tobytes()).hexdigest(),
            "correspondence_mapping_sha256":
            hashlib.sha256(np.ascontiguousarray(canonical, dtype="<i8").tobytes()).hexdigest()}


def compare_results(resident, shadow, source, target):
    import numpy as np
    left, right = summarize_result(resident, source, target), summarize_result(shadow, source, target)
    result = {"transformation_max_abs_delta": float(np.max(np.abs(
                  np.asarray(left["transformation"]) - np.asarray(right["transformation"])))),
              "fitness_abs_delta": abs(left["fitness"] - right["fitness"]),
              "rmse_abs_delta": abs(left["rmse"] - right["rmse"]),
              "correspondence_ids_equal": left["correspondence_count"] == right["correspondence_count"]
                  and left["correspondence_mapping_sha256"] == right["correspondence_mapping_sha256"]}
    result["passed"] = result["correspondence_ids_equal"] and all(
        result[field] <= 1e-8 for field in ("transformation_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta"))
    return result


def evidence(value):
    """Copy bounded gate outputs, including information matrices, without replay."""
    import numpy as np
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("Nonfinite gate evidence")
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        item = {"dtype": value.dtype.str, "shape": list(value.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
        if value.size <= 64:
            item["values"] = value.tolist()
        return item
    if isinstance(value, dict):
        return {str(k): evidence(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [evidence(v) for v in value]
    if hasattr(value, "transformation") and hasattr(value, "fitness"):
        pairs = np.asarray(value.correspondence_set)
        canonical = pairs[np.argsort(pairs[:, 0], kind="stable")] if len(pairs) else pairs
        return {"transformation": evidence(np.asarray(value.transformation)),
                "fitness": evidence(float(value.fitness)), "rmse": evidence(float(value.inlier_rmse)),
                "correspondence_set": evidence(pairs), "correspondence_mapping": evidence(canonical)}
    # Gate inputs may contain view/fragment/cloud objects. Only outputs selected
    # by observe() are captured; unknown output types must not be silently hashed.
    raise TypeError(f"Unsupported gate evidence type: {type(value).__name__}")


class FinishRegistrationScope:
    def __init__(self, registration_module, original_match, solver=None, *, mode="audit",
                 trace=None, authority=None, capture_inputs=capture_inputs,
                 result_summary=summarize_result, compare_results=compare_results,
                 thread_policy=THREAD_POLICY, close_solver=True):
        if mode not in ("native", "audit", "timing") or (mode != "native" and solver is None):
            raise ValueError("Require a native reference or separately owned resident solver")
        if mode == "timing" and authority is None:
            raise ValueError("Full-Finish timing requires distinct trajectory authority")
        self.module, self.original_match, self.solver = registration_module, original_match, solver
        self.mode, self.authority, self.trace = mode, authority, trace or (lambda row: None)
        self.capture, self.summarize, self.compare = capture_inputs, result_summary, compare_results
        self.thread_policy, self.close_solver = dict(thread_policy), close_solver
        self.bypass = contextvars.ContextVar("finish_original_cpu_bypass", default=False)
        self.gate_context = contextvars.ContextVar("finish_original_gate_context", default=())
        self.failure, self.cleanup_failures, self.calls, self.gates = None, [], [], []
        self.patches, self.targets = [], set()
        self.complete, self.restored, self.entered = False, False, False
        self.cpu_shadow_calls, self.full_cpu_fallback_calls = 0, 0

    def fail(self, error):
        if self.failure is None:
            self.failure = (error if isinstance(error, FinishResearchFailure) else
                            FinishResearchFailure(f"{type(error).__name__}: {error}"))
            self.failure.__cause__ = error if self.failure is not error else None
        raise self.failure

    def healthy(self):
        if self.failure is not None:
            raise self.failure

    def _cpu(self, source, target, initial):
        token = self.bypass.set(True)
        try:
            return self.original_match(source, target, initial)
        finally:
            self.bypass.reset(token)

    def _full_fallback(self, *args, **kwargs):
        self.full_cpu_fallback_calls += 1
        self.fail(ValueError("Unsupported complete resident call; strict Finish research forbids full CPU retry"))

    def _site(self):
        frame = inspect.currentframe().f_back
        try:
            while frame:
                path = Path(frame.f_code.co_filename).resolve()
                if path != Path(__file__).resolve() and "scanner_server" in path.parts:
                    return {"file": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "function": frame.f_code.co_name, "line": frame.f_lineno}
                frame = frame.f_back
            return {"file": "offline-test", "function": "original_match", "line": 1}
        finally:
            del frame

    def dispatch(self, source, target, initial):
        if self.bypass.get():
            return None  # captured refinement._match executes its exact legacy body
        self.healthy()
        site, context = self._site(), self.gate_context.get()
        row = {"event": "match", "call_index": len(self.calls), "complete": False}
        try:
            payload = self.capture(source, target, initial, site, context, self.thread_policy)
            signature = call_signature(payload)
            row.update(call_inputs=payload, call_signature=signature)
            self.targets.add(payload["target_points"]["sha256"])
            if self.mode == "timing":
                validate_expected_call(self.authority, len(self.calls), payload)
            before = dict(getattr(self.solver, "statistics", {})) if self.solver else {}
            fallback_before = self.full_cpu_fallback_calls
            started = time.perf_counter()
            result = self._cpu(source, target, initial) if self.mode == "native" else self.solver.match(source, target, initial)
            row["registration_wall_s"] = time.perf_counter() - started
            row["result"] = self.summarize(result, source, target)
            after = self.capture(source, target, initial, site, context, self.thread_policy)
            row["inputs_after_resident"] = call_signature(after)
            if row["inputs_after_resident"] != signature:
                raise ValueError("Resident registration mutated original inputs")
            row["resident_full_call_fallbacks"] = self.full_cpu_fallback_calls - fallback_before
            row["query_statistics_delta"] = {key: value - before.get(key, 0) for key, value in
                                               getattr(self.solver, "statistics", {}).items()} if self.solver else {}
            if self.mode == "audit":
                row["inputs_before_shadow"] = call_signature(self.capture(source, target, initial, site, context, self.thread_policy))
                started = time.perf_counter()
                shadow = self._cpu(source, target, initial)
                row["cpu_shadow_wall_s"] = time.perf_counter() - started
                self.cpu_shadow_calls += 1
                row["shadow_result"] = self.summarize(shadow, source, target)
                row["inputs_after_shadow"] = call_signature(self.capture(source, target, initial, site, context, self.thread_policy))
                row["cpu_shadow"] = self.compare(result, shadow, source, target)
                validate_shadow(row)
            row["complete"] = True
            self.calls.append(signature)
            self.trace(row)
            return result
        except BaseException as error:
            row["failure"] = {"type": type(error).__name__, "message": str(error)}
            try:
                self.trace(row)
            except BaseException as secondary:
                self.cleanup_failures.append(f"Failed trace preservation: {secondary}")
            self.fail(error)

    def observe(self, module, name, label=None):
        original = getattr(module, name)
        label = label or name

        def observed(*args, **kwargs):
            self.healthy()
            index = len(self.gates)
            self.gates.append(label)
            token = self.gate_context.set(self.gate_context.get() + ({"function": label, "invocation": index},))
            row = {"event": "gate", "gate_index": index, "function": label, "complete": False}
            try:
                result = original(*args, **kwargs)
                try:
                    row.update(complete=True, result=evidence(result))
                    self.trace(row)
                except BaseException as recording_error:
                    self.fail(recording_error)
                return result
            except BaseException as error:
                # Ordinary original gate exceptions retain original behavior;
                # a recording fault or existing research latch is fatal.
                row["failure"] = {"type": type(error).__name__, "message": str(error)}
                try:
                    self.trace(row)
                except BaseException as secondary:
                    self.fail(secondary)
                if self.failure is not None or row.get("complete"):
                    self.fail(error)
                raise
            finally:
                self.gate_context.reset(token)

        self.patches.append((module, name, original))
        setattr(module, name, observed)

    def __enter__(self):
        if self.entered:
            raise ValueError("Finish research context cannot be reused")
        self.entered = True
        try:
            self.patches.append((self.module, "match", self.module.match))
            self.module.match = self.dispatch
            if self.solver is not None:
                resident = getattr(self.solver, "resident", self.solver)
                self.patches.append((resident, "cpu_fallback", resident.cpu_fallback))
                resident.cpu_fallback = self._full_fallback
        except BaseException as error:
            # Python does not call __exit__ when __enter__ fails.
            self.__exit__(type(error), error, error.__traceback__)
        return self

    def finish(self):
        self.healthy()
        try:
            if self.mode == "timing":
                validate_complete_calls(self.authority, len(self.calls))
            if not self.calls:
                raise ValueError("No original Finish registration calls were exercised")
            self.complete = True
        except BaseException as error:
            self.fail(error)

    def __exit__(self, kind, error, traceback):
        if error is not None and self.failure is None:
            self.failure = error
        for owner, name, original in reversed(self.patches):
            try:
                setattr(owner, name, original)
            except BaseException as secondary:
                self.cleanup_failures.append(f"Restore {name}: {secondary}")
        self.restored = not self.cleanup_failures
        if self.close_solver and self.solver is not None:
            try:
                self.solver.close()
            except BaseException as secondary:
                self.cleanup_failures.append(f"Close resident solver: {secondary}")
        if self.failure is not None:
            for secondary in self.cleanup_failures:
                self.failure.add_note(secondary)
            raise self.failure
        if self.cleanup_failures:
            self.fail(RuntimeError("; ".join(self.cleanup_failures)))
        if not self.complete:
            self.fail(ValueError("Finish body did not validate healthy complete call scope"))
        return False

    def report(self):
        return {"complete": self.complete and self.failure is None, "calls": len(self.calls),
                "call_signatures": list(self.calls), "input_immutability_passed": self.failure is None,
                "cpu_shadow_calls": self.cpu_shadow_calls, "cpu_shadow_failures": int(self.failure is not None),
                "full_cpu_fallback_calls": self.full_cpu_fallback_calls,
                "target_digests": sorted(self.targets), "restored": self.restored,
                "failure": None if self.failure is None else {"type": type(self.failure).__name__, "message": str(self.failure)},
                "cleanup_failures": list(self.cleanup_failures),
                "resident": self.solver.resident.report() if self.solver and hasattr(self.solver, "resident") else None,
                "retrieval_statistics": dict(getattr(self.solver, "statistics", {})),
                "gate_calls": len(self.gates)}
