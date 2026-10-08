"""Bounded observation of the unchanged resident Eigen callable.

No numerical imports occur at module import. The owner supplies its already
loaded NumPy and original EigenSolve. Each call invokes the original exactly
once and returns its identical output object. Copies never seed registration.
"""

from contextlib import contextmanager
import hashlib
import json
import re
import sys

MAX_RECORDS = 20_000
ARRAY_BYTES_PER_RECORD = 468  # A288 + gradient48 + update128 + status4.
KIND = "observed-original-resident-eigen-equations-v1"


class EquationObservationError(RuntimeError):
    pass


def digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def restore_owned_hooks(hooks, saved_argv, *, argv_owner=sys, clear=True):
    """Attempt every owned restoration, including argv, after one failure."""
    failures = []
    for owner, name, original in reversed(hooks):
        try:
            setattr(owner, name, original)
            if getattr(owner, name) is not original:
                raise EquationObservationError("An owned observer hook did not restore its original object")
        except BaseException as error:
            failures.append(error)
    try:
        argv_owner.argv = saved_argv
        if argv_owner.argv is not saved_argv:
            raise EquationObservationError("Owned argv restoration did not preserve the saved object")
    except BaseException as error:
        failures.append(error)
    if clear:
        hooks.clear()
    return failures


class EquationRecorder:
    def __init__(self, np, *, max_records=MAX_RECORDS):
        if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS:
            raise ValueError("Require a bounded positive original equation record count")
        self.np, self.max_records = np, max_records
        self.records, self.contexts = [], []
        self.failure = None
        self.delegates = 0

    def check(self):
        if self.failure is not None:
            raise EquationObservationError("Equation observation failed; partial copies are diagnostic only") from self.failure

    def _latch(self, error):
        if self.failure is None:
            self.failure = error

    @contextmanager
    def scope(self, **values):
        self.check()
        # Freeze JSON values so external mutations cannot relabel a record.
        try:
            context = json.loads(json.dumps(values, allow_nan=False))
            self.contexts.append(context)
        except BaseException as error:
            self._latch(error)
            raise
        try:
            yield
        finally:
            self.contexts.pop()
            primary = sys.exception()
            if primary is not None and self.failure is not None:
                primary.add_note(f"Equation observer latched a failure: {self.failure}")
            else:
                self.check()

    def wrap(self, original, fallback_type):
        self.check()
        self.delegates += 1
        recorder, np = self, self.np

        class RecordedSolve:
            def __getattr__(self, name):
                return getattr(original, name)

            def __call__(self, matrix, gradient):
                recorder.check()
                try:
                    if len(recorder.records) >= recorder.max_records:
                        raise EquationObservationError("Original equation observation count cap exceeded")
                    arrays = (np.asarray(matrix), np.asarray(gradient))
                    for value, shape in zip(arrays, ((6, 6), (6,))):
                        if value.shape != shape or value.dtype != np.float64 or not value.flags.c_contiguous:
                            raise EquationObservationError("Expected original contiguous FP64 6x6 and six-gradient inputs")
                    before = tuple(digest(value) for value in arrays)
                    copied = tuple(np.array(value, copy=True, order="C") for value in arrays)
                    context = json.loads(json.dumps(recorder.contexts, allow_nan=False))
                except BaseException as error:
                    recorder._latch(error)
                    raise
                try:
                    result = original(matrix, gradient)
                except BaseException as primary:
                    try:
                        match = re.fullmatch(r"Eigen bridge rejected system \(status ([1-7])\)", str(primary))
                        code = int(match.group(1)) if isinstance(primary, fallback_type) and match else None
                        if before != tuple(digest(value) for value in arrays):
                            raise EquationObservationError("Original Eigen callable changed its equation inputs")
                        update = np.full((4, 4), np.nan, dtype=np.float64)
                        recorder.records.append({"matrix": copied[0], "gradient": copied[1], "update": update,
                            "status": code, "context": context, "matrix_sha256": before[0], "gradient_sha256": before[1],
                            "update_sha256": digest(update), "exception": {"type": type(primary).__name__, "message": str(primary)}})
                        if code is None:
                            raise EquationObservationError("Unexpected original Eigen exception has no authoritative status code")
                    except BaseException as observation_error:
                        recorder._latch(observation_error)
                        primary.add_note(f"Equation observation also failed: {type(observation_error).__name__}: {observation_error}")
                    raise
                try:
                    output = np.asarray(result)
                    if output.shape != (4, 4) or output.dtype != np.float64 or not output.flags.c_contiguous:
                        raise EquationObservationError("Original Eigen returned an incompatible update")
                    if not np.isfinite(output).all():
                        raise EquationObservationError("Accepted original Eigen update is nonfinite")
                    if before != tuple(digest(value) for value in arrays):
                        raise EquationObservationError("Original Eigen callable changed its equation inputs")
                    copied_output = np.array(output, copy=True, order="C")
                    recorder.records.append({"matrix": copied[0], "gradient": copied[1], "update": copied_output,
                        "status": 0, "context": context, "matrix_sha256": before[0], "gradient_sha256": before[1],
                        "update_sha256": digest(copied_output), "exception": None})
                except BaseException as error:
                    recorder._latch(error)
                    raise
                return result

        return RecordedSolve()

    def metadata(self):
        return {"kind": KIND, "count": len(self.records), "maximum_records": self.max_records,
            "maximum_retained_array_bytes": self.max_records * ARRAY_BYTES_PER_RECORD,
            "retained_array_bytes": len(self.records) * ARRAY_BYTES_PER_RECORD,
            "memory_scope": "Logical original equation/update/status copies only; JSON/Python objects, stacked export arrays, native libraries and allocator pools are additional process RSS. Export temporarily duplicates retained arrays.",
            "delegate_instances": self.delegates, "latched_failure": None if self.failure is None else
                {"type": type(self.failure).__name__, "message": str(self.failure)},
            "rows": [{"record_index": index, **{key: row[key] for key in
                ("status", "context", "matrix_sha256", "gradient_sha256", "update_sha256", "exception")}}
                for index, row in enumerate(self.records)],
            "previous_pose_scope": "Actual previous accumulated poses are not observed. A consumer may separately test synthetic or identity pose composition, without an actual trajectory-composition claim.",
            "policy": "One unchanged original Eigen callable invocation per row; exact original update object returned. No second DLL call, replacement solve, regularization or pose reconstruction."}

    def export(self, path):
        """Preserve partial copies even on failure; never overwrite an old file."""
        path = path.resolve()
        if path.exists():
            raise EquationObservationError("Require a fresh private equation array artifact")
        np = self.np
        rows = self.records
        matrices = np.stack([row["matrix"] for row in rows]) if rows else np.empty((0, 6, 6), dtype=np.float64)
        gradients = np.stack([row["gradient"] for row in rows]) if rows else np.empty((0, 6), dtype=np.float64)
        updates = np.stack([row["update"] for row in rows]) if rows else np.empty((0, 4, 4), dtype=np.float64)
        status = np.asarray([row["status"] if row["status"] is not None else -1 for row in rows], dtype=np.int32)
        for index, row in enumerate(rows):
            if (digest(matrices[index]), digest(gradients[index]), digest(updates[index])) != (
                    row["matrix_sha256"], row["gradient_sha256"], row["update_sha256"]):
                raise EquationObservationError("Retained observation copies changed before export")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            np.savez_compressed(target, matrices=matrices, gradients=gradients, cpu_update=updates, cpu_status=status)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
            "arrays": {"matrices": {"dtype": str(matrices.dtype), "shape": list(matrices.shape)},
                "gradients": {"dtype": str(gradients.dtype), "shape": list(gradients.shape)},
                "cpu_update": {"dtype": str(updates.dtype), "shape": list(updates.shape)},
                "cpu_status": {"dtype": str(status.dtype), "shape": list(status.shape)}}}
