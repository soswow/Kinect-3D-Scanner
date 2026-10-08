"""Meaningful stdlib-only observer contracts; no NumPy/native/CUDA imports.

The small array stand-in tests delegation, copied ownership, latching, context
order, count/domain failures and authoritative status extraction. It does not
establish numerical or NPZ interoperability proof; those require allocation.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.resident_equation_recorder import EquationRecorder, EquationObservationError, restore_owned_hooks


class Array:
    def __init__(self, values, shape, dtype="float64", contiguous=True):
        self.values, self.shape, self.dtype = list(values), tuple(shape), dtype
        self.flags = SimpleNamespace(c_contiguous=contiguous)

    def tobytes(self, order="C"):
        return struct.pack("<"+"d"*len(self.values), *self.values)


class NP:
    float64 = "float64"
    int32 = "int32"

    @staticmethod
    def asarray(value):
        return value

    @staticmethod
    def array(value, **kwargs):
        return Array(value.values, value.shape, value.dtype, value.flags.c_contiguous)

    @staticmethod
    def full(shape, value, dtype):
        return Array([value]*math.prod(shape), shape, dtype)

    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda: all(math.isfinite(item) for item in value.values))

    nan = math.nan


class Fallback(RuntimeError):
    pass


class Original:
    def __init__(self, *, action=None):
        self.calls = 0
        self.metadata = {"original": True}
        self.action = action
        self.output = Array([float(i % 5 == 0) for i in range(16)], (4, 4))

    def __call__(self, matrix, gradient):
        self.calls += 1
        if self.action:
            self.action(matrix, gradient, self.output)
        return self.output


def inputs():
    return Array([float(i % 7 == 0) for i in range(36)], (6, 6)), Array([0.]*6, (6,))


def fail(action, expected=Exception):
    try:
        action()
    except expected:
        return
    raise AssertionError("Expected a rejected observer contract")


def contracts():
    results = []

    def case(name, action):
        action()
        results.append({"name": name, "passed": True})

    def delegation():
        recorder, original = EquationRecorder(NP), Original()
        wrapped = recorder.wrap(original, Fallback)
        matrix, gradient = inputs()
        assert wrapped.metadata is original.metadata
        result = wrapped(matrix, gradient)
        assert result is original.output and original.calls == 1 and len(recorder.records) == 1
        assert recorder.records[0]["matrix"] is not matrix and recorder.records[0]["update"] is not result
        result.values[0] = 9.
        matrix.values[0] = 8.
        assert recorder.records[0]["matrix"].values[0] == 1. and recorder.records[0]["update"].values[0] == 1.
        assert recorder.metadata()["rows"][0]["status"] == 0
    case("exactly_one_original_call_and_same_output_object_private_copies", delegation)

    def context():
        recorder, original = EquationRecorder(NP), Original()
        wrapped = recorder.wrap(original, Fallback)
        pair = [8, 12]
        with recorder.scope(pair=pair):
            pair[0] = 99
            with recorder.scope(match_index=0):
                wrapped(*inputs())
        assert recorder.records[0]["context"] == [{"pair": [8, 12]}, {"match_index": 0}]
        assert recorder.contexts == []
    case("nested_context_frozen_and_restored", context)

    def invalid_context():
        recorder = EquationRecorder(NP)
        def invalid():
            with recorder.scope(unserializable=object()):
                pass
        fail(invalid, TypeError)
        assert recorder.failure is not None and recorder.contexts == []
        fail(recorder.check, EquationObservationError)
    case("context_freeze_failure_is_latched_before_scope_entry", invalid_context)

    def partial_restore():
        good = SimpleNamespace(value="patched")
        argv_owner = SimpleNamespace(argv=["patched"])
        class Broken:
            def __setattr__(self, name, value):
                raise RuntimeError("injected restoration failure")
        hooks = [(good, "value", "original"), (Broken(), "value", "original")]
        errors = restore_owned_hooks(hooks, ["original"], argv_owner=argv_owner)
        assert len(errors) == 1 and good.value == "original" and argv_owner.argv == ["original"] and hooks == []
    case("failed_hook_restore_does_not_skip_other_hooks_or_argv", partial_restore)

    def capacity():
        recorder, original = EquationRecorder(NP, max_records=1), Original()
        wrapped = recorder.wrap(original, Fallback)
        wrapped(*inputs())
        fail(lambda: wrapped(*inputs()), EquationObservationError)
        assert original.calls == 1 and recorder.failure is not None
        fail(recorder.check, EquationObservationError)
    case("bounded_capacity_stops_before_second_original_call_and_latches", capacity)

    for name, mutate in (("shape", lambda a: setattr(a, "shape", (36,))),
                         ("dtype", lambda a: setattr(a, "dtype", "float32")),
                         ("contiguity", lambda a: setattr(a.flags, "c_contiguous", False))):
        def bad_input(mutate=mutate):
            recorder, original = EquationRecorder(NP), Original()
            wrapped = recorder.wrap(original, Fallback)
            matrix, gradient = inputs(); mutate(matrix)
            fail(lambda: wrapped(matrix, gradient), EquationObservationError)
            assert original.calls == 0 and recorder.failure is not None
        case("invalid_original_"+name+"_before_delegate", bad_input)

    def mutation():
        original = Original(action=lambda a, b, result: a.values.__setitem__(0, 5.))
        recorder = EquationRecorder(NP);wrapped = recorder.wrap(original, Fallback)
        fail(lambda: wrapped(*inputs()), EquationObservationError)
        assert recorder.failure is not None and original.calls == 1
        fail(lambda: wrapped(*inputs()), EquationObservationError)
        assert original.calls == 1
    case("original_input_mutation_is_hard_latched", mutation)

    def output_nonfinite():
        original = Original(action=lambda a, b, result: result.values.__setitem__(0, math.inf))
        recorder = EquationRecorder(NP);wrapped = recorder.wrap(original, Fallback)
        fail(lambda: wrapped(*inputs()), EquationObservationError)
        assert recorder.failure is not None
    case("nonfinite_accepted_original_update_rejected", output_nonfinite)

    def expected_status():
        def reject(*_):
            raise Fallback("Eigen bridge rejected system (status 4)")
        recorder, original = EquationRecorder(NP), Original(action=reject)
        wrapped = recorder.wrap(original, Fallback)
        fail(lambda: wrapped(*inputs()), Fallback)
        assert original.calls == 1 and recorder.failure is None
        assert recorder.records[0]["status"] == 4 and recorder.records[0]["exception"]["type"] == "Fallback"
    case("original_rejection_status_parsed_without_second_dll_call", expected_status)

    for message in ("Eigen bridge rejected system (status 0)", "changed exception", "Eigen bridge rejected system (status 44)"):
        def unexpected_status(message=message):
            def reject(*_):
                raise Fallback(message)
            recorder, original = EquationRecorder(NP), Original(action=reject)
            fail(lambda: recorder.wrap(original, Fallback)(*inputs()), Fallback)
            assert recorder.failure is not None and recorder.records[0]["status"] is None
            fail(recorder.check, EquationObservationError)
        case("unknown_status_hard_latched_"+message, unexpected_status)

    def suppression():
        recorder, original = EquationRecorder(NP, max_records=1), Original()
        wrapped = recorder.wrap(original, Fallback);wrapped(*inputs())
        def suppressed():
            with recorder.scope(proposal=0):
                try:
                    wrapped(*inputs())
                except Exception:
                    pass
        fail(suppressed, EquationObservationError)
        assert recorder.contexts == []
    case("scope_exit_detects_suppressed_observation_failure", suppression)

    def preserve_primary():
        recorder = EquationRecorder(NP)
        primary = ValueError("primary remains")
        def body():
            with recorder.scope(proposal=0):
                recorder._latch(RuntimeError("observer failure"))
                raise primary
        try:
            body()
        except ValueError as error:
            assert error is primary and error.__notes__
        else:
            raise AssertionError("Expected original primary")
        assert recorder.contexts == []
    case("scope_failure_preserves_active_primary_and_adds_observer_note", preserve_primary)

    for value in (0, -1, True, 20001, 1.5):
        case("invalid_count_cap_"+str(value), lambda value=value: fail(lambda: EquationRecorder(NP, max_records=value), ValueError))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Require a fresh stdlib contract output")
    for path in (Path(__file__).with_name("resident_equation_recorder.py"), Path(__file__).with_name("capture_resident_equations.py")):
        ast.parse(path.read_text(encoding="utf-8"))
    rows = contracts()
    report = {"kind": "stdlib-resident-equation-observer-contracts", "status": "passed", "cases": rows,
        "artifact_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(__file__).with_name("resident_equation_recorder.py"), Path(__file__).with_name("capture_resident_equations.py"))},
        "scope": "Stdlib fake-array ownership/delegation/failure contracts only; no numerical/native/NPZ interoperability claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "cases": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
