"""Strict stdlib-only input guard for an observed Eigen micro-solve fixture.

An accepted fixture authorizes only comparison of its unchanged 6x6 systems.
It does not authorize different NN, ICP, convergence or field trajectories.
Numeric imports, GPU work and artifact writes are deliberately absent.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.validate_device_flat_grid_proof import (
    FROZEN, canonical_hash, current_core_hash, normalized_artifacts, resident_checks, validate_grid_proof)

KIND = "observed-original-resident-eigen-equations-v1"
MAX_RECORDS = 20_000
MAX_JSON_BYTES = 64 * 1024**2
ARTIFACTS = (
    "scripts/research/capture_resident_equations.py",
    "scripts/research/resident_equation_recorder.py",
    "scripts/research/check_resident_equation_recorder.py",
    "scripts/profile_session.py", "scripts/process_metrics.py")


class EquationFixtureError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EquationFixtureError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def contained(path, *, private=False):
    value = Path(path).resolve(strict=True)
    require(value.is_relative_to((ROOT / "benchmark-output" if private else ROOT).resolve()),
            "Equation artifact escapes its declared workspace")
    require(value.is_file(), "Equation artifact is not a regular file")
    return value


def record_path(record, *, private=True):
    require(isinstance(record, dict) and digest(record.get("sha256")), "Malformed closed artifact record")
    value = contained(record["path"], private=private)
    require(sha(value) == record["sha256"], "Closed equation artifact changed: " + str(value))
    return value


def read_json(path):
    require(0 < path.stat().st_size <= MAX_JSON_BYTES, "Equation JSON exceeds the bounded input scope")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(
        EquationFixtureError("Nonfinite equation JSON is unsupported")))
    require(isinstance(value, dict), "Equation JSON must be an object")
    return value


def read_numeric_npz(path, count):
    """Validate contained bounded NPY headers and return original numeric bits."""
    layouts = {"matrices": ("<f8", (count, 6, 6), 288),
               "gradients": ("<f8", (count, 6), 48),
               "cpu_update": ("<f8", (count, 4, 4), 128),
               "cpu_status": ("<i4", (count,), 4)}
    require(type(count) is int and 0 < count <= MAX_RECORDS, "Invalid original equation row count")
    require(0 < path.stat().st_size <= MAX_RECORDS * 468 + 64 * 1024,
            "Equation compressed artifact exceeds its byte cap")
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == 4 and {entry.filename for entry in entries} == {name + ".npy" for name in layouts},
                "Equation NPZ must contain exactly four original numeric arrays")
        for entry in entries:
            name = entry.filename[:-4]
            dtype, shape, bytes_per_row = layouts[name]
            require(entry.file_size <= count * bytes_per_row + 4096 and not entry.flag_bits & 1,
                    "Equation NPY descriptor exceeds its byte cap")
            raw = archive.read(entry)
            require(raw[:6] == b"\x93NUMPY" and raw[6:8] in (b"\x01\x00", b"\x02\x00"),
                    "Require an ordinary version1/2 numeric NPY")
            version = raw[6]
            size = 2 if version == 1 else 4
            require(len(raw) >= 8 + size, "Truncated equation NPY header")
            header_size = int.from_bytes(raw[8:8 + size], "little")
            require(0 < header_size <= 4096, "Equation NPY header exceeds its cap")
            start = 8 + size + header_size
            require(start <= len(raw), "Truncated equation NPY descriptor")
            header = ast.literal_eval(raw[8 + size:start].decode("latin1"))
            require(header == {"descr": dtype, "fortran_order": False, "shape": shape},
                    "Equation NPY dtype/shape/order differs from original recorded scope")
            bits = raw[start:]
            require(len(bits) == count * bytes_per_row, "Equation NPY payload length differs")
            arrays[name] = bits
    return arrays


@dataclass(frozen=True)
class ResidentEquationFixture:
    manifest_path: str
    manifest_sha256: str
    equation_path: str
    equation_sha256: str
    count: int
    manifest_json: str
    dependent_files: tuple[tuple[str, str], ...]

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    def check_unchanged(self):
        require(sha(self.manifest_path) == self.manifest_sha256 and sha(self.equation_path) == self.equation_sha256,
                "Observed equation fixture changed during the micro-experiment")
        require(current_core_hash() == FROZEN, "Original production source changed during micro-experiment")
        for path, fingerprint in self.dependent_files:
            require(sha(path) == fingerprint, "Equation dependency changed during micro-experiment: " + path)


def validate_equation_fixture(path, *, expected_solver_path):
    """Revalidate current component authority and every original equation bit row."""
    try:
        path = contained(path, private=True)
        before = sha(path)
        value = read_json(path)
        require(value.get("kind") == KIND and value.get("status") == "passed"
                and value.get("performance_attribution_valid") is False
                and value.get("new_solve_authority") is False and value.get("previous_pose_is_actual") is False,
                "Require a closed output-only original equation observation")
        require(not value.get("failure") and value.get("closure_failures") == []
                and value.get("observer_hooks_restored") is True and value.get("hook_cleanup_passed") is True,
                "Equation observation/cleanup did not close")
        require(value.get("hook_cleanup_failures") == [], "Equation hook cleanup failure was hidden")
        require(value.get("source_sha256") == value.get("source_sha256_after") == FROZEN == current_core_hash(),
                "Original equation production source changed")
        require(value.get("component_source_sha256") == value.get("component_source_sha256_after")
                and digest(value.get("component_source_sha256")), "Original component dependency source changed")
        sources = value["artifacts_sha256"]
        require(set(sources) == set(ARTIFACTS) and value.get("artifacts_sha256_after") == sources,
                "Equation observer/producer source bindings incomplete")
        for name, fingerprint in sources.items():
            require(digest(fingerprint) and sha(contained(ROOT / name)) == fingerprint,
                    "Current equation observer dependency changed: " + name)
        fixed = value["fixed_files_sha256"]
        require(isinstance(fixed, dict) and fixed and value.get("fixed_files_sha256_after") == fixed,
                "Original proof/fixture/solve files changed")
        for name, fingerprint in fixed.items():
            require(digest(fingerprint) and sha(contained(name)) == fingerprint, "Original fixed file changed: " + name)
        proofs = value["component_proofs"]
        synthetic, bridge = record_path(proofs["synthetic"]), record_path(proofs["bridge"])
        require(proofs["synthetic"]["sha256"] == value["proof_synthetic_sha256"]
                and proofs["bridge"]["sha256"] == value["proof_bridge_sha256"], "Original component proof records disagree")
        binding, fixture = value["proof_bindings"], value["fixture_binding"]
        authority = validate_grid_proof(synthetic, bridge, binding, fixture)
        require(authority.fixture_binding_sha256 == value["fixture_binding_sha256"] == canonical_hash(fixture),
                "Original raw-derived component fixture differs")
        require(value.get("actual_runtime_binding") == value.get("actual_runtime_binding_after") == binding,
                "Original actual runtime/configuration drifted")
        solver = contained(expected_solver_path)
        require(sha(solver) == binding["resident_math"]["solve_library_sha256"]
                and fixed.get(str(solver)) == sha(solver), "Micro-experiment Eigen DLL differs from observed original")
        component_path = record_path(value["component_report"])
        component = read_json(component_path)
        require(component.get("status") == "passed" and component.get("cleanup_passed") is True
                and component.get("fixture_arrays_unchanged") is True and component.get("cloud_arrays_unchanged") is True
                and component.get("fixture_binding") == fixture
                and component.get("proof_bindings") == component.get("proof_bindings_after") == binding,
                "Observed original nine-proposal component did not pass its unchanged authority")
        require(not any(component.get(field) for field in ("failure", "producer_current_failure",
                "timing_driver_current_failure", "cleanup_failures")), "Observed original component fault was hidden")
        require(value.get("solve_metadata") == component.get("solve_metadata")
                and isinstance(value.get("solve_metadata"), dict), "Actual original loaded Eigen metadata differs")
        native = [row for row in component["real_runs"] if row.get("mode") == "native_cpu"]
        gpu = [row for row in component["real_runs"] if row.get("mode") == "grid"]
        require(len(native) == len(gpu) == 1 and len(component["real_runs"]) == 2,
                "Require exactly one original native and resident observation pass")
        resident_checks(gpu[0], binding, normalized_artifacts(binding["artifacts_sha256"]))
        for run in (*native, *gpu):
            require(run.get("complete") is True and len(run.get("pairs", [])) == len(fixture["tasks"]),
                    "Observed original component pair coverage incomplete")
            for pair, task in zip(run["pairs"], fixture["tasks"]):
                rows = pair["proposal_results"]
                require(pair["position"] == task["position"] and pair["pair"] == task["pair"]
                        and pair.get("pair_verdict_same") is True
                        and [row["proposal_index"] for row in rows] == list(range(len(task["proposal_sha256"])))
                        and [row["input_sha256"] for row in rows] == task["proposal_sha256"],
                        "Observed original proposal order/bytes/gates changed")
                if run is gpu[0]:
                    require(all(row.get("quality", {}).get("passed") is True for row in rows),
                            "Observed original witness/pose/information gates failed")
        recorder = value["recorder"]
        count = recorder["count"]
        require(type(count) is int and 0 < count <= MAX_RECORDS and recorder.get("latched_failure") is None
                and recorder.get("delegate_instances") == 1 and len(recorder.get("rows", [])) == count,
                "Original equation observation coverage/failure scope incomplete")
        counts = value["actual_component_counts"]
        require(counts["actual_solve_calls"] == count <= counts["resident_pose_iterations"]
                and counts["resident_pose_iterations"] == gpu[0]["resident_statistics_delta"]["pose_iterations"]
                and counts["resident_calls"] == gpu[0]["resident_statistics_delta"]["calls"]
                and counts["query_rows"] == gpu[0]["statistics_delta"]["query_rows"]
                and counts["nn_device_calls"] == gpu[0]["statistics_delta"]["device_calls"],
                "Original equation/trajectory count mismatch")
        expected = [(task["pair"], index, fingerprint) for task in fixture["tasks"]
                    for index, fingerprint in enumerate(task["proposal_sha256"])]
        for mode in ("native_cpu", "grid"):
            scopes = [scope for scope in value["proposal_scopes"] if scope.get("mode") == mode]
            require(len(scopes) == len(expected) and all(scope.get("completed") is True for scope in scopes)
                    and [scope["sequence"] for scope in scopes] == list(range(len(expected)))
                    and [(scope["pair"], scope["proposal_index"], scope["input_sha256"]) for scope in scopes] == expected,
                    "Observed original proposal context/order changed")
        require(len(value["proposal_scopes"]) == 2 * len(expected), "Unexpected observed proposal mode")
        end = 0
        for index, scope in enumerate(value["match_scopes"]):
            require(scope.get("completed") is True and scope["match_index"] == index
                    and type(scope["first_record"]) is int and type(scope["end_record"]) is int
                    and scope["first_record"] == end <= scope["end_record"] <= count,
                    "Observed original match record coverage changed")
            end = scope["end_record"]
        require(end == count and len(value["match_scopes"]) == counts["resident_calls"],
                "Observed original match/solve coverage incomplete")
        equation_path = record_path(value["equation_artifact"])
        require(value["equation_artifact"]["bytes"] == equation_path.stat().st_size,
                "Original equation NPZ byte descriptor changed")
        arrays = read_numeric_npz(equation_path, count)
        expected_layout = {"matrices": {"dtype": "float64", "shape": [count, 6, 6]},
                           "gradients": {"dtype": "float64", "shape": [count, 6]},
                           "cpu_update": {"dtype": "float64", "shape": [count, 4, 4]},
                           "cpu_status": {"dtype": "int32", "shape": [count]}}
        require(value["equation_artifact"]["arrays"] == expected_layout, "Original numeric array descriptors changed")
        for index, row in enumerate(recorder["rows"]):
            require(row["record_index"] == index and row["status"] == 0 and row.get("exception") is None
                    and struct.unpack_from("<i", arrays["cpu_status"], index * 4)[0] == 0,
                    "Require original accepted statuses for this micro-fixture")
            for name, width, field in (("matrices", 288, "matrix_sha256"),
                                      ("gradients", 48, "gradient_sha256"), ("cpu_update", 128, "update_sha256")):
                bits = arrays[name][index * width:(index + 1) * width]
                require(hashlib.sha256(bits).hexdigest() == row[field], "Original equation/update bits changed")
            context = row["context"]
            require(isinstance(context, list) and len(context) == 2 and context[0].get("mode") == "grid"
                    and type(context[1].get("match_index")) is int,
                    "Original equation lacks actual proposal/match context")
            scope = value["match_scopes"][context[1]["match_index"]]
            require(scope["match_index"] == context[1]["match_index"] and scope.get("completed") is True
                    and scope["first_record"] <= index < scope["end_record"], "Original equation match membership changed")
            require(context[1] == {name: scope[name] for name in
                    ("match_index", "source", "target", "initial_sha256")}, "Original equation match inputs changed")
            proposals = [scope for scope in value["proposal_scopes"] if scope["mode"] == "grid"]
            proposal = proposals[context[0]["sequence"]]
            require(context[0] == {name: proposal[name] for name in
                    ("mode", "sequence", "pair", "proposal_index", "input_sha256")}
                    and proposal["first_record"] <= index < proposal["end_record"],
                    "Original equation proposal/seed membership changed")
        require(sha(path) == before and sha(equation_path) == value["equation_artifact"]["sha256"],
                "Observed equation fixture changed during validation")
        dependencies = {str(contained(name)): fingerprint for name, fingerprint in fixed.items()}
        dependencies.update({str(contained(ROOT / name)): fingerprint for name, fingerprint in sources.items()})
        dependencies.update({str(contained(ROOT / name)): fingerprint
                             for name, fingerprint in authority.artifact_sha256})
        dependencies[str(component_path)] = value["component_report"]["sha256"]
        return ResidentEquationFixture(str(path), before, str(equation_path), sha(equation_path), count,
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
                tuple(sorted(dependencies.items())))
    except EquationFixtureError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, OSError, struct.error, zipfile.BadZipFile) as error:
        raise EquationFixtureError("Malformed/stale original equation fixture: " + str(error)) from error
