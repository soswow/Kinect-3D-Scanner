"""Observe exact native ICP inputs; never cache or replace native results.

Source preparation only until an exclusive CPU fixture slot is allocated:
  python scripts/research_native_icp_signatures.py --run-pilot \
    --output benchmark-output/cuda-pipeline/native-signatures/chest-3.json -- \
    --fixture benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.fixture.pickle \
    --reference benchmark-output/cuda-pipeline/parallel-fragments/chest-3-dynamic.json \
    --threads 20 --output benchmark-output/cuda-pipeline/native-signatures/profile.json

Each call rehashes point/normal bytes; object identity is never a content key.
Original FP64 seed bytes, radius, exposed Huber/criteria state, current thread
policy and native binary identity are recorded. Opaque/custom estimators and
unsupported call shapes fail closed for repeat coverage but still execute the
original call. Result fingerprints expose nondeterminism among matching inputs.
Hashing/observation overhead is outside native spans but inside fixture wall
time, so this pilot cannot demonstrate a memoization speed gain.

Open3D v0.20.0 primary bindings expose PointToPlane.kernel, HuberLoss.k and
ICPConvergenceCriteria fields; repr strings are not used as kernel state:
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/pybind/pipelines/registration/registration.cpp
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/pybind/pipelines/registration/robust_kernels.cpp
"""

import argparse
import hashlib
import json
import os
import platform
import struct
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f64_bits(value):
    return struct.pack("<d", float(value)).hex()


class SignatureObserver:
    def __init__(self, numpy, open3d, cv2, max_records, max_record_bytes, input_binding):
        self.np, self.o3d, self.cv2 = numpy, open3d, cv2
        self.max_records, self.max_record_bytes = max_records, max_record_bytes
        self.records, self.retained_bytes, self.calls = [], 0, 0
        self.capture_s = self.result_capture_s = self.observer_s = 0.
        self.incomplete = Counter()
        self.script_before = file_hash(__file__)
        self.input_binding = input_binding
        class_module = open3d.geometry.PointCloud.__module__
        backend_name = class_module.split(".pybind")[0]+".pybind" if ".pybind" in class_module else None
        backend = sys.modules.get(backend_name)
        if backend is None or not getattr(backend, "__file__", None):
            raise RuntimeError("Native Open3D binary identity is unavailable")
        self.runtime = {"open3d": open3d.__version__, "numpy": numpy.__version__,
            "opencv": cv2.__version__, "python": sys.version,
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "native_binary": str(Path(backend.__file__).resolve()),
            "native_binary_sha256": file_hash(backend.__file__)}

    def threads(self):
        return {"open3d": int(self.o3d.utility.get_max_threads()),
            "opencv": int(self.cv2.getNumThreads()),
            "environment": {key: os.environ.get(key) for key in
                ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_SCHEDULE", "OMP_PROC_BIND",
                 "OMP_PLACES", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "KINECT_CUDA_REGISTRATION")}}

    def array(self, value, *, matrix=False):
        np = self.np
        value = np.asarray(value)
        if (value.dtype != np.float64 or (matrix and value.shape != (4, 4))
                or (not matrix and (value.ndim != 2 or value.shape[1] != 3))
                or value.nbytes > 32*1024**2 or not np.isfinite(value).all()):
            raise ValueError("Unsupported original FP64 array shape, extent or coordinates")
        data = np.ascontiguousarray(value).tobytes()
        result = {"shape": list(value.shape), "dtype": value.dtype.str,
                  "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if matrix:
            result["original_bytes_hex"] = data.hex()
        return result

    def cloud(self, value):
        if type(value) is not self.o3d.geometry.PointCloud:
            raise ValueError("Custom or opaque cloud class")
        return {name: self.array(getattr(value, name)) for name in ("points", "normals")}

    def signature(self, args, kwargs):
        # All current native calls use explicit arguments. Do not reconstruct
        # hidden/default state or claim coverage for an unobserved parameter.
        names = ("source", "target", "max_correspondence_distance", "init", "estimation_method", "criteria")
        if len(args) > len(names) or set(kwargs)-set(names):
            raise ValueError("Unsupported native argument layout")
        values = dict(zip(names, args))
        if set(values) & set(kwargs):
            raise ValueError("Duplicate positional/keyword argument")
        values.update(kwargs)
        if set(values) != set(names):
            raise ValueError("Unobserved default native argument")
        reg = self.o3d.pipelines.registration
        estimator, criteria = values["estimation_method"], values["criteria"]
        if type(estimator) is not reg.TransformationEstimationPointToPlane or type(criteria) is not reg.ICPConvergenceCriteria:
            raise ValueError("Opaque or unsupported estimator/convergence class")
        kernel = estimator.kernel
        if type(kernel) is not reg.HuberLoss:
            raise ValueError("Opaque or unsupported robust kernel")
        return {"source": self.cloud(values["source"]), "target": self.cloud(values["target"]),
            "initial": self.array(values["init"], matrix=True),
            "radius_fp64": f64_bits(values["max_correspondence_distance"]),
            "estimator": {"type": "TransformationEstimationPointToPlane", "kernel": "HuberLoss", "k_fp64": f64_bits(kernel.k)},
            "criteria": {"max_iteration": int(criteria.max_iteration),
                "relative_fitness_fp64": f64_bits(criteria.relative_fitness), "relative_rmse_fp64": f64_bits(criteria.relative_rmse)},
            "threads": self.threads(), "runtime": self.runtime}

    def result(self, value):
        pairs = self.np.asarray(value.correspondence_set)
        return {"transform": self.array(value.transformation, matrix=True),
            "fitness_fp64": f64_bits(value.fitness), "rmse_fp64": f64_bits(value.inlier_rmse),
            "correspondences": {"shape": list(pairs.shape), "dtype": pairs.dtype.str,
                "sha256": hashlib.sha256(self.np.ascontiguousarray(pairs).tobytes()).hexdigest()}}

    def observe(self, original, recorder, args, kwargs):
        observed_started = time.perf_counter()
        self.calls += 1
        row = {"call": self.calls, "phase": recorder.phase, "pair_position": recorder.pair_position,
               "proposal_index": recorder.proposal_index, "branch_path": recorder.scope()}
        signature = result = None
        started = time.perf_counter()
        try:
            signature = self.signature(args, kwargs)
            row["signature"] = signature
            row["signature_sha256"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        except Exception as error:
            row["incomplete_reason"] = str(error)
            self.incomplete[str(error)] += 1
        self.capture_s += time.perf_counter()-started
        before_events = len(recorder.events)
        native_started = time.perf_counter()
        try:
            result = original(*args, **kwargs)  # Always invoke original native call.
            return result
        except BaseException as error:
            row["native_error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            native_wrapper_s = time.perf_counter()-native_started
            if len(recorder.events) > before_events and recorder.events[-1]["name"] == "native.registration_icp":
                row["native_event_id"] = recorder.events[-1]["event_id"]
                row["native_s"] = recorder.events[-1]["inclusive_s"]
            started = time.perf_counter()
            if signature is not None and result is not None:
                try:
                    row["inputs_unchanged"] = signature == self.signature(args, kwargs)
                    if not row["inputs_unchanged"]:
                        self.incomplete["Inputs changed during native call"] += 1
                    row["result"] = self.result(result)
                    row["result_sha256"] = hashlib.sha256(json.dumps(row["result"], sort_keys=True).encode()).hexdigest()
                except Exception as error:
                    row["incomplete_reason"] = str(error)
                    self.incomplete[str(error)] += 1
            self.result_capture_s += time.perf_counter()-started
            serialized = len(json.dumps(row).encode())
            if len(self.records) < self.max_records and self.retained_bytes+serialized <= self.max_record_bytes:
                self.records.append(row)
                self.retained_bytes += serialized
            else:
                self.incomplete["Record extent exceeded"] += 1
            self.observer_s += time.perf_counter()-observed_started-native_wrapper_s

    def save(self, output, native_report=None, failure=None):
        native_binding = None
        native_verified = False
        expected_events = set()
        if native_report is not None:
            # One exact byte snapshot binds status, scope and events to the hash.
            payload = Path(native_report).read_bytes()
            native = json.loads(payload)
            metadata, quality = native["metadata"], native["quality"]
            from scripts import profile_fragment_verification as profiler
            expected_events = {row["event_id"] for row in native["events"]
                               if row["name"] == "native.registration_icp"}
            recorded_events = [row.get("native_event_id") for row in self.records]
            expected_order = metadata["fixture_metadata"]["fixture_pair_order"]
            actual_order = [row["pair"] for row in native["rows"]]
            native_verified = (native["status"] == "passed" and
                all(quality.get(key) is True for key in profiler.QUALITY_GATES) and
                metadata["source_sha256"] == profiler.FROZEN_SOURCE_SHA256 == metadata["source_sha256_after"] and
                actual_order == expected_order and len(actual_order) == 8 and
                len(recorded_events) == len(expected_events) == self.calls and
                len(set(recorded_events)) == len(recorded_events) and
                set(recorded_events) == expected_events and
                all(row.get("phase") == "fixed_verification" for row in self.records))
            native_binding = {"path": str(native_report), "sha256": hashlib.sha256(payload).hexdigest(),
                "status": native["status"], "quality": quality,
                "source_sha256": metadata["source_sha256"],
                "component_source_sha256": metadata["component_source_sha256"],
                "profiler_script_sha256": metadata["script_sha256"],
                "fixture_sha256": metadata["fixture_sha256"],
                "reference_sha256": metadata["reference_sha256"],
                "pair_order": actual_order, "native_icp_event_count": len(expected_events),
                "full_eight_pair_event_coverage": native_verified}
        groups = {}
        for row in self.records:
            if not row.get("inputs_unchanged") or not row.get("result_sha256"):
                continue
            item = groups.setdefault(row["signature_sha256"], {"calls": 0, "native_s": 0., "results": Counter()})
            item["calls"] += 1
            item["native_s"] += row.get("native_s", 0.)
            item["results"][row["result_sha256"]] += 1
        repeats = [{"signature_sha256": key, "calls": item["calls"], "native_s": item["native_s"],
                    "result_variants": dict(item["results"])} for key, item in groups.items() if item["calls"] > 1]
        script_after = file_hash(__file__)
        binary_unchanged = file_hash(self.runtime["native_binary"]) == self.runtime["native_binary_sha256"]
        input_after = {key: file_hash(item["path"]) for key, item in self.input_binding.items()}
        input_unchanged = all(input_after[key] == item["sha256"] for key, item in self.input_binding.items())
        complete_calls = sum(bool(row.get("inputs_unchanged") and row.get("result_sha256")) for row in self.records)
        complete = (self.calls > 0 and not self.incomplete and complete_calls == self.calls
                    and self.script_before == script_after and binary_unchanged and input_unchanged
                    and failure is None and native_verified)
        report = {"kind": "observation-only-exact-native-icp-signatures", "executed_native_calls": self.calls,
            "status": "complete_observation" if complete else "incomplete_observation",
            "recorded_calls": len(self.records), "signature_coverage_complete": complete,
            "incomplete_reasons": dict(self.incomplete), "observer_script_sha256": self.script_before,
            "observer_script_sha256_after": script_after, "observer_script_unchanged": self.script_before==script_after,
            "native_binary_unchanged": binary_unchanged,
            "input_binding": self.input_binding, "input_sha256_after": input_after,
            "input_artifacts_unchanged": input_unchanged,
            "runtime": self.runtime, "capture_s": self.capture_s, "result_capture_and_after_hash_s": self.result_capture_s,
            "observer_wall_outside_original_native_wrapper_s": self.observer_s,
            "native_report": native_binding,
            "failure": failure, "repeated_signatures": repeats, "records": self.records,
            "authority": "Original native calls always execute. Exposed input/runtime signatures establish candidate repeat coverage only; native result variability is explicit. No cache, speed gain, full-session decision proof or hidden-state determinism claim.",
            "limits": {"max_records": self.max_records, "max_record_bytes": self.max_record_bytes,
                "retained_serialized_bytes": self.retained_bytes, "per_array_bytes": 32*1024**2}}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pilot", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument("--max-record-mib", type=int, default=32)
    parser.add_argument("profile_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.run_pilot or min(args.max_records, args.max_record_mib) < 1:
        parser.error("Require --run-pilot in an allocated slot and positive record bounds")
    arguments = args.profile_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        parser.error("Pass original profile_fragment_verification.py arguments after --")
    from scripts import profile_fragment_verification as profiler
    from scripts.profile_session import source_hash
    if source_hash() != profiler.FROZEN_SOURCE_SHA256:
        raise RuntimeError("Require frozen production source before numerical imports")
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--fixture", type=Path, required=True)
    profile_parser.add_argument("--reference", type=Path, required=True)
    profile_parser.add_argument("--output", type=Path, required=True)
    profile_paths, _ = profile_parser.parse_known_args(arguments)
    if args.output.exists() or profile_paths.output.exists() or args.output.resolve() == profile_paths.output.resolve():
        raise RuntimeError("Observation and native report require distinct fresh output paths")
    reference_bytes = profile_paths.reference.read_bytes()
    reference_metadata = json.loads(reference_bytes)["metadata"]
    if reference_metadata.get("local_pose_source") != "measured Finish fragment report; no archived ZIP poses":
        raise RuntimeError("Require original measured-Finish fixture provenance")
    raw_path = Path(reference_metadata["session"])
    raw_sha256 = file_hash(raw_path)
    fixture_sha256 = file_hash(profile_paths.fixture)
    if raw_sha256 != reference_metadata["input_sha256"] or fixture_sha256 != reference_metadata["fixture_sha256"]:
        raise RuntimeError("Raw ZIP or original fixture differs from the reference")
    input_binding = {
        "raw_input": {"path": str(raw_path.resolve()), "sha256": raw_sha256},
        "fixture": {"path": str(profile_paths.fixture.resolve()), "sha256": fixture_sha256},
        "reference": {"path": str(profile_paths.reference.resolve()), "sha256": hashlib.sha256(reference_bytes).hexdigest()},
        "profiler": {"path": str(Path(profiler.__file__).resolve()), "sha256": file_hash(profiler.__file__)},
        "fixture_loader": {"path": str(ROOT/"scripts/benchmark_parallel_fragments.py"),
                           "sha256": file_hash(ROOT/"scripts/benchmark_parallel_fragments.py")}}
    import numpy as np
    import open3d as o3d
    import cv2
    observer = SignatureObserver(np, o3d, cv2, args.max_records, args.max_record_mib*1024**2, input_binding)
    original_proxy, original_writer, original_argv = profiler.RegistrationProxy, profiler.write_compact_summary, sys.argv
    saved = False

    class ObservedProxy(original_proxy):
        def __getattr__(self, name):
            original = super().__getattr__(name)
            if name != "registration_icp":
                return original
            return lambda *a, **kw: observer.observe(original, self.recorder, a, kw)

    def write_both(report, native_path):
        nonlocal saved
        result = original_writer(report, native_path)
        complete = observer.save(args.output, native_path)
        saved = True
        if not complete:
            raise RuntimeError("Exact native signature coverage or provenance is incomplete; observation report preserved")
        print(f"Exact native-call signatures: {args.output}", flush=True)
        return result

    try:
        profiler.RegistrationProxy, profiler.write_compact_summary = ObservedProxy, write_both
        sys.argv = [str(Path(profiler.__file__)), *arguments]
        # The existing Windows worker exits after its closed report. Save the
        # signature report in its normal writer hook, before that exact exit.
        profiler.main()
    except BaseException as error:
        if not saved:
            observer.save(args.output, failure={"type": type(error).__name__, "message": str(error)})
            saved = True
        raise
    finally:
        profiler.RegistrationProxy, profiler.write_compact_summary, sys.argv = original_proxy, original_writer, original_argv
        if not saved:
            observer.save(args.output, failure={"type": "IncompletePilot", "message": "Original profile did not close a report"})


if __name__ == "__main__":
    main()
