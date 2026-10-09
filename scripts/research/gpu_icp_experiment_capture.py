"""Capture bounded genuine ICP seed inputs from a current unseeded field replay.

Delegates the unchanged original raw/calibrated fragment preparation. Reported
fragment-local/live poses only reconstruct component inputs; they never seed a
new live scan. This fixture grants no numerical, graph, mesh or speed authority.
Private arrays and original ZIPs remain under ignored benchmark-output/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
KIND = "gpu-icp-current-field-pair-fixture-v2"
CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"
ARTIFACTS = ("scripts/research/gpu_icp_experiment_capture.py",
    "scripts/research/benchmark_parallel_fragments.py", "scripts/profile_session.py",
    "scripts/process_metrics.py", "scripts/tool_paths.py", "scripts/tool-catalog.json")
ENVIRONMENT = {"OMP_NUM_THREADS": "8", "KINECT_NATIVE": "on", "KINECT_CUDA_REGISTRATION": "cpu"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source_hash():
    h = hashlib.sha256()
    for folder in (ROOT / "scanner_server", ROOT / "shared", ROOT / "native"):
        for path in sorted(folder.rglob("*")):
            if path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".toml") and "build" not in path.parts:
                h.update(str(path.relative_to(ROOT)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def profile_contract(profile, session, actual_source, actual_raw):
    if (actual_source != CURRENT or profile.get("source_sha256") != actual_source
            or profile.get("input_sha256") != actual_raw
            or Path(profile.get("session", "")).name != Path(session).name
            or profile.get("source_changed_during_profile") is not False
            or profile.get("input_changed_during_profile") is not False
            or profile.get("pose_seeds_used") is not False
            or profile.get("finish_requested") is not True or profile.get("mesh_built") is not True
            or profile.get("experimental_visual_fallback") is not False
            or profile.get("native_mode") != "on" or profile.get("omp_threads") != "8"):
        raise ValueError("Require exact current-source completed unseeded full raw field replay")
    indices = profile.get("selected_indices")
    if (not isinstance(indices, list) or not indices or indices != list(range(len(indices)))
            or any(type(i) is not int for i in indices) or profile.get("frames") != len(indices)):
        raise ValueError("Require original all-view full selection with exact integer count")
    policy = profile.get("thread_policy", {})
    if policy.get("opencv_threads") != 20 or policy.get("open3d_threads") != 20:
        raise ValueError("Require matched OMP8/OpenCV20/Open3D20 input provenance")
    versions = profile.get("versions", {})
    if versions.get("open3d") != "0.20.0":
        raise ValueError("Require original Open3D0.20 legacy query/ICP semantics")
    native = profile.get("native_extension", {})
    if not native.get("path") or not native.get("sha256") or native.get("changed_during_profile") is not False:
        raise ValueError("Require immutable actual loaded native library provenance")
    fragments = profile.get("fragment_reconnection", {}).get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError("Require measured Final fragment-local inputs")
    return native


def descriptor(array):
    # Used only after reading this process's own freshly written fixture.
    import numpy as np
    array = np.asarray(array)
    if array.dtype != np.dtype("float64") or not np.isfinite(array).all():
        raise ValueError("Original point/normal/color/seed arrays must be finite float64")
    return {"dtype": array.dtype.str, "shape": list(array.shape), "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "original_strides": list(array.strides), "original_c_contiguous": bool(array.flags.c_contiguous),
        "byte_hash_order": "C value order; driver must use explicit bit-preserving contiguous copies for both CPU/GPU"}


def fixture_records(fixture):
    rows = []
    for position, task in enumerate(fixture["tasks"]):
        if task["position"] != position or len(task["pair"]) != 2:
            raise ValueError("Original fixture ordered task identity changed")
        a, b = task["pair"]
        clouds = {}
        for role, key in (("source", a), ("target", b)):
            clouds[role] = {name: descriptor(fixture["fragments"][key]["train"][name])
                            for name in ("points", "normals", "colors")}
        seeds = [descriptor(seed) for seed in task["proposals"]]
        if any(row["shape"] != [4, 4] for row in seeds) or len({r["sha256"] for r in seeds}) != len(seeds):
            raise ValueError("Genuine original deduplicated 4x4 proposal seeds required")
        rows.append({"position": position, "pair": task["pair"], "source": clouds["source"],
            "target": clouds["target"], "seeds": seeds, "proposal_count": len(seeds),
            "supported_batch_sizes": [size for size in (2, 4) if len(seeds) >= size]})
    if not rows:
        raise ValueError("No genuine eligible prepared pair inputs")
    return rows


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--run-allocated", action="store_true")
    args = p.parse_args()
    if not args.run_allocated or not 1 <= args.pairs <= 8:
        p.error("Require an allocated exclusive native slot and at most8 original pairs")
    args.session, args.profile, args.output = args.session.resolve(), args.profile.resolve(), args.output.resolve()
    if not args.output.is_relative_to(ROOT / "benchmark-output"):
        p.error("Keep private arrays under ignored benchmark-output")
    args.fixture = args.output.with_suffix(".fixture.pickle")
    args.snapshot = args.fixture.with_suffix(".runtime.zip")
    if any(path.exists() for path in (args.output, args.fixture, args.snapshot)):
        p.error("Require fresh report/fixture/source snapshot; preserve all earlier outputs")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            p.error("Stop the idle field server before allocated fixture preparation")
    return args


def run(args):
    before_source = source_hash()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    raw_sha, profile_sha = sha(args.session), sha(args.profile)
    native = profile_contract(profile, args.session, before_source, raw_sha)
    if sha(native["path"]) != native["sha256"]:
        raise RuntimeError("Actual native bytes differ from completed raw replay")
    artifacts = {path: sha(ROOT / path) for path in ARTIFACTS}
    report = {"kind": KIND, "status": "running", "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha256": before_source, "artifacts_sha256": artifacts,
        "session": {"path": str(args.session), "sha256": raw_sha},
        "profile": {"path": str(args.profile), "sha256": profile_sha}, "native_extension": native,
        "preparation_environment": ENVIRONMENT, "thread_policy": {"opencv_threads": 20, "open3d_threads": 20},
        "scope": "Raw calibrated current-source fragment component inputs using measured replay local poses only; no archived ZIP poses or new live pose seeds. Selected verified/candidate pairs are not a replay of adaptive frontier order.",
        "pose_seeds_used_for_live_tracking": False, "quality_authority": False,
        "performance_authority": False, "fixture_deserialization_scope": "Own fresh same-process trusted pickle only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    saved_env = {key: os.environ.get(key) for key in ENVIRONMENT}
    cv2 = o3d = None
    saved_threads = None
    failure = None
    try:
        os.environ.update(ENVIRONMENT)
        from scripts.research import benchmark_parallel_fragments as baseline
        import cv2
        import open3d as o3d
        import numpy as np
        saved_threads = (cv2.getNumThreads(), o3d.utility.get_max_threads())
        cv2.setNumThreads(20)
        o3d.utility.set_max_threads(20)
        actual_versions = {"python": sys.version.split()[0], "numpy": np.__version__,
            "open3d": o3d.__version__, "opencv": cv2.__version__}
        if actual_versions != profile["versions"]:
            raise RuntimeError("Loaded numerical versions differ from current raw replay")
        report["versions"] = actual_versions
        metadata = baseline.prepare_fixture(args.session, args.profile, args.fixture, args.pairs)
        report["preparation_metadata"] = metadata
        if (metadata["source_sha256"] != before_source or metadata["measured_fragment_source_sha256"] != before_source
                or metadata["runtime_changed_during_preparation"] or metadata["native_extension"] != {"path": native["path"], "sha256": native["sha256"]}):
            raise RuntimeError("Original preparation source/native identity failed closure")
        fixture_sha = sha(args.fixture)
        with args.fixture.open("rb") as stream:
            fixture = pickle.load(stream)
        if sha(args.fixture) != fixture_sha or fixture["metadata"] != metadata:
            raise RuntimeError("Own freshly captured fixture mutated during inspection")
        report["tasks"] = fixture_records(fixture)
        report["fixture"] = {"path": str(args.fixture), "sha256": fixture_sha, "bytes": args.fixture.stat().st_size}
        report["runtime_snapshot"] = {"path": str(args.snapshot), "sha256": sha(args.snapshot)}
        report["prepared_pair_count"] = len(report["tasks"])
        report["genuine_seed_count"] = sum(row["proposal_count"] for row in report["tasks"])
        report["actual_thread_policy"] = {"opencv_threads": cv2.getNumThreads(), "open3d_threads": o3d.utility.get_max_threads()}
        if report["actual_thread_policy"] != report["thread_policy"]:
            raise RuntimeError("Preparation selected threads changed")
    except BaseException as error:
        failure = error
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        cleanup = []
        if saved_threads is not None:
            for setter, value in ((cv2.setNumThreads, saved_threads[0]), (o3d.utility.set_max_threads, saved_threads[1])):
                try:
                    setter(value)
                except BaseException as error:
                    cleanup.append(error)
        for key, value in saved_env.items():
            try:
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            except BaseException as error:
                cleanup.append(error)
        try:
            report["source_sha256_after"] = source_hash()
            report["artifacts_sha256_after"] = {path: sha(ROOT / path) for path in ARTIFACTS}
            report["input_sha256_after"] = sha(args.session)
            report["profile_sha256_after"] = sha(args.profile)
            report["native_sha256_after"] = sha(native["path"])
            if (report["source_sha256_after"] != before_source or report["artifacts_sha256_after"] != artifacts
                    or report["input_sha256_after"] != raw_sha or report["profile_sha256_after"] != profile_sha
                    or report["native_sha256_after"] != native["sha256"]):
                raise RuntimeError("Input/source/native/helper changed during capture")
        except BaseException as error:
            cleanup.append(error)
        report["environment_restored"] = all(os.environ.get(k) == v for k, v in saved_env.items())
        try:
            report["threads_restored"] = saved_threads is None or (cv2.getNumThreads(), o3d.utility.get_max_threads()) == saved_threads
        except BaseException as error:
            cleanup.append(error)
            report["threads_restored"] = False
        report["cleanup_failures"] = [{"type": type(e).__name__, "message": str(e)} for e in cleanup]
        report["cleanup_passed"] = not cleanup and report["environment_restored"] and report["threads_restored"]
        if failure is None and not report["cleanup_passed"]:
            failure = cleanup[0] if cleanup else RuntimeError("Preparation ownership restoration failed")
        report["status"] = "passed" if failure is None else "failed"
        report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            save()
        except BaseException as write_error:
            if failure is not None:
                failure.add_note("Final report write also failed: " + repr(write_error))
                raise failure from write_error
            raise
    if failure is not None:
        raise failure
    print(f"Captured {report['prepared_pair_count']} current original pairs / {report['genuine_seed_count']} genuine seeds", flush=True)
    from scripts.process_metrics import finish_cuda_worker
    finish_cuda_worker()


if __name__ == "__main__":
    run(parse())
