"""Compare explicit convergence changes through original fixed-pair gates.

This changes the ICP method. It grants no CUDA, adaptive-frontier, Final pose,
surface or production authority. Original raw-derived proposals and every
reciprocal/camera/held-out/visual/information gate are retained. Wall timers run
without numerical shadows; comparisons take place after each complete pair.
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
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
METHODS = {
    "original": ((40, 30, 20), 1e-6),
    "epsilon-1e-5": ((40, 30, 20), 1e-5),
    "epsilon-1e-4": ((40, 30, 20), 1e-4),
    "budget-half": ((20, 15, 10), 1e-6),
    "budget-short": ((12, 8, 6), 1e-6),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def gate_comparison(native, candidate, np, module):
    if native is None or candidate is None:
        return {"accepted_equal": (native is None) == (candidate is None),
                "quality_candidate": native is None and candidate is None}
    a, b = np.asarray(native["transform"]), np.asarray(candidate["transform"])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Nonfinite original gate transform")
    translation, angle = module.motion(np.linalg.inv(a) @ b)
    support_equal = native.get("support") == candidate.get("support")
    scope_equal = native.get("validation_scope") == candidate.get("validation_scope")
    return {"accepted_equal": True, "support_equal": bool(support_equal),
        "validation_scope_equal": scope_equal, "translation_m": translation,
        "rotation_deg": angle,
        "information_max_abs_diff": float(np.max(np.abs(native["information"]-candidate["information"]))),
        "quality_candidate": bool(support_equal and scope_equal and translation <= .0005 and angle <= .1)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    p.add_argument("--run-allocated", action="store_true")
    args = p.parse_args()
    if not args.run_allocated or args.output.exists() or not 1 <= args.tasks <= 8 or not 1 <= args.repeats <= 5:
        p.error("Require exclusive allocated hardware, fresh output and bounded task/repeat counts")
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 8000)) == 0:
            p.error("Stop the scanner before a numerical experiment")
    report = {"kind": "icp-convergence-fixed-pair-tradeoff-v1", "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "methods": {name: {"iteration_budgets": list(METHODS[name][0]),
            "fitness_and_rmse_epsilon": METHODS[name][1]} for name in args.methods},
        "quality_authority": False, "whole_finish_authority": False,
        "scope": "Explicitly changed CPU convergence; fixed raw-derived pair components, every original gate and proposal verdict. No adaptive frontier, optimized graph, Final pose or surface authority.",
        "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    save()
    env = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "KINECT_CUDA_REGISTRATION", "KINECT_NATIVE")}
    failure = None
    original = module = np = None
    old_threads = None
    fixed = {}
    try:
        os.environ.update(OMP_NUM_THREADS="8", KINECT_CUDA_REGISTRATION="cpu", KINECT_NATIVE="on")
        from scripts.profile_session import source_hash
        from scripts.research.gpu_icp_experiment_driver import restore_matches, verdict
        from scripts.research.benchmark_parallel_fragments import unpack_fragment
        import numpy as np
        import open3d as o3d
        import cv2
        import scanner_server.fragments as module
        from scanner_server.cuda_registration import _device
        if _device.get() is not None:
            raise RuntimeError("Original CPU registration scope required")
        original = module._match
        old_threads = (o3d.utility.get_max_threads(), cv2.getNumThreads())
        o3d.utility.set_max_threads(20)
        cv2.setNumThreads(20)
        report["source_sha256"] = source_hash()
        capture = json.loads(args.capture.read_text())
        if (capture.get("kind") != "gpu-icp-current-field-pair-fixture-v2" or capture.get("status") != "passed"
            or capture.get("cleanup_passed") is not True or capture.get("source_sha256") != source_hash()
            or capture.get("source_sha256_after") != source_hash()
            or capture.get("pose_seeds_used_for_live_tracking") is not False
            or capture.get("artifacts_sha256") != capture.get("artifacts_sha256_after")):
            raise ValueError("Require closed current raw-derived capture")
        fixture_path = Path(capture["fixture"]["path"]).resolve()
        if not fixture_path.is_relative_to(ROOT/"benchmark-output") or sha(fixture_path) != capture["fixture"]["sha256"]:
            raise ValueError("Require hash-checked own trusted local fixture")
        fixed.update({str(ROOT/path): digest for path, digest in capture["artifacts_sha256"].items()})
        for record in (capture["session"], capture["profile"], capture["native_extension"], capture["fixture"], capture["runtime_snapshot"]):
            fixed[record["path"]] = record["sha256"]
        for path in (__file__, args.capture, "scripts/research/gpu_icp_experiment_driver.py",
                "scripts/research/benchmark_parallel_fragments.py"):
            fixed[str(Path(path).resolve())] = sha(path)
        backend = sys.modules[o3d.geometry.PointCloud.__module__.rsplit(".", 1)[0]]
        for path in (backend.__file__, sys.modules["numpy._core._multiarray_umath"].__file__):
            fixed[str(Path(path).resolve())] = sha(path)
        if any(sha(path) != digest for path, digest in fixed.items()):
            raise RuntimeError("Capture or loaded backend bytes changed")
        report["fixed_files_sha256"] = fixed
        report["runtime"] = {"numpy": np.__version__, "open3d": o3d.__version__, "opencv": cv2.__version__,
            "open3d_threads": o3d.utility.get_max_threads(), "opencv_threads": cv2.getNumThreads(), "omp_threads": os.environ["OMP_NUM_THREADS"]}
        with fixture_path.open("rb") as f:
            fixture = pickle.load(f)  # Own local capture, byte-checked above.
        fragments = {key: unpack_fragment(value) for key, value in fixture["fragments"].items()}
        def cloud_owners():
            records = []
            for key, fragment in sorted(fragments.items()):
                clouds = [fragment.train, fragment.heldout]
                for view in fragment.keys:
                    clouds.extend((view.train, view.heldout))
                    records.append(hashlib.sha256(np.asarray(view.pose).tobytes(order="C")).hexdigest())
                for cloud in clouds:
                    for attribute in ("points", "normals", "colors"):
                        array = np.asarray(getattr(cloud, attribute))
                        records.append([key, attribute, array.dtype.str, list(array.shape),
                            hashlib.sha256(array.tobytes(order="C")).hexdigest()])
            for task in fixture["tasks"]:
                for seed in task["proposals"]:
                    array = np.asarray(seed)
                    records.append(["seed", task["pair"], array.dtype.str, list(array.shape),
                        hashlib.sha256(array.tobytes(order="C")).hexdigest()])
            return records
        report["input_owners_before"] = cloud_owners()
        if not fixture["tasks"]:
            raise ValueError("No genuine original proposals to test")
        for task in fixture["tasks"][:args.tasks]:
            a, b = task["pair"]
            native_values = None
            names = list(dict.fromkeys(["original"]+args.methods))
            for repeat in range(args.repeats):
                schedules = names if repeat == 0 else names[repeat % len(names):]+names[:repeat % len(names)]
                for name in schedules:
                    calls = []
                    budgets, epsilon = METHODS[name]
                    def changed_match(source, target, initial):
                        start = time.perf_counter()
                        if name == "original":
                            result = original(source, target, initial)
                        else:
                            pose = initial
                            for radius, iterations in zip((.12, .06, .03), budgets):
                                result = module.REG.registration_icp(source, target, radius, pose,
                                    module.REG.TransformationEstimationPointToPlane(module.REG.HuberLoss(.01)),
                                    module.REG.ICPConvergenceCriteria(relative_fitness=epsilon,
                                        relative_rmse=epsilon, max_iteration=iterations))
                                pose = result.transformation
                        calls.append({"source_rows": len(source.points), "target_rows": len(target.points),
                            "wall_s": time.perf_counter()-start})
                        return result
                    module._match = changed_match
                    values = []
                    elapsed = 0.
                    for seed in task["proposals"]:
                        restore_matches(fixture, fragments, module)
                        start = time.perf_counter()
                        values.append(module._verify_bridge(fragments[a], fragments[b], seed, fixture["camera"]))
                        elapsed += time.perf_counter()-start
                    module._match = original
                    if native_values is None:
                        native_values = values
                    comparisons = [gate_comparison(x, y, np, module) for x, y in zip(native_values, values)]
                    same_verdict = verdict(native_values, module) == verdict(values, module)
                    row = {"pair": task["pair"], "repeat": repeat, "method": name,
                        "genuine_proposals": len(task["proposals"]), "whole_proposals_wall_s": elapsed,
                        "icp_calls": len(calls), "icp_wall_s": sum(c["wall_s"] for c in calls),
                        "call_sizes": [[c["source_rows"], c["target_rows"]] for c in calls],
                        "original_pair_verdict": verdict(native_values, module), "candidate_pair_verdict": verdict(values, module),
                        "proposal_gate_comparisons": comparisons, "quality_candidate": same_verdict and all(c["quality_candidate"] for c in comparisons)}
                    report["rows"].append(row)
                    save()
                    print(f"pair{a}/{b} {name} repeat{repeat}: {elapsed:.3f}s calls={len(calls)} quality_candidate={row['quality_candidate']}", flush=True)
        report["source_sha256_after"] = source_hash()
        report["fixed_files_sha256_after"] = {path: sha(path) for path in fixed}
        report["input_owners_after"] = cloud_owners()
        report["input_owners_unchanged"] = report["input_owners_after"] == report["input_owners_before"]
        report["runtime_policy_unchanged"] = (o3d.utility.get_max_threads() == 20 and cv2.getNumThreads() == 20
            and os.environ.get("OMP_NUM_THREADS") == "8" and _device.get() is None)
        if report["source_sha256_after"] != report["source_sha256"] or report["fixed_files_sha256_after"] != fixed:
            raise RuntimeError("Experiment source/input/runtime closure changed")
        if not report["input_owners_unchanged"] or not report["runtime_policy_unchanged"]:
            raise RuntimeError("Actual original input owner or CPU thread/registration scope changed")
    except BaseException as error:
        failure = error
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        cleanup_failures = []
        def cleanup(label, action):
            try:
                action()
            except BaseException as error:
                cleanup_failures.append(label+": "+str(error))
                if failure is not None:
                    failure.add_note(label+": "+str(error))
        if module is not None and original is not None:
            cleanup("original match", lambda: setattr(module, "_match", original))
        if old_threads is not None:
            cleanup("Open3D threads", lambda: o3d.utility.set_max_threads(old_threads[0]))
            cleanup("OpenCV threads", lambda: cv2.setNumThreads(old_threads[1]))
            def check_threads():
                report["threads_restored"] = (o3d.utility.get_max_threads(), cv2.getNumThreads()) == old_threads
            cleanup("restored thread check", check_threads)
        def restore_environment(key, value):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for key, value in env.items():
            cleanup("environment "+key, lambda key=key, value=value: restore_environment(key, value))
        report["cleanup_failures"] = cleanup_failures
        report["cleanup_passed"] = (not cleanup_failures and (module is None or original is None or module._match is original)
            and all(os.environ.get(k) == v for k,v in env.items())
            and (old_threads is None or report.get("threads_restored") is True))
        report["status"] = "passed" if failure is None and report["cleanup_passed"] else "failed"
        report["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            save()
        except BaseException as error:
            if failure is not None:
                failure.add_note("Final report write also failed: "+str(error))
            else:
                raise
    if failure is not None:
        raise RuntimeError("Convergence experiment failed; retain its diagnostics") from failure
    if not report["cleanup_passed"]:
        raise RuntimeError("Convergence experiment cleanup did not close")

if __name__ == "__main__":
    main()
