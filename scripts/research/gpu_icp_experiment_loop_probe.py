"""Small actual-input device-loop audit, not an unaudited performance benchmark.

Uses only a locally captured, hash-checked current field fixture. Compares full
CPU ICP with exhaustive-audited one-step/device-graph trajectories. Diagnostic
wall includes all CPU query audits, setup and graph capture. No scanner hooks.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, report, save):
    from scripts.profile_session import source_hash
    from scripts.research.device_loop_icp import DeviceLoopICP, source_contract
    capture = json.loads(args.capture.read_text())
    if (capture.get("kind") != "gpu-icp-current-field-pair-fixture-v2"
            or capture.get("status") != "passed" or capture.get("cleanup_passed") is not True
            or capture.get("source_sha256") != source_hash()
            or capture.get("source_sha256_after") != source_hash()
            or capture.get("artifacts_sha256") != capture.get("artifacts_sha256_after")):
        raise ValueError("Require fresh closed current-source component capture")
    fixture_path = Path(capture["fixture"]["path"]).resolve()
    if not fixture_path.is_relative_to(ROOT / "benchmark-output") or sha(fixture_path) != capture["fixture"]["sha256"]:
        raise ValueError("Require original local trusted fixture bytes")
    for path, digest in capture["artifacts_sha256"].items():
        if sha(ROOT/path) != digest:
            raise ValueError("Capture helpers changed")
    source = source_hash()
    before = source_contract()
    own_sha = sha(__file__)
    capture_sha = sha(args.capture)
    report.update(source_sha256=source, device_loop_source_contract=before, probe_sha256=own_sha,
        capture_sha256=capture_sha, fixture_sha256=sha(fixture_path), all_nearest_queries_cpu_audited=True,
        scope="Actual prepared fragment-pair component. No adaptive frontier, original complete bridge/witness gate closure, Final poses or mesh authority.")
    import cupy as cp
    import numpy as np
    import open3d as o3d
    import cv2
    from scripts.research.cuda_device_flat_grid_registration import DeviceFlatGridICP
    from scripts.research.archive.research_resident_icp import ResidentICP
    from scripts.research.benchmark_parallel_fragments import unpack_cloud
    cv2.setNumThreads(20)
    o3d.utility.set_max_threads(20)
    report["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__, "cupy": cp.__version__,
        "open3d": o3d.__version__, "opencv": cv2.__version__, "driver": cp.cuda.runtime.driverGetVersion(),
        "cuda": cp.cuda.runtime.runtimeGetVersion(), "device": "CUDA:0", "opencv_threads": cv2.getNumThreads(),
        "open3d_threads": o3d.utility.get_max_threads(), "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    with fixture_path.open("rb") as f:
        fixture = pickle.load(f)  # User-authorized own local capture, SHA checked above.
    if sha(fixture_path) != report["fixture_sha256"]:
        raise ValueError("Fixture changed during read")
    tasks = [task for task in fixture["tasks"] if len(task["proposals"]) >= args.seeds][:args.pairs]
    if not tasks:
        raise ValueError("No genuine eligible independent seed group; never pad it")
    for task in tasks:
        a, b = task["pair"]
        source_cloud = unpack_cloud(fixture["fragments"][a]["train"])
        target_cloud = unpack_cloud(fixture["fragments"][b]["train"])
        for index, seed in enumerate(task["proposals"][:args.seeds]):
            seed = np.ascontiguousarray(seed, dtype=np.float64)  # Bit-preserving layout copy for both controls.
            row = {"pair": task["pair"], "proposal_index": index, "seed_sha256": hashlib.sha256(seed.tobytes()).hexdigest(),
                "source_rows": len(source_cloud.points), "target_rows": len(target_cloud.points), "runs": []}
            report.setdefault("rows", []).append(row)
            start = time.perf_counter()
            cpu = ResidentICP._original_cpu_match(source_cloud, target_cloud, seed)
            row["original_cpu_wall_s"] = time.perf_counter()-start
            control = None
            for graph, chunk in ((False,1),(True,4)):
                retrieval = loop = None
                primary = None
                start = time.perf_counter()
                try:
                    retrieval = DeviceFlatGridICP("CUDA:0", max_clouds=1, audit_nearest=True,
                        audit_misses=True, miss_policy="direct-miss-research-v1")
                    loop = DeviceLoopICP(retrieval, cuda_graph=graph)
                    result = loop.match(source_cloud, target_cloud, seed, chunk_iterations=chunk)
                    relative = np.linalg.inv(cpu.transformation) @ result.transformation
                    translation = float(np.linalg.norm(relative[:3,3]))
                    angle = float(np.degrees(np.arccos(np.clip((np.trace(relative[:3,:3])-1)/2,-1,1))))
                    if translation > .0005 or angle > .1 or abs(result.fitness-cpu.fitness) > .002 or abs(result.inlier_rmse-cpu.inlier_rmse) > .0005:
                        raise RuntimeError("Actual GPU ICP failed fresh original CPU pose/metric bounds")
                    numerical = (result.transformation.tobytes(), result.fitness, result.inlier_rmse,
                        np.asarray(result.correspondence_set).tobytes(), loop.statistics["queries"], loop.statistics["updates"])
                    if control is None:
                        control = numerical
                    elif control != numerical:
                        raise RuntimeError("Graph chunk changed audited one-step result bytes, correspondence IDs or query/update counts")
                    row["runs"].append({"cuda_graph": graph, "chunk_steps": chunk,
                        "audited_total_wall_s": time.perf_counter()-start, "translation_delta_m": translation,
                        "rotation_delta_deg": angle, "fitness_delta": result.fitness-cpu.fitness,
                        "rmse_delta_m": result.inlier_rmse-cpu.inlier_rmse, "queries": loop.statistics["queries"],
                        "updates": loop.statistics["updates"], "graph_matches_one_step_bytes": control == numerical})
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    cleanup = []
                    completed = loop is None
                    if loop is not None:
                        try:
                            loop.close()
                            completed = True
                        except BaseException as error:
                            cleanup.append(error)
                    # The cache contains pointers used by the loop's graph.
                    # Retain it when lane completion could not be established.
                    if retrieval is not None and completed:
                        try:
                            retrieval.close()
                        except BaseException as error:
                            cleanup.append(error)
                    if loop is not None:
                        row.setdefault("loop_reports", []).append(loop.report())
                    if cleanup:
                        if primary is not None:
                            for error in cleanup:
                                primary.add_note("Probe cleanup also failed: "+repr(error))
                        else:
                            raise cleanup[0]
                save()
            print(f"Pair {a}/{b} seed{index}: complete device loop and graph actual CPU shadows passed", flush=True)
    report.update(source_sha256_after=source_hash(), probe_sha256_after=sha(__file__),
        device_loop_source_contract_after=source_contract())
    if (source != report["source_sha256_after"] or own_sha != report["probe_sha256_after"]
            or before != report["device_loop_source_contract_after"] or capture_sha != sha(args.capture)
            or report["fixture_sha256"] != sha(fixture_path)):
        raise RuntimeError("Input/source/helper closure failed")
    report["status"] = "passed"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--run-allocated", action="store_true")
    args = p.parse_args()
    if not args.run_allocated or not 1 <= args.pairs <= 8 or args.seeds not in (2,4):
        p.error("Require exclusive hardware and at most8 genuine2/4seed pair groups")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "benchmark-output") or args.output.exists():
        p.error("Require fresh private output in benchmark-output")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1",8000)) == 0:
            p.error("Stop idle field server first")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": "actual-input-exhaustive-device-loop-graph-probe-v1", "status": "running",
        "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "performance_authority": False,
        "timer_scope": "Audited diagnostic including setup/all CPU query audits/graph capture. No unaudited speed claim."}
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    code = 0
    try:
        save()
        run(args, report, save)
    except BaseException as error:
        code = 1
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
    report["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save()
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    if code == 0:
        from scripts.process_metrics import finish_cuda_worker
        finish_cuda_worker()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
