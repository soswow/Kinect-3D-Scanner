"""Run isolated CPU/CUDA field replays with the archive's actual settings.

This is a baseline producer, not resident-ICP proof authority. Every view is
processed from raw RGB-D; archived poses never seed replay. Stop the idle
scanner server first and allocate exclusive hardware use before running.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash
from scripts.process_metrics import gpu_info


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=("cpu", "cuda"), default=("cpu", "cuda"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--final-block-count", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_allocated or not 1 <= args.repeats <= 3:
        parser.error("Require an exclusive hardware slot and 1–3 repeats")
    if args.final_block_count is not None and not 1 <= args.final_block_count <= 50000:
        parser.error("Final block count must be 1–50000")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Modes must be distinct")
    sessions = [path.resolve(strict=True) for path in args.sessions]
    if len({path.stem for path in sessions}) != len(sessions):
        parser.error("Require distinct session stems")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Port 8000 is in use; stop the idle scanner before isolated measurements")
    folder = args.output_directory.resolve()
    try:
        folder.relative_to(ROOT / "benchmark-output")
    except ValueError:
        parser.error("Keep reports, logs and geometry in ignored benchmark-output")
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "execution.json"
    if manifest_path.exists() or any(folder.iterdir()):
        parser.error("Require an empty output directory; preserve previous measurements")
    artifacts = {str(path.relative_to(ROOT)): file_hash(path) for path in (
        Path(__file__), ROOT / "scripts/profile_session.py", ROOT / "scripts/process_metrics.py")}
    core = source_hash()
    hardware = gpu_info()
    execution = {
        "kind": "field-session-raw-replay-baselines-v1", "status": "running",
        "start_utc": now(), "source_sha256": core, "gpu_hardware": hardware,
        "artifacts_sha256": artifacts, "final_block_override": args.final_block_count,
        "runs": [],
        "scope": "Complete raw-selected-view Live plus Finish; archive settings unchanged except an explicit common final block budget. Imports, ZIP decoding, hashing, geometry export and report writes are outside profile processing times. No camera FPS or GPU resident ICP authority.",
    }

    def save():
        manifest_path.write_text(json.dumps(execution, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        for session in sessions:
            archive_digest = file_hash(session)
            for repeat in range(args.repeats):
                for mode in (args.modes if repeat % 2 == 0 else list(reversed(args.modes))):
                    name = f"{session.stem}-{mode}-{repeat}"
                    report_path = folder / (name + ".json")
                    log_path = folder / (name + ".log")
                    options = {
                        "OMP_NUM_THREADS": "8", "KINECT_NATIVE": "on", "KINECT_BLOCK_COUNT": "5000",
                        "KINECT_CUDA_REGISTRATION": "cpu", "KINECT_CUDA_ODOMETRY": "off",
                        "KINECT_CUDA_MATCHING": "cpu" if mode == "cpu" else "cuda",
                        "KINECT_CUDA_FUSION": "tensor" if mode == "cpu" else "fused",
                        "KINECT_MODEL_REFRESH": "lazy", "KINECT_KEYFRAME_CACHE": "on",
                        "KINECT_LIVE_RECOVERY": "full", "KINECT_VISUAL_FEATURES": "adaptive",
                        "KINECT_VISUAL_REFINEMENT": "icp", "KINECT_FINAL_VISUAL_FIRST": "off",
                        "KINECT_FINAL_LOCAL_REFINEMENT": "icp", "KINECT_ADAPTIVE_EXPERIMENTAL": "off",
                        "KINECT_CUDA_INPUT": "off" if mode == "cpu" else "auto",
                        "KINECT_CUDA_CONFIDENCE": "off", "PYTHONUTF8": "1",
                    }
                    command = [sys.executable, "-s", "-u", str(ROOT / "scripts/profile_session.py"),
                        str(session), "--device", mode, "--tracking", "legacy", "--finish", "--output", str(report_path)]
                    if args.final_block_count is not None:
                        command.extend(("--final-block-count", str(args.final_block_count)))
                    row = {"session": str(session), "input_sha256": archive_digest,
                        "mode": mode, "repeat": repeat, "command": command,
                        "environment_overrides": options, "report": str(report_path),
                        "log": str(log_path), "start_utc": now(), "status": "running"}
                    execution["runs"].append(row)
                    save()
                    print(f"{session.name}: {mode} repeat {repeat + 1}/{args.repeats}", flush=True)
                    started = time.perf_counter()
                    with log_path.open("wb") as log:
                        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **options),
                            stdout=log, stderr=subprocess.STDOUT)
                        row["child_pid"] = child.pid
                        save()
                        try:
                            code = child.wait(timeout=7200)
                        except BaseException:
                            child.kill()
                            child.wait()
                            raise
                    row.update(exit_code=code, child_wait_completed=True, elapsed_process_s=time.perf_counter() - started,
                        end_utc=now(), log_sha256=file_hash(log_path), status="exited")
                    save()
                    if code != 0:
                        raise RuntimeError(f"Replay exited {code}: {log_path}")
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if (report["source_sha256"] != core or source_hash() != core
                            or report["input_sha256"] != archive_digest or file_hash(session) != archive_digest
                            or report["input_changed_during_profile"] or report["source_changed_during_profile"]
                            or report["pose_seeds_used"] or report["gpu_hardware"] != hardware or gpu_info() != hardware
                            or any(file_hash(ROOT / path) != digest for path, digest in artifacts.items())):
                        raise RuntimeError("Source, raw inputs, runner or GPU identity changed during measurement")
                    row.update(report_sha256=file_hash(report_path), status="complete",
                        mesh_built=report["mesh_built"], live_s=report["live_s"], finish_s=report["finish_s"],
                        accepted_before_finish=report["accepted_before_finish"], accepted=report["accepted"],
                        final_reconstruction=report["final_reconstruction"])
                    save()
                    print(f"  Live {report['live_s']:.3f}s, Finish {report['finish_s']:.3f}s, "
                        f"views {report['accepted_before_finish']} -> {report['accepted']}, mesh={report['mesh_built']}", flush=True)
        execution["status"] = "complete"
    except BaseException as error:
        execution.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        execution["end_utc"] = now()
        save()


if __name__ == "__main__":
    run()
