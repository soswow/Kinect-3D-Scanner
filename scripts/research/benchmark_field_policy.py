"""Replay existing full/deferred recovery policies from raw field archives.

An offline workflow experiment: deferred recovery can alter Live acceptance and
Finish coverage, so lower latency alone cannot establish reconstruction quality.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import socket
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash
from scripts.research.profile_resident_finish import PIPELINE
from scripts.research.owned_profile_process import run_owned

PROFILE_OPTIONS = ("KINECT_CUDA_REGISTRATION", "KINECT_CUDA_ODOMETRY", "KINECT_CUDA_MATCHING",
    "KINECT_MODEL_REFRESH", "KINECT_KEYFRAME_CACHE", "KINECT_LIVE_RECOVERY", "KINECT_VISUAL_FEATURES",
    "KINECT_VISUAL_REFINEMENT", "KINECT_FINAL_VISUAL_FIRST", "KINECT_FINAL_LOCAL_REFINEMENT",
    "KINECT_ADAPTIVE_EXPERIMENTAL", "KINECT_CUDA_INPUT", "KINECT_CUDA_CONFIDENCE")
THREAD_OPTIONS = ("OMP_WAIT_POLICY", "KMP_BLOCKTIME", "OPENCV_FOR_THREADS_NUM",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--recovery", choices=("full", "deferred"), required=True)
    parser.add_argument("--final-block-count", type=int)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    if not args.run_allocated:
        parser.error("Require exclusive hardware allocation")
    if args.final_block_count is not None and not 1 <= args.final_block_count <= 50000:
        parser.error("Final budget must be 1-50000 blocks")
    sessions = [path.resolve(strict=True) for path in args.sessions]
    if len({path.stem for path in sessions}) != len(sessions):
        parser.error("Require distinct session stems")
    folder = args.output_directory.resolve()
    try:
        folder.relative_to(ROOT / "benchmark-output")
    except ValueError:
        parser.error("Keep all private outputs in ignored benchmark-output")
    if folder.exists() and any(folder.iterdir()):
        parser.error("Preserve previous measurements; require empty output directory")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle field server before isolated measurements")
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = {name: file_hash(ROOT / name) for name in (
        "scripts/research/benchmark_field_policy.py", "scripts/profile_session.py",
        "scripts/process_metrics.py", "scripts/research/profile_resident_finish.py",
        "scripts/research/owned_profile_process.py")}
    core = source_hash()
    environment = dict(PIPELINE, KINECT_LIVE_RECOVERY=args.recovery)
    expected_threads = {"opencv_threads":20,"open3d_threads":20,
        **{key:os.environ.get(key) for key in THREAD_OPTIONS}}
    native_spec = importlib.util.find_spec("_kinect_native")
    if native_spec is None or native_spec.origin is None:
        parser.error("Require the original native extension")
    native_path = Path(native_spec.origin).resolve(strict=True)
    native_sha = file_hash(native_path)
    report = {"kind": "raw-field-existing-recovery-policy-experiment-v1", "status": "running",
        "start_utc": now(), "source_sha256": core, "artifacts_sha256": artifacts,
        "environment_overrides": environment, "final_block_override": args.final_block_count,
        "expected_thread_policy":expected_threads,
        "native_extension": {"path":str(native_path), "sha256":native_sha},
        "runs": [], "scope": "Fresh raw selected-view CUDA replay, original existing policy. No archived pose seeds. Imports/decoding/exports excluded from processing times. A changed acceptance/coverage is an experiment, requiring separate physical surface and view-coverage assessment. No camera FPS or research resident-ICP authority."}
    manifest = folder / "execution.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        for session in sessions:
            output = folder / (session.stem + "-" + args.recovery + ".json")
            log = output.with_suffix(".log")
            archive_sha = file_hash(session)
            with zipfile.ZipFile(session) as archive:
                if archive.getinfo("manifest.json").file_size > 8 * 1024**2:
                    raise ValueError("Unsupported unbounded archive manifest")
                source_manifest = json.loads(archive.read("manifest.json"))
            frame_count = len(source_manifest["frames"])
            if not 1 <= frame_count <= 1000:
                raise ValueError("Unsupported raw field frame count")
            expected_settings = dict(source_manifest["settings"])
            settings_sha = hashlib.sha256(json.dumps(expected_settings,sort_keys=True,allow_nan=False).encode()).hexdigest()
            if args.final_block_count is not None:
                expected_settings["final_block_count"] = args.final_block_count
            command = [sys.executable, "-s", "-u", str(ROOT / "scripts/profile_session.py"),
                str(session), "--device", "cuda", "--tracking", "legacy", "--finish", "--output", str(output)]
            if args.final_block_count is not None:
                command.extend(("--final-block-count", str(args.final_block_count)))
            row = {"session": session.name, "input_sha256": archive_sha,
                "command": command, "profile": str(output), "log": str(log), "status": "running",
                "start_utc": now()}
            report["runs"].append(row)
            save()
            started = time.perf_counter()
            with log.open("wb") as stream:
                code = run_owned(command, cwd=ROOT, env=dict(os.environ, **environment),
                    stream=stream, timeout=7200, record=row, on_started=save)
            row.update(exit_code=code, child_wait_completed=True,
                elapsed_process_s=time.perf_counter() - started, end_utc=now(),
                log_sha256=file_hash(log), status="exited")
            save()
            if code != 0:
                raise RuntimeError(f"Replay exited {code}: {log}")
            profile = json.loads(output.read_text(encoding="utf-8"))
            backend = profile["backend"]
            native = profile["native_extension"]
            expected_pipeline = {key:environment[key] for key in PROFILE_OPTIONS}
            if (source_hash() != core or profile["source_sha256"] != core
                    or file_hash(session) != archive_sha or profile["input_sha256"] != archive_sha
                    or profile["source_changed_during_profile"] or profile["input_changed_during_profile"]
                    or profile["pose_seeds_used"] or not profile["finish_requested"]
                    or native["changed_during_profile"]
                    or Path(native["path"]).resolve() != native_path or native["sha256"] != native_sha
                    or file_hash(native_path) != native_sha
                    or profile["pipeline_options"] != expected_pipeline
                    or profile["native_mode"] != environment["KINECT_NATIVE"]
                    or profile["initial_blocks"] != environment["KINECT_BLOCK_COUNT"]
                    or profile["omp_threads"] != environment["OMP_NUM_THREADS"]
                    or profile["thread_policy"] != expected_threads
                    or backend["requested"] != "cuda" or backend["device"] != "CUDA:0"
                    or backend["cuda_available"] is not True or backend["fallback_reason"] is not None
                    or backend["tracking"] != "legacy" or backend["tracking_device"] != "CPU:0"
                    or backend["native_kernels"]["active"] is not True or backend["native_kernels"]["api_version"] != 2
                    or backend["confidence_cuda"]["requested"] != environment["KINECT_CUDA_FUSION"]
                    or backend["confidence_cuda"]["implementation"] != "fused" or backend["confidence_cuda"]["fallback_reason"] is not None
                    or backend["cuda_input"]["implementation"] != "cuda" or backend["cuda_input"]["fallback_batches"] != 0
                    or backend["depth_confidence"]["requested"] != "off" or backend["depth_confidence"]["implementation"] != "cpu"
                    or backend["depth_confidence"]["device"] != "CPU:0"
                    or backend["descriptor_matching"]["implementation"] != "cuda"
                    or backend["live_recovery"] != args.recovery
                    or profile["selected_indices"] != list(range(profile["frames"]))
                    or profile["frames"] != frame_count or profile["settings"] != expected_settings
                    or profile["original_settings_sha256"] != settings_sha
                    or profile["seed"] != 0 or profile["experimental_visual_fallback"] is not False
                    or profile["settings_overrides"] != ({"final_block_count":args.final_block_count} if args.final_block_count is not None else {})
                    or (args.final_block_count is not None and profile["settings"]["final_block_count"] != args.final_block_count)
                    or any(file_hash(ROOT / name) != digest for name, digest in artifacts.items())):
                raise RuntimeError("Raw/source/native/policy closure failed")
            geometry = output.with_suffix(".geometry.npz")
            row.update(status="complete", profile_sha256=file_hash(output),
                geometry_sha256=file_hash(geometry), live_s=profile["live_s"],
                finish_s=profile["finish_s"], processing_s=profile["processing_s"],
                mesh_built=profile["mesh_built"], accepted_before_finish=profile["accepted_before_finish"],
                accepted_after_finish=profile["accepted"], quality_proven=False)
            save()
            print(json.dumps({key: row[key] for key in ("session", "live_s", "finish_s",
                "mesh_built", "accepted_before_finish", "accepted_after_finish")}), flush=True)
        report.update(status="complete", end_utc=now())
        save()
    except BaseException as error:
        report.update(status="failed", end_utc=now(), failure={"type": type(error).__name__, "message": str(error)})
        try:
            save()
        except BaseException as write_error:
            error.add_note(f"Policy diagnostic write also failed: {write_error}")
        raise


if __name__ == "__main__":
    main()
