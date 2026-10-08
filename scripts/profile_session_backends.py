"""Matched sequential CPU/CUDA replays of lossless session ZIPs.

Example: python scripts/profile_session_backends.py export/*.zip --finish
Raw data, detailed diagnostics and geometry remain in ignored benchmark-output.
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--finish-runs", nargs="+", choices=("cpu", "cuda-tensor", "cuda-fused", "cuda-legacy"),
                        default=("cpu", "cuda-tensor", "cuda-fused", "cuda-legacy"),
                        help="Modes on which to additionally measure Finish")
    parser.add_argument("--resume", action="store_true", help="Reuse matching completed reports")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--runs", nargs="+", choices=("cpu", "cuda-tensor", "cuda-fused", "cuda-legacy"),
                        default=("cpu", "cuda-tensor", "cuda-fused"))
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-output/cuda-study/backends.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("Repeats and threads must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), KINECT_BLOCK_COUNT="5000")
    for session in args.sessions:
        with zipfile.ZipFile(session) as archive:
            expected_indices = list(range(len(json.loads(archive.read("manifest.json"))["frames"])))[:args.limit]
        # Alternate order to reduce consistent first/last-run thermal bias.
        for repeat in range(args.repeats):
            for mode in args.runs if repeat % 2 == 0 else reversed(args.runs):
                device = "cpu" if mode == "cpu" else "cuda"
                tracking = "legacy" if mode in ("cpu", "cuda-legacy") else "tensor"
                env["KINECT_CUDA_FUSION"] = "fused" if mode == "cuda-fused" else "tensor"
                destination = args.output.parent / f"{session.stem}-{mode}-{repeat}.json"
                command = [sys.executable, str(ROOT / "scripts/profile_session.py"), str(session),
                           "--device", device, "--tracking", tracking, "--output", str(destination)]
                if args.limit:
                    command += ["--limit", str(args.limit)]
                finish = args.finish and mode in args.finish_runs
                if finish:
                    command += ["--finish"]
                print(f"{session.name}: {mode}, run {repeat + 1}/{args.repeats}", flush=True)
                reusable = False
                if args.resume and destination.exists():
                    old = json.loads(destination.read_text())
                    sys.path.insert(0, str(ROOT))
                    from scripts.profile_session import file_hash, source_hash
                    reusable = (old["input_sha256"] == file_hash(session)
                        and old["source_sha256"] == source_hash()
                        and not old["source_changed_during_profile"]
                        and not old["input_changed_during_profile"]
                        and old["finish_requested"] == finish
                        and old["selected_indices"] == expected_indices
                        and not old["pose_seeds_used"]
                        and not old["settings_overrides"]
                        and old["backend"]["device"] == ("CPU:0" if device == "cpu" else "CUDA:0")
                        and old["backend"]["tracking"] == tracking
                        and old["initial_blocks"] == "5000"
                        and old["omp_threads"] == str(args.threads)
                        and old["native_mode"] == env.get("KINECT_NATIVE", "auto")
                        and old["backend"].get("confidence_cuda", {}).get("requested", env["KINECT_CUDA_FUSION"])
                            == env["KINECT_CUDA_FUSION"])
                if not reusable:
                    with destination.with_suffix(".log").open("w") as log:
                        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                                stderr=subprocess.STDOUT, timeout=7200)
                    if result.returncode:
                        raise RuntimeError(f"Replay failed; see {destination.with_suffix('.log')}")
                report = json.loads(destination.read_text())
                reports.append({"session": session.name, "mode": mode, "repeat": repeat,
                    "report": str(destination.resolve()), **{key: report[key] for key in (
                        "input_sha256", "source_sha256", "source_changed_during_profile", "frames",
                        "live_s", "finish_s", "processing_s", "accepted_before_finish", "accepted",
                        "accepted_indices_before_finish", "mesh_built", "backend", "live_stages",
                        "peak_process_rss_bytes", "geometry", "final_reconstruction",
                        "fragment_reconnection", "settings_overrides")}})
                args.output.write_text(json.dumps({"runs": reports}, indent=2, allow_nan=False))
                print(f"  live={report['live_s']:.2f}s finish={report['finish_s']} "
                      f"accepted={report['accepted_before_finish']}/{report['frames']}", flush=True)
    summaries = []
    for session in args.sessions:
        for mode in args.runs:
            runs = [r for r in reports if r["session"] == session.name and r["mode"] == mode]
            summaries.append({"session": session.name, "mode": mode,
                "median_live_s": float(np.median([r["live_s"] for r in runs])),
                "median_processing_s": float(np.median([r["processing_s"] for r in runs])),
                "accepted_range": [min(r["accepted_before_finish"] for r in runs),
                                   max(r["accepted_before_finish"] for r in runs)]})
    args.output.write_text(json.dumps({"runs": reports, "summary": summaries}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
