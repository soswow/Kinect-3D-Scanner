"""Matched, sequential CPU/CUDA replays in separate processes.

Never reset the user's running server. Detailed outputs/meshes stay ignored.
CUDA absence is reported as unavailable, never timed as CPU fallback.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    source = hashlib.sha256()
    for folder in (ROOT / "scanner_server", ROOT / "shared"):
        for path in sorted(folder.glob("*.py")):
            source.update(path.name.encode())
            source.update(path.read_bytes())
    return source.hexdigest()


def summarize(report):
    diagnostics = report.get("diagnostics", [])
    stages = report.get("stage_totals_ms") or {}
    total = sum(stages.values())
    timings = {}
    for name in stages:
        values = [
            d["timings_ms"][name]
            for d in diagnostics
            if name in d.get("timings_ms", {})
        ]
        timings[name] = {
            "total_ms": stages[name],
            "share": stages[name] / total if total else 0,
            "p50_ms": float(np.percentile(values, 50)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None,
        }
    return {
        k: report.get(k)
        for k in (
            "frames",
            "accepted",
            "elapsed_s",
            "mesh_built",
            "backend",
            "anchored_translation_rmse_m",
            "rotation_rmse_deg",
            "peak_process_rss_bytes",
        )
    } | {
        "stages": timings,
        "frames_per_processing_second": report["frames"]
        / max(report["elapsed_s"], 1e-6),
        "rejected": report["frames"] - report["accepted"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["redwood", "tum", "recording"], default="redwood"
    )
    parser.add_argument("--path", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=["cpu-legacy", "cpu-tensor", "cuda-tensor"],
        default=["cpu-legacy", "cpu-tensor", "cuda-tensor"],
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-output/backend-profile.json"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.parent / (args.output.stem + "-runs")
    work.mkdir(exist_ok=True)
    outputs = {}
    source_before = source_hash()
    env = dict(
        os.environ,
        OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"),
        KINECT_BLOCK_COUNT=os.environ.get("KINECT_BLOCK_COUNT", "5000"),
    )
    for name in dict.fromkeys(args.runs):
        device, tracking = name.split("-")
        runs = []
        for repeat in range(args.repeats):
            path = work / f"{name}-{repeat}.json"
            command = [
                sys.executable,
                str(ROOT / "scripts/replay_scan.py"),
                "--dataset",
                args.dataset,
                "--stride",
                str(args.stride),
                "--device",
                device,
                "--tracking",
                tracking,
                "--output",
                str(path),
            ]
            if args.path:
                command += ["--path", str(args.path)]
            if args.limit:
                command += ["--limit", str(args.limit)]
            print(
                f"Profiling {name}, repetition {repeat + 1}/{args.repeats}", flush=True
            )
            with path.with_suffix(".log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            if result.returncode:
                detail = path.with_suffix(".log").read_text()[-4000:]
                if device == "cuda" and "CUDA requested but unavailable" in detail:
                    outputs[name] = {
                        "available": False,
                        "reason": "CUDA unavailable on this host",
                    }
                    break
                raise RuntimeError(
                    f"{name} replay failed; see {path.with_suffix('.log')}"
                )
            runs.append(summarize(json.loads(path.read_text())))
        if runs:
            outputs[name] = {
                "available": True,
                "runs": runs,
                "median_elapsed_s": float(np.median([r["elapsed_s"] for r in runs])),
                "accepted_range": [
                    min(r["accepted"] for r in runs),
                    max(r["accepted"] for r in runs),
                ],
            }
    report = {
        "source_sha256": source_before,
        "source_changed_during_profile": source_hash() != source_before,
        "dataset": args.dataset,
        "stride": args.stride,
        "limit": args.limit,
        "repeats": args.repeats,
        "omp_threads": env["OMP_NUM_THREADS"],
        "initial_blocks": env["KINECT_BLOCK_COUNT"],
        "method": "sequential isolated processes; includes processing/final extraction; excludes startup",
        "memory": "peak process RSS; includes CPU allocations, not GPU VRAM",
        "runs": outputs,
        "note": "Compare pose error and acceptance before attributing a throughput gain",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
