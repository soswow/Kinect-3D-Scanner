"""Check CPU thread contention in the mixed CUDA scanning pipeline.

Sequential isolated matched replays. A quick prefix is a screening experiment,
not an entire-session speed claim. Each run records actual OpenCV threading.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
import json
import os
import subprocess

from scripts.profile_session import file_hash, source_hash
from scripts.profile_cuda_pipeline import KEYS, RECIPES

POLICIES = {
    "default-8": (8, None, None),
    "default-4": (4, None, None),
    "default-2": (2, None, None),
    "default-1": (1, None, None),
    "passive-8": (8, "PASSIVE", 1),
    "passive-4": (4, "PASSIVE", 1),
    "passive-1": (1, "PASSIVE", 1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=tuple(POLICIES))
    parser.add_argument("--recipe", choices=RECIPES, default="sift")
    parser.add_argument("--limit", type=int, default=28)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--finish", action="store_true", help="Include full Finish on each selected prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.limit < 1:
        parser.error("Require positive repeats and prefix limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = source_hash()
    results = []
    for path in args.sessions:
        checksum = file_hash(path)
        for repeat in range(args.repeats):
            for policy in args.policies if repeat % 2 == 0 else reversed(args.policies):
                threads, wait, cv_threads = POLICIES[policy]
                env = dict(os.environ, **dict(zip(KEYS, RECIPES[args.recipe])),
                           OMP_NUM_THREADS=str(threads), KINECT_NATIVE="on",
                           KINECT_BLOCK_COUNT="5000", KINECT_CUDA_FUSION="fused")
                for key in ("OMP_WAIT_POLICY", "KMP_BLOCKTIME", "OPENCV_FOR_THREADS_NUM", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    env.pop(key, None)
                if wait:
                    env.update(OMP_WAIT_POLICY=wait, KMP_BLOCKTIME="0",
                               OPENCV_FOR_THREADS_NUM=str(cv_threads), MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                destination = args.output.parent / f"{path.stem}-{policy}-{repeat}.json"
                print(f"{path.name}: {policy} {repeat+1}/{args.repeats}", flush=True)
                command = [sys.executable, str(ROOT/"scripts/profile_session.py"), str(path),
                    "--device", "cuda", "--tracking", "tensor", "--limit", str(args.limit), "--output", str(destination)]
                if args.finish:
                    command.append("--finish")
                with destination.with_suffix(".log").open("w") as log:
                    worker = subprocess.run(command,
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
                if worker.returncode:
                    raise RuntimeError(f"Thread experiment failed: {destination.with_suffix('.log')}")
                report = json.loads(destination.read_text())
                if (source_hash() != fingerprint or report["source_sha256"] != fingerprint
                        or report["source_changed_during_profile"] or report["input_sha256"] != checksum
                        or report["input_changed_during_profile"]):
                    raise RuntimeError("Source or input changed during thread sweep")
                row = {"session": path.name, "policy": policy, "repeat": repeat,
                       "report": str(destination.resolve()), "frames": report["frames"],
                       "live_s": report["live_s"], "accepted": report["accepted"],
                       "live_accepted": report["accepted_before_finish"], "finish_s": report["finish_s"],
                       "thread_policy": report["thread_policy"], "omp_threads": report["omp_threads"],
                       "accepted_indices": report["accepted_indices"], "stages": report["live_stages"]}
                results.append(row)
                args.output.write_text(json.dumps({"source_sha256": fingerprint, "recipe": args.recipe,
                    "limit": args.limit, "finish": args.finish, "input_sha256": {p.name: file_hash(p) for p in args.sessions},
                    "note": "Prefix screening only; differences in retained views must be assessed with timing.",
                    "runs": results}, indent=2, allow_nan=False)+"\n")
                print(f"  live={row['live_s']:.2f}s finish={row['finish_s']} retained={row['live_accepted']}/{row['frames']} final={row['accepted']} actual OpenCV threads={row['thread_policy']['opencv_threads']}", flush=True)


if __name__ == "__main__":
    main()
