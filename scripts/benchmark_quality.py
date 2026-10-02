"""Isolated repeated reconstruction scored against withheld RGB-D views.

Reference poses are read only by the scorer. Training/withheld RGB and depth
captures are disjoint. Input/source hashes and all failures stay in the report.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d
from replay_scan import capture_metadata, load_dataset, score

from scanner_server.engine import ScanEngine
from shared.view_metrics import heldout_view_metrics

VARIANTS = {
    "uniform": {},
    "confidence": {"confidence_fusion": True},
    "relocalize": {"relocalize": True},
    "refine": {"refine_poses": True},
}


def source_hash():
    digest = hashlib.sha256()
    paths = [
        p
        for folder in ("scanner_server", "shared")
        for p in (ROOT / folder).glob("*.py")
    ]
    paths += [Path(__file__), ROOT / "scripts/replay_scan.py"]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def input_hash(settings, training, withheld):
    digest = hashlib.sha256(json.dumps(settings.to_dict(), sort_keys=True).encode())
    for label, frames in ((b"training", training), (b"withheld", withheld)):
        digest.update(label)
        for rgb, depth, stamp, reference in frames:
            digest.update(str(stamp).encode())
            digest.update(rgb.tobytes())
            digest.update(depth.tobytes())
            digest.update(b"no pose" if reference is None else reference.tobytes())
    return digest.hexdigest()


def disjoint_views(training, withheld):
    # Timestamp identity includes the depth association, already one-to-one in the reader.
    if not training or not withheld:
        raise ValueError("Require nonempty training and withheld views")
    if {f[2] for f in training} & {f[2] for f in withheld}:
        raise ValueError("Training and withheld captures overlap")


def run_worker(args):
    cv2.setRNGSeed(args.seed)
    o3d.utility.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    settings, training = load_dataset(args.dataset, args.path, args.stride, args.limit)
    _, withheld = load_dataset(
        args.dataset, args.path, args.stride, args.limit, args.stride // 2
    )
    disjoint_views(training, withheld)
    digest = input_hash(settings, training, withheld)
    settings = replace(settings, **VARIANTS[args.worker])
    engine = ScanEngine(device="cpu")
    engine.reset(settings=settings)
    started = time.monotonic()
    for i, (rgb, depth, _, _) in enumerate(training):
        engine.store_frame(rgb, depth, capture_metadata(training[i], i))
        engine.process_frames()
    built, build_result = engine.build_mesh()
    elapsed = time.monotonic() - started
    paired = [
        (p, training[i][3]) for i, p in engine.poses if training[i][3] is not None
    ]
    if paired:
        estimated, reference = paired[0]
        anchor = estimated @ np.linalg.inv(reference)
        anchor_note = "First accepted camera with an evaluation reference pose"
    elif not built:
        # Empty geometry has no hits in any coordinate frame. Keep its penalty.
        anchor = np.eye(4)
        anchor_note = (
            "No accepted reference camera; absent surface receives missing penalty"
        )
    else:
        raise ValueError("Built mesh has no accepted evaluation reference camera")
    view_metrics = heldout_view_metrics(
        engine.mesh if built else None,
        withheld,
        settings,
        anchor,
        threshold_m=args.threshold_mm / 1000,
        pixel_stride=args.pixel_stride,
    )
    view_metrics["anchor_note"] = anchor_note
    report = {
        "variant": args.worker,
        "seed": args.seed,
        "input_sha256": digest,
        "source_sha256": source_hash(),
        "settings": settings.to_dict(),
        "backend": engine.backend,
        "versions": {
            "open3d": o3d.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "training_frames": len(training),
        "withheld_frames": len(withheld),
        "accepted": engine.frame_count,
        "mesh_built": built,
        "processing_s": elapsed,
        "build": build_result,
        "refinement": engine.refinement,
        "trajectory": score(engine.poses, training),
        "heldout_views": view_metrics,
        "rejections": [
            {"index": d["index"], "message": d["message"]}
            for d in engine.diagnostics
            if not d["success"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


def aggregate(runs):
    def distribution(values):
        values = [v for v in values if v is not None]
        return (
            {"min": min(values), "median": float(np.median(values)), "max": max(values)}
            if values
            else None
        )

    out = {}
    for name in dict.fromkeys(r["variant"] for r in runs):
        selected = [r for r in runs if r["variant"] == name]
        out[name] = {
            "runs": len(selected),
            "mesh_successes": sum(r["mesh_built"] for r in selected),
            "accepted": distribution([r["accepted"] for r in selected]),
            "trajectory_rmse_m": distribution(
                [r["trajectory"].get("anchored_translation_rmse_m") for r in selected]
            ),
            "depth_rmse_over_hits_m": distribution(
                [r["heldout_views"]["depth_rmse_over_hits_m"] for r in selected]
            ),
            "hit_fraction": distribution(
                [r["heldout_views"]["hit_fraction"] for r in selected]
            ),
            "completeness_within_threshold": distribution(
                [r["heldout_views"]["completeness_within_threshold"] for r in selected]
            ),
            "capped_rmse_including_missing_m": distribution(
                [
                    r["heldout_views"]["capped_rmse_including_missing_m"]
                    for r in selected
                ]
            ),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["tum", "recording", "redwood"], default="tum"
    )
    parser.add_argument("--path", type=Path)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=["uniform", "confidence"]
    )
    parser.add_argument("--threshold-mm", type=float, default=10)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-output/quality.json"
    )
    parser.add_argument("--worker", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.stride < 2
        or args.repeats < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Require stride >=2, positive repeats/limit")
    if not 0 < args.threshold_mm <= 100 or not 1 <= args.pixel_stride <= 16:
        parser.error("Require threshold 0–100 mm and pixel stride 1–16")
    if args.worker:
        run_worker(args)
        return
    before = source_hash()
    work = args.output.parent / (args.output.stem + "-runs")
    work.mkdir(parents=True, exist_ok=True)
    runs = []
    # Interleave variants, isolated processes, sequential so timings do not overlap.
    for seed in range(args.repeats):
        for name in dict.fromkeys(args.variants):
            path = work / f"{name}-{seed}.json"
            command = [
                sys.executable,
                str(Path(__file__)),
                "--worker",
                name,
                "--seed",
                str(seed),
                "--dataset",
                args.dataset,
                "--stride",
                str(args.stride),
                "--threshold-mm",
                str(args.threshold_mm),
                "--pixel-stride",
                str(args.pixel_stride),
                "--output",
                str(path),
            ]
            if args.path:
                command += ["--path", str(args.path)]
            if args.limit:
                command += ["--limit", str(args.limit)]
            print(f"Quality replay {seed + 1}/{args.repeats}: {name}", flush=True)
            with path.with_suffix(".log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            if result.returncode:
                raise RuntimeError(f"Replay failed; inspect {path.with_suffix('.log')}")
            runs.append(json.loads(path.read_text()))
            # Preserve completed checkpoints even if a later run is interrupted.
            report = {
                "dataset": args.dataset,
                "stride": args.stride,
                "limit": args.limit,
                "source_sha256": before,
                "source_changed": before != source_hash(),
                "input_sha256": runs[0]["input_sha256"],
                "matched_inputs": len({r["input_sha256"] for r in runs}) == 1,
                "requested_repeats": args.repeats,
                "completed_runs": len(runs),
                "summary": aggregate(runs),
                "runs": runs,
                "notes": [
                    "Seeds do not guarantee bitwise Open3D/hash-map determinism",
                    "Withheld depth shares sensor/calibration noise; this is not independent absolute accuracy",
                    "No reference pose/image initializes or refines reconstruction",
                    "Compare coverage and error together; missing mesh receives a penalty",
                ],
            }
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps(report["summary"][name]), flush=True)


if __name__ == "__main__":
    main()
