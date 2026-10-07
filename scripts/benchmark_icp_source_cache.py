"""Interleave identical ICP proposal batches with and without source-level reuse.

This isolates repeated source preparation, not full live capture throughput.
Synthetic reference poses score results only; they never initialize ICP.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d

from scanner_server.engine import ScanEngine


def setup(tracking):
    engine = ScanEngine(device="cpu", tracking=tracking)
    rng = np.random.default_rng(21)
    points = rng.uniform((-0.6, -0.5, 1.1), (0.6, 0.5, 2.2), (100000, 3))
    points[:40000, 2] = 2.1
    points[40000:70000, 0] = -0.4
    points[70000:, 1] = 0.3
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    target = target.voxel_down_sample(engine.reg_voxel)
    truth = np.eye(4)
    truth[:3, :3] = cv2.Rodrigues(np.array([0.004, 0.012, -0.003]))[0]
    truth[:3, 3] = [0.015, 0.003, 0]
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        (np.asarray(target.points) - truth[:3, 3]) @ truth[:3, :3]))
    engine.model_pcd = target
    for scale in (4, 2, 1):
        voxel = engine.reg_voxel * scale
        level = target.voxel_down_sample(voxel)
        level.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
        engine._model_pyramid[scale] = level
        if tracking == "tensor":
            engine._tensor_model_pyramid[scale] = o3d.t.geometry.PointCloud.from_legacy(
                level, dtype=o3d.core.float32, device=engine.device)
    return engine, source, target, truth


def run(engine, source, target, guesses, cached):
    results = []

    def proposals(current, rgbd):
        results.extend(engine._icp(current, target, guess) for guess in guesses)
        return results[-1]

    with patch.object(engine, "_visual_register", side_effect=proposals):
        started = time.perf_counter()
        if cached:
            engine._register(source)
        else:
            engine._register.__wrapped__(engine, source)
        elapsed = (time.perf_counter() - started) * 1000
    if hasattr(engine, "_icp_source_pyramid"):
        raise RuntimeError("Source cache leaked outside registration decision")
    return results, elapsed


def correspondences(result):
    pairs = np.asarray(result.correspondence_set)
    return pairs[np.lexsort(pairs.T)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--tracking", choices=("legacy", "tensor"), default="legacy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Require positive repeats")
    engine, source, target, truth = setup(args.tracking)
    report = {"schema_version": 1, "repeats": args.repeats,
              "backend": engine.backend, "source_points": len(source.points),
              "input_sha256": hashlib.sha256(np.asarray(source.points).tobytes() +
                                             np.asarray(target.points).tobytes()).hexdigest(),
              "source_sha256": hashlib.sha256(
                  (ROOT / "scanner_server/engine.py").read_bytes() +
                  (ROOT / "scanner_server/tracking_cache.py").read_bytes()).hexdigest(),
              "workloads": {},
              "notes": ["Synthetic static three-surface observation with metric pose used for scoring only",
                        "Interleaved registration decision batches after one warmup per variant",
                        "Identical 40/30/20 iteration budgets, robust losses and source/target voxel sizes",
                        "Full visual retrieval, feature verification, fusion and acquisition not benchmarked"]}
    for mode, current_target in (("cached_model", target), ("raw_observation", copy.deepcopy(target))):
        for attempts in (1, 5):
            guesses = [np.eye(4) for _ in range(attempts)]
            for i, guess in enumerate(guesses):
                guess[0, 3] = i * 0.004
            for cached in (False, True):
                run(engine, source, current_target, guesses, cached)
            times = {"before": [], "after": []}
            differences, rmse_differences, support_changed, match_changed, errors = [], [], [], [], []
            for repeat in range(args.repeats):
                results = {}
                for name, cached in (("before", False), ("after", True)) if repeat % 2 else (
                        ("after", True), ("before", False)):
                    results[name], elapsed = run(engine, source, current_target, guesses, cached)
                    times[name].append(elapsed)
                for a, b in zip(results["before"], results["after"]):
                    differences.append(float(np.max(np.abs(a.transformation - b.transformation))))
                    support_changed.append(a.fitness != b.fitness)
                    rmse_differences.append(abs(a.inlier_rmse - b.inlier_rmse))
                    match_changed.append(not np.array_equal(correspondences(a), correspondences(b)))
                    errors.append(float(np.linalg.norm((np.linalg.inv(truth) @ b.transformation)[:3, 3])))
            medians = {name: float(np.median(values)) for name, values in times.items()}
            record = {"median_ms": medians, "median_speedup": medians["before"] / medians["after"],
                      "samples_ms": times, "max_pose_matrix_difference": max(differences),
                      "max_rmse_difference_m": max(rmse_differences),
                      "changed_support": sum(support_changed), "changed_correspondences": sum(match_changed),
                      "max_translation_error_m": max(errors)}
            report["workloads"][f"{mode}_{attempts}_attempts"] = record
            print(mode, attempts, json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
