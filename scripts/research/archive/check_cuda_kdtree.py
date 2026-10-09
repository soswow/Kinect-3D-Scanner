"""Validate isolated CUDA KD-tree candidates against legacy Open3D searches."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import open3d as o3d
from scripts.research.archive.cuda_kdtree_registration import KDTreeICP
from scripts.process_metrics import finish_cuda_worker


def cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def main():
    rng = np.random.default_rng(72)
    solver = KDTreeICP("CUDA:0", max_clouds=2)
    total = 0
    for size in (1, 17, 257, 4099):
        target = cloud(rng.normal(size=(size, 3)))
        # Include duplicates, an exact radius boundary, and dense near ties.
        if size > 17:
            np.asarray(target.points)[2] = np.asarray(target.points)[1]
        points = np.concatenate((np.asarray(target.points)[:10], rng.normal(size=(250, 3))))
        for radius in (.03, .12, 3.):
            query = cloud(points)
            pairs, _, _ = solver._correspondences(query, target, solver._dataset(target, radius), radius)
            actual = np.full(len(points), -1, dtype=int)
            actual[pairs[:, 0]] = pairs[:, 1]
            tree = o3d.geometry.KDTreeFlann(target)
            expected = []
            for point in points:
                count, ids, _ = tree.search_hybrid_vector_3d(point, radius, 1)
                expected.append(ids[0] if count else -1)
            np.testing.assert_array_equal(expected, actual)
            total += len(points)
        # Content mutation invalidates a cached dataset even at the same identity.
        uploads = solver.statistics["cloud_uploads"]
        np.asarray(target.points)[0, 0] += .1
        solver._dataset(target, .03)
        assert solver.statistics["cloud_uploads"] == uploads+1
        assert len(solver.cache) <= 2
    target = cloud([[0., 0., 0.]])
    query = cloud([[.03, 0., 0.], [np.nextafter(.03, 0.), 0., 0.]])
    pairs, _, _ = solver._correspondences(query, target, solver._dataset(target, .03), .03)
    np.testing.assert_array_equal([[1, 0]], pairs)
    output = ROOT / "benchmark-output/cuda-pipeline/kdtree-check.json"
    output.write_text(json.dumps({"queries_checked": total+2, "exact_indices": True,
                                 "statistics": solver.statistics}, indent=2)+"\n")
    print(output.read_text(), flush=True)
    o3d.core.cuda.synchronize()
    finish_cuda_worker()


if __name__ == "__main__":
    main()
