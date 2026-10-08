"""CUDA float64 neighbour search with the original legacy robust pose solve.

All original points, normals, radius stages and convergence criteria remain.
Dataset indexes persist across repeated verification of unchanged clouds.
Content hashes invalidate indexes even for clouds mutated in place.
"""

import copy
import hashlib
from collections import OrderedDict
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import open3d as o3d

REG = o3d.pipelines.registration


@lru_cache(maxsize=8)
def check_support(device_name):
    device = o3d.core.Device(device_name)
    points = o3d.core.Tensor([[0., 0., 0.], [1., 0., 0.]], o3d.core.float64, device)
    search = o3d.core.nns.NearestNeighborSearch(points, index_dtype=o3d.core.int32)
    search.hybrid_index(.03)
    indices, _, counts = search.hybrid_search(points, .03, 1)
    if not np.array_equal(indices.cpu().numpy().reshape(-1), [0, 1]) or not np.all(counts.cpu().numpy() == 1):
        raise RuntimeError("CUDA float64 hybrid search failed its setup check")


class NearestICP:
    def __init__(self, device, max_clouds=64, reranked=False):
        self.device = o3d.core.Device(str(device))
        if not str(self.device).startswith("CUDA"):
            raise ValueError("CUDA neighbour search requires a CUDA device")
        check_support(str(self.device))
        self.max_clouds = max_clouds
        self.reranked = reranked
        self.cache = OrderedDict()
        self.statistics = {"cloud_uploads": 0, "index_builds": 0, "cache_hits": 0, "searches": 0,
                           "pose_iterations": 0, "exact_cpu_queries": 0}

    def _dataset(self, target, radius):
        points = np.asarray(target.points)
        digest = hashlib.blake2b(memoryview(points).cast("B"), digest_size=16).digest()
        key = id(target)
        cached = self.cache.pop(key, None)
        if cached is None or cached[0] is not target or cached[1] != digest:
            rounded = points.astype(np.float32) if self.reranked else points
            error = float(np.max(np.linalg.norm(points-rounded.astype(np.float64), axis=1)))
            data = o3d.core.Tensor(rounded, o3d.core.float32 if self.reranked else o3d.core.float64, self.device)
            cached = (target, digest, data, {}, error)
            self.statistics["cloud_uploads"] += 1
        else:
            self.statistics["cache_hits"] += 1
        self.cache[key] = cached
        while len(self.cache) > self.max_clouds:
            self.cache.popitem(last=False)
        indexes = cached[3]
        if radius not in indexes:
            search = o3d.core.nns.NearestNeighborSearch(cached[2], index_dtype=o3d.core.int32)
            search.hybrid_index(self._radius(radius))
            indexes[radius] = search
            self.statistics["index_builds"] += 1
        return indexes[radius], cached[4]

    def _radius(self, radius):
        return float(np.nextafter(np.float32(radius+1e-5), np.float32(np.inf))) \
            if self.reranked else radius*(1+1e-14)

    def _correspondences(self, source, target, search, radius):
        points = np.asarray(source.points)
        search, target_error = search
        rounded = points.astype(np.float32) if self.reranked else points
        query = o3d.core.Tensor(rounded, o3d.core.float32 if self.reranked else o3d.core.float64, self.device)
        indices, gpu_distances, counts = search.hybrid_search(query, self._radius(radius), 2 if self.reranked else 1)
        if self.reranked:
            candidates = indices.cpu().numpy()
            counts = counts.cpu().numpy().reshape(-1)
            valid = (np.arange(2) < counts[:, None]) & (candidates >= 0)
            difference = points[:, None, :] - np.asarray(target.points)[np.maximum(candidates, 0)]
            squared = np.sum(difference*difference, axis=2)
            squared[~valid] = np.inf
            best = np.argmin(squared, axis=1)
            nearest = candidates[np.arange(len(points)), best].copy()
            exact_distance = np.sqrt(squared[np.arange(len(points)), best])
            last = np.sqrt(np.max(gpu_distances.cpu().numpy(), axis=1).astype(np.float64))
            error = np.linalg.norm(points-rounded.astype(np.float64), axis=1) + target_error
            metric_error = 8*np.finfo(np.float32).eps*np.maximum(last, radius)
            # If another candidate could beat this one after exact reranking,
            # use the original CPU tree for that query. Exact ties also preserve
            # the legacy tree's choice. This is retrieval, not float32 geometry.
            uncertain = ((counts >= 2) & (last-error-metric_error <= exact_distance)) \
                        | (error+metric_error >= self._radius(radius)-radius)
            if np.any(uncertain):
                tree = o3d.geometry.KDTreeFlann(target)
                for row in np.flatnonzero(uncertain):
                    count, matched, _ = tree.search_hybrid_vector_3d(points[row], radius, 1)
                    nearest[row] = matched[0] if count else -1
                self.statistics["exact_cpu_queries"] += int(np.count_nonzero(uncertain))
            selected = np.flatnonzero(nearest >= 0)
        else:
            nearest = indices.cpu().numpy().reshape(-1)
            selected = np.flatnonzero(counts.cpu().numpy().reshape(-1) > 0)
        difference = points[selected] - np.asarray(target.points)[nearest[selected]]
        distances = np.sum(difference*difference, axis=1)
        # Preserve the legacy strict radius boundary using the original CPU
        # coordinates. The inflated search radius only guards GPU roundoff.
        inside = distances < radius*radius
        selected, distances = selected[inside], distances[inside]
        pairs = np.column_stack((selected, nearest[selected])).astype(np.int32)
        self.statistics["searches"] += 1
        # Keep metric accumulation on CPU alongside the original estimator.
        rmse = float(np.sqrt(np.sum(distances) / len(selected))) if len(selected) else 0.
        return pairs, len(selected)/len(points), rmse

    def match(self, source, target, initial):
        pose = np.asarray(initial, dtype=np.float64).copy()
        if not len(source.points) or not len(target.points):
            return SimpleNamespace(transformation=pose, fitness=0., inlier_rmse=0.,
                                   correspondence_set=np.empty((0, 2), dtype=np.int32))
        estimator = REG.TransformationEstimationPointToPlane(REG.HuberLoss(.01))
        for radius, iterations in ((.12, 40), (.06, 30), (.03, 20)):
            search = self._dataset(target, radius)
            moving = o3d.geometry.PointCloud()
            moving.points = o3d.utility.Vector3dVector(np.asarray(source.points))
            if not np.array_equal(pose, np.eye(4)):
                moving.transform(pose)
            pairs, fitness, rmse = self._correspondences(moving, target, search, radius)
            for _ in range(iterations):
                update = estimator.compute_transformation(moving, target, o3d.utility.Vector2iVector(pairs))
                pose = update @ pose
                moving.transform(update)
                old_fitness, old_rmse = fitness, rmse
                pairs, fitness, rmse = self._correspondences(moving, target, search, radius)
                self.statistics["pose_iterations"] += 1
                if abs(old_fitness-fitness) < 1e-6 and abs(old_rmse-rmse) < 1e-6:
                    break
        return SimpleNamespace(transformation=pose, fitness=fitness, inlier_rmse=rmse,
                               correspondence_set=pairs)
