"""Standalone resident FP64 ICP research; never imported by the scanner backend.

Prepare the tiny CPU bridge without starting CUDA:
  python scripts/research_resident_icp.py --emit-solve-source benchmark-output/cuda-pipeline/resident-icp/solve.cpp

The optional build command requires the exact Eigen archive pinned by Open3D
0.20 and a matching extracted include tree; it does not download dependencies.
Use a Visual Studio developer shell for --build-solver. Hardware execution is
reserved for an explicitly allocated benchmark slot after exact RTX retrieval
proof. ResidentICP expects OptixICP.nearest_device to resolve CPU-only ties and
uncertain/missing rows before returning device IDs. Those exceptional copies
remain in the retrieval counters, so "resident" does not imply zero fallback.
--radius-bins selects fractions of each original ICP radius; the exact increasing
sequence ending in 1 must match both prerequisite proof phases. Artifact hashes
also bind staged retrieval and its original CPU uncertainty/miss policy.
The optional direct-miss research policy additionally requires synthetic and
real proofs that shadow every declared GPU miss with the original CPU tree.

The GPU fast path uploads original points/normals once per match, transforms
points and reduces original Huber point-to-plane equations in FP64. It copies
21 symmetric matrix terms, six gradients and three validity/metric scalars,
then uses a separate Eigen LDLT bridge and Open3D's Rz*Ry*Rx convention. Final
correspondences are copied once for the unmodified independent CPU pose gates.
GPU reduction/transform order and bridge compilation change roundoff; this is
not the original Open3D binary solve or a claim of numerical equivalence.

Mathematical sources (Open3D v0.20.0, MIT):
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/pipelines/registration/TransformationEstimation.cpp
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/pipelines/registration/RobustKernel.cpp
https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/utility/Eigen.cpp
Eigen headers are MPL-2.0; retain the downloaded archive and COPYING.MPL2.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EIGEN_COMMIT = "da7909592376c893dabbc4b6453a8ffe46b1eb8e"
EIGEN_ARCHIVE_SHA256 = "37f71e1d7c408e2cc29ef90dcda265e2de0ad5ed6249d7552f6c33897eafd674"
EIGEN_ARCHIVE_URL = f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_COMMIT}/eigen-{EIGEN_COMMIT}.tar.gz"
STAGES = ((.12, 40), (.06, 30), (.03, 20))
MISS_POLICIES = ("cpu-fallback", "direct-miss-research-v1")
TERMS, BLOCK = 30, 128

CPU_BRIDGE_SOURCE = r"""
// SPDX-License-Identifier: MIT
// Standalone reimplementation of the Open3D v0.20.0 Eigen LDLT/Euler solve.
// Uses pinned Eigen headers (MPL-2.0); does not call the Open3D binary.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#if defined(_WIN32)
#define RESEARCH_API __declspec(dllexport)
#else
#define RESEARCH_API __attribute__((visibility("default")))
#endif
#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)
extern "C" RESEARCH_API const char* resident_eigen_version() {
    return STR(EIGEN_WORLD_VERSION) "." STR(EIGEN_MAJOR_VERSION) "." STR(EIGEN_MINOR_VERSION);
}
extern "C" RESEARCH_API int resident_solve(
        const double* matrix, const double* gradient, double* output) {
    if (!matrix || !gradient || !output) return 1;
    try {
        using Row6 = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;
        // The original utility solver materializes dynamic matrices/vectors.
        Eigen::MatrixXd A = Eigen::Map<const Row6>(matrix);
        Eigen::VectorXd b = -Eigen::Map<const Eigen::Matrix<double, 6, 1>>(gradient);
        if (!A.allFinite() || !b.allFinite()) return 2;
        auto factor = A.ldlt();
        if (factor.info() != Eigen::Success || !factor.isPositive()) return 3;
        const auto pivots = factor.vectorD().cwiseAbs().eval();
        const double largest = pivots.maxCoeff();
        // Research guard: ill-conditioned/degenerate systems redo original CPU ICP.
        if (!(largest > 0.) || pivots.minCoeff() <= largest * 1e-12) return 4;
        Eigen::VectorXd x = factor.solve(b);
        if (factor.info() != Eigen::Success || !x.allFinite()) return 5;
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) =
            (Eigen::AngleAxisd(x(2), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(x(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(x(0), Eigen::Vector3d::UnitX())).matrix();
        pose.block<3, 1>(0, 3) = x.tail<3>();
        Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> result(output);
        result = pose;
        return pose.allFinite() ? 0 : 6;
    } catch (...) {
        return 7;
    }
}
"""

GPU_SOURCE = r"""
extern "C" __global__ void transform_points(
        const double* original, double* moving, const double* pose, int count) {
    int row = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= count) return;
    double x = original[3*row], y = original[3*row+1], z = original[3*row+2];
    double w = ((pose[12]*x + pose[13]*y) + pose[14]*z) + pose[15];
    double a = ((pose[0]*x + pose[1]*y) + pose[2]*z) + pose[3];
    double b = ((pose[4]*x + pose[5]*y) + pose[6]*z) + pose[7];
    double c = ((pose[8]*x + pose[9]*y) + pose[10]*z) + pose[11];
    moving[3*row] = a/w;
    moving[3*row+1] = b/w;
    moving[3*row+2] = c/w;
}
extern "C" __global__ void normal_partials(
        const double* moving, const double* target, const double* normals,
        const int* nearest, int* filtered, int count, int target_count,
        double radius_squared, double* partials) {
    // 21 upper-triangle terms, 6 gradients, squared distance, count, invalid count.
    __shared__ double values[30][128];
    int lane = int(threadIdx.x);
    int row = int(blockIdx.x * blockDim.x + threadIdx.x);
    double J[6] = {0., 0., 0., 0., 0., 0.};
    double residual = 0., weight = 0., distance = 0., valid = 0., invalid = 0.;
    if (row < count) {
        int match = nearest[row];
        double x = moving[3*row], y = moving[3*row+1], z = moving[3*row+2];
        if (!isfinite(x) || !isfinite(y) || !isfinite(z) ||
                match < -1 || match >= target_count) {
            invalid = 1.;
        } else if (match >= 0) {
            double dx = x-target[3*match], dy = y-target[3*match+1],
                   dz = z-target[3*match+2];
            distance = (dx*dx + dy*dy) + dz*dz;
            if (!isfinite(distance)) {
                invalid = 1.;
            } else if (distance < radius_squared) {
                double nx = normals[3*match], ny = normals[3*match+1],
                       nz = normals[3*match+2];
                residual = (dx*nx + dy*ny) + dz*nz;
                weight = .01 / fmax(fabs(residual), .01);
                J[0] = y*nz-z*ny; J[1] = z*nx-x*nz; J[2] = x*ny-y*nx;
                J[3] = nx; J[4] = ny; J[5] = nz;
                valid = 1.;
                if (!isfinite(residual) || !isfinite(weight)) {
                    invalid = 1.; valid = 0.; weight = 0.;
                }
            } else {
                distance = 0.;
            }
        }
        filtered[row] = valid ? match : -1;
    }
    int term = 0;
    for (int a=0; a<6; ++a)
        for (int b=a; b<6; ++b)
            values[term++][lane] = valid ? (J[a]*weight)*J[b] : 0.;
    for (int a=0; a<6; ++a)
        values[term++][lane] = valid ? (J[a]*weight)*residual : 0.;
    values[27][lane] = valid ? distance : 0.;
    values[28][lane] = valid;
    values[29][lane] = invalid;
    __syncthreads();
    for (int stride=64; stride>0; stride/=2) {
        if (lane < stride)
            for (int k=0; k<30; ++k)
                values[k][lane] += values[k][lane+stride];
        __syncthreads();
    }
    if (lane == 0)
        for (int k=0; k<30; ++k) partials[30*blockIdx.x+k] = values[k][0];
}
extern "C" __global__ void collapse_partials(
        const double* partials, double* totals, int blocks) {
    int term = int(threadIdx.x);
    if (term >= 30) return;
    double value = 0.;
    for (int block=0; block<blocks; ++block) value += partials[30*block+term];
    totals[term] = value;
}
"""


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_hash():
    digest = hashlib.sha256()
    for folder in (ROOT / "scanner_server", ROOT / "shared", ROOT / "native"):
        for path in sorted(folder.rglob("*")):
            if path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".toml") and "build" not in path.parts:
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def checked_radius_bins(values):
    """Require an unambiguous policy without importing numerical libraries."""
    if (not isinstance(values, (list, tuple)) or not values
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or value <= 0 or value > 1 for value in values)):
        raise ValueError("Radius bins must be finite positive fractions no larger than 1")
    bins = tuple(float(value) for value in values)
    if bins[-1] != 1. or any(a >= b for a, b in zip(bins, bins[1:])):
        raise ValueError("Radius bins must be strictly increasing, unique and end at the full radius 1")
    return bins


class ResidentFallback(RuntimeError):
    """A numerical/unsupported research path must redo the full original CPU call."""


class EigenSolve:
    """Load a pinned standalone Eigen bridge; never substitute another solver."""

    def __init__(self, library):
        import numpy as np

        self.np = np
        self.path = Path(library).resolve()
        manifest = self.path.with_suffix(".build.json")
        self.metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (self.metadata["eigen_archive_sha256"] != EIGEN_ARCHIVE_SHA256
                or self.metadata["bridge_source_sha256"] != hashlib.sha256(CPU_BRIDGE_SOURCE.encode()).hexdigest()
                or self.metadata["library_sha256"] != file_hash(self.path)):
            raise ValueError("CPU solve bridge does not match its pinned source/header/library manifest")
        self.library = ctypes.CDLL(str(self.path))
        self.library.resident_solve.argtypes = (ctypes.c_void_p,) * 3
        self.library.resident_solve.restype = ctypes.c_int
        self.library.resident_eigen_version.restype = ctypes.c_char_p
        self.metadata["reported_eigen_version"] = self.library.resident_eigen_version().decode()

    def __call__(self, matrix, gradient):
        np = self.np
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        gradient = np.ascontiguousarray(gradient, dtype=np.float64)
        if matrix.shape != (6, 6) or gradient.shape != (6,):
            raise ValueError("Expected a 6x6 system and six gradients")
        result = np.empty((4, 4), dtype=np.float64)
        code = self.library.resident_solve(matrix.ctypes.data, gradient.ctypes.data, result.ctypes.data)
        if code:
            raise ResidentFallback(f"Eigen bridge rejected system (status {code})")
        return result


class ResidentICP:
    """Research adapter: retain original CPU gates; no production registration."""

    def __init__(self, retrieval, solve, *, cpu_fallback=None, max_points=1_000_000,
                 max_scratch_bytes=256 * 1024**2, gpu_timing=True, owns_retrieval=False):
        import cupy as cp
        import numpy as np

        if not callable(getattr(retrieval, "nearest_device", None)):
            raise ValueError("Require proven OptixICP.nearest_device with original CPU ambiguity fallback")
        if not callable(solve) or max_points < 1 or max_scratch_bytes < 1:
            raise ValueError("Require an explicit Eigen solve and positive memory limits")
        self.cp, self.np, self.retrieval, self.solve = cp, np, retrieval, solve
        self.device_id = retrieval.device_id
        self.owns_retrieval = owns_retrieval
        self.cpu_fallback = cpu_fallback or self._original_cpu_match
        self.max_points, self.max_scratch_bytes = max_points, max_scratch_bytes
        self.gpu_timing = gpu_timing
        self.events, self.fallback_reasons = [], Counter()
        self.provenance = {"source_sha256": runtime_source_hash(),
            "script_sha256": file_hash(Path(__file__)),
            "gpu_source_sha256": hashlib.sha256(GPU_SOURCE.encode()).hexdigest(),
            "cpu_bridge_source_sha256": hashlib.sha256(CPU_BRIDGE_SOURCE.encode()).hexdigest(),
            "retrieval_script_sha256": file_hash(ROOT / "scripts/cuda_optix_registration.py"),
            "solve_metadata": getattr(solve, "metadata", None),
            "retrieval_miss_policy": getattr(retrieval, "miss_policy", None),
            "authority": "unmeasured standalone roundoff-changing research; original CPU gates remain mandatory",
            "timing_scope": "GPU event times and host call/enqueue times overlap; never sum them as total wall time"}
        self.statistics = {key: 0 for key in ("calls", "pose_iterations", "cpu_fallback_calls",
            "host_to_device_bytes", "device_to_host_bytes", "peak_owned_gpu_bytes")}
        self.statistics.update({key: 0. for key in ("wall_s", "upload_s", "transform_enqueue_s",
            "nn_wall_s", "equation_enqueue_s", "system_copy_s", "solve_s", "final_pair_copy_s",
            "event_collection_s", "cpu_fallback_s", "gpu_transform_ms", "gpu_equations_ms")})
        with cp.cuda.Device(self.device_id), cp.cuda.Stream.null:
            self.transform_kernel = cp.RawKernel(GPU_SOURCE, "transform_points", options=("--fmad=false",))
            self.partial_kernel = cp.RawKernel(GPU_SOURCE, "normal_partials", options=("--fmad=false",))
            self.collapse_kernel = cp.RawKernel(GPU_SOURCE, "collapse_partials", options=("--fmad=false",))

    @staticmethod
    def _original_cpu_match(source, target, initial):
        import open3d as o3d

        pose = initial
        registration = o3d.pipelines.registration
        for radius, iterations in STAGES:
            result = registration.registration_icp(source, target, radius, pose,
                registration.TransformationEstimationPointToPlane(registration.HuberLoss(.01)),
                registration.ICPConvergenceCriteria(max_iteration=iterations))
            pose = result.transformation
        return result

    def _event_start(self):
        if not self.gpu_timing:
            return None
        event = self.cp.cuda.Event()
        event.record(self.cp.cuda.Stream.null)
        return event

    def _event_end(self, name, start):
        if start is not None:
            end = self.cp.cuda.Event()
            end.record(self.cp.cuda.Stream.null)
            self.events.append((name, start, end))

    def _collect_events(self):
        started = time.perf_counter()
        for name, start, end in self.events:
            end.synchronize()
            self.statistics[name] += self.cp.cuda.get_elapsed_time(start, end)
        self.events.clear()
        self.statistics["event_collection_s"] += time.perf_counter() - started

    def _transform(self, original, moving, pose, pose_buffer):
        started = time.perf_counter()
        pose_buffer.set(self.np.ascontiguousarray(pose, dtype=self.np.float64))
        self.statistics["host_to_device_bytes"] += pose_buffer.nbytes
        event = self._event_start()
        self.transform_kernel(((len(original)+BLOCK-1)//BLOCK,), (BLOCK,),
            (original, moving, pose_buffer, self.np.int32(len(original))))
        self._event_end("gpu_transform_ms", event)
        self.statistics["transform_enqueue_s"] += time.perf_counter() - started

    def _equations(self, moving, target, normals, nearest, filtered, radius, partials, totals):
        started = time.perf_counter()
        event = self._event_start()
        blocks = len(partials)
        self.partial_kernel((blocks,), (BLOCK,), (moving, target, normals, nearest, filtered,
            self.np.int32(len(moving)), self.np.int32(len(target)), self.np.float64(radius*radius), partials))
        self.collapse_kernel((1,), (32,), (partials, totals, self.np.int32(blocks)))
        self._event_end("gpu_equations_ms", event)
        self.statistics["equation_enqueue_s"] += time.perf_counter() - started
        started = time.perf_counter()
        values = self.cp.asnumpy(totals)
        self.statistics["system_copy_s"] += time.perf_counter() - started
        self.statistics["device_to_host_bytes"] += values.nbytes
        if values[29] or not self.np.isfinite(values).all():
            raise ResidentFallback("Invalid resident coordinates, correspondence IDs or equations")
        matrix = self.np.zeros((6, 6), dtype=self.np.float64)
        upper = self.np.triu_indices(6)
        matrix[upper] = values[:21]
        matrix[(upper[1], upper[0])] = values[:21]
        count = int(values[28])
        return matrix, values[21:27], count / len(moving), (values[27]/count)**.5 if count else 0., count

    def _iteration_data(self, moving, item, normals, filtered, radius, partials, totals):
        started = time.perf_counter()
        nearest, distances = self.retrieval.nearest_device(moving, item, radius)
        self.statistics["nn_wall_s"] += time.perf_counter() - started
        if (nearest.dtype != self.cp.int32 or nearest.shape != (len(moving),)
                or nearest.device.id != self.device_id or not nearest.flags.c_contiguous):
            raise ValueError("nearest_device returned an incompatible ID buffer")
        # Recompute strict-radius metrics from original FP64 coordinates, as legacy ICP does.
        return self._equations(moving, item["data"], normals, nearest, filtered, radius, partials, totals)

    def match(self, source, target, initial):
        started = time.perf_counter()
        try:
            with self.cp.cuda.Device(self.device_id), self.cp.cuda.Stream.null:
                try:
                    result = self._match_resident(source, target, initial)
                except ResidentFallback as error:
                    # Discard partial local pose; redo the original call from its original seed.
                    # A device execution fault must propagate, rather than pretend CPU recovered it.
                    self.cp.cuda.Stream.null.synchronize()
                    self.statistics["cpu_fallback_calls"] += 1
                    self.fallback_reasons[str(error)] += 1
                    fallback_started = time.perf_counter()
                    try:
                        result = self.cpu_fallback(source, target, initial)
                    finally:
                        self.statistics["cpu_fallback_s"] += time.perf_counter()-fallback_started
                self._collect_events()
                return result
        finally:
            self.events.clear()
            self.statistics["calls"] += 1
            self.statistics["wall_s"] += time.perf_counter() - started

    def _match_resident(self, source, target, initial):
        cp, np = self.cp, self.np
        pose = np.asarray(initial, dtype=np.float64).copy()
        points, target_points, normal_points = np.asarray(source.points), np.asarray(target.points), np.asarray(target.normals)
        if not len(points) or not len(target_points):
            return SimpleNamespace(transformation=pose, fitness=0., inlier_rmse=0.,
                                   correspondence_set=np.empty((0, 2), dtype=np.int32))
        if (pose.shape != (4, 4) or normal_points.shape != target_points.shape
                or not np.isfinite(pose).all() or not np.isfinite(points).all()
                or not np.isfinite(target_points).all() or not np.isfinite(normal_points).all()
                or len(points) > self.max_points or len(target_points) > self.max_points
                or np.max(np.abs(points)) > 1e6 or np.max(np.abs(target_points)) > 1e6):
            raise ResidentFallback("Unsupported/invalid original point, normal, seed or cloud-size regime")
        blocks = (len(points)+BLOCK-1)//BLOCK
        # Target XYZ/BVH storage belongs to the separate bounded retrieval cache.
        # This covers all owned arrays plus expected per-call nearest result buffers.
        scratch = (2*points.nbytes + normal_points.nbytes + blocks*TERMS*8 + TERMS*8 + 16*8
                   + len(points)*(4+8+4+4+8+16+8))
        if scratch > self.max_scratch_bytes:
            raise ResidentFallback("Resident scratch byte budget exceeded")
        self.statistics["peak_owned_gpu_bytes"] = max(self.statistics["peak_owned_gpu_bytes"], scratch)
        started = time.perf_counter()
        original, moving = cp.asarray(points), cp.empty(points.shape, dtype=cp.float64)
        normals = cp.asarray(normal_points)
        filtered = cp.empty(len(points), cp.int32)
        partials, totals = cp.empty((blocks, TERMS), cp.float64), cp.empty(TERMS, cp.float64)
        pose_buffer = cp.empty((4, 4), cp.float64)
        self.statistics["host_to_device_bytes"] += original.nbytes + normals.nbytes
        self.statistics["upload_s"] += time.perf_counter() - started
        for radius, iterations in STAGES:
            item = self.retrieval._dataset(target, radius)
            if (item["data"] is None or item["data"].dtype != cp.float64
                    or item["data"].shape != target_points.shape
                    or item["data"].device.id != self.device_id
                    or not item["data"].flags.c_contiguous):
                raise ResidentFallback("Retrieval cache is CPU-only or has incompatible resident XYZ")
            if not np.array_equal(pose, np.eye(4)):
                self._transform(original, moving, pose, pose_buffer)
            else:
                moving[...] = original
            matrix, gradient, fitness, rmse, count = self._iteration_data(
                moving, item, normals, filtered, radius, partials, totals)
            for _ in range(iterations):
                solve_started = time.perf_counter()
                try:
                    update = self.solve(matrix, gradient) if count else np.eye(4)
                finally:
                    self.statistics["solve_s"] += time.perf_counter() - solve_started
                if np.asarray(update).shape != (4, 4) or not np.isfinite(update).all():
                    raise ResidentFallback("Nonfinite/invalid resident solve update")
                pose = update @ pose
                self._transform(moving, moving, update, pose_buffer)
                old_fitness, old_rmse = fitness, rmse
                matrix, gradient, fitness, rmse, count = self._iteration_data(
                    moving, item, normals, filtered, radius, partials, totals)
                self.statistics["pose_iterations"] += 1
                if abs(old_fitness-fitness) < 1e-6 and abs(old_rmse-rmse) < 1e-6:
                    break
        started = time.perf_counter()
        # Reuse the kernel's exact strict-radius mask; avoid a second reduction order.
        selected = cp.flatnonzero(filtered >= 0)
        pairs = cp.column_stack((selected, filtered[selected])).astype(cp.int32)
        host_pairs = cp.asnumpy(pairs)
        self.statistics["device_to_host_bytes"] += host_pairs.nbytes
        self.statistics["final_pair_copy_s"] += time.perf_counter() - started
        return SimpleNamespace(transformation=pose, fitness=fitness, inlier_rmse=rmse,
                               correspondence_set=host_pairs)

    def close(self):
        with self.cp.cuda.Device(self.device_id), self.cp.cuda.Stream.null:
            failures = []
            for action in (self._collect_events, self.cp.cuda.Stream.null.synchronize):
                try:
                    action()
                except Exception as error:
                    failures.append(error)
            if self.owns_retrieval:
                try:
                    self.retrieval.close()
                except Exception as error:
                    failures.append(error)
            self.events.clear()
            if failures:
                for error in failures[1:]:
                    failures[0].add_note(f"Additional resident cleanup failure: {error}")
                raise failures[0]
        # CuPy's global allocator is shared; never purge another experiment's pool.

    def report(self):
        """Include wall counters and provenance without making a speed/quality claim."""
        return {"provenance": {**self.provenance,
                "source_unchanged": runtime_source_hash() == self.provenance["source_sha256"],
                "script_unchanged": file_hash(Path(__file__)) == self.provenance["script_sha256"],
                "retrieval_script_unchanged": file_hash(ROOT / "scripts/cuda_optix_registration.py")
                    == self.provenance["retrieval_script_sha256"]},
            "statistics": dict(self.statistics), "fallback_reasons": dict(self.fallback_reasons),
            "retrieval_statistics": dict(self.retrieval.statistics),
            "scratch_limit_bytes": self.max_scratch_bytes, "max_points": self.max_points,
            "gpu_timing": self.gpu_timing, "device": f"CUDA:{self.device_id}",
            "authority": "Component research only; independent original CPU pose gates are required"}


def verify_eigen_headers(archive_path, eigen_root):
    """Verify the exact pinned archive and every Eigen file used by the bridge."""
    archive_path, eigen_root = Path(archive_path), Path(eigen_root)
    if file_hash(archive_path) != EIGEN_ARCHIVE_SHA256:
        raise ValueError("Require the exact Eigen archive pinned by Open3D v0.20.0")
    digest, checked, license_found = hashlib.sha256(), 0, False
    with tarfile.open(archive_path, "r:gz") as archive:
        for item in sorted(archive.getmembers(), key=lambda value: value.name):
            parts = PurePosixPath(item.name).parts
            if not item.isfile() or len(parts) < 2:
                continue
            relative = PurePosixPath(*parts[1:])
            if relative.parts[0] != "Eigen" and str(relative) != "COPYING.MPL2":
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive header path")
            expected = archive.extractfile(item).read()
            actual = eigen_root.joinpath(*relative.parts).read_bytes()
            if actual != expected:
                raise ValueError(f"Extracted Eigen header/license differs from pinned archive: {relative}")
            digest.update(str(relative).encode())
            digest.update(actual)
            checked += 1
            license_found |= str(relative) == "COPYING.MPL2"
    if checked < 10 or not license_found:
        raise ValueError("Missing pinned Eigen headers or MPL-2.0 license")
    return {"eigen_commit": EIGEN_COMMIT, "eigen_archive_sha256": EIGEN_ARCHIVE_SHA256,
            "eigen_headers_sha256": digest.hexdigest(), "verified_header_files": checked}


def build_solver(output, eigen_root, eigen_archive, compiler):
    """Tiny CPU-only compilation; never initializes or measures CUDA."""
    output = Path(output).resolve()
    headers = verify_eigen_headers(eigen_archive, eigen_root)
    compiler = shutil.which(compiler) or compiler
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.with_suffix(".cpp")
    source.write_bytes(CPU_BRIDGE_SOURCE.encode())
    command = [compiler, "/nologo", "/O2", "/EHsc", "/LD", "/std:c++17",
        "/DEIGEN_DONT_PARALLELIZE", f"/I{Path(eigen_root).resolve()}", str(source),
        f"/Fo:{output.with_suffix('.obj')}", "/link", f"/OUT:{output}"]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=output.parent, capture_output=True, text=True, timeout=60)
    output.with_suffix(".build.log").write_text(result.stdout+result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"CPU bridge compilation failed; see {output.with_suffix('.build.log')}")
    manifest = {**headers, "bridge_source_sha256": hashlib.sha256(CPU_BRIDGE_SOURCE.encode()).hexdigest(),
        "library_sha256": file_hash(output), "compiler": str(compiler), "command": command,
        "compile_s": time.perf_counter()-started,
        "solver": "standalone Eigen LDLT/RzRyRx reimplementation; original Open3D binary is not called"}
    output.with_suffix(".build.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def validate_retrieval_proofs(adapter_path, real_paths, fixture_path, device, radius_bins, miss_policy):
    """Pure stdlib preflight; run before CUDA imports and never trust flags alone."""
    radius_bins = checked_radius_bins(radius_bins)
    if miss_policy not in MISS_POLICIES:
        raise ValueError("Unknown research RTX miss policy")
    direct_misses = miss_policy == "direct-miss-research-v1"
    artifacts = {name: ROOT / "scripts" / name for name in ("benchmark_optix_nn.py",
        "cuda_optix_registration.py", "research_optix_nn.cu", "research_optix_nn_host.cpp",
        "build_optix_nn.ps1", "benchmark_parallel_fragments.py")}
    artifacts["original_cpu_icp_adapter"] = ROOT / "scanner_server/cuda_nn_registration.py"
    nearest_root = ROOT / "benchmark-output/cuda-pipeline/optix-nearest"
    artifacts.update({name: nearest_root / name for name in
                      ("nearest.dll", "nearest.ptx", "vendor/optix-dev-v9.0.0.zip")})
    current = {name: file_hash(path) for name, path in artifacts.items()}
    source_hash, fixture_hash = runtime_source_hash(), file_hash(fixture_path)
    hashes, proof_records = {}, []

    def load(path):
        path = Path(path).resolve()
        payload = path.read_bytes()
        value = json.loads(payload)
        if (value.get("kind") != "standalone-exact-rtx-neighbour-search"
                or any(value.get(key) is not True for key in
                       ("passed", "source_unchanged", "math_unchanged", "artifacts_unchanged"))
                or value.get("source_sha256") != source_hash
                or not value.get("gpu")
                or any(value.get("artifacts_sha256", {}).get(key) != digest
                       for key, digest in current.items())
                or value.get("device") != device
                or checked_radius_bins(value.get("radius_bins")) != radius_bins
                or value.get("miss_policy") != miss_policy
                or not isinstance(value.get("certified_domain"), (str, dict))
                or not value["certified_domain"]):
            raise ValueError(f"RTX proof does not bind unchanged current source/artifacts/device: {path}")
        digest = hashlib.sha256(payload).hexdigest()
        hashes[str(path)] = digest
        proof_records.append(value)
        return value, str(path), digest

    def require_miss_audit(counts):
        declared, audited, false_misses = (counts.get(key) for key in
            ("declared_rt_miss_queries", "audited_rt_miss_queries", "audit_false_misses"))
        if (any(type(value) is not int or value < 0 for value in (declared, audited, false_misses))
                or declared != audited or false_misses != 0):
            raise ValueError("Direct-miss policy requires every declared RTX miss shadowed and zero false misses")
        return declared

    adapter, adapter_file, adapter_hash = load(adapter_path)
    synthetic = adapter.get("synthetic_indices")
    if (adapter.get("synthetic_device_adapter_checked") is not True or not synthetic
            or any(type(row.get("queries")) is not int or row["queries"] < 1
                or row.get("index_mismatches") != 0
                or row.get("device_adapter_index_mismatches") != 0
                or not isinstance(row.get("device_adapter_max_squared_distance_delta"), (int, float))
                or not math.isfinite(row["device_adapter_max_squared_distance_delta"])
                or row["device_adapter_max_squared_distance_delta"] < 0 for row in synthetic)):
        raise ValueError("Require completed synthetic device-adapter comparisons against original CPU indices/distances")
    if direct_misses and (adapter.get("cpu_miss_audit") is not True
            or require_miss_audit(adapter.get("synthetic_statistics", {})) < 1):
        raise ValueError("Require positive synthetic direct-miss coverage with complete original CPU miss audits")
    coverage, audited_pairs, total_hits, total_misses, native_records = set(), set(), 0, 0, []
    seen_native = set()
    raw_hashes = {}
    for path in real_paths:
        proof, proof_file, digest = load(path)
        if digest in seen_native:
            raise ValueError("Duplicate real-audit evidence cannot establish additional coverage")
        seen_native.add(digest)
        native = proof.get("real_pairs")
        if (proof.get("cpu_index_audit") is not True or not native
                or proof.get("fixture_unchanged") is not True
                or proof.get("raw_input_unchanged") is not True
                or proof.get("fixture_sha256") != fixture_hash):
            raise ValueError("Require nonempty unchanged real-fixture every-hit CPU audits, not a synthetic-only proof")
        if direct_misses and proof.get("cpu_miss_audit") is not True:
            raise ValueError("Real direct-miss proof must shadow declared RTX misses with the original CPU tree")
        provenance = proof.get("fixture_provenance", {})
        raw_path = Path(provenance["session"]).resolve()
        if str(raw_path) not in raw_hashes:
            raw_hashes[str(raw_path)] = file_hash(raw_path)
        if (not proof.get("raw_input_sha256") or proof["raw_input_sha256"] != provenance.get("input_sha256")
                or raw_hashes[str(raw_path)] != proof["raw_input_sha256"]
                or provenance.get("local_pose_source") != "measured Finish fragment report; no archived ZIP poses"):
            raise ValueError("Real audit no longer binds its original raw ZIP or excludes archived pose seeds")
        for pair in native:
            proposals = pair.get("proposals")
            accepted = pair.get("original_pair_verdict", {}).get("accepted")
            runs = pair.get("rtx_runs")
            if (type(proposals) is not int or proposals < 1 or type(accepted) is not bool
                    or not runs or len(pair.get("original_proposal_evidence", [])) != proposals):
                raise ValueError("Require original accepted/rejected real pairs with actual proposal comparisons")
            coverage.add(accepted)
            audited_pairs.add(tuple(pair["pair"]))
            for run in runs:
                checks = run.get("agreement", [])
                counts = run.get("statistics_delta", {})
                direct, shadowed, mismatches = (counts.get(key) for key in
                    ("direct_rt_hit_queries", "audited_rt_hit_queries", "audit_index_mismatches"))
                if (run.get("pair_verdict_agrees") is not True or len(checks) != proposals
                        or any(check.get("passed") is not True for check in checks)
                        or any(type(value) is not int or value < 0 for value in (direct, shadowed, mismatches))
                        or direct != shadowed or mismatches != 0):
                    raise ValueError("Real fixture did not preserve every proposal and audit every direct RTX hit")
                total_hits += direct
                if direct_misses:
                    total_misses += require_miss_audit(counts)
        native_records.append({"path": proof_file, "sha256": digest,
            "direct_rt_hit_queries": sum(run["statistics_delta"]["direct_rt_hit_queries"]
                                        for pair in native for run in pair["rtx_runs"]),
            "declared_rt_miss_queries": sum(run["statistics_delta"].get("declared_rt_miss_queries", 0)
                                            for pair in native for run in pair["rtx_runs"]),
            "retrieval_complete_on_tested_queries": proof.get("retrieval_complete_on_tested_queries"),
            "rt_miss_cpu_hits": proof.get("final_statistics", {}).get("rt_miss_cpu_hits")})
    if coverage != {False, True} or total_hits < 1:
        raise ValueError("Require both accepted and rejected real CPU pair audits and positive native direct-hit coverage")
    if direct_misses and total_misses < 1:
        raise ValueError("Require positive native direct-miss coverage, not vacuous zero-miss audits")
    if len({json.dumps(proof.get("gpu"), sort_keys=True) for proof in proof_records}) != 1:
        raise ValueError("Synthetic adapter and real audits used different GPU/driver identities")
    if len({json.dumps(proof["certified_domain"], sort_keys=True) for proof in proof_records}) != 1:
        raise ValueError("Synthetic adapter and real audits used different certified retrieval domains")
    return {"adapter": {"path": adapter_file, "sha256": adapter_hash},
        "native": native_records, "direct_native_hits": total_hits,
        "declared_native_misses": total_misses, "miss_policy": miss_policy,
        "certified_domain": adapter["certified_domain"], "gpu": adapter.get("gpu"),
        "audited_pairs": sorted(audited_pairs), "fixture_sha256": fixture_hash,
        "raw_inputs_sha256": raw_hashes, "proofs_sha256": hashes, "artifacts_sha256": current,
        "radius_bin_policy": {"fractions": list(radius_bins),
            "semantics": "Ascending fractions of each original ICP search radius ending at 1; original CPU hybrid search resolves uncertain rows; full-radius misses follow the separately proven miss_policy",
            "retrieval_script_sha256": current["cuda_optix_registration.py"],
            "retrieval_library_sha256": current["nearest.dll"],
            "retrieval_ptx_sha256": current["nearest.ptx"]},
        "scope": "Direct RTX completeness is separate from final-index parity preserved by original CPU fallback"}


def run_pair_proof(args):
    """Run only after exact retrieval proof and explicit external hardware allocation."""
    prerequisites = validate_retrieval_proofs(args.adapter_proof, args.retrieval_proof,
                                             args.fixture, args.device, args.radius_bins, args.miss_policy)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("KINECT_NATIVE", "on")
    os.environ["KINECT_CUDA_REGISTRATION"] = "cpu"
    import cupy as cp
    import cv2
    import numpy as np
    import open3d as o3d
    import scanner_server.fragments as fragments
    from scripts.benchmark_optix_nn import canonical_evidence, evidence_agreement, pair_verdict
    from scripts.benchmark_parallel_fragments import numerical_source_hash, unpack_fragment
    from scripts.cuda_optix_registration import ARTIFACTS, OptixICP
    from scripts.process_metrics import gpu_info, peak_rss_bytes

    o3d.utility.set_max_threads(20)
    cv2.setNumThreads(20)
    source_before, math_before = runtime_source_hash(), numerical_source_hash()
    paths = {name: ROOT / "scripts" / name for name in ("research_resident_icp.py",
        "cuda_optix_registration.py", "research_optix_nn.cu", "research_optix_nn_host.cpp",
        "benchmark_optix_nn.py", "benchmark_parallel_fragments.py")}
    paths.update({"solve_library": args.solver_dll,
                  "solve_manifest": args.solver_dll.with_suffix(".build.json"),
                  "adapter_proof": args.adapter_proof, "fixture": args.fixture,
                  "retrieval_library": ARTIFACTS / "nearest.dll",
                  "retrieval_ptx": ARTIFACTS / "nearest.ptx"})
    paths.update({f"retrieval_proof_{index}": path for index, path in enumerate(args.retrieval_proof)})
    paths.update({f"raw_input_{index}": Path(path)
                  for index, path in enumerate(prerequisites["raw_inputs_sha256"])})
    fingerprints = {key: file_hash(path) for key, path in paths.items()}
    hardware = gpu_info()
    if hardware != prerequisites["gpu"]:
        raise ValueError("GPU/driver changed after prerequisite audits")
    report = {"kind": "standalone-roundoff-changing-resident-fp64-icp",
        "source_sha256": source_before, "verification_dependencies_sha256": math_before,
        "artifacts_sha256": fingerprints, "gpu": hardware,
        "versions": {"numpy": np.__version__, "open3d": o3d.__version__, "cupy": cp.__version__,
                     "opencv": cv2.__version__},
        "thread_policy": {"omp": os.environ["OMP_NUM_THREADS"], "open3d": 20, "opencv": 20},
        "radius_bin_policy": prerequisites["radius_bin_policy"],
        "miss_policy": prerequisites["miss_policy"],
        "certified_domain": prerequisites["certified_domain"],
        "retrieval_prerequisites": prerequisites,
        "performance_attribution_valid": not (args.audit_nearest or args.audit_misses),
        "cpu_index_audit": args.audit_nearest, "cpu_miss_audit": args.audit_misses,
        "authority": "Fixed bridge component inputs and unchanged original CPU witnesses/gates; no archived pose seeds, live integration, graph frontier or mesh speed/quality claim",
        "roundoff": "FP64 GPU transforms/reductions and separately compiled Eigen bridge require empirical full CPU acceptance/evidence proof; resemblance to source is not equivalence",
        "real_pairs": []}
    original, solver, retrieval, failure = fragments._match, None, None, None
    try:
        with args.fixture.open("rb") as source:
            fixture = pickle.load(source)
        if (fixture["metadata"]["component_source_sha256"] != math_before
                or fixture["metadata"]["local_pose_source"] != "measured Finish fragment report; no archived ZIP poses"):
            raise ValueError("Fixture must preserve current component mathematics and exclude archived ZIP pose seeds")
        report["fixture_provenance"] = fixture["metadata"]
        tasks = [fixture["tasks"][index] for index in args.task_indices]
        if any(tuple(task["pair"]) not in prerequisites["audited_pairs"] for task in tasks):
            raise ValueError("Selected resident tasks have not passed the prerequisite real RTX hit audits")
        packed = {index: unpack_fragment(data) for index, data in fixture["fragments"].items()}
        for index, data in fixture["fragments"].items():
            for original_view, view in zip(data["keys"], packed[index].keys):
                for fragment in packed.values():
                    for other in fragment.keys:
                        matches = original_view["matches"].get(other.index)
                        if matches is not None:
                            fragments._cache_matches(view, other, matches.copy())
        references = []
        for task in tasks:
            a, b = task["pair"]
            started = time.perf_counter()
            reference = [fragments._verify_bridge(packed[a], packed[b], proposal, fixture["camera"])
                         for proposal in task["proposals"]]
            references.append((reference, time.perf_counter()-started, pair_verdict(reference)))
        if {verdict["accepted"] for _, _, verdict in references} != {False, True}:
            raise ValueError("Select task indices covering both an accepted and a rejected original CPU pair")
        setup_started = time.perf_counter()
        solve = EigenSolve(args.solver_dll)
        retrieval = OptixICP(device=args.device, radius_bins=args.radius_bins, max_clouds=args.max_clouds,
            max_cache_bytes=args.cache_mib*1024**2, audit_nearest=args.audit_nearest,
            miss_policy=args.miss_policy, audit_misses=args.audit_misses)
        solver = ResidentICP(retrieval, solve, max_scratch_bytes=args.scratch_mib*1024**2,
                             gpu_timing=args.gpu_timing, owns_retrieval=True)
        report["setup_s"] = time.perf_counter()-setup_started
        for task, (reference, cpu_s, verdict) in zip(tasks, references):
            a, b = task["pair"]
            row = {"pair": task["pair"], "original_cpu_s": cpu_s,
                "original_pair_verdict": verdict,
                "original_proposal_evidence": [canonical_evidence(value) for value in reference],
                "resident_runs": []}
            report["real_pairs"].append(row)
            try:
                fragments._match = solver.match
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    actual = [fragments._verify_bridge(packed[a], packed[b], proposal, fixture["camera"])
                              for proposal in task["proposals"]]
                    elapsed = time.perf_counter()-started
                    comparisons = [evidence_agreement(old, new) for old, new in zip(reference, actual)]
                    actual_verdict = pair_verdict(actual)
                    same = all(actual_verdict[key] == verdict[key]
                               for key in ("accepted", "ambiguous", "verified_proposals"))
                    row["resident_runs"].append({"repeat": repeat, "elapsed_s": elapsed,
                        "pair_verdict": actual_verdict, "same_pair_verdict": same,
                        "proposal_evidence": [canonical_evidence(value) for value in actual],
                        "evidence_agreement": comparisons, "resident": solver.report()})
                    if not same or not all(value["passed"] for value in comparisons):
                        raise RuntimeError("Resident ICP changed original CPU decisions/evidence; retain CPU authority")
            finally:
                fragments._match = original
    except Exception as error:
        failure = error
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    finally:
        fragments._match = original
        if solver:
            report["resident"] = solver.report()
        try:
            if solver:
                solver.close()
            elif retrieval:
                retrieval.close()
            with cp.cuda.Device(int(args.device.split(":")[1])):
                cp.cuda.Stream.null.synchronize()
        except Exception as error:
            report["cleanup_error"] = str(error)
            failure = failure or error
        report["source_unchanged"] = runtime_source_hash() == source_before
        report["math_unchanged"] = numerical_source_hash() == math_before
        report["artifacts_unchanged"] = all(file_hash(path) == fingerprints[key] for key, path in paths.items())
        report["hardware_unchanged"] = gpu_info() == hardware
        report["peak_process_rss_bytes"] = peak_rss_bytes()
        report["passed"] = bool(report["real_pairs"]) and failure is None and all(report[key]
            for key in ("source_unchanged", "math_unchanged", "artifacts_unchanged", "hardware_unchanged"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "passed": report["passed"],
                          "authority": report["authority"]}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-solve-source", type=Path)
    parser.add_argument("--build-solver", type=Path)
    parser.add_argument("--eigen-root", type=Path)
    parser.add_argument("--eigen-archive", type=Path)
    parser.add_argument("--compiler", default="cl.exe")
    parser.add_argument("--run-proof", action="store_true",
                        help="GPU execution: require separately allocated hardware and proven exact RTX retrieval")
    parser.add_argument("--adapter-proof", type=Path)
    parser.add_argument("--retrieval-proof", type=Path, nargs="+")
    parser.add_argument("--solver-dll", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--task-indices", type=int, nargs="+")
    parser.add_argument("--device", default="CUDA:0")
    parser.add_argument("--radius-bins", type=float, nargs="+", default=(1.,),
                        help="Exact proven increasing fractions of each original ICP radius, ending at 1")
    parser.add_argument("--miss-policy", choices=MISS_POLICIES, default="cpu-fallback",
                        help="Exact policy bound by synthetic and accepted/rejected real retrieval proofs")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-clouds", type=int, default=64)
    parser.add_argument("--cache-mib", type=int, default=512)
    parser.add_argument("--scratch-mib", type=int, default=256)
    parser.add_argument("--audit-nearest", action="store_true")
    parser.add_argument("--audit-misses", action="store_true",
                        help="Shadow declared resident-query misses on CPU; correctness only, timing confounded")
    parser.add_argument("--gpu-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-output/cuda-pipeline/resident-icp/proof.json")
    args = parser.parse_args()
    if sum(bool(option) for option in (args.emit_solve_source, args.build_solver, args.run_proof)) != 1:
        parser.error("Choose exactly one source-emission, CPU-build or explicitly allocated GPU-proof action")
    if args.emit_solve_source:
        args.emit_solve_source.parent.mkdir(parents=True, exist_ok=True)
        args.emit_solve_source.write_bytes(CPU_BRIDGE_SOURCE.encode())
        print(json.dumps({"source": str(args.emit_solve_source.resolve()),
            "source_sha256": file_hash(args.emit_solve_source), "eigen_archive": EIGEN_ARCHIVE_URL,
            "eigen_archive_sha256": EIGEN_ARCHIVE_SHA256,
            "runtime_source_sha256": runtime_source_hash(), "script_sha256": file_hash(Path(__file__)),
            "authority": "unmeasured standalone research"}))
    elif args.build_solver:
        if not args.eigen_root or not args.eigen_archive:
            parser.error("--build-solver requires --eigen-root and --eigen-archive")
        print(json.dumps(build_solver(args.build_solver, args.eigen_root, args.eigen_archive, args.compiler)))
    elif args.run_proof:
        if not all((args.adapter_proof, args.retrieval_proof, args.solver_dll, args.fixture, args.task_indices)):
            parser.error("--run-proof requires --adapter-proof, --retrieval-proof, --solver-dll, --fixture and --task-indices")
        if (min(args.task_indices) < 0 or len(set(args.task_indices)) != len(args.task_indices)
                or min(args.repeats, args.max_clouds, args.cache_mib, args.scratch_mib) < 1
                or not args.device.startswith("CUDA:") or not args.device.split(":")[1].isdigit()):
            parser.error("Require distinct nonnegative task indices, positive limits and a named CUDA device")
        try:
            args.radius_bins = checked_radius_bins(args.radius_bins)
        except ValueError as error:
            parser.error(str(error))
        run_pair_proof(args)
    else:
        parser.error("Choose CPU-only --emit-solve-source or --build-solver; import ResidentICP only in an allocated GPU proof")


if __name__ == "__main__":
    main()
