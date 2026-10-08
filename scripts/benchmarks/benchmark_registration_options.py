"""Compare registration alternatives on identical measured RGB-D proposals.

No archived pose initializes any method. CUDA timings synchronize the device.
This is an isolated component experiment, not an end-to-end speed claim.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
import json
import os
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")

import cv2
import numpy as np
import open3d as o3d
from scanner_server.cuda_nn_registration import NearestICP
from PIL import Image

from scanner_server.appearance import correspondences, extract_features, propose_transform
from scanner_server.engine import ScanEngine
from scanner_server.refinement import _match, motion
from scripts.process_metrics import finish_cuda_worker
from scripts.profile_session import file_hash, source_hash
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings
from shared.visual_tracking import feature_agreement


def tensor_match(source, target, initial, dtype, device):
    a = o3d.t.geometry.PointCloud.from_legacy(source, dtype=dtype, device=device)
    b = o3d.t.geometry.PointCloud.from_legacy(target, dtype=dtype, device=device)
    reg = o3d.t.pipelines.registration
    estimator = reg.TransformationEstimationPointToPlane(
        reg.robust_kernel.RobustKernel(reg.robust_kernel.RobustKernelMethod.HuberLoss, 0.01))
    pose = o3d.core.Tensor(initial, dtype=o3d.core.float64)
    for distance, iterations in ((0.12, 40), (0.06, 30), (0.03, 20)):
        result = reg.icp(a, b, distance, pose, estimator,
                         reg.ICPConvergenceCriteria(max_iteration=iterations))
        pose = result.transformation
    result.transformation.cpu().numpy()
    return result


def multiscale_match(source, target, initial, device=None):
    """Sample only coarse solves; final solve/score uses every original point."""
    if device is not None:
        a = o3d.t.geometry.PointCloud.from_legacy(source, dtype=o3d.core.float32, device=device)
        b = o3d.t.geometry.PointCloud.from_legacy(target, dtype=o3d.core.float32, device=device)
        reg = o3d.t.pipelines.registration
        estimator = reg.TransformationEstimationPointToPlane(
            reg.robust_kernel.RobustKernel(reg.robust_kernel.RobustKernelMethod.HuberLoss, .01))
        pose = o3d.core.Tensor(initial, dtype=o3d.core.float64)
    else:
        a, b = source, target
        reg = o3d.pipelines.registration
        estimator = reg.TransformationEstimationPointToPlane(reg.HuberLoss(.01))
        pose = initial
    for voxel, distance, iterations in ((.04, .12, 40), (.02, .06, 30), (None, .03, 20)):
        aa, bb = (a, b) if voxel is None else (a.voxel_down_sample(voxel), b.voxel_down_sample(voxel))
        if voxel is not None:
            bb.normalize_normals()
        result = reg.icp(aa, bb, distance, pose, estimator, reg.ICPConvergenceCriteria(max_iteration=iterations)) \
            if device is not None else reg.registration_icp(aa, bb, distance, pose, estimator,
                                                           reg.ICPConvergenceCriteria(max_iteration=iterations))
        pose = result.transformation
    return result


def timed(action):
    o3d.core.cuda.synchronize()
    start = time.perf_counter()
    result = action()
    o3d.core.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-multiscale", action="store_true")
    parser.add_argument("--include-nearest", action="store_true")
    parser.add_argument("--include-threads", action="store_true", help="Measure Open3D's actual TBB limits")
    parser.add_argument("--include-kdtree", action="store_true", help="Isolated double-precision CUDA KD-tree")
    parser.add_argument("--feature-method", choices=("orb", "sift"), default="orb")
    parser.add_argument("--methods", nargs="+", choices=("cpu_icp", "cuda_icp_f32", "cuda_icp_f64",
                        "cuda_rgbd_PointToPlane", "cuda_rgbd_Hybrid", "cpu_multiscale", "cuda_multiscale", "cuda_nearest_icp",
                        "cuda_reranked_icp", "cuda_kdtree_icp", "cpu_threads_8", "cpu_threads_4", "cpu_threads_1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Require positive repeats")
    engine = ScanEngine(device="cuda", tracking="tensor")
    nearest_solver = NearestICP(engine.device) if args.include_nearest else None
    reranked_solver = NearestICP(engine.device, reranked=True) if args.include_nearest else None
    if args.include_kdtree:
        from scripts.research.archive.cuda_kdtree_registration import KDTreeICP
        kdtree_solver = KDTreeICP(engine.device)
    original_threads = o3d.utility.get_max_threads()
    def legacy(threads, action):
        o3d.utility.set_max_threads(threads)
        return action()
    reports = []
    for path in args.sessions:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            settings = ScanSettings.from_dict(manifest["settings"])
            engine.reset(settings=settings)
            prepared = {}
            def get(index):
                if index not in prepared:
                    item = manifest["frames"][index]
                    with archive.open(item["rgb"]) as f, Image.open(f) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    with archive.open(item["depth"]) as f, Image.open(f) as image:
                        depth = np.asarray(image, dtype=np.uint16)
                    rgb, depth = prepare_rgbd(rgb, depth, settings)
                    rgbd = engine._make_rgbd(rgb, depth)
                    cloud = engine._make_reg_pcd(rgbd)
                    features = extract_features(rgb, depth, settings.camera, method=args.feature_method)
                    tensor_rgbd = o3d.t.geometry.RGBDImage(
                        o3d.t.geometry.Image(o3d.core.Tensor(rgb).to(engine.device)),
                        o3d.t.geometry.Image(o3d.core.Tensor(depth).to(engine.device)))
                    prepared[index] = cloud, features, tensor_rgbd
                return prepared[index]
            count = len(manifest["frames"])
            for i in (1, count // 4, count // 2, count * 3 // 4, count - 1):
                source, sf, srgbd = get(i)
                target, tf, trgbd = get(i-1)
                matches = correspondences(sf, tf)
                cv2.setRNGSeed(0)
                proposal = propose_transform(sf, tf, settings.camera, matches)
                # Also exercise difficult proposals for which appearance fails.
                initial = np.eye(4) if proposal is None else proposal
                methods = {
                    "cpu_icp": lambda: _match(source, target, initial),
                    "cuda_icp_f32": lambda: tensor_match(source, target, initial, o3d.core.float32, engine.device),
                    "cuda_icp_f64": lambda: tensor_match(source, target, initial, o3d.core.float64, engine.device),
                }
                if nearest_solver is not None:
                    methods["cuda_nearest_icp"] = lambda: nearest_solver.match(source, target, initial)
                    methods["cuda_reranked_icp"] = lambda: reranked_solver.match(source, target, initial)
                if args.include_kdtree:
                    methods["cuda_kdtree_icp"] = lambda: kdtree_solver.match(source, target, initial)
                if args.include_threads:
                    for threads in (8, 4, 1):
                        methods[f"cpu_threads_{threads}"] = lambda threads=threads: legacy(threads, lambda: _match(source, target, initial))
                    methods = {name: (lambda action=action: legacy(original_threads, action))
                               if not name.startswith("cpu_threads_") else action for name, action in methods.items()}
                odo = o3d.t.pipelines.odometry
                for name in ("PointToPlane", "Hybrid"):
                    methods["cuda_rgbd_" + name] = lambda name=name: odo.rgbd_odometry_multi_scale(
                        srgbd, trgbd, engine.intrinsic_tensor,
                        o3d.core.Tensor(initial, dtype=o3d.core.float64), depth_max=engine.max_depth_m,
                        criteria_list=[odo.OdometryConvergenceCriteria(n) for n in (20, 10, 5)],
                        method=getattr(odo.Method, name))
                if args.include_multiscale:
                    methods.update(cpu_multiscale=lambda: multiscale_match(source, target, initial),
                                   cuda_multiscale=lambda: multiscale_match(source, target, initial, engine.device))
                if args.methods:
                    if "cpu_icp" not in args.methods:
                        parser.error("Require cpu_icp as the shared reference")
                    methods = {name: action for name, action in methods.items() if name in args.methods}
                    if set(args.methods) != set(methods):
                        parser.error("Multiscale methods require --include-multiscale")
                samples = {name: [] for name in methods}
                results = {}
                for action in methods.values():
                    timed(action)
                for repeat in range(args.repeats):
                    for name in list(methods)[::1 if repeat % 2 else -1]:
                        results[name], elapsed = timed(methods[name])
                        samples[name].append(elapsed)
                cpu_pose = np.asarray(results["cpu_icp"].transformation)
                quality = {}
                for name, result in results.items():
                    pose = (np.asarray(result.transformation) if isinstance(result.transformation, np.ndarray)
                            else result.transformation.cpu().numpy())
                    forward = o3d.pipelines.registration.evaluate_registration(source, target, 0.0225, pose)
                    reverse = o3d.pipelines.registration.evaluate_registration(target, source, 0.0225, np.linalg.inv(pose))
                    delta_m, delta_deg = motion(np.linalg.inv(cpu_pose) @ pose)
                    good, support = (feature_agreement(sf.points[matches[:, 0]], tf.points[matches[:, 1]],
                                                      tf.pixels[matches[:, 1]], pose, settings.camera)
                                     if len(matches) else (False, {}))
                    quality[name] = {"delta_from_cpu_m": float(delta_m), "delta_from_cpu_deg": float(delta_deg),
                                     "forward_overlap": forward.fitness, "reverse_overlap": reverse.fitness,
                                     "forward_rmse_m": forward.inlier_rmse, "reverse_rmse_m": reverse.inlier_rmse,
                                     "feature_agreement": bool(good), "feature_support": support}
                    if name in ("cuda_nearest_icp", "cuda_reranked_icp", "cuda_kdtree_icp"):
                        quality[name]["same_correspondences"] = bool(np.array_equal(
                            np.asarray(results["cpu_icp"].correspondence_set), result.correspondence_set))
                row = {"session": path.name, "input_sha256": file_hash(path), "pair": [i, i-1],
                       "points": [len(source.points), len(target.points)], "matches": len(matches),
                       "visual_proposal": proposal is not None, "samples_ms": samples,
                       "median_ms": {k: float(np.median(v)) for k,v in samples.items()}, "quality": quality}
                reports.append(row)
                print(json.dumps({k:row[k] for k in ("session", "pair", "median_ms")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_sha256": source_hash(), "repeats": args.repeats,
                                       "default_open3d_threads": original_threads,
                                       "rows": reports}, indent=2, allow_nan=False) + "\n")
    o3d.core.cuda.synchronize()
    finish_cuda_worker()


if __name__ == "__main__":
    main()
