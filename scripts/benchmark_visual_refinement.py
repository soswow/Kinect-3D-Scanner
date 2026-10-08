"""Compare measured-identity refinement with ICP on real appearance proposals.

Research only. All methods use the same depth-supported SIFT matches and PnP
seed. Archive transforms never initialize a solve. Synthetic poses score results
only. Shared independent feature and reciprocal raw geometry gates are reported.
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from scanner_server.appearance import correspondences, extract_features, propose_transform
from scanner_server.engine import ScanEngine
from scanner_server.refinement import motion
from scripts.process_metrics import finish_cuda_worker
from scripts.profile_session import file_hash, source_hash
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings
from shared.visual_tracking import feature_agreement, measured_rigid_motion, refine_measured_motion
from tests.test_quality import scene_frames


def refine(source, target, matches, initial, camera, joint):
    a, b = matches.T
    pose = measured_rigid_motion(source.points[a], target.points[b], initial.copy())
    if pose is not None and joint:
        pose = refine_measured_motion(source.points[a], target.points[b], target.pixels[b], pose, camera)
    return pose


def timed(action):
    o3d.core.cuda.synchronize()
    start = time.perf_counter()
    result = action()
    o3d.core.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="*", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-fine", action="store_true",
                        help="Also test the original finest ICP level after measured feature refinement")
    parser.add_argument("--include-gpu-normals", action="store_true",
                        help="Include GPU float64 normal preparation on unchanged CPU voxel positions")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Require positive repeats")
    engine = ScanEngine(device="cuda", tracking="tensor")
    rows = []

    def prepare(rgb, depth):
        rgb, depth = prepare_rgbd(rgb, depth, engine.settings)
        frame = engine._make_rgbd(rgb, depth)
        return engine._make_reg_pcd(frame), extract_features(rgb, depth, engine.settings.camera, method="sift")

    def compare(label, i, j, source, target, truth=None):
        a, sf = source
        b, tf = target
        matches = correspondences(sf, tf)
        cv2.setRNGSeed(0)
        seed = propose_transform(sf, tf, engine.settings.camera, matches)
        if seed is None:
            rows.append({"session": label, "pair": [i, j], "matches": len(matches), "visual_proposal": False})
            return
        actions = {
            "pnp": lambda: seed.copy(),
            "measured_3d": lambda: refine(sf, tf, matches, seed, engine.settings.camera, False),
            "measured_joint": lambda: refine(sf, tf, matches, seed, engine.settings.camera, True),
            "cuda_multiscale_icp": lambda: engine._icp(a, b, seed).transformation,
        }
        if args.include_fine or args.include_gpu_normals:
            def fine(cuda, gpu_normals=False):
                initial = refine(sf, tf, matches, seed, engine.settings.camera, True)
                if initial is None:
                    return None
                voxel = engine.reg_voxel
                source_level = a.voxel_down_sample(voxel)
                target_level = b.voxel_down_sample(voxel)
                if not gpu_normals:
                    target_level.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=30))
                if not cuda:
                    return o3d.pipelines.registration.registration_icp(source_level, target_level, voxel*3,
                        initial, o3d.pipelines.registration.TransformationEstimationPointToPlane(
                            o3d.pipelines.registration.HuberLoss(max(.005, voxel))),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)).transformation
                core, reg = o3d.core, o3d.t.pipelines.registration
                source_tensor = o3d.t.geometry.PointCloud.from_legacy(source_level, core.float32, engine.device)
                if gpu_normals:
                    normal_cloud = o3d.t.geometry.PointCloud.from_legacy(target_level, core.float64, engine.device)
                    normal_cloud.estimate_normals(max_nn=30, radius=voxel*3)
                    target_tensor = o3d.t.geometry.PointCloud(engine.device)
                    target_tensor.point.positions = normal_cloud.point.positions.to(core.float32)
                    target_tensor.point.normals = normal_cloud.point.normals.to(core.float32)
                else:
                    target_tensor = o3d.t.geometry.PointCloud.from_legacy(target_level, core.float32, engine.device)
                result = reg.icp(source_tensor, target_tensor, voxel*3, core.Tensor(initial, core.float64),
                    reg.TransformationEstimationPointToPlane(reg.robust_kernel.RobustKernel(
                        reg.robust_kernel.RobustKernelMethod.HuberLoss, max(.005, voxel))),
                    reg.ICPConvergenceCriteria(max_iteration=20))
                return result.transformation.cpu().numpy()
            actions.update(cpu_measured_fine_icp=lambda: fine(False), cuda_measured_fine_icp=lambda: fine(True))
            if args.include_gpu_normals:
                actions["cuda_measured_fine_gpu64_normals"] = lambda: fine(True, True)
        timings = {name: [] for name in actions}
        results = {}
        for action in actions.values():
            timed(action)
        for repeat in range(args.repeats):
            for name in list(actions)[::1 if repeat % 2 else -1]:
                results[name], elapsed = timed(actions[name])
                timings[name].append(elapsed)
        reference = results["cuda_multiscale_icp"]
        quality = {}
        aa, bb = matches.T
        for name, pose in results.items():
            if pose is None or not np.isfinite(pose).all():
                quality[name] = {"valid": False, "passes_existing_gates": False}
                continue
            forward = o3d.pipelines.registration.evaluate_registration(a, b, .0225, pose)
            reverse = o3d.pipelines.registration.evaluate_registration(b, a, .0225, np.linalg.inv(pose))
            supported, support = feature_agreement(sf.points[aa], tf.points[bb], tf.pixels[bb], pose, engine.settings.camera)
            correction, angle = motion(np.linalg.inv(seed) @ pose)
            delta_m, delta_deg = motion(np.linalg.inv(reference) @ pose)
            passed = (supported and correction <= .03 and angle <= 3 and forward.fitness >= .6
                      and forward.inlier_rmse <= .015 and reverse.fitness >= .45 and reverse.inlier_rmse <= .015)
            quality[name] = {"valid": True, "passes_existing_gates": bool(passed),
                             "feature_support": support, "delta_from_icp_m": float(delta_m),
                             "delta_from_icp_deg": float(delta_deg), "forward_overlap": forward.fitness,
                             "reverse_overlap": reverse.fitness, "forward_rmse_m": forward.inlier_rmse,
                             "reverse_rmse_m": reverse.inlier_rmse, "pose": pose.tolist()}
            if truth is not None:
                distance, angle = motion(np.linalg.inv(truth) @ pose)
                quality[name].update(truth_error_m=float(distance), truth_error_deg=float(angle))
        row = {"session": label, "pair": [i, j], "matches": len(matches), "visual_proposal": True,
               "samples_ms": timings, "median_ms": {name: float(np.median(values)) for name, values in timings.items()},
               "quality": quality}
        rows.append(row)
        print(json.dumps({"session": label, "pair": [i, j], "median_ms": row["median_ms"],
                          "gate_pass": {name: value["passes_existing_gates"] for name, value in quality.items()}}), flush=True)

    # Ground truth is read here only for error scoring, never for initialization.
    engine.reset(settings=ScanSettings())
    synthetic = scene_frames(20, engine.settings.camera)
    prepared = [prepare(rgb, depth) for rgb, depth, _ in synthetic]
    for i, j in ((1, 0), (5, 4), (10, 9), (19, 18), (10, 0), (19, 0)):
        truth = np.linalg.inv(synthetic[j][2]) @ synthetic[i][2]
        compare("synthetic_known_motion", i, j, prepared[i], prepared[j], truth)
    for path in args.sessions:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            engine.reset(settings=ScanSettings.from_dict(manifest["settings"]))
            bank = {}
            def get(index):
                if index not in bank:
                    item = manifest["frames"][index]
                    with archive.open(item["rgb"]) as f, Image.open(f) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    with archive.open(item["depth"]) as f, Image.open(f) as image:
                        depth = np.asarray(image, dtype=np.uint16)
                    bank[index] = prepare(rgb, depth)
                return bank[index]
            count = len(manifest["frames"])
            for i in sorted({1, 11, 12, 13, 14, 15, 16, count//4, count//2, count*3//4, count-1}):
                if i < count:
                    compare(path.name, i, i-1, get(i), get(i-1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_sha256": source_hash(), "repeats": args.repeats,
        "input_sha256": {path.name: file_hash(path) for path in args.sessions}, "rows": rows,
        "note": "Component timings exclude extraction, matching, gate evaluation and fusion. Fine ICP includes measured seed, downsampling, normals and GPU conversion; no special prewarmed cloud cache. Archive poses unused. Synthetic ground truth scores only."},
        indent=2, allow_nan=False)+"\n")
    finish_cuda_worker()


if __name__ == "__main__":
    main()
