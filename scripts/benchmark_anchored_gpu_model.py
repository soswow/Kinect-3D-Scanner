"""Research GPU preview with independent raw-keyframe CPU pose corrections.

Known poses only score the moving synthetic input; they never seed tracking.
The GPU map uses uniform fusion and is preview-only. This does not authorize
final geometry, implement global loop repair or establish real-camera FPS.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d

from scanner_server.engine import ScanEngine
from scanner_server.fragments import View, _local_match, _matches, _view, _visual_witness
from scanner_server.appearance import extract_features, propose_transform
from scanner_server.visual_refinement import measured_pose
from scanner_server.refinement import motion
from scripts.process_metrics import finish_cuda_worker, gpu_info
from scripts.profile_session import file_hash, source_hash
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings
from tests.test_quality import scene_frames


def anchor_candidate(view, history, predicted, camera, settings, solver):
    # Search all recent measured-feature witnesses before falling back to ICP.
    # A close view may have lost the distinctive surface that an earlier one saw.
    methods = ("measured", "icp") if solver == "measured" else ("icp",)
    for method in methods:
        for reference, reference_pose in reversed(history):
            relative = None
            if method == "measured":
                matches = _matches(view, reference)
                proposal = propose_transform(view.features, reference.features, camera, matches)
                if proposal is not None:
                    candidate = measured_pose(view.features, reference.features, matches, proposal, camera)
                    if candidate is not None and _visual_witness(view, reference, candidate, camera, matches)[0]:
                        relative = candidate
            else:
                relative = _local_match(view, reference, camera, settings,
                    initial=np.linalg.inv(reference_pose) @ predicted)
            if relative is not None:
                translation, angle = motion(relative)
                if translation <= settings.max_translation_m and angle <= settings.max_rotation_deg:
                    return reference_pose @ relative, method, reference.index
    return None, None, None


def run(raw, prepared, settings, period, anchor_voxel=.015, anchor_solver="icp",
        reset_submaps=False, references=1):
    cv2.setRNGSeed(0)
    o3d.utility.random.seed(0)
    device = o3d.core.Device("CUDA:0")
    c = settings.camera
    intrinsic = o3d.core.Tensor([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], o3d.core.float64)
    pose = o3d.core.Tensor(np.eye(4), o3d.core.float64)
    model = o3d.t.pipelines.slam.Model(settings.voxel_m, 16, 5000, pose, device)
    current = o3d.t.pipelines.slam.Frame(c.height, c.width, intrinsic, device)
    rendered = o3d.t.pipelines.slam.Frame(c.height, c.width, intrinsic, device)
    context = SimpleNamespace(settings=settings, max_depth_m=settings.far_m,
        intrinsic=o3d.camera.PinholeCameraIntrinsic(c.width, c.height, c.fx, c.fy, c.cx, c.cy),
        raw_frames=[item[:2] for item in raw],
        frame_metadata=[{"rgb_depth_delta_ms": 0} for _ in raw])
    context._make_rgbd = lambda rgb, depth: ScanEngine._make_rgbd(context, rgb, depth)
    history = []
    local_frame = -1
    rows, anchor_attempts, anchor_accepted = [], 0, 0
    for index, (rgb, depth) in enumerate(prepared):
        o3d.core.cuda.synchronize()
        started = time.perf_counter()
        current.set_data_from_image("color", o3d.t.geometry.Image(o3d.core.Tensor(rgb).to(device)))
        current.set_data_from_image("depth", o3d.t.geometry.Image(o3d.core.Tensor(depth).to(device)))
        valid, anchor_ms, correction, solver, reference_id, checked = True, 0.0, None, None, None, False
        if index:
            result = model.track_frame_to_model(current, rendered, 1000.0, settings.far_m, .05,
                o3d.t.pipelines.odometry.Method.Hybrid,
                [o3d.t.pipelines.odometry.OdometryConvergenceCriteria(n) for n in (20, 10, 5)])
            candidate = (pose @ result.transformation).cpu().numpy()
            valid = np.isfinite(candidate).all() and np.isfinite(result.fitness) and result.fitness > 0
            if valid:
                pose = o3d.core.Tensor(candidate, o3d.core.float64)
        if valid and period and (index == 0 or index % period == 0 or index == len(raw)-1):
            anchor_started = time.perf_counter()
            if anchor_voxel == .015:
                view = _view(context, index)
            else:
                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                    context._make_rgbd(rgb, depth), context.intrinsic).voxel_down_sample(anchor_voxel)
                cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=anchor_voxel*4, max_nn=30))
                view = View(index, cloud.select_by_index(list(range(0, len(cloud.points), 2))),
                    cloud.select_by_index(list(range(1, len(cloud.points), 2))),
                    extract_features(rgb, depth, c, method="sift"))
            if view is None:
                valid = False
            elif not history:
                history.append((view, np.eye(4)))
                checked = True
            else:
                anchor_attempts += 1
                measured, solver, reference_id = anchor_candidate(view, history, pose.cpu().numpy(), c, settings, anchor_solver)
                valid = measured is not None
                if valid:
                    correction = motion(np.linalg.inv(pose.cpu().numpy()) @ measured)
                    pose = o3d.core.Tensor(measured, o3d.core.float64)
                    history.append((view, measured))
                    history = history[-references:]
                    checked = True
                    anchor_accepted += 1
            anchor_ms = (time.perf_counter()-anchor_started)*1000
        if valid:
            if reset_submaps and checked and index:
                model = o3d.t.pipelines.slam.Model(settings.voxel_m, 16, 5000, pose, device)
                local_frame = -1
                for reference, reference_pose in history:
                    a, b = prepared[reference.index]
                    current.set_data_from_image("color", o3d.t.geometry.Image(o3d.core.Tensor(a).to(device)))
                    current.set_data_from_image("depth", o3d.t.geometry.Image(o3d.core.Tensor(b).to(device)))
                    local_frame += 1
                    model.update_frame_pose(local_frame, o3d.core.Tensor(reference_pose, o3d.core.float64))
                    model.integrate(current, 1000.0, settings.far_m, settings.truncation_m/settings.voxel_m)
            else:
                local_frame += 1
                model.update_frame_pose(local_frame, pose)
                model.integrate(current, 1000.0, settings.far_m, settings.truncation_m/settings.voxel_m)
            model.synthesize_model_frame(rendered, 1000.0, settings.near_m, settings.far_m,
                settings.truncation_m/settings.voxel_m, True, .5)
        o3d.core.cuda.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        estimated = pose.cpu().numpy()
        # Ground truth enters only this scoring operation, after all decisions.
        error = np.linalg.inv(raw[index][2]) @ estimated
        rows.append({"index": index, "elapsed_ms": elapsed, "valid": bool(valid),
            "anchor_ms": anchor_ms, "anchor_correction": correction,
            "anchor_solver": solver,
            "anchor_reference": reference_id,
            "pose": estimated.tolist(), "translation_error_m": float(np.linalg.norm(error[:3,3])),
            "rotation_error_deg": motion(error)[1]})
        print(f"period={period} view={index+1} {elapsed:.1f}ms valid={valid} error={rows[-1]['translation_error_m']*1000:.2f}mm", flush=True)
        if not valid:
            break  # Reject partial, faster runs; never continue a failed preview chain.
    elapsed = [r["elapsed_ms"] for r in rows[1:]]
    errors = [r["translation_error_m"] for r in rows]
    anchor_errors = [r["translation_error_m"] for r in rows if r["anchor_ms"] and r["valid"]]
    return {"period": period, "anchor_voxel_m": anchor_voxel, "anchor_solver": anchor_solver,
        "reset_submaps": reset_submaps, "reference_bank": references,
        "frames": len(raw), "completed": len(rows),
        "all_frames_valid": len(rows) == len(raw) and all(r["valid"] for r in rows),
        "steady_mean_ms": float(np.mean(elapsed)), "steady_p95_ms": float(np.percentile(elapsed,95)),
        "steady_processing_fps": 1000/float(np.mean(elapsed)),
        "preview_translation_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "max_preview_translation_error_m": max(errors),
        "anchor_translation_rmse_m": float(np.sqrt(np.mean(np.square(anchor_errors)))) if anchor_errors else None,
        "anchor_attempts": anchor_attempts, "anchors_accepted": anchor_accepted,
        "anchor_total_ms": sum(r["anchor_ms"] for r in rows), "records": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--periods", nargs="+", type=int, default=(0, 6, 12))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-voxel", type=float, default=.015,
                        help="Independent raw-cloud keyframe resolution; .0075 tests the denser tracking input")
    parser.add_argument("--anchor-solver", choices=("icp", "measured"), default="icp")
    parser.add_argument("--reset-submaps", action="store_true",
                        help="Rebuild the preview map from recent checked keyframes at each correction")
    parser.add_argument("--references", type=int, default=1)
    args = parser.parse_args()
    if (args.frames < 2 or any(period < 0 for period in args.periods)
            or not .005 <= args.anchor_voxel <= .015 or not 1 <= args.references <= 3):
        parser.error("Require >=2 frames and nonnegative periods (0 means uncorrected)")
    fingerprint, script_fingerprint = source_hash(), file_hash(Path(__file__))
    hardware = gpu_info()
    settings = ScanSettings()
    raw = scene_frames(args.frames, settings.camera)
    prepared, preparation_ms = [], []
    for rgb, depth, _ in raw:
        started = time.perf_counter()
        prepared.append(prepare_rgbd(rgb, depth, settings))
        preparation_ms.append((time.perf_counter() - started) * 1000)
    # First CUDA RGB-D/raycast calls can JIT for tens of seconds after a driver
    # update. Keep that initialization record separate from warmed comparisons.
    warmup = run(raw[:2], prepared[:2], settings, 0)
    reports = [run(raw, prepared, settings, period, args.anchor_voxel, args.anchor_solver,
                   args.reset_submaps, args.references) for period in args.periods]
    if source_hash() != fingerprint or file_hash(Path(__file__)) != script_fingerprint or gpu_info() != hardware:
        raise RuntimeError("Prototype implementation or GPU driver changed during measurement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_sha256": fingerprint, "script_sha256": script_fingerprint,
        "input_generator_sha256": file_hash(ROOT / "tests/test_quality.py"),
        "synthetic_input_preparation": {"mean_ms": float(np.mean(preparation_ms)),
            "p95_ms": float(np.percentile(preparation_ms, 95)),
            "note": "Timed separately; synthetic metric VGA images do not require native Kinect RGB/depth projection."},
        "gpu_hardware": hardware, "warmup": {k:v for k,v in warmup.items() if k != "records"},
        "prototype": "GPU hybrid preview + existing CPU fragment-local raw-camera verification",
        "input": "noisy moving synthetic RGB-D, known poses used only for scoring",
        "settings": settings.to_dict(), "runs": reports,
        "limitations": "Preview-only uniform GPU map; keyframe checks are local, no global closure or final reconstruction. Map corrections are bounded optional submap rebuilds. Timings exclude synthetic generation, input depth preparation, USB/network and final fusion. No measured real-camera FPS."},
        indent=2, allow_nan=False)+"\n")
    o3d.core.cuda.synchronize()
    finish_cuda_worker()


if __name__ == "__main__":
    main()
