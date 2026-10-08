"""Research prototype: GPU raycast frame-to-model tracking on controlled input.

This deliberately compares a different architecture. The stock SLAM model uses
uniform TSDF fusion and lacks the server's recovery/independent acceptance gates.
It must never be presented as a validated scanner replacement or archive FPS.
Reference poses score synthetic results only; they never initialize tracking.
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
import numpy as np
import open3d as o3d
from PIL import Image

from scanner_server.engine import ScanEngine
from scripts.process_metrics import finish_cuda_worker
from scripts.profile_session import file_hash, source_hash
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings
from tests.test_quality import scene_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stationary", action="store_true",
                        help="Repeat one measured archive view; this is a controlled zero-motion test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("hybrid", "point-to-plane"), default="point-to-plane")
    args = parser.parse_args()
    if args.frames < 2 or (args.stationary and not args.session):
        parser.error("Require frames >= 2 and a session for stationary replay")
    if args.session:
        frames = []
        with zipfile.ZipFile(args.session) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            settings = ScanSettings.from_dict(manifest["settings"])
            for item in manifest["frames"][:1 if args.stationary else args.frames]:
                with archive.open(item["rgb"]) as f, Image.open(f) as image:
                    rgb = np.asarray(image.convert("RGB"))
                with archive.open(item["depth"]) as f, Image.open(f) as image:
                    depth = np.asarray(image, dtype=np.uint16)
                frames.append((*prepare_rgbd(rgb, depth, settings), None))
        if args.stationary:
            frames = [(*frames[0][:2], np.eye(4)) for _ in range(args.frames)]
    else:
        settings = ScanSettings()
        frames = [(prepare_rgbd(rgb, depth, settings)[0], prepare_rgbd(rgb, depth, settings)[1], truth)
                  for rgb, depth, truth in scene_frames(args.frames)]
    device = o3d.core.Device("CUDA:0")
    c = settings.camera
    intrinsic = o3d.core.Tensor([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], o3d.core.float64)
    pose = o3d.core.Tensor(np.eye(4), o3d.core.float64)
    model = o3d.t.pipelines.slam.Model(settings.voxel_m, 16, 5000, pose, device)
    current = o3d.t.pipelines.slam.Frame(c.height, c.width, intrinsic, device)
    rendered = o3d.t.pipelines.slam.Frame(c.height, c.width, intrinsic, device)
    odo = o3d.t.pipelines.odometry
    records = []
    for index, (rgb, depth, truth) in enumerate(frames):
        o3d.core.cuda.synchronize()
        started = time.perf_counter()
        current.set_data_from_image("color", o3d.t.geometry.Image(o3d.core.Tensor(rgb).to(device)))
        current.set_data_from_image("depth", o3d.t.geometry.Image(o3d.core.Tensor(depth).to(device)))
        result = None
        if index:
            result = model.track_frame_to_model(
                current, rendered, 1000.0, float(settings.far_m), 0.05,
                odo.Method.Hybrid if args.method == "hybrid" else odo.Method.PointToPlane,
                [odo.OdometryConvergenceCriteria(n) for n in (20, 10, 5)])
            pose = pose @ result.transformation
        model.update_frame_pose(index, pose)
        model.integrate(current, 1000.0, float(settings.far_m), settings.truncation_m/settings.voxel_m)
        model.synthesize_model_frame(rendered, 1000.0, float(settings.near_m), float(settings.far_m),
                                     settings.truncation_m/settings.voxel_m, args.method == "hybrid", 0.5)
        o3d.core.cuda.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        estimated = pose.cpu().numpy()
        relative = None if truth is None else np.linalg.inv(truth) @ estimated
        records.append({"index": index, "elapsed_ms": elapsed, "pose": estimated.tolist(),
                        "fitness": None if result is None else result.fitness,
                        "rmse_m": None if result is None else result.inlier_rmse,
                        "translation_error_m": None if relative is None else float(np.linalg.norm(relative[:3, 3])),
                        "rotation_error_deg": None if relative is None else float(np.degrees(np.arccos(np.clip((np.trace(relative[:3,:3])-1)/2, -1, 1))))})
        print(f"{index+1}/{len(frames)} {elapsed:.1f}ms error={records[-1]['translation_error_m']}", flush=True)
    times = [row["elapsed_ms"] for row in records[1:]]
    errors = [row["translation_error_m"] for row in records if row["translation_error_m"] is not None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_sha256": source_hash(), "prototype": "stock GPU raycast SLAM",
        "session": None if args.session is None else args.session.name,
        "input_sha256": None if args.session is None else file_hash(args.session),
        "input_type": "stationary repeated measured view" if args.stationary else "sparse archive" if args.session else "synthetic known-pose moving scene",
        "method": args.method, "settings": settings.to_dict(), "frames": len(records),
        "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times,95)),
        "processing_fps": 1000/float(np.mean(times)),
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "max_translation_error_m": max(errors) if errors else None,
        "records": records,
        "limitations": "Uniform fusion; no independent pose gates, recovery, global closure, final mesh, USB or network. Controlled FPS is not measured live scanning FPS."},
        indent=2, allow_nan=False)+"\n")
    o3d.core.cuda.synchronize()
    finish_cuda_worker()


if __name__ == "__main__":
    main()
