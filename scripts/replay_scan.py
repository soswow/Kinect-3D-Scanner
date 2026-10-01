"""Replay public RGB-D data; ground truth is used only for scoring, never tracking.

Run with OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset redwood.
TUM: --dataset tum --path datasets/rgbd_dataset_freiburg1_xyz --stride 10.
Recordings: --dataset recording --path recordings/my-scan (manifest.json).
"""

import argparse
import importlib.util
import json
import os
import resource
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

from scanner_server.engine import ScanEngine
from shared.settings import CameraCalibration, ScanSettings


class ReplayFrame(tuple):
    """Keep the historical four-field replay interface and capture metadata."""

    def __new__(cls, rgb, depth, stamp, reference, metadata=None):
        value = super().__new__(cls, (rgb, depth, stamp, reference))
        value.metadata = dict(metadata or {})
        return value


def capture_metadata(frame, index):
    # Evaluation reference poses are deliberately not tracking inputs.
    metadata = dict(getattr(frame, "metadata", {}))
    metadata.pop("reference_pose", None)
    metadata.update(frame_id=index, timestamp_s=frame[2])
    return metadata


def nearest_pairs(rgb, depth, max_delta=0.02):
    """Greedy timestamp association, with each depth image used at most once."""
    candidates = []
    times = np.array([row[0] for row in depth])
    for i, (stamp, _) in enumerate(rgb):
        center = np.searchsorted(times, stamp)
        for j in (center - 1, center):
            if 0 <= j < len(depth) and abs(stamp - depth[j][0]) <= max_delta:
                candidates.append((abs(stamp - depth[j][0]), i, j))
    used_rgb, used_depth, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            used_rgb.add(i)
            used_depth.add(j)
            pairs.append((rgb[i], depth[j]))
    return sorted(pairs)


def timestamp_rows(path):
    return [
        (float(parts[0]), parts[1:])
        for line in path.read_text().splitlines()
        if (parts := line.split()) and not line.startswith("#")
    ]


def tum_pose(values):
    tx, ty, tz, x, y, z, w = map(float, values)
    # Open3D quaternion order is w,x,y,z; source is x,y,z,w.
    pose = np.eye(4)
    pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([w, x, y, z])
    pose[:3, 3] = [tx, ty, tz]
    return pose


def load_dataset(kind, path=None, stride=1, limit=None):
    metadata_by_path = {}
    if kind == "redwood":
        sample = o3d.data.SampleRedwoodRGBDImages(
            data_root=str(ROOT / "datasets/redwood")
        )
        intrinsic = json.loads(Path(sample.camera_intrinsic_path).read_text())
        k = np.array(intrinsic["intrinsic_matrix"]).reshape(3, 3).T
        camera = CameraCalibration(
            name="Redwood PrimeSense", fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2]
        )
        lines = Path(sample.trajectory_log_path).read_text().splitlines()
        poses = [
            np.array(
                [[float(x) for x in line.split()] for line in lines[i + 1 : i + 5]]
            )
            for i in range(0, len(lines), 5)
        ]
        entries = [
            (i / 30, Path(c), Path(d), poses[i], 1000.0)
            for i, (c, d) in enumerate(zip(sample.color_paths, sample.depth_paths))
        ]
    elif kind == "tum":
        path = Path(path)
        pairs = nearest_pairs(
            timestamp_rows(path / "rgb.txt"), timestamp_rows(path / "depth.txt")
        )
        gt = timestamp_rows(path / "groundtruth.txt")
        gt_times = np.array([t for t, _ in gt])
        camera = CameraCalibration(
            name="TUM registered RGB / recommended ROS approximation"
        )
        entries = []
        for (stamp, c), (_, d) in pairs:
            j = int(np.argmin(np.abs(gt_times - stamp)))
            pose = tum_pose(gt[j][1]) if abs(gt_times[j] - stamp) <= 0.02 else None
            entries.append((stamp, path / c[0], path / d[0], pose, 5000.0))
    else:
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        metadata_by_path = {
            str(path / f["rgb"]): f.get("metadata", {}) for f in manifest["frames"]
        }
        settings = ScanSettings.from_dict(manifest["settings"])
        entries = [
            (
                f["timestamp_s"],
                path / f["rgb"],
                path / f["depth"],
                np.array(f["reference_pose"])
                if f.get("reference_pose") is not None
                else None,
                1000.0,
            )
            for f in manifest["frames"]
        ]
        camera = settings.camera
    entries = entries[::stride][:limit]
    frames = []
    for stamp, c, d, pose, scale in entries:
        rgb = cv2.cvtColor(cv2.imread(str(c)), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(d), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Cannot read depth image: {d}")
        if raw.dtype != np.uint16 or raw.shape != (480, 640):
            raise ValueError("Dataset depth must be uint16 640x480")
        depth = np.rint(raw.astype(np.float64) * 1000 / scale).astype(np.uint16)
        frames.append(
            ReplayFrame(rgb, depth, stamp, pose, metadata_by_path.get(str(c)))
        )
    return (settings if kind == "recording" else ScanSettings(camera=camera)), frames


def score(estimated, frames):
    pairs = [
        (pose, frames[index][3])
        for index, pose in estimated
        if frames[index][3] is not None
    ]
    if len(pairs) < 2:
        return {"scored_poses": len(pairs)}
    # Anchor both trajectories to their first camera. No scale adjustment, no GT seeding.
    e0, g0 = pairs[0]
    errors, rotations = [], []
    for e, g in pairs:
        delta = np.linalg.inv(np.linalg.inv(g0) @ g) @ (np.linalg.inv(e0) @ e)
        errors.append(np.linalg.norm(delta[:3, 3]))
        rotations.append(
            np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
        )
    return {
        "scored_poses": len(pairs),
        "anchored_translation_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotations)))),
    }


def baseline_engine(path, camera, legacy_intrinsics=False):
    spec = importlib.util.spec_from_file_location("scanner_server._baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not legacy_intrinsics:
        module.O3D_INTRINSIC = o3d.camera.PinholeCameraIntrinsic(
            640, 480, camera.fx, camera.fy, camera.cx, camera.cy
        )
    else:
        module.O3D_INTRINSIC = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 594.21, 591.04, 339.31, 242.74
        )
    module.O3D_INTRINSIC_TENSOR = o3d.core.Tensor(
        module.O3D_INTRINSIC.intrinsic_matrix, dtype=o3d.core.float64
    )
    return module.ScanEngine()


def replay_server(url, settings, frames, output):
    """Exercise the same binary HTTP pipeline as the GUI (resets server session)."""
    import httpx

    from shared.protocol import pack_frames

    start = time.monotonic()
    with httpx.Client(base_url=url, timeout=600, trust_env=False) as client:
        response = client.post("/api/scan/reset", json=settings.to_dict())
        response.raise_for_status()
        if not response.json().get("success"):
            raise RuntimeError(response.text)
        for first in range(0, len(frames), 25):
            payload = [
                (rgb, depth, capture_metadata(frames[i], i))
                for i, (rgb, depth, stamp, _) in enumerate(
                    frames[first : first + 25], first
                )
            ]
            response = client.post("/api/scan/frames", content=pack_frames(payload))
            response.raise_for_status()
            if response.json().get("rejected_count", 0) or not response.json().get(
                "success"
            ):
                raise RuntimeError(response.text)
        response = client.post("/api/scan/build")
        response.raise_for_status()
        result = response.json()
        response = client.get("/api/scan/diagnostics")
        response.raise_for_status()
        diagnostics = response.json()["frames"]
        estimates = [
            (d["index"], np.array(d["pose"]))
            for d in diagnostics
            if d.get("success") and "pose" in d
        ]
        report = {
            "dataset": "HTTP replay",
            "frames": len(frames),
            "accepted": result["frame_count"],
            "elapsed_s": time.monotonic() - start,
            "mesh_built": result["success"],
            **score(estimates, frames),
            "diagnostics": diagnostics,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        if result["success"]:
            response = client.get("/api/scan/export/ply")
            response.raise_for_status()
            output.with_suffix(".ply").write_bytes(response.content)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "diagnostics"}, indent=2
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["redwood", "tum", "recording"], default="redwood"
    )
    parser.add_argument("--path", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--server",
        help="Replay through HTTP instead of directly; resets the server session",
    )
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="Optional historical engine.py for before/after comparison",
    )
    parser.add_argument("--legacy-intrinsics", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-output/replay.json"
    )
    parser.add_argument("--near", type=float)
    parser.add_argument("--far", type=float)
    parser.add_argument(
        "--color-recovery",
        action="store_true",
        help="Enable experimental RGB-D recovery",
    )
    parser.add_argument(
        "--refine-poses",
        action="store_true",
        help="Enable experimental final pose graph and reintegration",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--tracking", choices=["auto", "legacy", "tensor"], default="auto"
    )
    args = parser.parse_args()
    if args.stride < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Stride and limit must be positive")
    settings, frames = load_dataset(args.dataset, args.path, args.stride, args.limit)
    settings = replace(
        settings,
        near_m=args.near if args.near is not None else settings.near_m,
        far_m=args.far if args.far is not None else settings.far_m,
    )
    if args.color_recovery:
        settings = replace(settings, color_recovery=True)
    if args.refine_poses:
        settings = replace(settings, refine_poses=True)
    if args.server:
        if args.baseline_source:
            parser.error("Baseline comparison uses the direct engine")
        replay_server(args.server, settings, frames, args.output)
        return
    camera = settings.camera
    engine = (
        baseline_engine(args.baseline_source, camera, args.legacy_intrinsics)
        if args.baseline_source
        else ScanEngine(device=args.device, tracking=args.tracking)
    )
    if not args.baseline_source:
        engine.reset(
            settings=replace(
                settings,
                near_m=args.near if args.near is not None else settings.near_m,
                far_m=args.far if args.far is not None else settings.far_m,
            )
        )
    start = time.monotonic()
    estimated, diagnostics = [], []
    for i, (rgb, depth, stamp, _) in enumerate(frames):
        stored = (
            engine.store_frame(rgb, depth)
            if args.baseline_source
            else engine.store_frame(rgb, depth, capture_metadata(frames[i], i))
        )
        if not stored["success"]:
            diagnostics.append(
                {"index": i, "accepted": False, "message": stored["message"]}
            )
            continue
        result = engine.process_frames()
        if result["errors"] == 0:
            estimated.append((i, engine.cumulative_T.copy()))
        diagnostics.append(
            {
                "index": i,
                "accepted": result["errors"] == 0,
                **(engine.diagnostics[-1] if hasattr(engine, "diagnostics") else {}),
            }
        )
        print(f"{i + 1}/{len(frames)} accepted={engine.frame_count}", flush=True)
    success, _ = engine.build_mesh()
    report = {
        "dataset": args.dataset,
        "frames": len(frames),
        "accepted": engine.frame_count,
        "elapsed_s": time.monotonic() - start,
        "mesh_built": success,
        "vertices": len(engine.mesh.vertices) if engine.mesh else 0,
        "triangles": len(engine.mesh.triangles) if engine.mesh else 0,
        "camera": camera.__dict__,
        **score(engine.poses if hasattr(engine, "poses") else estimated, frames),
        "tracking_score": score(estimated, frames),
        "backend": getattr(engine, "backend", None),
        "refinement": getattr(engine, "refinement", None),
        "stage_totals_ms": getattr(engine, "stage_totals_ms", None),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if success:
        engine.export_ply(str(args.output.with_suffix(".ply")))
    print(json.dumps({k: v for k, v in report.items() if k != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
