"""Rebuild an exported session ZIP with verified fragment reconnection.

Uses retained observations and calibration. Optional exported poses are seeds
that must be revalidated against raw geometry, never accepted graph authority.
The input ZIP and a running scanner server are left unchanged.
"""

import argparse
import json
import os
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from scanner_server.engine import ScanEngine
from scanner_server.session import export_session
from shared.settings import ScanSettings


def load_session(engine, path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        if not manifest["frames"] or len(manifest["frames"]) > engine.MAX_FRAMES:
            raise ValueError(f"Session must contain 1–{engine.MAX_FRAMES} frames")
        engine.reset(settings=replace(settings, reconnect_fragments=True))
        for index, frame in enumerate(manifest["frames"]):
            with archive.open(frame["rgb"]) as source, Image.open(source) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            with archive.open(frame["depth"]) as source, Image.open(source) as image:
                depth = np.array(image, dtype=np.uint16)
            metadata = dict(frame.get("metadata", {}))
            metadata.update(frame_id=index, timestamp_s=frame["timestamp_s"])
            result = engine.store_frame(rgb, depth, metadata)
            if not result["success"]:
                raise ValueError(f"Cannot load frame {index + 1}: {result['message']}")


def load_pose_seeds(engine, path):
    """Skip redundant live replay; every seeded relationship is revalidated."""
    from scanner_server.fragments import _rigid

    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        report = json.loads(archive.read(manifest.get("reconstruction", "reconstruction.json")))
    seeds = []
    seen = set()
    for item in report.get("poses", []):
        index, pose = item["index"], np.asarray(item["camera_to_world"], float)
        if type(index) is not int or not 0 <= index < engine.stored_count or index in seen or not _rigid(pose):
            raise ValueError("Invalid archived pose seed")
        seeds.append((index, pose))
        seen.add(index)
    if not seeds:
        raise ValueError("No archived poses available as seeds; omit --use-pose-seeds")
    engine.poses = sorted(seeds)
    source_results = {f["index"]: f for f in report.get("frames", [])}
    engine.diagnostics = [{**source_results.get(i, {}), "index": i, "success": i in seen}
                          for i in range(engine.stored_count)]
    engine.frame_count = len(seeds)
    engine._processed_count = engine.stored_count
    engine.cumulative_T = engine.poses[-1][1].copy()
    engine._pose_seeds_only = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-session", action="store_true", help="Also export a new ZIP with recovery diagnostics")
    parser.add_argument("--final-weight", type=float, help="Override final surface confidence")
    parser.add_argument("--block-budget", type=int, help="Maximum blocks for verified fresh fusion (1–50000)")
    parser.add_argument("--use-pose-seeds", action="store_true",
                        help="Revalidate archived pose guesses instead of repeating live tracking")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    destination = args.output_dir / "reconnected-session.zip"
    if args.save_session and destination.resolve() == args.session.resolve():
        parser.error("Output session must differ from the source ZIP")
    engine = ScanEngine(device=args.device)
    load_session(engine, args.session)
    if args.use_pose_seeds:
        load_pose_seeds(engine, args.session)
    if args.final_weight is not None:
        engine.settings = replace(engine.settings, final_weight=args.final_weight)
    if args.block_budget is not None:
        engine.settings = replace(engine.settings, final_block_count=args.block_budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def progress(current, total, result):
        print(f"{current}/{total}: {result.get('message', '')}", flush=True)

    success, result = engine.build_mesh(progress_cb=progress)
    report = engine.reconstruction_report()
    (args.output_dir / "reconstruction.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output_dir / "result.json").write_text(json.dumps({"success": success, **result}, indent=2, allow_nan=False))
    if success and not engine.export_ply(str(args.output_dir / "reconnected.ply")):
        raise OSError("Could not export the rebuilt mesh")
    if args.save_session:
        export_session(engine, destination)
    recovery = report["fragment_reconnection"]
    print(json.dumps({"mesh_built": success, "accepted": engine.frame_count,
                      "recovered_views": recovery.get("recovered_frames", 0),
                      "unconnected_fragments": recovery.get("unconnected_fragments", []),
                      "reason": recovery["reason"]}, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
