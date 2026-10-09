"""Read-only recorded loop/BA replay from connected estimates and raw ZIP images.

No fusion or server changes. A proposal only demonstrates measured consistency,
not ground-truth trajectory accuracy. Raw input checksums must remain unchanged.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d

from scripts.benchmark_bundle_adjustment import file_checksum, recording, validate_output
from scanner_server.bundle_adjustment import propose_bundle_poses
from scanner_server.engine import ScanEngine
from scanner_server.refinement import propose_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_output(args.output, args.session, args.poses)
    checksums = {path: file_checksum(path) for path in (args.session, args.poses)}
    cv2.setNumThreads(4)
    cv2.setRNGSeed(0)
    with zipfile.ZipFile(args.session) as archive:
        engine = recording(archive, args.poses)
        geometry = ScanEngine(device="cpu")
        geometry.reset(settings=engine.settings)
        engine.intrinsic = geometry.intrinsic
        engine._make_rgbd = geometry._make_rgbd
        started = time.monotonic()
        refined, refinement = propose_poses(engine)
        refinement["elapsed_ms"] = (time.monotonic() - started) * 1000
        print(json.dumps({"refinement": refinement}, indent=2), flush=True)
        if refined is not None:
            engine.poses = refined

        def progress(current, total, row):
            if current % 8 == 0 or current == total:
                print(f"{current}/{total}: {row['message']}", flush=True)

        bundled, bundle = propose_bundle_poses(engine, progress)
        final = bundled if bundled is not None else engine.poses
        report = {"schema_version": 1, "accepted_views": len(final),
                  "refinement_proposed": refined is not None, "refinement": refinement,
                  "bundle_proposed": bundled is not None, "bundle_adjustment": bundle,
                  "poses": [{"index": i, "camera_to_world": p.tolist()} for i, p in final],
                  "pose_provenance": engine.pose_provenance,
                  "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ("scanner_server/refinement.py", "scanner_server/bundle_adjustment.py",
                                 "scanner_server/pose_candidates.py")},
                  "versions": {"open3d": o3d.__version__, "opencv": cv2.__version__, "numpy": np.__version__},
                  "notes": ["Read-only pose proposals; no TSDF fusion or application",
                            "Archived estimates initialize fresh raw-measurement verification",
                            "No independent ground-truth trajectory; consistency is not absolute accuracy"],
                  "inputs": {str(path): checksum for path, checksum in checksums.items()}}
        report["inputs_unchanged"] = all(file_checksum(path) == checksum for path, checksum in checksums.items())
        if not report["inputs_unchanged"]:
            raise ValueError("Recorded inputs changed during evaluation")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"bundle_adjustment": bundle}, indent=2), flush=True)


if __name__ == "__main__":
    main()
