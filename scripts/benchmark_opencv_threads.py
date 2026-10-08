"""Measure actual OpenCV thread counts on unchanged real RGB-D features.

An isolated component experiment; no scanner defaults or archive poses change.
Every timed feature result must equal the original thread policy exactly.
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
from scanner_server.appearance import extract_features
from scripts.profile_session import file_hash, source_hash
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings


def same(a, b):
    np.testing.assert_array_equal(a.pixels, b.pixels)
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.descriptors, b.descriptors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Require positive repeats")
    original = cv2.getNumThreads()
    policies = list(dict.fromkeys((original, 1, 4, 8)))
    rows = []
    try:
        for path in args.sessions:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                settings = ScanSettings.from_dict(manifest["settings"])
                indices = np.unique(np.linspace(0, len(manifest["frames"])-1, 5).astype(int))
                for index in indices:
                    frame = manifest["frames"][index]
                    with archive.open(frame["rgb"]) as f, Image.open(f) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    with archive.open(frame["depth"]) as f, Image.open(f) as image:
                        depth = np.asarray(image, dtype=np.uint16)
                    rgb, depth = prepare_rgbd(rgb, depth, settings)
                    for method in ("orb", "sift"):
                        cv2.setNumThreads(original)
                        reference = extract_features(rgb, depth, settings.camera, method=method)
                        samples = {str(policy): [] for policy in policies}
                        for repeat in range(args.repeats):
                            for policy in policies[::1 if repeat % 2 == 0 else -1]:
                                cv2.setNumThreads(policy)
                                started = time.perf_counter()
                                actual = extract_features(rgb, depth, settings.camera, method=method)
                                elapsed = (time.perf_counter()-started)*1000
                                same(reference, actual)
                                samples[str(cv2.getNumThreads())].append(elapsed)
                        row = {"session": path.name, "index": int(index), "method": method,
                               "features": len(reference.points), "samples_ms": samples,
                               "median_ms": {key: float(np.median(value)) for key, value in samples.items()},
                               "exact_features": True}
                        rows.append(row)
                        print(json.dumps({key: row[key] for key in ("session", "index", "method", "median_ms")}), flush=True)
    finally:
        cv2.setNumThreads(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_sha256": source_hash(), "original_threads": original,
        "input_sha256": {path.name: file_hash(path) for path in args.sessions}, "repeats": args.repeats,
        "rows": rows}, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
