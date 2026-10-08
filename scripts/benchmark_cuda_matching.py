"""Interleave exact CPU/CUDA retrieval of a measured ORB keyframe bank."""

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
import numpy as np
import open3d as o3d
from PIL import Image

from scanner_server.appearance import correspondences, extract_features
from scanner_server.cuda_matching import Matcher
from scripts.process_metrics import finish_cuda_worker
from scripts.profile_session import file_hash, source_hash
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--method", choices=("orb", "sift"), default="orb")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Require positive repeats")
    records = []
    for path in args.sessions:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            settings = ScanSettings.from_dict(manifest["settings"])
            count = len(manifest["frames"])
            indices = sorted(set(np.linspace(0, count-1, 40).astype(int)) | {count//2})
            bank = {}
            for index in indices:
                item = manifest["frames"][index]
                with archive.open(item["rgb"]) as f, Image.open(f) as image:
                    rgb = np.asarray(image.convert("RGB"))
                with archive.open(item["depth"]) as f, Image.open(f) as image:
                    depth = np.asarray(image, dtype=np.uint16)
                rgb, depth = prepare_rgbd(rgb, depth, settings)
                bank[index] = extract_features(rgb, depth, settings.camera, method=args.method)
        targets = [bank[i] for i in indices[:40]]
        matcher = Matcher()
        for index in (indices[0], count//2, indices[-1]):
            source = bank[index]
            actions = {"cpu": lambda: [correspondences(source, target) for target in targets],
                       "cuda": lambda: matcher.match(source, targets)}
            for action in actions.values():
                action()
            samples = {name: [] for name in actions}
            for repeat in range(args.repeats):
                results = {}
                for name in list(actions)[::1 if repeat%2 else -1]:
                    o3d.core.cuda.synchronize()
                    start = time.perf_counter()
                    results[name] = actions[name]()
                    o3d.core.cuda.synchronize()
                    samples[name].append((time.perf_counter()-start)*1000)
                for before, after in zip(results["cpu"], results["cuda"]):
                    np.testing.assert_array_equal(before, after)
            medians = {name: float(np.median(values)) for name,values in samples.items()}
            row = {"session": path.name, "input_sha256": file_hash(path), "method": args.method, "source_index": int(index),
                   "target_indices": [int(i) for i in indices[:40]], "source_descriptors": len(source.points),
                   "target_descriptors": [len(f.points) for f in targets], "samples_ms": samples,
                   "median_ms": medians, "speedup": medians["cpu"]/medians["cuda"],
                   "exact_matches": True}
            records.append(row)
            print(json.dumps({k: row[k] for k in ("session", "source_index", "median_ms", "speedup")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_sha256": source_hash(), "repeats": args.repeats,
                                      "rows": records}, indent=2, allow_nan=False)+"\n")
    o3d.core.cuda.synchronize()
    finish_cuda_worker()


if __name__ == "__main__":
    main()
