"""Compare every real view pair, retaining any CUDA/CPU identity mismatch."""

import argparse
import json
import os
import sys
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
from scripts.process_metrics import finish_cuda_worker, gpu_info
from scripts.profile_session import file_hash, source_hash
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--methods", nargs="+", choices=("orb", "sift"), default=("orb", "sift"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only-previous-fallback", action="store_true",
                        help="Check banks excluded by the earlier feature-count bound")
    parser.add_argument("--previous-feature-bound", type=int, default=1200)
    args = parser.parse_args()
    reports = []
    for path in args.sessions:
        for method in args.methods:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                settings = ScanSettings.from_dict(manifest["settings"])
                bank = []
                for item in manifest["frames"]:
                    with archive.open(item["rgb"]) as stream, Image.open(stream) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    with archive.open(item["depth"]) as stream, Image.open(stream) as image:
                        depth = np.asarray(image, dtype=np.uint16)
                    rgb, depth = prepare_rgbd(rgb, depth, settings)
                    bank.append(extract_features(rgb, depth, settings.camera, method=method))
            matcher = Matcher()
            total = fallbacks = mismatched = skipped = 0
            examples = []
            for i, source in enumerate(bank):
                for start in range(i+1, len(bank), 40):
                    targets = bank[start:start+40]
                    if args.only_previous_fallback:
                        previously_rejected = (source.descriptors is None or len(source.descriptors) < 40
                            or len(source.descriptors) > args.previous_feature_bound
                            or any(target.descriptors is not None and
                                len(target.descriptors) > args.previous_feature_bound for target in targets))
                        if not previously_rejected:
                            skipped += len(targets)
                            continue
                    actual = matcher.match(source, targets)
                    if actual is None:
                        fallbacks += len(targets)
                        continue
                    for offset, (target, found) in enumerate(zip(targets, actual)):
                        expected = correspondences(source, target)
                        total += 1
                        if not np.array_equal(expected, found):
                            mismatched += 1
                            if len(examples) < 12:
                                examples.append({"source": i, "target": start+offset,
                                    "cpu": expected.tolist(), "cuda": found.tolist()})
                if i%20 == 0:
                    print(f"{path.name} {method}: {i}/{len(bank)} pairs={total} mismatches={mismatched}", flush=True)
            reports.append({"session": path.name, "input_sha256": file_hash(path), "method": method,
                            "pairs_compared": total, "cpu_fallback_pairs": fallbacks,
                            "skipped_previously_supported_pairs": skipped,
                            "descriptor_count_range": [min(len(v.points) for v in bank), max(len(v.points) for v in bank)],
                            "mismatched_pairs": mismatched, "exact_matches": mismatched == 0,
                            "mismatch_examples": examples})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"source_sha256": source_hash(),
                "gpu_hardware": gpu_info(),
                "previous_feature_bound": args.previous_feature_bound if args.only_previous_fallback else None,
                "rows": reports}, indent=2)+"\n")
    o3d.core.cuda.synchronize()
    finish_cuda_worker()


if __name__ == "__main__":
    main()
