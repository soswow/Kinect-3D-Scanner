"""Diagnose ID-based measured RGB-D proposals on the field's printed markers.

No board size, layout or archived pose is assumed. This analysis never fuses
geometry or authorizes tracking: marker corner identities only propose motion,
using the original measured-depth support and appearance acceptance functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--dictionary", choices=("DICT_4X4_250", "DICT_4X4_1000"), default="DICT_4X4_250")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_allocated or not 1 <= args.threads <= 20:
        parser.error("Require an allocated hardware slot and 1–20 CPU threads")
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT / "benchmark-output")
    except ValueError:
        parser.error("Keep diagnostic reports in ignored benchmark-output")
    if args.output.exists():
        parser.error("Require a fresh report; preserve earlier evidence")
    import cv2
    import numpy as np
    from PIL import Image
    from shared.calibration import prepare_rgbd
    from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS
    from shared.settings import ScanSettings
    from shared.visual_tracking import sampled_points, feature_agreement
    from scanner_server.appearance import Features, propose_transform

    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(0)
    before = source_hash()
    artifacts = {str(path.relative_to(ROOT)): file_hash(path) for path in (
        Path(__file__), ROOT / "shared/visual_tracking.py", ROOT / "scanner_server/appearance.py")}
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary)))
    report = {"kind": "field-marker-corner-proposal-diagnostic-v1", "status": "running",
        "source_sha256": before, "artifacts_sha256": artifacts,
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
        "dictionary": args.dictionary, "opencv_threads": cv2.getNumThreads(),
        "scope": "Raw RGB-D only. Original calibrated CPU preparation and measured 3x3 depth support; four canonical corners per decoded ID. No physical marker size/layout, archived pose seeds, synthetic samples, threshold changes, geometry integration or accepted scan claim. Proposal support is not independent geometric verification. Component wall excludes ZIP decoding, imports and reports; not camera FPS.",
        "sessions": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        for path in args.sessions:
            path = path.resolve(strict=True)
            archive_sha = file_hash(path)
            stored, rows = [], []
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                settings = ScanSettings.from_dict(manifest["settings"])
                for index, frame in enumerate(manifest["frames"]):
                    with archive.open(frame["rgb"]) as stream, Image.open(stream) as image:
                        rgb = np.array(image.convert("RGB"), np.uint8)
                    with archive.open(frame["depth"]) as stream, Image.open(stream) as image:
                        depth = np.array(image, np.uint16)
                    started = time.perf_counter()
                    color, metric = prepare_rgbd(rgb, depth, settings)
                    prepare_s = time.perf_counter() - started
                    started = time.perf_counter()
                    corners, ids, rejected = detector.detectMarkers(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY))
                    detection_s = time.perf_counter() - started
                    values = [] if ids is None else ids.ravel().tolist()
                    duplicates = {value for value in values if values.count(value) > 1}
                    keys, pixels = [], []
                    for marker, polygon in zip(values, corners):
                        if marker in duplicates:
                            continue
                        for corner, uv in enumerate(np.asarray(polygon).reshape(4, 2)):
                            keys.append((marker, corner))
                            pixels.append(uv)
                    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
                    points, valid = sampled_points(metric, pixels, settings.camera)
                    lag = frame.get("metadata", {}).get("rgb_depth_delta_ms")
                    assistance_allowed = lag is None or abs(lag) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS
                    if not assistance_allowed:
                        valid[:] = False
                    selected = np.flatnonzero(valid)
                    features = Features(pixels[selected], points[selected], None)
                    identities = {keys[int(i)]: j for j, i in enumerate(selected)}
                    stored.append((features, identities))
                    rows.append({"index": index, "ids": values, "duplicate_ids_rejected": sorted(duplicates),
                        "decoded_markers": len(values), "supported_corners": len(selected),
                        "assistance_allowed": assistance_allowed, "rejected_candidates": len(rejected),
                        "prepare_s": prepare_s, "detect_s": detection_s})
            pairs = []
            # All pair identities are independent of archived/fresh world poses.
            for i in range(len(stored)):
                for j in range(i + 1, len(stored)):
                    left, left_ids = stored[i]
                    right, right_ids = stored[j]
                    common = sorted(left_ids.keys() & right_ids.keys())
                    matches = np.asarray([(left_ids[key], right_ids[key]) for key in common], int).reshape(-1, 2)
                    if len(matches) < 40:
                        continue  # original propose_transform minimum, unchanged
                    started = time.perf_counter()
                    pose = propose_transform(left, right, settings.camera, matches)
                    accepted, support = False, None
                    if pose is not None:
                        a, b = matches.T
                        accepted, support = feature_agreement(left.points[a], right.points[b], right.pixels[b], pose, settings.camera)
                    pairs.append({"source": i, "target": j, "matched_corners": len(matches),
                        "shared_marker_ids": len({key[0] for key in common}), "original_proposal_passed": pose is not None,
                        "original_feature_support_passed": bool(accepted), "feature_support": support,
                        "proposal_wall_s": time.perf_counter() - started})
            if file_hash(path) != archive_sha:
                raise RuntimeError("Raw archive changed during analysis")
            item = {"archive": str(path), "archive_sha256": archive_sha,
                "settings_sha256": hashlib.sha256(json.dumps(settings.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest(),
                "frames": rows, "candidate_pairs": pairs,
                "summary": {"views": len(rows), "views_with_markers": sum(bool(row["ids"]) for row in rows),
                    "views_with_40_supported_corners": sum(row["supported_corners"] >= 40 for row in rows),
                    "median_detect_ms": statistics.median(row["detect_s"] for row in rows) * 1000,
                    "pairs_with_40_identity_matches": len(pairs),
                    "pairs_passing_original_proposal_and_feature_support": sum(row["original_feature_support_passed"] for row in pairs)}}
            report["sessions"].append(item)
            save()
            print(json.dumps({"archive": path.name, **item["summary"]}), flush=True)
        if source_hash() != before or any(file_hash(ROOT / path) != digest for path, digest in artifacts.items()):
            raise RuntimeError("Source changed during analysis")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
