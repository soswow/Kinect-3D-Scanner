"""Diagnose native-resolution marker identities supported by calibrated depth.

Maps each native RGB corner to an actually measured, visible depth-grid pixel.
This is an offline proposal experiment, with no fusion or pose authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import socket
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lookup", choices=("cpu", "cuda-audit"), default="cpu")
    parser.add_argument("--maximum-color-distance-px", type=float, default=3.)
    args = parser.parse_args()
    if not args.run_allocated or not 1 <= args.threads <= 20 or not 0 < args.maximum_color_distance_px <= 3:
        parser.error("Require allocated hardware, 1–20 threads and a mapping radius <=3 native RGB pixels")
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT / "benchmark-output")
    except ValueError:
        parser.error("Keep reports in ignored benchmark-output")
    if args.output.exists():
        parser.error("Preserve earlier reports; require a fresh output")
    if args.lookup == "cuda-audit":
        with socket.socket() as probe:
            probe.settimeout(.3)
            if probe.connect_ex(("127.0.0.1", 8000)) == 0:
                parser.error("Stop the idle field server before isolated CUDA research")
    import cv2
    import numpy as np
    from PIL import Image
    from shared.settings import ScanSettings
    from shared.calibration import _rectified_depth, _color_maps_numpy, prepare_metric_depth
    from shared.visual_tracking import sampled_points, feature_agreement
    from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS
    from scanner_server.appearance import Features, propose_transform

    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(0)
    core = source_hash()
    artifact = file_hash(Path(__file__))
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
    report = {"kind": "native-marker-measured-depth-proposal-diagnostic-v2", "status": "running",
        "source_sha256": core, "script_sha256": artifact, "opencv": cv2.__version__, "numpy": np.__version__,
        "lookup_mode": args.lookup,
        "opencv_threads": cv2.getNumThreads(), "maximum_native_rgb_distance_px": args.maximum_color_distance_px,
        "scope": "Native1280x1024RGB decoder, four canonical corners per unique decoded markerID. Original calibrated projection/occlusion and measured3x3depth support. Nearest visible depth-pixel association <=3nativeRGBpixels is a new explicitly bounded proposal policy; reject reused depth pixels, never fill holes or infer board dimensions/layout. Original PnP/feature support thresholds unchanged. No archived poses, graph acceptance, independent geometric verification, fusion, ground truth or camera FPS claim. Component times exclude decode/import/report writes.",
        "sessions": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        lookup = None
        if args.lookup == "cuda-audit":
            from scripts.research.native_corner_cuda import NativeCornerCuda
            helper = ROOT / "scripts/research/native_corner_cuda.py"
            report["cuda_lookup_helper_sha256"] = file_hash(helper)
            lookup = NativeCornerCuda()
            report["cuda_lookup_runtime"] = lookup.provenance
            report["cuda_lookup_self_check"] = lookup.self_check()
            report["cuda_lookup_scope"] = "Every actual query CPU-shadowed. Audit wall includes CPU brute checks; separate GPU component timer includes validation/hash/upload/allocation/kernels/download, excludes original projection/decoder/imports and reports. Compile/setup reported once, never camera FPS or full-Finish acceleration."
            save()
        for session in args.sessions:
            session = session.resolve(strict=True)
            archive_sha = file_hash(session)
            stored, rows = [], []
            with zipfile.ZipFile(session) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                settings = ScanSettings.from_dict(manifest["settings"])
                if settings.sensor_calibration is None:
                    raise ValueError("This diagnostic requires complete native RGB/depth calibration")
                for index, frame in enumerate(manifest["frames"]):
                    with archive.open(frame["rgb"]) as stream, Image.open(stream) as image:
                        rgb = np.array(image.convert("RGB"), np.uint8)
                    with archive.open(frame["depth"]) as stream, Image.open(stream) as image:
                        raw = np.array(image, np.uint16)
                    started = time.perf_counter()
                    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
                    detection_s = time.perf_counter() - started
                    values = [] if ids is None else ids.ravel().tolist()
                    duplicates = {value for value in values if values.count(value) > 1}
                    started = time.perf_counter()
                    metric = _rectified_depth(raw, settings)
                    depth = prepare_metric_depth(raw, settings)
                    map_x, map_y, visible = _color_maps_numpy(metric, depth, settings, rgb.shape)
                    ys, xs = np.nonzero(visible)
                    projections = np.column_stack((map_x[ys, xs], map_y[ys, xs])).astype(np.float64)
                    prepare_and_project_s = time.perf_counter() - started
                    keys, depth_pixels, distances = [], [], []
                    query_keys, queries = [], []
                    for marker, polygon in zip(values, corners):
                        if marker in duplicates:
                            continue
                        for corner, uv in enumerate(np.asarray(polygon).reshape(4, 2)):
                            query_keys.append((marker, corner))
                            queries.append(uv)
                    queries = np.asarray(queries, np.float64).reshape(-1, 2)
                    lookup_record = None
                    if len(queries) and len(projections):
                        if lookup is not None:
                            winners, squared, lookup_record = lookup.query(queries, projections)
                        else:
                            winners, squared = [], []
                            for uv in queries:
                                distance2 = np.sum((projections - uv) ** 2, axis=1)
                                nearest = int(np.argmin(distance2))
                                winners.append(nearest)
                                squared.append(distance2[nearest])
                        for key, nearest, distance in zip(query_keys, winners, squared):
                            if distance <= args.maximum_color_distance_px ** 2:
                                keys.append(key)
                                depth_pixels.append((int(xs[nearest]), int(ys[nearest])))
                                distances.append(float(np.sqrt(distance)))
                    reused = {pixel for pixel in depth_pixels if depth_pixels.count(pixel) > 1}
                    pixels = np.asarray(depth_pixels, np.float64).reshape(-1, 2)
                    points, valid = sampled_points(depth, pixels, settings.camera)
                    for i, pixel in enumerate(depth_pixels):
                        if pixel in reused:
                            valid[i] = False
                    lag = frame.get("metadata", {}).get("rgb_depth_delta_ms")
                    if lag is not None and abs(lag) > RGB_DEPTH_ASSISTANCE_LIMIT_MS:
                        valid[:] = False
                    selected = np.flatnonzero(valid)
                    features = Features(pixels[selected], points[selected], None)
                    stored.append((features, {keys[int(i)]: j for j, i in enumerate(selected)}))
                    rows.append({"index": index, "decoded_markers": len(values), "ids": values,
                        "duplicate_ids_rejected": sorted(duplicates), "mapped_corners": len(keys),
                        "supported_corners": len(selected), "reused_depth_pixels_rejected": len(reused),
                        "detect_s": detection_s, "mapping_and_depth_support_s": time.perf_counter() - started,
                        "prepare_and_project_s": prepare_and_project_s, "cuda_lookup_audit": lookup_record,
                        "supported_rgb_distance_px_p95": float(np.percentile(np.asarray(distances)[selected], 95)) if len(selected) else None})
            pairs = []
            for i in range(len(stored)):
                for j in range(i + 1, len(stored)):
                    a, a_ids = stored[i]
                    b, b_ids = stored[j]
                    common = sorted(a_ids.keys() & b_ids.keys())
                    if len(common) < 40:
                        continue
                    matches = np.asarray([(a_ids[key], b_ids[key]) for key in common], int)
                    started = time.perf_counter()
                    pose = propose_transform(a, b, settings.camera, matches)
                    passed, support = False, None
                    if pose is not None:
                        left, right = matches.T
                        passed, support = feature_agreement(a.points[left], b.points[right], b.pixels[right], pose, settings.camera)
                    pairs.append({"source": i, "target": j, "matches": len(matches), "proposal_passed": pose is not None,
                        "feature_support_passed": bool(passed), "feature_support": support, "wall_s": time.perf_counter() - started})
            if file_hash(session) != archive_sha:
                raise RuntimeError("Archive changed during analysis")
            item = {"archive": str(session), "archive_sha256": archive_sha, "views": rows, "pairs": pairs,
                "summary": {"views": len(rows), "views_with_markers": sum(row["decoded_markers"] > 0 for row in rows),
                    "views_with_40_supported_corners": sum(row["supported_corners"] >= 40 for row in rows),
                    "pairs_with_40_identity_matches": len(pairs), "pairs_passing_original_proposal_and_feature_support": sum(row["feature_support_passed"] for row in pairs),
                    "median_detector_ms": statistics.median(row["detect_s"] for row in rows) * 1000,
                    "median_mapping_ms": statistics.median(row["mapping_and_depth_support_s"] for row in rows) * 1000},
                "settings_sha256": hashlib.sha256(json.dumps(settings.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()}
            report["sessions"].append(item)
            save()
            print(json.dumps({"archive": session.name, **item["summary"]}), flush=True)
        if source_hash() != core or file_hash(Path(__file__)) != artifact:
            raise RuntimeError("Source changed during analysis")
        if lookup is not None and file_hash(helper) != report["cuda_lookup_helper_sha256"]:
            raise RuntimeError("Measured CUDA lookup helper changed during analysis")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
