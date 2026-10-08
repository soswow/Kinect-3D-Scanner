"""Check marker proposals against original reciprocal ICP and SIFT witnesses.

Offline diagnostics only: this never alters a graph, fuses geometry, or loads
archived poses. A passing camera pair is not a verified multi-view bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace, MethodType
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import file_hash, source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--marker-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-allocated", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--maximum-index-gap", type=int, default=3)
    args = parser.parse_args()
    if not args.run_allocated or not 1 <= args.max_pairs <= 200 or not 1 <= args.maximum_index_gap <= 10:
        parser.error("Require an allocated hardware slot and bounded pair selection")
    args.session = args.session.resolve(strict=True)
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT / "benchmark-output")
    except ValueError:
        parser.error("Keep private diagnostic outputs in ignored benchmark-output")
    if args.output.exists():
        parser.error("Require a fresh report")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            parser.error("Stop the idle field server before isolated measurements")
    os.environ.update(OMP_NUM_THREADS="8", KINECT_CUDA_REGISTRATION="cpu")
    import cv2
    import numpy as np
    import open3d as o3d
    from PIL import Image
    from shared.settings import ScanSettings
    from shared.calibration import prepare_rgbd
    from shared.visual_tracking import sampled_points, feature_agreement
    from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS
    from scanner_server.appearance import Features, propose_transform
    from scanner_server.engine import ScanEngine
    from scanner_server import fragments

    cv2.setNumThreads(20)
    core = source_hash()
    archive_sha = file_hash(args.session)
    previous = json.loads(args.marker_report.read_text(encoding="utf-8"))
    if previous["status"] != "complete" or previous["source_sha256"] != core:
        raise ValueError("Require a closed same-source marker diagnostic")
    for relative, digest in previous["artifacts_sha256"].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError("Marker diagnostic source changed")
    matching = [row for row in previous["sessions"] if row["archive_sha256"] == archive_sha]
    if len(matching) != 1:
        raise ValueError("Marker diagnostic must bind this exact raw archive")
    candidates = sorted((row["source"], row["target"]) for row in matching[0]["candidate_pairs"]
        if row["original_feature_support_passed"] and row["target"] - row["source"] <= args.maximum_index_gap)
    selected = candidates[:args.max_pairs]
    indices = sorted({index for pair in selected for index in pair})
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, previous["dictionary"])))
    report = {"kind": "marker-seed-original-geometric-witness-diagnostic-v1", "status": "running",
        "source_sha256": core, "script_sha256": file_hash(Path(__file__)), "archive_sha256": archive_sha,
        "marker_report_sha256": file_hash(args.marker_report), "selection": {"maximum_index_gap": args.maximum_index_gap,
            "eligible_pairs": len(candidates), "selected_pairs": selected, "order": "ascending raw source,target"},
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "open3d": o3d.__version__},
        "thread_policy": {"opencv": cv2.getNumThreads(), "omp": "8"}, "views": [], "pairs": [],
        "scope": "Raw selected-view pair diagnostics. Marker PnP seed must pass unchanged reciprocal point-to-plane ICP, original alternating held-out geometry, and separate original SIFT feature witness. Original thresholds unchanged. No archived poses, physical board dimensions, graph acceptance, mesh/ground-truth or camera FPS claim. Timers include this diagnostic's work; not full Finish timings."}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        prepared, views, markers = {}, {}, {}
        with zipfile.ZipFile(args.session) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            settings = ScanSettings.from_dict(manifest["settings"])
            raw_frames = [None] * len(manifest["frames"])
            engine = SimpleNamespace(settings=settings, raw_frames=raw_frames,
                frame_metadata=[dict(row.get("metadata", {})) for row in manifest["frames"]], max_depth_m=settings.far_m)
            camera = settings.camera
            engine.intrinsic = o3d.camera.PinholeCameraIntrinsic(camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy)
            engine._make_rgbd = MethodType(ScanEngine._make_rgbd, engine)
            for index in indices:
                frame = manifest["frames"][index]
                with archive.open(frame["rgb"]) as stream, Image.open(stream) as image:
                    rgb = np.array(image.convert("RGB"), np.uint8)
                with archive.open(frame["depth"]) as stream, Image.open(stream) as image:
                    depth = np.array(image, np.uint16)
                raw_frames[index] = (rgb, depth)
                color, metric = prepare_rgbd(rgb, depth, settings)
                prepared[index] = (color, metric)
                started = time.perf_counter()
                view = fragments._view(engine, index)
                view_s = time.perf_counter() - started
                views[index] = view
                started = time.perf_counter()
                corners, ids, _ = detector.detectMarkers(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY))
                values = [] if ids is None else ids.ravel().tolist()
                duplicates = {value for value in values if values.count(value) > 1}
                identities, pixels = [], []
                for marker, polygon in zip(values, corners):
                    if marker not in duplicates:
                        for corner, pixel in enumerate(np.asarray(polygon).reshape(4, 2)):
                            identities.append((marker, corner))
                            pixels.append(pixel)
                pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
                points, valid = sampled_points(metric, pixels, camera)
                lag = frame.get("metadata", {}).get("rgb_depth_delta_ms")
                if lag is not None and abs(lag) > RGB_DEPTH_ASSISTANCE_LIMIT_MS:
                    valid[:] = False
                kept = np.flatnonzero(valid)
                features = Features(pixels[kept], points[kept], None)
                markers[index] = (features, {identities[int(i)]: j for j, i in enumerate(kept)})
                report["views"].append({"index": index, "original_view_prepare_s": view_s,
                    "marker_extract_s": time.perf_counter() - started, "supported_marker_corners": len(kept),
                    "original_sift_features": None if view is None else len(view.features.points)})
        for source_index, target_index in selected:
            cv2.setRNGSeed(0)
            o3d.utility.random.seed(0)
            left, left_ids = markers[source_index]
            right, right_ids = markers[target_index]
            common = sorted(left_ids.keys() & right_ids.keys())
            matches = np.asarray([(left_ids[key], right_ids[key]) for key in common], int).reshape(-1, 2)
            started = time.perf_counter()
            proposal = propose_transform(left, right, camera, matches)
            proposal_s = time.perf_counter() - started
            item = {"source": source_index, "target": target_index, "marker_matches": len(matches),
                "proposal_s": proposal_s, "proposal_passed": proposal is not None, "reciprocal_icp_passed": False,
                "heldout_passed": False, "independent_sift_witness_passed": False}
            a, b = views[source_index], views[target_index]
            if proposal is not None and a is not None and b is not None:
                started = time.perf_counter()
                result = fragments._pair(a.train, b.train, proposal, .45)
                item["reciprocal_icp_s"] = time.perf_counter() - started
                item["reciprocal_icp_passed"] = result is not None
                if result is not None:
                    pose = result.transformation
                    valid, stats = fragments._heldout(a.heldout, b.heldout, pose, .4)
                    witness, witness_stats = fragments._visual_witness(a, b, pose, camera)
                    item.update(heldout_passed=bool(valid), heldout=stats,
                        independent_sift_witness_passed=bool(witness), sift_witness=witness_stats,
                        transformation_sha256=hashlib.sha256(np.asarray(pose, "<f8").tobytes()).hexdigest(),
                        marker_to_refined_translation_m=float(np.linalg.norm((np.linalg.inv(proposal) @ pose)[:3, 3])))
            report["pairs"].append(item)
            save()
            print(json.dumps(item), flush=True)
        if source_hash() != core or file_hash(args.session) != archive_sha or file_hash(Path(__file__)) != report["script_sha256"]:
            raise RuntimeError("Source or raw inputs changed during analysis")
        report["summary"] = {"pairs": len(report["pairs"]), "reciprocal_icp_passed": sum(row["reciprocal_icp_passed"] for row in report["pairs"]),
            "all_three_geometric_and_sift_checks_passed": sum(row["reciprocal_icp_passed"] and row["heldout_passed"] and row["independent_sift_witness_passed"] for row in report["pairs"])}
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
