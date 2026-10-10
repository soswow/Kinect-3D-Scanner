"""Measure tag evidence and a compact landmark experiment without changing a scan.

This is research, not a reconstruction replacement. The experiment uses only
server-redetected RGB-D tag corners and retains the existing full depth audit.
Archived camera poses are used for comparison, never to initialize the tag tree.
Run with the server environment; reports and captures belong in ignored export/.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
import numpy as np
from PIL import Image

from scanner_server.appearance import Features, extract_features
from scanner_server.motion_evidence import extract_tracks
from scanner_server.depth_graph import (
    _tree, bridge_visibility, components, edge_from_report, scene_visibility,
)
from scanner_server.geometry_registration import evaluate, pose_distance, prepare_view
from scanner_server.joint_depth_bundle import refine_motion_graph
from scanner_server.motion_evidence import MotionEvidence
from shared.apriltag import AprilTagDetector, tag_agreement, tag_motion
from shared.calibration import prepare_rgbd
from shared.settings import ScanSettings


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def quantiles(values):
    return dict(zip(("min", "median", "p95", "max"),
                    np.percentile(values, (0, 50, 95, 100)).tolist())) if values else {}


def tag_pairs(frames):
    bank, pairs = {}, set()
    for a, frame in enumerate(frames):
        for identity in frame.tags:
            previous = bank.setdefault(identity, [])
            pairs.update((a, b) for b in set(previous[:5] + previous[::4][-27:] + previous[-8:]))
            previous.append(a)
    return sorted(pairs)


def landmark_motion(frames, metadata, camera):
    tracked, groups = [], defaultdict(list)
    for view, frame in enumerate(frames):
        keys = sorted(frame.tags)
        points = np.concatenate([frame.tags[k].points for k in keys]) if keys else np.empty((0, 3))
        pixels = np.concatenate([frame.tags[k].pixels for k in keys]) if keys else np.empty((0, 2))
        tracked.append((np.arange(len(points)), Features(pixels, points, None)))
        for index, key in enumerate(keys):
            for corner in range(4):
                groups[key, corner].append((view, index * 4 + corner))
    motion = MotionEvidence(metadata, camera, [None] * len(frames),
                            gravity=False, visual=False, tracked=tracked)
    motion.direct_groups = [group for group in groups.values() if len(group) >= 2]
    return motion


def pose_quality(poses, frames, pairs, views, camera, reference):
    accepted, errors = 0, []
    for a, b in pairs:
        if a not in poses or b not in poses:
            continue
        relative = np.linalg.inv(poses[b]) @ poses[a]
        good, report = tag_agreement(frames[a], frames[b], relative, camera)
        accepted += good
        errors.append(report["median_pixel_error"])
    gauge = reference[min(poses)] @ np.linalg.inv(poses[min(poses)])
    differences = [pose_distance(reference[i], gauge @ p) for i, p in poses.items()]
    return {"tag_pairs_supported": int(accepted), "tag_pixel_error": quantiles(errors),
            "difference_from_saved_poses_m": quantiles([d[0] for d in differences]),
            "difference_from_saved_poses_deg": quantiles([d[1] for d in differences]),
            "depth_audit": scene_visibility(views, poses)}


def analyze(session, output, *, exclude_ids=(), compact_only=False, depth_refinement=False):
    cv2.setRNGSeed(0)
    np.random.seed(0)
    timing = Counter()
    views, frames, appearance, tracked, metadata = [], [], [], [], []
    with zipfile.ZipFile(session) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        saved = json.loads(archive.read(manifest.get("reconstruction", "reconstruction.json")))
        settings = ScanSettings.from_dict(manifest["settings"])
        if not settings.apriltag_tracking:
            raise ValueError("Require a session with AprilTag tracking enabled")
        detector = AprilTagDetector(settings.apriltag_dictionaries)
        journal = json.loads(archive.read(manifest["motion_journal"])) if "motion_journal" in manifest else None
        for index, item in enumerate(manifest["frames"]):
            start = time.perf_counter()
            rgb = np.array(Image.open(io.BytesIO(archive.read(item["rgb"]))).convert("RGB"))
            raw = np.array(Image.open(io.BytesIO(archive.read(item["depth"]))))
            timing["decode_s"] += time.perf_counter() - start
            start = time.perf_counter()
            views.append(prepare_view(index, raw, settings))
            rgb, depth = prepare_rgbd(rgb, raw, settings)
            timing["prepare_s"] += time.perf_counter() - start
            meta = item.get("metadata", {})
            metadata.append(meta)
            synchronized = abs(meta.get("rgb_depth_delta_ms", 0)) <= 20
            start = time.perf_counter()
            frame = detector.detect(rgb, depth, settings.camera, synchronized=synchronized)
            frame.tags = {key: tag for key, tag in frame.tags.items() if key[1] not in exclude_ids}
            frames.append(frame)
            timing["detection_s"] += time.perf_counter() - start
            start = time.perf_counter()
            appearance.append(extract_features(rgb, depth, settings.camera, method="sift")
                              if synchronized and not compact_only else None)
            tracked.append(extract_tracks(meta, depth, settings.camera)
                           if synchronized and not compact_only else None)
            timing["sift_and_tracks_s"] += time.perf_counter() - start
            if index % 20 == 0:
                print(f"Prepared {index+1}/{len(manifest['frames'])}", flush=True)
    motion = MotionEvidence(metadata, settings.camera, appearance, tracked=tracked,
                            gravity=settings.gravity_assistance, visual=True, journal=journal)
    pairs = tag_pairs(frames)
    measured, accepted, reasons, accepted_reports = [], {}, Counter(), []
    tag_durations, seed_durations = [], []
    for position, (a, b) in enumerate(pairs):
        if position % 400 == 0:
            print(f"Checking tags {position+1}/{len(pairs)}", flush=True)
        # The production tag path currently requests these general seeds and
        # then discards them. Isolate their cost from fitting known tag labels.
        if not compact_only:
            start = time.perf_counter()
            motion.seeds(a, b)
            duration = time.perf_counter() - start
            timing["unused_general_seeds_s"] += duration
            seed_durations.append(duration)
        start = time.perf_counter()
        pose, support = tag_motion(frames[a], frames[b], settings.camera)
        duration = time.perf_counter() - start
        timing["tag_fit_s"] += duration
        tag_durations.append(duration)
        if pose is None:
            reasons["no_consistent_tag_pose"] += 1
            continue
        measured.append({"source": a, "target": b, "transform": pose,
                         "score": support["inlier_tags"] / (1 + support["median_pixel_error"])})
        if compact_only:
            continue
        start = time.perf_counter()
        evidence = evaluate(views[a], views[b], pose, minimum_condition=0)
        timing["pair_depth_validation_s"] += time.perf_counter() - start
        reasons[evidence["reason"]] += 1
        report = {"source": a, "target": b, "method": "apriltag", **evidence,
                  "apriltag_support": support, "transform": pose.tolist()}
        edge = edge_from_report(report)
        if edge is None:
            continue
        accepted_reports.append(report)
        start = time.perf_counter()
        audit = bridge_visibility(views, list(accepted.values()), edge)
        timing["incremental_bridge_audit_s"] += time.perf_counter() - start
        if audit["accepted"]:
            accepted[a, b] = edge
        else:
            reasons["component_bridge_rejected"] += 1
    reference = {p["index"]: np.asarray(p["camera_to_world"]) for p in saved["poses"]}
    groups = components(len(views), measured)
    print(f"Tag-only pose components: {[len(g) for g in groups]}", flush=True)
    experiments = []
    for group in groups:
        if len(group) < 3:
            continue
        edges = [e for e in measured if e["source"] in group and e["target"] in group]
        poses, _ = _tree(group, edges)
        compact = landmark_motion(frames, metadata, settings.camera)
        compact.direct_groups = [[row for row in rows if row[0] in poses] for rows in compact.direct_groups]
        compact.direct_groups = [rows for rows in compact.direct_groups if len(rows) >= 2]
        initial_quality = pose_quality(poses, frames, pairs, views, settings.camera, reference)
        start = time.perf_counter()
        refined, report = refine_motion_graph(poses, [], compact, max_seconds=60, max_evaluations=80)
        elapsed = time.perf_counter() - start
        print(f"Compact tag solve: {elapsed:.3f}s", flush=True)
        experiment = {"captures": group, "initializer": "maximum evidence tree of redetected tag poses",
                            "archived_poses_used_for_initialization": False,
                            "complete_solve_wall_s": elapsed, "refinement": report,
                            "initial_quality": initial_quality,
                            "refined_quality": pose_quality(refined, frames, pairs, views, settings.camera, reference)}
        if depth_refinement:
            start = time.perf_counter()
            result, depth_report = refine_motion_graph(refined, [], compact, views=views,
                                                      max_seconds=60, max_evaluations=80)
            experiment["tag_and_depth_solve_wall_s"] = time.perf_counter() - start
            experiment["tag_and_depth_refinement"] = depth_report
            experiment["tag_and_depth_quality"] = pose_quality(result, frames, pairs, views, settings.camera, reference)
            print(f"Tag and depth solve: {experiment['tag_and_depth_solve_wall_s']:.3f}s", flush=True)
        experiments.append(experiment)
    reconnect = saved.get("fragment_reconnection", {})
    summary = {
        "kind": "apriltag-session-analysis-v1", "session_sha256": digest(session),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT / "shared/apriltag.py", ROOT / "scanner_server/depth_graph.py",
            ROOT / "scanner_server/joint_depth_bundle.py")},
        "session_id": saved.get("session_id"),
        "excluded_ids": sorted(exclude_ids), "pair_depth_benchmark_performed": not compact_only,
        "limitations": "One CPU research run; server idle but still resident. No full mesh rerun or calibrated pose ground truth. Compact tag fit is a proposal, not production authority.",
        "frames": len(frames), "distinct_ids": len({k for frame in frames for k in frame.tags}),
        "detections": sum(f.detected for f in frames), "usable_tags": sum(len(f.tags) for f in frames),
        "usable_per_capture": quantiles([len(f.tags) for f in frames]),
        "duplicate_ids_in_same_image": [{"capture": index + 1, "ids": duplicates}
            for index, frame in enumerate(frames)
            if (duplicates := sorted(key[1] for key, count in Counter(key for key, _ in frame.detections).items()
                                     if count > 1))],
        "shared_label_pairs": len(pairs), "tag_pose_pairs": len(measured),
        "tag_pose_component_sizes": [len(g) for g in groups],
        "pair_outcomes": dict(reasons), "timings_s": dict(timing),
        "tag_fit_per_pair_s": quantiles(tag_durations),
        "general_seed_per_pair_s": quantiles(seed_durations),
        "depth_validated_tag_components": ([len(g) for g in components(len(views), list(accepted.values()))]
                                           if not compact_only else None),
        "saved_stage_totals_ms": saved.get("stage_totals_ms"),
        "saved_pair_outcomes": dict(Counter(f"{p['method']}:{p.get('reason')}" for p in reconnect.get("pairs", ()))),
        "saved_final_edge_methods": dict(Counter(p["method"] for p in reconnect.get("verified_bridges", ()))),
        "saved_rejected_edges": len(reconnect.get("graph_rejected_edges", ())),
        "saved_final_joint_refinements": reconnect.get("joint_refinements", []),
        "experiments": experiments,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("experiments", "saved_final_joint_refinements")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-id", type=int, nargs="*", default=[],
                        help="Research exclusion of known repeated physical IDs; never silently infer from saved poses")
    parser.add_argument("--compact-only", action="store_true",
                        help="Skip general-seed and pair-depth timing; retain the final complete depth audit")
    parser.add_argument("--depth-refinement", action="store_true",
                        help="Also refine the compact tag proposal with selected-view depth surfaces and planes")
    args = parser.parse_args()
    analyze(args.session, args.output, exclude_ids=args.exclude_id,
            compact_only=args.compact_only, depth_refinement=args.depth_refinement)
