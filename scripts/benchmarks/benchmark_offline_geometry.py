"""Depth-only registration audit across every distinct locally supplied session.

Original ZIPs and running servers are never changed. Saved trajectories are
reported as a comparison only; they do not initialize or authorize matching.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from scanner_server.geometry_registration import prepare_view, register_pair
from scanner_server.depth_graph import recover_depth_graph, scene_visibility
from shared.settings import ScanSettings


def inventory(paths, registration_far_m=None):
    sessions, aliases = {}, []
    for path in sorted(set(Path(p).resolve() for p in paths)):
        with zipfile.ZipFile(path) as archive:
            if "manifest.json" not in archive.namelist():
                continue
            manifest = json.loads(archive.read("manifest.json"))
            if "frames" not in manifest or "settings" not in manifest:
                continue
            raw = [{key: (archive.getinfo(f[key]).CRC, archive.getinfo(f[key]).file_size)
                    for key in ("rgb", "depth")} for f in manifest["frames"]]
            settings = manifest["settings"]
            # Calibration and clipping affect measured geometry. Pose/backend
            # options do not distinguish copies of the same recorded session.
            geometry = {k: settings.get(k) for k in ("camera", "sensor_calibration", "near_m", "far_m", "roi", "filter_depth")}
            if registration_far_m is not None:
                geometry["registration_far_m"] = registration_far_m
            digest = hashlib.sha256(json.dumps([raw, geometry], sort_keys=True).encode()).hexdigest()
            if digest in sessions:
                aliases.append({"path": str(path), "same_observations_as": sessions[digest]["path"]})
                continue
            reconstruction = json.loads(archive.read(manifest.get("reconstruction", "reconstruction.json")))
            sessions[digest] = {"path": str(path), "observations_id": digest, "frames": len(raw),
                                "saved_accepted": len(reconstruction.get("poses", [])),
                                "session_id": reconstruction.get("session_id")}
    return list(sessions.values()), aliases


def load_views(path, registration_far_m=None):
    views = []
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        settings = ScanSettings.from_dict(manifest["settings"])
        if registration_far_m is not None:
            settings = replace(settings,far_m=registration_far_m)
        for index, frame in enumerate(manifest["frames"]):
            with archive.open(frame["depth"]) as member, Image.open(member) as image:
                raw = np.asarray(image, dtype=np.uint16)
            views.append(prepare_view(index, raw, settings))
    return views


def components(count, pairs):
    parents = list(range(count))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for row in pairs:
        if row["accepted"]:
            parents[root(row["source"])] = root(row["target"])
    groups = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def write_report(path, report):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["local"], choices=("local", "gicp", "projective", "fpfh", "multiscale", "pca", "ppf"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--failed-only", type=Path, help="Use all failures from an existing local audit, plus regular controls")
    parser.add_argument("--pair", type=int, nargs=2, help="One zero-based source/target pair")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--graph", action="store_true", help="Recover and validate complete depth pose graphs")
    parser.add_argument("--saved-trajectory-audit", action="store_true", help="Depth visibility control for archived poses; never used to initialize registration")
    parser.add_argument("--cache", type=Path, nargs="*", default=[], help="Reuse pair audits of the same observations")
    parser.add_argument("--registration-far", type=float, help="Ablate depth clipping for pose estimation; source/fusion settings are unchanged")
    args = parser.parse_args()
    o3d.utility.set_max_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))
    cv2.setNumThreads(max(1, int(os.environ["OMP_NUM_THREADS"])))
    if args.saved_trajectory_audit:
        sessions, aliases = inventory(args.sessions,args.registration_far)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        report = {"kind":"saved-trajectory-depth-control", "sessions":sessions,"aliases":aliases,
                  "poses_used_as_registration_authority":False,"results":[]}
        for session in sessions:
            views = load_views(session["path"],args.registration_far)
            with zipfile.ZipFile(session["path"]) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                saved = json.loads(archive.read(manifest.get("reconstruction","reconstruction.json")))
            poses = {item["index"]:np.asarray(item["camera_to_world"]) for item in saved["poses"]}
            validation = scene_visibility(views,poses)
            report["results"].append({"path":session["path"],"observations_id":session["observations_id"],
                                     "saved_accepted":len(poses),"validation":validation})
            write_report(args.output,report)
            print(json.dumps({"path":session["path"],"saved_accepted":len(poses),"validation":validation}),flush=True)
        return
    if args.graph:
        sessions, aliases = inventory(args.sessions,args.registration_far)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        source_root = Path(__file__).resolve().parents[2]
        source_hashes = {name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                         for name in ("scanner_server/geometry_registration.py","scanner_server/depth_graph.py")}
        report = {"device": args.device, "source_sha256":source_hashes,
                  "open3d_threads": o3d.utility.get_max_threads(),
                  "versions":{"open3d":o3d.__version__,"numpy":np.__version__},
                  "gpu_stages":["depth projective odometry","optional CuPy descriptor search"] if args.device == "cuda" else [],
                  "registration_far_m": args.registration_far, "sessions": sessions, "aliases": aliases, "results": []}
        cached = [json.loads(path.read_text(encoding="utf-8")) for path in args.cache]
        if args.output.exists():
            cached.append(json.loads(args.output.read_text(encoding="utf-8")))
        for session in sessions:
            rows = [pair for audit in cached for result in audit.get("results", audit.get("runs", []))
                    if result["observations_id"] == session["observations_id"]
                    for pair in result.get("pairs", [])]
            print(f"PREPARE {session['path']} frames={session['frames']} cached={len(rows)}", flush=True)
            views = load_views(session["path"],args.registration_far)
            started = time.monotonic()
            partial = {"observations_id": session["observations_id"], "path": session["path"], "pairs": [], "complete": False}
            report["results"].append(partial)
            def checkpoint(pairs):
                partial["pairs"] = pairs
                write_report(args.output, report)
            solved, result = recover_depth_graph(views, device=args.device,
                notify=lambda message: print(message, flush=True), cached_pairs=rows, checkpoint=checkpoint)
            result.update(observations_id=session["observations_id"], path=session["path"],
                registration_far_m=args.registration_far,
                seconds=time.monotonic()-started,
                components=[{"frame_indices": c["frame_indices"], "validation": c["validation"],
                    "poses": {str(i): p.tolist() for i,p in c["poses"].items()}} for c in solved])
            partial.clear()
            partial.update(result, complete=True)
            write_report(args.output, report)
            print(json.dumps({"session": session["path"], "component_sizes": result["component_sizes"],
                              "rejected_graph_edges": len(result["graph_rejected_edges"]), "seconds": result["seconds"]}), flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sessions, aliases = inventory(args.sessions,args.registration_far)
    report = {"kind": "offline-depth-registration-audit-v1", "device": args.device,
              "versions": {"open3d": o3d.__version__, "numpy": np.__version__},
              "rgb_used": False, "saved_pose_seeds_used": False,
              "session_inventory": sessions, "same_observation_copies": aliases, "runs": [],
              "limits": "Self-consistency and coverage, not absolute pose/surface truth"}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report["session_inventory"] != sessions or report["device"] != args.device:
            raise ValueError("Resume inputs/device differ")
    failed = json.loads(args.failed_only.read_text()) if args.failed_only else None
    for session in sessions:
        count = session["frames"]
        print(f"PREPARE {Path(session['path']).name}: {count} views", flush=True)
        views = load_views(session["path"],args.registration_far)
        for method in args.methods:
            run = next((r for r in report["runs"] if r["observations_id"] == session["observations_id"] and r["method"] == method), None)
            if run is None:
                run = {"observations_id": session["observations_id"], "path": session["path"], "method": method, "pairs": []}
                report["runs"].append(run)
            seen = {(r["source"], r["target"]) for r in run["pairs"]}
            pairs = [tuple(args.pair)] if args.pair else [(i, j) for i in range(1, count) for j in range(max(0, i - args.steps), i)]
            if failed:
                reference = next(r for r in failed["runs"] if r["observations_id"] == session["observations_id"] and r["method"] == "local")
                selected = {(r["source"], r["target"]) for r in reference["pairs"] if not r["accepted"] or r["source"] % 10 == 0}
                pairs = [p for p in pairs if p in selected]
            started = time.monotonic()
            for source, target in pairs:
                if (source, target) in seen:
                    continue
                _, result = register_pair(views[source], views[target], method=method, device=args.device)
                run["pairs"].append(result)
                if len(run["pairs"]) % 20 == 0:
                    write_report(args.output, report)
                    print(f"{Path(session['path']).name} {method} {len(run['pairs'])}/{len(pairs)} accepted={sum(p['accepted'] for p in run['pairs'])}", flush=True)
            run.update(complete=True, elapsed_s=time.monotonic() - started,
                       accepted=sum(p["accepted"] for p in run["pairs"]),
                       rejection_counts=dict(Counter(p["reason"] for p in run["pairs"] if not p["accepted"])),
                       components=components(count, run["pairs"]))
            write_report(args.output, report)
            print(json.dumps({"session": Path(session["path"]).name, "method": method, "accepted_pairs": run["accepted"],
                              "component_sizes": [len(c) for c in run["components"]], "seconds": run["elapsed_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
