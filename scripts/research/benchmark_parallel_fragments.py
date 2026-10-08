"""Measure unchanged Finish bridge verification in bounded CPU processes.

This is a component experiment, not a full scanner replay. Fragment-local poses
come from an already measured Finish report and only reproduce its inputs; each
bridge still runs all original reciprocal, normal-diversity, independent-camera
and held-out checks. Original raw images and calibration rebuild the point data.
RANSAC proposals are prepared serially to isolate Open3D's process-global RNG.
Worker decisions are consumed in the fixed fixture order. No server is changed.

Reproduce whole-pair load balancing with --scheduling dynamic --configurations
1x20 2x10 2x20 --thread-budget 40. Then reuse the exact --fixture with
--scheduling proposals --configurations 2x10 2x20 --thread-budget 40 --reference
the dynamic JSON. Proposal mode finishes every competing proposal for a fixture
pair before advancing to the next fixture pair, preserving proposal authority.
A production implementation must submit only the actual frontier-selected pair.
Forty TBB threads explicitly oversubscribe the
measured twenty-thread CPU. These timings exclude candidate generation, local
fragment estimation, optimized-graph revalidation and fresh final fusion.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sqlite3
import subprocess
import time
import zipfile
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KINECT_NATIVE", "on")
os.environ["KINECT_CUDA_REGISTRATION"] = "cpu"


def json_value(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


VERIFICATION_SYMBOLS = {
    "scanner_server/fragments.py": ("REG", "MAX_MATCH_CACHE", "_rigid", "_disagrees", "_strong", "_heldout",
        "_pair", "_matches", "_cache_matches", "_visual_witness", "_verify_bridge", "_independent_pairs",
        "_verify_partial_bridge", "_verify_visual_bridge"),
    "scanner_server/refinement.py": ("REG", "motion", "_match"),
    "scanner_server/cuda_registration.py": ("_device", "match", "as_legacy_result"),
    "scanner_server/appearance.py": ("correspondences",),
    "shared/visual_tracking.py": ("_feature_support", "feature_agreement"),
    "shared/calibration.py": ("camera_matrix",),
}


def verification_dependencies():
    """Only the actual helper dependency graph for prepared bridge validation."""
    result = {}
    for filename, names in VERIFICATION_SYMBOLS.items():
        source = (ROOT / filename).read_text()
        parsed = ast.parse(source)
        found = {}
        for node in parsed.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                found[node.name] = ast.get_source_segment(source, node)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        found[target.id] = ast.get_source_segment(source, node)
        if set(found) != set(names):
            raise RuntimeError(f"Verification dependency changed structure: {filename}")
        for name in names:
            result[f"{filename}:{name}"] = hashlib.sha256(found[name].encode()).hexdigest()
    return result


def numerical_source_hash():
    return hashlib.sha256(json.dumps(verification_dependencies(), sort_keys=True).encode()).hexdigest()


def preparation_dependencies():
    """Immutable image/geometry preparation is protected until fixture capture."""
    paths = [ROOT / filename for filename in ("scanner_server/fragments.py", "scanner_server/refinement.py",
        "scanner_server/appearance.py", "shared/calibration.py", "shared/depth.py", "shared/native.py",
        "shared/settings.py", "shared/sensor_calibration.py", "shared/capture.py")]
    paths.extend(path for path in (ROOT / "native").rglob("*")
                 if path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".toml") and "build" not in path.parts)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def pack_cloud(cloud):
    import numpy as np

    return {key: np.asarray(getattr(cloud, key)).copy() for key in ("points", "normals", "colors")}


def unpack_cloud(data):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    for key, array in data.items():
        setattr(cloud, key, o3d.utility.Vector3dVector(array))
    return cloud


def pack_view(view):
    return {"index": view.index, "train": pack_cloud(view.train), "heldout": pack_cloud(view.heldout),
            "features": view.features, "pose": view.pose.copy(),
            "matches": {index: item[2].copy() for index, item in view.match_cache.items()}}


def pack_fragment(fragment):
    return {"index": fragment.index, "train": pack_cloud(fragment.train),
            "heldout": pack_cloud(fragment.heldout), "keys": [pack_view(view) for view in fragment.keys]}


def unpack_fragment(data):
    from scanner_server.fragments import Fragment, View

    views = [View(item["index"], unpack_cloud(item["train"]), unpack_cloud(item["heldout"]),
                  item["features"], item["pose"]) for item in data["keys"]]
    by_index = {view.index: view for view in views}
    # Cache immutable descriptor evidence only; no pose-dependent validation.
    for item, view in zip(data["keys"], views):
        for index, matches in item["matches"].items():
            target = by_index.get(index)
            if target is not None:
                view.match_cache[index] = (view.features, target.features, matches)
    return Fragment(data["index"], keys=views, train=unpack_cloud(data["train"]),
                    heldout=unpack_cloud(data["heldout"]))


def prepare_fixture(session_path, report_path, fixture_path, pair_limit):
    import cv2
    import numpy as np
    import open3d as o3d
    from PIL import Image
    from scanner_server.fragments import (Fragment, _cache_matches, _disagrees, _global_seed,
                                         _has_independent_views, _matches, _prepare_fragment, _view)
    from scanner_server.appearance import propose_transform
    from scripts.profile_session import file_hash, source_hash
    from shared.settings import ScanSettings

    source_before, component_before = source_hash(), numerical_source_hash()
    preparation_before = preparation_dependencies()
    report = json.loads(report_path.read_text())
    if report["input_sha256"] != file_hash(session_path) or report["pose_seeds_used"]:
        raise ValueError("Require a measured replay of this exact ZIP without archived pose seeds")
    cv2.setRNGSeed(report["seed"])
    np.random.seed(report["seed"])
    o3d.utility.random.seed(report["seed"])
    o3d.utility.set_max_threads(20)
    settings = ScanSettings.from_dict(report["settings"])
    camera = settings.camera
    engine = SimpleNamespace(settings=settings, max_depth_m=settings.far_m,
        intrinsic=o3d.camera.PinholeCameraIntrinsic(camera.width, camera.height,
            camera.fx, camera.fy, camera.cx, camera.cy), raw_frames={}, frame_metadata={})
    def rgbd(rgb, depth):
        depth_f = depth.astype(np.float32)
        depth_f[(depth == 0) | (depth > settings.far_m * 1000)] = 0
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(depth_f)), depth_scale=1000,
            depth_trunc=settings.far_m, convert_rgb_to_intensity=False)
    engine._make_rgbd = rgbd
    learned = report["fragment_reconnection"]["fragments"]
    measured_live = {item["index"]: np.asarray(item["pose"], dtype=float)
                     for item in report["live_diagnostics"] if item["success"]}
    prepared_views, fragments = {}, {}
    prep_started = time.perf_counter()
    with zipfile.ZipFile(session_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for fragment in learned:
            poses = {item["index"]: np.asarray(item["pose"], dtype=float)
                     for item in fragment["camera_to_fragment"] + fragment["context_camera_to_fragment"]}
            keys = []
            for index in fragment["context_frame_indices"] + fragment["frame_indices"]:
                if index not in prepared_views:
                    item = manifest["frames"][report["selected_indices"][index]]
                    with archive.open(item["rgb"]) as stream, Image.open(stream) as image:
                        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
                    with archive.open(item["depth"]) as stream, Image.open(stream) as image:
                        depth = np.array(image, dtype=np.uint16)
                    engine.raw_frames[index] = (rgb, depth)
                    engine.frame_metadata[index] = report["live_diagnostics"][index]["metadata"]
                    prepared_views[index] = _view(engine, index)
                    del engine.raw_frames[index]
                keys.append(replace(prepared_views[index], pose=poses[index]))
            count = len(fragment["context_frame_indices"])
            current = Fragment(fragment["id"], views=keys[count:], context=keys[:count])
            for view in current.views:
                if view.index in measured_live:
                    current.seed = measured_live[view.index] @ np.linalg.inv(view.pose)
                    break
            _prepare_fragment(current)
            fragments[current.index] = current
    eligible = {index for index, fragment in fragments.items() if _has_independent_views(fragment)}
    original_edges = report["fragment_reconnection"]["verified_bridges"]
    sequential = {(edge["source"], edge["target"]) for edge in original_edges
                  if edge.get("validation_scope") == "sequential camera pair"}
    verified = [(edge["source"], edge["target"]) for edge in original_edges
                if (edge["source"], edge["target"]) not in sequential
                and edge["source"] in eligible and edge["target"] in eligible]
    candidates = [(i, j) for i in sorted(eligible) for j in sorted(eligible) if i < j
                  and (i, j) not in sequential and (i, j) not in verified]
    count = max(1, pair_limit // 2)
    selected = verified[:count] + candidates[:pair_limit - count]
    tasks = []
    for position, (a, b) in enumerate(selected):
        source, target = fragments[a], fragments[b]
        appearances = []
        if source.seed is not None and target.seed is not None:
            appearances.append((0, np.linalg.inv(target.seed) @ source.seed))
        appearances.append((0, np.eye(4)))
        for aa in source.keys:
            for bb in target.keys:
                matches = _matches(aa, bb)
                proposal = propose_transform(aa.features, bb.features, camera, matches)
                if proposal is not None:
                    appearances.append((len(matches), bb.pose @ proposal @ np.linalg.inv(aa.pose)))
        proposals = [pose for _, pose in sorted(appearances, key=lambda row: -row[0])[:4]]
        for seed in (a * 100 + b, a * 100 + b + 10000):
            proposal = _global_seed(source, target, seed)
            if proposal is not None:
                proposals.append(proposal)
        unique = []
        for proposal in proposals:
            if not any(not _disagrees(proposal, previous, .01, 1) for previous in unique):
                unique.append(proposal)
        tasks.append({"position": position, "pair": [a, b], "proposals": unique})
        print(f"Prepared pair {a}/{b}: {len(unique)} unchanged bridge proposals", flush=True)
    metadata = {"input_sha256": file_hash(session_path), "source_sha256": source_before,
                "component_source_sha256": component_before, "measured_fragment_report_sha256": file_hash(report_path),
                "verification_dependencies": verification_dependencies(),
                "preparation_dependencies": preparation_before,
                "measured_fragment_source_sha256": report["source_sha256"],
                "session": str(session_path.resolve()), "report": str(report_path.resolve()),
                "prepared_s": time.perf_counter() - prep_started,
                "scope": "fixed prepared bridge tasks; no live replay, graph search, fresh fusion or final mesh timing",
                "local_pose_source": "measured Finish fragment report; no archived ZIP poses",
                "rng_policy": "all original RANSAC seeds and visual proposals prepared serially",
                "versions": {"python": sys.version, "numpy": np.__version__,
                             "open3d": o3d.__version__, "opencv": cv2.__version__},
                "native_mode": os.environ["KINECT_NATIVE"],
                "omp_threads": os.environ["OMP_NUM_THREADS"],
                "fixture_pair_order": [task["pair"] for task in tasks]}
    native = importlib.util.find_spec("_kinect_native")
    metadata["native_extension"] = ({"path": native.origin, "sha256": file_hash(Path(native.origin))}
                                    if native and native.origin else None)
    if component_before != numerical_source_hash() or preparation_before != preparation_dependencies():
        raise RuntimeError("Numerically relevant implementation changed during fixture preparation")
    metadata["runtime_source_sha256_after_preparation"] = source_hash()
    metadata["runtime_changed_during_preparation"] = metadata["runtime_source_sha256_after_preparation"] != source_before
    snapshot_path = fixture_path.with_suffix(".runtime.zip")
    digest = hashlib.sha256()
    with zipfile.ZipFile(snapshot_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in (ROOT / "scanner_server", ROOT / "shared", ROOT / "native"):
            for path in sorted(folder.rglob("*")):
                if path.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".toml") and "build" not in path.parts:
                    payload = path.read_bytes()
                    relative = str(path.relative_to(ROOT))
                    digest.update(relative.encode())
                    digest.update(payload)
                    archive.writestr(relative.replace("\\", "/"), payload)
    metadata["runtime_snapshot"] = {"path": str(snapshot_path.resolve()), "sha256": file_hash(snapshot_path),
                                    "contained_source_sha256": digest.hexdigest()}
    fixture = {"metadata": metadata, "camera": camera,
               "fragments": {index: pack_fragment(fragment) for index, fragment in fragments.items()}, "tasks": tasks}
    with fixture_path.open("wb") as destination:
        pickle.dump(fixture, destination, protocol=pickle.HIGHEST_PROTOCOL)
    return metadata


def worker(fixture_path, output_path, worker_id, worker_count, threads):
    import cv2
    import numpy as np
    import open3d as o3d
    from scanner_server.fragments import _disagrees, _verify_bridge
    from scripts.process_metrics import finish_cuda_worker, peak_rss_bytes

    o3d.utility.set_max_threads(threads)
    cv2.setNumThreads(threads)
    with fixture_path.open("rb") as source:
        fixture = pickle.load(source)
    if fixture["metadata"]["component_source_sha256"] != numerical_source_hash():
        raise RuntimeError("Numerically relevant implementation changed before worker measurement")
    fragments = {index: unpack_fragment(data) for index, data in fixture["fragments"].items()}
    # Reinstall the immutable exact match cache using this worker's feature identities.
    from scanner_server.fragments import _cache_matches
    for index, data in fixture["fragments"].items():
        for original, view in zip(data["keys"], fragments[index].keys):
            for other_fragment in fragments.values():
                for other in other_fragment.keys:
                    matches = original["matches"].get(other.index)
                    if matches is not None:
                        _cache_matches(view, other, matches.copy())
    rows = []
    for task in fixture["tasks"][worker_id::worker_count]:
        start = time.perf_counter()
        a, b = task["pair"]
        verified = []
        for pose in task["proposals"]:
            result = _verify_bridge(fragments[a], fragments[b], pose, fixture["camera"])
            if result is not None:
                verified.append(result)
        ambiguous = bool(verified) and any(_disagrees(verified[0]["transform"], result["transform"])
                                           for result in verified[1:])
        best = verified[0] if verified and not ambiguous else None
        rows.append({"position": task["position"], "pair": task["pair"],
                     "proposal_count": len(task["proposals"]), "verified_proposals": len(verified),
                     "ambiguous": ambiguous, "accepted": best is not None,
                     "best": json_value(best), "elapsed_s": time.perf_counter() - start})
        print(f"Worker {worker_id}, pair {a}/{b}: {rows[-1]['elapsed_s']:.3f}s accepted={best is not None}", flush=True)
    if fixture["metadata"]["component_source_sha256"] != numerical_source_hash():
        raise RuntimeError("Numerically relevant implementation changed during worker measurement")
    output_path.write_text(json.dumps({"threads": o3d.utility.get_max_threads(), "opencv_threads": cv2.getNumThreads(), "worker_id": worker_id,
        "peak_process_rss_bytes": peak_rss_bytes(), "rows": rows}, indent=2, allow_nan=False) + "\n")
    finish_cuda_worker()


def queue_worker(fixture_path, output_path, worker_id, worker_count, threads, queue_path):
    """A persistent process claims jobs atomically after all processes are ready."""
    import cv2
    import open3d as o3d
    from scanner_server.fragments import _cache_matches, _disagrees, _verify_bridge
    from scripts.process_metrics import finish_cuda_worker, peak_rss_bytes

    o3d.utility.set_max_threads(threads)
    cv2.setNumThreads(threads)
    with fixture_path.open("rb") as source:
        fixture = pickle.load(source)
    if fixture["metadata"]["component_source_sha256"] != numerical_source_hash():
        raise RuntimeError("Numerically relevant implementation changed before measurement")
    fragments = {index: unpack_fragment(data) for index, data in fixture["fragments"].items()}
    for index, data in fixture["fragments"].items():
        for original, view in zip(data["keys"], fragments[index].keys):
            for other_fragment in fragments.values():
                for other in other_fragment.keys:
                    matches = original["matches"].get(other.index)
                    if matches is not None:
                        _cache_matches(view, other, matches.copy())
    connection = sqlite3.connect(queue_path, timeout=30)
    connection.execute("INSERT INTO workers VALUES (?, ?)", (worker_id, time.perf_counter()))
    connection.commit()
    while connection.execute("SELECT COUNT(*) FROM workers").fetchone()[0] < worker_count:
        time.sleep(.01)
    rows = []
    while True:
        connection.execute("BEGIN IMMEDIATE")
        mode, current_pair = connection.execute("SELECT mode, current_pair FROM control").fetchone()
        if mode == "proposals":
            job = connection.execute("SELECT id,payload FROM jobs WHERE state='pending' AND pair_position=? ORDER BY id LIMIT 1",
                                     (current_pair,)).fetchone()
        else:
            job = connection.execute("SELECT id,payload FROM jobs WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
        if job is None:
            unfinished = connection.execute("SELECT COUNT(*) FROM jobs WHERE state!='done'").fetchone()[0]
            active_current = connection.execute("SELECT COUNT(*) FROM jobs WHERE pair_position=? AND state!='done'",
                                                (current_pair,)).fetchone()[0]
            if mode == "proposals" and unfinished and not active_current:
                connection.execute("UPDATE control SET current_pair=current_pair+1")
            connection.commit()
            if not unfinished:
                break
            time.sleep(.001)
            continue
        job_id, payload = job
        connection.execute("UPDATE jobs SET state='active',worker=? WHERE id=?", (worker_id, job_id))
        connection.commit()
        task = pickle.loads(payload)
        started = time.perf_counter()
        a, b = task["pair"]
        verified, proposals = [], []
        for proposal_index, pose in enumerate(task["proposals"]):
            proposal_start = time.perf_counter()
            result = _verify_bridge(fragments[a], fragments[b], pose, fixture["camera"])
            proposals.append({"proposal_index": task.get("proposal_index", proposal_index),
                              "result": json_value(result), "elapsed_s": time.perf_counter() - proposal_start})
            if result is not None:
                verified.append(result)
        ambiguous = bool(verified) and any(_disagrees(verified[0]["transform"], result["transform"])
                                           for result in verified[1:])
        best = verified[0] if verified and not ambiguous else None
        row = {"position": task["position"], "pair": task["pair"], "job_id": job_id,
               "proposal_count": len(task["proposals"]), "verified_proposals": len(verified),
               "ambiguous": ambiguous, "accepted": best is not None, "best": json_value(best),
               "started_at": started, "ended_at": time.perf_counter(), "proposal_results": proposals}
        row["elapsed_s"] = row["ended_at"] - started
        rows.append(row)
        connection.execute("UPDATE jobs SET state='done' WHERE id=?", (job_id,))
        connection.commit()
        print(f"Worker {worker_id}, pair {a}/{b}, job {job_id}: {row['elapsed_s']:.3f}s", flush=True)
    connection.close()
    if fixture["metadata"]["component_source_sha256"] != numerical_source_hash():
        raise RuntimeError("Numerically relevant implementation changed during measurement")
    output_path.write_text(json.dumps({"threads": o3d.utility.get_max_threads(), "opencv_threads": cv2.getNumThreads(),
        "worker_id": worker_id, "peak_process_rss_bytes": peak_rss_bytes(), "rows": rows}, indent=2, allow_nan=False) + "\n")
    finish_cuda_worker()


def initialize_queue(queue_path, fixture_path, scheduling):
    if queue_path.exists():
        queue_path.unlink()
    with fixture_path.open("rb") as source:
        tasks = pickle.load(source)["tasks"]
    connection = sqlite3.connect(queue_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE jobs(id INTEGER PRIMARY KEY, pair_position INTEGER, payload BLOB, state TEXT, worker INTEGER)")
    connection.execute("CREATE TABLE workers(id INTEGER PRIMARY KEY, ready_at REAL)")
    connection.execute("CREATE TABLE control(mode TEXT, current_pair INTEGER)")
    connection.execute("INSERT INTO control VALUES (?,0)", (scheduling,))
    identifier = 0
    for task in tasks:
        pieces = ([dict(task, proposals=[proposal], proposal_index=index)
                   for index, proposal in enumerate(task["proposals"])] if scheduling == "proposals" else [task])
        for piece in pieces:
            connection.execute("INSERT INTO jobs VALUES (?,?,?,'pending',NULL)",
                               (identifier, task["position"], pickle.dumps(piece, protocol=pickle.HIGHEST_PROTOCOL)))
            identifier += 1
    connection.commit()
    connection.close()


def aggregate_proposal_rows(rows):
    """Consume competing proposal evidence in original order before acceptance."""
    import numpy as np
    from scanner_server.fragments import _disagrees

    result = []
    for position in sorted({row["position"] for row in rows}):
        selected = [row for row in rows if row["position"] == position]
        proposals = sorted((proposal for row in selected for proposal in row["proposal_results"]),
                           key=lambda item: item["proposal_index"])
        verified = [proposal["result"] for proposal in proposals if proposal["result"] is not None]
        ambiguous = bool(verified) and any(_disagrees(np.asarray(verified[0]["transform"]), np.asarray(item["transform"]))
                                           for item in verified[1:])
        best = verified[0] if verified and not ambiguous else None
        result.append({"position": position, "pair": selected[0]["pair"], "proposal_count": len(proposals),
                       "verified_proposals": len(verified), "ambiguous": ambiguous, "accepted": best is not None,
                       "best": best, "elapsed_s": max(row["ended_at"] for row in selected) - min(row["started_at"] for row in selected),
                       "proposal_results": proposals})
    return result


def compare_rows(reference, candidate):
    import numpy as np
    from scanner_server.refinement import motion

    differences = ([] if len(reference) == len(candidate) else [{"reason": "changed number of consumed pair results"}])
    max_translation = max_rotation = max_information = 0.0
    for old, new in zip(reference, candidate):
        if (old["pair"], old["accepted"], old["ambiguous"], old["verified_proposals"], old["proposal_count"]) != (
                new["pair"], new["accepted"], new["ambiguous"], new["verified_proposals"], new["proposal_count"]):
            differences.append({"pair": old["pair"], "reason": "changed decision or proposal evidence"})
            continue
        if old["accepted"]:
            a, b = old["best"], new["best"]
            translation, angle = motion(np.linalg.inv(np.asarray(a["transform"])) @ np.asarray(b["transform"]))
            info_delta = np.max(np.abs(np.asarray(a["information"]) - np.asarray(b["information"])))
            max_translation, max_rotation, max_information = (max(max_translation, translation),
                max(max_rotation, angle), max(max_information, float(info_delta)))
            if (a["source"], a["target"], a["support"], a["validation_scope"]) != (
                    b["source"], b["target"], b["support"], b["validation_scope"]):
                differences.append({"pair": old["pair"], "reason": "changed independent witness set"})
            if translation > 1e-6 or angle > 1e-4 or info_delta > 1e-5:
                differences.append({"pair": old["pair"], "reason": "changed transform/information beyond numerical tolerance"})
    return {"same_decisions_and_support": not differences, "differences": differences,
            "max_transform_translation_delta_m": max_translation, "max_transform_rotation_delta_deg": max_rotation,
            "max_information_absolute_delta": max_information}


def stop_owned_worker(process):
    """The Windows venv redirector may have a live Python child after kill()."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=10, check=False)
    else:
        process.kill()
    process.wait(timeout=10)


def cleanup_queue(path):
    if path is not None:
        for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
            candidate.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--measured-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--configurations", nargs="+", default=("1x20", "2x10", "4x5"))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--scheduling", choices=("static", "dynamic", "proposals"), default="static")
    parser.add_argument("--thread-budget", type=int, default=20,
                        help="Explicit total process TBB budget; 40 measures oversubscription on this twenty-thread CPU")
    parser.add_argument("--reference", type=Path, help="An earlier serial component result for exact decision comparison")
    parser.add_argument("--timeout-seconds", type=float, default=240,
                        help="Whole-configuration deadline, including all worker imports")
    args = parser.parse_args()
    if args.worker is not None:
        if args.queue:
            queue_worker(args.fixture, args.output, args.worker, args.worker_count, args.threads, args.queue)
        else:
            worker(args.fixture, args.output, args.worker, args.worker_count, args.threads)
        return
    if args.pairs < 2 or args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
        parser.error("Require at least two representative pairs and a positive timeout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = args.fixture or args.output.with_suffix(".fixture.pickle")
    if not args.fixture:
        if not args.session or not args.measured_report:
            parser.error("Require --session and --measured-report when preparing a new fixture")
        metadata = prepare_fixture(args.session, args.measured_report, fixture_path, args.pairs)
    else:
        with fixture_path.open("rb") as source:
            metadata = pickle.load(source)["metadata"]
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    results = []
    reference = None
    if args.reference:
        reference_report = json.loads(args.reference.read_text())
        if (reference_report["metadata"].get("fixture_sha256") != fixture_hash
                or reference_report["metadata"]["component_source_sha256"] != numerical_source_hash()):
            parser.error("Reference must use this exact prepared fixture and verification implementation")
        reference = reference_report["results"][0]
        if reference["workers"] != 1:
            parser.error("Reference must begin with a serial control")
    script_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for configuration in args.configurations:
        workers, threads = map(int, configuration.split("x"))
        if min(workers, threads) < 1 or workers * threads > args.thread_budget:
            parser.error("Configuration exceeds the explicit total TBB thread budget")
        if not results and reference is None and workers != 1:
            parser.error("Start with a serial worker or provide --reference")
        started = time.perf_counter()
        queue_path = None
        if args.scheduling != "static":
            queue_path = args.output.with_name(args.output.stem + f"-{configuration}.queue.sqlite")
            initialize_queue(queue_path, fixture_path, args.scheduling)
        jobs = []
        try:
            for index in range(workers):
                destination = args.output.with_name(args.output.stem + f"-{configuration}-worker-{index}.json")
                destination.unlink(missing_ok=True)
                log = destination.with_suffix(".log").open("w")
                command = [sys.executable, str(Path(__file__).resolve()), "--fixture", str(fixture_path),
                    "--output", str(destination), "--worker", str(index), "--worker-count", str(workers),
                    "--threads", str(threads)]
                if queue_path:
                    command.extend(("--queue", str(queue_path)))
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                except Exception:
                    log.close()
                    raise
                jobs.append((process, destination, log))
            for process, destination, log in jobs:
                remaining = max(.001, args.timeout_seconds - (time.perf_counter() - started))
                if process.wait(timeout=remaining):
                    raise RuntimeError(f"Worker failed; inspect {destination.with_suffix('.log')}")
        finally:
            cleanup_errors = []
            for process, _, log in jobs:
                try:
                    stop_owned_worker(process)
                except Exception as error:
                    cleanup_errors.append(error)
                finally:
                    log.close()
            try:
                cleanup_queue(queue_path)
            except Exception as error:
                cleanup_errors.append(error)
            if cleanup_errors:
                raise RuntimeError("Worker cleanup failed; retained logs identify the owned processes") from cleanup_errors[0]
        elapsed = time.perf_counter() - started
        parts = [json.loads(destination.read_text()) for _, destination, _ in jobs]
        if any(part["worker_id"] != index or part["threads"] != threads or part["opencv_threads"] != threads
               for index, part in enumerate(parts)):
            raise RuntimeError("Worker identity or configured resource limits changed")
        raw_rows = [row for part in parts for row in part["rows"]]
        rows = (aggregate_proposal_rows(raw_rows) if args.scheduling == "proposals"
                else sorted(raw_rows, key=lambda row: row["position"]))
        active_s = max(sum(row["elapsed_s"] for row in part["rows"]) for part in parts)
        compute_s = (max(row["ended_at"] for row in raw_rows) - min(row["started_at"] for row in raw_rows)
                     if raw_rows and args.scheduling != "static" else active_s)
        result = {"configuration": configuration, "workers": workers, "threads_per_worker": threads,
                  "wall_s_including_imports_and_fixture_loading": elapsed,
                  "critical_path_verification_s": compute_s,
                  "max_worker_active_verification_s": active_s,
                  "worker_peak_rss_bytes_sum": sum(part["peak_process_rss_bytes"] for part in parts),
                  "rows": rows}
        if raw_rows and args.scheduling != "static":
            result["persistent_workers_verification_span_s"] = max(row["ended_at"] for row in raw_rows) - min(row["started_at"] for row in raw_rows)
        reference_row = reference or (results[0] if results else None)
        if reference_row:
            result["agreement_vs_serial"] = compare_rows(reference_row["rows"], rows)
            result["wall_speedup_vs_serial"] = reference_row["wall_s_including_imports_and_fixture_loading"] / elapsed
            result["verification_speedup_vs_serial"] = reference_row["critical_path_verification_s"] / compute_s
        results.append(result)
        if script_before != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError("Experiment script changed during measurement")
        args.output.write_text(json.dumps({"metadata": dict(metadata, scheduling=args.scheduling,
            script_sha256=script_before, reference=str(args.reference.resolve()) if args.reference else None,
            tbb_thread_budget=args.thread_budget, fixture_sha256=fixture_hash), "results": results}, indent=2, allow_nan=False) + "\n")
        print(f"{configuration}: wall={elapsed:.3f}s verification={compute_s:.3f}s accepted={sum(row['accepted'] for row in rows)}", flush=True)
    from scripts.process_metrics import finish_cuda_worker
    finish_cuda_worker()


if __name__ == "__main__":
    main()
