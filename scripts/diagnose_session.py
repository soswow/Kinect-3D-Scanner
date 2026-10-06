"""Audit saved live tracking failures and surface extraction in an isolated engine.

Reintegrates archived accepted poses as estimates, without claiming they are
correct or reconnecting fragments. Recomputes the first rejected observation of
each loss episode against that saved prefix. Never contacts a running scanner.
Indices in JSON are zero based. Run with the scanner's Python environment:

python scripts/diagnose_session.py session.zip --output-dir benchmark-output/audit
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import open3d as o3d

from scanner_server.engine import ScanEngine
from scanner_server.fragments import capture_gap_limit
from scanner_server.refinement import _match, _trustworthy, motion
from scripts.reconnect_session import load_pose_seeds, load_session
from shared.calibration import prepare_rgbd


def match_metrics(result, target):
    matches = np.asarray(result.correspondence_set)
    normals = np.asarray(target.normals)[matches[:, 1]]
    return {
        "fitness": float(result.fitness),
        "rmse_m": float(result.inlier_rmse),
        "correspondences": len(matches),
        "normal_eigenvalues": (
            np.linalg.eigvalsh(normals.T @ normals / len(normals)).tolist()
            if len(normals) else []
        ),
        "trustworthy": bool(_trustworthy(result, target)),
    }


def audit_transition(engine, source, result):
    index, last = engine.poses[-1]
    relative = np.linalg.inv(last) @ result.transformation
    translation, angle = motion(relative)
    stamp = engine.frame_metadata[engine._processed_count]["timestamp_s"]
    last_stamp = engine.frame_metadata[index]["timestamp_s"]
    gap_limit = capture_gap_limit(
        engine.frame_metadata[max(0, index - 8):engine._processed_count + 1]
    )
    target = engine._last_reg_pcd
    forward = _match(source, target, relative)
    reverse = _match(target, source, np.linalg.inv(forward.transformation))
    cycle_m, cycle_deg = motion(reverse.transformation @ forward.transformation)
    difference_m, difference_deg = motion(
        np.linalg.inv(relative) @ forward.transformation
    )
    return {
        "anchor_index": index,
        "gap_s": stamp - last_stamp,
        "gap_limit_s": gap_limit,
        "translation_m": translation,
        "rotation_deg": angle,
        "forward": match_metrics(forward, target),
        "reverse": match_metrics(reverse, source),
        "cycle_m": cycle_m,
        "cycle_deg": cycle_deg,
        "model_anchor_difference_m": difference_m,
        "model_anchor_difference_deg": difference_deg,
        "rejection": engine._verify_tracking_transition(source, result),
    }


def surfaces(engine, output, export_meshes):
    rows = []
    for threshold in (0.01, 0.2, 0.5, 1.0, 2.0):
        mesh = engine.vbg.extract_triangle_mesh(weight_threshold=threshold).to_legacy()
        points = engine.vbg.extract_point_cloud(weight_threshold=threshold)
        row = {"weight": threshold, "points": len(points.point.positions),
               "raw_triangles": len(mesh.triangles)}
        del points
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        row["triangles_before_component_filter"] = len(mesh.triangles)
        if len(mesh.triangles) and engine.settings.min_component_triangles:
            labels, sizes, _ = mesh.cluster_connected_triangles()
            mask = (np.asarray(sizes)[np.asarray(labels)]
                    < engine.settings.min_component_triangles)
            mesh.remove_triangles_by_mask(mask)
            mesh.remove_unreferenced_vertices()
        row.update(triangles=len(mesh.triangles), vertices=len(mesh.vertices),
                   area_m2=float(mesh.get_surface_area()))
        if export_meshes and threshold in (0.2, 2.0):
            path = output / f"saved-live-poses-weight-{threshold:g}.ply"
            mesh.compute_vertex_normals()
            if not o3d.io.write_triangle_mesh(str(path), mesh):
                raise OSError(f"Cannot write {path}")
            row["mesh"] = str(path.resolve())
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def diagnose(path, output, export_meshes=False):
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        archived = json.loads(archive.read(manifest.get("reconstruction", "reconstruction.json")))
    engine = ScanEngine(device="cpu", tracking="legacy")
    load_session(engine, path)
    load_pose_seeds(engine, path)  # Validate saved indices and rigid transforms.
    saved_poses = dict(engine.poses)
    engine.poses = []
    engine.frame_count = 0
    engine._pose_seeds_only = False
    frames = {f["index"]: f for f in archived["frames"]}
    audits = []
    for index in range(engine.stored_count):
        engine._processed_count = index
        rgb, depth = prepare_rgbd(*engine.raw_frames[index], engine.settings)
        rgbd = engine._make_rgbd(rgb, depth)
        cloud = engine._make_reg_pcd(rgbd)
        if index in saved_poses:
            pose = saved_poses[index]
            engine._integrate_vbg(rgb, depth, np.linalg.inv(pose))
            engine.poses.append((index, pose))
            engine.cumulative_T = pose
            engine._last_rgbd = rgbd
            engine._last_reg_pcd = cloud
            engine.frame_count += 1
            engine._integrations_since_model += 1
        elif engine.poses and index - 1 in saved_poses:
            engine._tracking_lost_frames = 0
            engine._extract_model_pcd()
            result, method = engine._register(cloud, rgbd)
            audit = {"index": index, "archived_message": frames[index].get("message"),
                     "recomputed_method_or_error": method}
            if result is not None:
                audit.update(model_fitness=float(result.fitness),
                             model_rmse_m=float(result.inlier_rmse),
                             transition=audit_transition(engine, cloud, result))
            audits.append(audit)
            (output / "transitions.json").write_text(json.dumps(audits, indent=2, allow_nan=False))
            print(json.dumps(audit), flush=True)
    engine._extract_model_pcd()
    report = {
        "source": str(path.resolve()),
        "archived_session_id": archived.get("session_id"),
        "archived_accepted": len(saved_poses),
        "limitation": "Saved live poses are estimates, not ground truth. No offline reconnection or final refinement was applied. Recomputed candidates can differ from the original native solver.",
        "transitions": audits,
        "surface_sweep": surfaces(engine, output, export_meshes),
        "blocks": int(engine.vbg.hashmap().size()),
    }
    (output / "diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-meshes", action="store_true",
                        help="Export illustrative meshes from unvalidated saved live poses")
    args = parser.parse_args()
    diagnose(args.session, args.output_dir, args.export_meshes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
