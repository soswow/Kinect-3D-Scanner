"""Audit a saved project's storage and sampled final-surface contributions.

Reads an explicit session ZIP; never contacts a server, registers cameras, fuses
depth or deletes captures. Requires scanner numerical dependencies for surface
analysis. Optional compact output removes only reproducible display image copies,
retaining all native captures, poses, calibration, sensor records and saved mesh.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def same_file(left, right):
    left, right = Path(left), Path(right)
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def storage_audit(archive, manifest, accepted):
    """Count unique ZIP payloads; never equate a registered fragment with fusion."""
    members = {item.filename: item for item in archive.infolist()}
    if len(members) != len(archive.infolist()):
        raise ValueError("Duplicate ZIP member names are unsupported")
    rows = []
    native, display, unused, used = set(), set(), set(), set()
    for index, capture in enumerate(manifest["frames"]):
        original = {capture[key] for key in ("rgb", "depth")}
        shown = {capture[key] for key in ("display_rgb", "display_depth") if key in capture}
        native.update(original)
        display.update(shown)
        if index not in accepted:
            unused.update(original | shown)
        else:
            used.update(original | shown)
        rows.append({
            "index": index, "shot_number": index + 1,
            "in_final_poses": index in accepted,
            "native_payload_bytes": sum(members[name].compress_size for name in original),
            "display_payload_bytes": sum(members[name].compress_size for name in shown),
        })
    # A display member shared with a native capture is not removable.
    removable_display = display - native
    return rows, {
        "native_image_payload_bytes": sum(members[name].compress_size for name in native),
        "display_image_payload_bytes": sum(members[name].compress_size for name in removable_display),
        "outside_final_pose_image_payload_bytes": sum(members[name].compress_size for name in unused - used),
        "mesh_payload_bytes": members[manifest["mesh"]].compress_size if manifest.get("mesh") else 0,
        "zip_member_count": len(members),
        "note": "Compressed member payloads; excludes ZIP headers. Categories overlap.",
    }


def compact_display_copies(source_path, output_path):
    """Write a new lossless project, verifying retained bytes before publishing.

    Only the manifest's optional display keys and their exclusively used image
    members are omitted. Capture indices, native encodings and all other records
    stay intact. Existing destinations and input aliases are refused.
    """
    source_path, output_path = Path(source_path), Path(output_path)
    if same_file(source_path, output_path) or output_path.exists():
        raise ValueError("Compact output must be a new path distinct from the source")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with zipfile.ZipFile(source_path) as source:
            manifest = json.loads(source.read("manifest.json"))
            if "local_captures" in manifest:
                raise ValueError("Complete the local-capture transfer before compacting")
            names = source.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP member names are unsupported")
            changed = copy.deepcopy(manifest)
            display = set()
            for frame in changed["frames"]:
                for key in ("display_rgb", "display_depth"):
                    if key in frame:
                        display.add(frame.pop(key))
            # Retain anything referenced elsewhere in the manifest, even if
            # another capture also calls it a generated display image.
            def strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from strings(item)
                elif isinstance(value, list):
                    for item in value:
                        yield from strings(item)
            removed = display - set(strings(changed))
            if any(not name.startswith("display/") or name not in names for name in removed):
                raise ValueError("Expected generated display/ image members")
            with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".zip", delete=False) as handle:
                temporary = Path(handle.name)
            hashes = {}
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as target:
                for item in source.infolist():
                    if item.filename in removed or item.filename == "manifest.json":
                        continue
                    info = copy.copy(item)
                    digest = hashlib.sha256()
                    with source.open(item) as incoming, target.open(info, "w", force_zip64=True) as outgoing:
                        while chunk := incoming.read(1024 * 1024):
                            digest.update(chunk)
                            outgoing.write(chunk)
                    hashes[item.filename] = digest.hexdigest()
                target.writestr("manifest.json", json.dumps(changed, separators=(",", ":"), allow_nan=False),
                                compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            with zipfile.ZipFile(temporary) as candidate:
                if set(candidate.namelist()) != set(hashes) | {"manifest.json"}:
                    raise ValueError("Compact project members failed verification")
                if json.loads(candidate.read("manifest.json")) != changed:
                    raise ValueError("Compact manifest failed verification")
                for name, expected in hashes.items():
                    digest = hashlib.sha256()
                    with candidate.open(name) as member:
                        while chunk := member.read(1024 * 1024):
                            digest.update(chunk)
                    if digest.hexdigest() != expected:
                        raise ValueError(f"Compact member changed: {name}")
        # Publish without overwriting a destination created during verification.
        os.link(temporary, output_path)
        return {"path": str(output_path.resolve()), "bytes": output_path.stat().st_size,
                "removed_display_members": len(removed), "retained_members_sha256_verified": len(hashes)}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_redundant(weights, protected, min_observations=3, quality_ratio=0.9):
    """Reverse-delete views while preserving support and near-best incidence.

    This is a sampled fixed-pose proposal, not proof of unchanged TSDF fusion,
    texture output, or camera-graph connectivity. Row indices identify views.
    """
    import numpy as np

    visible = weights > 0
    counts = visible.sum(axis=0)
    required = np.minimum(counts, min_observations)
    best = weights.max(axis=0)
    quality = visible & (weights >= best[None, :] * quality_ratio)
    quality_counts = quality.sum(axis=0)
    kept = np.ones(len(weights), dtype=bool)
    for row in sorted(range(len(weights)), key=lambda i: (int(visible[i].sum()), i)):
        if row in protected:
            continue
        if np.any(counts[visible[row]] <= required[visible[row]]):
            continue
        if np.any(quality_counts[quality[row]] <= 1):
            continue
        kept[row] = False
        counts -= visible[row]
        quality_counts -= quality[row]
    return kept


def sample_surface(mesh, count, seed):
    import numpy as np

    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    total = lengths.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Saved mesh has no finite positive surface area")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(triangles), count, p=lengths / total)
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    points = (triangles[ids, 0] + uv[:, :1] * (triangles[ids, 1] - triangles[ids, 0])
              + uv[:, 1:] * (triangles[ids, 2] - triangles[ids, 0]))
    return points, cross[ids] / lengths[ids, None], total / 2


def surface_weights(world, normals, depth, pose, settings):
    """Depth-only visibility with the exporter's measured-depth tolerance."""
    import numpy as np

    camera = (world - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    c = settings.camera
    u = camera[:, 0] * c.fx / np.maximum(z, 1e-6) + c.cx
    v = camera[:, 1] * c.fy / np.maximum(z, 1e-6) + c.cy
    x = np.clip(np.rint(u), 0, c.width - 1).astype(int)
    y = np.clip(np.rint(v), 0, c.height - 1).astype(int)
    observed = depth[y, x]
    direction = pose[:3, 3] - world
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    cosine = np.abs(np.sum(normals * direction, axis=1))
    visible = ((z > settings.near_m) & (z < settings.far_m)
               & (u >= 0) & (u < c.width - 1) & (v >= 0) & (v < c.height - 1)
               & (observed > 0) & (np.abs(observed - z) <= np.maximum(0.015, 0.01 * z))
               & (cosine > 0.25))
    return np.where(visible, cosine**2 / np.maximum(z**2, 0.1), 0).astype(np.float32)


def analyze(path, output, samples=50000, min_observations=3, quality_ratio=0.9, storage_only=False):
    started = time.perf_counter()
    output = Path(output)
    for name in ("summary.json", "shots.csv", "compact-verification.json"):
        if same_file(path, output / name):
            raise ValueError("Report output aliases the source ZIP")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if "local_captures" in manifest:
            raise ValueError("Incomplete project transfer")
        reconstruction = json.loads(archive.read(manifest.get("reconstruction", "reconstruction.json")))
        poses = sorted(reconstruction.get("poses", []), key=lambda row: row["index"])
        accepted = {row["index"] for row in poses}
        if len(accepted) != len(poses) or any(type(i) is not int or not 0 <= i < len(manifest["frames"]) for i in accepted):
            raise ValueError("Invalid or duplicate saved pose index")
        diagnostics = reconstruction.get("frames", [])
        if accepted != {d["index"] for d in diagnostics if d.get("success")}:
            raise ValueError("Saved poses disagree with final frame diagnostics")
        rows, storage = storage_audit(archive, manifest, accepted)
        summary = {
            "source": str(Path(path).resolve()), "source_bytes": Path(path).stat().st_size,
            "source_sha256": file_digest(path),
            "analysis_source_sha256": file_digest(__file__),
            "session_id": reconstruction.get("session_id"),
            "captures": len(rows), "final_pose_captures": len(accepted),
            "outside_final_pose_indices": [i for i in range(len(rows)) if i not in accepted],
            "storage": storage, "surface_analysis": {"executed": False},
            "index_convention": "Indices are zero based; shot_number is one based.",
            "limitations": [
                "Outside-final captures can contain recoverable surfaces in disconnected maps.",
                "Coverage redundancy does not establish unchanged fusion, color or registration.",
                "No shots are deleted; evaluate fresh fusion and held-out depth before pruning accepted views.",
            ],
        }
        if not storage_only:
            if not poses or not manifest.get("mesh"):
                raise ValueError("Surface analysis requires accepted poses and a saved mesh; use --storage-only otherwise")
            import open3d as o3d
            import numpy as np
            from PIL import Image
            from types import SimpleNamespace
            from scanner_server.fragments import _rigid
            from scanner_server.texturing import _views
            from shared.calibration import prepare_metric_depth
            from shared.settings import ScanSettings

            if reconstruction.get("pose_convention") != "camera_to_world" or reconstruction.get("length_unit") != "metres":
                raise ValueError("Surface analysis requires camera_to_world poses in metres")
            settings = ScanSettings.from_dict(manifest["settings"])
            saved_poses = [(row["index"], np.asarray(row["camera_to_world"], float)) for row in poses]
            if not all(_rigid(pose) for _, pose in saved_poses):
                raise ValueError("Non-rigid saved camera pose")
            class Images:
                def __getitem__(self, index):
                    with archive.open(manifest["frames"][index]["rgb"]) as member, Image.open(member) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    if rgb.shape != (settings.rgb_camera.height, settings.rgb_camera.width, 3):
                        raise ValueError("RGB dimensions disagree with calibration")
                    return rgb, None
            proxy = SimpleNamespace(poses=saved_poses, raw_frames=Images(),
                                    frame_metadata=[f.get("metadata", {}) for f in manifest["frames"]])
            texture_indices = {i for i, _ in _views(proxy, 64)}
            with tempfile.TemporaryDirectory() as directory:
                mesh_path = Path(directory) / "mesh.ply"
                mesh_path.write_bytes(archive.read(manifest["mesh"]))
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            training = sample_surface(mesh, samples, 20261011)
            validation = sample_surface(mesh, samples, 20261012)
            points = np.concatenate((training[0], validation[0]))
            normals = np.concatenate((training[1], validation[1]))
            weights = np.empty((len(poses), 2 * samples), np.float32)
            for row, (index, pose) in enumerate(saved_poses):
                with archive.open(manifest["frames"][index]["depth"]) as member, Image.open(member) as image:
                    depth = np.asarray(image, dtype=np.uint16)
                if depth.shape != (settings.camera.height, settings.camera.width):
                    raise ValueError("Depth dimensions disagree with calibration")
                depth = prepare_metric_depth(depth, settings).astype(np.float32) / 1000
                weights[row] = surface_weights(points, normals, depth, pose, settings)
            protected = {row for row, (index, _) in enumerate(saved_poses) if index in texture_indices}
            kept = select_redundant(weights[:, :samples], protected, min_observations, quality_ratio)
            visible = weights > 0
            original_counts = visible.sum(axis=0)
            retained_counts = visible[kept].sum(axis=0)
            original_best = weights.max(axis=0)
            retained_best = weights[kept].max(axis=0) if kept.any() else np.zeros(2 * samples)
            def checks(part):
                observed = original_counts[part] > 0
                required = np.minimum(original_counts[part], min_observations)
                return {
                    "observed_samples": int(observed.sum()),
                    "lost_observed_samples": int(np.sum(observed & (retained_counts[part] == 0))),
                    "support_deficit_samples": int(np.sum(retained_counts[part] < required)),
                    "near_best_view_deficit_samples": int(np.sum(observed & (retained_best[part] < original_best[part] * quality_ratio))),
                }
            candidates = [index for row, (index, _) in enumerate(saved_poses) if not kept[row]]
            summary["surface_analysis"] = {
                "executed": True, "method": "area_weighted_saved_mesh_depth_visibility_reverse_delete",
                "sample_count_per_split": samples, "surface_area_m2": float(training[2]),
                "minimum_observations": min_observations, "near_best_incidence_ratio": quality_ratio,
                "protected_default_texture_indices": sorted(texture_indices),
                "candidate_redundant_indices": candidates,
                "proposed_retained_final_pose_count": int(kept.sum()),
                "candidate_native_payload_bytes": sum(rows[i]["native_payload_bytes"] for i in candidates),
                "selection": checks(slice(0, samples)), "held_out_surface": checks(slice(samples, None)),
                "numpy_version": np.__version__, "open3d_version": o3d.__version__,
                "scope": "Saved poses; depth agreement within max(15 mm, 1% of distance), incidence cosine > 0.25. No TSDF or texture rebake.",
            }
            for row, (index, _) in enumerate(saved_poses):
                rows[index].update(default_texture_selected=index in texture_indices,
                                   visible_surface_samples=int(visible[row, :samples].sum()),
                                   unique_surface_samples=int(np.sum(visible[row, :samples] & (original_counts[:samples] == 1))),
                                   proposed_redundant=not bool(kept[row]))
        summary["elapsed_s"] = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "shots.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Complete saved project/session ZIP (read only)")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-output/session-contribution",
                        help="Report directory (default: ignored benchmark-output/session-contribution)")
    parser.add_argument("--samples", type=int, default=50000, help="Area-weighted surface samples per selection/validation split (100..200000)")
    parser.add_argument("--min-observations", type=int, default=3, help="Preserve this many observations where originally available (1..20)")
    parser.add_argument("--quality-ratio", type=float, default=0.9, help="Preserve a view within this fraction of the best angle/distance score (0..1)")
    parser.add_argument("--storage-only", action="store_true", help="Skip numerical surface analysis")
    parser.add_argument("--compact-output", type=Path, help="Optional new project ZIP omitting only generated display image copies; never prunes shots")
    args = parser.parse_args(argv)
    if not 100 <= args.samples <= 200000 or not 1 <= args.min_observations <= 20 or not 0 < args.quality_ratio <= 1:
        parser.error("Invalid sample, observation or quality budget")
    sys.path.insert(0, str(ROOT))
    try:
        summary = analyze(args.session, args.output, args.samples, args.min_observations, args.quality_ratio, args.storage_only)
        if args.compact_output:
            result = compact_display_copies(args.session, args.compact_output)
            (args.output / "compact-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"captures": summary["captures"], "final_pose_captures": summary["final_pose_captures"],
                          "outside_final_pose_captures": len(summary["outside_final_pose_indices"]),
                          "surface_analysis": summary["surface_analysis"], "storage": summary["storage"],
                          "report": str((args.output / "summary.json").resolve())}, indent=2))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Cannot audit project: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
