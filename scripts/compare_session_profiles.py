"""Compare matched finished replay poses, vertices and actual triangle surfaces."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d

from scripts.profile_session import compare_quality
from shared.surface_metrics import surface_metrics


def mesh(report):
    with np.load(report["geometry"]["artifact"]) as data:
        result = o3d.geometry.TriangleMesh()
        result.vertices = o3d.utility.Vector3dVector(data["points"])
        result.triangles = o3d.utility.Vector3iVector(data["faces"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-preview-resolution-change", action="store_true",
                        help="Allow only different live voxels with identical effective and actual final detail")
    args = parser.parse_args()
    baseline_bytes = args.baseline.read_bytes()
    candidate_bytes = args.candidate.read_bytes()
    baseline = json.loads(baseline_bytes)
    candidate = json.loads(candidate_bytes)
    if not baseline["mesh_built"] or not candidate["mesh_built"]:
        parser.error("Both profiles must contain successful final meshes")
    result = compare_quality(candidate, args.baseline, np,
                             preview_resolution=args.allow_preview_resolution_change)
    result["triangle_surface_metrics_vs_cpu"] = surface_metrics(
        mesh(candidate), mesh(baseline), threshold_m=0.005, samples=30000)
    result["note"] = "CPU reconstruction is a matched reference, not ground truth; no alignment or scale fitting."
    if (args.baseline.read_bytes() != baseline_bytes
            or args.candidate.read_bytes() != candidate_bytes):
        raise RuntimeError("Input reports changed during comparison; rerun against fixed reports")
    result["baseline_report_sha256"] = hashlib.sha256(baseline_bytes).hexdigest()
    result["candidate_report_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
