"""Compare observable geometry of two offline field experiment profiles.

No fitting/alignment and no backend authority. A surface comparison does not
replace per-call numerical, graph/gate, source/runtime or acceptance proofs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import compare_quality, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-different-pipeline", action="store_true")
    parser.add_argument("--require-identical-final-poses", action="store_true")
    parser.add_argument("--run-allocated", action="store_true")
    args = parser.parse_args()
    if not args.run_allocated or args.output.exists():
        parser.error("Require exclusive hardware and fresh output")
    paths = [path.resolve(strict=True) for path in (args.reference, args.candidate)]
    pins = {str(path): file_hash(path) for path in paths}
    reference, candidate = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    report = {"kind": "offline-field-observable-surface-comparison-v1", "status": "failed",
        "reference": {"path": str(paths[0]), "sha256": pins[str(paths[0])]},
        "candidate": {"path": str(paths[1]), "sha256": pins[str(paths[1])]},
        "producer_sha256": file_hash(Path(__file__)), "whole_pipeline_equivalence_proven": False,
        "scope": "30,000 deterministic triangle-area surface samples in original coordinates. No alignment or scale fitting. Native reference is not ground truth. Source/input consistency and observable geometry only; no changed GPU, per-call, graph, proposal, runtime or production acceptance authority."}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        for value in (reference, candidate):
            if (value.get("input_changed_during_profile") is not False
                    or value.get("source_changed_during_profile") is not False
                    or value.get("pose_seeds_used") is not False
                    or value.get("mesh_built") is not True):
                raise ValueError("Require completed unseeded meshes with unchanged source/input")
            geometry = Path(value["geometry"]["artifact"])
            pins[str(geometry)] = file_hash(geometry)
        for name in ("input_sha256", "source_sha256", "settings", "selected_indices", "seed", "thread_policy"):
            if reference[name] != candidate[name]:
                raise ValueError("Matched profile scope differs: " + name)
        if not args.allow_different_pipeline and reference["pipeline_options"] != candidate["pipeline_options"]:
            raise ValueError("Pipeline differs; declare a workflow comparison explicitly")
        report["same_pipeline_options"] = reference["pipeline_options"] == candidate["pipeline_options"]
        report["same_live_accepted_indices"] = reference["accepted_indices_before_finish"] == candidate["accepted_indices_before_finish"]
        if args.require_identical_final_poses:
            # New allocation producer explicitly records original unrounded bytes.
            pose_hashes = [value.get("unrounded_final_pose_sha256") for value in (reference, candidate)]
            if not pose_hashes[0] or pose_hashes[0] != pose_hashes[1]:
                raise ValueError("Original unrounded Final poses differ")
            report["identical_unrounded_final_poses"] = True
        import numpy as np
        import open3d as o3d
        from shared.surface_metrics import surface_metrics
        comparison = compare_quality(candidate, paths[0], np)

        def mesh(value):
            with np.load(value["geometry"]["artifact"], allow_pickle=False) as arrays:
                result = o3d.geometry.TriangleMesh()
                result.vertices = o3d.utility.Vector3dVector(arrays["points"])
                result.triangles = o3d.utility.Vector3iVector(arrays["faces"])
            return result

        surface = surface_metrics(mesh(candidate), mesh(reference), threshold_m=.005, samples=30000)
        report.update(comparison=comparison, triangle_surface_metrics=surface,
            observable_geometry_passed=bool(comparison["same_accepted_indices"]
                and comparison["same_mesh_success"] and surface["surface_p95_m"] <= .0005
                and surface["precision"] >= .999 and surface["completeness"] >= .999))
        if any(file_hash(Path(path)) != digest for path, digest in pins.items()):
            raise ValueError("Measured profiles/geometries changed during comparison")
        report.update(status="complete", inputs_sha256=pins)
        save()
        print(json.dumps({"observable_geometry_passed": report["observable_geometry_passed"],
            "same_final_accepted_indices": comparison["same_accepted_indices"],
            "surface_p95_m": surface["surface_p95_m"], "precision": surface["precision"],
            "completeness": surface["completeness"]}))
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise


if __name__ == "__main__":
    main()
