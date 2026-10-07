"""Read-only joint RGB-D pose proposal evaluation; never runs or commits fusion.

Default: noisy raycast observations with known camera truth used only to create
an explicitly drifted input and score output. Optional session/poses inputs use
archived camera estimates as starting poses; without independent truth their
held-out geometry scores establish consistency only, never absolute accuracy.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import scipy
from PIL import Image

from scanner_server.bundle_adjustment import make_problem, propose_bundle_poses, solve_bundle
from shared.settings import ScanSettings


def synthetic():
    from tests.test_quality import scene_frames
    frames = scene_frames(6)
    poses = []
    for index, (_, _, truth) in enumerate(frames):
        estimate = truth.copy()
        estimate[0, 3] += index * 0.01
        estimate[2, 3] += index * 0.003
        poses.append((index, estimate))
    return SimpleNamespace(settings=ScanSettings(), poses=poses,
                           raw_frames=[(rgb, depth) for rgb, depth, _ in frames],
                           frame_metadata=[{} for _ in frames]), [truth for _, _, truth in frames]


def numerical():
    from tests.test_bundle_adjustment import bundle_fixture
    camera, world, truth, initial, features, tracks = bundle_fixture()
    problem = make_problem(initial, features, tracks)
    refined, landmarks, solver = solve_bundle(problem, camera)
    result = {"cameras": len(initial), "landmarks": len(world), "observations": len(problem.depths),
              "solver": solver}
    for label, poses, points in (("before", initial, problem.landmarks),
                                ("after", refined, landmarks)):
        translations = [np.linalg.norm(pose[:3, 3] - actual[:3, 3])
                        for pose, actual in zip(poses, truth)]
        result[f"translation_rmse_{label}_m"] = float(np.sqrt(np.mean(np.square(translations))))
        result[f"landmark_rmse_{label}_m"] = float(np.sqrt(np.mean(np.sum((points - world) ** 2, axis=1))))
    return result


def recording(archive, poses_path):
    manifest = json.loads(archive.read("manifest.json"))
    reconstruction = json.loads(poses_path.read_text())

    class Frames:
        def __getitem__(self, index):
            entry = manifest["frames"][index]
            return (np.array(Image.open(io.BytesIO(archive.read(entry["rgb"])))),
                    np.array(Image.open(io.BytesIO(archive.read(entry["depth"])))))

    return SimpleNamespace(settings=ScanSettings.from_dict(manifest["settings"]),
                           poses=[(entry["index"], np.array(entry["camera_to_world"]))
                                  for entry in reconstruction["poses"]], raw_frames=Frames(),
                           frame_metadata=[entry.get("metadata", {}) for entry in manifest["frames"]])


def score(engine, truth):
    proposals, report = propose_bundle_poses(engine)
    result = {"proposal_accepted": proposals is not None, "proposal": report,
              "accepted_view_count": len(engine.poses)}
    if truth is not None:
        for name, poses in (("before", engine.poses), ("after", proposals)):
            if poses is not None:
                errors = [float(np.linalg.norm(pose[:3, 3] - truth[index][:3, 3]))
                          for index, pose in poses]
                result[f"translation_rmse_{name}_m"] = float(np.sqrt(np.mean(np.square(errors))))
                result[f"translation_max_{name}_m"] = max(errors)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="Optional recorded session zip")
    parser.add_argument("--poses", type=Path, help="Connected estimated reconstruction.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.session) != bool(args.poses):
        parser.error("--session and --poses must be provided together")
    cv2.setNumThreads(4)
    engine, truth = synthetic()
    report = {"schema_version": 1,
              "source_sha256": hashlib.sha256((ROOT / "scanner_server/bundle_adjustment.py").read_bytes()).hexdigest(),
              "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__},
              "notes": ["Proposal optimization and independent depth validation only; no TSDF fusion or mesh export",
                        "Synthetic known poses are explicitly perturbed to simulate drift; ground truth only scores output",
                        "Recorded input uses pre-existing estimated poses, without independent pose or surface truth",
                        "Optional recorded result demonstrates acceptance/refusal and self-consistency, not absolute accuracy"],
              "synthetic_noisy_camera_landmark_solve": numerical(),
              "synthetic_sift_raycast": score(engine, truth)}
    if args.session:
        with zipfile.ZipFile(args.session) as archive:
            report["recorded_connected_estimates"] = score(recording(archive, args.poses), None)
        report["recorded_source"] = {"session": args.session.name, "poses": str(args.poses),
                                     "poses_sha256": hashlib.sha256(args.poses.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
