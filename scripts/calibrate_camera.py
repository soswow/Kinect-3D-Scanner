"""Calibrate/evaluate saved registered RGB-D captures with measured targets."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2

from shared.device_evidence import calibrate_corners, recording_evidence
from shared.settings import ScanSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("intrinsics")
    fit.add_argument("--columns", type=int, required=True, help="Inner corner columns")
    fit.add_argument("--rows", type=int, required=True, help="Inner corner rows")
    fit.add_argument("--square-mm", type=float, required=True)
    fit.add_argument("--name", default="Measured Kinect RGB")
    evidence = commands.add_parser("evidence")
    evidence.add_argument("--plane-roi", type=int, nargs=4)
    evidence.add_argument("--expected-z-m", type=float)
    args = parser.parse_args()
    manifest = json.loads((args.recording / "manifest.json").read_text())
    settings = ScanSettings.from_dict(manifest["settings"])
    if args.command == "intrinsics":
        corners, rejected = [], []
        frames = manifest["frames"]
        stride = max(1, len(frames) // 120)
        for index in range(0, len(frames), stride):
            frame = frames[index]
            image = cv2.imread(str(args.recording / frame["rgb"]), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (480, 640):
                raise ValueError("Calibration requires original 640x480 RGB images")
            ok, found = cv2.findChessboardCornersSB(
                image, (args.columns, args.rows), cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ok:
                corners.append(found)
            else:
                rejected.append(index)
        camera, report = calibrate_corners(
            corners, args.columns, args.rows, args.square_mm / 1000, args.name
        )
        report["undetected_frame_indices"] = rejected
        report["candidate_camera"] = camera.__dict__
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
        if not report["accepted"]:
            raise SystemExit("Calibration rejected: " + "; ".join(report["reasons"]))
        args.output.write_text(json.dumps(camera.__dict__, indent=2))
    else:
        if args.expected_z_m is not None and args.plane_roi is None:
            parser.error("Known plane depth requires --plane-roi")
        frames, metadata = [], []
        for frame in manifest["frames"]:
            color = cv2.imread(str(args.recording / frame["rgb"]))
            depth = cv2.imread(
                str(args.recording / frame["depth"]), cv2.IMREAD_UNCHANGED
            )
            if color is None or depth is None:
                raise ValueError("Recording image missing")
            frames.append((cv2.cvtColor(color, cv2.COLOR_BGR2RGB), depth))
            metadata.append(frame.get("metadata", {}))
        report = recording_evidence(
            frames, metadata, settings, args.plane_roi, args.expected_z_m
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
