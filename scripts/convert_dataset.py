"""Convert downloaded public datasets into the scanner's lossless recording format."""

import argparse
from pathlib import Path

from replay_scan import load_dataset

from shared.recording import RecordingWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["redwood", "tum"], required=True)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("Stride must be positive")
    settings, frames = load_dataset(args.dataset, args.path, args.stride, args.limit)
    writer = RecordingWriter(args.output, settings.to_dict())
    writer.manifest["source_dataset"] = args.dataset
    writer.manifest["reference_pose_convention"] = "camera_to_world; evaluation only"
    for rgb, depth, timestamp, pose in frames:
        writer.append(rgb, depth, {"timestamp_s": timestamp})
        if pose is not None:
            writer.manifest["frames"][-1]["reference_pose"] = pose.tolist()
            writer._save_manifest()
    print(f"Converted {len(frames)} frames into {args.output}")


if __name__ == "__main__":
    main()
