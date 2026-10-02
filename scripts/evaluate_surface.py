"""Compare metric meshes already expressed in the same fixed coordinate frame."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import open3d as o3d

from shared.surface_metrics import surface_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--threshold-mm", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = surface_metrics(
        o3d.io.read_triangle_mesh(str(args.mesh)),
        o3d.io.read_triangle_mesh(str(args.reference)),
        threshold_m=args.threshold_mm / 1000,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
