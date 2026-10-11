"""Convert a saved session to a Blender analysis scene, without a running server."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path, help="Saved scanner project/session ZIP")
    parser.add_argument("--output", type=Path, default=Path("benchmark-output/blender-export/scan.blend"), help="Output .blend (writes a sibling RGB sequence folder)")
    parser.add_argument("--blender", help="Blender 4.2+ executable or macOS .app; otherwise auto-detected")
    parser.add_argument("--textured-model", type=Path, help="Optional existing textured OBJ ZIP; otherwise bake saved mesh offline")
    parser.add_argument("--snapshot-stride", type=int, choices=(1, 2, 4, 8), default=2, help="Sample every N depth pixels (default: 2)")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kinect_scanner.blender_export import convert_session

    convert_session(args.session, args.output, blender=args.blender, textured_model=args.textured_model,
                    stride=args.snapshot_stride, progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
