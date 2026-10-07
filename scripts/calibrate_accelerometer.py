"""Fit Kinect accelerometer bias, scale, and camera alignment from static poses.

Input JSON: {"observations": [{"acceleration_m_s2": [x,y,z],
                             "up_camera": [x,y,z]}, ...],
             "validation": [... three or more independent held-out poses ...]}

Use six or more orientations spanning all axes. Camera up must be independently
known from a levelled fixture/reference, not inferred from the same accelerometer.
Only a successful held-out check marks the result verified. Physical capture is
separate; preserve its raw recordings and do not force the Kinect tilt mechanism.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.inertial import fit_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--id", help="Device-specific calibration identity")
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text())
        profile = fit_calibration(data["observations"], data.get("validation"), args.id)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "verified": profile["verified"],
                      "report": profile["evidence"]["report"]}, indent=2))


if __name__ == "__main__":
    main()
