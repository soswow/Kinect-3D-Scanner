"""Record guided Kinect poses, or fit previously measured stationary poses.

Guided mode (close the scanner first):
    python scripts/calibrate_accelerometer.py --interactive --output measured-accelerometer.json

File mode:
    python scripts/calibrate_accelerometer.py stationary-poses.json measured-accelerometer.json

Input JSON: {"observations": [{"acceleration_m_s2": [x,y,z],
                             "up_camera": [x,y,z]}, ...],
             "validation": [... three or more independent held-out poses ...]}

Use six or more orientations spanning all axes. Camera up must be independently
known from a levelled fixture/reference, not inferred from the same accelerometer.
Verification requires independently checked references and successful held-out
checks. In file mode, collect measurements separately. Do not force the tilt joint.
"""

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.inertial import fit_calibration


def fitted_profile(data, identity=None):
    profile = fit_calibration(data["observations"], data.get("validation"), identity or data.get("calibration_id"))
    # Residuals alone cannot establish an independently measured reference.
    if "reference_checked" in data:
        checked = data["reference_checked"] is True
        profile["verified"] = profile["verified"] and checked
        profile["evidence"]["reference_checked"] = checked
        profile["evidence"]["reference"] = data.get("reference", "Operator supplied camera-up directions")
    if "device" in data:
        profile["evidence"]["device"] = data["device"]
    return profile


def checkpoint(path, data):
    """Replace only this run's measurement file, keeping valid JSON on abort."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ask(prompt, input_fn):
    answer = input_fn(prompt).strip()
    if answer.lower() in ("q", "quit"):
        raise KeyboardInterrupt
    return answer


def guided_calibration(output, *, identity=None, device_index=0, seconds=3.0,
                       capture_factory=None, input_fn=input, print_fn=print):
    from shared.accelerometer_calibration import (CalibrationCapture, POSES, VALIDATION_POSES,
                                                  stationary_observation)

    output = Path(output).expanduser().resolve()
    measurements = output.with_name(output.stem + ".measurements.json")
    if output.exists() or measurements.exists():
        raise ValueError("Output or measurement file already exists. Choose a new --output filename; existing files are preserved.")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "preparing", "observations": [], "validation": [], "attempts": []}
    # Reserve the companion filename before prompting or touching hardware.
    with measurements.open("x") as stream:
        json.dump(data, stream, indent=2)
    print_fn("\nKinect accelerometer calibration\n")
    print_fn("Close the scanner and other Kinect apps. Connect USB and external power.")
    print_fn("You will position the WHOLE Kinect in six directions, then take three fresh checks.")
    print_fn("Support it securely; keep hands off while recording. Do not force the motor joint.")
    print_fn("Keep the head/base relationship fixed throughout, and use that same tilt when scanning.")
    print_fn("Directions describe the native camera image, before any automatic preview rotation.")
    print_fn("Use a level/square or a measured fixture to align the actual camera axes within 2 degrees.")
    print_fn("A level base alone does not prove the lenses are horizontal if the head is tilted.")
    print_fn("The sensor can check steadiness, but cannot independently check your physical alignment.")
    print_fn("Press Enter when each position is ready. Type q or press Ctrl-C to stop.")
    print_fn(f"Measurements are saved to: {measurements}")
    try:
        checked = ask("Will you independently check camera alignment for EVERY pose with a level/fixture? [y/N]: ", input_fn).lower()
        while checked not in ("", "n", "no", "y", "yes"):
            checked = ask("Please enter y or n: ", input_fn).lower()
        data["reference_checked"] = checked in ("y", "yes")
        data["reference"] = ("Operator confirmed independently levelled camera axes for each prescribed pose"
                             if data["reference_checked"] else "Approximate prescribed camera directions; independent alignment not confirmed")
        if not data["reference_checked"]:
            print_fn("The profile will remain UNVERIFIED. Residual checks still apply; approximate placement may fail them.")
        if identity is None:
            label = ask("Calibration name (Enter for an automatic name): ", input_fn)
            identity = label or "kinect-fixed-tilt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not 1 <= len(identity) <= 128:
            raise ValueError("Calibration name must contain 1 to 128 characters")
        data["calibration_id"] = identity
        checkpoint(measurements, data)
        factory = capture_factory or CalibrationCapture
        print_fn("Opening Kinect accelerometer…")
        with factory(device_index) as capture:
            data["device"] = capture.metadata
            data["status"] = "collecting"
            checkpoint(measurements, data)
            plan = [("observations", pose) for pose in POSES] + [("validation", pose) for pose in VALIDATION_POSES]

            def record(index):
                group, (name, instruction, up) = plan[index]
                print_fn(f"\nPosition {index + 1}/{len(plan)} — {'fresh validation' if group == 'validation' else 'calibration'}: {name.replace('_', ' ')}")
                print_fn(instruction)
                if group == "validation":
                    print_fn("Move away from this position first, then realign it independently. These are new readings.")
                while True:
                    ask("Align and support the Kinect; press Enter to record (q to quit): ", input_fn)
                    attempt = {"group": group, "pose": name, "up_camera": list(up), "samples": [], "accepted": False}
                    data["attempts"].append(attempt)
                    print_fn(f"Settling for 1 second, then recording for {seconds:g} seconds. Keep still…")
                    try:
                        capture.collect(seconds=seconds, settle=1.0, on_sample=attempt["samples"].append)
                        observation = stationary_observation(attempt["samples"], up, seconds)
                        observation["pose"] = name
                        attempt["accepted"] = True
                        # Replacing a failed-fit pose does not reuse validation data.
                        rows = data[group]
                        rows[:] = [row for row in rows if row.get("pose") != name]
                        rows.append(observation)
                        summary = observation["capture_summary"]
                        print_fn(f"Accepted {summary['sample_count']} readings; scatter {summary['scatter_rms_m_s2']:.3f} m/s².")
                        return
                    except ValueError as exc:
                        attempt["error"] = str(exc)
                        print_fn(f"Please retry this position: {exc}")
                    except BaseException as exc:
                        attempt["error"] = str(exc) or type(exc).__name__
                        raise
                    finally:
                        checkpoint(measurements, data)

            for index in range(len(plan)):
                record(index)
            while True:
                try:
                    profile = fitted_profile(data, identity)
                    break
                except ValueError as exc:
                    data["fit_error"] = str(exc)
                    checkpoint(measurements, data)
                    print_fn(f"\nCalibration did not pass: {exc}")
                    print_fn("Check camera alignment and keep the head/base relationship fixed. Measurements are retained.")
                    choice = ask("Re-record a position [1-9], all positions [a], or quit [q]: ", input_fn).lower()
                    if choice == "a":
                        for index in range(len(plan)):
                            record(index)
                    elif choice.isdigit() and 1 <= int(choice) <= len(plan):
                        record(int(choice) - 1)
                    else:
                        print_fn("Enter a position number, a, or q.")
            # Never overwrite an existing profile, including one created while prompting.
            with output.open("x") as stream:
                json.dump(profile, stream, indent=2, allow_nan=False)
                stream.write("\n")
            data.update(status="complete", report=profile["evidence"]["report"], verified=profile["verified"])
            data.pop("fit_error", None)
            checkpoint(measurements, data)
        print_fn(f"\nCalibration saved: {output}")
        print_fn(f"Verification: {'VERIFIED' if profile['verified'] else 'UNVERIFIED'}")
        for name, report in profile["evidence"]["report"].items():
            if isinstance(report, dict):
                print_fn(f"{name.capitalize()}: maximum error {report['max_m_s2']:.3f} m/s², {report['max_angle_deg']:.2f} degrees")
        print_fn("Open the scanner → experimental scan settings → Load Accelerometer Calibration…")
        print_fn(f"Select {output.name}, before starting a scan. Keep the same device and head tilt.")
        return profile
    except BaseException as exc:
        data.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, EOFError)) else "failed",
                    error=str(exc) or type(exc).__name__)
        checkpoint(measurements, data)
        print_fn(f"\nStopped. Recorded measurements are retained at: {measurements}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--interactive", action="store_true", help="Guide physical positioning and record directly from the Kinect")
    parser.add_argument("--output", dest="guided_output", type=Path, help="Guided profile filename (default: accelerometer-calibration.json)")
    parser.add_argument("--device-index", type=int, default=0, help="Kinect device index for guided capture")
    parser.add_argument("--seconds", type=float, default=3.0, help="Stationary recording seconds per pose, 2–30 (default: 3)")
    parser.add_argument("--id", help="Device-specific calibration identity")
    args = parser.parse_args(argv)
    if args.interactive:
        if args.input is not None or args.output is not None:
            parser.error("Guided mode uses --output, without positional input/output files")
        if args.device_index < 0 or not math.isfinite(args.seconds) or not 2 <= args.seconds <= 30:
            parser.error("Use a non-negative --device-index and --seconds between 2 and 30")
        try:
            guided_calibration(args.guided_output or Path("accelerometer-calibration.json"),
                               identity=args.id, device_index=args.device_index, seconds=args.seconds)
        except (KeyboardInterrupt, EOFError):
            return 130
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            print(f"Calibration stopped: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.input is None or args.output is None or args.guided_output is not None:
        parser.error("Supply input.json output.json, or use --interactive --output output.json")
    try:
        data = json.loads(args.input.read_text())
        profile = fitted_profile(data, args.id or data.get("calibration_id"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "verified": profile["verified"],
                      "report": profile["evidence"]["report"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
