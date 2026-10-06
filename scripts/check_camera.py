"""Check connected Kinect frames and bounded shutdown without saving images.

Run with the scanner closed. Use --window --offscreen to exercise Qt preview
and window closure as well as acquisition.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--rgb-mode", choices=("rgb_high_res", "rgb_low_res"), default="rgb_high_res")
    parser.add_argument("--exposure", choices=("auto", "manual"), default="auto")
    parser.add_argument("--shutter-speed", type=int, default=125,
                        help="Reciprocal seconds: 250 selects 1/250 s in manual mode")
    parser.add_argument("--gain", type=int, choices=(1, 2, 4, 8), default=1)
    args = parser.parse_args()
    if args.window and (args.exposure != "auto" or args.shutter_speed != 125
                        or args.rgb_mode != "rgb_high_res" or args.gain != 1):
        parser.error("Exposure/resolution overrides require running without --window")
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtCore import QCoreApplication, QTimer

    from kinect_scanner.worker import KinectWorker
    from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS

    window = None
    if args.window:
        from PyQt6.QtWidgets import QApplication

        from kinect_scanner.gui.main_window import MainWindow

        app = QApplication([])
        window = MainWindow()
        window.show()
        worker = window.worker
    else:
        app = QCoreApplication([])
        worker = KinectWorker(rgb_mode=args.rgb_mode, rgb_exposure_mode=args.exposure,
                              rgb_shutter_speed=args.shutter_speed, rgb_gain=args.gain)
    frames, deltas, errors, exposures, brightness, clipped = [], [], [], [], [], []
    started = time.monotonic()
    close_started = None
    capture_timer = QTimer()
    capture_timer.setSingleShot(True)

    def received(rgb, depth, metadata):
        if not frames:
            capture_timer.start(int(args.seconds * 1000))
        frames.append((int((depth > 0).sum()), rgb.shape, depth.shape))
        deltas.append(metadata["rgb_depth_delta_ms"])
        brightness.append(float(rgb.mean()))
        clipped.append(float((rgb >= 250).mean()))
        exposures.append({key: metadata.get(key) for key in
                          ("rgb_exposure_mode", "rgb_shutter_speed", "rgb_exposure_us", "rgb_gain", "rgb_exposure_controls")})

    worker.frame_pair_ready.connect(received)
    worker.error_occurred.connect(errors.append)

    def close():
        nonlocal close_started
        close_started = time.monotonic()
        if window is not None:
            window.close()
        else:
            worker.stop()
            if not worker.wait(2500):
                errors.append("Camera worker failed to stop within 2.5 seconds")
            app.quit()

    def fail_close():
        errors.append("Window failed to close within four seconds")
        worker.stop()
        worker.wait(2500)
        app.quit()

    if window is None:
        worker.start()
    capture_timer.timeout.connect(close)
    # Allow a startup timeout and a retry before judging missing hardware.
    QTimer.singleShot(int((args.seconds + 20) * 1000), close)
    QTimer.singleShot(int((args.seconds + 24) * 1000), fail_close)
    app.exec()
    elapsed = time.monotonic() - started
    shutdown = time.monotonic() - close_started if close_started is not None else None
    report = {
        "frames": len(frames),
        "capture_seconds": round(elapsed - (shutdown or 0), 2),
        "minimum_valid_depth_pixels": min((f[0] for f in frames), default=0),
        "max_rgb_depth_delta_ms": round(max(map(abs, deltas), default=0), 3),
        "median_abs_rgb_depth_delta_ms": round(statistics.median(map(abs, deltas)), 3) if deltas else None,
        "color_assistance_eligible_frames": sum(abs(d) <= RGB_DEPTH_ASSISTANCE_LIMIT_MS for d in deltas),
        "shutdown_seconds": round(shutdown, 3) if shutdown is not None else None,
        "camera_label": window.kinect_label.text() if window else None,
        "preview_rendered": (
            window.view_label.pixmap() is not None
            and not window.view_label.pixmap().isNull()
        )
        if window
        else None,
        "errors": errors,
        "exposure": exposures[-1] if exposures else None,
        "mean_rgb": round(sum(brightness) / len(brightness), 2) if brightness else None,
        "clipped_channel_fraction": round(sum(clipped) / len(clipped), 4) if clipped else None,
    }
    print(json.dumps(report, indent=2))
    success = (
        frames
        and not errors
        and shutdown is not None
        and shutdown < 4
        and (window is None or report["preview_rendered"])
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
