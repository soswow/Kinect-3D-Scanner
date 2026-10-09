"""Hardware-free bundle check: real Qt window, spawned capture and mesh helper."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def synthetic_capture(connection, stop_event, rgb_buffer, depth_buffer):
    # Import the actual extension in the frozen child to check its dylib closure.
    import freenect  # noqa: F401
    import numpy as np

    np.frombuffer(rgb_buffer, np.uint8)[:] = 42
    np.frombuffer(depth_buffer, np.uint16)[:] = 750
    while not stop_event.is_set():
        connection.send(("frame", {"rgb_depth_delta_ms": 0, "timestamp_s": time.time()}))
        connection.recv()
        stop_event.wait(0.1)


def run_check():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.pop("KINECT_AUTOCONNECT", None)
    from PyQt6.QtCore import QSettings, QTimer
    from PyQt6.QtWidgets import QApplication
    import open3d as o3d

    from shared.sensor_calibration import load_calibration
    from .gui import main_window
    from .gui.preferences import ScannerPreferences
    from .runtime import data_root, export_root, viewer_command
    from .worker import KinectWorker

    load_calibration()
    assets = Path(main_window.__file__).with_name("assets")
    for name in ("capture", "tracking_lost", "tracking_reacquired"):
        if (assets / f"{name}.wav").read_bytes()[:4] != b"RIFF":
            raise RuntimeError(f"Missing or invalid sound: {name}")
    app = QApplication([])
    with tempfile.TemporaryDirectory() as temporary:
        settings = QSettings(str(Path(temporary) / "preferences.ini"), QSettings.Format.IniFormat)
        settings.setValue("feedback/capture_sound", False)
        preferences = ScannerPreferences(settings)
        preferences.write("connection/host", "192.0.2.42")
        original = main_window.KinectWorker
        main_window.KinectWorker = lambda **kwargs: KinectWorker(capture_target=synthetic_capture, **kwargs)
        try:
            window = main_window.MainWindow(preferences=preferences)
        finally:
            main_window.KinectWorker = original
        window.show()
        errors = []
        window.worker.error_occurred.connect(errors.append)
        def close_when_ready():
            if window._frame_sequence:
                window.close()
        timer = QTimer()
        timer.timeout.connect(close_when_ready)
        timer.start(100)
        QTimer.singleShot(20_000, window.close)
        app.exec()
        frames = window._frame_sequence
        if not frames or errors or window.worker.isRunning():
            raise RuntimeError(f"Capture/window check failed: frames={frames}, errors={errors}")
        if window.server_ip_edit.text() != "192.0.2.42":
            raise RuntimeError("Server preferences were not restored")
        mesh_path = str(Path(temporary) / "box.ply")
        o3d.io.write_triangle_mesh(mesh_path, o3d.geometry.TriangleMesh.create_box())
        result = subprocess.run(viewer_command(json.dumps({"mode": "check", "filepath": mesh_path})),
                                capture_output=True, text=True, timeout=20)
        if result.returncode != 0 or "mesh helper ok" not in result.stdout:
            raise RuntimeError(f"Mesh helper failed: {result.stdout}\n{result.stderr}")
    report = {"status": "ok", "synthetic_frames": frames, "mesh_helper": "ok",
              "data": str(data_root()), "exports": str(export_root())}
    print(json.dumps(report), flush=True)
    return 0
