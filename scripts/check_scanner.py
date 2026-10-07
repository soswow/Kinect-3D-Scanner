"""Check the local HTTP/WebSocket reconstruction pipeline without hardware."""

import argparse
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import trimesh

# Exercise Qt widgets without a display or screen-recording permissions.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import httpx
import numpy as np
import open3d as o3d
import websocket

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.protocol import pack_frame, pack_frames


def check_client(port, rgb, depth):
    from PyQt6.QtCore import QSettings, QThread, pyqtSignal
    from PyQt6.QtWidgets import QApplication

    from kinect_scanner.gui import main_window
    from kinect_scanner.gui.preferences import ScannerPreferences

    class NoCameraWorker(QThread):
        frame_ready = pyqtSignal(np.ndarray, np.ndarray)
        error_occurred = pyqtSignal(str)

        def __init__(self, **kwargs):
            super().__init__()

        def run(self):
            pass

        def stop(self):
            pass

    # Use explicit synthetic input even if a Kinect becomes connected.
    main_window.KinectWorker = NoCameraWorker
    os.environ.update(
        KINECT_SERVER_HOST="127.0.0.1",
        KINECT_SERVER_PORT=str(port),
        KINECT_AUTOCONNECT="1",
    )
    # Preview launch is checked separately; avoid opening a window in this test.
    main_window.launch_viewer_subprocess = lambda path: None
    capture_cues = []
    main_window.CaptureSound.play = lambda self: capture_cues.append(True)  # Keep the check silent.
    app = QApplication([])
    settings_dir = tempfile.TemporaryDirectory()
    store = QSettings(str(Path(settings_dir.name) / "scanner.ini"), QSettings.Format.IniFormat)
    window = main_window.MainWindow(preferences=ScannerPreferences(store))
    window.rgb_mode_combo.setCurrentIndex(1)  # Native calibrated VGA fixture.

    def wait_until(predicate):
        deadline = time.monotonic() + 30
        while not predicate():
            app.processEvents()
            assert time.monotonic() < deadline, window.scan_status_label.text()
            time.sleep(0.01)
        app.processEvents()

    try:
        wait_until(lambda: window.server_client.is_connected and not window._connect_pending
                   and not window._restore_on_status)
        window.capture_mode_combo.setCurrentIndex(window.capture_mode_combo.findData("manual"))
        window._on_frame(rgb, depth)
        window.depth_near_spin.setValue(750)
        window.depth_far_spin.setValue(1500)
        window.final_voxel_spin.setValue(4)
        window.relocalize_cb.setChecked(True)
        window._start_scan()
        wait_until(lambda: window._scanning)
        assert not window.settings_group.isEnabled(), "Scan settings must freeze"
        settings = window.server_client.get_status()["settings"]
        assert settings["near_m"] == 0.75 and settings["far_m"] == 1.5, settings
        assert (
            settings["final_voxel_m"] == 0.004 and settings["final_block_count"] == 5000
        ), settings
        assert settings["relocalize"] and not settings["confidence_fusion"], settings
        for _ in range(3):
            window._on_frame(rgb, depth)
            window._capture_frame()
        wait_until(lambda: window._server_stored == 3)
        wait_until(lambda: window.live_view.snapshot.get("frame_count") == 3)
        assert window.live_view.snapshot["processing_interval_s"] > 0
        assert window._capture_pacer.outstanding_count == 0, "Completed captures must release pacing slots"
        assert window._effective_capture_interval() >= window.auto_capture_spin.interval_seconds
        assert capture_cues, "Accepted captures must request sound confirmation"
        assert len(window.live_view.points) > 0 and not window.live_view.isHidden()
        image = window.live_view.grab().toImage()
        assert not image.isNull(), "Persistent live view did not render"
        print("PASS: live fused surface reaches Qt without preview/build", flush=True)
        old_snapshot = window.live_view.snapshot
        window._on_live_updated({"session_id": "previous-session", "frame_count": 900})
        assert window.live_view.snapshot is old_snapshot
        window.live_view.snapshot = {**old_snapshot, "pending_count": 6}
        window._auto_capture_tick()
        assert (
            window.scan_status_label.text()
            == "Capturing automatically · paced by live reconstruction"
        )
        assert window.server_client.get_status()["stored_count"] == 3
        window.live_view.snapshot = old_snapshot
        window._on_frame(rgb, depth, {"captured_monotonic_s": time.monotonic() - 2})
        window._capture_frame()
        assert window.scan_status_label.text() == "Waiting for a fresh camera frame"
        assert window.server_client.get_status()["stored_count"] == 3
        window._preview_scan()
        wait_until(lambda: window._last_preview_path is not None)
        assert window._scanning and window.btn_preview_scan.isEnabled()
        assert not window.btn_start_scan.isEnabled()
        assert not window.btn_export_ply.isEnabled(), (
            "Preview must not enable final mesh export"
        )
        print(
            "PASS: Qt client connects, captures synthetic frames, previews and stays in scan mode",
            flush=True,
        )
        preview_path = window._last_preview_path
        window._session_dirty = False  # Save/discard protection is covered by GUI regression tests.
        window._cancel_scan()
        wait_until(lambda: not window._reset_pending and not window._scanning)
        assert window.server_client.get_status()["stored_count"] == 0
        assert not window.auto_capture_cb.isChecked() and window.settings_group.isEnabled()
        window._on_frame(rgb, depth)
        window._start_scan()
        wait_until(lambda: window._scanning)
        for _ in range(3):
            window._on_frame(rgb, depth)
            window._capture_frame()
        wait_until(lambda: window._server_stored == 3)
        print("PASS: Qt client cancels without a build and starts a fresh scan", flush=True)
        window._stop_and_build()
        wait_until(lambda: window.btn_export_ply.isEnabled())
        assert window.btn_export_obj.isEnabled()
        assert (
            window.btn_export_glb.isEnabled()
            and window.btn_export_texture_obj.isEnabled()
        )
        wait_until(lambda: window._last_preview_path is not None and not window._final_preview_pending)
        assert window._last_preview_path != preview_path, "Final inspection must fetch the final mesh"
        window._on_frame(rgb, depth)
        assert window.btn_start_scan.isEnabled(), "New Scan requires a fresh camera frame"
        final = window.server_client._http.get("/api/scan/diagnostics").json()[
            "final_reconstruction"
        ]
        assert (
            final["applied"] and final["voxel_m"] == 0.004 and final["blocks"] <= 5000
        ), final
        print(
            "PASS: Qt client final rebuild preserves live resolution and enables export controls",
            flush=True,
        )
        Path(preview_path).unlink(missing_ok=True)
    finally:
        window._close_approved = True  # This isolated check contains disposable synthetic captures.
        window.close()
        assert window.worker.wait(5000) and window.task_worker.wait(5000), (
            "Client threads did not stop"
        )
        app.processEvents()
        settings_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-data",
        action="store_true",
        help="Also download/replay the official five-frame Redwood sample",
    )
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = os.environ.copy()
    env.update(
        KINECT_SERVER_HOST="127.0.0.1",
        KINECT_SERVER_PORT=str(port),
        KINECT_BLOCK_COUNT="5000",
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    (ROOT / "logs").mkdir(exist_ok=True)
    ws = None
    with (ROOT / "logs/scanner-check-server.log").open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "scanner_server"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=120
            ) as http:
                deadline = time.monotonic() + 90
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(
                            "Server exited; see logs/scanner-check-server.log"
                        )
                    try:
                        if http.get("/api/health", timeout=1).json()["status"] == "ok":
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    assert time.monotonic() < deadline, "Server startup timed out"
                    time.sleep(0.25)
                ws = websocket.create_connection(
                    f"ws://127.0.0.1:{port}/ws/progress",
                    timeout=5,
                    http_no_proxy=["127.0.0.1"],
                )
                reset = http.post("/api/scan/reset", json={"rgb_mode": "rgb_low_res"}).json()
                assert reset["success"]
                assert reset["settings"]["sensor_calibration"]["camera_serial"] == "A00363W00948202A"

                # A curved depth surface with a raised central patch constrains ICP.
                y, x = np.mgrid[:480, :640]
                depth = (1100 + 100 * np.sin(x / 90) * np.cos(y / 70)).astype(np.uint16)
                depth[140:340, 220:420] -= 150
                from shared.sensor_calibration import load_calibration

                calibration = load_calibration()
                depth = np.rint(
                    (calibration.scale * 1000 / depth.astype(float) - calibration.b)
                    / calibration.a_per_code
                ).astype(np.uint16)
                rgb = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(
                    np.uint8
                )
                assert http.post(
                    "/api/scan/frame", content=pack_frame(rgb, depth)
                ).json()["success"]
                uploaded = http.post(
                    "/api/scan/frames", content=pack_frames([(rgb, depth)] * 2)
                ).json()
                assert uploaded["stored_count"] == 3, uploaded
                print("PASS: health, reset, single-frame and batch upload", flush=True)

                preview = http.post("/api/scan/preview")
                preview.raise_for_status()
                assert "octet-stream" in preview.headers["content-type"], preview.text
                status = http.get("/api/scan/status").json()
                assert (
                    status["frame_count"] == 3 and status["unprocessed_count"] == 0
                ), status
                print(
                    "PASS: three frames registered/integrated; preview generated",
                    flush=True,
                )
                messages = []
                while True:
                    message = json.loads(ws.recv())
                    messages.append(message)
                    if message["type"] == "done":
                        assert message["success"], message
                        break
                assert any(m["type"] == "progress" for m in messages), messages
                print("PASS: WebSocket progress and completion", flush=True)

                result = http.post("/api/scan/build").json()
                assert result["success"] and result["processed"] == 0, result
                print("PASS: final mesh build reuses integrated frames", flush=True)
                with tempfile.TemporaryDirectory(
                    prefix="kinect-scanner-check-"
                ) as folder:
                    for fmt in ("ply", "obj"):
                        response = http.get(f"/api/scan/export/{fmt}")
                        response.raise_for_status()
                        assert "octet-stream" in response.headers["content-type"]
                        path = Path(folder) / f"scan.{fmt}"
                        path.write_bytes(response.content)
                        mesh = o3d.io.read_triangle_mesh(str(path))
                        assert len(mesh.vertices) and len(mesh.triangles), fmt
                        print(
                            f"PASS: {fmt.upper()} export, {len(mesh.vertices):,} vertices, "
                            f"{len(mesh.triangles):,} triangles",
                            flush=True,
                        )
                    for fmt in ("glb", "obj.zip"):
                        response = http.get(
                            f"/api/scan/export/{fmt}",
                            params={
                                "size": 256,
                                "max_triangles": 10000,
                                "max_views": 3,
                                "exposure_correction": "true",
                                "blend_mode": "best",
                            },
                        )
                        response.raise_for_status()
                        path = Path(folder) / f"scan.{fmt}"
                        path.write_bytes(response.content)
                        if fmt == "obj.zip":
                            with zipfile.ZipFile(path) as archive:
                                report = json.loads(archive.read("texture-report.json"))
                                assert report["blend_mode"] == "best", report
                                assert (
                                    report["exposure_correction"]["reason"]
                                    != "Not requested"
                                ), report
                                archive.extractall(Path(folder) / "textured-obj")
                            path = Path(folder) / "textured-obj/scan.obj"
                        loaded = trimesh.load(path, force="scene")
                        assert all(
                            isinstance(g.visual, trimesh.visual.texture.TextureVisuals)
                            for g in loaded.geometry.values()
                        )
                        print(
                            f"PASS: textured {fmt.upper()} HTTP export preserves materials",
                            flush=True,
                        )
                    response = http.get("/api/scan/export/session")
                    response.raise_for_status()
                    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                        manifest = json.loads(archive.read("manifest.json"))
                        report = json.loads(archive.read("reconstruction.json"))
                        assert (
                            len(manifest["frames"]) == 3 and len(report["poses"]) == 3
                        )
                    print(
                        "PASS: lossless session HTTP export includes poses/settings/diagnostics",
                        flush=True,
                    )
                print(
                    "PASS: complete synthetic scan over loopback HTTP/WebSocket",
                    flush=True,
                )
                http.post("/api/scan/reset", json={"rgb_mode": "rgb_low_res"}).raise_for_status()
                check_client(port, rgb, depth)
                if args.public_data:
                    from replay_scan import load_dataset, replay_server

                    settings, frames = load_dataset("redwood")
                    output = ROOT / "benchmark-output/redwood-http.json"
                    replay_server(f"http://127.0.0.1:{port}", settings, frames, output)
                    report = json.loads(output.read_text())
                    assert report["accepted"] == 5 and report["mesh_built"], report
                    assert report["anchored_translation_rmse_m"] < 0.01, report
                    print(
                        "PASS: public RGB-D sequence reconstructed through HTTP with reference-pose scoring",
                        flush=True,
                    )
        finally:
            if ws is not None:
                ws.close()
            server.terminate()
            try:
                server.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
