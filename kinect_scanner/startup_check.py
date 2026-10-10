"""Hardware-free bundle check: real Qt window, spawned capture and mesh helper."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def synthetic_capture(connection, stop_event, rgb_buffer, depth_buffer):
    # Import the actual extension in the frozen child to check its dylib closure.
    import freenect  # noqa: F401
    import numpy as np

    np.frombuffer(rgb_buffer, np.uint8)[:] = 42
    np.frombuffer(depth_buffer, np.uint16)[:] = 750
    while not stop_event.is_set():
        stamp = time.monotonic()
        connection.send(("frame", {"rgb_depth_delta_ms": 0, "timestamp_s": time.time(),
                                   "orientation_host_monotonic_s": stamp,
                                   "accelerometer": {"valid": False, "reason": "Image-to-host timing uncertain"},
                                   "orientation_accelerometer": {"valid": True, "capture_generation": "synthetic",
                                                                "gravity": {"valid": True, "confidence": 1,
                                                                            "up_camera": [-1, 0, 0]}}}))
        connection.recv()
        stop_event.wait(0.1)


def check_offline_capture(window, temporary):
    """Exercise installed controls, real disk buffering and upload/build ordering."""
    import numpy as np

    from shared.protocol import pack_frames, unpack_frames
    from .capture_spool import CaptureSpool
    from .server_client import ServerClient
    from .server_task_worker import ServerTaskType, ServerTaskWorker

    client = ServerClient()
    worker = ServerTaskWorker(client)
    worker._spool = CaptureSpool(Path(temporary) / "capture-queue", "bundle-offline", {})
    uploaded, cues, checks = [], [], []

    def send(frames):
        # Check the installed new wire format, including exact sensor pixels.
        decoded = unpack_frames(pack_frames(frames, spatial_prediction=True), with_metadata=True)
        for original, restored in zip(frames, decoded):
            checks.append(np.array_equal(original[0], restored[0])
                          and np.array_equal(original[1], restored[1]))
        results = []
        for _, _, metadata in frames:
            uploaded.append(metadata["frame_id"])
            results.append({"success": True, "index": len(uploaded) - 1})
        return {"success": True, "stored_count": len(uploaded), "results": results}

    client.send_frame = lambda *frame: send([frame])
    client.send_frames_batch = send
    client.request_build = lambda *args, **kwargs: {"success": uploaded == list(range(8))}
    worker._save_reconstruction = lambda: None
    original_worker, original_sound = window.task_worker, window.capture_sound.play
    window.task_worker = worker
    window.capture_sound.play = lambda: cues.append(True)
    try:
        window._session_id = "bundle-offline"
        window.live_cb.setChecked(False)
        window.adaptive_capture_cb.setChecked(True)
        window.auto_capture_spin.set_interval_seconds(.5)
        window._paused = False
        for index in range(8):
            window._capture_selector.clear()
            window._last_frame_metadata = {**window._last_frame_metadata, "frame_id": index}
            window._last_frame_time = time.monotonic()
            window._capture_pacer.last_capture_at = window._last_frame_time - .5
            window._auto_capture_tick()
        checks.append(worker.pending_capture_count == 8 and len(cues) == 8)
        checks.append("Captured: 8" in window.frame_count_label.text()
                      and "Pending upload: 8" in window.frame_count_label.text())
        window._stop_and_build()
        checks.append("Uploading remaining captures" in window.scan_status_label.text())
        while not worker._queue.empty() or worker._pending is not None:
            task = worker._pending if worker._pending is not None else worker._queue.get_nowait()
            worker._pending = None
            if task.task_type == ServerTaskType.BUILD_MESH:
                checks.append(uploaded == list(range(8)) and worker.pending_capture_count == 0)
            worker._dispatch(task)
        checks.append(len(cues) == 8)
    finally:
        window.task_worker = original_worker
        window.capture_sound.play = original_sound
        window._build_pending = False
        window._session_dirty = False
        window._session_id = None
        worker._spool.cleanup()
    if len(checks) != 13 or not all(checks):
        raise RuntimeError(f"Offline capture/upload check failed: {checks}")


def run_check():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.pop("KINECT_AUTOCONNECT", None)
    import open3d as o3d
    from PyQt6.QtCore import QSettings, QTimer
    from PyQt6.QtWidgets import QApplication

    from shared.sensor_calibration import load_calibration

    from .gui import main_window
    from .gui.preferences import ScannerPreferences
    from .runtime import data_root, export_root, viewer_command
    from .server_task_worker import ServerTaskType
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
        check_texture_export_dialog(window)
        check_live_view(window)
        errors = []
        window.worker.error_occurred.connect(errors.append)
        log_checks = []
        motion_layout_checks = []
        reset_checks = []
        orientation_checks = []
        phase = "capture"
        reset_frame = 0
        camera_fault = "Synthetic check: camera disconnected"
        window.logs_panel.clear_logs()
        window._on_error(camera_fault)
        window.logs_panel.append("Synthetic check: connection unchanged", "Connection")
        window._on_error(camera_fault)
        window._switch_mode("logs")
        log_checks.append(window.view_stack.currentWidget() is window.logs_panel)
        log_checks.append(window.logs_panel.text.toPlainText().count(camera_fault) == 1)
        def close_when_ready():
            nonlocal phase, reset_frame
            if window._frame_sequence:
                if phase == "capture":
                    # Verify installed Auto rotation despite rejected tracking
                    # association, allowing the normal 0.3 s switch dwell.
                    if window._display_rotation != 90:
                        return
                    window._switch_mode(main_window.MODE_RGB)
                    image = window.view_label._image
                    orientation_checks.append(image is not None and image.width() < image.height())
                    orientation_checks.append(window.sensor_status_label.isVisible())
                    orientation_checks.append(not window._last_frame_metadata["accelerometer"]["valid"])
                    orientation_checks.append(window._last_depth.shape == (480, 640))
                    # The real spawned frame must re-arm this fault, while a
                    # second unchanged error must stay quiet in the Qt view.
                    window._on_error(camera_fault)
                    window._on_error(camera_fault)
                    window.logs_panel.flush()
                    history = window.logs_panel.text.toPlainText()
                    log_checks.append(history.count(camera_fault) == 2)
                    log_checks.append("Live color and depth frames received" in history)
                    log_checks.append("Resources {" not in history)
                    # Verify the installed Scan layout while offline camera
                    # tracking alternates between verified and unverified.
                    window.worker.blockSignals(True)
                    window.server_client._connected = True
                    window._session_settings = {"live_reconstruction": False, "color_recovery": True}
                    window._scanning = True
                    window._switch_mode(main_window.MODE_SCANNER)
                    window._last_frame_metadata["visual_tracking"] = {"valid": True}
                    window._refresh_status()
                    app.processEvents()
                    views = (window.view_stack, window.view_label, window.scan_depth_view)
                    geometry = [view.geometry() for view in views]
                    state = window.scan_status_label.text()
                    for valid in (False, False, True, False, True):
                        window._last_frame_metadata["visual_tracking"] = {"valid": valid}
                        window._refresh_status()
                        app.processEvents()
                        motion_layout_checks.append(
                            window.guidance_label.isHidden()
                            and geometry == [view.geometry() for view in views]
                            and state == window.scan_status_label.text()
                        )
                    motion_layout_checks.append(
                        window.logs_panel.text.toPlainText().count("Camera motion could not be verified") == 2
                    )
                    check_offline_capture(window, temporary)
                    check_reconstruction_progress(window)
                    window._scanning = False
                    window._session_settings = None
                    window.server_client._connected = False
                    window.worker.blockSignals(False)
                    window._stop_camera_after_finish()
                    phase = "stopped"
                elif phase == "stopped" and not window.worker.isRunning():
                    # Exercise the installed Reset button without touching a
                    # real server or camera, then observe fresh setup frames.
                    window.server_client._connected = True
                    window._session_id = "check-finished"
                    window._server_stored = 1
                    window._has_mesh = True
                    window._refresh_controls()
                    reset_checks.append(window.btn_reset_scan.isEnabled())
                    tasks = []
                    submit = window.task_worker.submit
                    window.task_worker.submit = lambda task: tasks.append(task) or True
                    try:
                        window.btn_reset_scan.click()
                    finally:
                        window.task_worker.submit = submit
                    reset_checks.append(len(tasks) == 1 and tasks[0].task_type == ServerTaskType.RESET)
                    reset_frame = window._frame_sequence
                    main_window.KinectWorker = lambda **kwargs: KinectWorker(capture_target=synthetic_capture, **kwargs)
                    try:
                        window._on_reset_done({"session_id": "check-empty", "settings": {}})
                    finally:
                        main_window.KinectWorker = original
                    window.worker.error_occurred.connect(errors.append)
                    phase = "setup"
                elif phase == "setup" and window._frame_sequence > reset_frame:
                    reset_checks.append(not window._scanning and not window.auto_capture_cb.isChecked()
                                        and not window._start_when_camera_ready)
                    reset_checks.append(window._session_id is None and not window._has_mesh
                                        and window._server_stored == window._server_integrated == 0)
                    reset_checks.append(window._mode == main_window.MODE_RGB and window.btn_start_scan.isEnabled())
                    window.server_client._connected = False
                    window._stop_camera_after_finish()
                    phase = "done"
                elif phase == "done" and not window.worker.isRunning():
                    window.close()
        timer = QTimer()
        timer.timeout.connect(close_when_ready)
        timer.start(100)
        QTimer.singleShot(20_000, window.close)
        app.exec()
        frames = window._frame_sequence
        if not frames or errors or window.worker.isRunning() or window._last_rgb is not None:
            raise RuntimeError(f"Capture/window check failed: frames={frames}, errors={errors}")
        if len(log_checks) != 5 or not all(log_checks):
            raise RuntimeError(f"Log changes check failed: {log_checks}")
        if len(motion_layout_checks) != 6 or not all(motion_layout_checks):
            raise RuntimeError(f"Camera motion layout check failed: {motion_layout_checks}")
        if len(reset_checks) != 5 or not all(reset_checks):
            raise RuntimeError(f"Reset setup check failed: {reset_checks}")
        if len(orientation_checks) != 4 or not all(orientation_checks):
            raise RuntimeError(f"Auto portrait check failed: {orientation_checks}")
        if window.server_ip_edit.text() != "192.0.2.42":
            raise RuntimeError("Server preferences were not restored")
        mesh_path = str(Path(temporary) / "box.ply")
        o3d.io.write_triangle_mesh(mesh_path, o3d.geometry.TriangleMesh.create_box())
        result = subprocess.run(viewer_command(json.dumps({"mode": "check", "filepath": mesh_path})),
                                capture_output=True, text=True, timeout=20, check=False)
        if result.returncode != 0 or "mesh helper ok" not in result.stdout:
            raise RuntimeError(f"Mesh helper failed: {result.stdout}\n{result.stderr}")
    report = {"status": "ok", "synthetic_frames": frames, "camera_shutdown": "ok", "mesh_helper": "ok", "log_changes": "ok", "scan_reset": "ok", "camera_motion_layout": "ok", "auto_portrait": "ok",
              "offline_capture": "ok", "reconstruction_progress": "ok", "texture_export_dialog": "ok", "live_view": "ok",
              "data": str(data_root()), "exports": str(export_root())}
    print(json.dumps(report), flush=True)
    return 0


def check_live_view(window):
    """Verify the installed Follow projection, control row and captured colors."""
    import numpy as np
    from PyQt6.QtWidgets import QApplication, QPushButton

    view = window.live_view
    pose = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1.]])
    points = np.array([[0, 0, 1], [0.3, 0, 1]]) @ pose[:3, :3].T + pose[:3, 3]
    try:
        view.show()
        view.set_snapshot({"points": points, "colors": [[0, 1, 0], [1, 0, 0]],
                           "camera_to_world": pose, "result": {"success": True}})
        QApplication.processEvents()
        controls = view.panel.findChildren(QPushButton)
        if ([button.text() for button in controls] != ["Follow", "Orbit", "Fit View"]
                or not all(button.isVisible() for button in controls)
                or len({button.y() for button in controls}) != 1):
            raise RuntimeError("Live view must have one navigation row")
        xy, depth, indices = view._project_points(640, 480)
        np.testing.assert_array_equal(xy, [[320, 240], [424, 240]])
        np.testing.assert_array_equal(depth, [1.5, 1.5])
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_array_equal(view.camera_to_world, pose)
        viewport = view.drawing_rect
        xy, _, _ = view._project_points(viewport.width(), viewport.height())
        x, y = xy[0] + [viewport.x(), viewport.y()]
        if view.grab().toImage().pixelColor(int(x), int(y)).getRgb()[:3] != (0, 255, 0):
            raise RuntimeError("Live view must display captured colors")
        view.orbit_button.click()
        if view.follow_cb.isChecked():
            raise RuntimeError("Orbit did not release Follow")
        view.follow_button.click()
        if not view.follow_cb.isChecked():
            raise RuntimeError("Follow did not resume")
    finally:
        view.reset()
        view.hide()


def check_texture_export_dialog(window):
    """Exercise installed texture choices without a server or preference writes."""
    from .gui.dialogs import ExportDialog

    dialog = ExportDialog(window)
    dialog.show()
    try:
        if "connected surface region" not in dialog.texture_description.text():
            raise RuntimeError("Texture export explanation is missing")
        for correction in (False, True):
            for sharp in (False, True):
                dialog.texture_exposure_cb.setChecked(correction)
                dialog.texture_best_cb.setChecked(sharp)
                expected = {"exposure_correction": correction, "blend_mode": "best" if sharp else "blend"}
                if dialog.texture_options != expected:
                    raise RuntimeError("Texture export choices are not independent")
        dialog.format_combo.setCurrentIndex(dialog.format_combo.findData("ply"))
        if dialog.texture_best_cb.isEnabled() or not dialog.texture_description.isHidden():
            raise RuntimeError("Texture choices shown for an untextured format")
    finally:
        dialog.reject()


def check_reconstruction_progress(window):
    """Verify phase changes using the installed WebSocket-to-widget path."""
    window._build_pending = True
    checks = []
    try:
        for number in (47, 48):
            window.server_client._handle_ws_message({
                "type": "progress", "current": 0, "total": 130,
                "result": {"stage": "fragment_reconnection",
                           "message": f"Depth registration: revisit 117 <-> 34; candidate {number}/307"},
            })
            checks.append(window.progress_bar.maximum() == 307
                          and window.progress_bar.value() == number
                          and window.progress_bar.text() == f"Revisit candidates: {number} / 307"
                          and window.progress_status_label.isVisible())
        window._on_process_progress(0, 130, {"stage": "fragment_reconnection",
                                             "message": "Optimizing and checking all measured depth constraints"})
        checks.append(window.progress_bar.maximum() == 0
                      and window.progress_status_label.isVisible()
                      and window.progress_status_label.text().startswith("Optimizing"))
        window._on_process_progress(1, 28, {"stage": "final_reintegration", "message": "Planning verified fusion"})
        checks.append(window.progress_bar.maximum() == 28 and window.progress_bar.value() == 1)
    finally:
        window._build_pending = False
        window._set_progress_visible(False)
        window._refresh_controls()
    checks.append(window.progress_bar.isHidden() and window.progress_status_label.isHidden())
    if not all(checks):
        raise RuntimeError(f"Reconstruction progress check failed: {checks}")
