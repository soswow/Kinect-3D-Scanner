"""MainWindow — primary application window (client/server mode)."""

import json
import os
import time
from datetime import datetime

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from shared.settings import CameraCalibration, ScanSettings

from ..config import MODE_DEPTH, MODE_RGB, MODE_SCANNER
from ..server_client import ServerClient
from ..server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker
from ..viewer import launch_viewer_subprocess
from ..worker import KinectWorker
from .live_view import LiveView
from .widgets import colorize_depth, numpy_to_qimage

# Default export directory (relative to where the app is launched)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
EXPORT_DIR = os.path.join(_PROJECT_ROOT, "export")
MESH_DIR = os.path.join(_PROJECT_ROOT, "mesh")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect 3D Scanner")
        self.setMinimumSize(960, 600)

        self._closing = False
        self._mode = MODE_RGB
        self._scanning = False
        self._fps_counter = 0
        self._fps_value = 0.0
        self._last_fps_time = time.time()
        self._last_rgb = None
        self._last_depth = None
        self._camera = CameraCalibration()
        self._frame_sequence = 0
        self._last_frame_metadata = {}
        self._last_frame_time = 0
        self._last_capture_id = None
        self._reset_pending = False
        self._preview_pending = False
        self._last_preview_path: str | None = None

        # Server counts (cached from server responses)
        self._server_stored = 0
        self._server_integrated = 0
        self._session_id = None
        self._has_mesh = False

        # Server client + task worker
        self.server_client = ServerClient(self)
        self.server_client.reset_done.connect(self._on_reset_done)
        self.server_client.frame_stored.connect(self._on_frame_stored)
        self.server_client.process_progress.connect(self._on_process_progress)
        self.server_client.build_mesh_done.connect(self._on_build_mesh_done)
        self.server_client.preview_done.connect(self._on_preview_done)
        self.server_client.export_done.connect(self._on_export_done)
        self.server_client.save_mesh_done.connect(self._on_save_mesh_done)
        self.server_client.task_started.connect(self._on_task_started)
        self.server_client.task_error.connect(self._on_task_error)
        self.server_client.live_updated.connect(self._on_live_updated)
        self.server_client.status_updated.connect(self._on_server_status)
        self.server_client.connected.connect(self._on_server_connected)
        self.server_client.disconnected.connect(self._on_server_disconnected)

        self.task_worker = ServerTaskWorker(self.server_client)
        self.task_worker.start()

        self._build_ui()
        self._build_toolbar()
        self._build_dock()
        self._build_statusbar()

        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

        self.worker = KinectWorker()
        if hasattr(self.worker, "frame_pair_ready"):
            self.worker.frame_pair_ready.connect(self._on_frame)
        else:
            self.worker.frame_ready.connect(self._on_frame)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

        # Disable scan controls until server connected
        self._set_scan_controls_enabled(False)
        if os.environ.get("KINECT_AUTOCONNECT") == "1":
            QTimer.singleShot(0, self._toggle_connection)

    # ── UI construction ───────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.view_label = QLabel("Connecting to Kinect...")
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_label.setMinimumSize(320, 240)
        self.view_label.setStyleSheet(
            "background-color: #1e1e1e; color: #aaa; font-size: 18px;"
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view_label)
        self.live_view = LiveView()
        self.live_view.hide()
        splitter.addWidget(self.live_view)
        layout.addWidget(splitter, stretch=1)

    def _build_toolbar(self):
        toolbar = QToolBar("Modes")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for mode in (MODE_RGB, MODE_DEPTH, MODE_SCANNER):
            action = QAction(mode, self)
            action.setCheckable(True)
            if mode == MODE_RGB:
                action.setChecked(True)
            action.triggered.connect(lambda checked, m=mode: self._switch_mode(m))
            toolbar.addAction(action)
        self._mode_actions = toolbar.actions()

    def _build_dock(self):
        dock = QDockWidget("Controls", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setFixedWidth(260)
        container = QWidget()
        layout = QVBoxLayout(container)

        # ── Server Connection ─────────────────────────────────────
        server_group = QGroupBox("Server Connection")
        sg_layout = QVBoxLayout(server_group)

        ip_row = QHBoxLayout()
        self.server_ip_edit = QLineEdit()
        self.server_ip_edit.setText(os.environ.get("KINECT_SERVER_HOST", "127.0.0.1"))
        self.server_ip_edit.setPlaceholderText("Server IP, e.g. 10.0.0.107")
        ip_row.addWidget(self.server_ip_edit, stretch=1)
        self.server_port_spin = QSpinBox()
        self.server_port_spin.setRange(1, 65535)
        self.server_port_spin.setValue(
            int(os.environ.get("KINECT_SERVER_PORT", "8000"))
        )
        self.server_port_spin.setFixedWidth(70)
        ip_row.addWidget(self.server_port_spin)
        sg_layout.addLayout(ip_row)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._toggle_connection)
        sg_layout.addWidget(self.btn_connect)

        self.server_status_label = QLabel("Not connected")
        self.server_status_label.setStyleSheet("color: #888;")
        self.server_status_label.setWordWrap(True)
        sg_layout.addWidget(self.server_status_label)

        self.backend_label = QLabel("Backend: unknown")
        self.backend_label.setWordWrap(True)
        sg_layout.addWidget(self.backend_label)
        layout.addWidget(server_group)

        # ── Depth visualisation range ─────────────────────────────
        viz_group = QGroupBox("Scan settings (apply to new scan)")
        self.settings_group = viz_group
        vg = QVBoxLayout(viz_group)

        vg.addWidget(QLabel("Near clip (mm):"))
        self.depth_near_spin = QSpinBox()
        self.depth_near_spin.setRange(0, 4000)
        self.depth_near_spin.setValue(500)
        self.depth_near_spin.setSingleStep(100)
        vg.addWidget(self.depth_near_spin)

        vg.addWidget(QLabel("Far clip (mm):"))
        self.depth_far_spin = QSpinBox()
        self.depth_far_spin.setRange(500, 8000)
        self.depth_far_spin.setValue(4000)
        self.depth_far_spin.setSingleStep(100)
        vg.addWidget(self.depth_far_spin)

        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setRange(2, 30)
        self.voxel_spin.setValue(5)
        self.voxel_spin.setSuffix(" mm")
        vg.addWidget(QLabel("Voxel size:"))
        vg.addWidget(self.voxel_spin)
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.5, 20)
        self.weight_spin.setValue(2)
        self.weight_spin.setSingleStep(0.5)
        vg.addWidget(QLabel("Final surface confidence (weight):"))
        vg.addWidget(self.weight_spin)
        self.crop_cb = QCheckBox("Crop to central region")
        self.crop_spin = QSpinBox()
        self.crop_spin.setRange(10, 100)
        self.crop_spin.setValue(70)
        self.crop_spin.setSuffix("% of image")
        vg.addWidget(self.crop_cb)
        vg.addWidget(self.crop_spin)
        self.calibration_label = QLabel(self._camera.name)
        self.calibration_label.setWordWrap(True)
        vg.addWidget(self.calibration_label)
        calibration_btn = QPushButton("Load RGB calibration JSON")
        calibration_btn.clicked.connect(self._load_calibration)
        vg.addWidget(calibration_btn)
        self.color_tracking_cb = QCheckBox("Color-assisted tracking (experimental)")
        self.color_tracking_cb.setToolTip(
            "Uses RGB-D motion to seed ICP; requires synchronized, textured views"
        )
        vg.addWidget(self.color_tracking_cb)
        self.live_cb = QCheckBox("Live fused surface feedback")
        self.live_cb.setChecked(True)
        self.live_cb.setToolTip(
            "Processes frames during capture; pending count shows when reconstruction falls behind"
        )
        vg.addWidget(self.live_cb)
        self.refine_cb = QCheckBox("Final pose refinement (experimental)")
        self.refine_cb.setToolTip(
            "Validate loop matches and rebuild fusion; uses additional memory and time"
        )
        vg.addWidget(self.refine_cb)
        self.record_cb = QCheckBox("Save local RGB-D recording")
        vg.addWidget(self.record_cb)
        layout.addWidget(viz_group)

        # ── Scanner Controls ──────────────────────────────────────
        scan_group = QGroupBox("3D Scanner")
        sg = QVBoxLayout(scan_group)

        self.btn_start_scan = QPushButton("Start Scan")
        self.btn_start_scan.clicked.connect(self._start_scan)
        sg.addWidget(self.btn_start_scan)

        self.btn_capture = QPushButton("Capture Frame")
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self._capture_frame)
        sg.addWidget(self.btn_capture)

        # Auto-capture
        auto_row = QHBoxLayout()
        self.auto_capture_cb = QCheckBox("Auto every")
        self.auto_capture_spin = QDoubleSpinBox()
        self.auto_capture_spin.setRange(0.03, 30.0)
        self.auto_capture_spin.setValue(0.5)
        self.auto_capture_spin.setSingleStep(0.01)
        self.auto_capture_spin.setDecimals(2)
        self.auto_capture_spin.setSuffix("s")
        self.auto_capture_cb.setEnabled(False)
        self.auto_capture_spin.setEnabled(False)
        self.auto_capture_cb.toggled.connect(self._toggle_auto_capture)
        auto_row.addWidget(self.auto_capture_cb)
        auto_row.addWidget(self.auto_capture_spin)
        sg.addLayout(auto_row)

        self.btn_preview_scan = QPushButton("Preview Scan")
        self.btn_preview_scan.setEnabled(False)
        self.btn_preview_scan.clicked.connect(self._preview_scan)
        sg.addWidget(self.btn_preview_scan)

        self.btn_stop_build = QPushButton("Stop && Build Mesh")
        self.btn_stop_build.setEnabled(False)
        self.btn_stop_build.clicked.connect(self._stop_and_build)
        sg.addWidget(self.btn_stop_build)

        export_row = QHBoxLayout()
        self.btn_export_ply = QPushButton("Export PLY")
        self.btn_export_ply.setEnabled(False)
        self.btn_export_ply.clicked.connect(self._export_ply)
        export_row.addWidget(self.btn_export_ply)
        self.btn_export_obj = QPushButton("Export OBJ")
        self.btn_export_obj.setEnabled(False)
        self.btn_export_obj.clicked.connect(self._export_obj)
        export_row.addWidget(self.btn_export_obj)
        sg.addLayout(export_row)
        self.btn_export_glb = QPushButton("Export textured GLB")
        self.btn_export_glb.setEnabled(False)
        self.btn_export_glb.clicked.connect(lambda: self._export_texture("glb"))
        sg.addWidget(self.btn_export_glb)
        self.btn_export_texture_obj = QPushButton("Export textured OBJ bundle")
        self.btn_export_texture_obj.setEnabled(False)
        self.btn_export_texture_obj.clicked.connect(
            lambda: self._export_texture("obj.zip")
        )
        sg.addWidget(self.btn_export_texture_obj)
        self.texture_exposure_cb = QCheckBox("Match texture exposures")
        self.texture_exposure_cb.setToolTip(
            "Bounded RGB gains must improve held-out, depth-visible overlap"
        )
        self.texture_best_cb = QCheckBox("Use one best view per texel")
        self.texture_best_cb.setToolTip(
            "Selects by viewing angle and distance; can sharpen detail but expose seams"
        )
        sg.addWidget(self.texture_exposure_cb)
        sg.addWidget(self.texture_best_cb)
        self.btn_export_session = QPushButton("Save full RGB-D session")
        self.btn_export_session.setEnabled(False)
        self.btn_export_session.clicked.connect(self._export_session)
        sg.addWidget(self.btn_export_session)

        # Preview 3D button (current scan)
        self.btn_preview_3d = QPushButton("Preview 3D")
        self.btn_preview_3d.setEnabled(False)
        self.btn_preview_3d.clicked.connect(self._preview_3d)
        sg.addWidget(self.btn_preview_3d)

        # Save / Load mesh
        mesh_row = QHBoxLayout()
        self.btn_save_mesh = QPushButton("Save Mesh")
        self.btn_save_mesh.setEnabled(False)
        self.btn_save_mesh.clicked.connect(self._save_mesh)
        mesh_row.addWidget(self.btn_save_mesh)
        self.btn_load_mesh = QPushButton("Load Mesh")
        self.btn_load_mesh.clicked.connect(self._load_mesh)
        mesh_row.addWidget(self.btn_load_mesh)
        sg.addLayout(mesh_row)

        # View 3D file from disk
        self.btn_view_file = QPushButton("View 3D File")
        self.btn_view_file.clicked.connect(self._view_3d_file)
        sg.addWidget(self.btn_view_file)

        self.frame_count_label = QLabel("Stored: 0 | Integrated: 0")
        sg.addWidget(self.frame_count_label)

        self.scan_status_label = QLabel("Idle")
        self.scan_status_label.setWordWrap(True)
        self.scan_status_label.setMaximumWidth(240)
        sg.addWidget(self.scan_status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        sg.addWidget(self.progress_bar)

        layout.addWidget(scan_group)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._capture_frame)

    def _build_statusbar(self):
        self.statusBar().showMessage("Ready")
        self.fps_label = QLabel("FPS: --")
        self.kinect_label = QLabel("Kinect: connecting...")
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().addPermanentWidget(self.kinect_label)

    def _set_scan_controls_enabled(self, enabled: bool):
        """Enable/disable scan controls based on server connection state."""
        self.btn_start_scan.setEnabled(enabled)
        if not enabled:
            self._scanning = False
            self._auto_timer.stop()
            self.auto_capture_cb.setChecked(False)
            for control in (
                self.btn_capture,
                self.btn_stop_build,
                self.btn_preview_scan,
                self.auto_capture_cb,
                self.auto_capture_spin,
                self.btn_export_ply,
                self.btn_export_obj,
                self.btn_export_glb,
                self.btn_export_texture_obj,
                self.btn_export_session,
                self.btn_save_mesh,
            ):
                control.setEnabled(False)
            self.settings_group.setEnabled(True)

    # ── Server connection ─────────────────────────────────────────────
    def _toggle_connection(self):
        if self.server_client.is_connected:
            self.server_client.disconnect()
            self.btn_connect.setText("Connect")
            self.server_status_label.setText("Disconnected")
            self.server_status_label.setStyleSheet("color: #888;")
            self.server_ip_edit.setEnabled(True)
            self.server_port_spin.setEnabled(True)
            self._set_scan_controls_enabled(False)
            return

        host = self.server_ip_edit.text().strip()
        port = self.server_port_spin.value()
        if not host:
            self.server_status_label.setText("Enter a server IP")
            self.server_status_label.setStyleSheet("color: #c00;")
            return

        self.server_status_label.setText("Connecting...")
        self.server_status_label.setStyleSheet("color: #888;")
        self.btn_connect.setEnabled(False)
        QApplication.processEvents()

        ok = self.server_client.connect_to_server(host, port)
        self.btn_connect.setEnabled(True)

        if ok:
            self.btn_connect.setText("Disconnect")
            self.server_ip_edit.setEnabled(False)
            self.server_port_spin.setEnabled(False)
        # Signal handlers below update the rest

    def _on_server_connected(self):
        self.server_status_label.setText("Connected")
        self.server_status_label.setStyleSheet("color: #0a0; font-weight: bold;")
        self._set_scan_controls_enabled(True)
        self.statusBar().showMessage("Connected to server")

    def _on_server_disconnected(self, reason: str):
        self.server_status_label.setText(f"Error: {reason}")
        self.server_status_label.setStyleSheet("color: #c00;")
        self._set_scan_controls_enabled(False)
        self.statusBar().showMessage(f"Server connection failed: {reason}")

    # ── mode switching ────────────────────────────────────────────────
    def _switch_mode(self, mode: str):
        self._mode = mode
        for action in self._mode_actions:
            action.setChecked(action.text() == mode)

    # ── frame display ─────────────────────────────────────────────────
    def _on_frame(self, video: np.ndarray, depth: np.ndarray, metadata=None):
        if self._closing:
            return
        self._fps_counter += 1
        self.kinect_label.setText("Kinect: connected")
        self._last_rgb = video
        self._last_depth = depth
        self._frame_sequence += 1
        self._last_frame_metadata = dict(metadata or {})
        self._last_frame_time = self._last_frame_metadata.get(
            "captured_monotonic_s", time.monotonic()
        )
        # GUI sequence remains unique if the USB device reconnects mid-session.
        self._last_frame_metadata["frame_id"] = self._frame_sequence
        self._last_frame_metadata.setdefault("timestamp_s", time.time())

        if self._mode == MODE_RGB:
            self._show_rgb(video)
        elif self._mode == MODE_DEPTH:
            self._show_depth(depth)
        elif self._mode == MODE_SCANNER:
            self._show_scanner(video, depth)

    def _show_rgb(self, rgb):
        self._set_pixmap(numpy_to_qimage(rgb))

    def _selected_roi(self):
        if not self.crop_cb.isChecked():
            return None
        fraction = self.crop_spin.value() / 100
        width, height = int(640 * fraction), int(480 * fraction)
        x, y = (640 - width) // 2, (480 - height) // 2
        return x, y, x + width, y + height

    def _depth_display(self, depth):
        roi = self._selected_roi()
        if roi is not None:
            x0, y0, x1, y1 = roi
            masked = np.zeros_like(depth)
            masked[y0:y1, x0:x1] = depth[y0:y1, x0:x1]
            depth = masked
        return colorize_depth(
            depth, self.depth_near_spin.value(), self.depth_far_spin.value()
        )

    def _show_depth(self, depth):
        self._set_pixmap(numpy_to_qimage(self._depth_display(depth)))

    def _show_scanner(self, rgb, depth):
        combined = np.hstack([rgb, self._depth_display(depth)])
        self._set_pixmap(numpy_to_qimage(combined))

    def _set_pixmap(self, qimg):
        self.view_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.view_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "RGB calibration", "", "JSON (*.json)"
        )
        if path:
            try:
                with open(path) as source:
                    self._camera = CameraCalibration(**json.load(source))
                self.calibration_label.setText(self._camera.name)
            except (ValueError, TypeError, OSError) as exc:
                self.scan_status_label.setText(f"Invalid calibration: {exc}")

    # ── scanning workflow ─────────────────────────────────────────────
    def _start_scan(self):
        if not self.server_client.is_connected:
            QMessageBox.warning(self, "Not Connected", "Connect to the server first.")
            return

        try:
            voxel = self.voxel_spin.value() / 1000
            roi = self._selected_roi()
            settings = ScanSettings(
                camera=self._camera,
                near_m=self.depth_near_spin.value() / 1000,
                far_m=self.depth_far_spin.value() / 1000,
                voxel_m=voxel,
                truncation_m=min(0.2, max(0.04, voxel * 8)),
                final_weight=self.weight_spin.value(),
                color_recovery=self.color_tracking_cb.isChecked(),
                live_reconstruction=self.live_cb.isChecked(),
                refine_poses=self.refine_cb.isChecked(),
                roi=roi,
            )
        except ValueError as exc:
            self.scan_status_label.setText(str(exc))
            return
        self._reset_pending = True
        self.settings_group.setEnabled(False)
        self.btn_start_scan.setEnabled(False)
        self.scan_status_label.setText("Starting scan...")
        self.task_worker.submit(
            ServerTask(
                ServerTaskType.RESET,
                {"settings": settings.to_dict(), "record": self.record_cb.isChecked()},
            )
        )

    def _on_reset_done(self, result):
        self._reset_pending = False
        if not self.server_client.is_connected or self._closing:
            return
        self._session_id = result.get("session_id")
        self._has_mesh = False
        self.live_view.reset()
        self.live_view.setVisible(
            result.get("settings", {}).get("live_reconstruction", False)
        )
        self._last_capture_id = None
        self._server_stored = 0
        self._server_integrated = 0

        self._scanning = True
        self._switch_mode(MODE_SCANNER)
        self.btn_start_scan.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.btn_stop_build.setEnabled(True)
        self.btn_export_ply.setEnabled(False)
        self.btn_export_obj.setEnabled(False)
        self.btn_export_glb.setEnabled(False)
        self.btn_export_texture_obj.setEnabled(False)
        self.btn_export_session.setEnabled(False)
        self.btn_preview_3d.setEnabled(False)
        self.btn_save_mesh.setEnabled(False)
        self.btn_preview_scan.setEnabled(True)
        self.auto_capture_cb.setEnabled(True)
        self.auto_capture_spin.setEnabled(True)
        self.frame_count_label.setText("Stored: 0 | Integrated: 0")
        self.scan_status_label.setText("Scanning — capture frames")
        self.statusBar().showMessage(
            "Scan started. Move Kinect and press Capture Frame."
        )

    def _capture_frame(self):
        if (
            not self._scanning
            or self._preview_pending
            or not self.server_client.is_connected
            or self._closing
        ):
            return
        if self._last_rgb is None or self._last_depth is None:
            self.scan_status_label.setText("No frame available yet")
            return

        frame_id = self._last_frame_metadata.get("frame_id")
        if (
            frame_id == self._last_capture_id
            or time.monotonic() - self._last_frame_time > 0.5
        ):
            self.scan_status_label.setText("Waiting for a fresh camera frame")
            return
        self._last_capture_id = frame_id
        queued = self.task_worker.submit(
            ServerTask(
                ServerTaskType.SEND_FRAME,
                {
                    "rgb": self._last_rgb.copy(),
                    "depth": self._last_depth.copy(),
                    "metadata": self._last_frame_metadata.copy(),
                },
            )
        )
        if not queued:
            self.scan_status_label.setText("Upload queue full; skipped capture")

    def _toggle_auto_capture(self, checked: bool):
        if checked and self._scanning:
            self._auto_timer.start(int(self.auto_capture_spin.value() * 1000))
        else:
            self._auto_timer.stop()

    def _stop_and_build(self):
        self._scanning = False
        self._auto_timer.stop()
        self.auto_capture_cb.setChecked(False)
        self.auto_capture_cb.setEnabled(False)
        self.auto_capture_spin.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_stop_build.setEnabled(False)
        self.btn_preview_scan.setEnabled(False)

        stored = self._server_stored
        self.scan_status_label.setText(f"Processing {stored} frames on server...")
        self.progress_bar.setRange(0, 0)  # indeterminate until progress arrives
        self.progress_bar.setVisible(True)

        self.task_worker.submit(ServerTask(ServerTaskType.BUILD_MESH))

    def _preview_scan(self):
        if self._server_stored == 0:
            QMessageBox.information(
                self, "Preview", "Capture at least one frame first."
            )
            return
        self._preview_pending = True
        self._auto_timer.stop()
        self.btn_capture.setEnabled(False)
        self.btn_stop_build.setEnabled(False)
        self.btn_preview_scan.setEnabled(False)
        self.scan_status_label.setText("Generating preview on server...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self.task_worker.submit(ServerTask(ServerTaskType.PREVIEW))

    def _preview_3d(self):
        if self._last_preview_path:
            launch_viewer_subprocess(self._last_preview_path)

    def _ensure_export_dir(self):
        os.makedirs(EXPORT_DIR, exist_ok=True)
        return EXPORT_DIR

    def _export_ply(self):
        default_path = os.path.join(self._ensure_export_dir(), "scan.ply")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PLY", default_path, "PLY files (*.ply)"
        )
        if path:
            self.btn_export_ply.setEnabled(False)
            self.task_worker.submit(
                ServerTask(ServerTaskType.EXPORT_PLY, {"path": path})
            )

    def _export_obj(self):
        default_path = os.path.join(self._ensure_export_dir(), "scan.obj")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export OBJ", default_path, "OBJ files (*.obj)"
        )
        if path:
            self.btn_export_obj.setEnabled(False)
            self.task_worker.submit(
                ServerTask(ServerTaskType.EXPORT_OBJ, {"path": path})
            )

    def _export_texture(self, fmt):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export textured surface",
            os.path.join(self._ensure_export_dir(), f"scan.{fmt}"),
            "GLB (*.glb)" if fmt == "glb" else "ZIP (*.zip)",
        )
        if path:
            self.btn_export_glb.setEnabled(False)
            self.btn_export_texture_obj.setEnabled(False)
            self.task_worker.submit(
                ServerTask(
                    ServerTaskType.EXPORT_TEXTURE,
                    {
                        "path": path,
                        "format": fmt,
                        "options": {
                            "exposure_correction": self.texture_exposure_cb.isChecked(),
                            "blend_mode": "best"
                            if self.texture_best_cb.isChecked()
                            else "blend",
                        },
                    },
                )
            )

    def _export_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save lossless RGB-D session",
            os.path.join(self._ensure_export_dir(), "scan-session.zip"),
            "ZIP (*.zip)",
        )
        if path:
            self.task_worker.submit(
                ServerTask(ServerTaskType.EXPORT_SESSION, {"path": path})
            )

    def _save_mesh(self):
        os.makedirs(MESH_DIR, exist_ok=True)
        filename = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
        path = os.path.join(MESH_DIR, filename)
        self.btn_save_mesh.setEnabled(False)
        self.task_worker.submit(ServerTask(ServerTaskType.SAVE_MESH, {"path": path}))

    def _load_mesh(self):
        start_dir = MESH_DIR if os.path.isdir(MESH_DIR) else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Mesh",
            start_dir,
            "PLY files (*.ply);;All 3D files (*.obj *.ply *.stl *.glb)",
        )
        if path:
            self.statusBar().showMessage(f"Opening: {path}")
            launch_viewer_subprocess(path)

    def _view_3d_file(self):
        start_dir = EXPORT_DIR if os.path.isdir(EXPORT_DIR) else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D File",
            start_dir,
            "3D files (*.obj *.ply *.stl *.glb);;OBJ files (*.obj);;PLY files (*.ply);;STL files (*.stl)",
        )
        if path:
            self.statusBar().showMessage(f"Opening: {path}")
            launch_viewer_subprocess(path)

    # ── Task/server signal handlers ──────────────────────────────────
    def _on_server_status(self, status):
        backend = status.get("backend", {})
        self.backend_label.setText(
            f"Fusion: {backend.get('device', 'unknown')} · ICP: {backend.get('tracking', 'unknown')}"
        )
        self.backend_label.setToolTip(backend.get("fallback_reason") or "")

    def _on_live_updated(self, snapshot):
        if (
            self._closing
            or self._reset_pending
            or snapshot.get("session_id") != self._session_id
        ):
            return
        self.live_view.set_snapshot(snapshot)
        self._on_server_status(snapshot)
        self._server_stored = max(self._server_stored, snapshot.get("stored_count", 0))
        self._server_integrated = snapshot.get("frame_count", 0)
        self.frame_count_label.setText(
            f"Stored: {self._server_stored} | Integrated: {self._server_integrated}"
        )
        if self._scanning and not self._preview_pending:
            self.scan_status_label.setText(
                snapshot.get("result", {}).get("message", "Scanning")
            )

    def _on_frame_stored(self, result: dict):
        if result.get("session_id") and result["session_id"] != self._session_id:
            return
        if not result.get("success"):
            self.auto_capture_cb.setChecked(False)
        self._server_stored = max(
            self._server_stored, result.get("stored_count", self._server_stored)
        )
        self.frame_count_label.setText(
            f"Stored: {self._server_stored} | Integrated: {self._server_integrated}"
        )
        self.btn_export_session.setEnabled(self._server_stored > 0)
        self.scan_status_label.setText(result.get("message", "Frame stored"))
        self.statusBar().showMessage(result.get("message", "Frame stored"))

    def _on_process_progress(self, current: int, total: int, result: dict):
        if result.get("session_id") and result["session_id"] != self._session_id:
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self._server_integrated = result.get("frame_count", self._server_integrated)
        self.frame_count_label.setText(
            f"Stored: {total} | Integrated: {self._server_integrated}"
        )
        self.scan_status_label.setText(
            f"Processing {current}/{total}: {result.get('message', '')}"
        )

    def _on_build_mesh_done(self, success: bool, detail: str):
        self._has_mesh = success
        self.progress_bar.setVisible(False)
        self.scan_status_label.setText(detail)

        if success:
            self.btn_export_ply.setEnabled(True)
            self.btn_export_obj.setEnabled(True)
            self.btn_export_glb.setEnabled(True)
            self.btn_export_texture_obj.setEnabled(True)
            self.btn_save_mesh.setEnabled(True)
            self.statusBar().showMessage("Mesh built. Ready to export or preview.")
        else:
            self.statusBar().showMessage("Mesh build failed.")

        self.btn_start_scan.setEnabled(True)
        self.settings_group.setEnabled(True)

    def _resume_capture(self):
        self._preview_pending = False
        if self._scanning:
            self.btn_capture.setEnabled(True)
            self.btn_stop_build.setEnabled(True)
            self.btn_preview_scan.setEnabled(True)
            self._toggle_auto_capture(self.auto_capture_cb.isChecked())

    def _on_preview_done(self, path: str):
        self._resume_capture()
        self.progress_bar.setVisible(False)
        if self._scanning:
            self.btn_preview_scan.setEnabled(True)

        if path:
            self._last_preview_path = path
            self.btn_preview_3d.setEnabled(True)
            self.scan_status_label.setText(
                f"Scanning — {self._server_stored} frames stored"
            )
            launch_viewer_subprocess(path)
        else:
            self.scan_status_label.setText("Preview extraction failed")

    def _on_export_done(self, success: bool, path: str):
        self.btn_export_ply.setEnabled(self._has_mesh)
        self.btn_export_obj.setEnabled(self._has_mesh)
        self.btn_export_glb.setEnabled(self._has_mesh)
        self.btn_export_texture_obj.setEnabled(self._has_mesh)
        if success:
            self.statusBar().showMessage(f"Exported: {path}")
            QMessageBox.information(self, "Export", f"Saved to:\n{path}")
        else:
            QMessageBox.warning(self, "Export", f"Failed to export:\n{path}")

    def _on_save_mesh_done(self, success: bool, path: str):
        self.btn_save_mesh.setEnabled(True)
        if success:
            self.statusBar().showMessage(f"Mesh saved: {path}")
            QMessageBox.information(self, "Save Mesh", f"Saved to:\n{path}")
        else:
            QMessageBox.warning(self, "Save Mesh", "Failed to save mesh.")

    def _on_task_started(self, msg: str):
        self.statusBar().showMessage(msg)

    def _on_task_error(self, msg: str):
        for button in (
            self.btn_export_ply,
            self.btn_export_obj,
            self.btn_export_glb,
            self.btn_export_texture_obj,
        ):
            button.setEnabled(self._has_mesh)
        self.progress_bar.setVisible(False)
        if self._reset_pending:
            self._reset_pending = False
            self.btn_start_scan.setEnabled(True)
            self.settings_group.setEnabled(True)
        elif self._preview_pending:
            self._resume_capture()
        elif not self._scanning:
            self.btn_start_scan.setEnabled(True)
            self.settings_group.setEnabled(True)
        self.scan_status_label.setText(f"Error: {msg}")
        self.statusBar().showMessage(f"Error: {msg}")

    # ── FPS / errors / cleanup ────────────────────────────────────────
    def _update_fps(self):
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed > 0:
            self._fps_value = self._fps_counter / elapsed
        self._fps_counter = 0
        self._last_fps_time = now
        self.fps_label.setText(f"FPS: {self._fps_value:.1f}")

    def _on_error(self, msg: str):
        if self._closing:
            return
        self.kinect_label.setText("Kinect: error")
        if self._last_rgb is None:
            self.view_label.setText(msg)
        self.statusBar().showMessage(f"Error: {msg}")

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self._auto_timer.stop()
            self._fps_timer.stop()
            self.worker.stop()
            self.task_worker.stop()
            self.server_client.disconnect()
        self.worker.wait(100)
        self.task_worker.wait(100)
        if self.worker.isRunning() or self.task_worker.isRunning():
            # QThreads must finish before their QObject owners are destroyed.
            message = (
                "Stopping Kinect camera..."
                if self.worker.isRunning()
                else "Finishing current server request before closing..."
            )
            self.statusBar().showMessage(message)
            event.ignore()
            QTimer.singleShot(200, self.close)
            return
        super().closeEvent(event)
