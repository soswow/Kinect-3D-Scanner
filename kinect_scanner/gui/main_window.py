"""MainWindow — primary application window (client/server mode)."""

import os
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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

from shared.calibration import raw_depth_to_mm
from shared.capture import RGB_MODE_FPS
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings

from ..config import MODE_DEPTH, MODE_RGB, MODE_SCANNER
from ..server_client import ServerClient
from ..server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker
from ..viewer import launch_viewer_subprocess
from ..worker import KinectWorker
from .components import CameraPreview, CollapsibleSection
from .dialogs import ExportDialog, SessionProtectionDialog
from .live_view import LiveView
from .widgets import (
    FrameIntervalSpinBox,
    colorize_depth,
    depth_legend_text,
    numpy_to_qimage,
)

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
        self._paused = False
        self._build_pending = False
        self._build_failed = False
        self._camera_ok = False
        self._capture_waiting = ""
        self._session_dirty = False
        self._capture_revision = 0
        self._saved_revision = 0
        self._capture_run_id = uuid.uuid4().hex[:12]
        self._pending_action = None
        self._export_pending = None
        self._connect_pending = False
        self._restore_on_status = False
        self._session_settings = None
        self._final_preview_pending = False
        self._final_preview_session = None
        self._preview_session = None
        self._close_approved = False
        self._operation_error = ""
        self._progress_link_ok = True
        self._server_operation = None
        self._reconcile_on_status = False
        self._status_pending = False
        self._fps_counter = 0
        self._fps_value = 0.0
        self._last_fps_time = time.time()
        self._last_rgb = None
        self._last_depth = None
        self._sensor_calibration = load_calibration()
        self._camera = self._sensor_calibration.depth
        self._frame_sequence = 0
        self._last_frame_metadata = {}
        self._last_frame_time = 0
        self._last_capture_id = None
        self._auto_frames_since_capture = 0
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
        self.server_client.final_preview_done.connect(self._on_final_preview_done)
        self.server_client.task_failed.connect(self._on_task_failed)
        self.server_client.websocket_status.connect(self._on_websocket_status)
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

        self._start_camera()

        # Disable scan controls until server connected
        self._set_scan_controls_enabled(False)
        if os.environ.get("KINECT_AUTOCONNECT") == "1":
            QTimer.singleShot(0, self._toggle_connection)

    def _start_camera(self):
        self.worker = KinectWorker(
            rgb_mode=self.rgb_mode_combo.currentData(),
        )
        worker = self.worker
        # Ignore any queued observation from the old worker after a mode change.
        def received(*args):
            if worker is self.worker:
                self._on_frame(*args)
        if hasattr(worker, "frame_pair_ready"):
            worker.frame_pair_ready.connect(received)
        else:
            worker.frame_ready.connect(received)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _restart_camera(self):
        self.worker.stop()
        if not self.worker.wait(2500):
            raise RuntimeError("Camera did not stop within 2.5 seconds")
        self._last_rgb = self._last_depth = None
        self._camera_ok = False
        self.view_label.set_stale(True, "Camera restarting…")
        self._last_frame_metadata = {}
        self._reset_auto_capture_cadence()
        self._start_camera()

    def _change_rgb_mode(self):
        self.auto_capture_spin.set_fps(RGB_MODE_FPS[self.rgb_mode_combo.currentData()])
        self._reset_auto_capture_cadence()
        try:
            self._restart_camera()
        except RuntimeError as exc:
            self.scan_status_label.setText(str(exc))

    # ── UI construction ───────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        self.guidance_label = QLabel(
            "Keep the subject stationary. Move the Kinect slowly around it with overlapping views."
        )
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setAccessibleName("Scanning guidance")
        layout.addWidget(self.guidance_label)
        self.camera_panel = QWidget()
        camera_layout = QVBoxLayout(self.camera_panel)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_title = QLabel("Live camera · Color")
        camera_layout.addWidget(self.camera_title)
        self.view_label = CameraPreview()
        camera_layout.addWidget(self.view_label, stretch=1)
        self.depth_legend_label = QLabel()
        self.depth_legend_label.setWordWrap(True)
        self.depth_legend_label.hide()
        camera_layout.addWidget(self.depth_legend_label)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.live_view = LiveView()
        self.live_view.hide()
        self.splitter.addWidget(self.live_view)
        self.splitter.addWidget(self.camera_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter, stretch=1)

    def _build_toolbar(self):
        toolbar = QToolBar("Views")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        group = QActionGroup(self)
        group.setExclusive(True)
        self._mode_actions = []
        for title, mode in (("Scan", MODE_SCANNER), ("Color", MODE_RGB), ("Depth", MODE_DEPTH)):
            action = QAction(title, self)
            action.setData(mode)
            action.setCheckable(True)
            action.setChecked(mode == self._mode)
            action.triggered.connect(lambda checked, m=mode: self._switch_mode(m))
            group.addAction(action)
            toolbar.addAction(action)
            self._mode_actions.append(action)
        toolbar.addSeparator()
        open_action = QAction("Open Model…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._view_3d_file)
        toolbar.addAction(open_action)
        self.pause_action = QAction("Pause / Resume", self)
        self.pause_action.setShortcut(QKeySequence("Space"))
        self.pause_action.triggered.connect(self._shortcut_pause)
        self.addAction(self.pause_action)
        self.capture_action = QAction("Capture Frame", self)
        self.capture_action.setShortcut(QKeySequence("C"))
        self.capture_action.triggered.connect(self._shortcut_capture)
        self.addAction(self.capture_action)

    def _build_dock(self):
        dock = QDockWidget("Scan", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setMinimumWidth(300)
        dock.setMaximumWidth(430)
        self.controls_dock = dock
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 6, 8, 6)

        self.scan_status_label = QLabel("Waiting for connection")
        self.scan_status_label.setWordWrap(True)
        self.scan_status_label.setAccessibleName("Scan state")
        self.scan_status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.scan_status_label)
        self.frame_count_label = QLabel("Captured: 0 · Added to model: 0")
        self.frame_count_label.setWordWrap(True)
        layout.addWidget(self.frame_count_label)
        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.addItem("Automatic", "automatic")
        self.capture_mode_combo.addItem("Manual", "manual")
        mode_label = QLabel("Capture mode")
        mode_label.setBuddy(self.capture_mode_combo)
        layout.addWidget(mode_label)
        layout.addWidget(self.capture_mode_combo)
        # Retain the cadence flag internally; operation is controlled by explicit actions.
        self.auto_capture_cb = QCheckBox(container)
        self.auto_capture_cb.hide()
        self.auto_capture_cb.toggled.connect(self._toggle_auto_capture)
        self.auto_capture_spin = FrameIntervalSpinBox(10)
        self.auto_capture_spin.valueChanged.connect(self._reset_auto_capture_cadence)
        self.interval_row = QWidget()
        interval_layout = QHBoxLayout(self.interval_row)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        interval_label = QLabel("Capture interval")
        interval_label.setBuddy(self.auto_capture_spin)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.auto_capture_spin)
        layout.addWidget(self.interval_row)
        self.interval_help = QLabel("Capture slows automatically while processing catches up.")
        self.interval_help.setWordWrap(True)
        layout.addWidget(self.interval_help)
        self.adaptive_capture_cb = QCheckBox(container)
        self.adaptive_capture_cb.setChecked(True)
        self.adaptive_capture_cb.hide()
        self.texture_exposure_cb = QCheckBox(container)
        self.texture_exposure_cb.hide()
        self.texture_best_cb = QCheckBox(container)
        self.texture_best_cb.hide()

        start_row = QHBoxLayout()
        self.btn_start_scan = QPushButton("Start Scan")
        self.btn_start_scan.clicked.connect(self._start_scan)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setToolTip("Pause or resume capture (Space)")
        self.btn_pause.clicked.connect(self._pause_or_resume)
        start_row.addWidget(self.btn_start_scan)
        start_row.addWidget(self.btn_pause)
        layout.addLayout(start_row)
        self.btn_capture = QPushButton("Capture Frame")
        self.btn_capture.setToolTip("Capture one fresh frame (C)")
        self.btn_capture.clicked.connect(self._capture_frame)
        layout.addWidget(self.btn_capture)
        finish_row = QHBoxLayout()
        self.btn_preview_scan = QPushButton("Inspect Scan")
        self.btn_preview_scan.clicked.connect(self._preview_scan)
        self.btn_stop_build = QPushButton("Finish Scan")
        self.btn_stop_build.clicked.connect(self._stop_and_build)
        finish_row.addWidget(self.btn_preview_scan)
        finish_row.addWidget(self.btn_stop_build)
        layout.addLayout(finish_row)
        output_row = QHBoxLayout()
        self.btn_export = QPushButton("Export…")
        self.btn_export.clicked.connect(self._choose_export)
        self.btn_export_session = QPushButton("Save Session…")
        self.btn_export_session.clicked.connect(self._export_session)
        output_row.addWidget(self.btn_export)
        output_row.addWidget(self.btn_export_session)
        layout.addLayout(output_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.readiness_label = QLabel("Connect the server and wait for live camera frames.")
        self.readiness_label.setWordWrap(True)
        layout.addWidget(self.readiness_label)
        # Old command entry points remain for API/pipeline compatibility, without UI duplication.
        for name, title, handler in (
            ("btn_export_ply", "Export PLY", self._export_ply),
            ("btn_export_obj", "Export OBJ", self._export_obj),
            ("btn_export_glb", "Export textured GLB", lambda: self._export_texture("glb")),
            ("btn_export_texture_obj", "Export textured OBJ", lambda: self._export_texture("obj.zip")),
            ("btn_preview_3d", "View snapshot", self._preview_3d),
            ("btn_save_mesh", "Save Mesh", self._save_mesh),
            ("btn_load_mesh", "Open Model", self._load_mesh),
            ("btn_view_file", "Open Model", self._view_3d_file),
        ):
            button = QPushButton(title, container)
            button.clicked.connect(handler)
            button.hide()
            setattr(self, name, button)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.settings_scroll = scroll
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 4, 0, 4)
        self.settings_group = QGroupBox("Setup for next scan")
        vg = QVBoxLayout(self.settings_group)
        self.depth_near_spin = QSpinBox()
        self.depth_near_spin.setRange(0, 7999)
        self.depth_near_spin.setValue(500)
        self.depth_near_spin.setSingleStep(100)
        self.depth_near_spin.setSuffix(" mm")
        self.depth_far_spin = QSpinBox()
        self.depth_far_spin.setRange(1, 8000)
        self.depth_far_spin.setValue(4000)
        self.depth_far_spin.setSingleStep(100)
        self.depth_far_spin.setSuffix(" mm")
        for title, control in (("Nearest surface", self.depth_near_spin), ("Farthest surface", self.depth_far_spin)):
            label = QLabel(title)
            label.setBuddy(control)
            vg.addWidget(label)
            vg.addWidget(control)
        self.crop_cb = QCheckBox("Crop to central region")
        self.crop_spin = QSpinBox()
        self.crop_spin.setRange(10, 100)
        self.crop_spin.setValue(70)
        self.crop_spin.setSuffix("% of image")
        self.crop_spin.setEnabled(False)
        self.crop_cb.toggled.connect(self.crop_spin.setEnabled)
        vg.addWidget(self.crop_cb)
        vg.addWidget(self.crop_spin)
        self.record_cb = QCheckBox("Record captures locally")
        self.record_cb.setToolTip("Lossless RGB/depth captures are saved in the recordings folder for later replay.")
        vg.addWidget(self.record_cb)
        self.settings_error_label = QLabel()
        self.settings_error_label.setWordWrap(True)
        vg.addWidget(self.settings_error_label)

        advanced = CollapsibleSection("Advanced reconstruction")
        av = advanced.content_layout
        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setRange(2, 30)
        self.voxel_spin.setValue(5)
        self.voxel_spin.setSuffix(" mm")
        self.final_voxel_spin = QDoubleSpinBox()
        self.final_voxel_spin.setRange(0, 5)
        self.final_voxel_spin.setValue(0)
        self.final_voxel_spin.setSuffix(" mm")
        self.final_voxel_spin.setSpecialValueText("Use live resolution")
        self.final_blocks_spin = QSpinBox()
        self.final_blocks_spin.setRange(128, 50000)
        self.final_blocks_spin.setValue(5000)
        self.final_blocks_spin.setToolTip("5000 blocks uses about 391 MiB, plus live reconstruction and working memory.")
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.5, 20)
        self.weight_spin.setValue(2)
        self.weight_spin.setSingleStep(0.5)
        for title, control in (("Live voxel size", self.voxel_spin), ("Final voxel size", self.final_voxel_spin), ("Final memory budget (blocks)", self.final_blocks_spin), ("Final surface confidence", self.weight_spin)):
            label = QLabel(title)
            label.setBuddy(control)
            av.addWidget(label)
            av.addWidget(control)
        self.rgb_mode_combo = QComboBox()
        self.rgb_mode_combo.addItem("Color detail · 1280 × 1024, 10 fps", "rgb_high_res")
        self.rgb_mode_combo.addItem("Motion detail · 640 × 480, 30 fps", "rgb_low_res")
        self.rgb_mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.rgb_mode_combo.setMinimumContentsLength(18)
        self.rgb_mode_combo.currentIndexChanged.connect(self._change_rgb_mode)
        av.addWidget(QLabel("Camera capture"))
        av.addWidget(self.rgb_mode_combo)
        self.live_cb = QCheckBox("Show live reconstruction")
        self.live_cb.setChecked(True)
        av.addWidget(self.live_cb)
        self.calibration_label = QLabel(self._sensor_calibration.name)
        self.calibration_label.setWordWrap(True)
        av.addWidget(self.calibration_label)
        calibration_btn = QPushButton("Load Calibration…")
        calibration_btn.clicked.connect(self._load_calibration)
        av.addWidget(calibration_btn)
        vg.addWidget(advanced)
        experimental = CollapsibleSection("Experimental options")
        ev = experimental.content_layout
        self.color_tracking_cb = QCheckBox("Color-assisted tracking")
        self.refine_cb = QCheckBox("Refine final camera poses")
        self.relocalize_cb = QCheckBox("Recover lost tracking")
        self.confidence_cb = QCheckBox("Use sensor confidence")
        for control, help_text in (
            (self.color_tracking_cb, "Uses synchronized color/depth observations to help initialize tracking."),
            (self.refine_cb, "Validates loop matches and rebuilds fusion; needs extra time and memory."),
            (self.relocalize_cb, "Attempts verified recovery after skipped frames; repeated scenes may be ambiguous."),
            (self.confidence_cb, "Weights depth using range, angle and edges; may require more observations."),
        ):
            control.setToolTip(help_text)
            ev.addWidget(control)
        vg.addWidget(experimental)
        settings_layout.addWidget(self.settings_group)
        connection = CollapsibleSection("Connection details")
        self.connection_section = connection
        cl = connection.content_layout
        self.server_ip_edit = QLineEdit(os.environ.get("KINECT_SERVER_HOST", "127.0.0.1"))
        self.server_ip_edit.setPlaceholderText("Server host")
        self.server_port_spin = QSpinBox()
        self.server_port_spin.setRange(1, 65535)
        self.server_port_spin.setValue(int(os.environ.get("KINECT_SERVER_PORT", "8000")))
        for title, control in (("Server host", self.server_ip_edit), ("Port", self.server_port_spin)):
            label = QLabel(title)
            label.setBuddy(control)
            cl.addWidget(label)
            cl.addWidget(control)
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._toggle_connection)
        cl.addWidget(self.btn_connect)
        self.server_status_label = QLabel("Not connected")
        self.server_status_label.setWordWrap(True)
        cl.addWidget(self.server_status_label)
        self.backend_label = QLabel("Backend: unknown")
        self.backend_label.setWordWrap(True)
        cl.addWidget(self.backend_label)
        settings_layout.addWidget(connection)
        settings_layout.addStretch()
        scroll.setWidget(settings)
        layout.addWidget(scroll, stretch=1)
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [310], Qt.Orientation.Horizontal)
        self.capture_mode_combo.currentIndexChanged.connect(self._capture_mode_changed)
        self.depth_near_spin.valueChanged.connect(self._validate_setup)
        self.depth_far_spin.valueChanged.connect(self._validate_setup)
        self.voxel_spin.valueChanged.connect(self._validate_setup)
        self.final_voxel_spin.valueChanged.connect(self._validate_setup)
        self._capture_mode_changed()

    def _build_statusbar(self):
        self.statusBar().showMessage("Ready")
        self.fps_label = QLabel("FPS: --")
        self.kinect_label = QLabel("Kinect: connecting...")
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().addPermanentWidget(self.kinect_label)

    def _set_scan_controls_enabled(self, enabled: bool):
        if not enabled:
            self._paused = self._scanning or self._paused
            self.auto_capture_cb.setChecked(False)
        self._refresh_controls()

    def _camera_ready(self):
        return bool(self._camera_ok and self._last_rgb is not None and self._last_depth is not None
                    and time.monotonic() - self._last_frame_time <= 0.5)

    def _refresh_controls(self):
        connected = self.server_client.is_connected
        ready = self._camera_ready()
        busy = self._reset_pending or self._preview_pending or self._build_pending or bool(self._export_pending)
        busy = busy or self._connect_pending or self._final_preview_pending
        busy = busy or bool(self._server_operation)
        active = self._scanning
        frames = self._server_stored > 0 or getattr(self.task_worker, "queued_task_count", 0) > 0
        valid_setup = self.depth_near_spin.value() < self.depth_far_spin.value() and (
            self.final_voxel_spin.value() == 0 or self.final_voxel_spin.value() >= 2
        )
        self.btn_start_scan.setEnabled(connected and ready and valid_setup and not busy and (not active or self._paused))
        self.btn_start_scan.setText("New Scan" if self._session_id else "Start Scan")
        self.btn_pause.setEnabled(connected and ready and not busy and (active or frames))
        self.btn_pause.setText("Resume Capture" if self._paused or not active else "Pause")
        self.btn_capture.setEnabled(connected and ready and active and not self._paused and not busy)
        self.btn_stop_build.setEnabled(connected and frames and not busy)
        self.btn_stop_build.setText("Retry Build" if self._build_failed else "Finish Scan")
        self.btn_preview_scan.setEnabled(connected and (frames or self._has_mesh) and not busy)
        self.btn_export.setEnabled(connected and self._has_mesh and not busy)
        self.btn_export_session.setEnabled(connected and frames and not busy)
        for button in (self.btn_export_ply, self.btn_export_obj, self.btn_export_glb,
                       self.btn_export_texture_obj, self.btn_save_mesh):
            button.setEnabled(connected and self._has_mesh and not busy)
        self.settings_group.setEnabled(not active and not busy)
        self.capture_mode_combo.setEnabled(not busy)
        self.auto_capture_spin.setEnabled(not busy)
        self.auto_capture_cb.setEnabled(connected and active and not busy)
        if not connected:
            reason = "Connect the reconstruction server."
        elif not ready:
            reason = "Waiting for fresh color and depth frames from the Kinect."
        else:
            reason = "Space: pause/resume · C: capture in Manual mode"
        self.readiness_label.setText(reason)
        self.btn_start_scan.setToolTip(reason if not self.btn_start_scan.isEnabled() else "Begin a new capture session")
        self.btn_connect.setEnabled(not (self._connect_pending or self._reset_pending or self._build_pending
                                        or self._preview_pending or self._export_pending or self._final_preview_pending))
        self.server_ip_edit.setEnabled(not connected and not self._connect_pending)
        self.server_port_spin.setEnabled(not connected and not self._connect_pending)
        self.btn_preview_3d.setEnabled(bool(self._last_preview_path) and not busy)
        self._refresh_status()

    def _refresh_status(self):
        if self._server_operation:
            state = "Server is still building · waiting for completion" if self._server_operation == "build" else "Server is preparing inspection · waiting"
        elif self._connect_pending:
            state = "Connecting to reconstruction server…"
        elif self._reset_pending:
            state = "Starting scan…"
        elif self._build_pending:
            state = "Building final surface…"
        elif self._export_pending:
            state = "Saving captured session…" if self._export_pending["kind"] == "session" else "Exporting final model…"
        elif self._final_preview_pending:
            state = "Loading final model for inspection…"
        elif self._preview_pending:
            state = "Preparing inspection · capture temporarily paused"
        elif not self.server_client.is_connected:
            state = "Server disconnected · reconnect to continue" if self._session_id else "Waiting for server connection"
        elif self._build_failed:
            state = "Build failed · retry or resume capture"
        elif self._operation_error:
            state = self._operation_error
        elif self._scanning and self._paused:
            state = "Paused · scan retained"
        elif self._scanning and self._capture_waiting:
            state = self._capture_waiting
        elif self._scanning and not self._camera_ready():
            state = "Waiting for fresh camera frames · scan retained"
        elif self._scanning:
            state = "Capturing automatically" if self.auto_capture_cb.isChecked() else "Manual capture · ready"
        elif self._has_mesh:
            state = "Final model ready · inspect or export"
        elif self._server_stored:
            state = "Scan retained · resume capture or finish"
        else:
            state = "Ready to scan" if self._camera_ready() else "Waiting for camera"
        self.scan_status_label.setText(state)

    def _capture_mode_changed(self):
        automatic = self.capture_mode_combo.currentData() == "automatic"
        self.interval_row.setVisible(automatic)
        self.interval_help.setVisible(automatic)
        self.btn_capture.setVisible(not automatic)
        self.auto_capture_cb.setChecked(automatic and self._scanning)
        self._capture_waiting = ""
        self._reset_auto_capture_cadence()
        self._refresh_controls()

    def _pause_or_resume(self):
        if self._reset_pending or self._build_pending or self._preview_pending or self._export_pending or self._final_preview_pending or self._server_operation or self._connect_pending:
            return
        if not self.server_client.is_connected:
            return
        if self._scanning and not self._paused:
            self._paused = True
        elif self._camera_ready() and (self._scanning or self._server_stored):
            self._apply_session_settings(self._session_settings)
            self._scanning = True
            self._paused = False
            self._build_failed = False
            self._has_mesh = False
            self._last_preview_path = None
            self._operation_error = ""
            self.auto_capture_cb.setChecked(self.capture_mode_combo.currentData() == "automatic")
        self._capture_waiting = ""
        self._reset_auto_capture_cadence()
        self._refresh_controls()

    def _shortcut_pause(self):
        if not isinstance(QApplication.focusWidget(), (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)):
            self._pause_or_resume()

    def _shortcut_capture(self):
        if self.capture_mode_combo.currentData() == "manual" and not isinstance(
                QApplication.focusWidget(), (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)):
            self._capture_frame()

    def _validate_setup(self):
        self.final_voxel_spin.setMaximum(self.voxel_spin.value())
        error = ""
        if self.depth_near_spin.value() >= self.depth_far_spin.value():
            error = "Nearest surface must be closer than farthest surface."
        elif 0 < self.final_voxel_spin.value() < 2:
            error = "Final voxel size must be at least 2 mm, or use live resolution."
        self.settings_error_label.setText(error)
        self.settings_error_label.setStyleSheet("color: #b4382c;" if error else "")
        self._refresh_controls()
        if error:
            self.btn_start_scan.setEnabled(False)
        if self._last_depth is not None and self._mode == MODE_DEPTH:
            self._show_depth(self._last_depth)

    def _choose_export(self):
        if not self.btn_export.isEnabled():
            return
        dialog = ExportDialog(self)
        if dialog.exec():
            fmt = dialog.selected_format
            self.texture_exposure_cb.setChecked(dialog.texture_options.get("exposure_correction", False))
            self.texture_best_cb.setChecked(dialog.texture_options.get("blend_mode") == "best")
            if fmt in ("glb", "obj.zip"):
                self._export_texture(fmt)
            elif fmt == "ply":
                self._export_ply()
            else:
                self._export_obj()


    def _toggle_connection(self):
        if self._connect_pending or self._reset_pending or self._build_pending or self._preview_pending or self._export_pending or self._final_preview_pending:
            return
        if self.server_client.is_connected:
            self._paused = True
            self.auto_capture_cb.setChecked(False)
            self._connect_pending = True
            self.task_worker.submit(ServerTask(ServerTaskType.DISCONNECT))
        else:
            host = self.server_ip_edit.text().strip()
            if not host:
                self.server_status_label.setText("Enter a server host.")
                self.connection_section.toggle.setChecked(True)
                return
            self._connect_pending = True
            self._restore_on_status = True
            self._operation_error = ""
            self.server_status_label.setText("Connecting…")
            self.task_worker.submit(ServerTask(ServerTaskType.CONNECT, {
                "host": host, "port": self.server_port_spin.value(),
            }))
        self._refresh_controls()

    def _on_server_connected(self):
        self._connect_pending = False
        self._status_pending = False
        self.btn_connect.setText("Disconnect")
        self.server_status_label.setText("Connected")
        self.server_status_label.setStyleSheet("color: #0a0;")
        self._operation_error = ""
        self._progress_link_ok = True
        self.live_view.set_feedback_connected(True)
        self._set_scan_controls_enabled(True)
        self.statusBar().showMessage("Connected to reconstruction server", 4000)

    def _on_server_disconnected(self, reason: str):
        self._connect_pending = False
        self._status_pending = False
        self._server_operation = None
        self.live_view.set_feedback_connected(False)
        self.btn_connect.setText("Reconnect" if self._session_id else "Connect")
        self.server_status_label.setText(reason)
        self.server_status_label.setStyleSheet("color: #b4382c;")
        self._set_scan_controls_enabled(False)
        self.connection_section.toggle.setChecked(True)
        self.statusBar().showMessage(f"Server: {reason}")

    def _on_websocket_status(self, state, detail):
        if not self.server_client.is_connected or self._closing:
            return
        self._progress_link_ok = state == "connected"
        self.live_view.set_feedback_connected(self._progress_link_ok)
        if not self._progress_link_ok:
            self.guidance_label.setText("Live feedback disconnected; reconnecting. Pause movement until updates return.")
        elif self._session_id:
            self.guidance_label.setText(self.live_view.snapshot.get("guidance", "Move slowly with overlapping views."))
            if self._reconcile_on_status:
                self._poll_server_status()
        self.server_status_label.setText("Connected" if self._progress_link_ok else "Connected · live feedback reconnecting")
        self.statusBar().showMessage(detail, 5000)

    def _switch_mode(self, mode: str):
        self._mode = mode
        self.live_view.setVisible(mode == MODE_SCANNER and bool(self._session_id) and self.live_cb.isChecked())
        self.guidance_label.setVisible(not self.live_view.isVisible())
        if self.live_view.isVisible():
            self.splitter.setSizes([max(300, self.splitter.width() * 2 // 3), max(240, self.splitter.width() // 3)])
        self.camera_title.setText("Live camera · Depth" if mode == MODE_DEPTH else "Live camera · Color")
        self.depth_legend_label.setVisible(mode == MODE_DEPTH)
        if self._last_rgb is not None and self._last_depth is not None:
            self._show_depth(self._last_depth) if mode == MODE_DEPTH else self._show_rgb(self._last_rgb)
        for action in self._mode_actions:
            action.setChecked(action.data() == mode)

    # ── frame display ─────────────────────────────────────────────────
    def _on_frame(self, video: np.ndarray, depth: np.ndarray, metadata=None):
        if self._closing:
            return
        self._fps_counter += 1
        self._camera_ok = True
        self.kinect_label.setText("Kinect: live")
        self._last_rgb = video
        self._last_depth = depth
        self._frame_sequence += 1
        self._last_frame_metadata = dict(metadata or {})
        self._last_frame_time = self._last_frame_metadata.get(
            "captured_monotonic_s", time.monotonic()
        )
        # GUI sequence remains unique if the USB device reconnects mid-session.
        self._last_frame_metadata["frame_id"] = f"{self._capture_run_id}:{self._frame_sequence}"
        self._last_frame_metadata.setdefault("timestamp_s", time.time())

        if self._mode == MODE_RGB:
            self._show_rgb(video)
        elif self._mode == MODE_DEPTH:
            self._show_depth(depth)
        elif self._mode == MODE_SCANNER:
            self._show_scanner(video, depth)

        if (
            self._scanning
            and not self._preview_pending
            and not self._reset_pending
            and not self._export_pending
            and not self._paused
            and self.auto_capture_cb.isChecked()
        ):
            self._auto_frames_since_capture += 1
            if self._auto_frames_since_capture >= self.auto_capture_spin.value():
                self._auto_capture_tick()
        self._refresh_controls()

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
        depth = raw_depth_to_mm(depth, self._sensor_calibration)
        roi = self._selected_roi()
        near, far = self.depth_near_spin.value(), self.depth_far_spin.value()
        if self._scanning and self._session_settings:
            roi = self._session_settings.get("roi")
            near = self._session_settings.get("near_m", near / 1000) * 1000
            far = self._session_settings.get("far_m", far / 1000) * 1000
        self.depth_legend_label.setText(depth_legend_text(near, far))
        return colorize_depth(
            depth, near, far, roi=roi
        )

    def _show_depth(self, depth):
        self._set_pixmap(numpy_to_qimage(self._depth_display(depth)))

    def _show_scanner(self, rgb, depth):
        # The reconstruction gets the main view; camera remains useful at its own aspect ratio.
        self._show_rgb(rgb)

    def _set_pixmap(self, qimg):
        self.view_label.set_image(qimg)

    def _load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Kinect calibration", "", "JSON (*.json)"
        )
        if path:
            try:
                profile = load_calibration(path)
                self._sensor_calibration = profile
                self._camera = profile.depth
                self.calibration_label.setText(profile.name)
                self._restart_camera()
            except (ValueError, TypeError, OSError, RuntimeError) as exc:
                self.scan_status_label.setText(f"Invalid calibration: {exc}")

    # ── scanning workflow ─────────────────────────────────────────────
    def _start_scan(self, checked=False, *, protected=False):
        if not self.server_client.is_connected:
            QMessageBox.warning(self, "Not Connected", "Connect to the server first.")
            return

        if self._reset_pending or self._build_pending or self._preview_pending or self._export_pending or self._final_preview_pending or self._server_operation or self._connect_pending or (self._scanning and not self._paused and not protected):
            return
        if not self._camera_ready():
            self._capture_waiting = "Waiting for a fresh camera frame"
            self._refresh_controls()
            return
        try:
            voxel = self.voxel_spin.value() / 1000
            roi = self._selected_roi()
            settings = ScanSettings(
                camera=self._camera,
                sensor_calibration=self._sensor_calibration,
                rgb_mode=self.rgb_mode_combo.currentData(),
                near_m=self.depth_near_spin.value() / 1000,
                far_m=self.depth_far_spin.value() / 1000,
                voxel_m=voxel,
                truncation_m=min(0.2, max(0.04, voxel * 8)),
                final_weight=self.weight_spin.value(),
                color_recovery=self.color_tracking_cb.isChecked(),
                live_reconstruction=self.live_cb.isChecked(),
                refine_poses=self.refine_cb.isChecked(),
                relocalize=self.relocalize_cb.isChecked(),
                confidence_fusion=self.confidence_cb.isChecked(),
                final_voxel_m=self.final_voxel_spin.value() / 1000
                if self.final_voxel_spin.value()
                else None,
                final_block_count=self.final_blocks_spin.value(),
                roi=roi,
            )
        except ValueError as exc:
            self.scan_status_label.setText(str(exc))
            return
        if not protected and not self._protect_session("new_scan"):
            return
        self._operation_error = ""
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
        self._session_settings = result.get("settings")
        self._session_dirty = False
        self._capture_revision = self._saved_revision = 0
        self._pending_action = None
        self._has_mesh = False
        self.live_view.reset()
        self.live_view.setVisible(
            result.get("settings", {}).get("live_reconstruction", False)
        )
        self._last_capture_id = None
        self._reset_auto_capture_cadence()
        self._server_stored = 0
        self._server_integrated = 0

        self._last_preview_path = None
        self._final_preview_pending = False
        self._preview_session = None
        self._final_preview_session = None
        self._operation_error = ""
        self._scanning = True
        self._paused = False
        self._build_pending = self._build_failed = False
        self._capture_waiting = ""
        self._switch_mode(MODE_SCANNER)
        self.guidance_label.setVisible(not self.live_view.isVisible())
        self.auto_capture_cb.setChecked(self.capture_mode_combo.currentData() == "automatic")
        self.frame_count_label.setText("Captured: 0 · Added to model: 0")
        self._refresh_controls()
        self.statusBar().showMessage("Scan started", 4000)

    def _auto_capture_tick(self):
        if self._server_operation or self._reset_pending or self._export_pending or self._paused or self._connect_pending:
            return
        snapshot = self.live_view.snapshot
        if not self._progress_link_ok and self.live_cb.isChecked():
            self._capture_waiting = "Auto capture waiting for live feedback to reconnect"
            self._refresh_status()
            return
        if self.adaptive_capture_cb.isChecked() and (
            self.task_worker.queued_task_count >= 5
            or snapshot.get("pending_count", 0) >= 5
            or snapshot.get("pending_age_s", 0) > 2
        ):
            self._capture_waiting = "Auto capture waiting for reconstruction to catch up"
            self._refresh_status()
            return
        self._capture_waiting = ""
        self._capture_frame()

    def _capture_frame(self):
        if (
            not self._scanning
            or self._reset_pending
            or self._server_operation
            or self._connect_pending
            or self._paused
            or self._build_pending
            or self._export_pending
            or self._preview_pending
            or not self.server_client.is_connected
            or self._closing
        ):
            return
        if self._last_rgb is None or self._last_depth is None:
            self._capture_waiting = "No frame available yet"
            self._refresh_status()
            return

        frame_id = self._last_frame_metadata.get("frame_id")
        if (
            frame_id == self._last_capture_id
            or time.monotonic() - self._last_frame_time > 0.5
        ):
            self._capture_waiting = "Waiting for a fresh camera frame"
            self._refresh_status()
            return
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
            self._capture_waiting = "Upload queue full; skipped capture"
            self._refresh_status()
        else:
            self._last_capture_id = frame_id
            self._capture_revision += 1
            self._session_dirty = True
            self._operation_error = ""
            self._capture_waiting = ""
            self._reset_auto_capture_cadence()
            self._refresh_controls()

    def _reset_auto_capture_cadence(self):
        self._auto_frames_since_capture = 0

    def _toggle_auto_capture(self, checked: bool):
        self._reset_auto_capture_cadence()

    def _stop_and_build(self):
        if self._build_pending or self._preview_pending or self._export_pending or self._final_preview_pending or not self.server_client.is_connected:
            return
        if not self._server_stored and not getattr(self.task_worker, "queued_task_count", 0):
            return
        self._build_pending = True
        self._build_failed = False
        self._operation_error = ""
        self._last_preview_path = None
        self._scanning = False
        self._reset_auto_capture_cadence()
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
        self._refresh_controls()

    def _preview_scan(self):
        if self._has_mesh:
            if self._last_preview_path:
                self._preview_3d()
            else:
                self._request_final_preview()
            return
        if self._preview_pending or self._build_pending:
            return
        if self._server_stored == 0:
            QMessageBox.information(
                self, "Preview", "Capture at least one frame first."
            )
            return
        self._preview_pending = True
        self._preview_session = self._session_id
        self._operation_error = ""
        self._reset_auto_capture_cadence()
        self.btn_capture.setEnabled(False)
        self.btn_stop_build.setEnabled(False)
        self.btn_preview_scan.setEnabled(False)
        self.scan_status_label.setText("Generating preview on server...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self.task_worker.submit(ServerTask(ServerTaskType.PREVIEW))
        self._refresh_controls()

    def _preview_3d(self):
        if self._last_preview_path:
            launch_viewer_subprocess(self._last_preview_path)

    def _ensure_export_dir(self):
        os.makedirs(EXPORT_DIR, exist_ok=True)
        return EXPORT_DIR

    def _export_ply(self):
        self._export_mesh("ply")

    def _export_obj(self):
        self._export_mesh("obj")

    def _export_texture(self, fmt):
        self._export_mesh(fmt)

    def _export_mesh(self, fmt):
        if not self._has_mesh or self._export_pending:
            return
        extension = "zip" if fmt == "obj.zip" else fmt
        path, _ = QFileDialog.getSaveFileName(
            self, "Export final model", os.path.join(self._ensure_export_dir(), f"scan.{extension}"),
            f"{extension.upper()} files (*.{extension})",
        )
        if not path:
            return
        kwargs = {"path": path}
        task_type = {"ply": ServerTaskType.EXPORT_PLY, "obj": ServerTaskType.EXPORT_OBJ}.get(fmt, ServerTaskType.EXPORT_TEXTURE)
        if task_type == ServerTaskType.EXPORT_TEXTURE:
            kwargs.update(format=fmt, options={
                "exposure_correction": self.texture_exposure_cb.isChecked(),
                "blend_mode": "best" if self.texture_best_cb.isChecked() else "blend",
            })
        self._begin_export("mesh", path, ServerTask(task_type, kwargs))

    def _export_session(self, checked=False):
        if self._export_pending or not self.server_client.is_connected:
            return False
        path, _ = QFileDialog.getSaveFileName(
            self, "Save captured session", os.path.join(self._ensure_export_dir(), "scan-session.zip"),
            "ZIP files (*.zip)",
        )
        if not path:
            return False
        return self._begin_export("session", path, ServerTask(ServerTaskType.EXPORT_SESSION, {"path": path}))

    def _begin_export(self, kind, path, task):
        restore_capture = (self._scanning, self._paused)
        self._paused = True
        self._export_pending = {
            "kind": kind, "path": path, "revision": self._capture_revision,
            "restore_capture": restore_capture,
        }
        self._operation_error = ""
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        if not self.task_worker.submit(task):
            self._on_export_done(False, path)
            return False
        self._refresh_controls()
        return True

    def _protect_session(self, reason):
        if not self._session_dirty:
            return True
        if self._export_pending:
            self.statusBar().showMessage("Wait for the current save or export to finish.")
            return False
        restore_capture = (self._scanning, self._paused)
        was_paused = self._paused
        self._paused = True
        self._refresh_controls()
        dialog = SessionProtectionDialog(reason, self)
        dialog.exec()
        if dialog.choice == "discard":
            self._paused = was_paused
            return True
        if dialog.choice == "save":
            self._pending_action = reason
            if self._export_session():
                self._export_pending["restore_capture"] = restore_capture
                return False  # Continue only after the saved session is confirmed.
            self._pending_action = None
        self._paused = was_paused
        self._reset_auto_capture_cadence()
        self._refresh_controls()
        return False

    def _save_mesh(self):
        os.makedirs(MESH_DIR, exist_ok=True)
        filename = f"scan_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}.ply"
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
        if self._restore_on_status and "settings" in status:
            self._restore_on_status = False
            self._restore_server_session(status)
        elif self._reconcile_on_status and "settings" in status:
            self._status_pending = False
            self._server_operation = status.get("operation")
            if self._server_operation:
                QTimer.singleShot(1000, self._poll_server_status)
            else:
                self._reconcile_on_status = False
                self._server_stored = status.get("stored_count", self._server_stored)
                self._server_integrated = status.get("frame_count", self._server_integrated)
                if status.get("has_mesh"):
                    self._on_build_mesh_done(True, "Server completed the final model; ready to inspect or export.")
            self._refresh_controls()

    def _poll_server_status(self):
        if self._closing or self._status_pending or not self.server_client.is_connected:
            return
        self._status_pending = True
        self.task_worker.submit(ServerTask(ServerTaskType.STATUS))

    def _restore_server_session(self, status):
        session_id = status.get("session_id")
        same_session = session_id == self._session_id
        previous_stored = self._server_stored
        self._session_id = session_id if status.get("stored_count", 0) or status.get("has_mesh") else None
        self._session_settings = status.get("settings")
        self._server_stored = status.get("stored_count", 0)
        self._server_integrated = status.get("frame_count", 0)
        self._has_mesh = bool(status.get("has_mesh"))
        self._scanning = self._server_stored > 0 and not self._has_mesh
        self._paused = self._server_stored > 0
        self.auto_capture_cb.setChecked(False)
        if not same_session:
            self._capture_revision = self._server_stored
            self._saved_revision = 0
            self._session_dirty = self._server_stored > 0
            self._last_preview_path = None
            self.live_view.reset()
        elif self._server_stored != previous_stored:
            self._capture_revision += max(1, self._server_stored - previous_stored)
            self._session_dirty = True
        self._server_operation = status.get("operation")
        if self._server_operation:
            self._scanning = False
            self._reconcile_on_status = True
            QTimer.singleShot(1000, self._poll_server_status)
        if self._server_stored or self._has_mesh:
            self._apply_session_settings(self._session_settings)
        if self._server_stored or self._has_mesh:
            self._switch_mode(MODE_SCANNER)
        else:
            self._switch_mode(self._mode)
        self.frame_count_label.setText(f"Captured: {self._server_stored} · Added to model: {self._server_integrated}")
        self._capture_waiting = ""
        self._operation_error = ""
        self._refresh_controls()
        if self._has_mesh and not self._server_operation:
            self._request_final_preview()

    def _apply_session_settings(self, settings):
        if not settings:
            return
        try:
            profile = ScanSettings.from_dict(settings)
        except (ValueError, TypeError) as exc:
            self._operation_error = f"Cannot restore scan settings: {exc}"
            return
        controls = (self.depth_near_spin, self.depth_far_spin, self.voxel_spin,
                    self.final_voxel_spin, self.final_blocks_spin, self.weight_spin,
                    self.rgb_mode_combo, self.crop_cb, self.crop_spin, self.live_cb,
                    self.color_tracking_cb, self.refine_cb, self.relocalize_cb, self.confidence_cb)
        previous = [control.blockSignals(True) for control in controls]
        rgb_changed = self.rgb_mode_combo.currentData() != profile.rgb_mode
        calibration_changed = profile.sensor_calibration is not None and profile.sensor_calibration != self._sensor_calibration
        try:
            self.depth_near_spin.setValue(round(profile.near_m * 1000))
            self.depth_far_spin.setValue(round(profile.far_m * 1000))
            self.voxel_spin.setValue(profile.voxel_m * 1000)
            self.final_voxel_spin.setMaximum(profile.voxel_m * 1000)
            self.final_voxel_spin.setValue((profile.final_voxel_m or 0) * 1000)
            self.final_blocks_spin.setValue(profile.final_block_count)
            self.weight_spin.setValue(profile.final_weight)
            self.rgb_mode_combo.setCurrentIndex(self.rgb_mode_combo.findData(profile.rgb_mode))
            self.live_cb.setChecked(profile.live_reconstruction)
            self.color_tracking_cb.setChecked(profile.color_recovery)
            self.refine_cb.setChecked(profile.refine_poses)
            self.relocalize_cb.setChecked(profile.relocalize)
            self.confidence_cb.setChecked(profile.confidence_fusion)
            self.crop_cb.setChecked(profile.roi is not None)
            if profile.roi:
                self.crop_spin.setValue(round((profile.roi[2] - profile.roi[0]) / profile.camera.width * 100))
            if profile.sensor_calibration:
                self._sensor_calibration = profile.sensor_calibration
                self._camera = profile.camera
                self.calibration_label.setText(profile.sensor_calibration.name)
        finally:
            for control, blocked in zip(controls, previous):
                control.blockSignals(blocked)
        self.crop_spin.setEnabled(self.crop_cb.isChecked())
        self.auto_capture_spin.set_fps(RGB_MODE_FPS[profile.rgb_mode])
        if rgb_changed or calibration_changed:
            try:
                self._restart_camera()
            except RuntimeError as exc:
                self._operation_error = str(exc)
        self._validate_setup()

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
            f"Captured: {self._server_stored} · Added to model: {self._server_integrated}"
        )
        self.guidance_label.setText(snapshot.get("guidance") or "Move slowly with overlapping views.")
        self._refresh_controls()

    def _on_frame_stored(self, result: dict):
        if result.get("session_id") and result["session_id"] != self._session_id:
            return
        if not result.get("success"):
            self._paused = True
            self._operation_error = result.get("message", "Capture rejected; scan retained")
        self._server_stored = max(
            self._server_stored, result.get("stored_count", self._server_stored)
        )
        self.frame_count_label.setText(
            f"Captured: {self._server_stored} · Added to model: {self._server_integrated}"
        )
        self.btn_export_session.setEnabled(self._server_stored > 0)
        self._refresh_controls()
        self.statusBar().showMessage(result.get("message", "Frame stored"), 3000)

    def _on_process_progress(self, current: int, total: int, result: dict):
        if result.get("session_id") and result["session_id"] != self._session_id:
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_bar.setVisible(self._build_pending or self._preview_pending)
        self.progress_bar.setFormat("Processing frames: %v / %m")
        if current == total and self._build_pending:
            self.progress_bar.setRange(0, 0)
        self._server_integrated = result.get("frame_count", self._server_integrated)
        self.frame_count_label.setText(
            f"Captured: {self._server_stored} · Added to model: {self._server_integrated}"
        )
        self.statusBar().showMessage(result.get("message", "Processing captured frames"), 3000)
        self._refresh_status()

    def _on_build_mesh_done(self, success: bool, detail: str):
        self._build_pending = False
        self._build_failed = not success
        self._has_mesh = success
        self._scanning = False
        self._paused = True
        self.progress_bar.hide()
        self._operation_error = "" if success else detail
        self.statusBar().showMessage(detail, 8000)
        self._refresh_controls()
        if success and not self._pending_action:
            self._request_final_preview()

    def _request_final_preview(self):
        if not self._has_mesh or self._final_preview_pending or self._export_pending:
            return
        self._final_preview_pending = True
        self._final_preview_session = self._session_id
        self._last_preview_path = None
        self._operation_error = ""
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.task_worker.submit(ServerTask(ServerTaskType.FINAL_PREVIEW))
        self._refresh_controls()

    def _on_final_preview_done(self, path):
        if self._closing or self._final_preview_session != self._session_id or not self._has_mesh:
            return
        self._final_preview_pending = False
        self._last_preview_path = path
        self.progress_bar.hide()
        self._refresh_controls()
        if path and not self._pending_action:
            launch_viewer_subprocess(path)

    def _resume_capture(self):
        self._preview_pending = False
        if self._scanning:
            self.btn_capture.setEnabled(True)
            self.btn_stop_build.setEnabled(True)
            self.btn_preview_scan.setEnabled(True)
            self._toggle_auto_capture(self.auto_capture_cb.isChecked())
        self._refresh_controls()

    def _on_preview_done(self, path: str):
        if self._closing or self._preview_session != self._session_id:
            return
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
        self._refresh_controls()

    def _on_export_done(self, success: bool, path: str):
        pending = self._export_pending
        if pending and pending["path"] != path:
            return
        self._export_pending = None
        self.progress_bar.hide()
        action, self._pending_action = self._pending_action, None
        if success:
            if pending and pending["kind"] == "session":
                self._saved_revision = pending["revision"]
                self._session_dirty = self._capture_revision != self._saved_revision
            self.statusBar().showMessage(f"Saved: {path}", 10000)
            self._operation_error = ""
        else:
            self._operation_error = "Save failed · scan retained; choose Save Session to retry"
            self.statusBar().showMessage(f"Could not save {path}; current scan is retained.")
        if pending and (not action or not success):
            self._scanning, self._paused = pending["restore_capture"]
            self._reset_auto_capture_cadence()
        self._refresh_controls()
        if success and action == "new_scan":
            self._start_scan(protected=True)
        elif success and action == "close":
            self._close_approved = True
            self.close()

    def _on_save_mesh_done(self, success: bool, path: str):
        self.statusBar().showMessage(f"Saved: {path}" if success else f"Save failed: {path}", 10000)
        self._refresh_controls()

    def _on_task_started(self, msg: str):
        self.statusBar().showMessage(msg)

    def _on_task_error(self, msg: str):
        # Recording/report warnings are independent of build or inspection success.
        self.statusBar().showMessage(msg, 10000)

    def _on_task_failed(self, task_type, message):
        self.progress_bar.hide()
        self._operation_error = f"{message} · current scan retained"
        if task_type == "CONNECT":
            self._connect_pending = False
            self.connection_section.toggle.setChecked(True)
        elif task_type == "RESET":
            self._reset_pending = False
        elif task_type == "BUILD_MESH":
            self._build_pending = False
            self._build_failed = True
            self._reconcile_on_status = True
            self._poll_server_status()
        elif task_type == "PREVIEW":
            self._resume_capture()
        elif task_type == "FINAL_PREVIEW":
            self._final_preview_pending = False
        elif task_type in ("EXPORT_PLY", "EXPORT_OBJ", "EXPORT_TEXTURE", "EXPORT_SESSION"):
            if self._export_pending:
                self._on_export_done(False, self._export_pending["path"])
        elif task_type == "SEND_FRAME":
            self._paused = True
        elif task_type == "STATUS":
            self._status_pending = False
            if self._reconcile_on_status:
                QTimer.singleShot(3000, self._poll_server_status)
        self._refresh_controls()


    def _update_fps(self):
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed > 0:
            self._fps_value = self._fps_counter / elapsed
        self._fps_counter = 0
        self._last_fps_time = now
        self.fps_label.setText(f"Camera: {self._fps_value:.1f} fps")
        if self._last_rgb is not None and not self._camera_ready():
            self.view_label.set_stale(True, "Camera delayed · last image")
            self.kinect_label.setText("Kinect: waiting for frames")
        self._refresh_controls()

    def _on_error(self, msg: str):
        if self._closing:
            return
        self._camera_ok = False
        self.kinect_label.setText("Kinect: unavailable")
        self.view_label.set_stale(True, "Camera unavailable · reconnecting")
        self._refresh_controls()
        if self._last_rgb is None:
            self.view_label.setText(msg)
        self.statusBar().showMessage(f"Error: {msg}")

    def closeEvent(self, event):
        if not self._closing and not self._close_approved and not self._protect_session("close"):
            event.ignore()
            return
        if not self._closing:
            self._closing = True
            self._reset_auto_capture_cadence()
            self._fps_timer.stop()
            self.worker.stop()
            self.task_worker.stop()
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
        self.server_client.disconnect(notify=False)
        super().closeEvent(event)
