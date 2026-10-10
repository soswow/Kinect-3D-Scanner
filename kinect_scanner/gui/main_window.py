"""MainWindow — primary application window (client/server mode)."""

import hashlib
import json
import logging
import os
import re
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
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from shared.calibration import raw_depth_to_mm
from shared.capture import RGB_GAIN_CHOICES, RGB_MODE_FPS
from shared.inertial import OrientationTracker, calibration_profile, rotate_display
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings

from ..capture_pacing import CapturePacer
from ..capture_selection import CaptureSelector
from ..config import MODE_DEPTH, MODE_RGB, MODE_SCANNER
from ..runtime import data_root, export_root
from ..server_client import ServerClient
from ..server_task_worker import ServerTask, ServerTaskType, ServerTaskWorker
from ..viewer import launch_viewer_subprocess
from ..worker import KinectWorker
from .components import CameraPreview, CollapsibleSection
from .apriltags import AprilTagDictionaries
from .dialogs import ExportDialog, SessionProtectionDialog
from .feedback import CaptureSound
from .live_view import LiveView
from .logs import LogsPanel
from .preferences import ScannerPreferences
from .tracking_debug import flow_image, flow_summary
from .widgets import (
    FrameIntervalSpinBox,
    colorize_depth,
    depth_legend_text,
    numpy_to_qimage,
)

# Bundled code/resources are read-only; keep generated files in user folders.
_PROJECT_ROOT = str(data_root())
EXPORT_DIR = str(export_root())
MESH_DIR = os.path.join(_PROJECT_ROOT, "mesh")
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, *, preferences=None):
        super().__init__()
        self.preferences = preferences if preferences is not None else ScannerPreferences()
        self.setWindowTitle("Kinect 3D Scanner")
        self.setMinimumSize(960, 600)

        self._closing = False
        self._mode = MODE_RGB
        self._scanning = False
        self._paused = False
        self._build_pending = False
        self._build_failed = False
        self._camera_ok = False
        self._camera_suspended = False
        self._start_when_camera_ready = False
        self._capture_waiting = ""
        self._session_dirty = False
        self._capture_revision = 0
        self._saved_revision = 0
        self._capture_run_id = uuid.uuid4().hex[:12]
        self._pending_action = None
        self._export_pending = None
        self._project_path = None
        self._project_to_open = None
        self._connect_pending = False
        self._restore_on_status = False
        self._session_settings = None
        self._final_preview_pending = False
        self._final_preview_session = None
        self._preview_session = None
        self._close_approved = False
        self._operation_error = ""
        self._progress_link_ok = True
        self._camera_motion_unverified = False
        self._server_operation = None
        self._reconcile_on_status = False
        self._status_pending = False
        self._fps_counter = 0
        self._fps_value = 0.0
        self._last_fps_time = time.time()
        self._last_rgb = None
        self._last_depth = None
        self._sensor_calibration = self.preferences.load_calibration(load_calibration())
        self._camera = self._sensor_calibration.depth
        self._frame_sequence = 0
        self._last_frame_metadata = {}
        self._last_tracking_debug = None
        self._orientation = OrientationTracker()
        self._display_rotation = 0
        self._sensor_recording_path = None
        self._sensor_counts_seen = {}
        try:
            saved_motion_calibration = self.preferences.read("camera/accelerometer_calibration", None)
            self._accelerometer_calibration = calibration_profile(saved_motion_calibration) if saved_motion_calibration else None
        except (ValueError, TypeError, KeyError):
            self._accelerometer_calibration = None
        self._last_frame_time = 0
        self._last_capture_id = None
        self._auto_frames_since_capture = 0
        self._capture_pacer = CapturePacer()
        self._capture_selector = CaptureSelector()
        self._reset_pending = False
        self._cancel_pending = False
        self._reset_to_setup_pending = False
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
        self.server_client.project_opened.connect(self._on_project_opened)
        self.server_client.transfer_progress.connect(self._on_transfer_progress)
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
        self.capture_sound = CaptureSound(self, settings=self.preferences.settings)
        self._build_toolbar()
        self._build_statusbar()
        self._build_dock()
        self._restore_preferences()

        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

        self._start_camera()

        # Disable scan controls until server connected
        self._set_scan_controls_enabled(False)
        if os.environ.get("KINECT_AUTOCONNECT") == "1":
            QTimer.singleShot(0, self._toggle_connection)

    def _restore_preferences(self):
        preferences = self.preferences
        with preferences.suspend():
            preferences.bind(self.rgb_mode_combo, "scan/rgb_mode")
            self.auto_capture_spin.set_fps(RGB_MODE_FPS[self.rgb_mode_combo.currentData()])
            preferences.bind(self.auto_capture_spin, "capture/interval_s")
            preferences.bind(self.voxel_spin, "scan/voxel_mm")
            self.final_voxel_spin.setMaximum(self.voxel_spin.value())
            for widget, key in (
                (self.capture_mode_combo, "capture/mode"),
                (self.depth_near_spin, "scan/near_mm"),
                (self.depth_far_spin, "scan/far_mm"),
                (self.crop_cb, "scan/crop_enabled"),
                (self.crop_spin, "scan/crop_percent"),
                (self.record_cb, "scan/record"),
                (self.full_camera_recording_cb, "scan/record_full_camera_streams"),
                (self.orientation_combo, "camera/orientation"),
                (self.gravity_tracking_cb, "scan/gravity_assistance"),
                (self.final_voxel_spin, "scan/final_voxel_mm"),
                (self.weight_spin, "scan/final_weight"),
                (self.live_cb, "scan/live_reconstruction"),
                (self.color_tracking_cb, "scan/color_tracking"),
                (self.apriltag_tracking_cb, "scan/apriltag_tracking"),
                (self.refine_cb, "scan/refine_poses"),
                (self.bundle_cb, "scan/bundle_adjustment"),
                (self.reconnect_fragments_cb, "scan/reconnect_fragments"),
                (self.offline_registration_combo, "scan/offline_registration"),
                (self.relocalize_cb, "scan/relocalize"),
                (self.confidence_cb, "scan/confidence"),
                (self.rgb_exposure_combo, "camera/exposure_mode"),
                (self.rgb_shutter_spin, "camera/shutter_speed"),
                (self.rgb_gain_combo, "camera/gain"),
                (self.flow_debug_cb, "debug/tracking_flow"),
                (self.flow_windows_cb, "debug/tracking_windows"),
            ):
                preferences.bind(widget, key)
            try:
                self.apriltag_dictionaries.set_dictionaries(preferences.read(
                    "scan/apriltag_dictionaries", ["DICT_APRILTAG_36h11"]))
            except ValueError:
                pass
            self.apriltag_dictionaries.changed.connect(lambda: preferences.write(
                "scan/apriltag_dictionaries", self.apriltag_dictionaries.dictionaries()))
            preferences.bind(self.server_ip_edit, "connection/host",
                             restore="KINECT_SERVER_HOST" not in os.environ)
            preferences.bind(self.server_port_spin, "connection/port",
                             restore="KINECT_SERVER_PORT" not in os.environ)
            self.crop_spin.setEnabled(self.crop_cb.isChecked())
            self._update_exposure_controls()
            self._update_offline_registration_controls()
            self._capture_mode_changed()
            self._validate_setup()
            self._update_tracking_debug()

    def _start_camera(self):
        self.worker = KinectWorker(
            rgb_mode=self.rgb_mode_combo.currentData(),
            rgb_exposure_mode=self.rgb_exposure_combo.currentData(),
            rgb_shutter_speed=self.rgb_shutter_spin.value(),
            rgb_gain=self.rgb_gain_combo.currentData(),
            accelerometer_calibration=self._accelerometer_calibration,
        )
        self._camera_configuration = self._selected_camera_configuration()
        worker = self.worker
        # Ignore any queued observation from the old worker after a mode change.
        def received(*args):
            if worker is self.worker and not self._camera_suspended:
                self._on_frame(*args)
        if hasattr(worker, "take_latest_frame"):
            worker.coalesce_frames = True
            def received_latest():
                frame = worker.take_latest_frame()
                if frame is not None:
                    received(*frame)
            worker.frame_available.connect(received_latest)
        elif hasattr(worker, "frame_pair_ready"):
            worker.frame_pair_ready.connect(received)
        else:
            worker.frame_ready.connect(received)
        self.worker.error_occurred.connect(
            lambda message: self._on_error(message) if worker is self.worker and not self._camera_suspended else None)
        self.worker.finished.connect(self._refresh_controls)
        if hasattr(self.worker, "sensor_recording_status"):
            self.worker.sensor_recording_status.connect(
                lambda status: self._on_sensor_recording_status(status) if worker is self.worker else None)
        self.worker.start()
        self._configure_camera_tracking()
        self._update_tracking_debug()

    def _restart_camera(self):
        if self._camera_suspended:
            return  # Changing setup or restoring a project must not reopen USB after Finish.
        self.worker.stop()
        if not self.worker.wait(2500):
            raise RuntimeError("Camera did not stop within 2.5 seconds")
        self._last_rgb = self._last_depth = None
        self._camera_ok = False
        self._set_camera_stale("Camera restarting…")
        self._last_frame_metadata = {}
        self._reset_auto_capture_cadence()
        self._start_camera()

    def _stop_camera_after_finish(self):
        if self._camera_suspended:
            return
        self._camera_suspended = True
        self._start_when_camera_ready = False
        self._camera_ok = False
        self._last_rgb = self._last_depth = self._last_tracking_debug = None
        self._last_frame_metadata = {}
        self._capture_selector.clear()
        self._fps_value = self._fps_counter = 0
        self.fps_label.setText("Camera: off")
        self.kinect_label.setText("Kinect: stopped")
        self.camera_title.setText("Camera off")
        self.sensor_status_label.setText("Sensor stopped")
        self.sensor_recording_label.setText("Sensor recording stopped")
        for view in (self.view_label, self.scan_depth_view):
            view.show_stopped()
        self.scan_depth_panel.hide()
        self.depth_legend_label.hide()
        logger.info("Capture finished; shutting down camera session=%s", self._session_id)
        if hasattr(self.worker, "finish_capture"):
            self.worker.finish_capture(self._sensor_recording_path)
        else:
            self.worker.stop()

    def _resume_camera(self):
        if not self._camera_suspended:
            return
        self._camera_suspended = False
        self.kinect_label.setText("Kinect: connecting…")
        self.camera_title.setText("Live camera · Color")
        self._set_camera_stale("Camera starting…")
        self._switch_mode(self._mode)
        self._start_camera()

    def _change_rgb_mode(self):
        self.auto_capture_spin.set_fps(RGB_MODE_FPS[self.rgb_mode_combo.currentData()])
        self._update_exposure_controls()
        self._reset_auto_capture_cadence()
        try:
            self._restart_camera()
        except RuntimeError as exc:
            self.scan_status_label.setText(str(exc))

    def _selected_camera_configuration(self):
        return (self.rgb_mode_combo.currentData(), self.rgb_exposure_combo.currentData(),
                self.rgb_shutter_spin.value(), self.rgb_gain_combo.currentData())

    def _update_offline_registration_controls(self):
        depth = self.offline_registration_combo.currentData() == "depth"
        self.reconnect_fragments_cb.setEnabled(not depth)
        self.refine_cb.setEnabled(not depth)
        self.bundle_cb.setEnabled(not depth)

    def _update_exposure_controls(self):
        previous_speed = self.rgb_shutter_spin.value()
        blocked = self.rgb_shutter_spin.blockSignals(True)
        try:
            self.rgb_shutter_spin.setMinimum(RGB_MODE_FPS[self.rgb_mode_combo.currentData()])
        finally:
            self.rgb_shutter_spin.blockSignals(blocked)
        if self.rgb_shutter_spin.value() != previous_speed:
            self.preferences.write("camera/shutter_speed", self.rgb_shutter_spin.value())
        manual = self.rgb_exposure_combo.currentData() == "manual"
        self.rgb_shutter_spin.setEnabled(manual)
        self.rgb_shutter_label.setEnabled(manual)
        self.rgb_gain_combo.setEnabled(manual)
        self.rgb_gain_label.setEnabled(manual)

    def _change_rgb_exposure(self):
        self._update_exposure_controls()
        if not hasattr(self, "worker"):
            return
        if self._selected_camera_configuration() == self._camera_configuration:
            return
        try:
            self.rgb_exposure_status_label.setText("Applying exposure settings…")
            self._restart_camera()
        except RuntimeError as exc:
            self.rgb_exposure_status_label.setText(str(exc))

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
        self.guidance_label.hide()
        self.camera_panel = QWidget()
        camera_layout = QVBoxLayout(self.camera_panel)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_title = QLabel("Live camera · Color")
        camera_layout.addWidget(self.camera_title)
        # The orientation status is placed beside its selector in scan settings.
        self.sensor_status_label = QLabel("Orientation: waiting for acceleration", self.camera_panel)
        self.sensor_status_label.setWordWrap(True)
        self.sensor_status_label.hide()
        self.sensor_recording_label = QLabel(self.camera_panel)
        self.sensor_recording_label.setWordWrap(True)
        camera_layout.addWidget(self.sensor_recording_label)
        self.sensor_recording_label.hide()
        self.view_label = CameraPreview()
        camera_layout.addWidget(self.view_label, stretch=1)
        self.scan_depth_panel = QWidget()
        scan_depth_layout = QVBoxLayout(self.scan_depth_panel)
        scan_depth_layout.setContentsMargins(0, 0, 0, 0)
        scan_depth_layout.addWidget(QLabel("Live camera · Depth"))
        self.scan_depth_view = CameraPreview()
        scan_depth_layout.addWidget(self.scan_depth_view, stretch=1)
        self.scan_depth_panel.hide()
        camera_layout.addWidget(self.scan_depth_panel, stretch=1)
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
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.splitter)
        self.logs_panel = LogsPanel()
        self.view_stack.addWidget(self.logs_panel)
        layout.addWidget(self.view_stack, stretch=1)
        self.view_label.setToolTip(self.guidance_label.text())
        self.camera_title.hide()

    def _build_toolbar(self):
        menu = self.menuBar().addMenu("File")
        self.open_project_action = menu.addAction("Open Project…", self._open_project)
        self.open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        self.save_project_action = menu.addAction("Save Project", self._save_project)
        self.save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_project_as_action = menu.addAction("Save Project As…", self._export_session)
        self.save_project_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        toolbar = QToolBar("Views")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        group = QActionGroup(self)
        group.setExclusive(True)
        self._mode_actions = []
        for title, mode in (("Scan", MODE_SCANNER), ("Color", MODE_RGB), ("Depth", MODE_DEPTH), ("Logs", "logs")):
            action = QAction(title, self)
            action.setData(mode)
            action.setCheckable(True)
            action.setChecked(mode == self._mode)
            action.triggered.connect(lambda checked, m=mode: self._switch_mode(m))
            group.addAction(action)
            toolbar.addAction(action)
            self._mode_actions.append(action)
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
        interval_label = QLabel("Minimum capture interval")
        interval_label.setBuddy(self.auto_capture_spin)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.auto_capture_spin)
        layout.addWidget(self.interval_row)
        self.interval_help = QLabel("Selects a sharp recent frame; slows to match live reconstruction.", container)
        self.interval_help.setWordWrap(True)
        self.interval_help.hide()
        interval_label.setToolTip(self.interval_help.text())
        self.interval_label = interval_label
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
        self.btn_cancel_scan = QPushButton("Cancel Scan")
        self.btn_cancel_scan.setToolTip("Return to setup without building a mesh; offer to save unsaved captures")
        self.btn_cancel_scan.clicked.connect(self._cancel_scan)
        finish_row.addWidget(self.btn_preview_scan)
        finish_row.addWidget(self.btn_stop_build)
        finish_row.addWidget(self.btn_cancel_scan)
        layout.addLayout(finish_row)
        self.btn_reset_scan = QPushButton("Reset Scan")
        self.btn_reset_scan.setToolTip("Clear the current scan and return to setup without starting capture; offer to save unsaved work")
        self.btn_reset_scan.clicked.connect(self._reset_scan)
        layout.addWidget(self.btn_reset_scan)
        output_row = QHBoxLayout()
        self.btn_export = QPushButton("Export…")
        self.btn_export.clicked.connect(self._choose_export)
        self.btn_export_session = QPushButton("Save Project…")
        self.btn_export_session.setToolTip("Save captures, calibration, reconstruction and final model in a reopenable ZIP project.")
        self.btn_export_session.clicked.connect(self._save_project)
        self.btn_open_project = QPushButton("Open Project…")
        self.btn_open_project.clicked.connect(self._open_project)
        output_row.addWidget(self.btn_export)
        output_row.addWidget(self.btn_export_session)
        layout.addWidget(self.btn_open_project)
        layout.addLayout(output_row)
        self.progress_status_label = QLabel()
        self.progress_status_label.setWordWrap(True)
        self.progress_status_label.hide()
        layout.addWidget(self.progress_status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.readiness_label = QLabel("Connect the server and wait for live camera frames.", container)
        self.readiness_label.setWordWrap(True)
        self.readiness_label.hide()
        # Old command entry points remain for API/pipeline compatibility, without UI duplication.
        for name, title, handler in (
            ("btn_export_ply", "Export PLY", self._export_ply),
            ("btn_export_obj", "Export OBJ", self._export_obj),
            ("btn_export_glb", "Export textured GLB", lambda: self._export_texture("glb")),
            ("btn_export_texture_obj", "Export textured OBJ", lambda: self._export_texture("obj.zip")),
            ("btn_preview_3d", "View snapshot", self._preview_3d),
            ("btn_save_mesh", "Save Mesh", self._save_mesh),
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
        self.record_cb = QCheckBox("Also keep selected captures locally")
        self.record_cb.setToolTip("Save Session keeps selected reconstruction images and the accelerometer log. This also keeps a separate local copy of the selected images.")
        vg.addWidget(self.record_cb)
        self.full_camera_recording_cb = QCheckBox("Record all camera frames (large files)")
        self.full_camera_recording_cb.setToolTip("Optional research recording of every RGB/depth frame: about 3.5 GB/min at high resolution, or 2.8 GB/min at VGA. Leave off for selected images plus the small accelerometer log.")
        vg.addWidget(self.full_camera_recording_cb)
        self.orientation_combo = QComboBox()
        for label, mode in (("Auto", "auto"), ("Landscape lock", "landscape"),
                            ("Portrait left lock", "portrait_left"), ("Portrait right lock", "portrait_right")):
            self.orientation_combo.addItem(label, mode)
        self.orientation_combo.setToolTip("Auto follows gravity. Lock portrait when looking up/down or while recording.")
        self.orientation_combo.currentIndexChanged.connect(self._change_orientation)
        orientation_label = QLabel("Camera orientation")
        orientation_label.setBuddy(self.orientation_combo)
        vg.addWidget(orientation_label)
        vg.addWidget(self.orientation_combo)
        vg.addWidget(self.sensor_status_label)
        self.sensor_status_label.show()
        self.gravity_tracking_cb = QCheckBox("Use accelerometer to assist tracking")
        self.gravity_tracking_cb.setToolTip("Optional gravity assistance for the RGB-D motion prediction. Auto portrait orientation works independently of this setting.")
        vg.addWidget(self.gravity_tracking_cb)
        self.settings_error_label = QLabel()
        self.settings_error_label.setWordWrap(True)
        vg.addWidget(self.settings_error_label)

        self.rgb_camera_section = CollapsibleSection("RGB camera")
        cv = self.rgb_camera_section.content_layout
        self.rgb_mode_combo = QComboBox()
        self.rgb_mode_combo.addItem("Color detail · 1280 × 1024, 10 fps", "rgb_high_res")
        self.rgb_mode_combo.addItem("Motion detail · 640 × 480, 30 fps", "rgb_low_res")
        self.rgb_mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.rgb_mode_combo.setMinimumContentsLength(18)
        self.rgb_mode_combo.currentIndexChanged.connect(self._change_rgb_mode)
        label = QLabel("Resolution and frame rate")
        label.setBuddy(self.rgb_mode_combo)
        cv.addWidget(label)
        cv.addWidget(self.rgb_mode_combo)
        self.rgb_exposure_combo = QComboBox()
        self.rgb_exposure_combo.addItem("Auto exposure", "auto")
        self.rgb_exposure_combo.addItem("Manual exposure", "manual")
        label = QLabel("Exposure")
        label.setBuddy(self.rgb_exposure_combo)
        cv.addWidget(label)
        cv.addWidget(self.rgb_exposure_combo)
        self.rgb_shutter_spin = QSpinBox()
        self.rgb_shutter_spin.setRange(10, 10000)
        self.rgb_shutter_spin.setValue(125)
        self.rgb_shutter_spin.setPrefix("1/")
        self.rgb_shutter_spin.setSuffix(" s")
        self.rgb_shutter_spin.setKeyboardTracking(False)
        self.rgb_shutter_spin.setToolTip("A larger denominator means a faster shutter and less motion blur. Shorter exposures need more light.")
        self.rgb_shutter_label = QLabel("Shutter speed")
        self.rgb_shutter_label.setBuddy(self.rgb_shutter_spin)
        cv.addWidget(self.rgb_shutter_label)
        cv.addWidget(self.rgb_shutter_spin)
        self.rgb_gain_combo = QComboBox()
        for gain in RGB_GAIN_CHOICES:
            self.rgb_gain_combo.addItem(f"{gain}×", gain)
        self.rgb_gain_combo.setToolTip("Higher gain brightens the image and increases noise.")
        self.rgb_gain_label = QLabel("Sensitivity (gain)")
        self.rgb_gain_label.setBuddy(self.rgb_gain_combo)
        cv.addWidget(self.rgb_gain_label)
        cv.addWidget(self.rgb_gain_combo)
        self.rgb_exposure_status_label = QLabel("Waiting for camera…")
        self.rgb_exposure_status_label.setWordWrap(True)
        cv.addWidget(self.rgb_exposure_status_label)
        self.rgb_exposure_combo.currentIndexChanged.connect(self._change_rgb_exposure)
        self.rgb_shutter_spin.valueChanged.connect(self._change_rgb_exposure)
        self.rgb_gain_combo.currentIndexChanged.connect(self._change_rgb_exposure)
        self._update_exposure_controls()
        vg.addWidget(self.rgb_camera_section)

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
        self.final_voxel_spin.setSpecialValueText("Use fusion resolution")
        self.final_voxel_spin.setToolTip("Controls reconstruction detail. Storage is allocated automatically for the scanned surface.")
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.5, 20)
        self.weight_spin.setValue(2)
        self.weight_spin.setSingleStep(0.5)
        for title, control in (("Fusion voxel size", self.voxel_spin), ("Final voxel size", self.final_voxel_spin), ("Final surface confidence", self.weight_spin)):
            label = QLabel(title)
            label.setBuddy(control)
            av.addWidget(label)
            av.addWidget(control)
        self.live_cb = QCheckBox("Show live reconstruction")
        self.live_cb.setChecked(False)
        av.addWidget(self.live_cb)
        self.offline_registration_combo = QComboBox()
        self.offline_registration_combo.addItem("Depth geometry (experimental)", "depth")
        self.offline_registration_combo.addItem("Existing fragment registration", "fragments")
        self.offline_registration_combo.setToolTip(
            "Depth geometry estimates camera positions from all saved depth captures at Finish. "
            "It does not require color or live tracking. Separate components are retained when their connection is unknown."
        )
        final_registration_label = QLabel("Final registration")
        final_registration_label.setBuddy(self.offline_registration_combo)
        av.addWidget(final_registration_label)
        av.addWidget(self.offline_registration_combo)
        self.reconnect_fragments_cb = QCheckBox("Reconnect separated views (fragment mode)")
        self.reconnect_fragments_cb.setChecked(True)
        self.reconnect_fragments_cb.setToolTip(
            "Search retained frames for overlapping fragments, verify links, and rebuild the connected scan. "
            "Adds processing time; views without a verified connection remain in the saved session."
        )
        av.addWidget(self.reconnect_fragments_cb)
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
        self.bundle_cb = QCheckBox("Joint RGB-D refinement at Finish")
        self.relocalize_cb = QCheckBox("Recover lost tracking")
        self.confidence_cb = QCheckBox("Use sensor confidence")
        for control, help_text in (
            (self.color_tracking_cb, "Tracks motion between camera frames while capturing, including when live reconstruction is off. Verifies color/depth matches against nearby saved views."),
            (self.refine_cb, "Validates loop matches and rebuilds fusion; needs extra time and memory."),
            (self.bundle_cb, "Refines camera positions and shared surface features together using color and measured depth. Adds processing time and memory; retains the current reconstruction if evidence is insufficient."),
            (self.relocalize_cb, "Attempts verified recovery after skipped frames; repeated scenes may be ambiguous."),
            (self.confidence_cb, "Weights depth using range, angle and edges; may require more observations."),
        ):
            control.setToolTip(help_text)
            ev.addWidget(control)
        vg.addWidget(experimental)
        apriltags = CollapsibleSection("AprilTag tracking")
        self.apriltag_tracking_cb = QCheckBox("Use AprilTags to assist tracking")
        self.apriltag_tracking_cb.setToolTip(
            "Detect all selected dictionaries in every camera frame and saved view. "
            "Static tags with valid measured depth provide extra camera-motion evidence. "
            "Tags are optional; color-assisted tracking also works without them."
        )
        apriltags.content_layout.addWidget(self.apriltag_tracking_cb)
        self.apriltag_dictionaries = AprilTagDictionaries()
        apriltags.content_layout.addWidget(self.apriltag_dictionaries)
        tag_help = QLabel(
            "Add every dictionary used by your printed labels. Different families can "
            "share an ID; each ID within one family must identify a single static label. "
            "No printed size is needed. Use clear tags with valid depth, at least 20 pixels per side."
        )
        tag_help.setWordWrap(True)
        apriltags.content_layout.addWidget(tag_help)
        vg.addWidget(apriltags)
        self.offline_registration_combo.currentIndexChanged.connect(self._update_offline_registration_controls)
        self._update_offline_registration_controls()
        motion_calibration_btn = QPushButton("Load Accelerometer Calibration…")
        motion_calibration_btn.clicked.connect(self._load_accelerometer_calibration)
        ev.addWidget(motion_calibration_btn)
        settings_layout.addWidget(self.settings_group)
        self.feedback_section = CollapsibleSection("Feedback")
        self.scan_sounds_cb = QCheckBox("Scan sounds")
        self.scan_sounds_cb.setChecked(self.capture_sound.enabled)
        self.scan_sounds_cb.setToolTip("Capture confirmations, tracking-loss and recovery alerts. Toggle to mute or enable.")
        self.scan_sounds_cb.toggled.connect(self.capture_sound.set_enabled)
        self.feedback_section.content_layout.addWidget(self.scan_sounds_cb)
        settings_layout.addWidget(self.feedback_section)
        diagnostics = CollapsibleSection("Tracking diagnostics")
        self.tracking_diagnostics_section = diagnostics
        dv = diagnostics.content_layout
        self.flow_debug_cb = QCheckBox("Show tracking flow")
        self.flow_debug_cb.setToolTip(
            "Show camera-side feature motion during scans with color-assisted tracking. "
            "Includes trails across the last 20 camera frames on the calibrated depth grid."
        )
        self.flow_windows_cb = QCheckBox("Show LK patch windows")
        self.flow_windows_cb.setToolTip(
            "Outline up to 24 of the 21 × 21 pixel tracking windows. "
            "Pyramid levels extend motion estimation; these boxes are not a fixed search boundary."
        )
        dv.addWidget(self.flow_debug_cb)
        dv.addWidget(self.flow_windows_cb)
        legend_text = (
            "Green: verified motion · fading green: 20-frame trails · cyan: new corners · red: flow lost · "
            "orange: round-trip rejection · purple: depth rejection · yellow: geometry rejection. "
            "Camera motion is checked independently by the server before fusion."
        )
        self.flow_debug_cb.setToolTip(self.flow_debug_cb.toolTip() + "\n" + legend_text)
        self.flow_debug_status = QLabel()
        self.flow_debug_status.setWordWrap(True)
        self.flow_debug_status.setAccessibleName("Camera tracking diagnostics")
        dv.addWidget(self.flow_debug_status)
        self.flow_debug_cb.toggled.connect(self._update_tracking_debug)
        self.flow_windows_cb.toggled.connect(self._update_tracking_debug)
        settings_layout.addWidget(diagnostics)
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
        self._add_field_tooltips(container)
        self._capture_mode_changed()

    def _add_field_tooltips(self, container):
        for control, help_text in (
            (self.capture_mode_combo, "Automatic selects sharp recent frames at the minimum interval. Manual captures one frame when you press C or Capture Frame."),
            (self.depth_near_spin, "Ignore surfaces closer than this distance from the Kinect."),
            (self.depth_far_spin, "Ignore surfaces farther than this distance from the Kinect."),
            (self.crop_cb, "Only reconstruct the central region of the depth image."),
            (self.crop_spin, "Width and height of the central region, as a percentage of the image."),
            (self.rgb_mode_combo, "Choose more color detail at 10 fps or smoother camera motion at 30 fps."),
            (self.rgb_exposure_combo, "Auto adjusts exposure to available light. Manual lets you choose shutter speed and gain."),
            (self.voxel_spin, "Smaller voxels show finer detail and use more reconstruction memory."),
            (self.weight_spin, "Higher confidence removes weakly observed surface from the final model; scan areas from overlapping views."),
            (self.live_cb, "Process captured views and show the fused point cloud while scanning. When off, reconstruction and its tracking checks wait until Finish; camera-side color tracking stays available."),
            (self.server_ip_edit, "Address of the computer running the reconstruction server."),
            (self.server_port_spin, "Port used by the reconstruction server (usually 8000)."),
        ):
            control.setToolTip(help_text)
        for label in container.findChildren(QLabel):
            if label.buddy() is not None:
                label.setToolTip(label.buddy().toolTip())

    def _build_statusbar(self):
        self.fps_label = QLabel("FPS: --")
        self.kinect_label = QLabel("Kinect: connecting...", self)
        self.statusBar().addPermanentWidget(self.fps_label)
        self.kinect_label.hide()

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
        busy = busy or bool(self._server_operation) or self._restore_on_status
        camera_stopping = self._camera_suspended and hasattr(self, "worker") and self.worker.isRunning()
        if self._camera_suspended:
            self.kinect_label.setText("Kinect: stopping…" if camera_stopping else "Kinect: stopped")
            self.camera_title.setText("Stopping camera…" if camera_stopping else "Camera off")
        busy = busy or camera_stopping
        can_start_camera = ready or self._camera_suspended
        active = self._scanning
        frames = self._server_stored > 0 or self._pending_upload_count() > 0
        valid_setup = self.depth_near_spin.value() < self.depth_far_spin.value() and (
            self.final_voxel_spin.value() == 0 or self.final_voxel_spin.value() >= 2
        )
        self.btn_start_scan.setEnabled(connected and can_start_camera and valid_setup and not busy and (not active or self._paused))
        self.btn_start_scan.setText("New Scan" if self._session_id else "Start Scan")
        self.btn_pause.setEnabled(connected and can_start_camera and not busy and (active or frames))
        self.btn_pause.setText("Resume Capture" if self._paused or not active else "Pause")
        self.btn_capture.setEnabled(connected and ready and active and not self._paused and not busy)
        self.btn_stop_build.setEnabled(connected and frames and not busy)
        self.btn_stop_build.setText("Retry Build" if self._build_failed else "Finish Scan")
        self.btn_cancel_scan.setEnabled(connected and not busy and (active or (frames and not self._has_mesh)))
        self.btn_reset_scan.setEnabled(connected and not busy and (active or frames or self._has_mesh or bool(self._session_id)))
        self.btn_preview_scan.setEnabled(connected and (frames or self._has_mesh) and not busy)
        self.btn_export.setEnabled(connected and self._has_mesh and not busy)
        self.btn_export_session.setEnabled(connected and (frames or bool(self._sensor_counts_seen)) and not busy)
        self.btn_open_project.setEnabled(connected and not busy)
        if hasattr(self, "open_project_action"):
            self.open_project_action.setEnabled(self.btn_open_project.isEnabled())
            self.save_project_action.setEnabled(self.btn_export_session.isEnabled())
            self.save_project_as_action.setEnabled(self.btn_export_session.isEnabled())
        for button in (self.btn_export_ply, self.btn_export_obj, self.btn_export_glb,
                       self.btn_export_texture_obj, self.btn_save_mesh):
            button.setEnabled(connected and self._has_mesh and not busy)
        self.settings_group.setEnabled(not active and not busy)
        self.capture_mode_combo.setEnabled(not busy)
        self.auto_capture_spin.setEnabled(not busy)
        self.auto_capture_cb.setEnabled(connected and active and not busy)
        if not connected:
            reason = "Connect the reconstruction server."
        elif self._camera_suspended:
            reason = "Stopping camera…" if camera_stopping else "Camera off · Resume Capture or New Scan turns it on."
        elif not ready:
            reason = "Waiting for fresh color and depth frames from the Kinect."
        else:
            reason = "Space: pause/resume · C: capture in Manual mode"
        self.readiness_label.setText(reason)
        self.scan_status_label.setToolTip(reason)
        self.fps_label.setToolTip(self.kinect_label.text())
        self.btn_start_scan.setToolTip(reason if not self.btn_start_scan.isEnabled() else "Begin a new capture session")
        self.btn_connect.setEnabled(not (self._connect_pending or self._reset_pending or self._build_pending
                                        or self._preview_pending or self._export_pending or self._final_preview_pending))
        self.server_ip_edit.setEnabled(not connected and not self._connect_pending)
        self.server_port_spin.setEnabled(not connected and not self._connect_pending)
        self.btn_preview_3d.setEnabled(bool(self._last_preview_path) and not busy)
        self._refresh_status()

    def _refresh_status(self):
        interval = self._effective_capture_interval()
        offline = not (self._session_settings or {}).get("live_reconstruction", self.live_cb.isChecked())
        pending_uploads = self._pending_upload_count()
        if offline and self._session_id:
            self.frame_count_label.setText(
                f"Captured: {self._server_stored + pending_uploads} · Pending upload: {pending_uploads}"
            )
        camera_motion_unverified = (
            self._scanning and not self._paused and offline
            and ((self._session_settings or {}).get("color_recovery", self.color_tracking_cb.isChecked())
                 or (self._session_settings or {}).get("apriltag_tracking", self.apriltag_tracking_cb.isChecked()))
            and self._last_frame_metadata.get("visual_tracking", {}).get("valid") is False
        )
        if camera_motion_unverified != self._camera_motion_unverified:
            self.logs_panel.append(
                "Camera motion could not be verified. Slow down and keep overlapping detail in view. "
                "Captures are retained; reconstruction is checked at Finish."
                if camera_motion_unverified else "Camera motion warning cleared.",
                "Camera", key="camera_motion",
            )
            self._camera_motion_unverified = camera_motion_unverified
        if self._camera_suspended:
            self.interval_help.setText("Capture finished · stopping camera…" if self.worker.isRunning()
                                       else "Capture finished · camera and sensors are off.")
        elif self._scanning and self.live_view.snapshot.get("fusion_paused"):
            self.interval_help.setText("Checking fresh views as soon as each recovery check finishes.")
        elif self._scanning and interval > self.auto_capture_spin.interval_seconds + 1e-9:
            self.interval_help.setText(
                f"Capture pace: ~{interval:g} s · adjusted for live reconstruction."
            )
        elif offline:
            self.interval_help.setText(
                f"Captures no faster than {self.auto_capture_spin.interval_seconds:g} s. "
                "Selects a sharp recent frame; buffers it locally and uploads in the background."
            )
        else:
            self.interval_help.setText(
                f"Captures no faster than {self.auto_capture_spin.interval_seconds:g} s. "
                "Selects a sharp recent frame; slows to match live reconstruction."
            )
        if self._server_operation:
            state = "Server is still building · waiting for completion" if self._server_operation == "build" else "Server is preparing inspection · waiting"
        elif self._connect_pending:
            state = "Connecting to reconstruction server…"
        elif self._restore_on_status and self.server_client.is_connected:
            state = "Checking server scan…"
        elif self._reset_pending:
            state = "Resetting scan…" if self._reset_to_setup_pending else (
                "Cancelling scan…" if self._cancel_pending else "Starting scan…"
            )
        elif self._build_pending:
            state = "Uploading remaining captures before reconstruction…" if pending_uploads else "Building final surface…"
        elif self._export_pending:
            state = {"session": "Saving project…", "open": "Opening project…"}.get(self._export_pending["kind"], "Exporting final model…")
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
        elif self._scanning and self.live_view.snapshot.get("fusion_paused"):
            state = "TRACKING LOST · model paused · match the highlighted last good view"
        elif self._scanning and self._paused:
            state = "Paused · scan retained"
        elif self._scanning and self._capture_waiting:
            state = self._capture_waiting
        elif self._scanning and not self._camera_ready():
            state = "Waiting for fresh camera frames · scan retained"
        elif self._scanning:
            state = "Capturing automatically" if self.auto_capture_cb.isChecked() else "Manual capture · ready"
            if offline:
                state += " · reconstruction checked at Finish"
        elif self._has_mesh:
            state = "Scan finished · camera off · inspect, save or export"
        elif self._server_stored:
            state = "Scan retained · resume capture or finish"
        else:
            state = "Ready to scan" if self._camera_ready() else "Waiting for camera"
        self.auto_capture_spin.set_capture_help(self.interval_help.text())
        self.interval_label.setToolTip(self.interval_help.text())
        if self.scan_status_label.text() != state:
            self.logs_panel.append(state, "Scan")
        self.scan_status_label.setText(state)
        self.guidance_label.setVisible(
            self._scanning and self.live_view.isHidden()
            and self.view_stack.currentWidget() is self.splitter
            and (bool(self.live_view.snapshot.get("fusion_paused")) or not self._progress_link_ok)
        )

    def _capture_mode_changed(self):
        automatic = self.capture_mode_combo.currentData() == "automatic"
        self.interval_row.setVisible(automatic)
        self.interval_help.hide()
        self.btn_capture.setVisible(not automatic)
        self.auto_capture_cb.setChecked(automatic and self._scanning)
        self._capture_waiting = ""
        self._reset_auto_capture_cadence()
        self._refresh_controls()

    def _pause_or_resume(self):
        if self._reset_pending or self._build_pending or self._preview_pending or self._export_pending or self._final_preview_pending or self._server_operation or self._connect_pending or self._restore_on_status:
            return
        if not self.server_client.is_connected:
            return
        if self._scanning and not self._paused:
            self._paused = True
        elif (self._camera_ready() or self._camera_suspended) and (self._scanning or self._server_stored):
            if self._camera_suspended and self.worker.isRunning():
                return
            self._apply_session_settings(self._session_settings)
            self._resume_camera()
            self._scanning = True
            self._paused = False
            self._configure_camera_tracking()
            self._build_failed = False
            self._has_mesh = False
            self._last_preview_path = None
            self._operation_error = ""
            self.auto_capture_cb.setChecked(self.capture_mode_combo.currentData() == "automatic")
        self._capture_waiting = ""
        logger.info("Capture state session=%s scanning=%s paused=%s", self._session_id, self._scanning, self._paused)
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
        if self._last_depth is not None:
            if self._mode == MODE_DEPTH:
                self._show_depth(self._last_depth)
            elif self._mode == MODE_SCANNER:
                self.scan_depth_view.set_image(numpy_to_qimage(rotate_display(self._depth_display(self._last_depth), self._display_rotation)))
            if not self._camera_ready():
                self._set_camera_stale("Camera delayed · last image")

    def _choose_export(self):
        if not self.btn_export.isEnabled():
            return
        dialog = ExportDialog(self, preferences=self.preferences)
        if dialog.exec():
            fmt = dialog.selected_format
            self.texture_exposure_cb.setChecked(dialog.texture_options.get("exposure_correction", False))
            self.texture_best_cb.setChecked(dialog.texture_options.get("blend_mode") == "best")
            self.preferences.write("export/format", fmt)
            self.preferences.write("export/exposure_correction", self.texture_exposure_cb.isChecked())
            self.preferences.write("export/best_source", self.texture_best_cb.isChecked())
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
        self._show_message("Connected to reconstruction server", 4000)

    def _on_server_disconnected(self, reason: str):
        self._start_when_camera_ready = False
        self._connect_pending = False
        self._status_pending = False
        self._server_operation = None
        self.live_view.set_feedback_connected(False)
        self.btn_connect.setText("Reconnect" if self._session_id else "Connect")
        self.server_status_label.setText(reason)
        self.server_status_label.setStyleSheet("color: #b4382c;")
        self._set_scan_controls_enabled(False)
        self.connection_section.toggle.setChecked(True)
        self._show_message(f"Server: {reason}")

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
        self.logs_panel.append(detail, "Connection")
        self._refresh_status()

    def _switch_mode(self, mode: str):
        self.view_stack.setCurrentWidget(self.logs_panel if mode == "logs" else self.splitter)
        for action in self._mode_actions:
            action.setChecked(action.data() == mode)
        if mode == "logs":
            self.guidance_label.hide()
            self.logs_panel.flush()
            return
        self._mode = mode
        self.camera_title.setVisible(mode == MODE_SCANNER)
        self.live_view.setVisible(mode == MODE_SCANNER and bool(self._session_id) and self.live_cb.isChecked())
        self.guidance_label.hide()
        if self.live_view.isVisible():
            self.splitter.setSizes([max(300, self.splitter.width() * 2 // 3), max(240, self.splitter.width() // 3)])
        self.camera_title.setText("Camera off" if self._camera_suspended else
                                  "Live camera · Depth" if mode == MODE_DEPTH else "Live camera · Color")
        self.scan_depth_panel.setVisible(mode == MODE_SCANNER and not self._camera_suspended)
        self.depth_legend_label.setVisible(mode in (MODE_DEPTH, MODE_SCANNER) and not self._camera_suspended)
        if self._last_rgb is not None and self._last_depth is not None:
            if mode == MODE_SCANNER:
                self._show_scanner(self._last_rgb, self._last_depth)
            elif mode == MODE_DEPTH:
                self._show_depth(self._last_depth)
            else:
                self._show_rgb(self._last_rgb)
            if not self._camera_ready():
                self._set_camera_stale("Camera delayed · last image")
        for action in self._mode_actions:
            action.setChecked(action.data() == mode)

    # ── frame display ─────────────────────────────────────────────────
    def _on_frame(self, video: np.ndarray, depth: np.ndarray, metadata=None):
        if self._closing or self._camera_suspended:
            return
        self._fps_counter += 1
        if not self._camera_ok:
            self.logs_panel.append("Live color and depth frames received", "Camera")
        self._camera_ok = True
        self.kinect_label.setText("Kinect: live")
        self._last_rgb = video
        self._last_depth = depth
        self._frame_sequence += 1
        self._last_frame_metadata = dict(metadata or {})
        self._update_orientation()
        # Arrays and the calibrated preview belong only to this frame's UI.
        # Remove them before capture selection, upload, or local recording.
        debug = self._last_frame_metadata.pop("_tracking_debug", None)
        self._last_tracking_debug = debug if self.flow_debug_cb.isChecked() else None
        if self._last_frame_metadata.get("rgb_exposure_controls") is False:
            self.rgb_exposure_status_label.setText("Default auto exposure · manual controls unavailable in this driver")
        elif self._last_frame_metadata.get("rgb_exposure_mode") == "manual":
            duration = self._last_frame_metadata.get("rgb_exposure_us")
            if duration is not None:
                gain = self._last_frame_metadata.get("rgb_gain", self.rgb_gain_combo.currentData())
                self.rgb_exposure_status_label.setText(f"Manual · {duration / 1000:.2f} ms · {gain}× gain")
        elif self._last_frame_metadata.get("rgb_exposure_mode") == "auto":
            self.rgb_exposure_status_label.setText("Auto exposure active")
        self._last_frame_time = self._last_frame_metadata.get(
            "captured_monotonic_s", time.monotonic()
        )
        # GUI sequence remains unique if the USB device reconnects mid-session.
        self._last_frame_metadata["frame_id"] = f"{self._capture_run_id}:{self._frame_sequence}"
        self._last_frame_metadata.setdefault("timestamp_s", time.time())
        self._capture_selector.offer(video, depth, self._last_frame_metadata, self._last_frame_time)
        if self._start_when_camera_ready:
            self._start_when_camera_ready = False
            if self.server_client.is_connected:
                self._start_scan(protected=True)
        self._update_tracking_debug_status()

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
            if (self._auto_frames_since_capture >= self.auto_capture_spin.value()
                    or self.live_view.snapshot.get("fusion_paused")):
                self._auto_capture_tick()
        self._refresh_controls()

    def _show_rgb(self, rgb):
        if self.flow_debug_cb.isChecked() and self._last_tracking_debug is not None:
            self.camera_title.setText("Live camera · Tracking RGB (depth grid)")
            image = flow_image(self._last_tracking_debug, show_windows=self.flow_windows_cb.isChecked())
        else:
            self.camera_title.setText("Live camera · Color")
            image = numpy_to_qimage(rgb)
        from PyQt6.QtGui import QTransform
        self._set_pixmap(image.transformed(QTransform().rotate(self._display_rotation)))

    def _update_tracking_debug_status(self):
        if not self.flow_debug_cb.isChecked():
            return
        if self._last_tracking_debug is not None:
            self.flow_debug_status.setText(flow_summary(self._last_tracking_debug))
        else:
            reason = self._last_frame_metadata.get("visual_tracking", {}).get("reason")
            self.flow_debug_status.setText(reason or
                "Waiting for tracking frames. Start a scan with Color-assisted tracking enabled.")

    def _update_tracking_debug(self):
        enabled = self.flow_debug_cb.isChecked()
        self.flow_windows_cb.setEnabled(enabled)
        self.flow_debug_status.setVisible(enabled)
        if not enabled:
            self._last_tracking_debug = None
        if hasattr(self, "worker") and hasattr(self.worker, "set_tracking_debug"):
            self.worker.set_tracking_debug(enabled)
        self._update_tracking_debug_status()
        if self._last_rgb is not None and self._mode != MODE_DEPTH:
            self._show_rgb(self._last_rgb)

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
        self._set_pixmap(numpy_to_qimage(rotate_display(self._depth_display(depth), self._display_rotation)))

    def _show_scanner(self, rgb, depth):
        self._show_rgb(rgb)
        self.scan_depth_view.set_image(numpy_to_qimage(rotate_display(self._depth_display(depth), self._display_rotation)))

    def _update_orientation(self):
        metadata = self._last_frame_metadata
        decision = self._orientation.update(metadata.get("orientation_accelerometer", metadata.get("accelerometer", {})),
                                            metadata.get("orientation_host_monotonic_s", metadata.get("depth_host_monotonic_s", metadata.get("captured_monotonic_s", time.monotonic()))),
                                            self.orientation_combo.currentData())
        metadata["orientation"] = decision
        self._display_rotation = decision["rotation_cw_degrees"]
        self.sensor_status_label.setText(f"{decision['reason']} · {self._display_rotation}°")
        self.orientation_combo.setToolTip(
            "Auto follows gravity. Lock portrait when looking up/down or while recording.\n"
            + self.sensor_status_label.text()
        )

    def _change_orientation(self):
        self._update_orientation()
        if hasattr(self, "worker") and hasattr(self.worker, "record_sensor_event"):
            self.worker.record_sensor_event({"type": "orientation_mode", "value": self.orientation_combo.currentData(),
                                             "host_monotonic_s": time.monotonic(), "timestamp_s": time.time()})
        self._switch_mode(self._mode)

    def _on_sensor_recording_status(self, status):
        signature = (status.get("root"), status.get("recording_segment"), status.get("error"),
                     status.get("closed"), tuple(sorted(status.get("dropped", {}).items())))
        if signature != getattr(self, "_logged_sensor_status", None):
            self._logged_sensor_status = signature
            logger.info("Sensor recording session=%s status=%s", self._session_id, status)
        if status.get("root") and status["root"] != self._sensor_recording_path:
            return
        if self._session_id and status.get("root") == self._sensor_recording_path:
            segment = status.get("recording_segment")
            count = sum(status.get("counts", {}).values()) + sum(status.get("dropped", {}).values())
            if count > self._sensor_counts_seen.get(segment, 0):
                self._sensor_counts_seen[segment] = count
                if not (self._export_pending and self._export_pending["kind"] == "session"):
                    self._capture_revision += 1
                    self._session_dirty = True
        if not status.get("complete", True):
            drops = sum(status.get("dropped", {}).values())
            self.sensor_recording_label.setText(f"Sensor recording incomplete · {status.get('error') or str(drops) + ' dropped observations'}")
            self.sensor_recording_label.setStyleSheet("color: #ffb45b;")
            self.logs_panel.append(self.sensor_recording_label.text(), "Warning")
            self.sensor_recording_label.show()
        elif self._scanning:
            self.sensor_recording_label.hide()
            counts = status.get("counts", {})
            if status.get("record_full_camera_streams"):
                message = f"Recording all camera frames · {counts.get('rgb', 0)} RGB / {counts.get('depth', 0)} depth / {counts.get('accelerometer', 0)} acceleration"
            else:
                message = f"Recording accelerometer log · {counts.get('accelerometer', 0)} readings · selected images saved with session"
            self.sensor_recording_label.setText(message)
            self.sensor_recording_label.setStyleSheet("")
        self._refresh_controls()

    def _load_accelerometer_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load accelerometer calibration", "", "JSON (*.json)")
        if not path:
            return
        try:
            from pathlib import Path
            profile = calibration_profile(json.loads(Path(path).read_text()))
            self._accelerometer_calibration = profile
            self.preferences.write("camera/accelerometer_calibration", profile)
            self._restart_camera()
            self._show_message(f"Motion calibration loaded: {profile['id']}")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            self._show_message(f"Invalid motion calibration: {exc}")

    def _set_camera_stale(self, message):
        for view in (self.view_label, self.scan_depth_view):
            view.set_stale(True, message)

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
                self.preferences.save_calibration(profile)
                self._restart_camera()
            except (ValueError, TypeError, OSError, RuntimeError) as exc:
                self.scan_status_label.setText(f"Invalid calibration: {exc}")

    # ── scanning workflow ─────────────────────────────────────────────
    def _start_scan(self, checked=False, *, protected=False):
        if not self.server_client.is_connected:
            QMessageBox.warning(self, "Not Connected", "Connect to the server first.")
            return

        if self._reset_pending or self._build_pending or self._preview_pending or self._export_pending or self._final_preview_pending or self._server_operation or self._connect_pending or self._restore_on_status or (self._scanning and not self._paused and not protected):
            return
        if self._camera_suspended:
            if self.worker.isRunning() or (not protected and not self._protect_session("new_scan")):
                return
            self._start_when_camera_ready = True
            self._resume_camera()
            self._refresh_controls()
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
                rgb_exposure_mode=self.rgb_exposure_combo.currentData(),
                rgb_shutter_speed=self.rgb_shutter_spin.value(),
                rgb_gain=self.rgb_gain_combo.currentData(),
                gravity_assistance=self.gravity_tracking_cb.isChecked(),
                record_full_camera_streams=self.full_camera_recording_cb.isChecked(),
                accelerometer_calibration=self._accelerometer_calibration,
                orientation_mode=self.orientation_combo.currentData(),
                near_m=self.depth_near_spin.value() / 1000,
                far_m=self.depth_far_spin.value() / 1000,
                voxel_m=voxel,
                truncation_m=min(0.2, max(0.04, voxel * 8)),
                final_weight=self.weight_spin.value(),
                color_recovery=self.color_tracking_cb.isChecked(),
                apriltag_tracking=self.apriltag_tracking_cb.isChecked(),
                apriltag_dictionaries=self.apriltag_dictionaries.dictionaries(),
                live_reconstruction=self.live_cb.isChecked(),
                refine_poses=self.refine_cb.isChecked() and self.offline_registration_combo.currentData() != "depth",
                bundle_adjustment=self.bundle_cb.isChecked() and self.offline_registration_combo.currentData() != "depth",
                reconnect_fragments=self.reconnect_fragments_cb.isChecked(),
                offline_registration=self.offline_registration_combo.currentData(),
                relocalize=self.relocalize_cb.isChecked(),
                confidence_fusion=self.confidence_cb.isChecked(),
                final_voxel_m=self.final_voxel_spin.value() / 1000
                if self.final_voxel_spin.value()
                else None,
                roi=roi,
            )
        except ValueError as exc:
            self.scan_status_label.setText(str(exc))
            return
        if not protected and not self._protect_session("new_scan"):
            return
        self._operation_error = ""
        self._cancel_pending = False
        self._reset_to_setup_pending = False
        self._reset_pending = True
        self.capture_sound.stop()
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
        self.guidance_label.setStyleSheet("")
        cancelled = self._cancel_pending
        reset_to_setup = self._reset_to_setup_pending
        self._cancel_pending = False
        self._reset_to_setup_pending = False
        self._reset_pending = False
        if not self.server_client.is_connected or self._closing:
            return
        self._session_id = None if cancelled else result.get("session_id")
        self._session_settings = None if cancelled else result.get("settings")
        self._session_dirty = False
        self._project_path = None
        self._capture_revision = self._saved_revision = 0
        self._sensor_counts_seen = {}
        self.sensor_recording_label.hide()
        self.sensor_recording_label.setText("" if cancelled else "Starting sensor recording…")
        self._pending_action = None
        self._has_mesh = False
        self.live_view.reset()
        self.capture_sound.reset_tracking()
        self.live_view.setVisible(
            not cancelled and result.get("settings", {}).get("live_reconstruction", False)
        )
        self._last_capture_id = None
        self._capture_pacer.reset()
        self._capture_selector.clear()
        self._reset_auto_capture_cadence()
        self._server_stored = 0
        self._server_integrated = 0

        self._last_preview_path = None
        self._final_preview_pending = False
        self._preview_session = None
        self._final_preview_session = None
        self._operation_error = ""
        self._scanning = not cancelled
        self._configure_camera_tracking()
        self._paused = False
        self._build_pending = self._build_failed = False
        self._capture_waiting = ""
        self._switch_mode(MODE_RGB if cancelled else MODE_SCANNER)
        if reset_to_setup:
            self._start_when_camera_ready = False
            self._resume_camera()
        if cancelled:
            self.guidance_label.setText("Keep the subject stationary. Move the Kinect slowly around it with overlapping views.")
        self.guidance_label.hide()
        self.auto_capture_cb.setChecked(self._scanning and self.capture_mode_combo.currentData() == "automatic")
        self.frame_count_label.setText("Captured: 0 · Added to model: 0")
        self._set_progress_visible(False)
        self._refresh_controls()
        self._show_message("Scan reset · press Start Scan when ready" if reset_to_setup else (
            "Scan cancelled · ready for a new scan" if cancelled else "Scan started"
        ), 4000)

    def _configure_camera_tracking(self):
        self._configure_sensor_recording()
        if hasattr(self.worker, "set_tracking_settings"):
            try:
                settings = ScanSettings.from_dict(self._session_settings) if self._session_settings else None
            except (TypeError, ValueError):
                settings = None
            if settings is not None and not (self._scanning and (settings.color_recovery or settings.apriltag_tracking)):
                settings = None
            configuration = (self.worker, self._session_id, settings)
            if getattr(self, "_tracking_configuration", None) == configuration:
                return
            self._tracking_configuration = configuration
            self._last_tracking_debug = None
            self.worker.set_tracking_settings(settings)

    def _configure_sensor_recording(self):
        if not hasattr(self.worker, "set_sensor_recording"):
            return
        if self._session_id:
            identity = hashlib.sha256(str(self._session_id).encode()).hexdigest()[:20]
            self._sensor_recording_path = os.path.join(_PROJECT_ROOT, "recordings", "sensors-" + identity)
        else:
            self._sensor_recording_path = None
        recording = self._scanning and not (self._export_pending and self._export_pending["kind"] == "session")
        settings = {**self._session_settings, "orientation_mode": self.orientation_combo.currentData()} if self._session_settings else None
        self.worker.set_sensor_recording(self._sensor_recording_path if recording else None,
                                         settings)

    def _cancel_scan(self, checked=False, *, protected=False):
        self._clear_scan("cancel_scan", protected=protected)

    def _reset_scan(self, checked=False, *, protected=False):
        self._clear_scan("reset_scan", protected=protected)

    def _clear_scan(self, reason, *, protected=False):
        if self._closing or not self.server_client.is_connected:
            return
        if self._reset_pending or self._build_pending or self._preview_pending or self._export_pending or self._final_preview_pending or self._server_operation or self._connect_pending or self._restore_on_status:
            return
        if self._camera_suspended and self.worker.isRunning():
            return
        if reason == "cancel_scan" and not self._scanning and (not self._server_stored or self._has_mesh):
            return
        if not (self._scanning or self._session_id or self._server_stored or self._has_mesh):
            return
        if not protected and not self._protect_session(reason):
            return
        self._cancel_pending = self._reset_pending = True
        self._reset_to_setup_pending = reason == "reset_scan"
        self._start_when_camera_ready = False
        self.capture_sound.stop()
        self._paused = True
        self.auto_capture_cb.setChecked(False)
        self._operation_error = ""
        self._capture_waiting = ""
        self._reset_auto_capture_cadence()
        self.progress_bar.setRange(0, 0)
        self._set_progress_visible(True)
        queued = self.task_worker.submit(ServerTask(
            ServerTaskType.RESET, {"settings": self._session_settings, "record": False}
        ))
        if not queued:
            self._on_task_failed("RESET", "Reset could not be queued" if reason == "reset_scan" else "Cancellation could not be queued")
        self._refresh_controls()

    def _auto_capture_tick(self):
        if self._server_operation or self._reset_pending or self._export_pending or self._paused or self._connect_pending or self._restore_on_status:
            return
        snapshot = self.live_view.snapshot
        now = time.monotonic()
        self._capture_pacer.observe(snapshot, now)
        outstanding = max(
            self._capture_pacer.outstanding_count,
            self.task_worker.queued_task_count,
            snapshot.get("pending_count", 0)
            if snapshot.get("processed_count", 0) >= self._capture_pacer.processed_count else 0,
        )
        if not self._progress_link_ok and self.live_cb.isChecked():
            self._capture_waiting = "Auto capture waiting for live feedback to reconnect"
            self._refresh_status()
            return
        if self.live_cb.isChecked() and snapshot.get("fusion_paused") and outstanding:
            self._capture_waiting = "Model paused · waiting for recovery check; match the last good image"
            self._refresh_status()
            return
        if self._adaptive_live_capture() and outstanding >= 2:
            # At most one frame being processed and one waiting, including uploads.
            self._capture_waiting = "Capturing automatically · paced by live reconstruction"
            self._refresh_status()
            return
        if self._adaptive_live_capture() and self.task_worker.queued_task_count >= 5:
            self._capture_waiting = "Auto capture waiting for uploads to catch up"
            self._refresh_status()
            return
        self._capture_waiting = ""
        recovering = self.live_cb.isChecked() and bool(snapshot.get("fusion_paused"))
        if not self._capture_pacer.ready(
            now, 0.1 if recovering else self.auto_capture_spin.interval_seconds,
            RGB_MODE_FPS[self.rgb_mode_combo.currentData()],
            adaptive=self._adaptive_live_capture() and not recovering,
        ):
            self._refresh_status()
            return
        self._capture_frame(select_best=True)

    def _adaptive_live_capture(self):
        return self.adaptive_capture_cb.isChecked() and self.live_cb.isChecked()

    def _pending_upload_count(self):
        return getattr(self.task_worker, "pending_capture_count", self.task_worker.queued_task_count)

    def _effective_capture_interval(self):
        return self._capture_pacer.interval_seconds(
            self.auto_capture_spin.interval_seconds,
            RGB_MODE_FPS[self.rgb_mode_combo.currentData()],
            adaptive=self._adaptive_live_capture(),
        )

    def _capture_frame(self, *, select_best=False):
        if (
            not self._scanning
            or self._reset_pending
            or self._server_operation
            or self._connect_pending
            or self._restore_on_status
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
        selected = (self._capture_selector.choose(time.monotonic(), self._capture_pacer.last_capture_at)
                    if select_best else None)
        rgb, depth, metadata = selected if selected is not None else (
            self._last_rgb, self._last_depth, self._last_frame_metadata)
        frame_id = metadata.get("frame_id")
        queued = self.task_worker.submit(
            ServerTask(ServerTaskType.SEND_FRAME, {
                "rgb": rgb.copy(), "depth": depth.copy(), "metadata": metadata.copy(),
            })
        )
        if not queued:
            self._paused = True
            self._operation_error = getattr(self.task_worker, "capture_error", "") or "Capture paused: upload buffer full"
            self._refresh_controls()
        else:
            self._last_capture_id = frame_id
            self._capture_pacer.captured(frame_id, time.monotonic(), live=self.live_cb.isChecked())
            self._capture_selector.clear()
            self._capture_revision += 1
            self._session_dirty = True
            self._operation_error = ""
            self._capture_waiting = ""
            if not self.live_cb.isChecked():
                self.capture_sound.play()
            self._reset_auto_capture_cadence()
            self._refresh_controls()

    def _reset_auto_capture_cadence(self):
        self._auto_frames_since_capture = 0

    def _toggle_auto_capture(self, checked: bool):
        self._reset_auto_capture_cadence()

    def _stop_and_build(self):
        if self._build_pending or self._preview_pending or self._export_pending or self._final_preview_pending or not self.server_client.is_connected:
            return
        if not self._server_stored and not self._pending_upload_count():
            return
        logger.info("Finish scan requested session=%s stored=%s upload_queue=%s final_voxel_mm=%s",
                    self._session_id, self._server_stored, self.task_worker.queued_task_count,
                    self.final_voxel_spin.value())
        self._build_pending = True
        self._build_failed = False
        self._operation_error = ""
        self._last_preview_path = None
        self._scanning = False
        self._stop_camera_after_finish()
        self.capture_sound.reset_tracking()
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
        self._set_progress_visible(True)

        options = {"final_voxel_m": self.final_voxel_spin.value() / 1000 or None,
                   "offline_registration": self.offline_registration_combo.currentData()}
        if self._session_settings:
            self._session_settings = {**self._session_settings, **options}
        self._session_dirty = True
        self._capture_revision += 1
        self.task_worker.submit(ServerTask(ServerTaskType.BUILD_MESH, {"options": options}))
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
        self._set_progress_visible(True)

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

    def _save_project(self, checked=False):
        return self._export_session(path=self._project_path)

    def _export_session(self, checked=False, path=None):
        if self._export_pending or not self.server_client.is_connected:
            return False
        if path is None:
            filename = f"scan-session_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}.zip"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Project", self._project_path or os.path.join(self._ensure_export_dir(), filename),
                "Scanner projects and sessions (*.zip)",
            )
        if not path:
            return False
        options = {"path": path}
        if self._sensor_recording_path and hasattr(self.worker, "flush_sensor_recording"):
            options.update(sensor_recorder=self.worker, sensor_path=self._sensor_recording_path)
        return self._begin_export("session", path, ServerTask(ServerTaskType.EXPORT_SESSION, options))

    def _open_project(self, checked=False, *, path=None, protected=False):
        if not self.btn_open_project.isEnabled():
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open Project or Session", self._ensure_export_dir(),
                                                "Scanner projects and sessions (*.zip)")
        if not path:
            return
        self._project_to_open = path
        if not protected and not self._protect_session("open_project"):
            return
        self.auto_capture_cb.setChecked(False)
        self._begin_export("open", path, ServerTask(ServerTaskType.OPEN_PROJECT, {"path": path}))

    def _on_project_opened(self, status, path):
        self._export_pending = None
        self._pending_action = None
        self._project_to_open = None
        self._set_progress_visible(False)
        self._sensor_recording_path = None
        self._sensor_counts_seen = {}
        self._build_failed = False
        self._restore_server_session(status)
        self._scanning = False
        self._project_path = path
        self._saved_revision = self._capture_revision
        self._session_dirty = False
        self._configure_sensor_recording()
        self._show_message(f"Opened project: {path}", 10000)
        self._refresh_controls()

    def _on_transfer_progress(self, phase, done, total):
        if not self._export_pending:
            return
        self.progress_bar.setRange(0, 100 if total else 0)
        self.progress_bar.setValue(int(done / total * 100) if total else 0)
        self.progress_bar.setFormat(f"{phase}: %p%")
        self._set_progress_visible(True)
        detail = f"{phase}: {done / 1024**2:.1f}"
        detail += f" / {total / 1024**2:.1f} MB" if total else " MB"
        self._show_message(detail)

    def _begin_export(self, kind, path, task):
        restore_capture = (self._scanning, self._paused)
        self._paused = True
        self._export_pending = {
            "kind": kind, "path": path, "revision": self._capture_revision,
            "restore_capture": restore_capture,
        }
        self._configure_sensor_recording()
        self._operation_error = ""
        self.progress_bar.setRange(0, 0)
        self._set_progress_visible(True)
        if not self.task_worker.submit(task):
            self._on_export_done(False, path)
            return False
        self._refresh_controls()
        return True

    def _protect_session(self, reason):
        if not self._session_dirty:
            return True
        if self._export_pending:
            self._show_message("Wait for the current save or export to finish.")
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
            if self._save_project():
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

    # ── Task/server signal handlers ──────────────────────────────────
    def _on_server_status(self, status):
        backend = status.get("backend", {})
        self.backend_label.setText(
            f"Fusion: {backend.get('device', 'unknown')} · ICP: {backend.get('tracking', 'unknown')}"
        )
        self.backend_label.setToolTip(backend.get("fallback_reason") or "")
        if "settings" in status:
            self._status_pending = False
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
        if self._closing or self._status_pending or not self.server_client.is_connected or not (self._restore_on_status or self._reconcile_on_status):
            return
        self._status_pending = True
        self.task_worker.submit(ServerTask(ServerTaskType.STATUS))

    def _restore_server_session(self, status):
        if self._reset_to_setup_pending and not status.get("operation"):
            if not status.get("stored_count", 0) and not status.get("has_mesh"):
                self._cancel_pending = True
                self._on_reset_done(status)
                return
            self._reset_to_setup_pending = False
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
        # The connection command is a queue barrier. Use authoritative server
        # counts, and avoid learning the disconnected time as processing cost.
        self._capture_pacer.reset()
        self._capture_selector.clear()
        self._capture_pacer.observe(status, time.monotonic())
        if not same_session:
            self._capture_revision = self._server_stored
            self._saved_revision = 0
            self._session_dirty = self._server_stored > 0
            self._last_preview_path = None
            self.live_view.reset()
            self.capture_sound.reset_tracking()
        elif self._server_stored != previous_stored:
            self._capture_revision += max(1, self._server_stored - previous_stored)
            self._session_dirty = True
        self._server_operation = status.get("operation")
        if self._has_mesh or self._server_operation == "build":
            self._stop_camera_after_finish()
        if self._server_operation:
            self._scanning = False
            self._reconcile_on_status = True
            QTimer.singleShot(1000, self._poll_server_status)
        if self._server_stored or self._has_mesh:
            self._apply_session_settings(self._session_settings)
        else:
            self._configure_camera_tracking()
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
        with self.preferences.suspend():
            self._restore_session_settings(settings)
        self._configure_camera_tracking()

    def _restore_session_settings(self, settings):
        if not settings:
            return
        try:
            profile = ScanSettings.from_dict(settings)
        except (ValueError, TypeError) as exc:
            self._operation_error = f"Cannot restore scan settings: {exc}"
            return
        controls = (self.depth_near_spin, self.depth_far_spin, self.voxel_spin,
                    self.final_voxel_spin, self.weight_spin,
                    self.rgb_mode_combo, self.crop_cb, self.crop_spin, self.live_cb,
                    self.rgb_exposure_combo, self.rgb_shutter_spin, self.rgb_gain_combo,
                    self.orientation_combo, self.gravity_tracking_cb, self.full_camera_recording_cb,
                    self.color_tracking_cb, self.apriltag_tracking_cb, self.refine_cb, self.bundle_cb, self.reconnect_fragments_cb,
                    self.offline_registration_combo, self.relocalize_cb, self.confidence_cb)
        previous = [control.blockSignals(True) for control in controls]
        rgb_changed = self.rgb_mode_combo.currentData() != profile.rgb_mode
        exposure_changed = (self.rgb_exposure_combo.currentData() != profile.rgb_exposure_mode
                            or self.rgb_shutter_spin.value() != profile.rgb_shutter_speed
                            or self.rgb_gain_combo.currentData() != profile.rgb_gain)
        calibration_changed = profile.sensor_calibration is not None and profile.sensor_calibration != self._sensor_calibration
        motion_calibration_changed = profile.accelerometer_calibration != self._accelerometer_calibration
        try:
            self.depth_near_spin.setValue(round(profile.near_m * 1000))
            self.depth_far_spin.setValue(round(profile.far_m * 1000))
            self.voxel_spin.setValue(profile.voxel_m * 1000)
            self.final_voxel_spin.setMaximum(profile.voxel_m * 1000)
            self.final_voxel_spin.setValue((profile.final_voxel_m or 0) * 1000)
            self.weight_spin.setValue(profile.final_weight)
            self.rgb_mode_combo.setCurrentIndex(self.rgb_mode_combo.findData(profile.rgb_mode))
            self.rgb_exposure_combo.setCurrentIndex(self.rgb_exposure_combo.findData(profile.rgb_exposure_mode))
            self.rgb_shutter_spin.setMinimum(RGB_MODE_FPS[profile.rgb_mode])
            self.rgb_shutter_spin.setValue(profile.rgb_shutter_speed)
            self.rgb_gain_combo.setCurrentIndex(self.rgb_gain_combo.findData(profile.rgb_gain))
            self.live_cb.setChecked(profile.live_reconstruction)
            self.color_tracking_cb.setChecked(profile.color_recovery)
            self.apriltag_tracking_cb.setChecked(profile.apriltag_tracking)
            self.apriltag_dictionaries.set_dictionaries(profile.apriltag_dictionaries)
            self.refine_cb.setChecked(profile.refine_poses)
            self.bundle_cb.setChecked(profile.bundle_adjustment)
            self.reconnect_fragments_cb.setChecked(profile.reconnect_fragments)
            self.offline_registration_combo.setCurrentIndex(self.offline_registration_combo.findData(profile.offline_registration))
            self.relocalize_cb.setChecked(profile.relocalize)
            self.confidence_cb.setChecked(profile.confidence_fusion)
            self.gravity_tracking_cb.setChecked(profile.gravity_assistance)
            self.full_camera_recording_cb.setChecked(profile.record_full_camera_streams)
            self.orientation_combo.setCurrentIndex(self.orientation_combo.findData(profile.orientation_mode))
            self._accelerometer_calibration = profile.accelerometer_calibration
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
        self._update_offline_registration_controls()
        self._update_exposure_controls()
        self.auto_capture_spin.set_fps(RGB_MODE_FPS[profile.rgb_mode])
        if rgb_changed or calibration_changed or exposure_changed or motion_calibration_changed:
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
        self._capture_pacer.observe(
            snapshot, time.monotonic(),
            learn_completion=not (self._preview_pending or self._build_pending),
        )
        for key in ("guidance", "surface_description"):
            if snapshot.get(key) and snapshot.get(key) != self.live_view.snapshot.get(key):
                self.logs_panel.append(snapshot[key], "Reconstruction", key=key)
        self.live_view.set_snapshot(snapshot)
        self._on_server_status(snapshot)
        if not self._scanning:
            self.capture_sound.reset_tracking()
        elif "fusion_paused" in snapshot:
            self.capture_sound.set_tracking_lost(snapshot["fusion_paused"])
        self._server_stored = max(self._server_stored, snapshot.get("stored_count", 0))
        self._server_integrated = snapshot.get("frame_count", 0)
        self.frame_count_label.setText(
            f"Captured: {self._server_stored} · Added to model: {self._server_integrated}"
        )
        self.guidance_label.setText(snapshot.get("guidance") or "Move slowly with overlapping views.")
        self.guidance_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: white; background: #a52c29; padding: 8px;"
            if snapshot.get("fusion_paused") else ""
        )
        self._refresh_controls()

    def _on_frame_stored(self, result: dict):
        if result.get("session_id") and result["session_id"] != self._session_id:
            return
        self._capture_pacer.acknowledge(result.get("capture_acknowledgements", []), time.monotonic())
        if not result.get("success"):
            self._paused = True
            self._operation_error = result.get("message", "Capture rejected; scan retained")
        elif not self._closing and not self._reset_pending and self.live_cb.isChecked():
            self.capture_sound.play()
        self._server_stored = max(
            self._server_stored, result.get("stored_count", self._server_stored)
        )
        self.frame_count_label.setText(
            f"Captured: {self._server_stored} · Added to model: {self._server_integrated}"
        )
        self.btn_export_session.setEnabled(self._server_stored > 0)
        self._refresh_controls()
        self.logs_panel.append(result.get("message", "Frame stored"), "Capture")

    def _on_process_progress(self, current: int, total: int, result: dict):
        if result.get("session_id") and result["session_id"] != self._session_id:
            return
        if "index" in result and "elapsed_ms" in result:
            # Preview/build may process frames before another live snapshot.
            # Their deliberate operation barriers are not live capture latency.
            self._capture_pacer.observe(
                {"processed_count": result["index"] + 1, "result": result},
                time.monotonic(), learn_completion=False,
            )
        message = result.get("message", "")
        is_activity = bool(result.get("stage") or (message and "index" not in result))
        progress_format = "Processing frames: %v / %m"
        if is_activity:
            progress_format = "Reconstruction: %v / %m"
            # Older servers send depth-graph counters only in their messages,
            # with current=0 and total=the capture count for every phase.
            for pattern, label in (
                (r"Depth registration: revisit .*; candidate (\d+)/(\d+)$", "Revisit candidates"),
                (r"Depth registration: local evidence (\d+)/(\d+)$", "Local depth pairs"),
                (r"Preparing depth view (\d+)/(\d+)$", "Preparing depth views"),
                (r"Searching accumulated depth components (\d+)/(\d+)$", "Component candidates"),
                (r"Checking complete depth component: view (\d+)/(\d+)$", "Checking depth views"),
            ):
                match = re.fullmatch(pattern, message)
                if match:
                    current, total = map(int, match.groups())
                    progress_format = f"{label}: %v / %m"
                    break
        self.progress_bar.setRange(0, max(0, total))
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(progress_format)
        if total <= 0 or (is_activity and current == 0):
            self.progress_bar.setRange(0, 0)
        elif not is_activity and current == total and self._build_pending:
            self.progress_bar.setRange(0, 0)
            message = "Preparing the reconstructed model…"
        self._set_progress_visible(self._build_pending or self._preview_pending,
                                   message if is_activity or current == total else "")
        self._server_integrated = result.get("frame_count", self._server_integrated)
        self.frame_count_label.setText(
            f"Captured: {self._server_stored} · Added to model: {self._server_integrated}"
        )
        self.logs_panel.append(result.get("message", "Processing captured frames"), "Reconstruction")
        self._refresh_status()

    def _set_progress_visible(self, visible: bool, message: str = ""):
        # Native macOS busy bars do not draw format text, so keep the activity
        # in a separate label and clear it when another operation takes over.
        self.progress_status_label.setText(message)
        self.progress_status_label.setVisible(visible and bool(message))
        self.progress_bar.setVisible(visible)

    def _on_build_mesh_done(self, success: bool, detail: str):
        logger.info("Build finished session=%s success=%s detail=%s", self._session_id, success, detail)
        self._build_pending = False
        self._build_failed = not success
        self._has_mesh = success
        self._scanning = False
        self._paused = True
        self._configure_camera_tracking()
        self._set_progress_visible(False)
        self._operation_error = "" if success else detail
        self._show_message(detail, 8000)
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
        self._set_progress_visible(True)
        self.task_worker.submit(ServerTask(ServerTaskType.FINAL_PREVIEW))
        self._refresh_controls()

    def _on_final_preview_done(self, path):
        if self._closing or self._final_preview_session != self._session_id or not self._has_mesh:
            return
        self._final_preview_pending = False
        self._last_preview_path = path
        self._set_progress_visible(False)
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
        self._set_progress_visible(False)
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
        logger.info("Export finished session=%s success=%s path=%s", self._session_id, success, path)
        pending = self._export_pending
        if pending and pending["path"] != path:
            return
        self._export_pending = None
        self._set_progress_visible(False)
        action, self._pending_action = self._pending_action, None
        if success:
            if pending and pending["kind"] == "session":
                self._project_path = path
                self._saved_revision = pending["revision"]
                self._session_dirty = self._capture_revision != self._saved_revision
            self._show_message(f"Saved: {path}", 10000)
            self._operation_error = ""
        else:
            self._operation_error = "Save failed · scan retained; choose Save Project to retry"
            self._show_message(f"Could not save {path}; current scan is retained.")
        if pending and (not action or not success):
            self._scanning, self._paused = pending["restore_capture"]
            self._reset_auto_capture_cadence()
            self._configure_sensor_recording()
        self._refresh_controls()
        if success and action == "new_scan":
            self._start_scan(protected=True)
        elif success and action == "cancel_scan":
            self._cancel_scan(protected=True)
        elif success and action == "reset_scan":
            self._reset_scan(protected=True)
        elif success and action == "close":
            self._close_approved = True
            self.close()
        elif success and action == "open_project":
            self._open_project(path=self._project_to_open, protected=True)

    def _on_save_mesh_done(self, success: bool, path: str):
        self._show_message(f"Saved: {path}" if success else f"Save failed: {path}", 10000)
        self._refresh_controls()

    def _on_task_started(self, msg: str):
        logger.info("Operation session=%s %s", self._session_id, msg)
        self._show_message(msg)
        if self._export_pending:
            self.progress_bar.setRange(0, 0)
            self._set_progress_visible(True)

    def _on_task_error(self, msg: str):
        logger.error("Operation error session=%s %s", self._session_id, msg, extra={"ui_log": False})
        # Recording/report warnings are independent of build or inspection success.
        self._show_message(msg, 10000)
        if "incomplete sensor recording" in msg:
            self.sensor_recording_label.setText("Saved session has incomplete sensor recording · see its recording report")
            self.sensor_recording_label.setStyleSheet("color: #ffb45b;")
            self.logs_panel.append(self.sensor_recording_label.text(), "Warning")
            self.sensor_recording_label.show()

    def _on_task_failed(self, task_type, message):
        logger.error("Task failed session=%s task=%s %s", self._session_id, task_type, message, extra={"ui_log": False})
        self._set_progress_visible(False)
        self._operation_error = f"{message} · current scan retained"
        if task_type == "CONNECT":
            self._connect_pending = False
            self.connection_section.toggle.setChecked(True)
        elif task_type == "RESET":
            self._reset_pending = False
            if self._cancel_pending:
                self._cancel_pending = False
                self._operation_error = "Could not confirm reset · checking server scan" if self._reset_to_setup_pending else "Could not confirm cancellation · checking server scan"
                self._restore_on_status = True
                self._poll_server_status()
        elif task_type == "BUILD_MESH":
            self._build_pending = False
            self._build_failed = True
            self._reconcile_on_status = True
            self._poll_server_status()
        elif task_type == "PREVIEW":
            self._resume_capture()
        elif task_type == "FINAL_PREVIEW":
            self._final_preview_pending = False
        elif task_type == "OPEN_PROJECT":
            pending, self._export_pending = self._export_pending, None
            if pending:
                self._scanning, self._paused = pending["restore_capture"]
            self._configure_sensor_recording()
        elif task_type in ("EXPORT_PLY", "EXPORT_OBJ", "EXPORT_TEXTURE", "EXPORT_SESSION"):
            if self._export_pending:
                self._on_export_done(False, self._export_pending["path"])
        elif task_type == "SEND_FRAME":
            self._paused = True
        elif task_type == "STATUS":
            self._status_pending = False
            if self._restore_on_status:
                self._operation_error = "Scan state unavailable · retrying server check"
            if self._reconcile_on_status or self._restore_on_status:
                QTimer.singleShot(3000, self._poll_server_status)
        self._refresh_controls()


    def _update_fps(self):
        now = time.monotonic()
        if now - getattr(self, "_last_diagnostic_at", 0) >= 15:
            self._last_diagnostic_at = now
            logger.info("Client state session=%s scanning=%s paused=%s build=%s preview=%s stored=%s integrated=%s upload_queue=%s preview_frames_replaced=%s",
                        self._session_id, self._scanning, self._paused, self._build_pending,
                        self._final_preview_pending or self._preview_pending, self._server_stored,
                        self._server_integrated, self.task_worker.queued_task_count,
                        getattr(self.worker, "preview_frames_replaced", 0))
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed > 0:
            self._fps_value = self._fps_counter / elapsed
        self._fps_counter = 0
        self._last_fps_time = now
        self.fps_label.setText("Camera: off" if self._camera_suspended else f"Camera: {self._fps_value:.1f} fps")
        if self._last_rgb is not None and not self._camera_ready():
            self._set_camera_stale("Camera delayed · last image")
            self.kinect_label.setText("Kinect: waiting for frames")
        self._refresh_controls()

    def _show_message(self, message, timeout=5000):
        self.logs_panel.append(message)
        self.statusBar().showMessage(message, timeout)

    def _on_error(self, msg: str):
        if self._closing:
            return
        self._camera_ok = False
        self.kinect_label.setText("Kinect: unavailable")
        self._set_camera_stale("Camera unavailable · reconnecting")
        self._refresh_controls()
        if self._last_rgb is None:
            self.view_label.setText(msg)
            self.scan_depth_view.setText(msg)
        self.logs_panel.append(msg, "Camera")

    def closeEvent(self, event):
        if not self._closing and not self._close_approved and not self._protect_session("close"):
            event.ignore()
            return
        if not self._closing:
            self._closing = True
            self.capture_sound.stop()
            self._reset_auto_capture_cadence()
            self._fps_timer.stop()
            if not self._camera_suspended:
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
            self._show_message(message)
            event.ignore()
            QTimer.singleShot(200, self.close)
            return
        self.logs_panel.stop()
        self.server_client.disconnect(notify=False)
        super().closeEvent(event)
