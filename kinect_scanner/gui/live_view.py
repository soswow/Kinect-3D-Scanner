"""Persistent, bounded 3D feedback rendered with Qt, without an OpenGL context."""

import time

import numpy as np
from PyQt6.QtCore import QRect, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

from shared.config import LIVE_MAX_POINTS
from shared.settings import CameraCalibration


class LiveView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setToolTip("Orbit: drag to rotate and wheel to zoom. Fit View frames the current cloud.")
        self.follow_cb = QCheckBox("Follow scanner", self)
        self.follow_cb.hide()  # Retained for integrations using the original API.
        self.follow_button = QPushButton("Follow", self)
        self.orbit_button = QPushButton("Orbit", self)
        self.color_button = QPushButton("Color", self)
        self.shape_button = QPushButton("Shape", self)
        self.fit_button = QPushButton("Fit View", self)
        self.details_button = QPushButton("Details", self)
        self.details_button.setCheckable(True)
        for names in (("follow_button", "orbit_button"), ("color_button", "shape_button")):
            group = QButtonGroup(self)
            for name in names:
                button = getattr(self, name)
                button.setCheckable(True)
                group.addButton(button)
        for button, label in ((self.follow_button, "Follow last tracked scanner view"),
                              (self.orbit_button, "Orbit the fused point cloud"),
                              (self.color_button, "Display captured colors"),
                              (self.shape_button, "Display shape shading"),
                              (self.fit_button, "Fit entire current point cloud"),
                              (self.details_button, "Show reconstruction diagnostics")):
            button.setAccessibleName(label)
            button.setToolTip(label)
        self.follow_button.clicked.connect(lambda: self.follow_cb.setChecked(True))
        self.orbit_button.clicked.connect(lambda: self.follow_cb.setChecked(False))
        self.follow_cb.toggled.connect(self._sync_view_mode)
        self.color_button.clicked.connect(lambda: self._set_colored(True))
        self.shape_button.clicked.connect(lambda: self._set_colored(False))
        self.fit_button.clicked.connect(self.fit_view)
        self.details_button.toggled.connect(self._show_details)
        self.panel = QWidget(self)
        self.panel.setStyleSheet(
            "QWidget { color: #e1eaf0; background: #15202b; }"
            "QPushButton { padding: 3px 6px; border: 1px solid #536574; border-radius: 3px; }"
            "QPushButton:checked { background: #32566c; border-color: #88cde8; }"
        )
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        self.title_label = QLabel("Fused point cloud", self.panel)
        layout.addWidget(self.title_label)
        for buttons in ((self.follow_button, self.orbit_button, self.fit_button),
                        (self.color_button, self.shape_button, self.details_button)):
            row = QHBoxLayout()
            row.setSpacing(4)
            for button in buttons:
                row.addWidget(button)
            layout.addLayout(row)
        self.guidance_label = QLabel(self.panel)
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setAccessibleName("Scanning guidance")
        self.guidance_label.setStyleSheet("font-weight: bold; color: #c8e9f6;")
        layout.addWidget(self.guidance_label)
        self.status_label = QLabel(self.panel)
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Reconstruction status")
        layout.addWidget(self.status_label)
        self.details_label = QLabel(self.panel)
        self.details_label.setWordWrap(True)
        self.details_label.setAccessibleName("Reconstruction diagnostics")
        self.details_label.hide()
        layout.addWidget(self.details_label)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self.reset()

    def reset(self):
        self.snapshot = {}
        self.camera_to_world = np.eye(4)
        self.camera = CameraCalibration()
        self.follow_cb.setChecked(True)
        self.points = np.empty((0, 3))
        self.colors = np.empty((0, 3))
        self.center = None
        self.radius = 1
        self.yaw = self.pitch = 0
        self.zoom = 1
        self._set_colored(True)
        self._drag = None
        self._received = None
        self._refresh_labels()
        self.update()

    def set_snapshot(self, snapshot):
        points = np.asarray(snapshot.get("points", []), dtype=float).reshape(-1, 3)
        colors = np.asarray(snapshot.get("colors", []), dtype=float).reshape(-1, 3)
        # Bound both network payload and GUI work even with an older server.
        step = max(1, int(np.ceil(len(points) / LIVE_MAX_POINTS)))
        self.points = points[::step]
        self.colors = colors[::step]
        self.snapshot = snapshot
        pose = np.asarray(snapshot.get("camera_to_world", self.camera_to_world))
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            self.camera_to_world = pose.copy()
        if "camera" in snapshot:
            self.camera = CameraCalibration(**snapshot["camera"])
        if self.center is None and len(self.points):
            lo, hi = np.percentile(self.points, [1, 99], axis=0)
            self.center = (lo + hi) / 2
            self.radius = max(0.1, float(np.linalg.norm(hi - lo) / 2))
        self._received = time.monotonic()
        self._refresh_labels()
        self.update()

    def _sync_view_mode(self, follow):
        self.follow_button.setChecked(follow)
        self.orbit_button.setChecked(not follow)
        self.update()

    def _set_colored(self, colored):
        self.colored = colored
        self.color_button.setChecked(colored)
        self.shape_button.setChecked(not colored)
        self.update()

    def fit_view(self):
        """Reframe all current finite points, including points omitted for rendering."""
        points = np.asarray(self.snapshot.get("points", []), dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            lo, hi = points.min(axis=0), points.max(axis=0)
            self.center = (lo + hi) / 2
            self.radius = max(0.1, float(np.linalg.norm(hi - lo) / 2))
        self.yaw = self.pitch = 0
        self.zoom = 1
        self.follow_cb.setChecked(False)
        self.update()

    def _show_details(self, visible):
        self.details_label.setVisible(visible)
        self._layout_panel()
        self.update()

    @property
    def drawing_rect(self):
        """Point-cloud viewport in widget coordinates, below the visible panel."""
        top = self.panel.geometry().bottom() + 1
        return QRect(0, top, self.width(), max(0, self.height() - top))

    def _tick(self):
        self._refresh_labels()
        self.update()

    def _layout_panel(self):
        self.panel.setFixedWidth(self.width())
        self.panel.adjustSize()
        self.panel.move(0, 0)

    def resizeEvent(self, event):
        self._layout_panel()
        super().resizeEvent(event)

    def _refresh_labels(self):
        s = self.snapshot
        result = s.get("result", {})
        self.guidance_label.setText(s.get("guidance", "Move slowly with overlap"))
        self.status_label.setText(
            f"{s.get('frame_count', 0)} integrated · {s.get('pending_count', 0)} pending"
        )
        state = ("tracking accepted" if result.get("success") else "tracking skipped") if result else "waiting"
        age = "waiting for frames" if self._received is None else f"last update {time.monotonic() - self._received:.1f}s ago"
        self.details_label.setText(
            f"{state} · {age}\n"
            f"Processing: {result.get('elapsed_ms', 0):.0f} ms/frame · displayed: {len(self.points):,} points\n"
            f"Skipped: {s.get('skipped_count', 0)} · queue age: {s.get('pending_age_s', 0):.1f}s · geometry frames: {s.get('geometry_frame_count', 0)}"
        )
        self._layout_panel()

    def _project_points(self, width, height):
        """Return visible pixel positions, depths, and source point indices."""
        if self.follow_cb.isChecked():
            # The pose maps camera to world; invert its rigid transform for
            # projection. Camera coordinates are +X right, +Y down, +Z forward.
            pose = self.camera_to_world
            points = (self.points - pose[:3, 3]) @ pose[:3, :3]
            indices = np.flatnonzero(np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-4))
            points = points[indices]
            c = self.camera
            uv = points[:, :2] / points[:, 2, None] * [c.fx, c.fy] + [c.cx, c.cy]
            in_frame = (
                (uv[:, 0] >= 0) & (uv[:, 0] < c.width)
                & (uv[:, 1] >= 0) & (uv[:, 1] < c.height)
            )
            indices, points, uv = indices[in_frame], points[in_frame], uv[in_frame]
            scale = min(width / c.width, height / c.height)
            offset = [(width - c.width * scale) / 2, (height - c.height * scale) / 2]
            xy = np.rint(uv * scale + offset).astype(int)
        else:
            cy, sy = np.cos(self.yaw), np.sin(self.yaw)
            cp, sp = np.cos(self.pitch), np.sin(self.pitch)
            rotation = np.array(
                [[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]]
            )
            points = (self.points - self.center) @ rotation.T
            scale = min(width, height) * 0.42 * self.zoom / self.radius
            xy = np.rint(points[:, :2] * scale + [width / 2, height / 2]).astype(int)
            indices = np.arange(len(points))
        inside = (
            (xy[:, 0] >= 1) & (xy[:, 0] < width - 1)
            & (xy[:, 1] >= 1) & (xy[:, 1] < height - 1)
        )
        return xy[inside], points[inside, 2], indices[inside]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15202b"))
        viewport = self.drawing_rect
        width, height = viewport.width(), viewport.height()
        painter.setClipRect(viewport)
        if width > 2 and height > 2 and len(self.points) and self.center is not None:
            xy, depth, indices = self._project_points(width, height)
            order = np.argsort(depth)[::-1]
            pixels = np.full((height, width, 3), [21, 32, 43], dtype=np.uint8)
            if self.colored and len(self.colors) == len(self.points):
                colors = np.clip(self.colors[indices] * 255, 0, 255).astype(np.uint8)
            else:
                shade = (depth - (depth.min() if len(depth) else 0)) / max(
                    float(np.ptp(depth)) if len(depth) else 0, 0.01
                )
                colors = np.clip(
                    np.array([100, 195, 225]) * (1 - shade[:, None] * 0.6), 0, 255
                ).astype(np.uint8)
            # Draw 3x3 splats. Resolve overlapping pixels in depth order,
            # including neighbors, so a distant dot cannot cover a near one.
            offsets = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            splats = xy[order, None, :] + offsets
            pixel_ids = (splats[:, :, 1] * width + splats[:, :, 0]).ravel()
            _, reverse_indices = np.unique(pixel_ids[::-1], return_index=True)
            winners = len(pixel_ids) - 1 - reverse_indices
            pixels.reshape(-1, 3)[pixel_ids[winners]] = colors[order[winners // 9]]
            image = QImage(
                pixels.data,
                width,
                height,
                pixels.strides[0],
                QImage.Format.Format_RGB888,
            )
            painter.drawImage(viewport.topLeft(), image)

    def mousePressEvent(self, event):
        if not self.follow_cb.isChecked() and self.drawing_rect.contains(event.position().toPoint()):
            self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is not None and not self.follow_cb.isChecked():
            delta = event.position() - self._drag
            self.yaw += delta.x() * 0.008
            self.pitch = np.clip(self.pitch + delta.y() * 0.008, -1.5, 1.5)
            self._drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def wheelEvent(self, event):
        if not self.follow_cb.isChecked():
            self.zoom = np.clip(self.zoom * np.exp(event.angleDelta().y() / 1200), 0.1, 20)
            self.update()

    def mouseDoubleClickEvent(self, event):
        self._set_colored(not self.colored)
