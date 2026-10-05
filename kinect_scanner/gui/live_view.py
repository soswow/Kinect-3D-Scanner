"""Persistent, bounded 3D feedback rendered with Qt, without an OpenGL context."""

import time

import numpy as np
from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QCheckBox, QWidget

from shared.config import LIVE_MAX_POINTS
from shared.settings import CameraCalibration


class LiveView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 240)
        self.setToolTip(
            "Follow the last tracked scanner view; uncheck Follow scanner to drag "
            "to orbit and wheel to zoom; double-click for color/shape"
        )
        self.follow_cb = QCheckBox("Follow scanner", self)
        self.follow_cb.setStyleSheet("color: #e1eaf0; background: #15202b;")
        self.follow_cb.toggled.connect(self.update)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
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
        self.colored = True
        self._drag = None
        self._received = None
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
        self.update()

    def resizeEvent(self, event):
        self.follow_cb.adjustSize()
        self.follow_cb.move(12, 30)
        super().resizeEvent(event)

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
        width, height = self.width(), self.height()
        if len(self.points) and self.center is not None:
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
            painter.drawImage(0, 0, image)
        painter.setPen(QColor("#e1eaf0"))
        mode = "last tracked view" if self.follow_cb.isChecked() else "orbit view"
        painter.drawText(12, 22, f"Live fused point cloud · {mode}")
        s = self.snapshot
        result = s.get("result", {})
        age = (
            "waiting for frames"
            if self._received is None
            else f"last update {time.monotonic() - self._received:.1f}s ago"
        )
        state = "tracking accepted" if result.get("success") else "tracking skipped"
        if not result:
            state = "waiting"
        lines = [
            f"{s.get('frame_count', 0)} integrated · {s.get('pending_count', 0)} pending · {age}",
            f"{state} · processing {result.get('elapsed_ms', 0):.0f} ms/frame · {len(self.points):,} points",
            f"{s.get('skipped_count', 0)} skipped · queue age {s.get('pending_age_s', 0):.1f}s · geometry {s.get('geometry_frame_count', 0)}",
        ]
        for i, line in enumerate(lines):
            text = painter.fontMetrics().elidedText(
                line, Qt.TextElideMode.ElideRight, max(1, width - 24)
            )
            painter.drawText(12, height - 84 + 16 * i, text)
        painter.drawText(
            QRect(12, height - 42, max(1, width - 24), 40),
            Qt.TextFlag.TextWordWrap.value,
            s.get("guidance", "Move slowly with overlap"),
        )

    def mousePressEvent(self, event):
        if not self.follow_cb.isChecked():
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
        self.colored = not self.colored
        self.update()
