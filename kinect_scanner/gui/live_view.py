"""Persistent, bounded 3D feedback rendered with Qt, without an OpenGL context."""

import time

import numpy as np
from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget


class LiveView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 240)
        self.setToolTip("Drag to orbit; wheel to zoom; double-click for color/shape")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(1000)
        self.reset()

    def reset(self):
        self.snapshot = {}
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
        self.points = points[:5000]
        self.colors = colors[:5000]
        self.snapshot = snapshot
        if self.center is None and len(self.points):
            lo, hi = np.percentile(self.points, [1, 99], axis=0)
            self.center = (lo + hi) / 2
            self.radius = max(0.1, float(np.linalg.norm(hi - lo) / 2))
        self._received = time.monotonic()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15202b"))
        width, height = self.width(), self.height()
        if len(self.points) and self.center is not None:
            cy, sy = np.cos(self.yaw), np.sin(self.yaw)
            cp, sp = np.cos(self.pitch), np.sin(self.pitch)
            rotation = np.array(
                [[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]]
            )
            points = (self.points - self.center) @ rotation.T
            scale = min(width, height) * 0.42 * self.zoom / self.radius
            xy = np.rint(points[:, :2] * scale + [width / 2, height / 2]).astype(int)
            inside = (
                (xy[:, 0] >= 1)
                & (xy[:, 0] < width - 1)
                & (xy[:, 1] >= 1)
                & (xy[:, 1] < height - 1)
            )
            order = np.argsort(points[:, 2])[::-1]
            order = order[inside[order]]
            pixels = np.full((height, width, 3), [21, 32, 43], dtype=np.uint8)
            if self.colored and len(self.colors) == len(self.points):
                colors = np.clip(self.colors * 255, 0, 255).astype(np.uint8)
            else:
                shade = (points[:, 2] - points[:, 2].min()) / max(
                    float(np.ptp(points[:, 2])), 0.01
                )
                colors = np.clip(
                    np.array([100, 195, 225]) * (1 - shade[:, None] * 0.6), 0, 255
                ).astype(np.uint8)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                pixels[xy[order, 1] + dy, xy[order, 0] + dx] = colors[order]
            image = QImage(
                pixels.data,
                width,
                height,
                pixels.strides[0],
                QImage.Format.Format_RGB888,
            )
            painter.drawImage(0, 0, image)
        painter.setPen(QColor("#e1eaf0"))
        painter.drawText(12, 22, "Live fused surface · drag to orbit")
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
            f"{state} · processing {result.get('elapsed_ms', 0):.0f} ms/frame",
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
        self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position() - self._drag
            self.yaw += delta.x() * 0.008
            self.pitch = np.clip(self.pitch + delta.y() * 0.008, -1.5, 1.5)
            self._drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def wheelEvent(self, event):
        self.zoom = np.clip(self.zoom * np.exp(event.angleDelta().y() / 1200), 0.1, 20)
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.colored = not self.colored
        self.update()
