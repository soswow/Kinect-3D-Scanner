"""Overhead orientation aid with accepted poses and the last good RGB view."""

import base64

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget


class TrackingOverview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 140)
        self.setAccessibleName("Overhead camera trajectory and recovery reference")
        self.setToolTip("Dots are accepted camera positions. The red camera is the last good view. "
                        "While tracking is lost, your current position is unknown. Up is estimated from a plane.")
        self.snapshot = {}
        self.reference = QImage()

    def set_snapshot(self, snapshot):
        self.snapshot = snapshot
        encoded = snapshot.get("last_tracked_rgb_png")
        self.reference = QImage.fromData(base64.b64decode(encoded)) if encoded else QImage()
        self.update()

    def projection_basis(self):
        up = np.asarray(self.snapshot.get("world_up", [0, -1, 0]), float)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
            up = np.array([0., -1., 0.])
        up /= np.linalg.norm(up)
        right = np.array([1., 0., 0.])
        right -= up * right.dot(up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0., 0., 1.])
            right -= up * right.dot(up)
        right /= np.linalg.norm(right)
        return np.column_stack((right, np.cross(up, right)))

    def accepted_poses(self):
        poses = []
        for entry in self.snapshot.get("trajectory", [])[:500]:
            pose = np.asarray(entry.get("camera_to_world"), float)
            if pose.shape == (4, 4) and np.isfinite(pose).all():
                poses.append((entry["index"], pose))
        return poses

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101b25"))
        painter.setPen(QColor("#dce9f1"))
        lost = self.snapshot.get("fusion_paused", False)
        painter.drawText(8, 17, "Overhead" if self.snapshot.get("up_estimated") else "Overview · camera up")
        anchor_index = self.snapshot.get("last_tracked_index")
        if lost and not self.reference.isNull():
            painter.drawText(int(self.width() * 0.55) + 4, 17, "Last good view")
        poses = self.accepted_poses()
        if not poses:
            painter.drawText(8, 42, "Waiting for first tracked view")
            return
        basis = self.projection_basis()
        points = np.asarray(self.snapshot.get("points", []), float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)][::5]
        cameras = np.array([pose[:3, 3] @ basis for _, pose in poses])
        cloud = points @ basis
        # Cap distant depth points for a useful subject-and-path overview.
        bounds = cameras if not len(cloud) else np.vstack((cameras, np.percentile(cloud, [5, 95], axis=0)))
        lo, hi = bounds.min(axis=0), bounds.max(axis=0)
        width = self.width() * (0.55 if lost and not self.reference.isNull() else 1)
        area = QRectF(12, 30, max(30, width - 24), max(30, self.height() - 58))
        scale = min(area.width(), area.height()) / max(float(np.max(hi - lo)), 0.3) * 0.85
        center = (lo + hi) / 2

        def project(xy):
            return (xy - center) * [scale, -scale] + [area.center().x(), area.center().y()]

        painter.save()
        painter.setClipRect(area)
        painter.setPen(QPen(QColor("#426172"), 1))
        for xy in project(cloud):
            painter.drawPoint(QPointF(*xy))
        xy = project(cameras)
        painter.setPen(QPen(QColor("#6dcbe8"), 1.5))
        # Never draw an odometry edge across skipped frames.
        for n in range(1, len(poses)):
            if poses[n][0] == poses[n - 1][0] + 1:
                painter.drawLine(QPointF(*xy[n - 1]), QPointF(*xy[n]))
        for n, (index, pose) in enumerate(poses):
            is_anchor = index == anchor_index
            color = QColor("#ff625b" if lost and is_anchor else "#6dcbe8")
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(*xy[n]), 2, 2)
            direction = pose[:3, 2] @ basis * [1, -1]
            length = np.linalg.norm(direction)
            if length > 1e-6:
                direction /= length
                side = np.array([-direction[1], direction[0]])
                triangle = [xy[n] + direction * 10, xy[n] - direction * 4 + side * 4,
                            xy[n] - direction * 4 - side * 4]
                painter.drawPolyline(QPolygonF([QPointF(*p) for p in triangle + triangle[:1]]))
            if is_anchor:
                painter.drawEllipse(QPointF(*xy[n]), 8, 8)
        painter.restore()
        if lost and not self.reference.isNull():
            rect = QRectF(width + 4, 30, self.width() - width - 12, self.height() - 57)
            size = self.reference.size().scaled(int(rect.width()), int(rect.height()), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF(rect.x(), rect.y(), size.width(), size.height())
            painter.drawImage(target, self.reference)
        painter.setPen(QColor("#ff8a83" if lost else "#b4c9d6"))
        frame = "none" if anchor_index is None else str(anchor_index + 1)
        painter.drawText(8, self.height() - 8, f"Return to Frame {frame} · current pose unknown" if lost
                         else f"Last tracked Frame {frame} · accepted cameras")
