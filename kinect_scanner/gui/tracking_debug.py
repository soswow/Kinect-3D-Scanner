"""Paint the actual camera-side flow on its calibrated RGB image."""

import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen

from shared.visual_tracking import VisualTracker

from .widgets import numpy_to_qimage


COLORS = ("#ff6262", "#ff9f43", "#e08bff", "#ffd166", "#53e895")
SEED_COLOR = "#55dfff"
MAX_WINDOWS = 24


def flow_image(snapshot, *, show_windows=False):
    """Keep pixels in the depth-aligned tracking grid, including lens remaps.

    Raw native RGB has different intrinsics/parallax. Scaling these vectors
    onto that image would misplace them. Work on a copy so capture stays raw.
    """
    image = numpy_to_qimage(snapshot["image"]).copy()
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    try:
        corners = snapshot.get("corners")
        if corners is not None:
            painter.setPen(QPen(QColor(SEED_COLOR), 2))
            for x, y in corners.reshape(-1, 2):
                painter.drawPoint(QPointF(float(x), float(y)))
        source = snapshot.get("source", np.empty((0, 2)))
        target = snapshot.get("target", np.empty((0, 2)))
        statuses = snapshot.get("status", np.empty(0, np.uint8))
        for a, b, status in zip(source, target, statuses):
            if not np.isfinite(a).all():
                continue
            painter.setPen(QPen(QColor(COLORS[status]), 1.4))
            start = QPointF(float(a[0]), float(a[1]))
            if (status == VisualTracker.FLOW_LOST or not np.isfinite(b).all()
                    or not (0 <= b[0] < image.width() and 0 <= b[1] < image.height())):
                painter.drawLine(start + QPointF(-3, -3), start + QPointF(3, 3))
                painter.drawLine(start + QPointF(-3, 3), start + QPointF(3, -3))
                continue
            end = QPointF(float(b[0]), float(b[1]))
            painter.drawLine(start, end)
            painter.drawEllipse(end, 1.8, 1.8)
            vector = b - a
            length = float(np.linalg.norm(vector))
            if length >= 3:
                direction = vector / length
                wing = np.array([-direction[1], direction[0]])
                for sign in (-1, 1):
                    tip = b - direction * min(5, length) + sign * wing * 2.5
                    painter.drawLine(end, QPointF(float(tip[0]), float(tip[1])))
        if show_windows:
            candidates = np.flatnonzero(statuses == VisualTracker.VERIFIED)
            if not len(candidates):
                candidates = np.flatnonzero(statuses == VisualTracker.GEOMETRY_REJECTED)
            selected = candidates[np.linspace(0, len(candidates) - 1,
                                             min(MAX_WINDOWS, len(candidates)), dtype=int)]
            size = snapshot["window_size"]
            painter.setPen(QPen(QColor("#ffffff"), 0.8))
            for index in selected:
                x, y = target[index]
                if np.isfinite([x, y]).all():
                    painter.drawRect(QRectF(float(x - size / 2), float(y - size / 2), size, size))
    finally:
        painter.end()
    return image


def flow_summary(snapshot):
    status = snapshot.get("status", np.empty(0, np.uint8))
    counts = np.bincount(status, minlength=5)
    age = snapshot.get("reference_age_ms")
    reference = f" · reference {age:.0f} ms ago" if age is not None else ""
    return (
        f"{snapshot['reason']}\n"
        f"Verified {counts[4]} / {len(status)} · flow lost {counts[0]} · "
        f"round-trip {counts[1]} · depth {counts[2]} · geometry {counts[3]}\n"
        f"Detected {snapshot['detected']} · {snapshot['elapsed_ms']:.1f} ms{reference}\n"
        f"LK {snapshot['window_size']} × {snapshot['window_size']} px · "
        f"pyramid 0–{snapshot['pyramid_level']} · fresh corners on each accepted reference"
    )
