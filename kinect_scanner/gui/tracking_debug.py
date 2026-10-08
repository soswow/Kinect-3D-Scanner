"""Paint the actual camera-side flow on its calibrated RGB image."""

import numpy as np
from PyQt6.QtCore import QLineF, QPointF, QRectF
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
        segments = snapshot.get("trail_segments", np.empty((0, 2, 2)))
        ages = snapshot.get("trail_age_frames", np.empty(0, np.uint8))
        finite = np.isfinite(segments).all(axis=(1, 2))
        limit = snapshot.get("trail_frame_limit", VisualTracker.DEBUG_TRAIL_FRAMES)
        # Batch equal-age segments, oldest first, beneath current flow markers.
        for age in np.unique(ages)[::-1]:
            color = QColor(COLORS[VisualTracker.VERIFIED])
            color.setAlpha(round(35 + 110 * (1 - age / max(1, limit - 1))))
            painter.setPen(QPen(color, 1.1))
            painter.drawLines([
                QLineF(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
                for a, b in segments[(ages == age) & finite]
            ])
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
    tracks = snapshot.get("tracks")
    field = (
        f"Tracks {tracks['active']} · kept {tracks['retained']} · new {tracks['added']} · "
        f"retired for quality {tracks['retired']}\n"
        f"Median age {tracks['median_age_s']:.1f} s (oldest {tracks['max_age_s']:.1f} s)\n"
        f"Coverage {tracks['occupied_cells']} / {tracks['eligible_cells']} depth cells · "
        f"replenishment: {tracks['replenishment']}\n" if tracks else ""
    )
    trails = (f"Trails: {snapshot['trail_frame_count']} / {snapshot['trail_frame_limit']} camera frames "
              f"· older segments fade\n" if "trail_frame_limit" in snapshot else "")
    return (
        f"{snapshot['reason']}\n"
        f"Verified {counts[4]} / {len(status)} · flow lost {counts[0]} · "
        f"round-trip {counts[1]} · depth {counts[2]} · geometry {counts[3]}\n"
        f"New candidates {snapshot['detected']} · {snapshot['elapsed_ms']:.1f} ms{reference}\n"
        f"{field}"
        f"{trails}"
        f"LK {snapshot['window_size']} × {snapshot['window_size']} px · "
        f"pyramid 0–{snapshot['pyramid_level']} · surviving identities retained"
    )
