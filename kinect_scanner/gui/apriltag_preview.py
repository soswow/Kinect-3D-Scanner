"""Paint detected AprilTag outlines and numeric IDs on a calibrated preview."""

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF


USABLE_COLOR = "#53e895"
DETECTED_COLOR = "#ffd166"


def paint_apriltags(image, detections):
    """Copy before painting so recorded pixels and flow snapshots remain raw."""
    result = image.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = painter.font()
    font.setPixelSize(16)
    font.setBold(True)
    painter.setFont(font)
    try:
        for tag in detections:
            corners = np.asarray(tag["corners"], float)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                continue
            color = QColor(USABLE_COLOR if tag["usable"] else DETECTED_COLOR)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 2.5))
            painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in corners]))
            # Numeric ID only; families remain part of tracking identity.
            text = str(tag["id"])
            metrics = painter.fontMetrics()
            width = min(result.width(), metrics.horizontalAdvance(text) + 12)
            height = min(result.height(), metrics.height() + 6)
            x = float(np.clip(corners[:, 0].min(), 0, result.width() - width))
            y = float(np.clip(corners[:, 1].min() - height - 3, 0, result.height() - height))
            rect = QRectF(x, y, width, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 23, 31, 220))
            painter.drawRoundedRect(rect, 3, 3)
            painter.setPen(color)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    finally:
        painter.end()
    return result
