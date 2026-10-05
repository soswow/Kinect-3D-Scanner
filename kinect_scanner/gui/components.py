"""Small layout components shared by the scanner workflow."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 4, 0, 4)
        self.content.hide()
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self.set_expanded)

    def set_expanded(self, expanded):
        self.content.setVisible(expanded)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class CameraPreview(QLabel):
    """Aspect-fit image which resizes independently of the last pixmap."""

    def __init__(self, parent=None):
        super().__init__("Connecting to Kinect…", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: #1e1e1e; color: #ddd;")
        self._image = None
        self.stale_label = QLabel("Camera unavailable · last image", self)
        self.stale_label.setWordWrap(True)
        self.stale_label.setStyleSheet("background: #352b15; color: #ffe4a3; padding: 8px;")
        self.stale_label.hide()

    def set_image(self, image):
        self._image = image.copy()
        self.set_stale(False)
        self._refresh_image()

    def set_stale(self, stale, message="Camera unavailable · last image"):
        self.stale_label.setText(message)
        self.stale_label.setVisible(stale)
        self.stale_label.setGeometry(8, 8, max(1, self.width() - 16), 50)
        self.stale_label.raise_()

    def _refresh_image(self):
        if self._image is not None:
            self.setPixmap(QPixmap.fromImage(self._image).scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        self._refresh_image()
        self.stale_label.setGeometry(8, 8, max(1, self.width() - 16), 50)
        super().resizeEvent(event)
