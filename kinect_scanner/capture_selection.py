"""Choose a fresh, sharp RGB-D observation without retaining camera history."""

from collections import deque

import cv2
import numpy as np


class MotionCapturePolicy:
    """Retain overlap before the ordinary cadence skips a large/weak step."""
    def __init__(self):
        self.reference = None

    def captured(self, metadata):
        self.reference = metadata

    def needed(self, metadata):
        if self.reference is None or not metadata.get("visual_tracking"):
            return False
        elapsed = metadata.get("captured_monotonic_s", 0) - self.reference.get("captured_monotonic_s", 0)
        if elapsed < .15:
            return False
        a, b = (m.get("visual_tracking", {}) for m in (self.reference, metadata))
        if not (a.get("valid") and b.get("valid") and a.get("segment") == b.get("segment")):
            return elapsed >= .2
        try:
            delta = np.linalg.inv(np.asarray(a["camera_to_local"], float)) @ np.asarray(b["camera_to_local"], float)
            angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
            return bool(np.linalg.norm(delta[:3, 3]) >= .1 or angle >= 8.)
        except (KeyError, ValueError, TypeError, np.linalg.LinAlgError):
            return elapsed >= .2


def sharpness(rgb):
    # A fixed scoring resolution makes the two RGB modes comparable and keeps
    # this cheap enough for every camera arrival. This is a relative score.
    height, width = rgb.shape[:2]
    scale = min(1.0, 320 / width)
    image = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


class CaptureSelector:
    MAX_AGE_S = 0.3

    def __init__(self):
        self.frames = deque(maxlen=5)

    def clear(self):
        self.frames.clear()

    def offer(self, rgb, depth, metadata, captured_at):
        score = sharpness(rgb)
        self.frames.append((captured_at, score, rgb, depth, dict(metadata)))

    def choose(self, now, after=None):
        candidates = [f for f in self.frames if 0 <= now - f[0] <= self.MAX_AGE_S
                      and (after is None or f[0] > after)]
        if not candidates:
            return None
        # Prefer valid continuous motion when available, then compare sharpness.
        # Missing motion metadata (capture-only mode) is neutral, not a failure.
        selected = max(candidates, key=lambda f: (
            f[4].get('visual_tracking', {}).get('valid', True), f[1], f[0]))
        stamp, score, rgb, depth, metadata = selected
        metadata = {**metadata, 'capture_selection': {
            'sharpness': score, 'candidates': len(candidates), 'age_ms': (now - stamp) * 1000}}
        return rgb, depth, metadata
