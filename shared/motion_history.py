"""Compact sensor evidence transported with captures, without full image recording.

Host read intervals and camera packet times retain their distinct meanings.
Bounds keep preview coalescing safe; unavailable history is explicitly reported.
"""

from collections import deque
import heapq
import math


class MotionHistory:
    WINDOW_S = 8.
    MAX_ACCELERATION = 256
    MAX_VISUAL = 256
    MAX_FEATURES_PER_OBSERVATION = 192
    MAX_FEATURE_OBSERVATIONS = 80

    def __init__(self):
        self.acceleration = deque(maxlen=self.MAX_ACCELERATION)
        self.visual = deque(maxlen=self.MAX_VISUAL)
        self.started = None

    def add_acceleration(self, sample):
        keys = ("capture_generation", "sequence", "valid", "reason", "raw_counts",
                "acceleration_m_s2", "read_start_s", "read_end_s", "host_monotonic_s",
                "gravity", "tilt_degrees", "tilt_status_raw")
        self.acceleration.append({k: sample[k] for k in keys if k in sample})

    def add_frame(self, metadata):
        stamp = metadata.get("captured_monotonic_s")
        if stamp is None or not math.isfinite(stamp):
            return
        if self.started is None:
            self.started = stamp
        tracking = metadata.get("visual_tracking", {})
        row = {"host_monotonic_s": stamp,
                            "device_timestamp_s": metadata.get("depth_device_timestamp_unwrapped_s"),
                            "rgb_depth_delta_ms": metadata.get("rgb_depth_delta_ms"),
                            "depth_host_monotonic_s": metadata.get("depth_host_monotonic_s"),
                            "host_mapping_uncertainty_s": metadata.get("host_mapping_uncertainty_s"),
                            "capture_generation": metadata.get("capture_generation"),
                            "sensor_frame_sequences": metadata.get("sensor_frame_sequences"),
                            "visual_tracking": {k: tracking[k] for k in (
                                "valid", "reason", "segment", "camera_to_local", "steps",
                                "reference_timestamp_s", "inliers", "matches", "support_fraction",
                                "median_pixel_error", "median_depth_error_m", "distributed") if k in tracking}}
        measured = tracking.get("measured_tracks")
        if measured:
            # Retained identities link successive observations. Their measured
            # axial depths are camera-side observations, not integrated poses.
            chosen = self._feature_selection(measured)
            row["feature_observations"] = {"version": 1, "image_size": measured["image_size"],
                **{key: [measured[key][i] for i in chosen] for key in ("ids", "pixels", "depths_m")}}
        self.visual.append(row)

    def _feature_selection(self, measured):
        """Keep long-lived identities across the observed image, not one patch."""
        width, height = measured["image_size"]
        cells = {}
        for i in sorted(range(len(measured["ids"])), key=lambda i: -measured["observations"][i]):
            x, y = measured["pixels"][i]
            cell = min(7, max(0, int(x*8/width))) + 8*min(5, max(0, int(y*6/height)))
            cells.setdefault(cell, deque()).append(i)
        queue = [(0, cell) for cell in cells]
        heapq.heapify(queue)
        selected = []
        while queue and len(selected) < self.MAX_FEATURES_PER_OBSERVATION:
            count, cell = heapq.heappop(queue)
            selected.append(cells[cell].popleft())
            if cells[cell]:
                heapq.heappush(queue, (count+1, cell))
        return selected

    def snapshot(self, end):
        start = max(self.started if self.started is not None else end, end - self.WINDOW_S)
        acceleration = [s for s in self.acceleration if start <= s["host_monotonic_s"] <= end]
        visual = [s for s in self.visual if start <= s["host_monotonic_s"] <= end]
        feature_rows = [s for s in visual if s.get("feature_observations")]
        feature_start = start
        if len(feature_rows) > self.MAX_FEATURE_OBSERVATIONS:
            feature_start = feature_rows[-self.MAX_FEATURE_OBSERVATIONS]["host_monotonic_s"]
            visual = [{k:v for k,v in s.items() if k != "feature_observations" or s["host_monotonic_s"] >= feature_start} for s in visual]
        # Count bounds can shorten a window at unusually high camera/read rates.
        if len(self.acceleration) == self.MAX_ACCELERATION:
            start = max(start, self.acceleration[0]["host_monotonic_s"])
        if len(self.visual) == self.MAX_VISUAL:
            start = max(start, self.visual[0]["host_monotonic_s"])
        return {"version": 1, "available_from_host_s": start, "end_host_s": end,
                "feature_observations_from_host_s": feature_start,
                "accelerometer": [s for s in acceleration if s["host_monotonic_s"] >= start],
                "visual": [s for s in visual if s["host_monotonic_s"] >= start],
                "timing_reference": "host_read_intervals_and_device_camera_packets"}


def capture_interval(history, after=None):
    """Keep every reading since the previous selected image, or report a gap."""
    start, end = history["available_from_host_s"], history["end_host_s"]
    requested = start if after is None else after
    return {**history, "requested_from_host_s": requested,
            "complete_interval": requested >= start - .001,
            "complete_feature_interval": requested >= history.get("feature_observations_from_host_s", start) - .001,
            "accelerometer": [s for s in history["accelerometer"] if s["host_monotonic_s"] > requested],
            "visual": [s for s in history["visual"] if s["host_monotonic_s"] > requested],
            "duration_s": max(0., end - requested)}
