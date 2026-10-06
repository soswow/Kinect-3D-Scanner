"""Automatic capture cadence and frames awaiting live processing."""

import math


class CapturePacer:
    HEADROOM = 1.15

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_capture_at = None
        self._processing_seconds = 0.0
        self._completion_seconds = 0.0
        self._processed_count = 0
        self._stored_count = 0
        self._timed_count = 0
        # A capture stays here across upload, acknowledgement and processing.
        self._pending = {}

    @property
    def pending_count(self):
        return len(self._pending)

    @property
    def processed_count(self):
        return self._processed_count

    @property
    def outstanding_count(self):
        # Indexed captures are already included in the server's stored count.
        # Unacknowledged uploads count too, even if queue.qsize() is now zero.
        return max(0, self._stored_count - self._processed_count) + sum(
            index is None for index, _ in self._pending.values()
        )

    @staticmethod
    def _smooth(previous, sample):
        if previous == 0:
            return sample
        # Respond promptly to slowdowns; regain speed gradually.
        return previous + (0.5 if sample > previous else 0.1) * (sample - previous)

    def interval_seconds(self, minimum, fps, *, adaptive=True):
        seconds = minimum
        if adaptive:
            seconds = max(seconds, self.HEADROOM * max(
                self._processing_seconds, self._completion_seconds
            ))
        # Round up to a whole camera frame, preserving the minimum delay.
        return math.ceil(seconds * fps - 1e-9) / fps

    def ready(self, now, minimum, fps, *, adaptive=True):
        return self.last_capture_at is None or (
            now - self.last_capture_at + 1e-9
            >= self.interval_seconds(minimum, fps, adaptive=adaptive)
        )

    def captured(self, frame_id, now, *, live):
        self.last_capture_at = now
        if live:
            self._pending[frame_id] = (None, now)

    def _completed(self, frame_id, now, *, learn=True):
        _, captured_at = self._pending.pop(frame_id)
        if learn:
            self._completion_seconds = self._smooth(
                self._completion_seconds, max(0, now - captured_at)
            )

    def acknowledge(self, acknowledgements, now):
        for ack in acknowledgements:
            frame_id = ack.get("frame_id")
            if frame_id not in self._pending:
                continue  # Live feedback can arrive before the HTTP response.
            if not ack.get("success"):
                del self._pending[frame_id]
                continue
            index = ack.get("index")
            if index is not None:
                self._stored_count = max(self._stored_count, index + 1)
            if index is not None and index < self._processed_count:
                self._completed(frame_id, now)
            else:
                self._pending[frame_id] = (index, self._pending[frame_id][1])

    def observe(self, snapshot, now, *, learn_completion=True):
        result = snapshot.get("result", {})
        processed = snapshot.get("processed_count", (
            snapshot["stored_count"] - snapshot["unprocessed_count"]
            if "unprocessed_count" in snapshot else result.get("index", -1) + 1
        ))
        self._processed_count = max(self._processed_count, processed)
        self._stored_count = max(self._stored_count, snapshot.get("stored_count", processed))
        if processed > self._timed_count:
            # New servers average every frame, including periodic model refresh.
            # Older servers still provide the latest frame's full processing time.
            sample = snapshot.get("processing_interval_s", result.get("elapsed_ms", 0) / 1000)
            if math.isfinite(sample) and sample > 0:
                self._processing_seconds = self._smooth(self._processing_seconds, sample)
            self._timed_count = processed

        frame_id = result.get("metadata", {}).get("frame_id")
        # Processing is sequential. A matching result also completes earlier
        # local uploads whose HTTP acknowledgements have not arrived yet.
        pending_ids = list(self._pending)
        completed_ids = (
            pending_ids[:pending_ids.index(frame_id) + 1]
            if frame_id in self._pending else []
        )
        for key, (index, _) in self._pending.items():
            if index is not None and index < self._processed_count and key not in completed_ids:
                completed_ids.append(key)
        for key in completed_ids:
            self._completed(key, now, learn=learn_completion)
