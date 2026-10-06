"""Live processing must preserve the existing cancellation and command barriers."""

import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import asyncio
import threading
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from scanner_server import app as server
from scanner_server.engine import ScanEngine
from shared.config import LIVE_MAX_POINTS
from shared.live import decode_live_geometry
from shared.protocol import pack_frame, pack_frames
from shared.sensor_calibration import load_calibration
from tests.test_quality import scene_frames as metric_scene_frames


def scene_frames(count):
    """Express the synthetic surfaces as Kinect-native disparity observations."""
    calibration = load_calibration()
    frames = []
    for rgb, metric, pose in metric_scene_frames(count, camera=calibration.depth):
        # Put the synthetic scene in the calibration's measured 0.8–1.6 m
        # range. Scale both scene depths and the known camera translations.
        metric = metric.astype(float) * 0.7
        pose = pose.copy()
        pose[:3, 3] *= 0.7
        raw = np.full(metric.shape, 2047, np.uint16)
        valid = metric > 0
        raw[valid] = np.rint(
            (calibration.scale * 1000 / metric[valid] - calibration.b)
            / calibration.a_per_code
        ).astype(np.uint16)
        frames.append((rgb, raw, pose))
    return frames


class LiveApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original = server.engine
        server.engine = ScanEngine(device="cpu")
        server._build_lock = asyncio.Lock()
        server._live_task = None
        server._latest_live = None
        server._shutting_down = False
        server._exclusive = False
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://scanner"
        )

    async def asyncTearDown(self):
        server._shutting_down = True
        if server._live_task is not None:
            server._live_task.cancel()
            try:
                await server._live_task
            except asyncio.CancelledError:
                pass
        await self.http.aclose()
        server.engine = self.original
        server._shutting_down = False
        server._latest_live = None
        server._exclusive = False

    async def test_live_batch_acknowledgements_and_bounded_snapshot(self):
        reset = (
            await self.http.post("/api/scan/reset", json={"live_reconstruction": True, "rgb_mode": "rgb_low_res"})
        ).json()
        frames = scene_frames(2)
        payload = [
            (rgb, depth, {"frame_id": i}) for i, (rgb, depth, _) in enumerate(frames)
        ]
        payload.append(payload[0])  # duplicate must not create a recording/pose index
        result = (
            await self.http.post("/api/scan/frames", content=pack_frames(payload, compression_level=0))
        ).json()
        self.assertEqual([True, True, False], [r["success"] for r in result["results"]])
        self.assertEqual(
            [0, 1], [r["index"] for r in result["results"] if r["success"]]
        )
        task = server._live_task
        self.assertIsNotNone(task)
        await asyncio.wait_for(asyncio.shield(task), 20)
        snapshot = server._latest_live
        self.assertEqual(reset["session_id"], snapshot["session_id"])
        self.assertEqual(2, snapshot["frame_count"])
        self.assertEqual(0, snapshot["pending_count"])
        self.assertGreater(len(snapshot["points"]), 5000)
        self.assertLessEqual(len(snapshot["points"]), LIVE_MAX_POINTS)
        self.assertEqual(len(snapshot["points"]), len(snapshot["colors"]))
        self.assertEqual(server.engine.settings.to_dict()["camera"], snapshot["camera"])
        np.testing.assert_array_equal(snapshot["camera_to_world"], server.engine.cumulative_T)
        with patch.object(server.engine, "vbg") as volume:
            volume.extract_point_cloud.side_effect = AssertionError(
                "Feedback must reuse cached extraction"
            )
            limited = server.engine.live_snapshot(max_points=200)
        self.assertLessEqual(len(limited["points"]), 200)
        report = (await self.http.get("/api/scan/diagnostics")).json()
        self.assertEqual(2, len(report["poses"]))
        self.assertIn("fusion", report["stage_totals_ms"])

        # A rejected frame must keep the previous scanner viewpoint, even
        # though the latest tracking result and processing counts advance.
        tracked_pose = np.asarray(snapshot["camera_to_world"])
        rgb, depth, _ = frames[-1]
        await self.http.post("/api/scan/frame", content=pack_frame(rgb, np.full_like(depth, 2047)))
        await asyncio.wait_for(asyncio.shield(server._live_task), 20)
        skipped = server._latest_live
        self.assertFalse(skipped["result"]["success"])
        self.assertEqual(2, skipped["frame_count"])
        np.testing.assert_array_equal(skipped["camera_to_world"], tracked_pose)

    async def test_build_waits_for_live_frame_instead_of_rejecting(self):
        await self.http.post("/api/scan/reset", json={"live_reconstruction": True, "rgb_mode": "rgb_low_res"})
        entered, release = threading.Event(), threading.Event()
        original = server.engine.process_frames
        events = []

        def processing(*args, **kwargs):
            if kwargs.get("max_frames") == 1:
                events.append("live-start")
                entered.set()
                release.wait(5)
                result = original(*args, **kwargs)
                events.append("live-end")
                return result
            events.append("build-process")
            return original(*args, **kwargs)

        rgb, depth, _ = scene_frames(1)[0]
        with patch.object(server.engine, "process_frames", side_effect=processing):
            await self.http.post("/api/scan/frame", content=pack_frame(rgb, depth))
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(entered.is_set())
            build = asyncio.create_task(self.http.post("/api/scan/build"))
            try:
                await asyncio.sleep(0.05)
                self.assertFalse(build.done())
            finally:
                release.set()
            result = (await build).json()
        self.assertNotEqual("Build already in progress", result.get("message"))
        self.assertEqual(["live-start", "live-end", "build-process"], events)

    async def test_reset_waits_for_native_live_processing(self):
        old_session = (
            await self.http.post("/api/scan/reset", json={"live_reconstruction": True, "rgb_mode": "rgb_low_res"})
        ).json()["session_id"]
        entered, release = threading.Event(), threading.Event()
        original = server.engine.process_frames

        def processing(*args, **kwargs):
            entered.set()
            release.wait(5)
            return original(*args, **kwargs)

        rgb, depth, _ = scene_frames(1)[0]
        with patch.object(server.engine, "process_frames", side_effect=processing):
            await self.http.post("/api/scan/frame", content=pack_frame(rgb, depth))
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(entered.is_set())
            reset = asyncio.create_task(self.http.post("/api/scan/reset", json={}))
            try:
                await asyncio.sleep(0.05)
                self.assertFalse(reset.done())
            finally:
                release.set()
            new_session = (await reset).json()["session_id"]
        self.assertNotEqual(old_session, new_session)
        self.assertEqual(0, server.engine.frame_count)
        self.assertEqual(0, server.engine.stored_count)

    async def test_packed_and_legacy_subscribers_receive_the_same_snapshot(self):
        import json

        class Socket:
            def __init__(self, packed):
                self.query_params = {"geometry": "xyzrgb-f32le"} if packed else {}
                self.messages = []

            async def send_text(self, data):
                self.messages.append(json.loads(data))

        packed, legacy = Socket(True), Socket(False)
        snapshot = server.engine.live_snapshot(array_geometry=True)
        snapshot["points"] = np.array([[0.12345, 0, 1]], np.float32)
        snapshot["colors"] = np.array([[1, 0.5, 0]], np.float32)
        server._ws_clients.update((packed, legacy))
        try:
            await server._broadcast(snapshot)
        finally:
            server._ws_clients.difference_update((packed, legacy))
        self.assertIn("geometry", packed.messages[0])
        self.assertIn("points", legacy.messages[0])
        for socket in (packed, legacy):
            decoded = decode_live_geometry(socket.messages[0])
            np.testing.assert_array_equal(snapshot["points"], decoded["points"])
            np.testing.assert_array_equal(snapshot["colors"], decoded["colors"])
            self.assertEqual(snapshot["session_id"], decoded["session_id"])

    async def test_feedback_serialization_leaves_http_event_loop_responsive(self):
        class Socket:
            def __init__(self):
                self.query_params = {"geometry": "xyzrgb-f32le"}

            async def send_text(self, data):
                pass

        entered, release = threading.Event(), threading.Event()
        original = server.encode_live_message

        def encode(*args, **kwargs):
            entered.set()
            release.wait(5)
            return original(*args, **kwargs)

        socket = Socket()
        server._ws_clients.add(socket)
        task = None
        try:
            with patch.object(server, "encode_live_message", side_effect=encode):
                task = asyncio.create_task(server._broadcast(server.engine.live_snapshot(array_geometry=True)))
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(entered.is_set())
                self.assertFalse(task.done())
                status = await asyncio.wait_for(self.http.get("/api/scan/status"), 1)
                self.assertEqual(200, status.status_code)
                self.assertFalse(task.done())
                release.set()
                await task
        finally:
            release.set()
            if task is not None:
                await task
            server._ws_clients.discard(socket)


if __name__ == "__main__":
    unittest.main()
