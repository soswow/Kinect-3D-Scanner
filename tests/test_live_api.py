"""Live processing must preserve the existing cancellation and command barriers."""

import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import asyncio
import threading
import unittest
from unittest.mock import patch

import httpx

from scanner_server import app as server
from scanner_server.engine import ScanEngine
from shared.protocol import pack_frame, pack_frames
from tests.test_quality import scene_frames


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
            await self.http.post("/api/scan/reset", json={"live_reconstruction": True})
        ).json()
        frames = scene_frames(2)
        payload = [
            (rgb, depth, {"frame_id": i}) for i, (rgb, depth, _) in enumerate(frames)
        ]
        payload.append(payload[0])  # duplicate must not create a recording/pose index
        result = (
            await self.http.post("/api/scan/frames", content=pack_frames(payload))
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
        self.assertLessEqual(len(snapshot["points"]), 5000)
        report = (await self.http.get("/api/scan/diagnostics")).json()
        self.assertEqual(2, len(report["poses"]))
        self.assertIn("fusion", report["stage_totals_ms"])

    async def test_build_waits_for_live_frame_instead_of_rejecting(self):
        await self.http.post("/api/scan/reset", json={"live_reconstruction": True})
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
            await self.http.post("/api/scan/reset", json={"live_reconstruction": True})
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


if __name__ == "__main__":
    unittest.main()
