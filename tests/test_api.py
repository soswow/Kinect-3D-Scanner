"""API settings validation and serialization of builds, resets and uploads."""

import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import asyncio
import threading
import unittest

import httpx
import numpy as np

from scanner_server import app as server
from shared.protocol import pack_frame


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original = server.engine
        server._build_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://scanner"
        )

    async def asyncTearDown(self):
        await self.http.aclose()
        server.engine = self.original

    async def test_empty_reset_uses_complete_measured_kinect_default(self):
        response = await self.http.post("/api/scan/reset", json={})
        self.assertEqual(200, response.status_code)
        settings = response.json()["settings"]
        self.assertEqual("rgb_high_res", settings["rgb_mode"])
        self.assertEqual("native_depth", settings["camera"]["image_space"])
        self.assertEqual(
            "A00363W00948202A", settings["sensor_calibration"]["camera_serial"]
        )
        self.assertEqual(1280, server.engine.settings.rgb_camera.width)

    async def test_invalid_settings_do_not_reset_session(self):
        before = server.engine.vbg
        response = await self.http.post(
            "/api/scan/reset", json={"near_m": 5, "far_m": 1}
        )
        self.assertEqual(422, response.status_code)
        self.assertIs(before, server.engine.vbg)
        response = await self.http.post(
            "/api/scan/reset", json={"camera": {"image_space": "ir"}}
        )
        self.assertEqual(422, response.status_code)

    async def test_status_exposes_active_operation_without_waiting_for_build_lock(self):
        async with server._exclusive_operation("build"):
            response = await self.http.get("/api/scan/status")
            self.assertEqual("build", response.json()["operation"])
        response = await self.http.get("/api/scan/status")
        self.assertIsNone(response.json()["operation"])

    async def test_build_reset_and_upload_serialize(self):
        entered = threading.Event()
        release = threading.Event()
        events = []

        class Fake:
            mesh = None

            def build_mesh(self, progress_cb=None):
                events.append("build-start")
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test release missing")
                events.append("build-end")
                return False, {"frame_count": 0, "errors": 0, "skipped_count": 0}

            def reset(self, settings=None):
                events.append("reset")

            def store_frame(self, *args):
                events.append("upload")
                return {"success": True, "stored_count": 1}

            def live_snapshot(self):
                return {"type": "live", "frame_count": 0}

        fake = Fake()
        fake.settings = self.original.settings
        server.engine = fake
        build = asyncio.create_task(self.http.post("/api/scan/build"))
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(entered.is_set())
            reset = asyncio.create_task(self.http.post("/api/scan/reset", json={}))
            rgb = np.zeros((480, 640, 3), np.uint8)
            depth = np.zeros((480, 640), np.uint16)
            upload = asyncio.create_task(
                self.http.post("/api/scan/frame", content=pack_frame(rgb, depth))
            )
            await asyncio.sleep(0.05)
            self.assertEqual(["build-start"], events)
        finally:
            release.set()
        await asyncio.gather(build, reset, upload)
        self.assertEqual(["build-start", "build-end", "reset", "upload"], events)

    async def test_cancelled_worker_retains_lock_until_completion(self):
        entered = threading.Event()
        release = threading.Event()
        events = []

        def work():
            events.append("start")
            entered.set()
            release.wait(5)
            events.append("end")

        async def call():
            async with server._build_lock:
                await server._engine_call(work)

        task = asyncio.create_task(call())
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.02)
        try:
            self.assertTrue(server._build_lock.locked())
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(["start", "end"], events)
        self.assertFalse(server._build_lock.locked())

    async def test_progress_disconnect_during_broadcast(self):
        class Socket:
            async def send_text(self, data):
                server._ws_clients.discard(self)

        server._ws_clients.update([Socket(), Socket()])
        await server._broadcast({"type": "progress"})
        self.assertFalse(server._ws_clients)


if __name__ == "__main__":
    unittest.main()
