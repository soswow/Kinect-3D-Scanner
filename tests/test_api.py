"""API settings validation and serialization of builds, resets and uploads."""

import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import asyncio
import threading
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from scanner_server import app as server
from shared.protocol import pack_frame


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original = server.engine
        server._build_lock = asyncio.Lock()
        server._broadcast_lock = asyncio.Lock()
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://scanner"
        )

    async def asyncTearDown(self):
        await self.http.aclose()
        server.engine = self.original

    async def test_health_advertises_bounded_normal_motion_and_feature_transport(self):
        from shared.protocol import MAX_METADATA_BYTES
        result = (await self.http.get("/api/health")).json()["capture_protocol"]
        for capability in ("motion_history", "motion_journal", "visual_tracks", "feature_history"):
            self.assertEqual(1, result[capability])
        self.assertEqual(MAX_METADATA_BYTES, result["metadata_max_bytes"])

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

    async def test_malformed_motion_journal_preserves_active_scan(self):
        before = server.engine
        journal = before.motion_journal
        for payload in ([], None, {"session_id": "other-scan"}):
            with self.subTest(payload=payload):
                response = await self.http.post("/api/scan/motion", json=payload)
                self.assertEqual(422, response.status_code)
                self.assertIs(before, server.engine)
                self.assertIs(journal, before.motion_journal)

    async def test_motion_journal_is_received_by_the_active_session(self):
        from scanner_server.engine import ScanEngine
        from tests.test_motion_journal import journal
        source = ScanEngine(device="cpu")
        self.addCleanup(source.shutdown)
        server.engine = source
        payload = journal()
        response = await self.http.post("/api/scan/motion", json={**payload, "session_id": source.session_id})
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(40, response.json()["reads"])
        self.assertEqual(40, len(source.motion_journal["segments"][0]["samples"]))

    async def test_invalid_project_preserves_active_scan(self):
        before = server.engine
        volume = before.vbg
        response = await self.http.post("/api/scan/project", content=b"not a ZIP")
        self.assertEqual(422, response.status_code)
        self.assertIs(before, server.engine)
        self.assertIs(volume, server.engine.vbg)
        self.assertIsNone(server._exclusive_kind)

    async def test_project_open_and_save_roundtrip(self):
        from scanner_server.engine import ScanEngine
        from scanner_server.session import export_session
        from shared.settings import ScanSettings
        import io
        import json
        import zipfile

        source = ScanEngine(device="cpu")
        self.addCleanup(source.shutdown)
        source.reset(settings=ScanSettings())
        source.store_frame(np.full((480, 640, 3), 128, np.uint8),
                           np.full((480, 640), 1000, np.uint16), {"frame_id": "saved"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.zip"
            export_session(source, path)
            response = await self.http.post("/api/scan/project", content=path.read_bytes())
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, response.json()["stored_count"])
        self.assertEqual(1, response.json()["unprocessed_count"])
        self.assertIsNot(self.original, server.engine)
        opened = server.engine
        try:
            saved = await self.http.get("/api/scan/export/session")
            self.assertEqual(200, saved.status_code)
            with zipfile.ZipFile(io.BytesIO(saved.content)) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual("saved", manifest["frames"][0]["metadata"]["frame_id"])
            archive_path = opened._project_archive_path
            await self.http.post("/api/scan/reset", json={})
            self.assertFalse(Path(archive_path).exists())
        finally:
            opened.shutdown()

    async def test_invalid_final_build_overrides_preserve_settings(self):
        before = server.engine.settings
        for overrides in ({"near_m": 1}, {"final_block_count": -1}):
            response = await self.http.post("/api/scan/build", json=overrides)
            self.assertEqual(422, response.status_code)
            self.assertEqual(before, server.engine.settings)

    async def test_final_registration_switch_invalidates_pose_cache_without_resetting_captures(self):
        from scanner_server.engine import ScanEngine
        from shared.settings import ScanSettings
        engine = ScanEngine(device="cpu")
        self.addCleanup(engine.shutdown)
        engine.reset(settings=ScanSettings())
        engine.store_frame(np.zeros((480,640,3),np.uint8),np.full((480,640),1000,np.uint16))
        engine._reconnection_count = engine._refined_count = engine._bundle_count = 1
        session_id, volume = engine.session_id, engine.vbg
        server.engine = engine
        with patch.object(engine,"build_mesh",return_value=(False,{"message":"Deliberate failed build"})):
            response = await self.http.post("/api/scan/build",json={"offline_registration":"depth"})
        self.assertEqual(200,response.status_code)
        self.assertEqual("depth",engine.settings.offline_registration)
        self.assertIsNone(engine._reconnection_count)
        self.assertIsNone(engine._refined_count)
        self.assertIsNone(engine._bundle_count)
        self.assertEqual(1,engine.stored_count)
        self.assertEqual(session_id,engine.session_id)
        self.assertIs(volume,engine.vbg)

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
            session_id = "fake"

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

            def live_snapshot(self, **kwargs):
                return {"type": "live", "session_id": self.session_id, "frame_count": 0}

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
