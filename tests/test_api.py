"""API settings validation and serialization of builds, resets and uploads."""

import os

os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from scanner_server import app as server
from shared.protocol import pack_frame, pack_frames


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
        self.assertEqual(1, result["session_guard"])

    async def test_stale_single_and_batch_uploads_preserve_replacement_scan(self):
        from scanner_server.engine import ScanEngine
        from shared.settings import ScanSettings
        server.engine = ScanEngine(device="cpu")
        self.addCleanup(server.engine.shutdown)
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.full((480, 640), 1000, np.uint16)
        frame = (rgb, depth, {"frame_id": "stale"})
        stale = server.engine.session_id
        server.engine.reset(settings=ScanSettings())
        current = server.engine.session_id
        for endpoint, body in (("frame", pack_frame(*frame)), ("frames", pack_frames([frame, frame]))):
            with self.subTest(endpoint=endpoint):
                response = await self.http.post("/api/scan/" + endpoint, content=body,
                                                headers={"X-Scan-Session": stale})
                self.assertEqual(409, response.status_code)
                self.assertEqual(0, server.engine.stored_count)
                self.assertEqual(current, server.engine.session_id)
        # Updated and legacy clients can still upload to the current session.
        for index, headers in enumerate(({"X-Scan-Session": current}, {})):
            response = await self.http.post("/api/scan/frame", headers=headers,
                content=pack_frame(rgb, depth, {"frame_id": str(index)}))
            self.assertTrue(response.json()["success"])
        response = await self.http.post("/api/scan/frames", headers={"X-Scan-Session": current},
            content=pack_frames([(rgb, depth, {"frame_id": "batch"})]))
        self.assertEqual(1, response.json()["batch_size"])

    async def test_session_guard_is_checked_after_waiting_for_engine_lock(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.zeros((480, 640), np.uint16)
        for endpoint, body in (("frame", pack_frame(rgb, depth)), ("frames", pack_frames([(rgb, depth)]))):
            fake = SimpleNamespace(session_id="old", store_frame=Mock())
            server.engine = fake
            await server._build_lock.acquire()
            upload = asyncio.create_task(self.http.post("/api/scan/" + endpoint,
                content=body, headers={"X-Scan-Session": "old"}))
            try:
                await asyncio.sleep(0.05)
                self.assertFalse(upload.done())
                fake.session_id = "replacement"
            finally:
                server._build_lock.release()
            self.assertEqual(409, (await upload).status_code)
            fake.store_frame.assert_not_called()

    async def test_reset_reports_busy_until_replacement_is_ready(self):
        entered, release = threading.Event(), threading.Event()
        original_reset = server.engine.reset

        def reset(**kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test release missing")
            original_reset(**kwargs)

        old_session = server.engine.session_id
        with patch.object(server.engine, "reset", side_effect=reset):
            task = asyncio.create_task(self.http.post("/api/scan/reset", json={}))
            try:
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(entered.is_set())
                status = (await self.http.get("/api/scan/status")).json()
                self.assertEqual("reset", status["operation"])
                self.assertEqual(old_session, status["session_id"])
            finally:
                release.set()
            self.assertTrue((await task).json()["success"])
        status = (await self.http.get("/api/scan/status")).json()
        self.assertIsNone(status["operation"])
        self.assertNotEqual(old_session, status["session_id"])

    async def test_queued_exclusive_operation_remains_visible_after_first_finishes(self):
        first_ready, finish_first = asyncio.Event(), asyncio.Event()
        second_ready, finish_second = asyncio.Event(), asyncio.Event()

        async def operation(kind, ready, finish):
            async with server._exclusive_operation(kind):
                ready.set()
                await finish.wait()

        first = asyncio.create_task(operation("save", first_ready, finish_first))
        second = None
        try:
            await first_ready.wait()
            second = asyncio.create_task(operation("reset", second_ready, finish_second))
            await asyncio.sleep(0)
            self.assertEqual("save", (await server.scan_status())["operation"])
            finish_first.set()
            await second_ready.wait()
            self.assertEqual("reset", (await server.scan_status())["operation"])
            self.assertTrue(server._exclusive)
        finally:
            finish_first.set()
            finish_second.set()
            await asyncio.gather(first, *([second] if second else []))
        self.assertIsNone((await server.scan_status())["operation"])

    async def test_project_export_options_and_local_reuse_reach_exporter(self):
        def save(engine, path, **options):
            Path(path).write_bytes(b"project archive")

        health = (await self.http.get("/api/health")).json()["project_export"]
        self.assertEqual(["png", "jpeg"], health["rgb_formats"])
        with patch("scanner_server.session.export_session", side_effect=save) as export:
            response = await self.http.get("/api/scan/export/session", params={"rgb_format": "jpeg"})
            self.assertEqual(200, response.status_code)
            self.assertEqual({"rgb_format": "jpeg", "reuse": None}, export.call_args.kwargs)
            reuse = {"session_id": server.engine.session_id, "captures": []}
            response = await self.http.post("/api/scan/export/session", json={"rgb_format": "png", "reuse": reuse})
            self.assertEqual(200, response.status_code)
            self.assertEqual(reuse, export.call_args.kwargs["reuse"])
        for body in ([], {"unexpected": True}):
            self.assertEqual(422, (await self.http.post("/api/scan/export/session", json=body)).status_code)
        self.assertEqual(413, (await self.http.post("/api/scan/export/session", content=b" " * (2 * 1024**2 + 1))).status_code)
        self.assertEqual(422, (await self.http.get("/api/scan/export/session", params={"rgb_format": "invalid"})).status_code)
        self.assertEqual(422, (await self.http.post("/api/scan/export/session", json={"reuse": {"session_id": "stale", "captures": []}})).status_code)

    async def test_compact_review_and_stale_export_are_bound_to_final_session(self):
        from scanner_server.engine import ScanEngine
        import open3d as o3d

        candidate = ScanEngine(device="cpu")
        self.addCleanup(candidate.shutdown)
        server.engine = candidate
        candidate.store_frame(np.full((480, 640, 3), 80, np.uint8), np.full((480, 640), 1000, np.uint16), {})
        candidate.poses = [(0, np.eye(4))]
        candidate.diagnostics = [{"index": 0, "success": True}]
        candidate._processed_count = candidate.frame_count = 1
        candidate.mesh = o3d.geometry.TriangleMesh.create_box(.1, .2, .3)
        review = (await self.http.get("/api/scan/project/compact")).json()
        self.assertTrue(review["available"])
        response = await self.http.post("/api/scan/export/session", json={"compact": True, "compact_revision": review["revision"]})
        self.assertEqual(200, response.status_code)
        import io, zipfile, json
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual([0], manifest["project_compaction"]["retained_source_indices"])
        for options in ({"compact": True, "compact_revision": "stale"}, {"compact": "false"}):
            self.assertEqual(422, (await self.http.post("/api/scan/export/session", json=options)).status_code)
        self.assertIs(candidate, server.engine)
        self.assertEqual(1, candidate.stored_count)

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

    async def test_texture_export_reports_brightness_decision_in_both_formats(self):
        from tests.test_features import texture_scene
        server.engine = texture_scene()
        for fmt in ("glb", "obj.zip"):
            response = await self.http.get(
                f"/api/scan/export/{fmt}",
                params={"size": 256, "exposure_correction": True, "blend_mode": "blend"},
            )
            self.assertEqual(200, response.status_code, response.text[:100])
            self.assertIn("Unchanged:", response.headers["x-texture-brightness"])
            self.assertGreater(len(response.content), 100)

    async def test_texture_page_options_and_default_source_budget_reach_exporter(self):
        def write_asset(engine, path, fmt, **options):
            Path(path).write_bytes(b"asset")
            return {"exposure_correction": {"applied": False, "reason": "Not requested"}}

        with patch("scanner_server.texturing.export_texture", side_effect=write_asset) as export:
            result = await self.http.get("/api/scan/export/glb", params={"max_pages": 8, "texels_per_m": 384})
        self.assertEqual(200, result.status_code)
        self.assertEqual(64, export.call_args.kwargs["max_views"])
        self.assertEqual(200000, export.call_args.kwargs["max_triangles"])
        self.assertEqual(8, export.call_args.kwargs["max_pages"])
        self.assertEqual(384, export.call_args.kwargs["texels_per_m"])

    async def test_explicit_texture_triangle_budget_reaches_exporter(self):
        def write_asset(engine, path, fmt, **options):
            Path(path).write_bytes(b"asset")
            return {"exposure_correction": {"applied": False, "reason": "Not requested"}}

        for budget in (100, 50000, 200000):
            with patch("scanner_server.texturing.export_texture", side_effect=write_asset) as export:
                result = await self.http.get("/api/scan/export/glb", params={"max_triangles": budget})
            self.assertEqual(200, result.status_code)
            self.assertEqual(budget, export.call_args.kwargs["max_triangles"])

    async def test_invalid_texture_page_options_are_rejected(self):
        from tests.test_features import texture_scene
        server.engine = texture_scene()
        for options in ({"max_pages": 0}, {"max_pages": 33}, {"texels_per_m": 0}):
            result = await self.http.get("/api/scan/export/glb", params=options)
            self.assertEqual(422, result.status_code)

    async def test_invalid_project_preserves_active_scan(self):
        before = server.engine
        volume = before.vbg
        response = await self.http.post("/api/scan/project", content=b"not a ZIP")
        self.assertEqual(422, response.status_code)
        self.assertIs(before, server.engine)
        self.assertIs(volume, server.engine.vbg)
        self.assertIsNone(server._exclusive_kind)

    async def test_project_open_and_save_roundtrip(self):
        import io
        import json
        import zipfile

        from scanner_server.engine import ScanEngine
        from scanner_server.session import export_session
        from shared.settings import ScanSettings

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

    async def test_repeated_cancellation_keeps_native_work_serialized(self):
        for processing in (False, True):
            for worker_fails in (False, True):
                with self.subTest(processing=processing, worker_fails=worker_fails):
                    entered, release = threading.Event(), threading.Event()
                    events = []

                    def work():
                        events.append("native-start")
                        entered.set()
                        if not release.wait(5):
                            raise TimeoutError("test release missing")
                        events.append("native-end")
                        if worker_fails:
                            raise RuntimeError("deliberate native failure")

                    async def drain_progress():
                        pass

                    async def operation():
                        async with server._build_lock:
                            if processing:
                                native = asyncio.create_task(asyncio.to_thread(work))
                                await server._wait_for_processing(native, drain_progress)
                            else:
                                await server._engine_call(work)

                    async def queued_reset():
                        async with server._build_lock:
                            events.append("reset")

                    task = asyncio.create_task(operation())
                    reset = None
                    try:
                        for _ in range(100):
                            if entered.is_set():
                                break
                            await asyncio.sleep(0.01)
                        self.assertTrue(entered.is_set())
                        reset = asyncio.create_task(queued_reset())
                        for _ in range(3):
                            task.cancel()
                            await asyncio.sleep(0.01)
                            self.assertTrue(server._build_lock.locked())
                            self.assertFalse(task.done())
                            self.assertFalse(reset.done())
                            self.assertEqual(["native-start"], events)
                    finally:
                        release.set()
                        await asyncio.gather(task, *([reset] if reset else []), return_exceptions=True)
                    self.assertTrue(task.cancelled())
                    self.assertEqual(["native-start", "native-end", "reset"], events)
                    self.assertFalse(server._build_lock.locked())

    async def test_progress_failure_waits_for_native_completion(self):
        entered, release = threading.Event(), threading.Event()
        finished = threading.Event()

        def work():
            entered.set()
            release.wait(5)
            finished.set()

        async def broken_progress():
            raise RuntimeError("progress failed")

        async def operation():
            async with server._build_lock:
                native = asyncio.create_task(asyncio.to_thread(work))
                await server._wait_for_processing(native, broken_progress)

        task = asyncio.create_task(operation())
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)
            self.assertTrue(server._build_lock.locked())
            self.assertFalse(task.done())
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertTrue(server._build_lock.locked())
        finally:
            release.set()
        with self.assertRaisesRegex(RuntimeError, "progress failed"):
            await task
        self.assertTrue(finished.is_set())


if __name__ == "__main__":
    unittest.main()
