"""Hard input failures stop processing without rejecting or corrupting raw views."""

import asyncio
import io
import json
import os
import unittest
import zipfile
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np

from scanner_server.cuda_fusion import FusionUpdateError
from scanner_server.cuda_input import CudaInputError, InputPreparation
from scanner_server.engine import ScanEngine
from shared.protocol import pack_frame
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings


def recording(engine):
    rgb = np.zeros((480, 640, 3), np.uint8)
    raw = np.full((480, 640), 800, np.uint16)
    for _ in range(2):
        engine.store_frame(rgb, raw)
    return rgb, raw


def environment():
    return patch.dict(os.environ, KINECT_DEVICE="cpu", KINECT_BLOCK_COUNT="32",
        KINECT_CUDA_INPUT="off", KINECT_CUDA_MATCHING="cpu", KINECT_CUDA_REGISTRATION="cpu",
        KINECT_CUDA_ODOMETRY="off", KINECT_VISUAL_FEATURES="orb")


class InputFailureTests(unittest.TestCase):
    def setUp(self):
        self.environment = environment()
        self.environment.start()
        self.addCleanup(self.environment.stop)
        with patch.object(ScanEngine, "BLOCK_COUNT", 32):
            self.engine = ScanEngine(device="cpu", tracking="legacy")
        self.rgb, self.raw = recording(self.engine)

    def test_pre_fusion_failure_preserves_failed_raw_index_and_valid_volume(self):
        engine = self.engine
        original = engine.vbg
        with patch.object(engine, "_prepare_input", side_effect=CudaInputError("Incompatible calibration")) as prepare:
            for action in (engine.process_frames, engine.process_frames, engine.build_mesh, engine.extract_preview):
                with self.assertRaisesRegex(CudaInputError, "Save Session"):
                    action()
            self.assertEqual(1, prepare.call_count)
        self.assertEqual(2, engine.unprocessed_count)
        self.assertEqual([], engine.diagnostics)
        self.assertIs(original, engine.vbg)
        self.assertIsNone(engine.fusion_failure)
        snapshot = engine.live_snapshot()
        self.assertEqual("error", snapshot["tracking_state"])
        self.assertTrue(snapshot["input_requires_reset"])
        self.assertFalse(snapshot["volume_requires_reset"])
        self.assertEqual(engine.input_failure, engine.reconstruction_report()["input_failure"])
        with patch.object(ScanEngine, "BLOCK_COUNT", 32):
            engine.reset()
        self.assertIsNone(engine.input_failure)

    def test_real_relocalization_and_tracking_catches_propagate_input_failure(self):
        engine = self.engine
        engine.frame_count = engine._processed_count = 1
        engine.poses = [(0, np.eye(4))]
        engine.diagnostics = [{"index": 0, "success": True}]
        engine._tracking_lost_frames = 2
        cloud = SimpleNamespace(points=np.zeros((200, 3)))
        with patch("scanner_server.engine.prepare_rgbd", side_effect=lambda rgb, raw, _: (rgb, raw)), \
                patch.object(engine, "_make_rgbd", return_value=object()), \
                patch.object(engine, "_make_reg_pcd", return_value=cloud), \
                patch.object(engine, "_visual_register", return_value=None), \
                patch.object(engine, "_ensure_raw_normals"), \
                patch.object(engine, "_recover_anchor", return_value=None), \
                patch.object(engine, "_relocalize", side_effect=CudaInputError("Target preparation failed")), \
                patch.object(engine, "_integrate_vbg", side_effect=AssertionError("No write is authorized")) as integrate:
            with self.assertRaisesRegex(CudaInputError, "Save Session"):
                engine.process_frames()
            integrate.assert_not_called()
        self.assertEqual(1, engine.unprocessed_count)
        self.assertEqual([True], [result["success"] for result in engine.diagnostics])
        self.assertIn("Target preparation failed", engine.input_failure["reason"])
        self.assertIsNone(engine.fusion_failure)

    def test_cleanup_sync_failure_preserves_primary_input_error_and_balances_stages(self):
        engine = self.engine
        engine.device = "CUDA:0"
        def prepare_failure(rgb, raw):
            with engine._stage("depth_filter"):
                raise CudaInputError("Sticky input fault")
        with patch.object(engine, "_process_single_frame", side_effect=prepare_failure), \
                patch("scanner_server.engine.o3c.cuda.synchronize", side_effect=[None, RuntimeError("Cleanup failed")]):
            with self.assertRaisesRegex(CudaInputError, "Sticky input fault"):
                engine.process_frames()
        self.assertEqual([], engine._stage_children)
        self.assertIn("Sticky input fault", engine.input_failure["reason"])
        self.assertIsNone(engine.fusion_failure)

    def test_input_error_after_live_write_uses_volume_quarantine(self):
        engine = self.engine
        def write_then_fail(rgb, raw):
            engine.vbg.attribute("weight").numpy().reshape(-1)[0] = 17
            engine._live_fusion_generation += 1
            raise CudaInputError("Late input preparation fault")
        with patch.object(engine, "_process_single_frame", side_effect=write_then_fail):
            with self.assertRaisesRegex(FusionUpdateError, "after fusion"):
                engine.process_frames()
        self.assertIsNone(engine.input_failure)
        self.assertTrue(engine.fusion_failure["completed_live_update"])
        self.assertEqual([], engine.diagnostics)
        self.assertEqual(2, engine.unprocessed_count)

    def test_candidate_input_failure_preserves_live_volume_and_pause_state(self):
        engine = self.engine
        original = engine.vbg
        engine.settings = replace(engine.settings, final_voxel_m=.003, final_block_count=32)
        engine.poses = [(0, np.eye(4))]
        with patch.object(engine, "_prepare_input", side_effect=CudaInputError("Candidate input failed")):
            with self.assertRaises(CudaInputError):
                engine._final_volume()
        self.assertIs(original, engine.vbg)
        self.assertIsNone(engine.fusion_failure)
        self.assertIsNone(engine.input_failure)

    def test_safe_auto_compatibility_fallback_can_still_authorize_fusion(self):
        class IncompatibleGpu:
            def __init__(self, settings):
                pass
            def prepare_host(self, rgb, raw):
                depth = raw.copy()
                depth[0, 0] += 1
                return rgb.copy(), depth
            def synchronize(self):
                pass
        engine = self.engine
        engine.settings = ScanSettings(sensor_calibration=load_calibration(), rgb_mode="rgb_low_res")
        with patch.dict(os.environ, KINECT_CUDA_INPUT="auto"):
            engine._input_preparation = InputPreparation("CUDA:0", engine.backend, factory=IncompatibleGpu)
        cloud = SimpleNamespace(points=np.zeros((200, 3)),
            segment_plane=lambda *_: ([0, 1, 0, 0], np.arange(200)))
        with patch("scanner_server.engine.prepare_rgbd", side_effect=lambda rgb, raw, _: (rgb, raw)), \
                patch.object(engine, "_make_rgbd", return_value=object()), \
                patch.object(engine, "_make_reg_pcd", return_value=cloud), \
                patch.object(engine, "_integrate_vbg") as integrate, \
                patch.object(engine, "_extract_model_pcd"):
            result = engine.process_frames(max_frames=1)
        self.assertEqual(0, result["errors"])
        integrate.assert_called_once()
        self.assertIsNone(engine.input_failure)
        self.assertIsNone(engine.fusion_failure)
        self.assertEqual(1, engine.backend["cuda_input"]["fallback_batches"])
        self.assertEqual("cpu", engine.backend["cuda_input"]["implementation"])


class InputFailureApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.environment = environment()
        self.environment.start()
        from scanner_server import app as server
        self.server = server
        self.original = server.engine
        with patch.object(ScanEngine, "BLOCK_COUNT", 32):
            server.engine = ScanEngine(device="cpu", tracking="legacy")
        server._build_lock = asyncio.Lock()
        server._broadcast_lock = asyncio.Lock()
        server._live_task = server._feedback_task = None
        server._latest_live = server._pending_live = None
        server._shutting_down = server._exclusive = False
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://scanner")
        self.rgb, self.raw = recording(server.engine)

    async def asyncTearDown(self):
        self.server._shutting_down = True
        if self.server._live_task is not None:
            await self.server._live_task
        await self.http.aclose()
        self.server.engine = self.original
        self.server._shutting_down = self.server._exclusive = False
        self.server._latest_live = self.server._pending_live = None
        self.environment.stop()

    async def test_api409_status_raw_export_and_reset_distinguish_valid_live_volume(self):
        engine = self.server.engine
        with patch.object(engine, "_prepare_input", side_effect=CudaInputError("Explicit input failure")) as prepare:
            for route in ("build", "preview", "build"):
                response = await self.http.post("/api/scan/" + route)
                self.assertEqual(409, response.status_code)
                self.assertTrue(response.json()["input_requires_reset"])
                self.assertFalse(response.json()["volume_requires_reset"])
                self.assertIn("Save Session", response.json()["message"])
            self.assertEqual(1, prepare.call_count)
        status = (await self.http.get("/api/scan/status")).json()
        self.assertEqual("error", status["tracking_state"])
        self.assertTrue(status["fusion_paused"])
        self.assertEqual(2, status["unprocessed_count"])
        self.assertEqual(0, status["skipped_count"])
        self.assertFalse(status["volume_requires_reset"])
        self.assertTrue((await self.http.get("/api/health")).json()["input_requires_reset"])
        self.assertTrue((await self.http.post("/api/scan/frame", content=pack_frame(self.rgb, self.raw))).json()["success"])
        exported = await self.http.get("/api/scan/export/session")
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            self.assertEqual(3, len(json.loads(archive.read("manifest.json"))["frames"]))
            self.assertEqual(engine.input_failure, json.loads(archive.read("reconstruction.json"))["input_failure"])
        await self.http.post("/api/scan/reset", json=ScanSettings().to_dict())
        self.assertFalse((await self.http.get("/api/scan/status")).json()["input_requires_reset"])

    async def test_live_error_broadcasts_once_and_raw_upload_does_not_restart_processing(self):
        engine = self.server.engine
        engine.settings = replace(engine.settings, live_reconstruction=True)
        with patch.object(engine, "_prepare_input", side_effect=CudaInputError("Live input failure")) as prepare, \
                patch.object(self.server, "_broadcast", new_callable=AsyncMock) as broadcast:
            self.server._ensure_live_worker()
            await asyncio.wait_for(self.server._live_task, 5)
            self.assertIsNone(self.server._live_task)
            uploaded = await self.http.post("/api/scan/frame", content=pack_frame(self.rgb, self.raw))
            self.assertTrue(uploaded.json()["success"])
            self.assertIsNone(self.server._live_task)
            self.assertEqual(1, prepare.call_count)
            errors = [call.args[0] for call in broadcast.await_args_list if call.args[0]["type"] == "error"]
            self.assertEqual(1, len(errors))
            self.assertTrue(errors[0]["input_requires_reset"])
            self.assertFalse(errors[0]["volume_requires_reset"])


if __name__ == "__main__":
    unittest.main()
