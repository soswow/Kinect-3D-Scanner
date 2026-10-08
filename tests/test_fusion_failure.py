"""A potentially partial CUDA update must never authorize later live geometry."""

import asyncio
import io
import json
import os
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np

from scanner_server.cuda_fusion import FusionUpdateError
from scanner_server.engine import ScanEngine
from scanner_server.session import export_session
from shared.protocol import pack_frame
from shared.settings import ScanSettings


def raw_frames(engine, count=2):
    rgb = np.zeros((480, 640, 3), np.uint8)
    depth = np.full((480, 640), 1000, np.uint16)
    for _ in range(count):
        engine.store_frame(rgb, depth)
    return rgb, depth


class FusionFailureTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, KINECT_CUDA_MATCHING="cpu",
            KINECT_CUDA_REGISTRATION="cpu", KINECT_CUDA_ODOMETRY="off", KINECT_VISUAL_FEATURES="orb")
        self.environment.start()
        self.addCleanup(self.environment.stop)
        with patch.object(ScanEngine, "BLOCK_COUNT", 32):
            self.engine = ScanEngine(device="cpu", tracking="legacy")
        raw_frames(self.engine)

    def test_partial_live_failure_stops_processing_and_preserves_the_recording(self):
        engine = self.engine
        original_volume = engine.vbg
        def partial_update(rgb, depth):
            original_volume.attribute("weight").numpy().reshape(-1)[0] = 17
            raise FusionUpdateError("Injected failure after first CUDA chunk")
        with patch.object(engine, "_process_single_frame", side_effect=partial_update) as process:
            for action in (engine.process_frames, engine.process_frames, engine.build_mesh, engine.extract_preview):
                with self.assertRaisesRegex(FusionUpdateError, "Save Session"):
                    action()
            self.assertEqual(1, process.call_count)
        self.assertEqual(2, engine.unprocessed_count)
        self.assertEqual(0, engine.frame_count)
        self.assertEqual([], engine.diagnostics)
        self.assertEqual(0, engine.fusion_failure["index"])
        self.assertTrue(engine.live_snapshot()["volume_requires_reset"])
        self.assertTrue(engine.live_snapshot()["fusion_paused"])
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "retained.zip"
            export_session(engine, destination)
            with zipfile.ZipFile(destination) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(2, len(manifest["frames"]))
                for frame in manifest["frames"]:
                    self.assertIn(frame["rgb"], archive.namelist())
                    self.assertIn(frame["depth"], archive.namelist())
                report = json.loads(archive.read("reconstruction.json"))
                self.assertEqual(engine.fusion_failure, report["fusion_failure"])
        with patch.object(ScanEngine, "BLOCK_COUNT", 32):
            engine.reset()
        self.assertIsNone(engine.fusion_failure)
        self.assertIsNot(original_volume, engine.vbg)

    def test_ordinary_registration_errors_remain_rejected_captures(self):
        with patch.object(self.engine, "_process_single_frame", side_effect=[
                RuntimeError("Registration rejected"), {"success": True}]) as process:
            result = self.engine.process_frames()
        self.assertEqual(2, process.call_count)
        self.assertEqual(1, result["errors"])
        self.assertEqual(0, self.engine.unprocessed_count)
        self.assertIsNone(self.engine.fusion_failure)

    def test_cleanup_sync_errors_cannot_mask_a_typed_partial_update(self):
        engine = self.engine
        engine.device = "CUDA:0"
        def partial_update(rgb, depth):
            with engine._stage("outer"):
                with engine._stage("fusion"):
                    raise FusionUpdateError("Injected partial update")
        with patch.object(engine, "_process_single_frame", side_effect=partial_update) as process, \
                patch("scanner_server.engine.o3c.cuda.synchronize", side_effect=[
                    None, None, RuntimeError("Failed fusion cleanup"), RuntimeError("Failed outer cleanup")]):
            with self.assertRaisesRegex(FusionUpdateError, "Save Session"):
                engine.process_frames()
        self.assertEqual(1, process.call_count)
        self.assertIn("Injected partial update", engine.fusion_failure["reason"])
        self.assertEqual([], engine._stage_children)
        self.assertIn("fusion", engine.stage_totals_ms)
        self.assertIn("outer", engine.stage_totals_ms)

    def test_post_fusion_sync_failure_quarantines_a_completed_update(self):
        engine = self.engine
        engine.device = "CUDA:0"
        def update(rgb, depth):
            with engine._stage("fusion"):
                engine.vbg.attribute("weight").numpy().reshape(-1)[0] = 17
            return {"success": True}
        with patch.object(engine, "_process_single_frame", side_effect=update) as process, \
                patch("scanner_server.engine.o3c.cuda.synchronize", side_effect=[
                    None, RuntimeError("Failed completion synchronization")]):
            with self.assertRaisesRegex(FusionUpdateError, "Save Session"):
                engine.process_frames()
        self.assertEqual(1, process.call_count)
        self.assertEqual(2, engine.unprocessed_count)
        self.assertIn("completion synchronization", engine.fusion_failure["reason"])
        self.assertEqual([], engine._stage_children)

    def test_failure_in_a_fresh_final_candidate_does_not_quarantine_the_live_volume(self):
        engine = self.engine
        original_volume = engine.vbg
        engine.settings = replace(engine.settings, final_voxel_m=.003, final_block_count=32)
        engine.poses = [(0, np.eye(4))]
        with patch.object(ScanEngine, "_integrate_vbg", side_effect=FusionUpdateError("Candidate update failed")), \
                patch("scanner_server.engine.prepare_rgbd", side_effect=lambda rgb, depth, _: (rgb, depth)):
            with self.assertRaises(FusionUpdateError):
                engine._final_volume()
        self.assertIsNone(engine.fusion_failure)
        self.assertIs(original_volume, engine.vbg)
        self.assertIsNone(engine._final_vbg)

    def post_fusion_preparation_failure(self, *, later=False, lazy=False):
        engine = self.engine
        cloud = SimpleNamespace(points=np.zeros((200, 3)),
            segment_plane=lambda *_: ([0, 1, 0, 0], np.arange(200)))
        if later:
            engine.frame_count = 1
            engine._processed_count = 1
            engine.poses = [(0, np.eye(4))]
            engine.diagnostics = [{"index": 0, "success": True}]
        if lazy:
            engine.backend["model_preparation"] = "lazy"
            engine._visual_evidence = {"feature_method": "orb"}
        method = "keyframe+visual" if lazy else "global"
        proposal = SimpleNamespace(transformation=np.eye(4), fitness=1, inlier_rmse=0)
        prepare = "_refresh_live_points" if lazy else "_extract_model_pcd"
        def write(rgb, depth, extrinsic):
            engine.vbg.attribute("weight").numpy().reshape(-1)[0] += 1
        with patch("scanner_server.engine.prepare_rgbd", side_effect=lambda rgb, depth, _: (rgb, depth)), \
                patch.object(engine, "_make_rgbd", return_value=object()), \
                patch.object(engine, "_make_reg_pcd", return_value=cloud), \
                patch.object(engine, "_integrate_vbg", side_effect=write) as integrate, \
                patch.object(engine, prepare, side_effect=RuntimeError("Injected post-fusion preparation failure")), \
                patch.object(engine, "_register", return_value=(proposal, method)), \
                patch.object(engine, "_verify_tracking_transition", return_value=None), \
                patch.object(engine, "MODEL_REFRESH_INTERVAL", 1):
            with self.assertRaisesRegex(FusionUpdateError, "Save Session"):
                engine.process_frames()
            with self.assertRaises(FusionUpdateError):
                engine.process_frames()
            self.assertEqual(1, integrate.call_count)
        self.assertEqual(1 if later else 2, engine.unprocessed_count)
        self.assertEqual([True] if later else [], [item["success"] for item in engine.diagnostics])
        self.assertIn("preparation failure", engine.fusion_failure["reason"])
        self.assertEqual(2, engine.stored_count)

    def test_first_reference_preparation_failure_quarantines_successfully_fused_geometry(self):
        self.post_fusion_preparation_failure()

    def test_later_eager_preparation_failure_does_not_mark_committed_pose_rejected(self):
        self.post_fusion_preparation_failure(later=True)

    def test_later_lazy_snapshot_failure_blocks_subsequent_processing(self):
        self.post_fusion_preparation_failure(later=True, lazy=True)


class FusionFailureApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from scanner_server import app as server
        self.server = server
        self.original = server.engine
        self.environment = patch.dict(os.environ, KINECT_CUDA_MATCHING="cpu",
            KINECT_CUDA_REGISTRATION="cpu", KINECT_CUDA_ODOMETRY="off", KINECT_VISUAL_FEATURES="orb")
        self.environment.start()
        with patch.object(ScanEngine, "BLOCK_COUNT", 32):
            server.engine = ScanEngine(device="cpu", tracking="legacy")
        server._build_lock = asyncio.Lock()
        server._broadcast_lock = asyncio.Lock()
        server._live_task = server._feedback_task = None
        server._latest_live = server._pending_live = None
        server._shutting_down = server._exclusive = False
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://scanner")

    async def asyncTearDown(self):
        self.server._shutting_down = True
        if self.server._live_task is not None:
            await self.server._live_task
        await self.http.aclose()
        self.server.engine = self.original
        self.server._shutting_down = self.server._exclusive = False
        self.server._latest_live = self.server._pending_live = None
        self.environment.stop()

    async def test_manual_processing_reports409_and_raw_exports_remain_available(self):
        engine = self.server.engine
        rgb, depth = raw_frames(engine)
        with patch.object(engine, "_process_single_frame", side_effect=FusionUpdateError("Injected CUDA failure")) as process:
            for route in ("build", "preview", "build"):
                response = await self.http.post("/api/scan/" + route)
                self.assertEqual(409, response.status_code)
                self.assertIn("Save Session", response.json()["message"])
            self.assertEqual(1, process.call_count)
        status = (await self.http.get("/api/scan/status")).json()
        self.assertEqual("error", status["tracking_state"])
        self.assertTrue(status["volume_requires_reset"])
        upload = await self.http.post("/api/scan/frame", content=pack_frame(rgb, depth))
        self.assertTrue(upload.json()["success"])
        exported = await self.http.get("/api/scan/export/session")
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            self.assertEqual(3, len(json.loads(archive.read("manifest.json"))["frames"]))
        reset = await self.http.post("/api/scan/reset", json=ScanSettings().to_dict())
        self.assertEqual(200, reset.status_code)
        self.assertFalse((await self.http.get("/api/scan/status")).json()["volume_requires_reset"])

    async def test_live_failure_broadcasts_once_and_new_upload_does_not_restart_it(self):
        engine = self.server.engine
        engine.settings = replace(engine.settings, live_reconstruction=True)
        rgb, depth = raw_frames(engine)
        with patch.object(engine, "_process_single_frame", side_effect=FusionUpdateError("Injected CUDA failure")) as process, \
                patch.object(self.server, "_broadcast", new_callable=AsyncMock) as broadcast:
            self.server._ensure_live_worker()
            await asyncio.wait_for(self.server._live_task, 5)
            self.assertIsNone(self.server._live_task)
            uploaded = await self.http.post("/api/scan/frame", content=pack_frame(rgb, depth))
            self.assertTrue(uploaded.json()["success"])
            self.assertIsNone(self.server._live_task)
            self.assertEqual(1, process.call_count)
            errors = [call.args[0] for call in broadcast.await_args_list if call.args[0]["type"] == "error"]
            self.assertEqual(1, len(errors))
            self.assertIn("Save Session", errors[0]["message"])


if __name__ == "__main__":
    unittest.main()
