"""Opt-in capture receipts recover uncertain ACKs without relabeling pixels."""

import asyncio
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from scanner_server.engine import ScanEngine
from shared.capture_receipts import capture_digest, receipt_for, validate_upload_receipt
from shared.protocol import pack_frame, pack_frames, unpack_frame_with_metadata, unpack_frames
from shared.settings import ScanSettings


class ReceiptHelpersTests(unittest.TestCase):
    def test_receipt_survives_all_native_packet_shapes_and_predictions(self):
        depth = np.arange(480 * 640, dtype=np.uint16).reshape(480, 640)
        for shape in ((480, 640, 3), (1024, 1280, 3)):
            rgb = np.zeros(shape, np.uint8)
            metadata = {"frame_id": "wire", "motion_history": {"complete": True},
                        "upload_receipt": {"stream_id": "wire", "sequence": 0, "expected_index": 0}}
            expected = receipt_for(rgb, depth, metadata)
            for prediction in (False, True):
                with self.subTest(shape=shape, prediction=prediction):
                    decoded = unpack_frame_with_metadata(pack_frame(rgb, depth, metadata,
                        spatial_prediction=prediction))
                    self.assertEqual(expected, receipt_for(*decoded))
                    decoded_batch = unpack_frames(pack_frames([(rgb, depth, metadata)],
                        spatial_prediction=prediction), with_metadata=True)
                    self.assertEqual(expected, receipt_for(*decoded_batch[0]))
            self.assertNotIn("rgb_shape", metadata)
            with self.assertRaisesRegex(ValueError, "rgb_shape"):
                receipt_for(rgb, depth, {**metadata, "rgb_shape": [1, 2, 3]})

    def test_digest_is_portable_and_canonical(self):
        rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        depth = np.arange(8, dtype=np.uint16).reshape(2, 4)
        metadata = {"z": [1, {"b": 2, "a": 3}], "a": "text"}
        expected = capture_digest(rgb, depth, metadata)
        self.assertEqual(expected, capture_digest(rgb, depth.astype(">u2"), {"a": "text", "z": [1, {"a": 3, "b": 2}]}))
        self.assertEqual(expected, capture_digest(rgb[:, ::-1][:, ::-1], depth, metadata))
        with self.assertRaises(ValueError):
            capture_digest(rgb, depth, {"motion": float("nan")})

    def test_receipt_validation_is_strict_and_copied(self):
        valid = {"stream_id": "stream", "sequence": 0, "expected_index": 0}
        self.assertEqual(valid, validate_upload_receipt({"upload_receipt": valid}))
        self.assertIsNot(valid, validate_upload_receipt({"upload_receipt": valid}))
        for change in ({"stream_id": ""}, {"stream_id": "x" * 129}, {"sequence": True},
                       {"sequence": -1}, {"expected_index": False}, {"expected_index": 1.0}, {"extra": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_upload_receipt({"upload_receipt": {**valid, **change}})


class CaptureReceiptTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {"KINECT_NATIVE": "off"}), patch.object(ScanEngine, "BLOCK_COUNT", 100):
            self.engine = ScanEngine(device="cpu")
        self.engine.BLOCK_COUNT = 100
        self.engine.reset(settings=ScanSettings())
        self.addCleanup(self.engine.shutdown)
        self.rgb = np.zeros((480, 640, 3), np.uint8)
        self.depth = np.full((480, 640), 1000, np.uint16)
        self.metadata = {"frame_id": "capture", "motion_history": {"visual": [{"valid": False}]},
                         "upload_receipt": {"stream_id": "stream", "sequence": 7, "expected_index": 0}}

    def store(self, metadata=None, **kwargs):
        return self.engine.store_frame(self.rgb, self.depth, metadata or self.metadata, **kwargs)

    def test_lost_ack_exact_repeat_preserves_processed_and_final_state(self):
        original = self.store(retry_safe=True)
        self.assertEqual(receipt_for(self.rgb, self.depth, self.metadata), original["capture_receipt"])
        self.engine.mesh = object()
        self.engine._final_vbg = object()
        self.engine._processed_count = 1
        self.engine.frame_metadata[0]["server_enrichment"] = True
        state = self.engine.__dict__.copy()
        with patch.object(self.engine, "MAX_FRAMES", 1):
            result = self.store(copy.deepcopy(self.metadata), retry_safe=True)
        self.assertTrue(result["success"])
        self.assertEqual(0, result["index"])
        self.assertEqual(original["capture_receipt"], result["capture_receipt"])
        for name, value in state.items():
            self.assertIs(value, self.engine.__dict__[name], name)
        result["capture_receipt"]["sha256"] = "changed response"
        self.assertEqual(original["capture_receipt"], self.store(retry_safe=True)["capture_receipt"])

    def test_conflicting_retry_pixels_metadata_or_expected_index_preserves_scan(self):
        self.store(retry_safe=True)
        for field in ("pixels", "motion", "expected_index"):
            metadata = copy.deepcopy(self.metadata)
            rgb = self.rgb.copy()
            if field == "pixels":
                rgb[0, 0, 0] = 1
            elif field == "motion":
                metadata["motion_history"]["visual"][0]["valid"] = True
            else:
                metadata["upload_receipt"]["expected_index"] = 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs"):
                self.engine.store_frame(rgb, self.depth, metadata, retry_safe=True)
            self.assertEqual(1, self.engine.stored_count)

    def test_cross_stream_order_and_frame_id_reuse_are_rejected(self):
        wrong = copy.deepcopy(self.metadata)
        wrong["upload_receipt"]["expected_index"] = 1
        with self.assertRaisesRegex(ValueError, "next stored index"):
            self.store(wrong, retry_safe=True)
        self.store(retry_safe=True)
        wrong["upload_receipt"].update(stream_id="another", sequence=0)
        self.assertFalse(self.store(wrong, retry_safe=True)["success"])
        wrong["frame_id"] = "different"
        wrong["upload_receipt"]["expected_index"] = 0
        with self.assertRaisesRegex(ValueError, "next stored index"):
            self.store(wrong, retry_safe=True)
        wrong["upload_receipt"]["expected_index"] = 1
        self.assertTrue(self.store(wrong, retry_safe=True)["success"])

    def test_bool_id_and_legacy_duplicate_behavior(self):
        with self.assertRaisesRegex(ValueError, "frame_id"):
            self.store({**self.metadata, "frame_id": True}, retry_safe=True)
        legacy = copy.deepcopy(self.metadata)
        legacy["upload_receipt"]["expected_index"] = 100
        self.assertTrue(self.store(legacy)["success"])
        self.assertFalse(self.store(legacy)["success"])
        self.assertEqual({}, self.engine._capture_receipts)

    def test_reset_and_project_load_do_not_restore_receipt_authority(self):
        from scanner_server.session import export_session, load_session
        self.store(retry_safe=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.zip"
            export_session(self.engine, path)
            old_session = self.engine.session_id
            self.engine.reset(settings=ScanSettings())
            self.assertNotEqual(old_session, self.engine.session_id)
            self.assertEqual({}, self.engine._capture_receipts)
            load_session(self.engine, path)
        self.assertEqual({}, self.engine._capture_receipts)
        with self.assertRaisesRegex(ValueError, "next stored index"):
            self.store(retry_safe=True)


class CaptureReceiptApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from scanner_server import app as server
        self.server = server
        self.original = server.engine
        with patch.object(ScanEngine, "BLOCK_COUNT", 100):
            server.engine = ScanEngine(device="cpu")
        server.engine.BLOCK_COUNT = 100
        server.engine.reset(settings=ScanSettings())
        server._build_lock = asyncio.Lock()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://isolated")
        self.rgb = np.zeros((480, 640, 3), np.uint8)
        self.depth = np.full((480, 640), 1000, np.uint16)
        self.headers = {"X-Capture-Receipt": "1", "X-Scan-Session": server.engine.session_id}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.server.engine.shutdown()
        self.server.engine = self.original

    def frame(self, index):
        return (self.rgb, self.depth, {"frame_id": f"id-{index}",
            "upload_receipt": {"stream_id": "stream", "sequence": index, "expected_index": index}})

    async def test_capability_session_headers_and_single_retry(self):
        health = (await self.client.get("/api/health")).json()
        self.assertEqual(1, health["capture_protocol"]["capture_receipts"])
        body = pack_frame(*self.frame(0))
        for headers in ({"X-Capture-Receipt": "1"}, {**self.headers, "X-Capture-Receipt": "0"},
                        {**self.headers, "X-Scan-Session": "old"}):
            self.assertIn((await self.client.post("/api/scan/frame", content=body, headers=headers)).status_code, (409, 422))
        first = (await self.client.post("/api/scan/frame", content=body, headers=self.headers)).json()
        again = (await self.client.post("/api/scan/frame", content=body, headers=self.headers)).json()
        self.assertTrue(again["success"])
        self.assertEqual(first["capture_receipt"], again["capture_receipt"])
        self.assertEqual(1, self.server.engine.stored_count)

    async def test_partial_batch_cannot_append_successor_over_a_missing_capture(self):
        invalid = self.frame(0)
        invalid[2]["upload_receipt"]["sequence"] = True
        body = pack_frames([invalid, self.frame(1)])
        result = (await self.client.post("/api/scan/frames", content=body, headers=self.headers)).json()
        self.assertEqual([False, False], [r["success"] for r in result["results"]])
        self.assertEqual(0, self.server.engine.stored_count)
        body = pack_frames([self.frame(0), self.frame(1)])
        for _ in range(2):
            result = (await self.client.post("/api/scan/frames", content=body, headers=self.headers)).json()
            self.assertEqual([0, 1], [r["index"] for r in result["results"]])
            self.assertTrue(all(r["success"] for r in result["results"]))
        self.assertEqual(2, self.server.engine.stored_count)


if __name__ == "__main__":
    unittest.main()
