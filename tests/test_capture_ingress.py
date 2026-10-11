"""Bound upload allocation before decoding and while native work is draining."""

import asyncio
import os
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("KINECT_DEVICE", "cpu")
os.environ.setdefault("KINECT_BLOCK_COUNT", "100")

import numpy as np
from fastapi import HTTPException

from scanner_server import app as server
from scanner_server.engine import ScanEngine
from shared.protocol import MAX_BATCH_BYTES, MAX_FRAME_BYTES, pack_frame, pack_frames
from shared.settings import ScanSettings


class StreamRequest:
    def __init__(self, chunks, *, content_length=None, gate=None):
        self.chunks = chunks
        self.headers = {} if content_length is None else {"content-length": str(content_length)}
        self.consumed = 0
        self.entered = asyncio.Event()
        self.gate = gate

    async def stream(self):
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def body(self):
        raise AssertionError("Capture ingress must stream, never buffer request.body()")


class CaptureIngressTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous = server.engine
        with patch.object(ScanEngine, "BLOCK_COUNT", 100):
            server.engine = ScanEngine(device="cpu")
        server.engine.BLOCK_COUNT = 100
        server.engine.reset(settings=ScanSettings())
        server._capture_ingress_lock = asyncio.Lock()
        server._build_lock = asyncio.Lock()
        self.rgb = np.zeros((480, 640, 3), np.uint8)
        self.depth = np.full((480, 640), 1000, np.uint16)
        self.frame = pack_frame(self.rgb, self.depth, {"frame_id": "single"})
        self.batch = pack_frames([(self.rgb, self.depth, {"frame_id": "batch"})])

    async def asyncTearDown(self):
        server.engine.shutdown()
        server.engine = self.previous

    async def wait_until(self, predicate):
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(.01)
        self.fail("Timed out waiting for isolated ingress event")

    async def test_known_oversize_does_not_consume_or_decode(self):
        for handler, constant in ((server.scan_frame, "MAX_FRAME_BYTES"),
                                  (server.scan_frames_batch, "MAX_BATCH_BYTES")):
            request = StreamRequest([b"unused"], content_length=17)
            with patch.object(server, constant, 16), patch.object(server, "unpack_frame_with_metadata") as decode:
                with self.assertRaises(HTTPException) as error:
                    await handler(request)
                self.assertEqual(413, error.exception.status_code)
                self.assertEqual(0, request.consumed)
                decode.assert_not_called()
        self.assertEqual(0, server.engine.stored_count)

    async def test_unknown_or_underdeclared_chunked_oversize_stops_input(self):
        for length in (None, 1):
            for handler, constant in ((server.scan_frame, "MAX_FRAME_BYTES"),
                                      (server.scan_frames_batch, "MAX_BATCH_BYTES")):
                request = StreamRequest([b"a" * 8, b"b" * 9, b"never read"], content_length=length)
                with patch.object(server, constant, 16):
                    with self.assertRaises(HTTPException) as error:
                        await handler(request)
                self.assertEqual(413, error.exception.status_code)
                self.assertEqual(2, request.consumed)
        self.assertEqual(0, server.engine.stored_count)

    async def test_exact_limit_valid_packets_and_legacy_hundred_frame_bound(self):
        self.assertEqual(6_000_000, MAX_FRAME_BYTES)
        self.assertEqual(600_000_408, MAX_BATCH_BYTES)
        for handler, constant, body in ((server.scan_frame, "MAX_FRAME_BYTES", self.frame),
                                        (server.scan_frames_batch, "MAX_BATCH_BYTES", self.batch)):
            request = StreamRequest([body[:8], body[8:]], content_length=len(body))
            with patch.object(server, constant, len(body)):
                result = await handler(request)
            self.assertTrue(result["success"])
        self.assertEqual(2, server.engine.stored_count)
        # Preserve the original encoder's 100-frame contract without decoding
        # hundreds of native arrays or allocating a near-limit body.
        from shared import protocol
        with patch.object(protocol, "pack_frame", return_value=b"12345678"):
            self.assertEqual(1208, len(protocol.pack_frames([(None, None)] * 100)))
            with self.assertRaises(ValueError):
                protocol.pack_frames([(None, None)] * 101)

    async def test_empty_and_malformed_packet_leave_entire_scan_untouched(self):
        state = server.engine.__dict__.copy()
        for handler in (server.scan_frame, server.scan_frames_batch):
            for body in (b"", b"invalid"):
                result = await handler(StreamRequest([body]))
                self.assertFalse(result["success"])
                self.assertFalse(server._capture_ingress_lock.locked())
        for name, value in state.items():
            self.assertIs(value, server.engine.__dict__[name], name)

    async def test_single_and_batch_share_admission_while_finish_holds_engine(self):
        await server._build_lock.acquire()
        first, second = StreamRequest([self.frame]), StreamRequest([self.batch])
        one = asyncio.create_task(server.scan_frame(first))
        two = asyncio.create_task(server.scan_frames_batch(second))
        try:
            await first.entered.wait()
            await asyncio.sleep(.03)
            self.assertFalse(second.entered.is_set())
            self.assertFalse(one.done())
            self.assertFalse(two.done())
            status = await server.scan_status()
            self.assertEqual(0, status["stored_count"])
        finally:
            server._build_lock.release()
        results = await asyncio.gather(one, two)
        self.assertTrue(all(result["success"] for result in results))
        self.assertEqual(2, server.engine.stored_count)

    async def test_slow_body_does_not_block_status_or_reset(self):
        gate = asyncio.Event()
        request = StreamRequest([self.frame], gate=gate)
        upload = asyncio.create_task(server.scan_frame(request))
        try:
            await request.entered.wait()
            self.assertFalse(server._build_lock.locked())
            old = (await server.scan_status())["session_id"]
            class ResetRequest:
                async def body(self):
                    return b'{"live_reconstruction":false}'
            result = await server.scan_reset(ResetRequest())
            self.assertTrue(result["success"])
            self.assertNotEqual(old, result["session_id"])
            self.assertFalse(upload.done())
        finally:
            upload.cancel()
            await asyncio.gather(upload, return_exceptions=True)
        self.assertFalse(server._capture_ingress_lock.locked())

    async def test_repeated_decode_cancellation_retains_admission_until_thread_finishes(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def decode(body):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Missing test native release")
            finished.set()
            return self.rgb, self.depth, {"frame_id": "cancelled"}
        first, second = StreamRequest([self.frame]), StreamRequest([self.batch])
        with patch.object(server, "unpack_frame_with_metadata", side_effect=decode):
            one = asyncio.create_task(server.scan_frame(first))
            two = None
            try:
                await self.wait_until(entered.is_set)
                two = asyncio.create_task(server.scan_frames_batch(second))
                for _ in range(3):
                    one.cancel()
                    await asyncio.sleep(.01)
                    self.assertTrue(server._capture_ingress_lock.locked())
                    self.assertFalse(second.entered.is_set())
                    self.assertFalse(finished.is_set())
            finally:
                release.set()
                results = await asyncio.gather(one, *([two] if two else []), return_exceptions=True)
        self.assertIsInstance(results[0], asyncio.CancelledError)
        self.assertTrue(finished.is_set())
        self.assertTrue(results[1]["success"])
        self.assertEqual(1, server.engine.stored_count)
        self.assertFalse(server._capture_ingress_lock.locked())

    async def test_repeated_storage_cancellation_retains_both_locks_for_single_and_batch(self):
        for handler, body in ((server.scan_frame, self.frame), (server.scan_frames_batch, self.batch)):
            with self.subTest(handler=handler.__name__):
                server.engine.reset(settings=ScanSettings())
                entered, release = threading.Event(), threading.Event()
                original = server.engine.store_frame
                def store(*args, **kwargs):
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("Missing test native release")
                    return original(*args, **kwargs)
                first, second = StreamRequest([body]), StreamRequest([b"invalid"])
                with patch.object(server.engine, "store_frame", side_effect=store):
                    one = asyncio.create_task(handler(first))
                    two = None
                    try:
                        await self.wait_until(entered.is_set)
                        two = asyncio.create_task(server.scan_frame(second))
                        for _ in range(3):
                            one.cancel()
                            await asyncio.sleep(.01)
                            self.assertTrue(server._build_lock.locked())
                            self.assertTrue(server._capture_ingress_lock.locked())
                            self.assertFalse(second.entered.is_set())
                    finally:
                        release.set()
                        results = await asyncio.gather(one, *([two] if two else []), return_exceptions=True)
                self.assertIsInstance(results[0], asyncio.CancelledError)
                self.assertEqual(1, server.engine.stored_count)
                self.assertFalse(results[1]["success"])
                self.assertFalse(server._capture_ingress_lock.locked())
                self.assertFalse(server._build_lock.locked())


if __name__ == "__main__":
    unittest.main()
