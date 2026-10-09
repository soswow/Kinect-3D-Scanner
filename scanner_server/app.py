"""FastAPI application wrapping ScanEngine for remote 3D scanning."""

import asyncio
import logging
import os
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from shared.live import encode_live_message
from shared.protocol import unpack_frame_with_metadata, unpack_frames
from shared.sensor_calibration import load_calibration
from shared.settings import ScanSettings

from .engine import ScanEngine
from .cuda_fusion import FusionUpdateError
from .cuda_input import CudaInputError

logger = logging.getLogger("scanner_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app):
    global _shutting_down
    _shutting_down = False
    try:
        yield
    finally:
        _shutting_down = True
        if _live_task is not None:
            _live_task.cancel()
            try:
                await _live_task
            except asyncio.CancelledError:
                pass
        if _feedback_task is not None:
            _feedback_task.cancel()
            try:
                await _feedback_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Kinect 3D Scanner Server", lifespan=lifespan)
engine = ScanEngine()
engine.reset(settings=ScanSettings(sensor_calibration=load_calibration()))
_build_lock = asyncio.Lock()
_broadcast_lock = asyncio.Lock()
_live_task = None
_feedback_task = None
_pending_live = None
_latest_live = None
_shutting_down = False
_exclusive = False
_exclusive_kind = None


@asynccontextmanager
async def _exclusive_operation(kind):
    global _exclusive, _exclusive_kind
    _exclusive = True
    _exclusive_kind = kind
    try:
        async with _build_lock:
            # Finish earlier live updates before manual progress/final messages.
            await _flush_live_feedback()
            yield
    finally:
        _exclusive = False
        _exclusive_kind = None


def _ensure_live_worker():
    global _live_task
    if (
        not _shutting_down
        and engine.settings.live_reconstruction
        and not getattr(engine, "fusion_failure", None)
        and not getattr(engine, "input_failure", None)
        and not getattr(engine, "_project_processing_paused", False)
        and engine.unprocessed_count
        and (_live_task is None or _live_task.done())
    ):
        _live_task = asyncio.create_task(_live_worker())


async def _live_worker():
    global _live_task, _latest_live
    last_sent = 0
    failed = False
    try:
        while engine.settings.live_reconstruction and engine.unprocessed_count:
            async with _build_lock:
                if (
                    not engine.settings.live_reconstruction
                    or not engine.unprocessed_count
                    or getattr(engine, "_project_processing_paused", False)
                ):
                    break
                await _engine_call(engine.process_frames, max_frames=1)
                snapshot = None
                now = time.monotonic()
                if now - last_sent >= 0.5 or not engine.unprocessed_count:
                    snapshot = await _engine_call(engine.live_snapshot, array_geometry=True)
                    _latest_live = snapshot
                    last_sent = now
            if snapshot is not None:
                _queue_live_feedback(snapshot)
            await asyncio.sleep(0)  # give queued mutations the next lock turn
    except asyncio.CancelledError:
        failed = True
        raise
    except (FusionUpdateError, CudaInputError) as exc:
        failed = True
        logger.error("Live CUDA reconstruction stopped: %s", exc)
        await _flush_live_feedback()
        async with _build_lock:
            snapshot = await _engine_call(engine.live_snapshot, array_geometry=True)
            _latest_live = snapshot
        if snapshot.get("volume_requires_reset") or snapshot.get("input_requires_reset"):
            await _broadcast(snapshot)
            await _broadcast({"type": "error", "session_id": snapshot["session_id"],
                              "message": str(exc),
                              "volume_requires_reset": snapshot["volume_requires_reset"],
                              "input_requires_reset": snapshot.get("input_requires_reset", False)})
    except Exception:
        failed = True
        logger.exception("Live processing stopped; manual build can retry")
        await _flush_live_feedback()
        await _broadcast(
            {"type": "error", "message": "Live processing stopped; use Build to retry"}
        )
    finally:
        _live_task = None
        if not failed:
            _ensure_live_worker()


# Connected WebSocket clients for progress broadcasting
_ws_clients: set[WebSocket] = set()


def _queue_live_feedback(snapshot):
    """Retain one pending update while the previous update is being sent."""
    global _pending_live, _feedback_task
    if _shutting_down or not _ws_clients:
        return
    _pending_live = snapshot
    if _feedback_task is None or _feedback_task.done():
        _feedback_task = asyncio.create_task(_live_feedback_worker())


async def _live_feedback_worker():
    global _pending_live, _feedback_task
    try:
        while _pending_live is not None:
            snapshot, _pending_live = _pending_live, None
            await _broadcast(snapshot)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Live feedback delivery stopped")
    finally:
        _feedback_task = None
        _pending_live = None


async def _flush_live_feedback():
    if _feedback_task is not None:
        await asyncio.shield(_feedback_task)


async def _broadcast(msg: dict):
    """Send a JSON message to all connected WebSocket clients."""
    # Serialize sends across live feedback and manual build/preview progress.
    async with _broadcast_lock:
        await _broadcast_unlocked(msg)


async def _broadcast_unlocked(msg):
    if msg.get("type") == "live" and msg.get("session_id") != engine.session_id:
        return
    clients = tuple(_ws_clients)
    if not clients:
        return
    # Build each negotiated representation once, away from the HTTP event loop.
    modes = {ws: _packed_feedback(ws) for ws in clients}
    data = {}
    for mode in set(modes.values()):
        data[mode] = await asyncio.to_thread(encode_live_message, msg, packed_geometry=mode)
    # Reset may have completed while the immutable snapshot was being encoded.
    if msg.get("type") == "live" and msg.get("session_id") != engine.session_id:
        return

    async def send(ws):
        try:
            await asyncio.wait_for(ws.send_text(data[modes[ws]]), timeout=1.0)
        except Exception:  # noqa: BLE001 — a failed subscriber must not interrupt scan progress.
            _ws_clients.discard(ws)

    await asyncio.gather(*(send(ws) for ws in clients))


def _packed_feedback(websocket):
    return getattr(websocket, "query_params", {}).get("geometry") == "xyzrgb-f32le"


async def _engine_call(function, *args, **kwargs):
    # Cancellation cannot stop a native worker thread. Keep its lock held until
    # it finishes, so a cancelled HTTP request cannot race a subsequent reset.
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def _wait_for_processing(task, drain_progress):
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.05)
            await drain_progress()
        await drain_progress()
    finally:
        if not task.done():
            await asyncio.shield(task)


# ── Health / Status ────────────────────────────────────────────────────


@app.exception_handler(FusionUpdateError)
async def fusion_update_error(request: Request, exc: FusionUpdateError):
    return JSONResponse(status_code=409, content={
        "success": False, "message": str(exc), "detail": str(exc),
        "volume_requires_reset": engine.fusion_failure is not None,
        "fusion_failure": engine.fusion_failure,
    })


@app.exception_handler(CudaInputError)
async def cuda_input_error(request: Request, exc: CudaInputError):
    return JSONResponse(status_code=409, content={
        "success": False, "message": str(exc), "detail": str(exc),
        "volume_requires_reset": engine.fusion_failure is not None,
        "input_requires_reset": engine.input_failure is not None,
        "input_failure": engine.input_failure,
    })


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "backend": engine.backend,
        "session_id": engine.session_id,
        "stored_count": engine.stored_count,
        "frame_count": engine.frame_count,
        "has_mesh": engine.mesh is not None,
        "volume_requires_reset": engine.fusion_failure is not None,
        "input_requires_reset": engine.input_failure is not None,
        "operation": _exclusive_kind if _exclusive else None,
    }


@app.get("/api/scan/status")
async def scan_status():
    return {
        "stored_count": engine.stored_count,
        "frame_count": engine.frame_count,
        "unprocessed_count": engine.unprocessed_count,
        "backend": engine.backend,
        "session_id": engine.session_id,
        "settings": engine.settings.to_dict(),
        "skipped_count": sum(not r["success"] for r in engine.diagnostics),
        "has_mesh": engine.mesh is not None,
        "tracking_state": "error" if engine.fusion_failure or engine.input_failure else "recovering" if engine._tracking_lost_frames else "tracking" if engine.poses else "waiting",
        "fusion_paused": bool(engine._tracking_lost_frames) or engine.fusion_failure is not None or engine.input_failure is not None,
        "volume_requires_reset": engine.fusion_failure is not None,
        "fusion_failure": engine.fusion_failure,
        "input_requires_reset": engine.input_failure is not None,
        "input_failure": engine.input_failure,
        "operation": _exclusive_kind if _exclusive else None,
    }


@app.get("/api/scan/diagnostics")
async def scan_diagnostics():
    async with _build_lock:
        return await _engine_call(engine.reconstruction_report)


# ── Scan Control ───────────────────────────────────────────────────────


@app.post("/api/scan/reset")
async def scan_reset(request: Request):
    global _latest_live, _pending_live
    body = await request.body()
    try:
        import json

        value = json.loads(body) if body else {}
        # Live Kinect sessions use the measured default. Public dataset replay
        # supplies its own metric camera explicitly.
        if "camera" not in value and "sensor_calibration" not in value:
            value["sensor_calibration"] = load_calibration().to_dict()
        settings = ScanSettings.from_dict(value)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    async with _build_lock:
        await _engine_call(engine.reset, settings=settings)
        _latest_live = None
        _pending_live = None
    logger.info("Scan reset")
    return {
        "success": True,
        "message": "Scan reset",
        "session_id": getattr(engine, "session_id", None),
        "settings": engine.settings.to_dict(),
    }


@app.post("/api/scan/frame")
async def scan_frame(request: Request):
    """Accept a compressed frame (binary body from pack_frame)."""
    body = await request.body()
    if not body:
        return {"success": False, "message": "Empty body"}
    try:
        rgb, depth, metadata = await asyncio.to_thread(unpack_frame_with_metadata, body)
    except Exception as e:  # noqa: BLE001 — report malformed protocol input at the HTTP boundary.
        return {"success": False, "message": f"Unpack error: {e}"}

    try:
        async with _build_lock:
            result = await _engine_call(engine.store_frame, rgb, depth, metadata)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if result.get("success"):
        _ensure_live_worker()
    return result


@app.post("/api/scan/frames")
async def scan_frames_batch(request: Request):
    """Accept a batch of compressed frames (binary body from pack_frames)."""
    body = await request.body()
    if not body:
        return {"success": False, "message": "Empty body"}

    def _unpack_and_store(raw_data: bytes):
        frames = unpack_frames(raw_data, with_metadata=True)
        results = []
        for rgb, depth, metadata in frames:
            try:
                result = engine.store_frame(rgb, depth, metadata)
            except (ValueError, TypeError) as exc:
                result = {
                    "success": False,
                    "stored_count": engine.stored_count,
                    "message": f"Invalid frame: {exc}",
                }
            results.append(result)
        count = sum(bool(r["success"]) for r in results)
        return count, engine.stored_count, len(frames), results

    try:
        async with _build_lock:
            count, total, submitted, results = await _engine_call(
                _unpack_and_store, body
            )
    except Exception as e:  # noqa: BLE001 — return a batch failure without losing the scan session.
        return {"success": False, "message": f"Batch error: {e}"}

    if count:
        _ensure_live_worker()
    return {
        "success": count > 0,
        "results": results,
        "session_id": engine.session_id,
        "stored_count": total,
        "batch_size": count,
        "rejected_count": submitted - count,
        "message": f"{count} frames stored, {submitted - count} rejected (total: {total})",
    }


@app.post("/api/scan/build")
async def scan_build(request: Request):
    """Process all frames and build the final mesh."""
    global _latest_live
    if _exclusive:
        return {"success": False, "message": "Build already in progress"}

    async with _exclusive_operation("build"):
        body = await request.body()
        if body:
            import json
            try:
                overrides = json.loads(body)
                allowed = {"final_voxel_m", "final_block_count", "final_weight", "min_component_triangles"}
                if not isinstance(overrides, dict) or overrides.keys() - allowed:
                    raise ValueError("Only final surface settings can change during a build")
                settings = ScanSettings.from_dict({**engine.settings.to_dict(), **overrides})
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc
            if settings != engine.settings:
                engine.settings = settings
                engine._final_vbg = None
        loop = asyncio.get_event_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def progress_cb(current, total, result):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {
                    "type": "progress",
                    "current": current,
                    "total": total,
                    "message": result.get("message", ""),
                    "result": result,
                },
            )

        async def drain_progress():
            while True:
                try:
                    msg = progress_queue.get_nowait()
                    await _broadcast(msg)
                except asyncio.QueueEmpty:
                    break

        build_task = asyncio.create_task(
            asyncio.to_thread(engine.build_mesh, progress_cb=progress_cb)
        )

        await _wait_for_processing(build_task, drain_progress)

        success, proc_result = build_task.result()
        _latest_live = await _engine_call(engine.live_snapshot, array_geometry=True)
        await _broadcast(_latest_live)

        if success:
            nv = len(engine.mesh.vertices) if engine.mesh else 0
            nt = len(engine.mesh.triangles) if engine.mesh else 0
            detail = (
                f"Mesh ready: {nv:,} vertices, {nt:,} triangles "
                f"({proc_result['frame_count']} frames integrated, "
                f"{proc_result['skipped_count']} skipped)"
            )
            refinement = proc_result.get("refinement", {})
            recovery = proc_result.get("fragment_reconnection", {})
            if engine.settings.reconnect_fragments:
                detail += "; " + recovery.get("reason", "Fragment reconnection finished")
                remaining = len(recovery.get("unconnected_fragments", []))
                if remaining:
                    detail += f"; {remaining} unconnected fragments retained in session"
                if recovery.get("budget_limited"):
                    detail += "; reconnection search budget reached"
            if engine.settings.refine_poses:
                detail += "; " + refinement.get("reason", "Refinement finished")
            if engine.settings.bundle_adjustment:
                detail += "; " + proc_result.get("bundle_adjustment", {}).get("reason", "Joint RGB-D refinement finished")
            if engine.settings.final_voxel_m is not None:
                final = proc_result["final_reconstruction"]
                detail += (
                    f"; final {final['voxel_m'] * 1000:g} mm"
                )
            await _broadcast({"type": "done", "success": True, "detail": detail})
        else:
            detail = proc_result.get(
                "message", "Mesh build failed. Try capturing more frames."
            )
            await _broadcast({"type": "done", "success": False, "detail": detail})

        return {"success": success, "detail": detail, **proc_result}


@app.post("/api/scan/preview")
async def scan_preview():
    """Process frames and return the preview mesh as PLY bytes."""
    if _exclusive:
        return Response(status_code=409, content="Build in progress")

    async with _exclusive_operation("preview"):
        loop = asyncio.get_event_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def progress_cb(current, total, result):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {
                    "type": "progress",
                    "current": current,
                    "total": total,
                    "message": result.get("message", ""),
                    "result": result,
                },
            )

        async def drain_progress():
            while True:
                try:
                    msg = progress_queue.get_nowait()
                    await _broadcast(msg)
                except asyncio.QueueEmpty:
                    break

        preview_task = asyncio.create_task(
            asyncio.to_thread(engine.extract_preview, progress_cb=progress_cb)
        )

        await _wait_for_processing(preview_task, drain_progress)

        mesh, pcd, _proc_result = preview_task.result()

        if mesh is None and pcd is None:
            await _broadcast(
                {
                    "type": "done",
                    "success": False,
                    "detail": "Preview extraction failed",
                }
            )
            return {"success": False, "message": "No geometry to preview"}

        # Write to temp PLY and stream back
        import open3d as o3d

        fd, tmp_path = tempfile.mkstemp(suffix=".ply")
        os.close(fd)

        try:
            if mesh is not None and len(mesh.vertices) > 0:
                await _engine_call(o3d.io.write_triangle_mesh, tmp_path, mesh)
            elif pcd is not None and len(pcd.points) > 0:
                await _engine_call(o3d.io.write_point_cloud, tmp_path, pcd)

            data = await asyncio.to_thread(Path(tmp_path).read_bytes)
        finally:
            os.unlink(tmp_path)

        await _broadcast(
            {
                "type": "done",
                "success": True,
                "detail": f"Preview ready ({len(data)} bytes)",
            }
        )
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=preview.ply"},
        )


# ── Export endpoints ───────────────────────────────────────────────────


@app.get("/api/scan/export/ply")
async def export_ply():
    """Download the mesh as PLY."""
    if engine.mesh is None and engine.point_cloud is None:
        return {"success": False, "message": "No mesh available. Build first."}

    fd, tmp_path = tempfile.mkstemp(suffix=".ply")
    os.close(fd)

    try:
        async with _build_lock:
            success = await _engine_call(engine.export_ply, tmp_path)
        if not success:
            return {"success": False, "message": "Export failed"}
        data = await asyncio.to_thread(Path(tmp_path).read_bytes)
    finally:
        os.unlink(tmp_path)

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=scan.ply"},
    )


@app.get("/api/scan/export/obj")
async def export_obj():
    """Download the mesh as OBJ."""
    if engine.mesh is None:
        return {"success": False, "message": "No mesh available. Build first."}

    fd, tmp_path = tempfile.mkstemp(suffix=".obj")
    os.close(fd)

    try:
        async with _build_lock:
            success = await _engine_call(engine.export_obj, tmp_path)
        if not success:
            return {"success": False, "message": "Export failed"}
        data = await asyncio.to_thread(Path(tmp_path).read_bytes)
    finally:
        os.unlink(tmp_path)

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=scan.obj"},
    )


@app.get("/api/scan/export/session")
async def export_recording():
    from .session import export_session

    fd, path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        async with _exclusive_operation("save"):
            await _engine_call(export_session, engine, path)
        response = FileResponse(
            path,
            media_type="application/octet-stream",
            filename="scan-session.zip",
            background=BackgroundTask(os.unlink, path),
        )
        response.chunk_size = 1024 * 1024
        return response
    except BaseException:
        os.unlink(path)
        raise


@app.post("/api/scan/project")
async def open_project(request: Request):
    """Stage and fully validate a project before replacing the active scan."""
    global engine, _latest_live, _pending_live
    from .session import load_session

    fd, path = tempfile.mkstemp(suffix=".zip")
    candidate = None
    committed = False
    try:
        size = 0
        with os.fdopen(fd, "wb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 8 * 1024**3:
                    raise HTTPException(413, "Project upload exceeds 8 GiB")
                await asyncio.to_thread(output.write, chunk)
        async with _exclusive_operation("open"):
            def stage():
                nonlocal candidate
                candidate = ScanEngine(device="cuda" if str(engine.device).startswith("CUDA") else "cpu",
                                       tracking=engine.backend["tracking"])
                load_session(candidate, path)

            try:
                await _engine_call(stage)
            except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
                raise HTTPException(422, f"Cannot open project: {exc}") from exc
            previous = engine
            candidate._project_archive_path = path
            candidate._project_processing_paused = True
            engine = candidate
            committed = True
            _latest_live = None
            _pending_live = None
            previous.shutdown()
        # Leave pending observations paused until the user previews, builds or resumes.
        return {"success": True, **await scan_status()}
    finally:
        if not committed:
            os.unlink(path)
            if candidate is not None:
                candidate.shutdown()


@app.get("/api/scan/export/{fmt}")
async def export_textured(
    fmt: str,
    size: int = 1024,
    max_triangles: int = 50000,
    max_views: int = 24,
    use_images: bool = True,
    exposure_correction: bool = False,
    blend_mode: str = "blend",
):
    """Portable UV texture export; options explicitly bound work and output size."""
    if fmt not in ("glb", "obj.zip"):
        raise HTTPException(404, "Unknown export format")
    from .texturing import export_texture

    fd, path = tempfile.mkstemp(suffix="." + fmt)
    os.close(fd)
    try:
        async with _build_lock:
            try:
                await _engine_call(
                    export_texture,
                    engine,
                    path,
                    fmt,
                    size=size,
                    max_triangles=max_triangles,
                    max_views=max_views,
                    use_images=use_images,
                    exposure_correction=exposure_correction,
                    blend_mode=blend_mode,
                )
            except ValueError as exc:
                logger.warning("Texture export %s rejected: %s", fmt, exc)
                raise HTTPException(422, str(exc)) from exc
        data = await asyncio.to_thread(Path(path).read_bytes)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=scan.{fmt}"},
        )
    finally:
        os.unlink(path)


# ── WebSocket for progress ─────────────────────────────────────────────


@app.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        async with _broadcast_lock:
            snapshot = _latest_live
            if snapshot is not None and snapshot["session_id"] == engine.session_id:
                data = await asyncio.to_thread(
                    encode_live_message, snapshot, packed_geometry=_packed_feedback(websocket)
                )
                if snapshot["session_id"] == engine.session_id:
                    await asyncio.wait_for(websocket.send_text(data), timeout=1.0)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))
