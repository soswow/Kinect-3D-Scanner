"""Isolated libfreenect acquisition; native USB calls may never return."""

import time

import numpy as np

from shared.capture import RGB_MODE_FPS, timestamp_delta_ms, validate_rgb_exposure

from .rgb_exposure import ExposureControlUnavailable, RGBExposureControl

RGB_SHAPE = (480, 640, 3)
DEPTH_SHAPE = (480, 640)


def capture_frames(connection, stop_event, rgb_buffer, depth_buffer, high_res=True,
                   rgb_exposure_mode="auto", rgb_shutter_speed=125, rgb_gain=1):
    """Publish one shared-memory pair at a time, awaiting a copy acknowledgement.

    Only small messages cross the pipe. The child never overwrites a published
    frame until the parent has copied it. Terminating a stalled driver therefore
    cannot leave the GUI blocked on a frame payload or a shared-memory lock.
    """
    ctx = dev = None
    depth_started = video_started = False
    try:
        import freenect

        validate_rgb_exposure(rgb_exposure_mode, rgb_shutter_speed,
                              "rgb_high_res" if high_res else "rgb_low_res", rgb_gain)

        ctx = freenect.init()
        if ctx is None:
            raise RuntimeError("Cannot initialise libfreenect")
        if freenect.num_devices(ctx) == 0:
            raise RuntimeError(
                "No Kinect camera detected. Connect USB and external power."
            )
        dev = freenect.open_device(ctx, 0)
        if dev is None:
            raise RuntimeError(
                "Cannot open Kinect camera. Close other camera applications."
            )
        for result in (
            freenect.set_depth_mode(
                dev,
                freenect.RESOLUTION_MEDIUM,
                freenect.DEPTH_11BIT,
            ),
            freenect.set_video_mode(
                dev, freenect.RESOLUTION_MEDIUM, freenect.VIDEO_IR_10BIT
            ),
        ):
            if result < 0:
                raise RuntimeError("Cannot configure Kinect video/depth streams")
        latest = {}
        warming = True
        warmup_frames = 30
        settling = 0
        rgb_deadline = None
        warmup_retry = True
        rgb_resolution = (
            freenect.RESOLUTION_HIGH if high_res else freenect.RESOLUTION_MEDIUM
        )
        exposure_metadata = {}
        exposure_controls = None
        exposure_phase = "ready"
        exposure_checked = 0

        def switch_video(ir=False):
            nonlocal video_started, settling, exposure_metadata, exposure_controls, exposure_phase
            if freenect.stop_video(dev) < 0:
                raise RuntimeError("Cannot stop video for IR/RGB switch")
            video_started = False
            if (
                freenect.set_video_mode(
                    dev,
                    freenect.RESOLUTION_MEDIUM if ir else rgb_resolution,
                    freenect.VIDEO_IR_10BIT if ir else freenect.VIDEO_RGB,
                )
                < 0
            ):
                raise RuntimeError("Cannot configure IR/RGB mode")
            if freenect.start_video(dev) < 0:
                raise RuntimeError("Cannot restart video after mode switch")
            video_started = True
            settling = 2
            if not ir:
                # Let RGB firmware startup and white balance finish before locking
                # shutter/gain. Register readback accesses pending frame values.
                try:
                    exposure_controls = RGBExposureControl(freenect, dev)
                except ExposureControlUnavailable:
                    if rgb_exposure_mode == "manual":
                        raise
                    exposure_controls = None
                exposure_metadata = {"rgb_exposure_mode": "auto", "rgb_exposure_controls": False}
                if exposure_controls is not None:
                    exposure_controls.apply("auto", rgb_shutter_speed, rgb_gain)
                    exposure_metadata = exposure_controls.verify()
                exposure_phase = "auto_settling" if rgb_exposure_mode == "manual" else "ready"
                if rgb_exposure_mode == "manual":
                    settling = RGB_MODE_FPS["rgb_high_res" if high_res else "rgb_low_res"]
            latest.clear()
            connection.send(("phase", "warming IR" if ir else "waiting for RGB"))

        def depth_callback(device, array, stamp):
            latest["depth"] = (array.copy(), int(stamp))

        def video_callback(device, array, stamp):
            nonlocal warmup_frames, rgb_deadline, settling
            if warming:
                warmup_frames -= 1
                return
            rgb_deadline = None
            if settling:
                settling -= 1
                return
            latest["rgb"] = (array.copy(), int(stamp))

        freenect.set_depth_callback(dev, depth_callback)
        freenect.set_video_callback(dev, video_callback)
        if freenect.start_depth(dev) < 0:
            raise RuntimeError("Cannot start Kinect depth stream")
        depth_started = True
        if freenect.start_video(dev) < 0:
            raise RuntimeError("Cannot start Kinect RGB stream")
        video_started = True
        rgb_out = np.frombuffer(rgb_buffer, np.uint8).reshape(
            (1024, 1280, 3) if high_res else RGB_SHAPE
        )
        depth_out = np.frombuffer(depth_buffer, np.uint16).reshape(DEPTH_SHAPE)
        awaiting_copy = False
        while not stop_event.is_set():
            # This call can block inside libusb; the parent enforces a deadline.
            process_events = getattr(
                freenect, "process_events_timeout", freenect.process_events
            )
            if process_events(ctx) < 0:
                raise RuntimeError("Kinect USB stream disconnected or failed")
            if warming:
                if warmup_frames <= 0:
                    switch_video()
                    warming = False
                    rgb_deadline = time.monotonic() + 1.5
                continue
            if (
                rgb_deadline is not None
                and warmup_retry
                and time.monotonic() > rgb_deadline
            ):
                warmup_retry = False
                switch_video(ir=True)
                warming = True
                warmup_frames = 30
                continue
            if exposure_phase != "ready":
                if settling:
                    continue
                latest.clear()
                if exposure_phase == "auto_settling":
                    exposure_controls.apply("manual", rgb_shutter_speed, rgb_gain)
                    settling = 2
                    exposure_phase = "manual_settling"
                else:
                    exposure_metadata = exposure_controls.verify()
                    exposure_checked = time.monotonic()
                    exposure_phase = "ready"
                continue
            if awaiting_copy:
                if not connection.poll():
                    continue
                connection.recv()
                awaiting_copy = False
            if "rgb" not in latest or "depth" not in latest:
                continue
            if rgb_exposure_mode == "manual" and time.monotonic() - exposure_checked >= 0.5:
                exposure_metadata = exposure_controls.verify()
                exposure_checked = time.monotonic()
            rgb, rgb_stamp = latest["rgb"]
            depth, depth_stamp = latest["depth"]
            delta = timestamp_delta_ms(rgb_stamp, depth_stamp)
            if abs(delta) > 50:
                # Keep the newer frame and wait for its matching stream.
                latest.pop("depth" if delta > 0 else "rgb")
                continue
            rgb_out[:] = rgb
            depth_out[:] = depth
            latest.clear()
            connection.send(
                (
                    "frame",
                    {
                        "captured_monotonic_s": time.monotonic(),
                        "timestamp_s": time.time(),
                        "depth_timestamp_ticks": depth_stamp,
                        "rgb_timestamp_ticks": rgb_stamp,
                        "device_timestamp_hz": 60_000_000,
                        "rgb_depth_delta_ms": delta,
                        "rgb_mode": "rgb_high_res" if high_res else "rgb_low_res",
                        "rgb_fps": RGB_MODE_FPS[
                            "rgb_high_res" if high_res else "rgb_low_res"
                        ],
                        "depth_encoding": "raw_11bit",
                        **exposure_metadata,
                    },
                )
            )
            awaiting_copy = True
    except Exception as exc:  # noqa: BLE001 -- relay driver failures to the supervisor
        try:
            connection.send(("error", str(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        # Cleanup may itself hang after USB failure. The parent bounds it too.
        if dev is not None:
            if video_started:
                freenect.stop_video(dev)
            if depth_started:
                freenect.stop_depth(dev)
            freenect.close_device(dev)
        if ctx is not None:
            freenect.shutdown(ctx)
        connection.close()
