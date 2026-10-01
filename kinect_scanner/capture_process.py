"""Isolated libfreenect acquisition; native USB calls may never return."""

import time

import numpy as np

from shared.capture import timestamp_delta_ms

RGB_SHAPE = (480, 640, 3)
DEPTH_SHAPE = (480, 640)


def capture_frames(connection, stop_event, rgb_buffer, depth_buffer):
    """Publish one shared-memory pair at a time, awaiting a copy acknowledgement.

    Only small messages cross the pipe. The child never overwrites a published
    frame until the parent has copied it. Terminating a stalled driver therefore
    cannot leave the GUI blocked on a frame payload or a shared-memory lock.
    """
    ctx = dev = None
    depth_started = video_started = False
    try:
        import freenect

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
                dev, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_REGISTERED
            ),
            freenect.set_video_mode(
                dev, freenect.RESOLUTION_MEDIUM, freenect.VIDEO_RGB
            ),
        ):
            if result < 0:
                raise RuntimeError(
                    "Cannot configure Kinect RGB/registered depth streams"
                )
        latest = {}

        def depth_callback(device, array, stamp):
            latest["depth"] = (array.copy(), int(stamp))

        def video_callback(device, array, stamp):
            latest["rgb"] = (array.copy(), int(stamp))

        freenect.set_depth_callback(dev, depth_callback)
        freenect.set_video_callback(dev, video_callback)
        if freenect.start_depth(dev) < 0:
            raise RuntimeError("Cannot start Kinect depth stream")
        depth_started = True
        if freenect.start_video(dev) < 0:
            raise RuntimeError("Cannot start Kinect RGB stream")
        video_started = True
        rgb_out = np.frombuffer(rgb_buffer, np.uint8).reshape(RGB_SHAPE)
        depth_out = np.frombuffer(depth_buffer, np.uint16).reshape(DEPTH_SHAPE)
        awaiting_copy = False
        while not stop_event.is_set():
            # This call can block inside libusb; the parent enforces a deadline.
            if freenect.process_events(ctx) < 0:
                raise RuntimeError("Kinect USB stream disconnected or failed")
            if awaiting_copy:
                if not connection.poll():
                    continue
                connection.recv()
                awaiting_copy = False
            if "rgb" not in latest or "depth" not in latest:
                continue
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
