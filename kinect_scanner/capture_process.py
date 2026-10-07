"""Isolated libfreenect acquisition; native USB calls may never return."""

import time
import queue
import uuid

import numpy as np

from shared.capture import DeviceClockMapper, RGB_MODE_FPS, RGBDepthPairer, timestamp_delta_ms, validate_rgb_exposure
from shared.inertial import AccelerometerPoller, GravityEstimator
from shared.sensor_recording import SensorJournal

from .rgb_exposure import ExposureControlUnavailable, RGBExposureControl

RGB_SHAPE = (480, 640, 3)
DEPTH_SHAPE = (480, 640)


def capture_frames(connection, stop_event, rgb_buffer, depth_buffer, high_res=True,
                   rgb_exposure_mode="auto", rgb_shutter_speed=125, rgb_gain=1,
                   controls=None, accelerometer_enabled=True, accelerometer_calibration=None):
    """Publish one shared-memory pair at a time, awaiting a copy acknowledgement.

    Only small messages cross the pipe. The child never overwrites a published
    frame until the parent has copied it. Terminating a stalled driver therefore
    cannot leave the GUI blocked on a frame payload or a shared-memory lock.
    """
    ctx = dev = None
    depth_started = video_started = False
    journal = None
    retired_journals = []
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
        generation = uuid.uuid4().hex
        acceleration = AccelerometerPoller(freenect, dev, generation, calibration=accelerometer_calibration, enabled=accelerometer_enabled)
        clock = DeviceClockMapper()
        image_info = {"rgb": {}, "depth": {}}
        image_sequence = {"rgb": 0, "depth": 0}
        pending_flushes = []
        recording_status_sent = 0
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
        pairer = RGBDepthPairer()
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
            pairer.clear()
            connection.send(("phase", "warming IR" if ir else "waiting for RGB"))

        def image_metadata(stream, stamp):
            image_sequence[stream] += 1
            info = {"sequence": image_sequence[stream], "device_timestamp_ticks": int(stamp),
                    "capture_generation": generation, "timestamp_s": time.time(),
                    **clock.observe(stamp, time.monotonic())}
            image_info[stream][int(stamp)] = info
            if len(image_info[stream]) > 64:
                del image_info[stream][next(iter(image_info[stream]))]
            return info

        def depth_callback(device, array, stamp):
            info = image_metadata("depth", stamp)
            if journal is not None:
                journal.submit("depth", info, array)
            pairer.add_depth(array, stamp)

        def video_callback(device, array, stamp):
            nonlocal warmup_frames, rgb_deadline, settling
            if warming:
                warmup_frames -= 1
                return
            info = image_metadata("rgb", stamp)
            if journal is not None:
                mode = "rgb_high_res" if high_res else "rgb_low_res"
                journal.submit("rgb", {**info, **exposure_metadata, "rgb_mode": mode,
                                       "rgb_fps": RGB_MODE_FPS[mode], "exposure_phase": exposure_phase,
                                       "settling_frames_remaining": settling}, array)
            rgb_deadline = None
            if settling:
                settling -= 1
                return
            pairer.add_rgb(array, stamp)

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
            # Recording commands use a separate bounded queue; image ACKs keep
            # their original pipe ownership protocol and cannot consume commands.
            if controls is not None:
                for _ in range(8):
                    try:
                        command, payload = controls.get_nowait()
                    except queue.Empty:
                        break
                    if command == "record":
                        if journal is not None:
                            retired_journals.append(journal)
                            journal.close()
                            journal = None
                        if payload is not None:
                            try:
                                requested_settings = payload.get("settings") or {}
                                acceleration.estimator = GravityEstimator(requested_settings.get("accelerometer_calibration", accelerometer_calibration))
                                effective_settings = {**requested_settings, "accelerometer_calibration": acceleration.estimator.calibration}
                                journal = SensorJournal(payload["path"], generation + "-" + uuid.uuid4().hex[:8], effective_settings, capture_generation=generation)
                                journal.submit("events", {"type": "accelerometer_capability", "enabled": acceleration.enabled,
                                                          "reason": acceleration.reason, "host_monotonic_s": time.monotonic()})
                                acceleration.samples.clear()
                                pairer.clear()  # First pair must belong to this recording.
                            except (OSError, ValueError, TypeError) as exc:
                                connection.send(("sensor_status", {"complete": False, "error": str(exc)}))
                    elif command == "flush":
                        pending_flushes.append(payload)
                    elif command == "event" and journal is not None:
                        journal.submit("events", payload)
                    elif command == "stop_recording" and journal is not None and str(journal.path.parent) == payload:
                        retired_journals.append(journal)
                        journal.close()
                        journal = None
            for token in pending_flushes[:]:
                root = token["root"]
                if any(str(old.path.parent) == root and old.thread.is_alive() for old in retired_journals):
                    continue
                if journal is None or str(journal.path.parent) != root:
                    connection.send(("sensor_flush", {"request_id": token["id"], "path": None}))
                    pending_flushes.remove(token)
                elif journal.request_flush(token["id"]):
                    pending_flushes.remove(token)
            if journal is not None:
                while not journal.notifications.empty():
                    connection.send(("sensor_flush", journal.notifications.get()))
                if time.monotonic() - recording_status_sent > 1:
                    connection.send(("sensor_status", journal.status()))
                    recording_status_sent = time.monotonic()
            if acceleration.enabled and time.monotonic() >= acceleration.next_poll:
                # The supervisor disables acceleration on the next connection if
                # this native call hangs, instead of entering a restart loop.
                connection.send(("accelerometer_poll", True))
                if journal is not None:
                    journal.submit("events", {"type": "accelerometer_read_started", "sequence": acceleration.sequence + 1,
                                              "host_monotonic_s": time.monotonic()})
                sample = acceleration.poll()
                if sample is not None:
                    if journal is not None:
                        journal.submit("accelerometer", sample)
                    connection.send(("accelerometer", sample))
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
                pairer.clear()
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
            pair = pairer.pop_pair()
            if pair is None:
                continue
            if rgb_exposure_mode == "manual" and time.monotonic() - exposure_checked >= 0.5:
                exposure_metadata = exposure_controls.verify()
                exposure_checked = time.monotonic()
            rgb, depth, rgb_stamp, depth_stamp = pair
            depth_observation = image_info["depth"][depth_stamp]
            rgb_observation = image_info["rgb"][rgb_stamp]
            delta = timestamp_delta_ms(rgb_stamp, depth_stamp)
            rgb_out[:] = rgb
            depth_out[:] = depth
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
                        "rgb_depth_pairing": "nearest_depth",
                        "device_timestamp_reference": "packet_end",
                        "rgb_mode": "rgb_high_res" if high_res else "rgb_low_res",
                        "rgb_fps": RGB_MODE_FPS[
                            "rgb_high_res" if high_res else "rgb_low_res"
                        ],
                        "depth_encoding": "raw_11bit",
                        "capture_generation": generation,
                        "sensor_recording_segment": journal.path.name if journal is not None else None,
                        "depth_host_monotonic_s": depth_observation["estimated_host_monotonic_s"],
                        "rgb_host_monotonic_s": rgb_observation["estimated_host_monotonic_s"],
                        "host_mapping_uncertainty_s": depth_observation["host_mapping_uncertainty_s"],
                        "sensor_frame_sequences": {"rgb": rgb_observation["sequence"], "depth": depth_observation["sequence"]},
                        "accelerometer": acceleration.associate(depth_observation["estimated_host_monotonic_s"],
                                                                depth_observation["host_mapping_uncertainty_s"]),
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
        if journal is not None:
            journal.close()
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
