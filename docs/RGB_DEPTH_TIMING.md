# RGB/depth timing and color assistance

The warning means that a requested RGB-assisted pose estimate is unavailable
for the current stored observation because its RGB and depth device timestamps
are more than 20 ms apart. Geometric tracking can still run. RGB is still saved
and used for fused point/vertex color; it is not discarded by this warning.
The image-projected UV texture exporter also selects views within 20 ms, while
remaining texels can use fused vertex color. Color recovery is conditional even
with good timing: it needs useful texture and independent geometric validation.

## Second chest session: 6 October 2026

Measured directly from `manifest.json` and `reconstruction.json` inside
`chest-2scan-session_20261006_090302.zip`, without changing the archive:

| Measurement | Result |
| --- | --- |
| Stored observations | 200 |
| RGB mode | 1280 × 1024, 10 fps, automatic exposure |
| Absolute timestamp gap: minimum / median / maximum | 8.87 / 35.23 / 49.99 ms |
| Gap above the 20 ms color-assistance limit | 183 of 200 (91.5%) |
| Observations individually eligible by timing | 17 of 200 (8.5%) |
| Successive accepted observations both eligible | 0 |
| Accepted poses | 123 |
| Accepted observations eligible for image-projected textures by timing | 12 |

All signed gaps are negative: the stored RGB timestamp precedes the depth
timestamp. Adjacent RGB-D odometry requires both observations to pass the timing
check; it therefore had no eligible pair against the preceding accepted pose in
this scan. Appearance relocalization can use non-adjacent eligible references,
but the recorded successful tracking methods were ICP (120), anchor + ICP (2),
and the initial reference (1). This establishes that RGB-assisted tracking did
not succeed here; it does not establish that timing caused every tracking loss.

## Concurrent streams are not simultaneous observations

The [libfreenect mode table](https://github.com/OpenKinect/libfreenect/blob/master/src/cameras.c)
lists high-resolution RGB at 10 fps, low-resolution RGB at 30 fps, and raw depth
at 30 fps. High-resolution RGB therefore produces one frame every 100 ms while
depth produces one every roughly 33 ms. The callbacks arrive independently.
Having both cameras running does not provide a single simultaneous RGB-D frame.

The [driver's stream parser](https://github.com/OpenKinect/libfreenect/blob/master/src/cameras.c)
passes the final packet's device timestamp to each callback. Its
[OpenNI adapter](https://github.com/OpenKinect/libfreenect/blob/master/OpenNI2-FreenectDriver/src/VideoStream.hpp)
documents the 60 MHz clock. The scanner correctly handles its approximately
72-second wrap before conversion. The saved timing values agree with the raw
timestamp ticks; this is not a millisecond/unit-conversion mistake.

These packet-end timestamps are a useful pairing signal, but do not measure
the cameras' exposure midpoints. Exposure duration, readout, and stream transfer
timing can leave an actual temporal offset even when reported timestamps match.
We have not calibrated that offset on this device.

## Why retain a limit?

Color-assisted tracking assigns RGB features to measured 3D depth points. If
the camera moves between the observations, a color feature may be attached to
the wrong surface, especially near edges. At 10 degrees/second, a true 40 ms
observation offset spans 0.4 degrees: about four pixels with this session's
depth focal length (~586 px). That can matter for a feature solver using a
2 px reprojection gate. This example illustrates motion sensitivity; the
reported packet gap is not a measurement of the exposure offset.

**20 ms is an engineering guard, not a Kinect specification or a calibrated
guarantee.** It can reject usable data when the camera is stationary, and cannot
guarantee good correspondence during rapid motion. Raising it to 50 ms would
make all 200 saved pairs eligible by timing, but would not improve their
alignment. A future motion-dependent policy or exposure-time correction needs
measurement and validation, especially when tracking is already lost.

The separate 50 ms capture limit bounds gross pairing mismatch and admits RGB
for recording and fused model color. That less demanding use can still show
motion-related color blur; passing its limit is not a color-quality guarantee.

## Pairing improvement

Previously acquisition kept only the latest depth frame. A late RGB callback
could pair its older RGB timestamp with a newer depth frame after the closer
depth had already been overwritten. A 40 ms gap against the latest depth can
be about 7 ms against the preceding depth at 30 fps. The one-sided gaps in this
archive are consistent with this failure, though the discarded depth frames
were not recorded and cannot be inspected retrospectively.

Acquisition now retains eight depth frames (~267 ms), chooses the nearest unused
depth timestamp, and waits for depth to reach a leading RGB timestamp before
choosing the closer surrounding observation. Newer unused depth frames are
retained. Slow consumers keep only the newest pending RGB, and mode/exposure
changes clear the buffer. The 20 ms assistance and 50 ms capture limits remain
shared across capture, tracking, refinement, texture selection, and the warning.

With uninterrupted 30 fps depth and enough history, a nearest timestamp should
normally be within roughly half a depth period (~17 ms), including 10 fps RGB.
Dropped frames or delayed callbacks beyond the history can still produce a
warning. This expected bound concerns device timestamps, not exposure accuracy.
Recorded metadata now identifies `nearest_depth` pairing and `packet_end`
timestamps. Deterministic tests cover delayed callbacks, leading RGB,
wraparound, buffer ownership, missing depth, backpressure, and both stream rates.
Validation: 75 tests passed across acquisition, exposure, recovery, appearance,
fragments, textures, and the live API; one CUDA test was skipped on this Mac.
The simulated delayed-arrival check paired all 20 observations within 17 ms
in each RGB mode. This is software verification, not a hardware measurement.

The connected scanner was running during this change, so it was not interrupted
for a hardware comparison. Restart the scanner using the updated checkout to
apply the change. With the scanner closed, run:

```bash
python scripts/check_camera.py --seconds 10 --rgb-mode rgb_high_res
python scripts/check_camera.py --seconds 10 --rgb-mode rgb_low_res
```

The report includes the median/maximum absolute gap and the number of frames
eligible for color assistance. In the GUI, **RGB camera → Resolution and frame
rate → Motion detail · 640 × 480, 30 fps** offers more frequent RGB observations
and less motion between RGB frames, at the cost of color resolution. Good light,
short exposures, slow movement, and a stationary subject improve correspondence;
moving slowly alone will not eliminate a timestamp-based warning. Better pairing
cannot repair this archive: its intervening depth frames were never saved.
