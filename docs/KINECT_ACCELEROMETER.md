# Kinect v1 accelerometer: acquisition, tracking, and portrait capture

Feasibility study, 7 October 2026. This is a proposed design, not an implemented
scanner feature. Driver support and the installed Python API were checked;
the hardware probe found no connected Kinect. No simultaneous-capture rate,
orientation accuracy, or tracking improvement has been measured on this device.

## Findings

| Question | Finding | Practical consequence |
| --- | --- | --- |
| Can we access the internal motion sensor? | Yes: Kinect v1 exposes a three-axis accelerometer. Its libfreenect interface exposes no gyroscope or magnetometer. | Treat this as acceleration/gravity sensing, not a complete six-axis IMU. |
| Can it run alongside RGB and depth? | Yes in the driver: the upstream example starts both image streams and polls acceleration in the event loop. | Use the existing device handle. Measure USB-call latency and image delivery before choosing a polling rate. |
| Is it a synchronized stream? | The API returns polled state without a sensor timestamp, sample sequence, or FIFO. | Host timestamps describe the read, not the physical sample or image exposure. |
| Can it improve tracking? | A gravity direction supplies two orientation constraints during sufficiently gentle motion. | Test roll/pitch initialization and consistency checks alongside visual/depth evidence. It supplies no absolute position or yaw. |
| Can it recognize portrait orientation? | Yes when gravity has a clear projection into the camera image plane. | Rotate previews and export portrait copies with hysteresis and a manual lock. Looking straight up/down is ambiguous. |

These hardware/API findings follow the [OpenKinect project](https://github.com/OpenKinect/libfreenect),
its [public API](https://github.com/OpenKinect/libfreenect/blob/master/include/libfreenect.h),
and its [streaming example](https://github.com/OpenKinect/libfreenect/blob/master/examples/glview.c).
The model and implementation choices below are engineering proposals for this repository.

## What is accessible today

The existing asynchronous acquisition in
[`kinect_scanner/capture_process.py`](../kinect_scanner/capture_process.py) owns
one context/device and runs RGB plus native raw depth. Acceleration can be polled
on that same device, without switching image modes or moving the tilt motor:

```python
read_start_s = time.monotonic()
result = freenect.update_tilt_state(dev)
read_end_s = time.monotonic()
if result < 0:
    # Mark this measurement unavailable; do not reuse cached state as fresh.
    ...
else:
    state = freenect.get_tilt_state(dev)
    acceleration_m_s2 = freenect.get_mks_accel(state)
```

Check for negative errors, rather than requiring a zero success return. The
ordinary motor-interface implementation returns a positive transfer byte count
on success, despite the header's zero-success description. Also verify capability
and plausible samples: a disabled motor interface can return zero without a new
measurement. The convenience `get_accel(dev)` discards the update return value,
so it can hand back stale cached data after a read failure. Copy numeric values
immediately; `get_tilt_state()` references the driver's mutable state.

`get_mks_accel()` scales raw counts to m/s². It does **not** remove gravity,
despite the header's potentially confusing wording. A stationary measurement
should have magnitude near 9.81 m/s² after calibration. Use all three axes;
`get_tilt_degs()` alone is insufficient to recognize left/right portrait roll.
The [tilt implementation](https://github.com/OpenKinect/libfreenect/blob/master/src/tilt.c)
also has an alternative audio-interface path for some hardware revisions.
Identify the actual device and verify scale/error behavior instead of assuming
every v1 revision follows the ordinary motor path.

The installed module at
`/Users/sasha/hobby/xbox360/.venv/lib/python3.13/site-packages/freenect.cpython-313-darwin.so`
exposes `update_tilt_state`, `get_tilt_state`, `get_mks_accel`, and `get_accel`.
The local Python wrapper's `init()` selects both camera and motor interfaces;
it does not export `select_subdevices()`. Thus this environment needs no new
Python binding for the ordinary accelerometer API. The local libfreenect source
has base revision `09a1f098040d00e6070c18174904547ec31d2774` with local wrapper
modifications. Other installations still need capability detection.

### Local probe and remaining hardware evidence

The bounded probe attempted VGA and high-resolution RGB modes on 7 October 2026.
Both reported `device_count: 0` and `No Kinect connected` before opening a device.
The [probe record](benchmarks/accelerometer-probe.json) preserves that result.
No images were saved and no motor commands were issued. The reproducible local
probe is retained outside Git at
`/Users/sasha/hobby/xbox360/accelerometer-investigation-20261007/probe.py`:

```bash
/Users/sasha/hobby/xbox360/.venv/bin/python \
  /Users/sasha/hobby/xbox360/accelerometer-investigation-20261007/probe.py
```

With the scanner closed and a Kinect connected, it warms IR, switches to RGB,
then measures 4 s without polling, 7 s requesting 20 Hz polls, and another 4 s
without polling, in each RGB mode. Each mode runs in a subprocess with a 28 s
deadline. It reports callback counts, callback interval p95, read failures,
read latency, acceleration mean/scatter, and distinct raw vectors. Its observed
poll frequency is a host-read rate, not a verified sensor sampling frequency.
This is a first concurrency check; it does not exercise pairing, uploads, or
tracking. Sustained end-to-end acquisition must be checked separately.

## Acquisition and timing design

Poll initially at a configurable **20 Hz trial rate**, on the acquisition child
between event-processing calls. Keep polling independent of selected uploads,
frame-copy acknowledgements, and reconstruction backpressure. Do not poll from
image callbacks, the GUI thread, or a second process that opens the same Kinect.
Do not assume the device handle supports concurrent calls from several threads.
The API is blocking; its small response does not guarantee a small latency.
The current subprocess supervisor remains necessary for native-driver stalls.

Retain a bounded ring of copied samples. For every read, store its start/end
monotonic host times, midpoint estimate, latency, raw counts, scaled acceleration,
read status, and capture-generation identifier. Reconnects reset the history
and clock mapping. A failed read marks data unavailable; it must not refresh
the age of the previous sample. Repeated ordinary read errors disable acceleration
and keep image capture running. A native call that hangs still requires the
existing capture restart; remember the acceleration failure across that restart
so it cannot trigger an endless restart loop.

RGB/depth use a wrapping 60 MHz device clock, while acceleration reads have no
device clock. Record host receipt time in each image callback and maintain a
bounded, wrap-aware device-clock-to-host mapping per connection. Associate a
sample with the estimated time of the chosen depth observation, rather than
with the later time at which the pair is published. Keep receipt/transfer delay
uncertainty visible. A read midpoint is only an association estimate: unknown
internal sample age and packet-end versus exposure timing remain unresolved.
Nearest-sample association is adequate to trial slow orientation decisions;
precise inertial integration would require stronger timing evidence.

Send a compact versioned `metadata.accelerometer` summary with the selected
pair: validity/reason, acceleration, calibrated gravity direction if available,
read interval, sample age, connection identity, and calibration identity.
[`shared/protocol.py`](../shared/protocol.py) already carries JSON metadata but
limits it to **4096 bytes**, including visual tracking and exposure fields.
Do not attach an unbounded sample history. For statistical experiments, retain
the full bounded-rate sample sequence in a recording sidecar, with an explicit
clock epoch and links from frames. Current recordings preserve selected-frame
metadata, not all intermediate camera observations or acceleration samples.
Missing acceleration remains compatible with older recordings.

Before enabling polling by default, compare image rates, pairing deltas, frame
age, and restart behavior with polling off/on in both RGB modes and the normal
manual-exposure path. Lower the poll rate or disable the feature if reads consume
too much of the event loop. More host polls do not necessarily yield new samples.

## How it can help the current tracker

The scanner already has two relevant components:

- [`shared/visual_tracking.py`](../shared/visual_tracking.py) follows incoming
  RGB-D pairs with optical flow, PnP, measured 3D correspondences, and joint
  refinement. A broken chain gets a new segment.
- [`scanner_server/engine.py`](../scanner_server/engine.py) uses that motion as an
  initializer, otherwise predicts from previous accepted poses, and checks
  visual/depth/model evidence before fusion.

Keep these components. Add acceleration first as independent diagnostic evidence,
then trial an orientation prior. The camera-side path can use gravity when color
tracking is unavailable, but gravity cannot restore a broken visual chain.
Preserve the existing 20 ms RGB/depth assistance guard, geometry checks, and
pause-on-loss behavior. In particular, acceleration cannot compensate for bad
RGB/depth pairing or authorize relocalization by itself.

### Proposed statistical model

Use **robust gravity-constrained RGB-D tracking** rather than integrating the
accelerometer into a position estimate. In conventional notation its observation
is approximately

```text
f_sensor = R_world_from_sensorᵀ (a_world - g_world) + bias + noise
```

At rest or sufficiently gentle motion, calibrated `f_sensor` points upward.
During acceleration it mixes gravity and actual motion. The
[Analog Devices inclination note](https://www.analog.com/en/resources/app-notes/an-1057.html)
explains this ambiguity and the response delay introduced by filtering. A norm
near 1 g alone is insufficient to certify a gravity observation.

Calibrate accelerometer bias/scale in several stationary orientations and fit
the fixed rotation from sensor axes to the scanner's depth-camera axes. Verify
that relationship across supported tilt settings. Do not assume driver axes,
sign, or a fixed sensor/camera relationship from names alone. Bind this extra
calibration to the device and camera-calibration publication.

Let `u_i` be the measured unit up vector in camera coordinates and `R_i` the
camera-to-world rotation. Establish `u_world` from an accepted stationary
reference: `u_world = R_0 u_0`. This preserves the scan's existing coordinate
system rather than forcing the first camera pose to a new world-up convention.
A proposed two-component residual is

```text
r_gravity = Bᵀ (R_i u_i - u_world)
E = E_existing_RGBD + w_i · robust_loss(r_gravityᵀ Σ_i⁻¹ r_gravity)
```

Here `B` spans the plane perpendicular to world up. Restrict gravity correction
to the local solution hemisphere, or additionally check the dot product, because
this projected residual alone is also zero for an inverted up vector. Gravity
constrains two rotation directions; yaw around gravity and all translation
remain unconstrained. This is a design equation, not a claim that Open3D's current
ICP accepts an extra residual or exposes a calibrated pose covariance.

For a first implementation, compute a bounded minimal gravity alignment of the
existing rotation seed, retain its translation and unconstrained heading, and
run the normal raw-evidence verification. Report gravity disagreement alongside
accepted candidates. Do not simply adjust the accepted pose after ICP: that would
change the projection without rechecking the observations. A later joint solver
could add the residual to camera-side refinement and final graph optimization,
with fresh validation and reintegration after pose changes.

Make confidence adaptive. Start with a robust short-window direction estimate,
measured stationary scatter, sample age, read latency, norm deviation, and
disagreement with accepted visual motion. Transport samples using visual rotation
before interpreting their directional scatter: raw sensor-axis changes can be
legitimate camera rotation. A small two-state probabilistic model or HMM can
estimate **gravity reliable / acceleration contaminated** from these signals;
its reliability probability sets `w_i`. Increase uncertainty or drop the gravity
term when motion/timing/calibration is suspect. Initial weights and thresholds
are engineering heuristics until held-out captures calibrate them. Do not count
a filtered estimate and its underlying raw samples as independent observations.

This should be tried before a full EKF or learned model. The current tracker
does not provide a measured pose covariance, and the Kinect supplies no angular
velocity. An EKF would need a validated motion/noise model; its existence would
not solve those observability limits. A neural model needs motion ground truth
and varied device recordings that the existing archives do not contain.

### Expected gains and limits

Potential gains are steadier roll/pitch, a better rotational initializer after
gentle viewpoint changes, rejection/ranking of implausibly tilted candidates,
and a gravity-based up direction for the tracking overview. A portrait decision
can continue even when RGB-D tracking fails, provided acceleration is reliable;
the current camera position must still be shown as unknown.

Gravity cannot disambiguate sliding translation on a blank plane, recognize a
place, prevent yaw drift, or recover a trajectory through a long visual outage.
Even a hypothetical constant 0.01 g acceleration bias creates about **1.23 m**
position error after 5 s of double integration (`0.5 × 0.01 × 9.81 × 5²`). This
is an illustrative calculation, not a measured Kinect bias. Tilt error also
leaks gravity into apparent translation. Deriving angular velocity by differencing
noisy gravity vectors still leaves yaw missing.

For genuinely stronger tracking during fast motion or image loss, a rigidly
mounted, timestamped accelerometer-plus-gyroscope would be a separate option,
requiring timing/extrinsic calibration and new evaluations. The research paper
[Dense RGB-D-Inertial SLAM with Map Deformations](https://arxiv.org/abs/2207.10940)
is a useful reference for joint estimation of pose, velocity, biases, and gravity;
its full-inertial results are not evidence for gains from Kinect's accelerometer alone.

## Portrait previews and recordings

This is the most direct first user-facing feature. In calibrated camera axes
`x = right`, `y = down`, `z = forward`, let `d = -u` be the down direction. Use
its image-plane projection:

```text
roll = atan2(d_x, d_y)
display_rotation_cw_degrees = nearest_quarter_turn(roll)
```

An upright image has down along `+y` and needs 0°. Down along `+x` needs 90°
clockwise. Verify every sign with physical upright/left/right/upside-down tests.
The [upstream viewer](https://github.com/OpenKinect/libfreenect/blob/master/examples/glview.c)
already demonstrates accelerometer-driven image rotation, though its filtering
and synchronization are too simple to adopt unchanged.

Use four discrete states, circular angle comparisons, hysteresis, and a dwell
time. Starting trial values could be a 0.2–0.3 s filter, 15° extra margin beyond
the 45° state boundary, and 0.3 s of consistent evidence before changing state.
Hold the last state when data is stale/contaminated or the normalized image-plane
projection is small (trial threshold 0.35). A camera pointing straight up/down
has no gravity-defined image roll. Offer **Auto / Landscape / Portrait left /
Portrait right**, including a lock during recording. These values must be tuned
on physical movement, especially genuine slow roll versus linear acceleration.

Apply the committed quarter-turn to both RGB and depth preview copies, including
the RGB view, Depth view, Scan view, and stored recovery reference thumbnails.
Colorize depth and draw its crop outline in native coordinates before rotating
the presentation. Preserve aspect fit. A 90° turn displays VGA RGB/depth as
480 × 640 and high-resolution RGB as 1024 × 1280 (width × height); the sensor's
resolution and field of view do not increase.

Record the committed orientation and its validity/reason **for each captured
pair**, so replay uses the capture decision rather than reclassifying it with
different thresholds. Continue to save original lossless RGB and uint16 depth
plus their existing calibration. To make ordinary image viewers actually show
portrait files, optionally save/export rotated PNG copies under separate
`display/rgb/` and `display/depth/` paths. A private manifest field alone will
not rotate a PNG in an external viewer. Quarter-turn copies need no interpolation;
depth values, holes, and units must remain unchanged. Longer video exports should
lock orientation or use a fixed canvas, because frame dimensions cannot switch
casually within a video stream.

Do **not** replace the scanner's observation arrays with these rotated copies.
Its [wire format](../shared/protocol.py), shared buffers, ROI conventions, and
[`prepare_rgbd()`](../shared/calibration.py) expect native dimensions and camera
coordinates. Making portrait arrays authoritative would require changes to
intrinsics, distortion, RGB/depth extrinsics, masks, serialization, and all pose
conventions. Rotating presentation/export copies gives the requested portrait
view and saved images while retaining replayable measurement geometry. It must
also avoid double rotation when a portrait derivative is opened again.

## Suggested implementation order and acceptance

1. **Acquire and characterize acceleration.** Extend the existing child and
   `scripts/check_camera.py` with an optional acceleration probe. Check both RGB
   modes, exposure settings, unplug/reconnect, stale/error reads, and bounded
   shutdown. Compare off/on image rates, pairing, frame age, and read-latency
   distributions. Measure stationary vectors and different physical orientations;
   confirm useful fresh-data frequency and the sensor/camera relationship.
2. **Deliver portrait display and export.** Add an independent orientation helper,
   frame metadata, manual controls/preferences, preview transforms, and optional
   portrait PNG copies. Test all four orientations, boundary jitter, look-up/down
   ambiguity, rotation while recording, sensor absence, and reconnect. Exact
   uint16 round trips and unchanged source RGB/depth/calibration must pass;
   reopening the portrait copies must produce the expected dimensions and view.
3. **Collect motion evidence.** Record synchronized-enough raw RGB-D, every
   acceleration read and timing diagnostics, and independent orientation/pose
   reference. Include slow scans, rapid translation, rolls, vertical views, blank
   planes, repeated texture, tracking loss, and stationary intervals. Old chest
   archives contain no acceleration, so they cannot prove a benefit from it.
4. **Trial gravity assistance off/on.** First log advisory residuals, then compare
   bounded seed correction or joint residuals on the same new held-out recordings.
   Measure roll/pitch and position errors, yaw drift, wrong registrations, loss
   duration, surface disagreement, and acquisition/processing cost. More accepted
   frames alone is insufficient. Corrupted/stale acceleration must degrade to the
   existing tracker; gravity must never turn a failed visual/geometric match into
   permission to fuse. Promote the feature only after it improves accuracy or
   recovery without increasing incorrect registrations.

The recommended first implementation is acquisition plus portrait handling.
Gravity-assisted tracking is a separate measured experiment; inertial position
tracking from this built-in sensor is not a dependable recovery strategy.
