# Kinect v1 accelerometer: acquisition, tracking, and portrait capture

## Update: continuous motion transport and offline constraints, 10 October 2026

Kinect v1 exposes three accelerometer axes and a motor tilt encoder, **no
gyroscope**. The camera-side visual pose is estimated from RGB-D between camera
frames; it does not fuse acceleration into a visual-inertial trajectory.

Normal captures now transmit the intervening raw acceleration read intervals
and compact visual observations. Finish automatically uploads the complete
acceleration journal before reconstruction, independently of **Record all
camera frames**. Pauses, failed reads, reconnect generations, calibration,
dropped observations and completeness remain explicit. The server preserves
the received journal in normal project exports.

Image timing keeps one shared packet-clock epoch and least-delayed host offset,
but measures receipt spread separately for RGB and depth. Their different fixed
delivery delays no longer masquerade as within-stream clock jitter. This does
not establish exact exposure or accelerometer sample times.

Offline depth registration uses reliable local gravity windows as a soft pose
search weight and a broad veto against incompatible roll/pitch, with wider
bounds for unverified calibration and uncertain association. Intermediate
visual history supplies initial poses and a soft search weight. Fixed measured
RGB-D identities constrain forward and reverse pose fits and graph checks.
Spatially distributed intermediate feature pixels and depths also support a
sparse joint camera/landmark fit with selected raw surfaces and shared planes.
Neither acceleration double integration nor gyro preintegration is performed.
These are engineering weights, not calibrated sensor-noise probabilities.
Accelerometers cannot determine yaw, and fast dynamic acceleration reduces
gravity reliability. Optional evidence never replaces measured depth checks.

## Earlier implementation and research notes, 7 October 2026

Concurrent acceleration reads,
full sensor recording/replay, portrait presentation/export, and a conservative
gravity-assisted initializer are implemented. Driver support and the installed
Python API were checked; hardware enumeration still found no connected Kinect.
No simultaneous USB capture rate, physical orientation accuracy, or tracking
improvement has been measured on this device. The inertial preintegration and
calibrated statistical solver discussed later remain research proposals; the
implemented RGB-D joint fit is described above.

Normal recording keeps selected RGB-D captures and the continuous accelerometer
log. **Record all camera frames (large files)** is off by default; the multi-GB/min
stream cost below applies only when explicitly enabled. **Use accelerometer to
assist tracking** is also off by default, while portrait orientation defaults
to **Auto** independently.

## Implemented behavior

- The existing USB child polls acceleration at a trial maximum of 20 Hz using
  the same device as RGB/depth. Each completed read retains counts, m/s²,
  return code, read start/end, host midpoint, wall time, and available tilt
  angle/status. Every failure is retained. The event log also records attempts
  before entering the native call, so an interrupted read is visible.
- Missing APIs, three consecutive failures, or three consecutive reads over
  50 ms disable acceleration without disabling RGB-D. An isolated delayed read
  is rejected by the gravity timing check but allows later fast reads to recover.
  Implausible vectors are retained as failed reads. An image timeout during
  a native sensor call causes a bounded child restart with acceleration disabled
  for the next connection. Sensor traffic cannot conceal a stalled image stream.
- Gravity uses a short robust median and exponential filter. Norm deviation,
  scatter, read latency, and sample age reduce reliability. Image timing uses an
  unwrapped 60 MHz packet clock and the least-delayed host receipts. A host mapping
  spread over 30 ms disables association; read latency must be at most 30 ms and
  paired acceleration age at most 150 ms. These bounds are trial heuristics, and
  host receipt spread does not measure absolute exposure/sample-time error.
- The server blends minimal gravity alignment into its existing motion seed.
  It preserves translation, requires consistent connection/calibration and
  reliable observations, rejects disagreements over 25°, and caps correction at
  3°. Unverified profiles receive one quarter of the verified weight. RGB-D
  registration and acceptance still determine whether a frame is integrated.
  A verified first gravity observation can supply overview up.
- Auto orientation uses a 60° switch threshold, 0.3 s dwell, and holds the last
  quarter turn when gravity is unreliable or nearly along the optical axis.
  It uses the latest reliable host-timed gravity read, at most 250 ms old, without
  requiring exposure-clock association. Thus uncertain camera timing can disable
  tracking assistance while allowing the preview to rotate. The orientation status
  is visible below the Auto/lock selector. Recorded orientation and display copies
  follow this coarse presentation decision; tracking retains its stricter timing
  limits. Full sensor replay uses causal reads at camera host receipt time for
  the same presentation decision.
  Manual landscape/left/right locks persist in preferences and are usable during
  scans. RGB, colorized depth, crop outlines, and recovery references rotate as
  presentation copies. Selected local/server exports include rotated PNGs at
  `display/rgb/` and `display/depth/`; source arrays/intrinsics remain native.

The provisional sensor-to-camera rotation is `diag(-1, -1, +1)`. Its XY roll
signs are inferred from the upstream viewer's Y-up OpenGL projection and rotation
formula; Z merely completes a proper rotation. This is **not measured extrinsic
calibration**. Physically check upright, both portrait directions, upside down,
and optical-axis tilt on this device. Tilt-mechanism position may require a
different measured relationship. Factory-profile data are saved as unverified.

## Compact sessions and optional full camera recording

Accelerometer recording starts with a GUI scan, runs independently of selected captures
and image-consumer backpressure, continues during ordinary Pause, and stops at
Finish/cancel. Save Session stops and drains recording at a fixed boundary,
downloads the server's selected frames/report, and atomically adds the local
sensor journal before replacing the destination. Ordinary saves resume recording
in a new segment afterward. Failed saves preserve an existing destination.
The journal remains locally available under `recordings/sensors-<session hash>/`;
it is not deleted automatically. A server-only API export contains selected
frames and the acceleration journal received at Finish. Optional full image
streams still belong to the client.

Selected lossless RGB-D images remain in the server's normal session archive.
The default local journal retains every accelerometer read and orientation event
without copying, queuing, or writing camera arrays. The checkbox **Record all
camera frames (large files)** adds independent RGB/depth streams for research.
Live camera-side visual tracking continues processing intermediate pairs when
full recording is off; only their storage changes. The compact default cannot
reproduce that exact intermediate visual chain offline, but retains selected
images, their computed motion/decision metadata, and the raw acceleration series
for future reanalysis.

The augmented ZIP uses manifest version 2. Existing `frames`, `settings`, and
`reconstruction.json` remain compatible with selected-frame replay. The additional
`sensor_archive` names its root, ordered segments, and aggregate `complete` flag.
Each `sensors/<segment>/` contains:

| File | Retained evidence |
| --- | --- |
| `configuration.json` | Settings, effective accelerometer calibration/evidence, units, encodings, start times, hardware connection generation, clock semantics. |
| `rgb.jsonl`, `depth.jsonl` | Empty by default. With full camera recording enabled: every image callback's sequence, native/unwrapped device time, host times, mapping spread, array path/shape/type, and RGB exposure/mode/settling state. |
| `rgb/*.npy`, `depth/*.npy` | Only with full camera recording enabled: exact uint8 RGB and uint16 native raw disparity arrays, with pickle disabled. |
| `accelerometer.jsonl` | Every completed read attempt, including errors, raw and converted vectors, host interval, sequence, and versioned derived gravity. |
| `events.jsonl` | Orientation controls, capability status, and read-start attempts. |
| `status.json` | Written/dropped counts, errors, checkpoint prefix lengths, close/completeness state. |

Optional full camera streams include images discarded during RGB exposure settling and
depth during startup. IR warmup images and audio are outside the scan's RGB-D
recording scope. Acceleration is a polled read series, not a hardware-timestamped
stream. Reconnection creates a new hardware clock generation; every save/resume
creates a new segment. Raw acceleration is retained even for intervals with rejected
or unselected images. Optional full recording also retains those camera images. Existing selected-frame
metadata retains the live gravity/orientation decision and initializer report.

A bounded background writer always retains acceleration and events. When full
camera recording is explicitly enabled, it copies borrowed callback arrays
immediately and writes NPY without compression. At nominal rates high-resolution
mode produces 57.75 MB/s of image payload (~3.47 GB/min); VGA produces 46.08 MB/s
(~2.76 GB/min), plus small indices. ZIP storage preserves these streams without
extra compression overhead. Queue overflow or disk errors keep capture responsive
and mark data incomplete explicitly. Periodic immutable index checkpoints make
surviving prefixes exportable after a crash; an unclean segment remains incomplete.
A resumed remote session contains full streams only for intervals this client
actually recorded, not earlier clients' unrecorded observations.

After unzipping a normal compact session, replay selected captures as before:

```bash
OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset recording --path /path/to/session
```

For a session recorded with all camera frames enabled, rebuild pairs and gravity
from the optional independent camera streams:

```bash
OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset recording --path /path/to/session \
  --sensor-streams --recompute-motion --stride 5 --gravity-assistance
# Repeat against the identical raw session for an initializer ablation:
OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset recording --path /path/to/session \
  --sensor-streams --recompute-motion --stride 5 --no-gravity-assistance
```

Full replay recomputes gravity from raw reads, applies recorded orientation-control
events, and optionally recomputes visual motion on **every intermediate pair before
stride selection**. It never treats old estimated poses as ground truth. This
supports improving tracking and comparing versions from the same measurements.
Compact sessions retain selected-frame replay and raw acceleration, but cannot
recreate discarded camera images. Old ZIPs without sensor logs cannot recreate
acceleration either. Record independent reference motion for accuracy
evaluation; accepted-frame counts alone do not demonstrate improvement.

## Accelerometer calibration

For guided recording, close the scanner and other Kinect applications, connect
the Kinect's USB and external power, and run this in the scanner's Python
environment (requires `numpy` and the working `freenect` binding, not a server):

```bash
python scripts/calibrate_accelerometer.py --interactive --output ~/Documents/measured-accelerometer.json
```

The terminal describes each physical position and waits for Enter. It records
one second of settling followed by three seconds of acceleration, averages the
original driver m/s² readings, and retries positions with motion, too few reliable
reads, excessive read latency, or implausible acceleration. `--seconds 5` extends
each recording, `--device-index 1` selects another Kinect, and `--id my-kinect`
sets the profile identity instead of prompting for a name. Native USB acquisition
runs in a spawned process with bounded waits and shutdown; RGB/depth images are
not recorded and the tilt motor is never commanded.

Individual stationary readings can fluctuate because of sensor noise. The guide
checks sustained linear drift and differences between three interval averages
separately from individual scatter. It accepts scatter up to 0.35 m/s² and peaks
up to 0.8 m/s², while rejecting drift or interval shifts above 0.2 m/s². These
are practical capture-quality limits, not proof of motion or physical accuracy;
the fitter's separate residual limits below still apply. Unstable-read messages
identify motion, vibration, and sensor noise as possible causes. Accepted
observations retain scatter, peak, drift, and interval-shift diagnostics.

Move and securely support the **whole Kinect**, preserving the head/base
relationship throughout. Do not force the tilt joint. Align the actual camera
axes using a level/square or an independently measured fixture; a level base
does not guarantee horizontal lenses when the head is tilted. For approximate
mode, follow the directions as closely as practical without measured alignment.
The positions are:

| Position | Camera-up vector in native depth-camera axes |
| --- | --- |
| Upright, lenses horizontal, top toward ceiling | `[0, -1, 0]` |
| Upside down, lenses horizontal, top toward floor | `[0, 1, 0]` |
| Right side toward ceiling, viewed from behind, lenses horizontal | `[1, 0, 0]` |
| Left side toward ceiling, viewed from behind, lenses horizontal | `[-1, 0, 0]` |
| Lenses straight toward ceiling | `[0, 0, 1]` |
| Lenses straight toward floor | `[0, 0, -1]` |

Here x points right in the native image, y down, and z forward through the lenses.
Automatic preview rotation does not change these directions. The guide then
requests three fresh held-out captures (upright, right side up, and lenses up).
Move away and independently realign each validation position; fitting samples
are never reused. The accelerometer can detect changing readings, but cannot
independently establish physical pose accuracy. At the start, the guide asks
whether every reference alignment will be independently checked. Answering no
selects **approximate mode**, which always keeps the output **unverified**.
This mode permits up to 1 m/s² vector disagreement and 6° direction disagreement
against the nominal poses in both fitting and fresh checks. These are practical
trial limits, not independent accuracy measurements. They allow ordinary
approximate placement without a measured fixture, while retaining the checks
for adequate pose diversity, plausible bias/scale, and a proper axis rotation.
They do not prove improved tracking or alignment accuracy. Confirmed independent
references keep the strict 0.25 m/s² and 2° limits for verification. The report
records the mode and limits used. Loading an approximate profile retains the
scanner's reduced assistance weight for unverified calibration.

The guide saves every completed attempt's raw readings, rejected attempts,
accepted means, and reports in `measured-accelerometer.measurements.json`,
checkpointed after each attempt and retained on quit/error. Type `q` or press
Ctrl-C to stop. If the fit fails, re-record a numbered position or the complete
set. Existing output files are preserved; choose a new filename for a new run.
The guide does not automatically resume an interrupted run. A completed
measurement file can also be refitted with the file-mode command below; an
unconfirmed reference automatically uses approximate mode and remains unverified
on refit. There is no need to repeat completed captures solely to change mode:

```bash
python scripts/calibrate_accelerometer.py ~/Documents/measured-accelerometer.measurements.json ~/Documents/measured-accelerometer-approximate.json
```

After success, use **Load Accelerometer Calibration…** to select
`measured-accelerometer.json` (the profile, not the measurements file) before
starting a scan. Use the same device and fixed head tilt as during calibration.
No physical alignment accuracy or real hardware performance is claimed by
the automated synthetic tests of this guide.

`scripts/calibrate_accelerometer.py` fits bias, diagonal scale, and a proper
sensor-to-camera rotation from at least six stationary orientations spanning all
axes. Each input supplies the measured `acceleration_m_s2` vector and an
**independently known** unit `up_camera` vector. Use a levelled fixture or separate
reference; deriving the reference from this same accelerometer would be circular.
For independently checked references, the fitter rejects poorly conditioned
orientations or residuals above 0.25 m/s² or 2°. At least three held-out reference
poses passing these checks are required
to mark the profile verified. Without them it remains unverified.

```json
{"observations": [{"acceleration_m_s2": [0.01, 9.79, 0.04], "up_camera": [0, -1, 0]}],
 "validation": []}
```

This abbreviated structure example is deliberately insufficient to fit. Supply
the full pose sets, then run:

```bash
python scripts/calibrate_accelerometer.py stationary-poses.json measured-accelerometer.json --id my-kinect-fixed-tilt
```

Load the resulting profile with **Load accelerometer calibration…** in the
experimental scan settings before capture. Profiles preserve their calibration
observations, validation, fit algorithm, and residual report in saved sessions.
Physical data collection is still required; no synthetic fit is shipped as a
verified device calibration.

## Validation checkpoint

The integrated full suite ran 366 tests successfully with two skips; 73 targeted
checks against the subsequently merged tracking caches passed with one skip.
Regression checks cover raw retention under image backpressure, read failures, timing
rejection, orientation/native-array separation, exact uint16 portrait exports,
intermediate-motion replay, atomic save failure, and the actual child-process
save/checkpoint barrier. The HTTP/WebSocket/Qt synthetic scan check also passed.
A three-second synthetic
native-array workload wrote 30 high-resolution RGB frames, 90 depth frames, and
60 acceleration records (~173 MB, ~57 MB/s) with zero drops on the local filesystem.
This short run includes OS caching; sustained long-scan storage performance and
disk durability were not measured. See the [validation record](benchmarks/accelerometer-implementation.json).
USB latency, fresh accelerometer frequency,
physical signs/extrinsics, and tracking accuracy still require a connected Kinect.

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
currently limits it to **1 MiB**, including visual tracking and exposure fields.
Do not attach an unbounded sample history. For statistical experiments, retain
the full bounded-rate sample sequence in a recording sidecar, with an explicit
clock epoch and links from frames. The earlier selected-frame implementation
did not preserve intermediate observations. Current compact motion transport
and the complete acceleration journal are described at the top of this guide.
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

## Hardware acceptance and future extensions

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

The first implementation now provides acquisition, portrait handling, raw
archival/replay, and the bounded gravity seed. Validate these on hardware before
claiming tracking gains or extending the joint solver. Inertial position tracking
from this built-in sensor is not a dependable recovery strategy.
