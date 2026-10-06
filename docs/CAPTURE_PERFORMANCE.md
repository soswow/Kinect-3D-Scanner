# Capture and live feedback performance

The capture pipeline now avoids expensive image compression on local connections,
uses a bounded NumPy implementation of confidence-weighted fusion on CPU, reuses
its observed recovery reference cloud, and sends packed float32 live geometry.
Dense calibrated RGB projection also avoids allocating an unused OpenCV Jacobian.
Voxel size, confidence weights, clipping, timestamp limits, and tracking/recovery
acceptance thresholds are preserved.

## Measurements

Measured on Apple Silicon, Python 3.13.7, Open3D 0.20.0, CPU legacy tracking,
`OMP_NUM_THREADS=4`, and 5,000 initial TSDF blocks. The supplied recording
`chest_20261005_230320` has 50 high-resolution RGB/raw-depth captures with
confidence fusion, color assistance, and relocalization enabled. Settings and
capture timestamps were retained. Baseline source: `76ca332`.

Two sequential isolated replays per version produced these results. Stage values
are medians of each run's per-frame median, rather than one pooled percentile.
Detailed captures, poses, and diagnostics remain ignored in `benchmark-output/`;
the committed [summary JSON](benchmarks/capture-profile.json) contains aggregate
measurements only.

| Measurement | Before | After |
| --- | ---: | ---: |
| Complete 50-frame replay, median | 77.8 s | 60.4 s |
| Complete replay, individual runs | 76.2 / 79.3 s | 51.4 / 69.5 s |
| Processing time summed over the 14 accepted frames, median | 21.6 s | 13.1 s |
| Accepted frame processing, median | 1,345 ms | 952 ms |
| Lossless capture packing on loopback | 85.6 ms | 2.3 ms |
| Capture unpacking | 11.3 ms | 1.0 ms |
| Calibrated RGB-D preparation | 51.5 ms | 25.0 ms |
| Confidence-weighted fusion | 854 ms | 383 ms |
| Live snapshot preparation | 4.07 ms | 0.14 ms |
| Live feedback encoding | 82.1 ms | 3.34 ms |
| Live feedback decoding | 37.5 ms | 2.88 ms |
| Live message size, median | 3.51 MB | 1.04 MB |

The full replay's median elapsed time decreased **22%**, and processing the
accepted observations took **40% less time**. Recovery attempts account for most
of the remaining runtime and vary considerably between runs, so the fastest run
is not a general latency promise.

The profiler includes lossless frame packing/unpacking, storage, incremental
processing, and one feedback encode/decode per frame. It excludes startup, image
loading, final mesh/refinement, USB acquisition, actual HTTP/WebSocket transport,
and GUI rendering. It does not reproduce concurrent acquisition or an automatic
capture backlog. The common feedback helper serializes legacy JSON lists for the
baseline and packed geometry for the optimized run.

Both versions accepted exactly Frames 1–14. Their common accepted poses differed
by at most **0.28 mm / 0.017 degrees**. The original saved report contains 42
integrations from before the conservative recovery rules were introduced; it is
not the baseline for this performance comparison. Later unverified transitions
remain rejected. This recording has no reference trajectory: pose agreement is a
regression check, not evidence that the complete chest reconstruction is correct.

## Changes and tradeoffs

- **Local capture:** `localhost` and loopback IP connections use zlib level 0.
  This retains the existing lossless packet format and works with older servers.
  It increases the median high-resolution packet from 3.69 to 4.55 MB. Remote
  connections retain level 1 compression; local replay timings therefore do not
  predict remote bandwidth or upload latency.
- **CPU confidence fusion:** NumPy views update Open3D's CPU attribute buffers
  directly in batches of at most 128 blocks. The TSDF/color equations, confidence
  rejection, truncation gates, and half-away-from-zero projection rounding match
  the tensor implementation. CUDA retains its tensor path.
- **Native RGB projection:** the same measured transform and five-coefficient
  Brown–Conrady model run without `projectPoints`' unused dense derivative matrix.
  Projection is checked against OpenCV for both calibrated RGB resolutions.
- **Recovery:** the registration cloud of the last accepted observation is
  retained and reused. Failed probes cannot replace it. Reciprocal matching,
  normal diversity, motion, cycle consistency, and model-agreement checks remain.
- **Feedback:** the client negotiates `xyzrgb-f32le`. Each point carries the same
  six float32 XYZ/RGB values in bounded base64 data. Decoding occurs on the
  listener thread; serialization occurs outside the engine lock and away from
  the HTTP event loop. Live delivery overlaps the next reconstruction frame and
  holds one in-flight update plus the latest pending snapshot. Old-session updates
  are invalidated on reset, and manual build/preview drain earlier live feedback.
  Single-frame decompression also runs away from the HTTP event loop.
  Legacy subscribers keep JSON lists, and the new client
  accepts legacy servers. Point counts and rendering density are preserved.
- **Diagnostics:** model refreshes called from tracking/recovery are reported
  under `model_refresh`; nested timing excludes them from the parent tracking
  total. Historical reports charged those refreshes to tracking.

## Reproduce and investigate further

Use the scanner's existing Python environment and an extracted lossless session:

```bash
OMP_NUM_THREADS=4 KINECT_BLOCK_COUNT=5000 python scripts/profile_capture.py \
  --path /absolute/path/to/session \
  --output benchmark-output/capture-profile.json
```

The default models local capture: packed feedback and zlib level 0. To measure
compressed capture or legacy feedback separately, use `--compression-level 1`
and/or `--feedback json`. These switches change transport costs without changing
reconstruction settings. `--compare previous-report.json` checks accepted indices
and pose differences. `--limit 14` isolates this recording's continuously tracked
prefix. `--profile` additionally writes a `.prof` file for `pstats`; profiler
instrumentation should not be mixed into ordinary wall-time comparisons.

Recovery ICP is the largest remaining measured cost on this session. Further
changes to its sampling or iteration policy need nearby-return and ambiguous
scene measurements before adoption. Registration normals/model pyramids also
cost time, and uploads still wait for the active native frame under the engine
lock. Faster packing cannot remove that wait. A camera-to-feedback measurement
on hardware is needed to quantify the complete live delay.

High-resolution RGB remains 10 fps; the existing VGA mode offers 30 fps with a
separate calibration. These improvements do not change that sensor cadence,
disable confidence fusion, reduce geometry resolution, or repair missing overlap.
The feedback cloud still uses the cached model refresh schedule. CUDA performance
and physical Kinect throughput have not been measured for this change.

## CPU parallelism

Acquisition already runs in a separate camera process, supervised by a Qt worker.
Upload/recording work uses a Qt worker, WebSocket receiving/decoding uses another
thread, and reconstruction runs in a native worker away from the server's HTTP
event loop. Feedback encoding/delivery now overlaps reconstruction rather than
making the frame processor wait for each broadcast. Slow feedback can replace
intermediate display snapshots; it never drops captured reconstruction frames.

Whole-frame reconstruction remains ordered. Tracking uses the previous accepted
pose, the latest fused model, and recovery history. Fusion updates overlapping
voxel attributes in one shared volume. Running independent complete frames
against that mutable state would change acceptance decisions or race updates.

The strongest remaining parallelism candidates are:

1. A bounded lookahead of one or two frames for RGB-D calibration/filtering and
   registration-cloud preparation, while the current frame tracks and integrates.
   These inputs can be prepared independently using immutable session settings;
   results must be consumed in order and invalidated on reset. Native point-cloud
   construction/normal-estimation calls need GIL/CPU-budget measurements first.
2. Confidence fusion over disjoint voxel-block batches, after serial block
   activation. Read-only observation arrays can be shared; each worker must own
   non-overlapping attribute indices. This requires numerical parity, bounded
   scratch memory, and evidence that memory bandwidth does not erase the gain.
3. Independent offline feature/matching work for final reconstruction, retaining
   ordered pose decisions and a single owner of the mutable volume.

Open3D operations already use native parallelism. Its Open3D 0.20 legacy ICP
[Python binding](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/pybind/pipelines/registration/registration.cpp)
explicitly releases the GIL, so another Python thread can perform native work
while ICP runs. Ordinary Python bytecode is subject to the
[CPython GIL](https://docs.python.org/3/library/threading.html#gil-and-performance-considerations).
More outer threads do not automatically mean more effective CPU cores.

On the test Mac, there are eight performance and two efficiency cores. The
launcher defaults to four OpenMP threads, while `python -m scanner_server`
defaults to fourteen; both respect an explicit `OMP_NUM_THREADS`. OpenCV reports
ten threads in this environment. Nested pools should share a deliberate CPU
budget. Replays with 1/4/8/14 OpenMP threads were performed, but substantial
concurrent host CPU load made those measurements unsuitable for selecting a
default. No thread-count change or parallel-frame speedup is claimed. The replay
timings above measure the numerical/transport changes, not overlapped feedback
delivery; the latter is validated by blocked-encoder/slow-subscriber tests.

## Validation

Numerical tests compare CPU weighted TSDF, weight, and color attributes with the
existing tensor reference, and calibrated projection with OpenCV. Transport tests
cover VGA/high-resolution level-0 packets, metadata and batch losslessness,
packed/legacy feedback, bounded malformed-message rejection, and remote/local
compression selection. Recovery and live API tests cover cached references,
tracking freeze, cancellation/reset/build barriers, and HTTP responsiveness
while feedback encoding runs. Additional concurrency tests confirm the next frame
can finish with feedback encoding blocked, pending snapshots coalesce, terminal
messages remain ordered, and reset/shutdown invalidate pending delivery.

After integration with the current local `master`, full verification:
**257 tests passed, 2 CUDA tests skipped**. The synthetic
HTTP/WebSocket/Qt scan workflow passed, including live fused feedback, final mesh
builds, exports, and lossless session download. Changed-file lint passed; the
engine retains its pre-existing broad-exception lint findings.
