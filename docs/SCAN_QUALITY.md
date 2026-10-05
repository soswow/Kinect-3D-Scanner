# Scan quality: changes and reproducible evaluation

The active client uses `scanner_server/engine.py`. The older local engine is
retained as reference code and is not used by the GUI.

See [IMPLEMENTATION_MILESTONES.md](IMPLEMENTATION_MILESTONES.md) for the newer
ordered work: calibration tools, profiling, distant appearance matching,
confidence fusion, texture correction and separate live/final budgets. The
results below describe the preceding tracking baseline.

## What changed

- **Consistent camera model.** Live Kinect scans use the complete measured
  [calibration default](../calibration/default.json): native depth intrinsics,
  IR/depth grid correspondence, lens distortion, metric conversion and RGB pose.
  RGB defaults to 1280×1024 at 10 fps. The GUI requires the complete JSON format;
  see the [calibration guide](../calibration/README.md). Public dataset replays
  supply their own explicit camera model and metric depth.
- **Real depth bounds and cropping.** Near/far settings now mask both tracking
  and fusion. The optional central crop removes the outer image region from both.
  Settings are frozen during capture. Keep enough distinct geometry inside the
  crop for tracking; a crop is not automatic object segmentation.
- **Conservative filtering.** Isolated depth pixels need two nearby neighbours
  at similar depth. Missing depth is not filled and sharp boundaries are retained.
- **Tracking checks.** ICP runs at 30/15/7.5 mm registration scales with a Huber
  loss (scales increase for larger voxels). Minimum overlap, maximum residual,
  translation/rotation limits, and matched-normal diversity guard fusion.
  Weak tracking refreshes recent geometry and tries bounded motion prediction. FGR recovery
  must pass the same checks and uses at most 5,000 coarse feature points per
  cloud. Ambiguous or poorly aligned frames are skipped.
  Geometry on a single flat surface still cannot constrain all camera motion.
- **Optional color assistance.** The experimental checkbox runs hybrid RGB-D
  odometry before ICP, then applies all geometric acceptance checks. Weak texture
  or known pairing lag above 20 ms disables that candidate. It improves the tested
  desk sequence but slightly worsens XYZ, so it is off by default. It does not
  bypass flat-surface ambiguity. CLI replay enables it with `--color-recovery`.
- **Correct TSDF truncation.** The configured distance now reaches both block
  allocation and integration, rather than always using Open3D's default.
- **Confidence-aware final mesh.** Preview uses weight 0.5 for early feedback;
  final extraction defaults to weight 2. Tiny components below 30 triangles,
  duplicate triangles/vertices, and degenerate triangles are removed. A final
  scan with insufficient repeated observations fails with an explanation. The
  largest component is not selected automatically and holes are not filled.
- **Reliable capture ordering.** Batches include only consecutive captures;
  reset/preview/build barriers retain their order. Engine mutations and exports
  share a server lock, which remains held until native work finishes even if
  an HTTP request is cancelled. WebSocket disconnects cannot interrupt a build. Capture waits for reset confirmation and pauses for preview.
- **Fresh, timed pairs.** Camera timestamps are retained with wraparound handling.
  Pairs separated by more than 50 ms and duplicate capture IDs are rejected.
  The GUI uploads a camera frame only once and rejects stale cached frames.
- **Useful diagnostics.** Accepted poses, per-frame quality, rejection reasons,
  and actual integration counts are retained. `GET /api/scan/diagnostics` exposes
  them; skipped frames do not inflate the displayed integration count.
- **Bounded capture.** The client limits pending captures to 100. The server
  defaults to 500 raw frames per session (`KINECT_MAX_FRAMES`), roughly 2.3 GB
  for full-resolution RGB/native depth arrays, in addition to reconstruction memory.
- **Replayable recordings.** “Save local RGB-D recording” saves lossless PNGs and
  a manifest with the session settings under ignored `recordings/`. The option
  starts a new folder per scan and is off by default.

## Practical starting settings

Leave the subject still and move the Kinect slowly around it. Keep corners,
curved surfaces, and overlapping views visible. The current method assumes a
static scene: rotating an object in front of a stationary background is not a
supported turntable workflow. Matte, opaque surfaces are easier for structured
light than reflective, transparent, or very dark surfaces. Stay within the
sensor's usable distance; tighter clipping cannot make invalid close-range
measurements valid.

Start at 5 mm voxels, 500–2000 mm clipping for a nearby subject, auto capture
around 0.3–0.5 seconds, and final confidence 2. Capture several overlapping views
of each surface. Use the crop only when it keeps sufficient geometry visible.
Lower final confidence to 0.5 if inspecting a very short scan; expect more noise.
A finer voxel grid cannot recover detail absent from the depth measurements.

## Public datasets and normalization

[Open3D's Redwood sample](https://www.open3d.org/docs/release/python_api/open3d.data.SampleRedwoodRGBDImages.html)
contains five registered 640×480 RGB-D frames, intrinsic parameters, and a
reference trajectory. [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
provides real Kinect sequences with timestamped images and motion-capture poses.

The public dataset replay standard is uint8 RGB, uint16 depth in **millimetres**, registered
RGB pixels, and camera-to-world poses in metres. Live Kinect sessions retain
native raw disparity with the complete calibration. Redwood depth is already mm.
TUM depth values are divided by five and rounded to mm; its supplied depth
correction is not applied again. RGB/depth files are associated one-to-one within
20 ms; reference poses are matched within 20 ms. TUM's recommended approximate
ROS intrinsics are used without independently undistorting the registered depth,
following its [file-format guidance](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
Reference poses are used only by the scorer, never to initialize tracking.

Downloaded source data, converted recordings, meshes, logs, and full diagnostic
reports remain ignored. Only aggregate results and tools are included in Git.

```bash
python scripts/download_dataset.py redwood
python scripts/download_dataset.py tum-xyz
python scripts/download_dataset.py tum-desk

OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset redwood
OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset tum \
  --path datasets/rgbd_dataset_freiburg1_xyz --stride 10 \
  --output benchmark-output/tum-improved.json

# Convert a public sample into the same standard as locally recorded scans.
python scripts/convert_dataset.py --dataset redwood \
  --output datasets/converted-redwood
python scripts/replay_scan.py --dataset recording --path datasets/converted-redwood

# Exercise actual network transport (this resets the selected server's session).
python scripts/replay_scan.py --dataset recording --path datasets/converted-redwood \
  --server http://127.0.0.1:8000
```

Dataset download requires a Python version with `tarfile`'s safe `data` extraction
filter (tested with Python 3.13). To reproduce the historical comparison:

```bash
mkdir -p benchmark-output
git show 609e64c:scanner_server/engine.py > benchmark-output/baseline_engine.py
python scripts/replay_scan.py --dataset tum \
  --path datasets/rgbd_dataset_freiburg1_xyz --stride 10 \
  --baseline-source benchmark-output/baseline_engine.py \
  --output benchmark-output/tum-baseline.json
```

The baseline helper overrides the historical globals with the **same** camera
parameters as the new engine. `--legacy-intrinsics` instead reproduces its old
IR constants. Use the same sampled frames and report acceptance alongside error;
rejecting difficult frames alone can improve the reported error.

## Measurements

See [aggregate benchmark results](benchmarks/rgbd-summary.json). The scoring
anchors both trajectories to the first evaluated camera, with no scale fitting
or trajectory-wide alignment. Translation RMSE measures relative camera-position
error in this fixed frame; rotation RMSE measures relative orientation error.
It is not the usual rigid-aligned TUM ATE, and it is not a direct surface-accuracy
measurement. Runs can vary slightly because of registration and voxel hashing.

| Sequence | Historical engine, same camera | Improved engine | Accepted frames |
|---|---:|---:|---:|
| Redwood, five frames: translation RMSE | 1.97 mm | 1.36 mm | 5/5 both |
| Redwood: rotation RMSE | 0.054° | 0.064° | 5/5 both |
| TUM fr1/xyz, every tenth associated pair: translation RMSE | 219.8 mm | 43.6 mm | 80/80 both |
| TUM fr1/xyz: rotation RMSE | 5.39° | 1.94° | same |
| TUM fr1/desk: translation RMSE | 1542 mm | 631 mm | 58/58 baseline; 52/58 improved |
| TUM fr1/desk: rotation RMSE | 40.13° | 35.62° | same |
| TUM fr1/desk with optional color: translation RMSE | 1542 mm | 325 mm | 58/58 baseline; 57/58 color |
| TUM fr1/desk with optional color: rotation RMSE | 40.13° | 21.95° | same |

The XYZ run covers the full approximately 30-second sequence. Its final mesh
has about 1.27 million triangles versus 4.06 million for the baseline. This lower
count reflects tracking/filtering/confidence changes and is not independently
proof of better geometry. On Apple Silicon with four OpenMP threads and 5,000
initial TSDF blocks, the current XYZ replay took about 147 seconds versus 47
seconds for the baseline. The desk replay took about 102 seconds, or 212 seconds
with color assistance. Timings are indicative: some checks ran concurrently.
Caching model scales and computing FPFH only on recovery reduced an earlier,
stricter configuration from about 162 to 86 seconds with comparable pose errors.

The desk sequence includes much stronger camera rotation and scene changes.
Remaining errors of 0.3–0.6 m and 22–36 degrees are still poor; tracking coverage
and lower error do not make it a solved reconstruction problem. Geometry guards
can reject local failures but cannot eliminate cumulative drift or all wrong
matches. Optional color slightly worsened XYZ (48.1 mm, 2.25 degrees; all 80
frames accepted), which is why it is not the default.

These results demonstrate improvement on the tested recordings, not calibrated
accuracy for an individual Kinect or arbitrary subjects. The Redwood rotation
metric slightly regressed. Physical capture timing, per-device calibration, and
your subjects still require a hardware comparison.

## Verification

```bash
OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 python scripts/check_scanner.py --public-data
```

Tests raycast an asymmetric static scene from known moving camera poses, add
noise and holes, then check trajectory error, rejected empty frames, and mesh
construction. Other checks cover clipping/ROI, TSDF truncation, calibration
validation, planar ambiguity, pose jumps, protocol bounds, timestamp rollover,
duplicates, command barriers, concurrent/cancelled API mutation, bounded feature
matching, motion/color recovery, failed integration, portable recordings, TUM
unit conversion, timestamp association, and scoring conventions. The offscreen
GUI check uses synthetic camera input; the network check also replays Redwood.

## Next improvements, in order

1. **Device-specific calibration and live comparison.** Measure RGB intrinsics
   in the registered image space. Compare the same subject and motion before/
   after with local recording enabled. Inspect rejected frame messages and USB
   pairing lag. This is the first step that requires access to the actual device.
2. **Pose graph and reintegration.** The harder desk sequence shows why this is
   needed. Add reliable loop-closure candidates, confidence checks, trajectory
   optimization, and reconstruction from corrected poses. Measure loop drift and
   surface consistency on longer recordings. Retain the original trajectory when
   refinement cannot find reliable constraints.
3. **Color-assisted tracking validation.** The opt-in RGB-D/ICP prototype is
   implemented and tested against synthetic motion and public recordings. Extend
   tests for weak texture, blur, exposure changes, and timing before considering a
   default change. A successful optimizer alone must not authorize fusion. Flat
   textured scenes need explicit observability checks before relaxing plane rejection.
4. **Foreground/turntable mode.** Separate tracking and reconstruction masks and
   make stationary-camera/rotating-subject capture explicit. Background removal
   alone is insufficient if the tracker estimates the background's stationary pose.
5. **Surface accuracy evaluation and adaptive capture.** Compare meshes against
   a reference reconstruction, report completeness alongside error, and use motion/
   coverage to pick useful frames. Then tune noise filtering and voxel presets.

Global refinement is implemented experimentally; turntable capture remains planned. The color-assisted
prototype stays opt-in because the tested sequences show a tradeoff rather than
a uniform improvement. No live device measurements were performed in this batch.
