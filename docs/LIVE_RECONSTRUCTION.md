# Live reconstruction, compute backends, textures, and final refinement

Current use: new UI scans start with live reconstruction **off** and experimental
depth/color/motion Finish registration selected. Saved preferences and projects
retain their settings. See [final registration](FRAGMENT_RECONNECTION.md) and
[offline buffering](OFFLINE_CAPTURE_UPLOADS.md). Measurements and test totals
below describe their historical checkpoints.

These additions build on the tracking and recording changes in
[SCAN_QUALITY.md](SCAN_QUALITY.md). They add working CPU paths and an explicitly
selected CUDA path. NVIDIA hardware was unavailable at that initial checkpoint;
later [CUDA studies](CUDA_EXPERIMENTS.md) and [field results](FIELD_CUDA_RESEARCH.md)
provide hardware measurements. The
subsequent capture/shutdown repair was checked with a connected Kinect v1:
258 RGB/registered-depth pairs in 9.3 seconds, at most 23.6 ms pairing offset,
and Qt window shutdown in 0.32 seconds. This validates acquisition and closure,
not the reconstruction accuracy of that physical scene.

The next six software milestones are now implemented. See
[development history](DEVELOPMENT_HISTORY.md) for completed checkpoints and the
[script index](../scripts/README.md) for commands. Earlier benchmark results below
remain historical evidence rather than predictions for a new device.

## Compute selection

Set environment variables on the **server**, independently of the Kinect client:

| Variable | Values | Behavior |
|---|---|---|
| `KINECT_DEVICE` | `auto` (default), `cpu`, `cuda` | Selects the VoxelBlockGrid device |
| `KINECT_TRACKING` | `auto` (default), `legacy`, `tensor` | Auto chooses tensor ICP on CUDA, legacy ICP on CPU |
| `KINECT_BLOCK_COUNT` | Integer | Initial voxel block allocation; start with 5000 |

An explicit CUDA request fails at startup if Open3D cannot use CUDA. Auto reports
its fallback reason. `/api/health`, `/api/scan/status`, diagnostics, and the GUI
show the actual backend. `KINECT_TRACKING=tensor KINECT_DEVICE=cpu` exercises the
tensor registration code on a machine without CUDA.

TSDF integration and extraction run on the selected device. Tensor geometric ICP
uses device point clouds and a robust Huber loss, then passes the same overlap,
residual, pose-jump, and normal-diversity gates as legacy tracking. Depth filtering,
registration-cloud construction, model downsampling/normals, feature recovery,
RGB-D color recovery, pose-graph optimization, UV generation, and image projection
currently run on CPU. This is a hybrid pipeline, not an entirely GPU-resident one.

CUDA needs an NVIDIA driver and a compatible Open3D build. Apple Silicon runs the
CPU path; these changes do not add Metal acceleration. Do not use the macOS lock
file as a Linux/CUDA environment specification. Run the CUDA test on the target
server before claiming hardware support or throughput:

```bash
KINECT_DEVICE=cuda KINECT_TRACKING=tensor KINECT_BLOCK_COUNT=5000 \
  OMP_NUM_THREADS=4 python -m unittest tests.test_features.FeatureTests.test_cuda_tracking_and_fusion -v
```

Frame reports include wall times for filtering, cloud creation, tracking, fusion,
and model refresh. CUDA timings synchronize at stage boundaries, which adds
profiling overhead. Model refreshes requested inside tracking or recovery now
have their own `model_refresh` timing; the parent tracking stage excludes that
time so totals do not count it twice. Final refinement has a separate total. On one CPU tensor XYZ
replay, fusion took 1.7 s, tracking 40.8 s, and scheduled model refreshes 50.2 s.
Accelerating fusion alone would therefore leave most of that run's cost intact.

## Continuous feedback

Enable **Show live reconstruction** for live fused feedback. API clients use
`{"live_reconstruction": true}` in reset settings; omission retains the previous
on-demand processing workflow.

Uploads store raw frames. A background task processes one frame while holding the
engine lock, then releases it before the next frame. Reset, build, preview, and
export share that lock. Build waits for an active live frame instead of reporting
that a build is already running. Cancellation waits for native processing to
finish before releasing the lock. Client frame batching preserves command
barriers and uses batches of at most eight in live mode.

WebSocket `live` messages contain at most 30,000 sampled fused points with colors,
camera-to-world pose, camera calibration, session ID, counts, backend, and the latest tracking result.
Messages are limited to approximately two per second plus the final update of a
drained queue. Slow sockets time out independently. The persistent Qt view uses
software rendering and follows the latest accepted scanner pose by default,
from 50 cm behind the scanner along its viewing axis. The view keeps the scanner's
orientation and uses perspective projection with the session's calibrated depth
intrinsics, fitting the viewport without stretching or changing its field of view.
Skipped tracking frames hold the last accepted viewpoint. Select **Orbit**
to drag to orbit and wheel to zoom; select **Follow** to resume following. **Fit View**
frames the whole cloud. Captured colors are always used when available.
The view ignores snapshots from previous sessions
and needs no OpenGL context or screen capture.

This is responsive reconstruction feedback, not a guaranteed camera-rate mesh.

The desktop client requests `?geometry=xyzrgb-f32le` on the progress WebSocket.
Those live messages replace the `points`/`colors` lists with a `geometry` object:
`encoding: "xyzrgb-f32le"`, `count`, and base64 `data`. Each point contains six
little-endian float32 values, XYZ followed by RGB in 0–1. Counts and payload sizes
are bounded to 30,000 points; decoding happens on the listener thread. Clients
without that query parameter receive the existing JSON lists, and the new client
also accepts older servers. Serialization runs outside the engine lock and away
from the HTTP event loop. A separate delivery worker lets reconstruction continue
while feedback is encoded and sent. It keeps one update in flight and only the
newest pending snapshot, preventing slow viewers from creating a stale backlog.
Reset invalidates queued/encoded updates from the previous session; build and
preview drain earlier live updates before their progress/final messages.
See [capture performance](CAPTURE_PERFORMANCE.md) for
measurements, remaining costs, and the recording profiler.
Geometry is sampled from the fused cloud before tracking downsamples it, using
the existing model extraction, refreshed every three accepted integrations.
The client draws 3×3 dots with depth ordering; **Inspect Scan** opens the mesh.
`geometry_frame_count` makes that lag visible. Processing can fall
behind capture; the view shows pending frames, server processing time, and time
since its last update when hovering over the reconstruction counts. These values
are not an end-to-end latency measurement.
**Inspect Scan** requests a full snapshot and temporarily suspends capture.
**Finish Scan** opens the final exported mesh after the build succeeds. If live
feedback disconnects, capture waits and guidance asks the operator to pause movement.

## Portable texture exports

Plain OBJ already contains vertex colors through a widely used extension; some
readers ignore those colors. It does not contain UV coordinates or an MTL/texture.
PLY is a reliable vertex-color export. The new formats are:

- `/api/scan/export/glb`: a mesh with UVs and embedded texture/material.
- `/api/scan/export/obj.zip`: OBJ, MTL, PNG, and `texture-report.json`.

Both accept `size=256|512|1024|2048`, `max_triangles=100..200000`,
`max_views=1..64`, and `use_images=true|false`. Defaults are 1024, 50000, 24, and
true. Optional `exposure_correction=true` enables bounded gain matching on
held-out depth-visible overlap; `blend_mode=best` selects the strongest
angle/distance view instead of blending. Both have GUI export checkboxes. The texture mesh is a simplified copy;
the full final mesh remains available through PLY/plain OBJ. UV seams duplicate
export vertices deliberately. Non-manifold edge connections and disconnected
vertex fans are separated into UV seams on the export copy before atlas
generation. This preserves triangle positions and colors; the topology repair
counts are included in `texture-report.json`.
Contradictory winding is also cut into separate sheets. If native atlas
generation still rejects a partition, independently repaired charts are packed
into texture tiles, subdividing only rejected charts. The report records the
atlas method, chart count and subdivisions; this fallback preserves every face
of the simplified export mesh.
Connections that cannot be separated return an actionable error.

The portable projection algorithm works on ARM CPUs:

1. Select synchronized accepted views across the trajectory, favoring sharper
   images in each interval. Known pairing offsets above 20 ms are excluded;
   legacy captures without offsets retain the assumption of registered RGB-D.
2. Generate a partitioned Open3D UV atlas and bake position, normal, and fused
   color maps. Partitioning avoids expensive single-chart solves on scan meshes.
3. Project surface texels into original RGB views using calibrated intrinsics and
   inverse camera-to-world poses. Require measured depth to agree with surface
   depth within `max(15 mm, 1% of depth)` and exclude grazing views.
4. Bilinearly sample RGB and blend with view-angle/distance weights. Missing
   texels retain fused vertex color. No generated image fills unobserved areas.

The pipeline supports measured lens correction and optional relative exposure
gains. It does not optimize patch seams or photometric poses. Inaccurate calibration/poses can blur
texture, and a larger atlas cannot create detail absent from 640x480 RGB. The
depth tolerance is an engineering default, not a device-specific noise model.
GLB/OBJ material round trips and asymmetric image orientation are tested. The
viewer preserves texture materials when loading GLB or a textured OBJ extracted
from its ZIP.

## Full session preservation

**Save full RGB-D session**, or `/api/scan/export/session`, returns a ZIP with
lossless RGB PNGs, uint16 millimetre depth PNGs, `manifest.json`, and
`reconstruction.json`. The report stores settings/calibration, backend, accepted
camera-to-world poses in metres, original poses when refinement applied,
per-frame tracking diagnostics, timings, and refinement outcome. It deliberately
keeps estimated poses separate from evaluation reference poses.

Unzip and replay using:

```bash
python scripts/replay_scan.py --dataset recording --path recordings/unzipped-session
```

Optional client recordings also save a reconstruction sidecar after preview or
final build. Per-frame batch acknowledgements identify accepted stored frames
and map their server indices into recordings. Older servers without individual
acknowledgements preserve captures but cannot establish partial-batch mappings.
Datasets, captures, exports, and detailed replay artifacts remain ignored by Git.

## Experimental final refinement

Enable **Final pose refinement**, reset with `refine_poses: true`, or pass
`--refine-poses` to replay. It remains off by default. The current implementation:

- Uses at most 64 accepted keyframes and 40 loop candidates, up to half proposed
  by mutual ORB/measured-depth PnP outside the estimated position radius, and a
  fixed first-camera anchor.
- Requires reciprocal robust ICP, sufficient overlap, low residual, normal
  diversity, inverse consistency, and bounded corrections for loop constraints.
- Optimizes an Open3D pose graph and checks separate point samples with a
  saturated distance loss. It requires at least 2% aggregate improvement and
  rejects inconsistent individual constraints.
- Interpolates bounded keyframe corrections into all accepted poses and fuses
  raw frames into a **fresh** TSDF. The old volume/poses remain intact if native
  reintegration fails. Unchanged sessions do not repeat refinement.

For current sampling, protected-boundary and all-output validation rules, see
[camera loop refinement](POSE_REFINEMENT.md). Experimental depth registration
skips this optional legacy stage.

Appearance proposals can discover distant loops, but many scenes lack enough
trustworthy RGB/depth correspondences. It cannot recover already discarded frames
or reliably disambiguate repeated geometry. The optional online recovery checkbox
can accept later views after two rejected frames through reciprocal verification. Held-out samples reduce fitting bias but are not external ground truth.
Two volumes coexist during reintegration; allow approximately twice the voxel
attribute allocation plus raw frames, cached clouds, and extraction overhead.

## Evidence and reproducibility

Validation used Apple Silicon, Python 3.13, Open3D 0.20, four OpenMP threads, and
5000 initial blocks. Aggregate results are in
[feature-summary.json](benchmarks/feature-summary.json).

| Check | Result | Interpretation |
|---|---|---|
| Public TUM XYZ, 80 strided frames, tensor ICP on CPU | 80/80 accepted; 43.8 mm camera-position RMSE | Similar to preceding legacy result of 43.6 mm; not a CUDA benchmark |
| Public Redwood through HTTP | 5/5 accepted; about 1.36 mm camera-position RMSE | Client/server reconstruction still works |
| Synthetic loop, injected trajectory drift | Detail-surface RMSE 14.9 mm → 3.3 mm | Fresh fusion improves this controlled geometry; not measured Kinect accuracy |
| Harder public desk run with refinement requested | 51/58 accepted; 785 mm camera-position RMSE; no trusted loop | Poses retained; global drift remains unresolved |
| UV orientation, occlusion, GLB and OBJ material reload | Passed | Portable measured color and materials survive export |
| HTTP/Qt check | Live view, stale-session rejection, preview/build, textures, session ZIP passed | Acquisition is replaced with synthetic input |

Public scores use a fixed first-camera anchor with no scale fit or trajectory-wide
alignment. They are not standard aligned TUM ATE and do not measure mesh accuracy.
The detail-surface synthetic score compares vertices below 2 m against exact
fixture surfaces, excluding the large back wall. Tracking can vary on difficult
sequences because voxel ordering and recovery can select different alignments;
single-run gains must not be generalized. In a seeded paired run, the preceding
source accepted 57/58 frames at 116 mm while the new source accepted 53/58 at
544 mm. Two instances of the **same preceding source**, with the same seed
schedule, also diverged: 55/58 at 967 mm versus 51/58 at 1067 mm. This establishes
existing instability but does not establish statistical non-regression of the
new code. These experiments remain opt-in pending broader tracking and hardware
validation. The changes have been merged; these historical timings include concurrent experiments.

```bash
OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 python scripts/check_scanner.py --public-data
OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset tum \
  --path datasets/rgbd_dataset_freiburg1_xyz --stride 10 --tracking tensor \
  --output benchmark-output/xyz-tensor.json
OMP_NUM_THREADS=4 python scripts/replay_scan.py --dataset tum \
  --path datasets/rgbd_dataset_freiburg1_desk --stride 10 --color-recovery --refine-poses \
  --output benchmark-output/desk-refinement.json
```

The corresponding software tools are implemented; physical calibration, matched
live scans and broader scene comparisons remain evidence requirements. Later
NVIDIA profiling is linked above.
Patch seam leveling and turntable capture require subsequent work. See the
research roadmap and milestone log for current status.


## Separate live and final budgets

**Final voxel size** defaults to the live volume. Set 2 mm up to the live voxel
size to request fresh final fusion of accepted views at their final poses. Live
tracking, cached feedback and raw observations keep their original resolution.
The final build has a hard **Final block budget** (default 5000); it checks newly
required blocks before each integration and stops with a coarser-voxel/budget
suggestion if the limit would be exceeded. It does not silently lower quality.

5000 blocks allocate about 391 MiB of voxel attributes, in addition to the live
volume, raw frames, hash maps, temporary tensors and mesh extraction. This is a
block limit, not a process-RAM or VRAM cap. Finer voxels can exhaust it quickly.
A failed final build preserves the live volume and any existing final asset.
An unchanged successful session reuses its final volume; new frames invalidate
it. Build/session reports record final voxel, block count and elapsed time.

Live feedback shows integrated/skipped frames, pending count, server queue age,
and guidance for weak depth, lost tracking or backlog. Queue age uses only the
server's monotonic clock and is not end-to-end capture latency. Automatic capture
treats **Minimum capture interval** as the fastest permitted cadence. It learns
from the mean processing time of the last twelve frames (including model refresh
and skipped frames) and local capture-to-feedback latency, adds 15% headroom, and
rounds the resulting delay up to a whole camera frame. Slowdowns adjust promptly;
speed increases gradually. Captures never occur closer together than the user's
minimum interval.

The client also limits automatic live capture to two outstanding frames: one
processing and one waiting. Captures count from queue submission through live
processing, so delayed uploads or HTTP acknowledgements cannot conceal a backlog.
The adjusted pace appears below the interval. Feedback disconnects suspend live
automatic capture; recovery checks still wait for the previous check to finish.
Only fresh camera frames are selected, with no catch-up bursts. Scans without
live reconstruction keep the chosen cadence and wait at five queued uploads.
Manual capture remains available. It does not estimate whole-object coverage
or choose keyframes by geometric information gain.
