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

- `/api/scan/export/glb`: a mesh with UVs and embedded textures/materials.
- `/api/scan/export/obj.zip`: OBJ, MTL, texture PNGs, and `texture-report.json`.

Both formats export **one scan object with multiple material assignments**.
Texture pages are internal baking groups. OBJ switches materials with `usemtl`
within one `o scan`; GLB stores one mesh/node with a primitive for each page
material. Importing in Blender should create one mesh object, with a material
slot for each texture page.
The optional Blender import regression check runs in a separate background
instance with factory settings; set `BLENDER_EXECUTABLE` to its executable path
and run `python -m unittest tests.test_texture_pages`. It checks object count,
faces and all texture material slots using Blender's default import options.

Both accept `size=256|512|1024|2048`, `max_triangles=100..200000`,
`max_views=1..64`, and `use_images=true|false`. Defaults are 1024 **per page**,
200000 triangles, 64 views, and true. The export dialog's **Triangle budget**
persists this choice and sends it as `max_triangles`. Choose a smaller budget
for less export work and smaller geometry files. This limit simplifies an export
copy; it does not limit the reconstruction or change the saved project.
Texture exports now add spatially grouped
pages as mesh surface area grows. `texels_per_m=1..1024` (default 512) sets a
planning target; `max_pages=1..32` (default 16) bounds image storage and work.
`max_pages=1` requests the previous single-atlas layout. A 55% packing allowance
guides page allocation; actual chart occupancy, source detail and the page cap
can reduce attained density. The report includes page count, surface area,
mapped/projected texel counts, occupancy and `page_budget_limited`. Bake working
maps are processed one page at a time and selected RGB/depth observations are
prepared once. Pages skip views only when their entire bounding box lies outside
a depth-camera frustum; sparse visibility sampling does not discard source views.
More pages and views increase projection work and file size;
these defaults do not promise faster exports.

Default `blend_mode=blend` matches low-frequency photo colors across the whole
surface before atlas cuts. It keeps the selected original photo's sharp detail.
`blend_mode=best` keeps original photo colors and sharp boundaries.
Optional `exposure_correction=true` additionally enables bounded per-camera
gain matching on held-out depth-visible overlap. Both options have GUI export
checkboxes. The texture mesh is a simplified copy;
the full final mesh remains available through PLY/plain OBJ. UV seams duplicate
export vertices deliberately. Non-manifold edge connections and disconnected
vertex fans are separated into UV seams on the export copy before atlas
generation. This preserves triangle positions and colors; the topology repair
counts are included in `texture-report.json`.
Contradictory winding is also cut into separate sheets. If native atlas
generation still rejects a partition, independently repaired charts are packed
into texture tiles, subdividing only rejected charts. Rectangular sections in a
texture are these UV chart groups, not snapshots or independent source photos.
Fallback packing assigns space in proportion to surface area instead of giving
every repaired chart the same small square. Charts share pages; each chart does
not automatically need its own 1024-pixel image. The report records the
atlas method, chart count and subdivisions; this fallback preserves every face
of the simplified export mesh.
Connections that cannot be separated return an actionable error.

The portable projection algorithm works on ARM CPUs:

1. Select synchronized accepted views across the trajectory, favoring sharper
   images in each interval. Known pairing offsets above 20 ms are excluded;
   legacy captures without offsets retain the assumption of registered RGB-D.
   Only accepted poses in the selected reconstruction can supply texture;
   captures in disconnected maps cannot fill holes without verified registration.
2. Select coherent source-photo regions on the whole simplified mesh before
   page/UV cuts. Four interior samples per face supply depth-tested
   angle/distance scores; physical face adjacency favors connected labels.
   A separate scoring copy receives winding/manifold repair so signed adjacency
   does not treat a reversed coplanar face as a sharp fold. Face IDs and positions
   stay unchanged; the report records that repair.
   Every page receives the same global labels, remapped to its unculled views.
   This makes source selection independent of page boundaries. It does not
   optimize photographic alignment.
3. For `blend_mode=blend`, project low-pass native RGB onto physical mesh
   vertices with the same measured-depth visibility checks. Normal-aware short
   mesh edges give observations a shared surface footprint; robust, capped
   geometric weights estimate common low-frequency color. Fit bounded RGB
   log gains (0.5–2), shrinking near-black channel gains towards luminance gain
   to limit sensor-noise amplification. Incident normal axes resist winding
   cancellation; consistently oriented bisectors handle sharp mesh corners.
   The original photographs are never blurred for the final texture.
   Interior surface points withheld from vertex fitting verify relative
   low-frequency continuity. Unsupported regions keep original colors;
   regressing disconnected components are rejected. These checks compare
   photographs at saved poses, not reflectance or geometric ground truth.
   The fit retains at most 200,000 original vertices and 64 photos, chunks
   fitting temporaries, and blurs one image at a time. Several numerical buffers
   coexist; input photos and atlas baking need additional memory. The report
   records fit size/time, validation and rejection counts.
4. Generate a partitioned Open3D UV atlas and bake position, normal, and fused
   color maps on each page. Partitioning avoids expensive single-chart solves on scan meshes.
5. Project surface texels into original RGB views using calibrated intrinsics and
   inverse camera-to-world poses. Require measured depth to agree with surface
   depth within `max(15 mm, 1% of depth)` and exclude grazing views.
6. Bilinearly sample the chosen region photo. Where that photo has missing
   depth, retain the strongest independently depth-tested source. Apply the
   shared physical gain field to the actual chosen photo, including these
   fallback observations. Face and view remapping preserve the field across
   UV winding changes and page cuts. If no supported color field is available,
   the existing narrow compatible seam feather remains available. Missing
   observations retain fused vertex color. No generated image fills them.

The pipeline supports measured lens correction and optional relative exposure
gains and shared spatial color matching. It does not optimize photometric poses. Inaccurate calibration/poses can blur
texture, and more atlas pixels cannot create detail absent from the source RGB
(640x480 or native 1280x1024). A 50,000-triangle export of a large scan also loses
geometric detail independently of texture resolution. The new 200,000 default
preserves more of the reconstructed surface; use full PLY/plain OBJ to retain
all final triangles. Final voxel size, measured depth, fusion, meshing and
component filtering determine the full reconstruction's geometry. The
depth tolerance is an engineering default, not a device-specific noise model.
GLB/OBJ material round trips and asymmetric image orientation are tested. The
viewer preserves multiple texture materials when loading GLB or a textured OBJ extracted
from its ZIP. Bright clipped highlights, motion blur, inconsistent lighting,
occlusion and missing depth remain input limitations. Brightness correction uses
one shared solution across pages and still refuses inconsistent held-out overlap.

### Geometry, image alignment and seam limits

More triangles reduce simplification error; they cannot repair misregistered
photos or reconstructed geometry. A straight line slanted against an imported
coordinate grid can also reflect the chosen orientation; local bends and steps
across photo regions require separate investigation. Gravity does not constrain
the room's horizontal heading. Compare the full saved mesh, exports at matched
budgets, and native-image reprojection before changing poses or calibration.

In the recorded half-room investigation at source `035b3a4`, the full mesh had
2,732,087 triangles. With 65,536 seeded area-weighted full-surface samples,
nearest-triangle distance to 50K/200K exports had 95th percentiles 2.58/1.17 mm.
These measure export simplification relative to the reconstructed mesh, not
accuracy against the real room. A bounded 24-view floor-tag check found
leave-one-view-out native RGB reprojection disagreement of 4.10 px median and
8.42 px at the 95th percentile, compared with 0.86/2.12 px for same-view
projection. This supports cross-view inconsistency without isolating pose,
depth, calibration or timing error. Selected door views had no shared tags.
The existing pose-graph proposal improved average held-out loss by 2.31% but
worsened four camera pairs and was rejected; sparse RGB-D bundle adjustment
had no valid three-view landmarks. These rejected poses were not applied.

The shared color field works across distant charts/pages. The fallback narrow
seam feather still only sees neighboring atlas texels on physically linked
faces. Color matching cannot recover clipped detail, remove every glossy
reflection or repair shifted photo edges. Some source grain is amplified even
with dark-channel damping. Two
bounded offline color-leveling prototypes (per-view affine fields and physical
patch constants) were also rejected on that recording; neither is a production
feature or a demonstrated seam fix. Raw measurements, prototypes and captures
stay in ignored local investigation storage.

The production field separates smooth color differences from selected-photo
detail. [Let There Be Color! and the authors' implementation](https://github.com/nmoehrle/mvs-texturing)
use global mesh-patch color leveling followed by local seam correction.
[Color Map Optimization](https://vladlen.info/publications/color-map-optimization-for-3d-reconstruction-with-consumer-depth-cameras/)
jointly optimizes camera poses and image correction functions to address color
mapping errors. These methods motivate physical seam matching, not a
claim that this exporter implements them. Use overlapping local camera windows,
fixed measured calibration, independent depth validation, and matched renders;
photometric consistency alone does not establish geometric accuracy.

On the recorded half-room, matched 200K-face/16-page exports with identical
geometry, UVs, source labels and cameras reduced the 95th percentile RGB gap
across coplanar photo boundaries from 65.2 to 50.6/255 on the floor and 74.4 to
56.4/255 on the wall. Independent low-frequency photo disagreement at 1,544
interior points fell about 85%; this is not an 85% improvement in overall scan
quality. Printed-marker high-frequency correlations remained 0.995–0.999.
Selected upper/middle wood patches had about 14%/10% more normalized fine
variation, so noise removal is not claimed. The strongest wall transitions and
large floor brightness blocks visibly improve; reflections and alignment
artifacts remain. These are one-recording results, not universal accuracy or
throughput guarantees. Captures, matched renders, rejected additive/pose
experiments and detailed metrics stay in ignored local investigation storage.
The comparison used Windows, Open3D 0.20.0, NumPy 2.5.3 and four OpenMP
threads; exported views used saved measured calibration and poses. Its source
base was `7716a2a`, with Git file blobs `c33e8b1` (surface color), `4cb35a3`
(texturing) and `1b6f772` (pages). The private project SHA256 is
`b00bab6d12535eb365916ebfe8328606e98b840371f4f81efab3829c66e8b0c6`;
the fixed 200K mesh SHA256 is
`53ee9bedf0fbd87f3a9f0681386b67ba788a413ad76d713783044e55435828cf`.

## Blender analysis export

Choose **Blender — animated RGB-D scene** in **Export mesh** after Finish Scan.
The client finds Blender on PATH or in common macOS/Windows install locations;
use **Browse** to choose another executable or a macOS `Blender.app`. The
selected executable and depth sampling are remembered. Blender 4.2 or newer
runs in a separate background process; it does not modify an open Blender scene.
The server supplies the finished session and the textured fused OBJ bundle.
The normal texture options and triangle budget apply to the fused mesh.

The `.blend` contains three top-level collections: **Fused mesh**, **Depth
snapshots**, and **Camera**. The fused mesh retains its texture material slots.
There is exactly one camera and one background-image slot. Frame 1 keeps that
camera visible, hides every depth snapshot and makes the background transparent.
Frames 2 onward step through accepted final camera poses, their background
photos and exactly one visible depth mesh. The fused mesh's visibility has no
animation; manually hiding it stays effective while scrubbing. Pose and
visibility switches use constant keyframes, with no Python frame handlers or
automatic script execution required when reopening the file.

Keep the sibling `scan.images-<id>/` directory beside `scan.blend`. It contains
the lossless, undistorted RGB sequence used by the camera's single background
slot; Blender image sequences are external dependencies. Ordinary material
textures are packed in the scene. Each export writes a new sequence directory
so replacing the scene cannot corrupt an existing export's images. Old image
directories can be removed when no retained `.blend` still references them.
Moving the scene and its image directory together preserves the background.

Each snapshot mesh uses only its own depth and RGB observations. Preparation
applies the saved raw-disparity calibration, nearest-neighbour depth
rectification, clipping, crop and conservative depth filter. Triangulation
leaves holes and rejects sampled cells with invalid pixels or a depth jump
larger than `max(30 mm, 2% of depth)`. RGB projection uses the measured separate
camera transform; the Blender camera also includes that fixed offset, calibrated
focal lengths and principal point. Images are undistorted with the matching K.
These meshes are partial measured surfaces, not watertight reconstructions.
Depth sampling defaults to every two pixels; choose every pixel for full depth
resolution, or wider sampling to reduce scene memory. Wider cells can omit thin
features. All accepted frames remain included. Unregistered captures have no
shared-world pose and are omitted, with their indices retained in the scene's
`scanner_export` property. Object metadata and timeline markers retain source
frame IDs, capture times and animation frame mapping. Large scans can require
substantial Blender memory even with only one snapshot visible.

For an already saved project, without a server or camera:

```bash
python scripts/export_blender_session.py --session recordings/session.zip \
  --output export/scan.blend --blender /Applications/Blender.app
```

The standalone tool textures the saved fused mesh without rerunning registration
or fusion. Supply `--textured-model export/scan.obj.zip` to reuse an existing
textured OBJ bundle, and `--snapshot-stride 1|2|4|8` to choose depth sampling.
Without `--output`, artifacts go to ignored `benchmark-output/blender-export/`.
Help loads no optional numerical packages. Failed conversions preserve an
existing `.blend` and remove the new sequence directory.

The synthetic regression in `tests/test_blender_export.py` checks native Kinect
RGB offsets, pixel-center UV projection, holes and depth edges, failed-write
preservation, then saves, moves and reopens a Blender scene to inspect its one
camera/one slot, calibrated projection, material packing and animation. Set
`BLENDER_EXECUTABLE` to run that round trip in a separate factory-startup process.
It establishes export behavior, not scan-pose accuracy or large-session memory
capacity. Mac packaged-client verification remains a separate installed-app check.

## Full session preservation

**Save full RGB-D session**, or `/api/scan/export/session`, returns a ZIP with
lossless RGB PNGs, uint16 depth PNGs, `manifest.json`, and
`reconstruction.json`. The report stores settings/calibration, backend, accepted
camera-to-world poses in metres, original poses when refinement applied,
per-frame tracking diagnostics, timings, and refinement outcome. It deliberately
keeps estimated poses separate from evaluation reference poses.
Native Kinect projects retain raw disparity and its complete sensor calibration;
metric dataset projects retain millimetre depth. Use `manifest.depth_unit` and
the saved calibration rather than assuming every uint16 PNG is millimetres.

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
