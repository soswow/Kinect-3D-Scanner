# Development checkpoints and durable decisions

This is historical implementation evidence, consolidated from the completed
milestone, usability and RGB-D improvement plans. Current setup is in the
[scanner README](../README.md); current architecture and measurements are indexed
in [docs/README.md](README.md). Test totals below belong to their checkpoints.

## Reconstruction milestones

The ordered work started at `50d448c`. Calibration, profiling, distant appearance
retrieval, sensor-aware fusion, texture exposure/best-view export and separate
live/final budgets were implemented, followed by withheld-view evaluation,
depth-supported appearance proposals and benchmark metadata provenance.

| Decision retained | Current guide or tool |
| --- | --- |
| Load a complete Kinect calibration; single-camera analysis output is not a GUI calibration. Intrinsics need sharp checkerboard views with measured square size and tilt/coverage. RGB boards do not determine depth scale; use known plane distances. Precision residuals are not absolute accuracy. | [Calibration](../calibration/README.md), `scripts/calibrate_camera.py` |
| Profile identical recordings in isolated sequential processes; report stage placement, accepted indices, source/input hashes and host RSS separately from VRAM. CUDA requests must be explicit about execution/fallback. | [Script index](../scripts/README.md), [CUDA reference](CUDA_EXPERIMENTS.md) |
| Appearance retrieval can propose distant loops, but measured depth, reciprocal checks and separate validation authorize them. Three-by-three feature support requires a valid center, seven measured samples and bounded spread (30 mm or 4% of range); it does not fill depth holes. | [Loop refinement](POSE_REFINEMENT.md), [recorded evaluation](RECORDED_RGBD_EVALUATION.md) |
| Fractional TSDF confidence is a relative evidence score. Local inverse-depth plane normals change weights, not observed depth. Calibrated variance needs device noise, pose uncertainty and correlation evidence. | [Depth confidence](STATISTICAL_DEPTH_FUSION.md) |
| Exposure fitting uses at most 8,192 depth-visible samples, channel gains 2/3–1.5 and held-out improvement. Best-view texels may expose seams. Unobserved texels retain measured fused color. | [Texture exports](LIVE_RECONSTRUCTION.md#portable-texture-exports) |
| Pose corrections require fresh fusion and atomic commit. A failed candidate preserves prior geometry; a final block budget is not a total-memory cap or sensor-resolution claim. | [Final registration](FRAGMENT_RECONNECTION.md), [field allocation](FIELD_CUDA_RESEARCH.md) |
| Withheld sensor views never enter reconstruction. Missing geometry counts in quality loss. Metadata belongs in input hashes because timing changes appearance eligibility. Historical version-1 hashes must stay unchanged. | [Quality benchmarks](QUALITY_BENCHMARKS.md) |

Early validation progressed from 60 passing tests/two CUDA skips to 68 passing
tests/two CUDA skips. The five-view Redwood network smoke check had 1.36 mm
anchored position RMSE. The TUM desk run still had large drift and no trustworthy
loop; neither result establishes Kinect surface accuracy. Later native/CUDA and
field studies supersede the old statements that NVIDIA execution was pending.

The RGB-D improvement batch separately implemented camera/reference caching,
per-registration ICP source caching, local-plane confidence and optional sparse
camera/landmark optimization. Reports retain operation-level results at
[camera cache](benchmarks/live-tracking-depth-cache.json),
[ICP cache](benchmarks/icp-source-cache.json),
[confidence](benchmarks/confidence-fusion.json) and
[bundle adjustment](benchmarks/rgbd-bundle-adjustment.json). Source-cache timing
gains varied by recording; oblique confidence coverage could worsen; recorded
bundle proposals were refused. The batch's 371-test checkpoint had three CUDA
skips. See the focused guides for solver bounds and actual usage.

## Usability checkpoints

The usability baseline was `fdd978f` (95 tests, two hardware skips). Its completed
snapshots were `86603cb` (fixed actions, explicit Automatic/Manual and Pause/Resume),
`ac5464e` (unsaved-session protection, reconnect/retry and atomic exports),
`a044201` (prominent reconstruction, aspect-fit cameras and depth/crop legend),
and `56da9c8` (operation phases, build-timeout reconciliation and shortcut guards).

The durable workflow rules are:

- Start explicitly; Pause stops capture and Finish builds. Cancel protects
  unsaved captures and waits for confirmed reset; timeout leaves capture paused
  while status/retry reconciles server state.
- Keep primary actions outside the settings scroll area. The original 960×600
  and 1280×800 layouts were rendered and checked with synthetic input.
- Preserve camera aspect ratio, native calibrated crop and depth inclusion.
  Follow uses accepted poses; Orbit is independent. Scan view stacks color/depth.
- Confirm successful capture with a bounded 70 ms sound cue. Failed or stale
  acknowledgements stay silent. Offline capture now confirms durable local spool
  writes; live capture confirms upload (see [buffering](OFFLINE_CAPTURE_UPLOADS.md)).
- Save validated user preferences independently of a restored server session.
  Restore resolution before cadence and live voxel before final voxel; server
  restore suppresses preference writes. Environment connection values override
  that launch. Tests use isolated preference stores.

The initial UI checkpoint passed 142 tests/two CUDA skips and the synthetic
HTTP/WebSocket/Qt workflow, then cancellation, stacked previews, sound and
preference follow-ups were checked separately. These checks establish software
workflow behavior, not physical ergonomics or scan quality.

## Remaining research

Measured per-device error and timing, repeated physical accuracy/completeness,
long sequences and weak/repeated scenes remain needed. Overlapping bundle
windows, bounded projective/local-map tracking and calibrated robust fusion are
research candidates. [The roadmap](RESEARCH_ROADMAP.md) preserves named comparison
systems and integration tradeoffs. Seam leveling and an explicit measured
turntable model remain separate work: background masking cannot supply an
object's rotation axis or angle.

## Full historical records

Detailed completed plans, commands, numeric probes and validation chronology
remain in Git at the review baseline `762a6dac3b5a48b5865d382bbf18cd1746c37999`:
[implementation milestones](https://github.com/soswow/Kinect-3D-Scanner/blob/762a6dac3b5a48b5865d382bbf18cd1746c37999/docs/IMPLEMENTATION_MILESTONES.md),
[usability implementation](https://github.com/soswow/Kinect-3D-Scanner/blob/762a6dac3b5a48b5865d382bbf18cd1746c37999/docs/UX_IMPLEMENTATION.md), and
[RGB-D improvement plan](https://github.com/soswow/Kinect-3D-Scanner/blob/762a6dac3b5a48b5865d382bbf18cd1746c37999/docs/RGBD_IMPROVEMENT_PLAN.md).
Use the recorded revision for historical reproduction; running a command on
today's checkout measures today's code.
