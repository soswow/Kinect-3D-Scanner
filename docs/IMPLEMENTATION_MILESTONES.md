# Ordered implementation milestones

Continue from `50d448c`, following RESEARCH_ROADMAP.md. Each milestone is a
separate tested commit; experimental options stay opt-in. Existing uncommitted
UI dock sizing edits are preserved outside these commits.

1. **Device evidence and calibration.** Checkerboard fitting with held-out
   reprojection, coverage and tilt checks; common RGB/depth lens rectification;
   measured depth-scale correction; plane precision and pairing reports.
   Implemented and verified with known-camera fixtures and the regression suite.
   Physical accuracy acceptance needs a measured
   checkerboard/plane and repeated recordings; no fabricated calibration.
2. **Compute profiling.** Matched replay reports and backend/stage attribution.
   Actual NVIDIA throughput remains unverified until run on an NVIDIA server.
3. **Global tracking.** Appearance retrieval outside estimated pose radius,
   reciprocal geometric verification, held-out checks and fresh fusion.
4. **Sensor-aware fusion.** Explicit range/angle/edge confidence, controlled
   surface tests, observed-space preservation; compare error and completeness.
5. **Final appearance.** Measured-view exposure consistency and sharp view
   selection, with coverage reports and an observed-color fallback.
6. **Live/final budgets.** Separate final reintegration and memory limits;
   show actionable tracking/backlog feedback.

The older SCAN_QUALITY.md roadmap also requests turntable capture and adaptive
capture. Turntable support remains a separate subsequent milestone: foreground
masking alone cannot give the object's rotation. It needs measured angles and
an axis, or a validated object-pose tracker. Surface evaluation and capture
feedback belong alongside milestones 4 and 6.

## Calibration workflow

Save a full RGB-D session and unzip it into ignored `recordings/`. For intrinsics,
collect at least 16 sharp views of a flat checkerboard spanning the image and
several tilts. Columns/rows count **inner corners**; measure printed square size.

```bash
python scripts/calibrate_camera.py recordings/board \
  --output benchmark-output/measured-camera.json intrinsics \
  --columns 9 --rows 6 --square-mm 25
python scripts/calibrate_camera.py recordings/plane \
  --output benchmark-output/plane.json evidence \
  --plane-roi 100 100 540 380 --expected-z-m 1.0
```

Only accepted fits write a loadable camera JSON; a separate report records
held-out errors and rejection reasons. Load that camera through the GUI's
calibration control. Depth scale is **not** estimated from RGB checkerboards:
use several known front-facing plane depths to validate any scale correction.
Plane residuals describe precision, not absolute accuracy. Recordings remain
raw; correction is applied consistently before tracking, fusion and texturing.
Lens rectification keeps K unchanged and uses nearest-neighbour depth sampling
so missing depth stays missing. Recordings replay their pairing metadata.

Calibration follows [OpenCV's calibrated camera model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

## Compute profiling milestone

Implemented isolated, sequential replays with per-stage total/p50/p95, stage
placement, acceptance/error, source hash and peak process RSS. It detects source
changes during a run. CUDA requests fail explicitly rather than falling back.

```bash
OMP_NUM_THREADS=4 python scripts/profile_backends.py --dataset redwood --repeats 3
# On an NVIDIA server, use the same recorded input and settings:
python scripts/profile_backends.py --dataset recording --path recordings/session \
  --repeats 3 --runs cpu-legacy cuda-tensor
```

Two sequential five-frame Redwood repeats on this Mac accepted 5/5 in both CPU
modes: median processing/final-extraction 3.37 s legacy, 4.06 s tensor. These short
runs are a profiling smoke check, not a general throughput result. CUDA was
unavailable. Peak RSS includes CPU allocations and imports, not GPU VRAM.
[Aggregate evidence](benchmarks/backend-profile.json) contains no captured images.
NVIDIA throughput validation remains pending target hardware. The current hybrid
stage placement is explicit; no new GPU optimization is justified by a CPU-only
profile.

## Global tracking milestone

Implemented bounded mutual ORB retrieval and calibrated measured-depth PnP
proposals, independent of estimated camera separation. Final refinement spends
at most half its loop budget on appearance candidates, verifies reciprocal ICP,
and retains the independent geometry checks and transactional fresh fusion.
Unverified sequential odometry is a weak prior. Verified non-planar RGB-D seeds
can initialize poorly conditioned sequential edges; flat targets cannot use that
fallback. Graph optimization starts from these sequential constraints.

Tests recover a revisit with 0.45 m injected position drift outside the previous
0.35 m radius, and recover a lost pose 0.6 m from its correct location. These are
controlled fixtures, not calibrated Kinect accuracy. Wrong depth, blank/blurred
RGB, and known pairing lag >20 ms cannot authorize appearance recovery. Existing
flat-loop rejection and failed-reintegration preservation checks still pass.

Online recovery remains **off by default**, activated by reset `relocalize: true`
or replay `--relocalize`; it runs only after two rejected frames and tests up to
three candidates from at most 32 accepted keyframes. Final refinement remains
opt-in through the existing GUI checkbox/`--refine-poses`. The desk sequence and
repeated-scene failure cases remain important limitations; this is not a full
SLAM replacement or an RTAB-Map integration.

## Sensor-aware fusion milestone

`confidence_fusion: true` / replay `--confidence-fusion` enables fractional TSDF
and RGB weights from range, surface angle and measured depth discontinuities.
Missing depth contributes zero; holes are not filled. The range model is an
engineering prior, **not fitted Kinect noise coefficients**. Native uniform
integration remains the default. CPU integration is tested; the same tensor
operations can run on CUDA but need target-hardware verification.

Controlled tests show less displacement from a deliberately biased far-depth
observation than uniform averaging. This demonstrates the weighting mechanism,
not real sensor accuracy. A five-frame Redwood smoke replay accepted 5/5;
`--final-weight 0.5` produced a mesh. The usual final weight of 2 can leave no
surface after so few fractional observations: collect more views, or explicitly
choose a lower confidence threshold. Do not interpret that threshold as a
probability. The custom path also adds work; it is not a throughput optimization.

Evaluate measured reference surfaces in their known shared coordinate system:

```bash
python scripts/evaluate_surface.py scan.ply reference.ply \
  --threshold-mm 10 --output benchmark-output/surface.json
```

Area-weighted bidirectional samples report error, precision, completeness and
F-score. There is no alignment or scale fit that could hide drift. A half-plane
fixture has low error but only about half completeness. Real improvement still
requires calibrated physical reference surfaces and repeated scans.

## Tracking evidence still pending

A single 58-frame TUM desk run accepted 54 frames, proposed 13 appearance loops,
and verified none. It retained the original poses; anchored position RMSE was
0.430 m. This difficult scene is not solved by the new retrieval. The run
coincided with tests, so its elapsed time is not a performance comparison.

## Final appearance milestone

Textured GLB/OBJ bundle exports now offer **Match texture exposures** and
**Use one best view per texel**. Both are off by default. API query parameters
are `exposure_correction=true` and `blend_mode=best`; `blend` retains the existing
weighted blend. OBJ bundles contain OBJ, MTL, PNG and the texture report. A plain
OBJ's vertex colors remain less portable than these materials.

Exposure fitting uses at most 8192 shared, depth-visible atlas samples, rejects
saturation and spatially varying ratios, bounds per-channel gains to 2/3–1.5,
and requires held-out overlap improvement before applying them. Disconnected
view groups have independent anchors. Tests recover imposed exposure changes
and reject variable lighting; actual UV projection and occlusion also pass.
This is relative RGB gain matching, not full radiometric calibration.

The best-view mode chooses the strongest angle/distance observation at each
texel from the existing sharp synchronized view selection. It can reduce blend
blur but may reveal seams: there is no seam optimization or photometric pose
warping yet. Missing/occluded observations keep fused vertex color. The report's
projected fraction describes atlas coverage, not whole-object completeness.

## Live/final budget milestone

Implemented optional `final_voxel_m` (2 mm up to live voxel size) and
`final_block_count` (default 5000). GUI settings expose these plus the experimental
relocalization and confidence-fusion toggles. Final reconstruction fuses accepted
poses into a fresh volume without changing live tracking or raw frames. The block
limit is checked before each integration; no implicit coarse fallback is used.
A failed allocation, insufficient surface or budget overflow preserves previous
geometry. Successful unchanged builds reuse the final volume; new captures
invalidate it. Controlled tests cover both paths.

```bash
python scripts/replay_scan.py --dataset recording --path recordings/session \
  --final-voxel-mm 3 --final-block-count 5000 \
  --output benchmark-output/final-detail.json
```

5000 blocks use about 391 MiB of voxel attributes **plus** the live volume,
frames, hash maps and scratch/extraction memory. This is not a total RAM cap.
Finer grids are a representation option, not evidence of finer sensor accuracy.
The live payload remains bounded, adds skipped count/server queue age and
actionable guidance. Auto capture can wait for server or local-upload backlog and resume automatically;
manual capture still works. Motion/coverage-based keyframe selection is future
work and no whole-object completeness percentage is inferred from a live cloud.

## Next evidence and subsequent work

1. Collect checkerboard/known-plane recordings and repeated scans of a measured
   object; use the calibration and surface tools before changing defaults.
2. Run the profiling harness and CUDA tests on an NVIDIA server. CPU smoke tests
   cannot establish CUDA speed, VRAM behavior or weighted-kernel correctness.
3. Test repeated/weakly textured scenes and longer loops; compare an external
   RTAB-Map pose provider if the bounded tracker still fails.
4. Tune confidence only against measured sensor errors and reference completeness.
5. Compare texture seams/sharpness on real RGB; patch seam leveling is still open.
6. Implement an explicit turntable model with measured rotation axis/angles and
   separate foreground masks. Do not infer object rotation from a static background.

The six software milestones are reviewable checkpoints; physical/NVIDIA evidence
is still required for claims of calibrated accuracy or a default change.


## Final batch validation

The final suite passed 60 tests; two CUDA tests were skipped (62 total).
It covers measured-camera recovery, rejected matches,
transactional pose/final reintegration, weighted depth, error/completeness,
texture projection and material reloads. CUDA tests are present but skipped on
this Mac. The actual HTTP/WebSocket and offscreen Qt workflow covers live
feedback, settings transfer, backlog pausing, previews, finer final rebuilding,
textured exports and lossless sessions. A fresh public five-frame Redwood HTTP
replay accepted 5/5 with anchored position RMSE 1.36 mm; this is camera tracking
on that short sequence, not Kinect mesh accuracy.

[Aggregate validation](benchmarks/milestone-validation.json) preserves results
without captured images. Detailed logs and session data remain ignored.

## Milestone 7: withheld-view quality and repeated comparisons

Added a disjoint-phase RGB-D reader, held-out mesh raycasting and isolated
repeated quality comparisons. The scorer reports depth residuals together with
coverage and a missing-surface penalty. Exact-plane tests verify metric depth
under a rigid anchor; partial/empty surfaces cannot masquerade as lower-error
improvements. Reader and existing replay tests pass (9 checks). A Redwood smoke
run exercises successful uniform extraction and missing weighted extraction.

See [QUALITY_BENCHMARKS.md](QUALITY_BENCHMARKS.md) for commands, source/input
hashes, repeat statistics and limits. Public sensor depth is a consistency
reference, not an independently accurate physical surface. Reconstruction never
receives the held-out views or reference poses. Longer comparisons are being
collected before choosing the next tracking/fusion change.

## Milestone 8: depth-supported appearance proposals

Appearance features now require a measured center with at least seven valid
samples in its 3x3 patch. Discontinuous patches are rejected; supported patches
use the median of measured neighbours for proposal depth. RGB matching, PnP,
cross-depth checks, reciprocal ICP, normal diversity and independent refinement
validation retain their acceptance limits. There is no filling of missing
centers and raw depth/fusion inputs are unchanged.

The patch-spread limit (30 mm or 4% of range) is an engineering edge heuristic,
not calibrated Kinect noise. Tests cover depth edges, holes, unsupported centers
and noisy measured planes. The noisy-plane fixture recovers the correct proposal
where single-pixel cross-depth checks fail; a flat target still cannot authorize
final camera motion through the geometric gates. Existing distant-loop/lost-pose
checks pass. The options that use these proposals remain experimental.

The quality harness can compare `appearance_raw` and `appearance_supported`
with the same relocalization/final-refinement settings. The raw variant restores
only single-pixel feature sampling inside its isolated worker. The report records
that override. Better or more numerous proposal seeds are not evidence of better
final geometry; use the withheld-view and repeat results together.

Final regression: 68 tests passed, two CUDA tests skipped (70 total). The desk
comparison completed all four runs with unchanged source and matched inputs.
The modest two-repeat median improvement is documented without claiming solved
tracking or statistically established quality. Earlier unrelated working edits
are preserved outside these commits.
