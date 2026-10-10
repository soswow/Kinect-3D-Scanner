# Reconnecting separated scan fragments

## Experimental depth-only final registration

The **Final registration** selector provides **Depth, color and motion (experimental)**
alongside the existing fragment algorithm described below. Live reconstruction
is a separate option. New UI scans start with live reconstruction off; saved
preferences and old session settings retain their choices. Depth registration
does not depend on accepted live poses. It prepares every retained raw capture
and runs at Finish. With color recovery/relocalization enabled, fixed measured
RGB-D correspondences constrain poses and retrieve loops both within a map and
between maps. Continuous camera-side poses propose initial alignments; reliable
gravity optionally weights and constrains orientation. These inputs are
uncertain evidence. Depth-only operation remains supported without them.
Ordinary updated clients also send bounded intermediate feature identities,
pixels and axial depths. These permit a sparse joint camera/landmark fit without
uploading intermediate images. Selected depth surfaces and measured shared planes
add partial surface constraints; gravity adds roll/pitch evidence. The first
camera fixes the coordinate gauge. None of these supplies ground-truth poses.
With this continuous evidence, internal drift is corrected before accumulated
maps are rigidly aligned, avoiding the earlier per-frame geometric revisit
search. Maps without that evidence retain the geometric fallback search.
An opened project can switch final registration before Finish without resetting
its captures. Switching modes invalidates the previous registration cache;
calibration and capture settings remain fixed. Depth mode also skips legacy
RGB pose refinement retained in an older project's settings.

The old requirement for two distinct camera positions on each side was a
conservative defense against repeated surfaces and partial false matches, not
a mathematical requirement for rigid registration. A pair of asymmetric depth
views can determine a camera transform. A plane, sphere, repeated corner, or
insufficient overlap may leave several transforms plausible, regardless of how
many times the same view was captured.

Depth mode replaces the fixed camera-count rule with separately sampled depth,
bidirectional overlap, six-direction pose conditioning, measured empty-space
checks, competing hypotheses, reciprocal refinement, graph checks, and
component-wide visibility validation. Millimetre range noise can falsely make a
plane appear determined; validation normals use larger neighborhoods to reduce
that effect. Samples are disjoint before point-cloud downsampling, but depth
filtering mixes neighboring pixels, so they are not independent sensor-noise
draws. Numerical thresholds are engineering policies, not calibrated confidence
probabilities.

It searches robust local ICP, GICP, multiscale FPFH/RANSAC, a bounded point-pair voting
prototype, and dense depth odometry. Accumulated component geometry supplies
additional global proposals. Temporal interpolation supplies guesses only; it
never authorizes a pose. There is no 32-fragment/16-view ceiling, minimum camera
count, fixed timestamp-gap rejection, or 30 cm/30 degree pose rejection in this
mode. Search still uses computational budgets: nearby captures and selected
descriptor candidates. Unsearched or ambiguous connections remain unresolved.

All existing views on both sides contribute to bridge validation. Final
components are checked against every other depth view, including pairs that
were never graph edges. A pose graph can have mutually consistent edges yet
place a desk in space another view measured as empty. Such a candidate must not
be promoted merely because it includes more frames. The selected validated
component is fused; separate components and their independent camera poses
remain in the report/session. Their relative placement is unknown. Rigid poses,
metric calibration, real overlap and available memory remain necessary.
The final audit allows isolated moving surfaces: its mean measured empty-space
conflict limit is 8%, while severe individual view-pair conflicts remain in the
report. A single changed object cannot veto an otherwise supported room. This
engineering policy does not certify absolute accuracy or explain every conflict.

CUDA is used for dense depth odometry, optional CuPy descriptor matching and
TSDF fusion on a CUDA engine. FPFH computation, RANSAC, point-pair voting,
verification and graph optimization currently run on CPU. CPU execution is
supported. GICP, single-scale FPFH and PCA proposals are also benchmarked; a
method's pair-match count is not evidence of a correct complete reconstruction.

### Assumptions reconsidered

| Previous policy | Decision in depth mode | What the evidence still requires |
| --- | --- | --- |
| Two distinct camera positions on each side of a bridge | Removed; one pair can connect isolated views | Six determined pose directions, plausible alternative poses checked, raw-depth consistency |
| Live tracking determines which captures are usable | Removed from offline registration | Every stored depth frame is considered; unconnected frames keep independent component poses |
| The first accepted fragment fixes the model | Largest component that passes the final checks is selected | Each independent component needs a coordinate gauge; its relative placement remains unknown |
| Motion above 30 cm / 30 degrees or long capture gaps is unacceptable | Removed as an acceptance rule | Motion and time may order guesses; measured depth determines whether a guess works |
| Up to 32 fragments and 16 views represent the session | Removed | All frames participate; descriptor retrieval still limits proposal search and can miss a connection |
| RGB supplies necessary disambiguation | Optional measured RGB-D constraints supplement depth | Textured planes can be determined by distributed color identities; untextured planes remain ambiguous |
| More connected frames or agreeing graph edges imply a better model | Rejected | Optimize local maps before placement and check non-edge depth observations for contradictions |
| ICP refinement necessarily improves a good seed | Rejected | Test multiple initial poses; a coarse refinement can drift along a plane |
| Diverse surface normals determine the camera pose | Replaced with a centered six-direction Jacobian check | A sphere has diverse normals but ambiguous rotation; noisy planes can create false apparent information |
| Reconstruction clipping is also the right registration range | Tested separately with `--registration-far` | Source depth must actually contain extra useful measurements; fusion settings remain unchanged |
| GPU support decides which methods are worth trying | Rejected | CPU implementations remain eligible; measure useful CUDA stages separately |

This is an experimental consistency filter, not a proof of physical identity or
absolute accuracy. No recorded trajectory is ground truth. Moving objects,
mirrors, calibration errors, repeated room structure and genuinely missing
overlap can still defeat these checks. Component-wide verification can reject a
bad connection without identifying the correct replacement; retained captures
remain available to later methods.

The October 10 update uses `offline_depth_graph_v2`. It retains fixed color
identities in the reverse fit rather than letting anonymous depth ICP erase
them. Batched point-pair votes cache oriented pairs, count pose bins together,
and discard gross gravity conflicts before ranking. Loop counts report actual
non-tree constraints in the selected map. Saved v1 registration results are
invalidated when Finish is requested. The benchmark results below describe
the earlier depth-only implementation unless explicitly stated otherwise.

```sh
python scripts/reconnect_session.py export/your-session.zip \
  --depth-geometry --output-dir export/depth-registration --save-session
python scripts/benchmarks/benchmark_offline_geometry.py export/*.zip \
  --graph --device cuda --output benchmark-output/depth-graphs.json
```

The audit deduplicates raw observation/calibration copies and reports saved
trajectories only as comparisons. Its optional `--registration-far` experiment
tests whether reconstruction clipping removed useful pose-estimation context;
it does not edit recordings or change fusion clipping. Pair caches are matched
to observation/calibration fingerprints and depth-revalidated. Keep captures,
meshes and large diagnostic reports in ignored output directories.

Further viable candidates include [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus)
for robust correspondence fitting, and learned depth features/matching from
[FCGF](https://github.com/chrischoy/FCGF) and
[GeoTransformer](https://github.com/qinzheng93/GeoTransformer). These are not
implemented or qualified here. Their build/runtime and pretrained-model
requirements need separate evaluation on the same recordings. A robust pose
solver does not certify that repeated surfaces represent the same physical
place. GPU availability determines an implementation route, not eligibility.
This Windows host also has an Ubuntu WSL2 instance with Python 3.12 and the RTX
3080 Ti visible to `nvidia-smi`; an isolated Linux learning runtime is feasible.
PyTorch and WarpConvNet are not installed there yet. Rendering the diagnostic
figures additionally needs Matplotlib; CuPy is optional for descriptor matching.

### Methods in the shared chat

Descriptors, correspondence solvers, local refinement and spatial indexing are
different stages. Adding a descriptor still needs a pose solver and the same
raw-depth checks; changing the solver cannot repair correspondences between two
different but similarly shaped objects. The following assessment concerns this
repository and the current Windows server, rather than claiming every possible
implementation has the same backend.

| Method | Present implementation / feasible addition | GPU route and priority |
| --- | --- | --- |
| FPFH and multiscale FPFH | Existing fragment FPFH plus new separate radii and concatenated descriptors; RANSAC proposals validated on all seven available recordings | CPU feature extraction/RANSAC; optional CUDA descriptor matching executed here |
| ICP / GICP | Existing robust ICP; new raw-view GICP recovery and multiscale refinement | New verifier/refinement uses CPU. Existing tensor CUDA ICP is available elsewhere in the server; it needs separate parity/quality qualification before replacing this refinement |
| RANSAC | Existing and new geometric correspondence fitting | CPU Open3D implementation; eligible regardless of backend |
| TEASER++ | Feasible robust correspondence solver, absent from this runtime | CPU C++/Python integration is useful; not dependent on a GPU. Test correspondence failure separately from pose fitting |
| ISS + FPFH | [Open3D ISS](https://open3d.org/docs/latest/tutorial/geometry/iss_keypoint_detector.html) is callable, but not integrated into this pipeline | Low-cost CPU candidate. Test whether selecting stable corners improves matches or removes needed overlap |
| SHOT, 3D Shape Context, Spin Images | [PCL supplies these descriptors](https://pointclouds.org/documentation/group__features.html); they need a compiled adapter and pose fitting here | CPU first. Alternative shape descriptions are worth testing on failed pairs; no demonstrated improvement on these scans yet |
| PPF | New sampled oriented point-pair voting prototype; absent from the installed OpenCV surface-matching build | CPU voting executed here. It is not a complete implementation of OpenCV's Drost detector |
| Super4PCS | Feasible [OpenGR](https://github.com/STORM-IRIT/OpenGR) global geometric proposer, absent here | CPU C++ adapter. Useful because it supplies hypotheses without descriptor correspondences; no recorded-session result yet |
| FCGF | Feasible pretrained depth feature proposer, not implemented here | CUDA. The maintained [FCGF](https://github.com/chrischoy/FCGF) WarpConvNet route currently specifies Linux x86_64, PyTorch and CUDA; isolate its environment instead of replacing the scanner runtime |
| GeoTransformer | Feasible pretrained geometric matcher, not implemented here | CUDA/PyTorch candidate for low-overlap components; model/operator installation and Kinect-domain evaluation are still required ([upstream](https://github.com/qinzheng93/GeoTransformer)) |
| PointNet++ | A feature-learning backbone, not a ready camera-pose solution; upstream provides classification/segmentation networks | GPU learning is possible, but [PointNet++](https://github.com/charlesq34/pointnet2) alone does not supply pretrained registration correspondences. Prefer testing FCGF/GeoTransformer first |
| Octrees / spatial hierarchies | Spatial indexing can reduce search work; the current geometry uses voxel sampling and KD-trees | CPU/GPU implementations possible; changing the index does not resolve repeated geometry by itself |
| RGB + depth | Existing legacy visual assistance remains available | Not used for camera estimation in this experiment; stored RGB can still color the final surface |

The next substantive feature experiment is pretrained geometry matching on the
unconnected components, followed by TEASER++ or Super4PCS as alternative pose
proposers. This is a priority judgment, not a measured superiority claim. Each
must face all available recordings and the same non-edge visibility checks;
the hard scan cannot be declared solved on pair overlap alone.

### Recorded qualification, 10 October 2026

All seven distinct full recordings available on this Windows host were tested,
covering 776 raw captures. Copies with revised trajectories were deduplicated;
the 21-view box subset is covered by the full hard recording. Historical
chest-1/chest-2 recordings on the Mac were not accessible here. Measurements and
exact registration-source hashes are in
[offline-depth-registration.json](benchmarks/offline-depth-registration.json).

| Recording | Captures | Saved accepted count | Components passing the new checks | Rejected component sizes |
| --- | ---: | ---: | --- | --- |
| chest-3 | 126 | 68 | 126 | none |
| chest-4 | 164 | 4 | 120, 1 | 43 |
| chest-5 | 27 | 7 | 27 | none |
| chest-6 | 38 | 38 | 38 | none |
| chest-7 | 115 | 108 | 31, 1 | 83 |
| chest-8 | 145 | 143 | 57 | 88 |
| Live-off hard scan, `912b6d41` | 161 | 22 | 55, 40, 25, 21, 7, 1 | 12 |

These are consistency results, not accuracy measurements. Saved accepted counts
are comparisons, not reference truth: the archived trajectories for chest-3,
chest-6, chest-7, chest-8 and the hard scan fail the same empty-space audit. The
new thresholds may also overreject. In particular, chest-7/chest-8 remain partial;
the depth mode is an experimental alternative, not a universal replacement.

For the hard scan, an earlier 157-capture candidate was rejected after 18.85% of
24.44 million sampled surface projections contradicted measured empty space.
The final selected 55-view map has 2.30% conflicts across 3.39 million tested
projections and passes the current supported-pair thresholds. Its new CUDA-fused
mesh has 614,651 vertices and 1,173,231 triangles. All 161 observations remain in
the exported session; all 322 PNG CRCs, timestamps and capture metadata match
the original. The other passing maps have independent camera coordinates, so
149 captures across those maps do not constitute a complete registered room.
The original ZIP and the running 161-capture server scan are unchanged.

The focused 86-test geometry/UI/project/legacy-fragment run passes with explicit
CPU fusion; the subsequent 123-test UI/API/transport/workflow run also passes.
CUDA descriptor checks pass, and real hard-scan CUDA fusion produced the verified
artifacts. This Open3D build raises a CUDA driver-shutdown error at interpreter
exit after export, reproduced with the unchanged master's legacy CLI. CPU CLI
execution exits cleanly. A broad 1,834-test run exposed 24 existing failures/errors
reproduced on master, plus a new script-catalog issue that was fixed. No historical
source-proof hashes or unrelated guards were relaxed to make that suite green.

## Existing fragment mode

Enable **Reconnect separated views at Finish** under advanced scan settings,
then capture and press **Finish Scan**. The GUI enables it for new scans and
remembers your choice. The server setting is `reconnect_fragments`; legacy API
requests and saved profiles that omit it retain their previous behavior.

Recovery runs exclusively during Finish. Progress shows fragment preparation,
link verification, graph optimization, and fresh fusion. It does not run in the
background while you move. Live tracking still pauses fusion on loss and asks
you to return to the last tracked image and highlighted camera position.
Continue capturing overlapping recovery views or retrace your path; those raw
observations can also help the Finish pass.

[Continuous visual tracking](CONTINUOUS_VISUAL_TRACKING.md) now associates
selected captures with nearby measured keyframes while scanning. The full
fragment graph optimization described here remains a Finish operation.

## What the algorithm does

1. Reconstruct calibrated, filtered point clouds from retained raw observations.
   Split sequences at timestamp gaps over three times the typical capture interval
   (at least two seconds), missing geometry, and failed adjacent registration.
   Each fragment gets its own local coordinates.
   Adjacent registration uses measured visual/depth correspondences or
   reciprocal geometric ICP, held-out points,
   and the configured motion limits, including for previously accepted views.
   Fragments contain at most sixteen views. Verified adjacent motion across that
   storage boundary supplies a measured sequential edge. A size limit alone
   does not break a connected track. Live poses provide initial guesses;
   only the first retained fragment fixes the world coordinate system. A size
   boundary also retains the two preceding cameras as local overlap witnesses;
   those context captures are not owned or fused a second time.
2. Preserve additional short-range RGB-D connections across fragment boundaries.
   Finding one local reference must not discard a different valid reference in
   another fragment. Search at most three capture steps, within the ordinary
   timestamp-gap limit, using measured feature refinement, distributed visual
   identities, held-out depth and the existing per-step motion limits. These
   temporal ties require synchronized RGB-D and do not accept archived or client
   poses as authority. Contradictory measurements are reported as ambiguity.
   Then search for fragment overlap independently of the broken live trajectory.
   Synchronized ORB/PnP matches propose transforms; FPFH descriptors and bounded
   RANSAC also propose transforms using depth alone. RGB-D pairs over 20 ms
   apart cannot provide appearance proposals.
3. Verify each proposal with reciprocal coarse-to-fine point-to-plane ICP,
   nonplanar normal coverage, bidirectional overlap, and independent held-out
   samples. Partially overlapping unions can be verified using their shared
   camera observations. Require supporting camera pairs from at least two
   distinct positions on each side, separated by over 2 cm or 2°. Distributed
   measured visual correspondences can constrain planar overlap when those same
   independent camera pairs also pass held-out depth verification. Stationary duplicate captures and
   fragments with only one measured viewpoint cannot authorize a global bridge.
   A short fragment can use its measured overlap context. Reject competing verified
   transforms that disagree by over 5 cm or 5°.
4. Optimize the anchored fragment pose graph with Open3D's Levenberg–Marquardt
   optimizer. Build the measured spanning forest before choosing world poses:
   sequential and verified temporal connections come first, followed by visual
   bridges, then geometric-only bridges. This preserves measured connectivity
   even inside a component that has not yet reached the world anchor. The forest
   supplies the initial trajectory;
   additional bridges are uncertain constraints. Recompute connectivity after edge
   pruning, then validate optimized bridges against held-out geometry and any
   supporting visual correspondences again.
   Revalidate every measured temporal/storage boundary whose endpoints remain
   connected, even if the optimizer pruned its original edge. Pruning a constraint
   cannot authorize a contradictory pose through a different graph route.
   If the adjustment fails those checks, revalidate the measured spanning-tree
   poses and retain only still-valid surviving links connected to the first
   fragment. Optimization-pruned edges remain removed, and diagnostics explicitly
   record the fallback. This preserves measured connectivity without claiming
   that global drift was corrected.
   If fallback poses still contradict a measured boundary, fail before fusion
   and preserve the previous reconstruction, including the full diagnostic report.
   Re-estimate accepted poses as well as skipped views. Exclude observations
   without a verified connection to the first fragment. Optional final pose
   refinement can subsequently refine that connected trajectory using its
   separate validation rules.
5. Count the required frustum blocks before allocating and fusing a fresh TSDF
   volume. Commit poses, diagnostics, and tracking state only after native fusion and model extraction
   succeed. A failed allocation, exhausted fusion budget, or invalid proposal
   preserves the existing reconstruction and reports Finish as failed. The full
   verification report survives a fusion failure. The live feedback snapshot is
   refreshed after Finish, and exports use the resulting connected poses.

The implementation follows Open3D's [global registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
and [multiway pose-graph registration](https://www.open3d.org/docs/latest/tutorial/pipelines/multiway_registration.html)
approach with additional conservative verification. These checks reduce false
matches; they cannot establish unique identity from perfectly repeated surfaces
or invent missing overlap. Include corners, asymmetric details, and a stable
background when capturing a bridge.

## Budgets and diagnostics

The pass considers at most 32 fragments and 256 global fragment pairs, with the
first and last five owned key views plus a midpoint and two overlap witnesses per
fragment, 12,000 aggregate training points, 30,000 validation points,
and two geometric RANSAC proposals of at most 12,000 iterations each. Pairs
crossing from the anchored reconstruction to an unconnected fragment are tested
first. Once no crossing pair can connect a remaining component, pairs entirely
inside that unreachable component are skipped. Original evidence ranking still
orders global candidates within their evidence class and determines which loop
candidates precede early completion. The separate temporal pass considers only
cross-fragment cameras at most three capture steps apart; cameras lacking 40
mutual feature matches skip that pass's registration work.
Exact mutual descriptor matches are cached for at most sixteen target views per
prepared camera; pose-dependent verification is never cached. Preparation is bounded by the
scanner's raw frame limit. Search time varies with captured geometry; it is not
a real-time tracking path.

Fresh recovery fusion uses the live voxel size and the explicit final block
budget. It needs memory for the original and candidate volumes concurrently.
Optional finer final fusion then uses its separately selected resolution. A
repeated Finish reuses recovery results until another frame is stored; native
failures remain retryable.

`reconstruction.json` and `/api/scan/diagnostics` include `fragment_reconnection`:
local camera poses for every prepared fragment, connected/unconnected fragment
IDs, geometric bridge evidence and connection status, ambiguous pairs, recovered
view count, corrected/excluded accepted views, elapsed time, required fusion blocks,
fusion budget, and whether a search cap was reached.
`temporal_bridges` counts additional measured boundary edges;
`ambiguous_temporal_pairs` records contradictory nearby measurements.
`rejected_optimized_boundaries` and `rejected_fallback_boundaries` identify
output placements that violate retained raw boundary evidence.
Unassigned indices identify observations beyond the fragment cap. All retained
raw frames remain in **Save Session…**, including unconnected and unassigned
views. They are excluded from the world mesh until a connection is verified.
Original live poses and the previous rejection message are preserved in exports.

## Rebuild an older session ZIP

From the project environment:

```bash
python scripts/reconnect_session.py export/your-session.zip \
  --output-dir export/reconnected --save-session --device cpu
```

The command reads calibration, settings, lossless images, and capture metadata.
It replays tracking under the current guards and attempts reconnection. It never
uses the archive's saved pose estimates as registration authority or changes the
source ZIP. Outputs are `reconstruction.json`, `result.json`, `reconnected.ply`
when a mesh is available, and optionally `reconnected-session.zip`. A successful
mesh build may still contain only the trusted portion: read the recovered count
and unconnected fragment list. `--final-weight` can change extraction confidence;
it does not relax registration verification.

`--use-pose-seeds` skips repeating live tracking. It reads archived camera guesses,
then independently revalidates their local motion and all fragment connections
against the retained raw depth. It never exports a mesh from the guesses alone.
`--block-budget 10000` permits a larger candidate volume; the command reports the
measured requirement and fails before fusion if that budget is insufficient.

Live tracking also verifies camera-to-camera geometry after unusually long capture
gaps or steps over 10 cm / 8°. Slow regular adaptive capture does not create a
separate fragment for every image. A confident match to the accumulated model
cannot by itself authorize those transitions.

## Validation

Actual raycast RGB-D tests separate two overlapping camera runs with a capture
gap and a motion jump that live tracking rejects. Depth-only fragment matching
recovers the missing views within 3 cm / 3° of a reference trajectory used only
for test evaluation, then builds and exports the connected model. Tests also
cover planar ambiguity, repeated stationary captures, conflicting proposals,
pruned bridges, search caps, cache invalidation, native fusion failure rollback,
raw ZIP replay, server counts/progress/snapshot refresh, and GUI preference
restoration. The saved Kinect session repair below was verified on CPU.
Updated live tracking guards have not yet been exercised in a new physical
Kinect capture. CUDA reconnection remains unverified.

The regression suite includes synthetic HTTP/WebSocket and Qt scan workflows
through capture, Finish, rebuilding, and exports. New raycast tests exercise
confidently accepted but drifted components, exclusion of unverifiable accepted
views, raw-camera verification after tracking gaps, archived pose seed
revalidation, and fusion-budget failure before any candidate integration.
After integration with continuous visual tracking and the algorithm review, the
full suite ran 285 tests: 283 passed and two CUDA tests were skipped. The separate synthetic
HTTP/WebSocket and Qt workflow passed capture, Finish, rebuilding, and all exports.

## Chest session repair, 6 October 2026

The 200-capture session `chest-2scan-session_20261006_090302.zip` originally
accepted 123 captures, including a displaced returning chest. Its old Finish
pass retained accepted fragments as fixed roots instead of revalidating their
alignment. Fresh fusion also exceeded its 5,000-block budget, leaving the live
volume in use.

Rebuilding with the revised verifier and `--use-pose-seeds --block-budget 16000`
produced 174 connected captures: all 77 rejected views recovered, 95 accepted
poses corrected, and 26 unverified accepted captures excluded. There are 16
connected fragments with 47 surviving measured bridges. The remaining six
fragments are preserved in local coordinates in the diagnostics. Excluded
captures are 137–158, 163–164, and 177–178 (one-based).

The returning capture 193 now agrees with early capture 3: bidirectional raw
depth overlap at 25 mm is 92% / 65%, with approximately 12 mm nearest-neighbour
RMSE. Its relative camera pose differs from direct raw-depth alignment by
14 mm / 0.65°, compared with 0.95 m / 41° in the original trajectory. These are
alignment checks, not absolute dimensional accuracy measurements. Both oblique
and overhead mesh previews show one chest without the displaced second copy.

At 5 mm voxels, the verified fresh fusion requires 6,541 blocks. The mesh contains
243,771 vertices and 471,464 triangles. CPU repair took about 19 minutes. Outputs
are in `export/chest-2-repaired/`: `reconnected.ply`, `reconnected-session.zip`,
the reconstruction and verification reports, and mesh/alignment previews. All
400 RGB/depth image CRCs and capture timestamps match the source session. The
source archive SHA-256 is unchanged.

For this scan size, select a final block budget of at least 6,541; 10,000 gives
room for additional observations. Keep **Reconnect separated views at Finish**
enabled. The normal pose-refinement pass still reports no further loop
constraints on this session; the correction comes from the independently
measured fragment graph.

## Earlier 50-capture session

An earlier implementation replayed `chest_20261005_230320.zip` and accepted the
first 14 of 50 captures. The offline pass creates 17 local fragments and tests
96 of 136 candidate pairs, including every pair involving the trusted anchor.
It finds one geometrically supported link between later fragments, but no
verified link to the trusted first fragment. **Zero additional views are fused.**
Sixteen fragments remain unconnected; no observation exceeds the fragment cap.
The pair search cap is reported explicitly. This is a failed full reconnection,
with a safe partial reconstruction, rather than evidence that every possible
matching algorithm would fail.

The partial mesh contains 39,803 vertices and 74,512 triangles at the archived
surface confidence. Outputs are in
`/Users/sasha/hobby/xbox360/chest-session-analysis/fragment-reconnection/`.
All 100 RGB/depth PNG CRCs in the new session ZIP match the source archive's
50 captures. The original ZIP is unchanged. A new capture sequence that overlaps
both the trusted view and a later fragment, with distinctive corners or texture,
was recommended to supply an unambiguous bridge for that earlier session.
The 6–7 October [algorithm review](ALGORITHM_REVIEW.md) reran this archive with
the current verifier and its archived pose seeds. It retains 16 of 50 views,
including two previously rejected captures, and excludes 28 unverified accepted
poses. Fresh fusion produces 82,058 triangles, but the reconstruction remains
partial. That initialization differs from the earlier live replay above.
