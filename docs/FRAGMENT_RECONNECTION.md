# Reconnecting separated scan fragments

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
   Fragments contain at most sixteen views. Live poses provide initial guesses;
   only the first retained fragment fixes the world coordinate system.
2. Search for fragment overlap independently of the broken live trajectory.
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
   single-view fragments cannot authorize a bridge. Reject competing verified
   transforms that disagree by over 5 cm or 5°.
4. Optimize the anchored fragment pose graph with Open3D's Levenberg–Marquardt
   optimizer. Its measured spanning tree supplies the initial trajectory;
   additional bridges are uncertain constraints. Recompute connectivity after edge
   pruning, then validate optimized bridges against held-out geometry and any
   supporting visual correspondences again.
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

The pass considers at most 32 fragments and 256 fragment pairs, with up to five
key views per fragment, 12,000 aggregate training points, 30,000 validation points,
and two geometric RANSAC proposals of at most 12,000 iterations each. Pairs
involving the world anchor or consecutive fragments are prioritized when
appearance scores are equal. Preparation is bounded by the
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
After integration with the current capture-performance changes, the full suite
ran 266 tests: 264 passed and two CUDA tests were skipped. The separate synthetic
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
That historical result has not been rerun under the revised verifier.
