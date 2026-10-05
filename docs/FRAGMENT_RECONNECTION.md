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

## What the algorithm does

1. Reconstruct calibrated, filtered point clouds from retained raw observations.
   Split sequences at timestamp gaps over two seconds, missing geometry, and
   failed adjacent registration. Each fragment gets its own local coordinates.
   Adjacent registration uses measured geometry, reciprocal ICP, held-out points,
   and the configured motion limits. Existing live poses provide world anchors.
2. Search for fragment overlap independently of the broken live trajectory.
   Synchronized ORB/PnP matches propose transforms; FPFH descriptors and bounded
   RANSAC also propose transforms using depth alone. RGB-D pairs over 20 ms
   apart cannot provide appearance proposals.
3. Verify each proposal with reciprocal coarse-to-fine point-to-plane ICP,
   nonplanar normal coverage, bidirectional overlap, and independent held-out
   samples. Require supporting camera pairs from at least two distinct positions
   on each side, separated by over 2 cm or 2°. Stationary duplicate captures and
   single-view fragments cannot authorize a bridge. Reject competing verified
   transforms that disagree by over 5 cm or 5°.
4. Optimize the anchored fragment pose graph with Open3D's Levenberg–Marquardt
   optimizer and uncertain bridge edges. Recompute connectivity after edge
   pruning, then validate optimized bridges against held-out geometry again.
   Existing accepted live camera poses remain unchanged; recovered poses extend
   their connected model. Optional final pose refinement can subsequently refine
   that connected trajectory using its separate validation rules.
5. Fuse all connected observations into a fresh TSDF volume. Commit poses,
   diagnostics, and tracking state only after native fusion and model extraction
   succeed. A failed allocation, exhausted fusion budget, or invalid proposal
   preserves the existing reconstruction. The live feedback snapshot is refreshed
   after Finish, and exports use the resulting connected poses.

The implementation follows Open3D's [global registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
and [multiway pose-graph registration](https://www.open3d.org/docs/latest/tutorial/pipelines/multiway_registration.html)
approach with additional conservative verification. These checks reduce false
matches; they cannot establish unique identity from perfectly repeated surfaces
or invent missing overlap. Include corners, asymmetric details, and a stable
background when capturing a bridge.

## Budgets and diagnostics

The pass considers at most 32 fragments and 96 fragment pairs, with up to three
key views per fragment, 6,000 aggregate cloud points, and two geometric RANSAC
proposals of at most 12,000 iterations each. Pairs involving a world anchor are
prioritized when appearance scores are equal. Preparation is bounded by the
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
view count, elapsed time, fusion budget, and whether a search cap was reached.
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

## Validation

Actual raycast RGB-D tests separate two overlapping camera runs with a capture
gap and a motion jump that live tracking rejects. Depth-only fragment matching
recovers the missing views within 3 cm / 3° of a reference trajectory used only
for test evaluation, then builds and exports the connected model. Tests also
cover planar ambiguity, repeated stationary captures, conflicting proposals,
pruned bridges, search caps, cache invalidation, native fusion failure rollback,
raw ZIP replay, server counts/progress/snapshot refresh, and GUI preference
restoration. Physical Kinect and CUDA reconnection remain unverified.

After combining with the current master, the full suite ran 214 tests: 212
passed and two CUDA tests were skipped. The synthetic HTTP/WebSocket and Qt
scan workflow passed through capture, Finish, rebuilding, and all exports.

## Chest session result, 6 October 2026

Replaying `chest_20261005_230320.zip` under the current loss guards accepts the
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
is still needed to supply an unambiguous bridge for this chest.
