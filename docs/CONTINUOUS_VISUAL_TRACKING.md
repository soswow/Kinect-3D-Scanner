# Continuous visual tracking

This guide combines current behavior with historical trials. Current defaults
and the separate offline mode are in [final registration](FRAGMENT_RECONNECTION.md).

Enable **Live fused point cloud feedback** and **Color-assisted tracking**, then
start a new scan. The camera worker now estimates motion on incoming camera
pairs while the server processes selected captures. Capture pacing and the
bounded upload queue remain in place. Restart the client and server to load the
changes; a running scan is not replaced by this development work.

## The two tracking speeds

```mermaid
flowchart LR
  Camera[Incoming camera frames] --> Motion[Fast local motion estimate]
  Camera --> Capture[Selected captures]
  Motion --> Capture
  Capture --> Verify[Verify against saved views]
  Verify --> Fusion[Live fused surface]
  Verify --> Links[Measured camera links]
  Links --> Finish[Finish: verify graph and rebuild]
```

The camera thread follows up to 500 image corners with persistent identities and pyramidal optical flow,
checks forward/backward agreement, and requires measured depth at both ends.
The reverse flow pass is the costlier half of that check, and every later test
only removes points from the forward set, so a reference that already lost most
of its forward tracks stops there instead of running the reverse pass.
PnP proposes motion, paired 3D points constrain metric motion, and a small joint
pixel/depth refinement retains the feature identities. This is a short-baseline
motion estimate, not a trusted world map. Unknown depth is never filled. It
tries up to five recent verified references, newest first. A blurred image,
insufficient support, inconsistent motion, or RGB/depth timing over 20 ms leaves
those references intact for the next image. A gap over 0.75 s since the last
verified reference starts a new local coordinate system. Camera restarts and
session/calibration changes also start new chains. Visual failure leaves
acquisition running.

Selected uploads carry `metadata.visual_tracking`: a segment identifier,
camera-to-local transform, validity, steps, residuals, and timing. The server
can use relative motion only between rigid, bounded poses in the same segment.
It independently checks captured observations before fusion. A pose hint cannot
authorize a frame, bypass loss recovery, or join disconnected camera chains.
Intermediate camera images are not uploaded or saved, so exported sessions
retain motion summaries but cannot re-run the intermediate optical flow.

The server attempts synchronized ORB/depth registration against a bounded bank
of up to forty accepted views: the recent eight, the initial five, and up to
twenty-seven spaced historical landmarks. Descriptor matches rank the bank;
only the best five candidates undergo geometric registration. Stable landmarks
avoid rebuilding a uniformly resampled cache on every capture. A view can
match an earlier measured keyframe during ordinary
tracking or recovery. ICP may refine the measured feature proposal, but the
result must still satisfy the original pixel and 3D correspondences. If geometry
slides off that evidence, the observed proposal is checked directly against
bidirectional raw depth instead. Distributed visual support can constrain a
textured plane; an anonymous or blank plane still cannot authorize motion.
The dense RGB-D fallback also rejects model refinements over 3 cm or 3° away
from its color seed. Ordinary geometric paths retain their existing gates.

Successful visual links are recorded immediately as `tracking_edges` and in
each frame's `visual_evidence`. This is online keyframe association, not online
global graph optimization. Full fragment graph optimization and fresh final
fusion still run at Finish. Model extraction follows its scheduled cadence
instead of being forced before every color-assisted frame.

Recovery also checks up to five recent accepted raw depth observations instead
of depending on the last one alone. Wider appearance relocalization and Finish
use scale-tolerant SIFT features; ordinary live matching keeps the cheaper ORB
features. Descriptor similarity proposes a connection, which still needs the
existing measured color/depth and reciprocal geometry checks.

## Capture flow diagnostics

Open **Tracking diagnostics** in the Scan sidebar and enable **Show tracking
flow**. This preference defaults off, is saved locally, and can be toggled
during capture. The scan must have **Show live reconstruction** and
**Color-assisted tracking** enabled. The diagnostic preview uses the actual
calibrated RGB image on the 640 × 480 depth grid; native RGB has a different
resolution, lens model, and parallax, so scaling these coordinates onto it
would misplace the arrows. Color and Scan views show the overlay; the Depth
view keeps its depth display.

Arrows run from the reference location to the current image location. Green
marks correspondences consistent with verified camera motion; cyan marks
newly added corners. Red crosses mark failed optical
flow, orange marks forward/backward disagreement, purple marks unavailable or
unstable measured depth, and yellow marks geometry/pose rejection. A seeded
reference explicitly says that no motion has been measured yet. This local
verification still does not authorize server fusion.

Fading green trails show feature paths across the **last 20 incoming camera
frames**, including the current frame. They follow persistent feature IDs;
newly replenished corners start new paths. Older segments fade behind the
current flow arrows and rejection markers. Paths grow as observations arrive,
so new or short-lived features have shorter trails. Missing/rejected
observations count toward the 20-frame window and leave gaps in a path;
recovery does not invent intermediate positions. Recent retired paths fade
out as their observations leave the window.

The status shows the collected trail frames, up to 20. Trail history starts
when visualization is enabled and clears when it is disabled or the tracking
chain resets. It stores only IDs and pixel positions while diagnostics are
enabled. The motion estimator still uses its five-reference bank, and trail
arrays stay inside the local diagnostic snapshot rather than capture reports,
recordings or uploads. Positions remain on the calibrated depth grid;
portrait display rotates the complete overlay afterward.

The [trail display check](benchmarks/twenty-frame-flow-trails.json) compares
60 synthetic camera observations with the previous diagnostic implementation.
Measured motion counts and pose matrices are unchanged. With 500 surviving
features, the 20-frame buffer holds 10,000 positions and renders up to 9,500
segments. In that dense case, median tracker update time increases from 5.93
to 6.83 ms, and median overlay rendering from 6.48 to 20.43 ms on the local
ARM Mac. These measurements exclude physical acquisition, native color
registration and the rest of the UI; trails incur this extra display cost
only when visualization is enabled.

The status shows the rejection counts, detected corner count, camera tracking
time, and reference age. The reference may be an earlier good image from the
five-view history, so arrows are not necessarily adjacent-frame motion. When
all attempts fail, the overlay describes the last attempted eligible reference.
Timing failures clear the overlay and display the failure reason.
**Show LK patch windows** outlines up to 24 evenly sampled feature windows to
keep the display readable. These are 21 × 21 pixel windows at pyramid level
zero, not a fixed search boundary.

The field retains up to 500 corners with persistent IDs, birth timestamps, and
successful observation counts. Verified survivors keep their subpixel locations
and measured depth in each new reference. The five-reference fallback can
recover those same identities from an earlier good image. Failed observations
within a current chain never replace the field, allocate IDs, or trigger
replenishment. A chain reset
starts fresh lifetimes; IDs are local to their segment, not world landmarks.

Replenishment runs after a verified motion step when fewer than 400 reliable
tracks remain, or an eligible cell in the 8 × 6 image grid has fewer than three
tracks. It keeps existing survivors, masks at least nine pixels around
them, and selects new measured-depth corners from the least populated cells
first, within the 500-track budget. Cells with insufficient included depth are
excluded from coverage checks. The detector can inspect up to 2,000 candidates
with corner quality 0.015 and nine-pixel spacing, but retains at most 500 tracks.
Healthy fields skip detection. Ordinary top-ups are spaced by at least 200 ms;
fewer than 60 surviving tracks bypass that delay. A new origin requires at
least 60 corners with stable measured depth. An unusable replacement reference
is discarded while earlier references remain available.

Maintaining an identity uses a stricter quality check than authorizing one
short motion step: retained tracks must have pixel residual at most 1.5 pixels
and paired 3D residual at most 30 mm. Tracks with weaker evidence are retired
and can be replaced. This limits accumulation of uncertain subpixel locations.
These are engineering bounds, not a calibrated sensor uncertainty model. The
existing motion acceptance, forward/backward, depth, and server verification
gates remain in effect. Lucas–Kanade still uses 21 × 21 windows at pyramid
levels 0–3. No camera image history beyond the five-reference bank is added.

Diagnostics show tracks kept/new/retired for quality, median/oldest track age,
grid coverage, and why replenishment ran or was skipped. Only newly added
corners are cyan. Green flow can support the current pose yet fail the stricter
lifetime check, so the green count can exceed the retained count. On a failed
frame, ages and active counts describe the last retained reference. Scalar
field summaries are included in capture metadata; per-feature IDs and ages
remain local preview data. Algorithm parameters currently use code defaults;
the sidebar controls visualization.

Debug arrays and the calibrated preview travel only with their matching
camera pair to the UI and are removed before manual/automatic capture
selection, upload, or recording. Drawing works on a copy. Toggling diagnostics
does not reset the motion chain, change acceptance thresholds, or restart the
camera. The cost of snapshot collection and overlay drawing applies only
while diagnostics are enabled. Physical Kinect capture with this overlay
still requires validation.

The adaptive-field tests cover identity survival across twelve updates,
count/coverage top-ups, minimum spacing, bounded detection cadence, quality
retirement, older-reference recovery, timing/depth/blank-frame rejection,
segment resets, and manual/automatic capture isolation. The existing noisy
raycast sequence still passes its metric motion checks: 9.31 mm translation
RMS, 8.95 mm final translation error, and 1.20° final rotation error over
twelve observations. Repeating with five PnP random seeds gives the same result.
This tests a synthetic camera sequence, not physical scanning accuracy.

The interleaved [compute benchmark](benchmarks/adaptive-feature-tracking.json)
compares thirty-frame textured-plane sequences over three repeats. Whole-update
median time changes from 9.83 to 7.42 ms, and from 9.21 to 7.92 ms with 20%
depth holes. Both versions accept all thirty observations; the feature support
sets differ by design. Synthetic translation RMS stays below 0.05 mm and maximum
rotation error below 0.006° on those idealized planes. These times exclude
acquisition, upload, server processing, and drawing; they do not establish
Kinect throughput or real-world pose accuracy.

## Adaptive tracking on archived captures

The [archived evaluation](benchmarks/archived-adaptive-feature-tracking.json)
compares the adaptive camera tracker with its immediate predecessor, `fdf144b`,
using every raw RGB/depth pair in four unchanged session ZIPs: 540 captures.
Calibration and depth filtering are held constant, and saved poses never
initialize either tracker. Two complete runs agree on motion counts, detector
calls, feature-count distributions, and input checksums. Timing below comes
from the second run on the same ARM Mac with four OpenCV threads, after warmup;
no validation tests ran concurrently with that run.

These archives contain selected captures, with gaps of 0.97–57.92 seconds,
and no intermediate camera streams. Every gap exceeds the camera tracker's
0.75-second reference lifetime. Replay with original timestamps produces
**zero measured motion steps in either version**. Bootstrap references are
counted separately and do not establish motion. The first two sessions also
have excessive RGB/depth skew: 45 of 50 and 183 of 200 captures exceed 20 ms.
All 126 Chest 3 and 164 Chest 4 pairs satisfy the timing guard. None of these
results measure a change in server pose verification or reconstruction quality.

A separate stationary diagnostic repeats each original raw pair for a seed
and five subsequent observations, with artificial 10 Hz timestamps and zero
skew. It preserves the recorded images, calibration and depth processing.
These are identical-image comparisons; they contain no real camera motion,
occlusion, blur, exposure changes or newly sampled sensor noise.

| Session | Raw captures | Median initial measured features, before → after | Median stationary motion update, before → after |
| --- | ---: | ---: | ---: |
| Chest, Oct 5 | 50 | 397.5 → 500 | 17.54 → 16.35 ms |
| Chest 2, Oct 6 | 200 | 427 → 500 | 17.58 → 16.26 ms |
| Chest 3, Oct 6 | 126 | 439 → 500 | 17.58 → 16.22 ms |
| Chest 4, Oct 7 | 164 | 316.5 → 317 | 16.98 → 16.04 ms |

Both versions verify all 2,700 stationary motion steps. Every original adaptive
feature identity survives its five updates, reaching 0.5 seconds of diagnostic
age; the predecessor selects a fresh field every time. Corner-detector calls
fall from 3,240 to 1,124, a **65.3% reduction** including initialization. The
combined median motion-update time falls from 17.44 to 16.23 ms, **6.9% lower**.
The 90th percentile increases slightly, from 18.25 to 18.55 ms; replenishment
still costs work. Median new-reference seeding increases from 13.78 to 14.40 ms.
Static translation drift stays below 0.001 mm in both versions. That measures
numerical consistency on duplicated observations, not Kinect accuracy.

Chest 4 also stores 164 capture-selected live reports from the tracker used
during that earlier scan: 159 have measured support, five report chain loss,
and 27 distinct local segments appear. Median supported feature count is 279,
median reported pixel residual is 0.336 px, and median reported depth residual
is 1.71 mm. These are fitting residuals and sampled old-tracker diagnostics,
not independent pose truth or an evaluation of adaptive lifetimes.

Reproduce the selected-capture evaluation with:

```sh
OMP_NUM_THREADS=4 python scripts/evaluate_archived_visual_tracking.py \
  export/sessions/*.zip --baseline-revision fdf144b --stationary-steps 5 \
  --threads 4 --output benchmark-output/archived-tracking.json \
  --details-output benchmark-output/archived-tracking.jsonl
```

The evaluator reads selected captures even if a newer archive includes sensor
streams. To measure moving-camera acceptance, recovery and feature lifetimes,
make a short new scan with **Record all camera frames (large files)** enabled
and replay its complete stream. The four evaluated archives cannot supply
that missing evidence.

## Sharp capture selection and recovery pacing

Automatic capture chooses from the last five incoming RGB-D pairs, limited to
300 ms of age and captured after the preceding upload. It prefers a valid
camera motion estimate, then the highest grayscale Laplacian variance at a
fixed scoring resolution, with the newest image breaking ties. RGB, measured
depth, timestamps and motion metadata always come from the same candidate.
This is a relative sharpness score, not a guarantee that the selected image is
sharp. Manual capture continues to send the displayed latest image.

While fusion is paused, automatic capture checks each arriving image and sends
the next fresh selected pair as soon as the previous recovery check finishes,
with a 100 ms minimum spacing. It bypasses the ordinary capture interval and
learned processing delay, but allows only one outstanding recovery probe.
Unused candidates are discarded; frames actually sent to the server remain in
the session for later verification. Successful recovery resumes normal pacing.

The approach uses OpenCV's [pyramidal optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
and [PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html). The residual
scales and acceptance thresholds are engineering bounds, not a measured
per-device noise model. Repeated texture, moving subjects, occlusion, and
long-term drift remain limitations.

## Finish and the preview

The fragment pass now retains measured feature constraints during adjacent
registration. It tries the previous five prepared views, so a bad capture need
not split the next good view from an earlier reference. Nonadjacent local links
allow the same per-capture motion budget over at most three capture steps; the
reciprocal, visual and held-out depth checks are unchanged. Fragments retain
their first five, last five and middle views as retrieval witnesses, plus
measured boundary context, to preserve short overlap arcs. A planar fragment
bridge can pass only when at least two distinct
camera positions on each side support the same transform with distributed
visual matches and independent held-out depth. Stationary duplicate images and
a single matching view remain insufficient. Optimized visual bridges must pass
both checks again before fresh fusion. Geometric bridges keep their existing
reciprocal and normal-diversity verification.

The combined revision also preserves measured neighboring-camera links across
the sixteen-view storage limit. A storage boundary no longer creates an
artificial tracking loss. If global optimization moves verified links beyond
their raw-data checks, its adjustment is rejected. The measured bridge poses
are rechecked, inconsistent surviving links are discarded, and connectivity
is recomputed from the first view. Pruned edges stay pruned. The report records
this fallback; it does not establish that accumulated drift has been corrected.
The report counts independent cycles in the surviving verified fragment graph,
after pruning and fallback. The final guidance displays that count together
with any applied refinement loops. A redundant edge that becomes the only
remaining connection does not count as a closed loop.

The live point cloud still includes low-weight tentative surface. Inspection
during capture uses weight 0.5; Finish uses the selected final confidence
threshold (2.0 in the chest archive), cleanup, and verified connectivity. The
panel now explains this persistently and reports how many captured views the
final model retains, including previously fused views excluded by verification.
No confidence or geometric gate was weakened simply to increase face count.

## Validation

A raycast sequence with noisy measured depth and known moving-camera poses tests
all intermediate camera frames while the server receives only the first and
last. The local motion estimate ends within 9 mm / 1.25° over an 11-step path;
the server independently verifies sparse fusion. Textured planar tests check
metric translation, rejection of an apparently perfect but sliding geometric
fit, earlier-keyframe recovery, and final fragment reconnection using two
independent visual/depth witnesses. Blank texture, inconsistent depth, timing
failure, invalid hints, stationary bridge evidence, and acquisition error
containment are also covered.

The Chest 3 replay uses the original 126 raw observations, calibration, timing
guards, and confidence settings. New live tracking accepted 104 captures versus
68 archived, with nine loss episodes versus fifteen. Eighty-two captures used
verified visual keyframe links, including six links to a view earlier than
the last accepted view. Median replay processing was 1.45 s versus 2.47 s recorded;
this is not a controlled throughput benchmark. More accepted views alone do
not establish absolute pose accuracy. The archive does not contain intermediate
camera frames, so it cannot validate the camera-rate path on the physical scan.

A compute-only test repeats the first archived native 1280 × 1024 pair with
artificial 10 fps timestamps: 18 measured motion updates take a median 43 ms
and at most 58 ms. This includes calibration, filtering, flow, and pose fitting;
it does not validate real acquisition throughput or physical moving-camera
accuracy. The full suite on the combined revision ran 283 tests (281 passed,
two CUDA skips), followed by 12 session-lifetime checks after the final client
reconnection change. The separate synthetic HTTP/WebSocket and Qt
capture/preview/Finish/export workflow passed.

Final replay outputs and complete raw-evidence reports are preserved outside Git
in `/Users/sasha/hobby/xbox360/chest-3-session-analysis/continuous-tracking/`.
The original archive is unchanged. A new physical capture with the running
client/server changes, and CUDA execution, remain unverified.

### Where live tracking time goes

A CPU profile of `export/chest-3-scan-session_20261006_215208.zip` (first 40
frames, `OMP_NUM_THREADS=4`, legacy tracking backend) puts the live tracking
budget in point-to-plane ICP: `registration_icp` took 50.1 s of 82.9 s accounted
self time across 444 calls, about 11 per frame. Descriptor matching is far
smaller — about 3.2 ms per keyframe pair, so roughly 127 ms per frame for a full
40-view bank. It is not the bottleneck.

Most of those calls are verification rather than tracking. `refinement._match`
supplied 285 of the 444 calls (64%), because `_recover_anchor`, `_relocalize` and
`_verify_tracking_transition` each run a forward and a reverse match.
`_verify_tracking_transition` now skips the reverse match when the forward match
is already untrustworthy: its acceptance test is a conjunction, so the reverse
result cannot change an already-failed verdict. On those 40 frames this removes
exactly one `_match` (three ICP calls). The saving is deterministic and provably
outcome-identical, but small — do not expect a visible throughput change from it.

The visual keyframe overlap checks in `_visual_register_primary` use
`reg_voxel * 3` instead of a hard-coded 0.0225 m radius. At the default preset
`reg_voxel` is 0.0075, so the value is bit-identical and nothing changes; for a
coarser preset the literal was narrower than the corridor ICP converges in, which
scored feature and reverse-overlap evidence against the wrong distance.

Two results worth keeping, both measured on the same 40 frames:

- **Loosening ICP convergence makes tracking worse.** Open3D defaults to
  `relative_fitness` and `relative_rmse` of 1e-6, so a level rarely converges
  before its iteration budget runs out, which looks like waste. Setting both to
  1e-5 in `engine._icp` and `refinement._match` dropped accepted frames from 28/40
  to 15/40 and raised live time from 84.7 s to 106.8 s. These solves genuinely use
  their iterations, and the shorter budget only sends frames into recovery.
- **Live replay is not bit-reproducible on this machine.** Two runs of identical
  code produced 29 784 and 29 767 surface vertices, different surface hashes, and
  live times differing by about 2%. Treat the accepted index set, the
  tracking-method histogram and ICP call counts as the stable signals; neither a
  single timing run nor a surface hash is an equivalence test for a replay change.

### Recovery and relocalization cost

Attributing every ICP call to the engine method that requested it, on the same
40 frames, shows where recovery effort actually goes (405 calls, 42.4 s):

| Requesting method | ICP calls | Seconds | Average |
|---|---|---|---|
| `_recover_anchor` → `_match` | 153 | 27.76 s | 181 ms |
| `_visual_register_primary` → `_icp` | 69 | 2.13 s | 31 ms |
| `_color_recovery` → `verify` → `_icp` | 54 | 1.84 s | 34 ms |
| `_relocalize` → `_match` | 48 | 6.27 s | 131 ms |
| `_verify_tracking_transition` → `_match` | 45 | 2.67 s | 59 ms |
| `_register` → `_icp` | 24 | 0.91 s | 38 ms |
| `_relocalize` → `_icp` | 6 | 0.40 s | 67 ms |
| `_fpfh_fallback` → `_icp` | 3 | 0.15 s | 50 ms |
| `_recover_anchor` → `_icp` | 3 | 0.25 s | 83 ms |

`_recover_anchor` is **65% of all ICP time**, and on this session it never
succeeded once: no `anchor+icp` frame appears in the method histogram, so all 51
forward matches failed `_trustworthy`. Cutting that cost is the largest remaining
opportunity, but any change trades recovery recall, and this session cannot show
what recall would be lost because nothing recovers here. It needs a capture where
anchor recovery actually fires before it can be tuned.

Two safe reductions were applied on top of that measurement:

- `_relocalize` now skips the reverse match when the forward match is already
  untrustworthy, on the same conjunction argument used for transition
  verification. That removed 36 ICP calls and 3.1 s of the 82.7 s live replay,
  with an identical accepted set and method histogram.
- Recovery prepares each accepted anchor's registration cloud once instead of on
  every lost frame. The raw frame behind an accepted index never changes, so this
  is pure memoization; it is worth a further 1.1 s and about 15 MB.

Together those took the same replay from 82.65 s to 78.40 s (median of three
runs each, disjoint ranges) and from 441 to 405 ICP calls.

Anchor verification can only succeed while the camera is still near an accepted
view, so it is now **probed through the start of a loss episode and sampled
after that** instead of being repeated on every frame of a long outage
(`ANCHOR_PROBE_FRAMES = 3`, `ANCHOR_PROBE_INTERVAL = 4`). The tail of an outage
is where this matters most:

| Session | Frames | Live time before | Live time after | Accepted |
|---|---|---|---|---|
| chest-3 | 40 | 78.40 s | **73.63 s** | 28 → 28 |
| chest-8 | 80 | 84.24 s | **34.14 s** | 6 → 6 |

chest-3 also drops from 405 to 375 ICP calls. Both sessions kept an identical
accepted index set and tracking-method histogram, so the outage tail is sampled
rather than dropped.

### Chest 8 loses tracking at frame 6 and never recovers

`export/chest-8-scan-20261009-145-captures.zip` captures roughly **3.95 s apart**,
so the camera moves far more between frames than in chest-3. Current tracking
accepts **6 of the first 80 frames**, and the whole loss follows from one
decision:

```
[  5] accepted, fitness 0.699, RMSE 13.6 mm
[  6] verify gap=3.95s (limit 12.14) trans=0.299m ang=19.0deg -> REJECT
      forward/target  fit=0.650 rmse=9.9mm corr=14239  trustworthy
      reverse/source  fit=0.691 rmse=9.2mm corr=12917  trustworthy
      cycle     13.3mm 0.61deg (limit 10mm 2deg) -> FAIL
      agreement 43.0mm 2.38deg (limit 30mm 3deg) -> FAIL
```

Both reciprocal matches are individually trustworthy, so the rejection comes from
the two numeric bounds — and **both of them bind**. Frames 1 and 2, at 11.0° and
14.6°, closed at 1.6 mm and 3.2 mm and agreed within 10.3 mm and 10.5 mm, so they
passed the same gates.

Earlier notes here blamed the cycle-consistency gate alone. That was wrong: the
probe that produced those numbers checked the reverse match against `target`
instead of `source`, so the reciprocal closure it computed was not the engine's.
The corrected probe above shows the 43 mm agreement correction is the larger
violation.

After that, `_recover_anchor` runs on every subsequent frame, never succeeds — no
`anchor+icp` frame is ever produced, and appearance relocalization finds no
candidates — so 74 frames are spent failing. Two changes were measured against
this and one was reverted:

- **Anchor probing (kept).** Sampling the outage tail cut this session from
  84.24 s to 34.14 s with identical results.
- **Seeding older anchors from tracked motion instead of identity (reverted).**
  Using `inv(anchor_pose) @ last_accepted_pose` as the ICP seed is more
  informed, and reduces to identity for the newest anchor, but it recovered
  nothing: 6 accepted either way, 33.62 s against 34.14 s. No measured benefit on
  a safety-critical path, so it was not kept.

#### The two transition bounds were widened, measured, and left alone

The bounds were promoted to named class constants so an A/B could move them
together, and the widened configuration scaled agreement with the step being
verified (`max(0.03, 0.2 * translation)`) and widened cycle closure to 0.03 m /
3° — the corridor `_match` already searches, which a 19° rotation can plausibly
resolve. Across all eight archived sessions:

| Configuration | chest-3 | chest-4 | chest-5 | chest-6 | chest-8 | scan-912b6d41 | full-room | with-markers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current bounds | 28 | 34 | 5 | 33 | 6 | 3 | 13 | 40 |
| Widened bounds | 28 | 34 | 5 | 33 | **18** | 3 | 13 | 40 |

Only chest-8 changes. Widening either bound on its own is a no-op there — the
agreement-scaled run and the cycle-widened run both stay at 6 accepted, because
the other bound still binds — so the two had to move together to change
anything.

**It does not deliver.** With `--finish`, chest-8 ends at the *same* six accepted
indices `[0, 1, 2, 3, 4, 5]` either way, with the same three verified bridges,
the same seven fragments, the same five corrected poses and the same one
connected fragment. Finish reports
`Reconnected 0 views; 5 poses corrected, 12 excluded`: every one of the twelve
newly tracked views is dropped again as a view whose position could not be
verified, and the mesh grows by 0.17% (60,816 → 60,921 vertices). The reconnection
was not budget-limited (`budget_limited: false`, 256 pairs, 32 fragments, 0
ambiguous pairs), so this is genuine absence of corroboration rather than a
truncated search.

Two further signals point the same way. The twelve new views include frames 9
and 12, accepted at overlap 0.572 and 0.516 — below the 0.6 floor the same module
applies in `_match_failure` and `_recover_anchor`. And live time rises from
18.92 s to 32.30 s, because twelve more frames are tracked instead of failing
fast. `tests/test_tracking_recovery.py` pins both bounds at their current values
with the measured rationale, so revisiting this takes a deliberate change rather
than a silent one.

chest-5 (5 of 27) and scan-912b6d41 (3 of 40) are **not** explained by either
bound: widening them changed nothing in those sessions either, and the condition
that rejects them remains undiagnosed.

Measured decisions, all "do not change this":

- **`KINECT_KEYFRAME_CACHE=on` is not a speedup here.** Enabling it left the ICP
  count identical at 405, moved the live median from 79.53 s to 80.83 s, widened
  the three-run spread from 0.39 s to 3.47 s, and raised peak RSS from about
  2060 MB to as much as 2308 MB. The default `off` is correct.
- **A cheaper first RGB-D odometry pass is not worth it.** `_color_recovery`
  succeeds about 10 times per 18 attempts. Trying a reduced `[16,8,4]` pyramid
  before the full `[40,20,10]` one breaks even at roughly a 29% cheap-pass
  success rate, and a cheap pass that converges but disagrees with ICP still pays
  a second ICP verification. Against the measured success rate the expected saving
  is around 1.4% of live time, for a real chance of losing a recovery. Not taken.
- **No existing backend knob beats the defaults.** Measured on chest-3 at 40
  frames against the 78.40 s / 405-call baseline: `KINECT_MODEL_REFRESH=lazy`
  was neutral (78.43 s, same accepted set and methods, tracking slightly worse);
  `KINECT_VISUAL_REFINEMENT=measured-fine` was slower (80.32 s) with no gain;
  `KINECT_LIVE_RECOVERY=deferred` accepted 1 frame instead of 28, because it is a
  mode change that hands recovery to Finish rather than an optimisation. The
  remaining cost is not sitting behind a configuration flag.

## Combined final replay result

The combined tracking and fragment-boundary fixes rebuild the archive with
**120 of 126 captured views**, compared with sixteen in the earlier final
replay. Nineteen rejected views are recovered, one hundred live poses corrected,
and three previously accepted views excluded. Missing captures are 38, 57, 81,
93, 104, and 121 (one based); the latter three were accepted during live replay.
All observations remain in the unchanged original ZIP.

| Final result | Earlier replay | Combined trial |
| --- | ---: | ---: |
| Retained views | 16 | 120 |
| Vertices | 52,722 | 200,081 |
| Triangles | 96,845 | 389,817 |
| Surface area | 0.826 m² | 3.315 m² |
| Final confidence threshold | 2.0 | 2.0 |
| Voxel size | 5 mm | 5 mm |

Thirteen fragments connect through twelve surviving measured bridges, including
three verified storage-boundary links. The search checks nineteen of sixty-four
candidate pairs. Global graph optimization passes its final raw-data checks;
the optimization fallback is not needed in this successful combined replay.
Additional pose refinement finds no further trustworthy loops. Fresh fragment
fusion uses 7,095 of the archive's 10,000-block budget and takes about 11.6 minutes
including verification. No separate finer voxel resolution was requested.

Matched oblique and overhead renders show a single chest with substantially
better lid and wall coverage. More captured floor also contributes to the
triangle increase; this is greater verified coverage, not a change in voxel
resolution or a measurement of absolute accuracy. The fixed-camera comparison
is `mesh-comparison.jpg`; the reconstructed mesh is `final.ply` in the artifact
directory above. The compact [benchmark record](benchmarks/chest-3-visual-tracking.json)
contains counts, exclusions, unchanged source checksum, and verification scope.

The later [Chest 4 recovery result](CHEST_4_TRACKING_RECOVERY.md) tests sharper
capture selection, recent-reference fallback and wider returning-loop retrieval
on the following scan.
