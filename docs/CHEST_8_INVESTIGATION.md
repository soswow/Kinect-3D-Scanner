# Chest 8: final reconstruction introduces a half-turn

Investigated on 2026-10-09 against source commit `db0db2d`.
Source archive: `chest-8-scan-20261009-145-captures.zip` (145 captures).
The source ZIP was read without modification. No running server was contacted.

The saved final trajectory contains two large jumps introduced by fragment
reconnection. They bracket a middle section whose original accepted poses were
rotated approximately 174–177 degrees. Raw RGB-D measurements contradict that
placement. This is a reconstruction error, rather than an image-display rotation
or a camera-pose convention problem.

## Exact boundaries

Capture numbers below are **one-based**. JSON frame indices are zero-based.
Angles are computed from the relative rotation matrices, rather than Euler
angles, so these are actual changes of orientation rather than angle wrapping.

| Captures | JSON indices | Final rotation | Final displacement | Capture interval |
| --- | --- | ---: | ---: | ---: |
| 57 → 58 | 56 → 57 | 155.366° | 2.069 m | 4.016 s |
| 128 → 129 | 127 → 128 | 175.280° | 2.079 m | 3.919 s |

The first boundary is between fragments 8 and 9; the second is between fragments
15 and 16. Connected views in captures 58–128 belong to fragments 9, 10, 11, 13,
14 and 15. Their original accepted poses received approximately 174–177° world
corrections. Some captures within this interval remain excluded.
Fragment 8 received only 2.56–3.46° corrections; fragment 16 received 0.23–6.79°.

There is no original accepted pose for capture 58. Comparing the last accepted
capture 57 to capture 60 instead gives 16.622° / 0.296 m in the original live
trajectory, versus 161.654° / 2.096 m in the final trajectory.
At captures 128 → 129, original live motion was 10.915° / 0.107 m.

All 145 image-orientation records specify 0°. The final matrices have valid
proper rotations (maximum orthogonality error about 1.14e-14). Every final pose
equals `fragment_to_world @ camera_to_fragment` exactly in this archive. The
jumps therefore exist in reconstruction data, before rendering.

## Which graph decisions caused the placement

The archive records 103 original accepted poses and 134 final accepted poses:
35 recovered views, 97 corrected poses and four formerly accepted views excluded.
Fragment reconnection was applied. Final refinement was not applied because no
trustworthy loop constraints were found. Bundle adjustment was not applied
because 134 views exceeded its 128-view limit. Fresh final fusion then integrated
the reconnection poses. Neither subsequent refinement stage created these flips.

The important graph conflict is visible in `fragment_reconnection.verified_bridges`:

* **1 → 14:** a geometric-only bridge remains `connected_to_scan: true`. Its
  supporting zero-based camera pairs are `[18, 116]` and `[20, 106]`. It anchors
  fragment 14 in the half-turned placement. Bridge 9 → 14 and the other retained
  connections propagate this placement through the middle section.
* **15 → 16:** a bridge validated with visual features and held-out depth has
  five supporting pairs, including `[123, 128]` and `[127, 129]`. It ends with
  `connected_to_scan: false`. The final placement disagrees with this measured
  bridge by **179.889° / 0.833 m** in fragment-transform coordinates.
* **0 → 16** and **1 → 16:** retained geometric connections place the last
  section near its original orientation, creating the second discontinuity.

The archive reports `optimization_fallback: Revalidated measured bridge poses`
and `rejected_optimized_bridges: [[1, 2]]`. The fallback uses the previously
constructed spanning-tree placement and retains bridges that validate there.
This preserves the contradictory geometry-only placement and discards the
visual bridge across fragments 15 and 16. The archive has no global ambiguity
flag: `ambiguous_pairs` is empty.

## Fresh checks against raw observations

An isolated CPU audit decoded nine selected RGB-D observations and used their
saved native-depth calibration, current SIFT extraction, alternating held-out
point samples, and the production `_heldout` / `_visual_witness` checks. It also
re-estimated two local connections with `_local_match`. Saved live transforms
were seeds for those local fits; their acceptance still required raw evidence.
No volume was constructed and no server state was changed.

| Pair, one-based captures | Pose evaluated | Forward / reverse overlap | Forward / reverse RMSE | Visual witness |
| --- | --- | --- | --- | --- |
| 57 → 58 | Final saved pose | 0.205 / 0.189 | 16.80 / 16.92 mm | Fail |
| 57 → 60 | Final saved pose | 0.234 / 0.236 | 16.38 / 16.28 mm | Fail |
| 57 → 60 | Fresh local ICP/visual fit | 0.706 / 0.729 | 11.98 / 11.96 mm | Pass: 97 / 119 feature inliers |
| 128 → 129 | Final saved pose | 0.102 / 0.103 | 16.27 / 17.53 mm | Fail |
| 128 → 129 | Fresh measured local fit | 0.699 / 0.769 | 12.16 / 14.73 mm | Pass: 163 / 175 feature inliers |
| 19 → 117 | Final geometric bridge support | 0.671 / 0.712 | 11.49 / 11.36 mm | Fail |
| 21 → 107 | Final geometric bridge support | 0.556 / 0.564 | 13.13 / 12.97 mm | Fail |

The fresh measured local fit across captures 128 → 129 gives **10.958° / 0.111 m**.
The fresh local fit across captures 57 → 60 gives **16.622° / 0.296 m**.
Both pass the existing raw visual and held-out geometric checks.
The original, unrefitted 128 → 129 live pose narrowly fails held-out RMSE
(15.30 mm reverse), so treating all original poses as unquestioned truth would
also be inappropriate. The measured local fit resolves this small discrepancy.

The bad 1 → 14 bridge demonstrates why geometric verification alone is
insufficient here: its supporting depth surfaces meet the overlap and RMSE
thresholds in the incorrect placement, while visual identity does not verify it.
The box/chest and surrounding planar surfaces offer similar geometry from
different directions. This symmetry explanation is an inference from the raw
images and the reproduced geometric/visual disagreement; a complete ground-truth
trajectory was not measured.

The continuous client visual tracker also reports the same segment across both
boundaries. Its estimated rotations are 22.804° at captures 57 → 58 and 11.931°
at captures 128 → 129. These are corroborating measurements, not pose authority:
some unrefined client transforms fail the held-out depth thresholds.

## Relevant implementation and recommended correction

In `scanner_server/fragments.py`:

* `_verify_bridge` and `_verify_partial_bridge` can accept purely geometric
  witnesses without requiring visual identity. Multiple cameras and held-out
  depth can still validate a repeated shape at a false global orientation.
* Local fragment construction retains the first successful recent reference.
  A useful additional connection across an existing fragment boundary can be
  lost: the audit verified captures 57 → 60 directly even though captures 58
  and 59 had started fragment 9.
* Spanning-tree initialization establishes one world placement before graph
  optimization. Fallback then revalidates against that same placement and can
  reject a contradictory measured visual edge while preserving false geometry.
* Output validation checks surviving bridges; it does not independently
  revalidate all available short-range raw connections across output boundaries.

A correction should retain independently verified short-range RGB-D ties across
fragments, use them to detect and reject conflicting geometric loop matches,
and validate proposed output boundary motion against those measurements before
fusion. The graph fallback should not resolve a half-turn conflict merely by
keeping whichever spanning-tree placement was initialized first. Unresolved
components should remain excluded rather than be fused at the ambiguous pose.

Switching local refinement to measured RGB-D can recover the second boundary in
this audit. That isolated result does **not** establish that the setting alone
repairs the entire reconstruction. No production behavior was changed, no full
alternative Finish run was performed, and no repaired session is claimed.

## Local evidence and provenance

The ignored directory `benchmark-output/chest8-investigation/` contains
`audit.py`, `audit.json`, `audit.log`, `boundary-captures.jpg`, `plot.py` and
`pose-jumps.png`. Images and pose arrays were not added to Git.

Run the audit with the scanner's existing Python environment; it uses four
OpenMP/OpenCV threads. The tested environment was Open3D 0.20.0, OpenCV 5.0.0,
NumPy 2.5.3. `raw_local` uses the default ICP/visual path;
`raw_measured` uses `measured_first=True`. The archived Finish used
`final_local_refinement: icp` and `final_visual_first: off`.

Exact SHA256 fingerprints:

* `manifest.json`: `32027b8146332962632379a01cd675c2465c9e32a5dc8cf29718fb94c7208cb5`
* `reconstruction.json`: `90a2165ef7f4003b90dd0f943775db6f51b696e29f8a88af2afe923fe3445729`
* Local `audit.py`: `6d58b67b9ae484f6a6087e264e08760c2ad1e37f46bdd07a22e864eed26547a1`

This investigation documents an observed failure and targeted raw-data checks.
It does not certify the remaining poses or the resulting mesh as correct.

## Algorithm correction and rebuilt session

The subsequent production correction retains verified short-range camera ties
across fragment boundaries, constructs the spanning forest in temporal/visual
evidence order before anchoring it, and checks all measured output boundaries
even when optimization prunes their original edges. If fallback still violates
those measurements, Finish fails before fusion and preserves the previous
volume and diagnostic evidence. See [the algorithm guide](FRAGMENT_RECONNECTION.md).

Eight additional temporal bridges are found in chest 8, including the direct
zero-based `[56, 59]` and `[127, 128]` RGB-D connections reproduced above. Three
existing storage-boundary bridges remain. The resulting graph tests seven
global pairs rather than the archive's 88. Early completion follows the existing
connected-plus-redundant-cycle rule after these measurements establish coverage.
This comparison describes graph work; the archived CUDA timing and the fresh CPU
timing are not interchangeable performance benchmarks.

The corrected algorithm was checked in two isolated replays:

1. Original live poses as seeds: reconnects 143 of 145 observations.
2. Already-flipped final poses as seeds: full Finish succeeds with 143 accepted
   observations, fresh 5 mm fusion, mesh extraction and session export. None of
   the original 134 accepted observations is excluded. Nine additional views
   are recovered and 82 previously accepted poses are corrected. Raw indices
   25 and 42 remain unconnected.

| Captures, one-based | Archived final rotation / displacement | Rebuilt final rotation / displacement |
| --- | --- | --- |
| 57 → 58 | 155.366° / 2.069 m | 23.490° / 0.435 m |
| 128 → 129 | 175.280° / 2.079 m | 10.958° / 0.111 m |

The largest relative rotation between retained output captures is now 38.518°;
that transition spans a missing capture. All eleven retained temporal/storage
boundaries pass a fresh calibrated raw visual/held-out check with the current
source after export. The optional later pose-refinement and bundle-adjustment
stages did not apply, so the repaired trajectory is the reconnection result.
Final fusion activates 15,926 blocks at 5 mm with automatic capacity allocation.

The new archive is
`export/chest-8-scan-20261009-145-captures-temporal-fixed.zip`. It contains the
rebuilt mesh and all 145 raw RGB/depth captures. Every image retains the original
CRC and uncompressed size, and the entire new ZIP passes CRC verification.
The original archive remains intact. Local evidence is in the ignored
`benchmark-output/chest8-improved/` directory, including `boundary-validation.json`,
the complete rebuilt report, logs, mesh, and `pose-jumps-fixed.png`.

To reproduce a full isolated rebuild from the already-flipped estimates:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:KINECT_NATIVE = 'off'
$env:KINECT_CUDA_INPUT = 'off'
$env:KINECT_CUDA_REGISTRATION = 'cpu'
python scripts/reconnect_session.py export/chest-8-scan-20261009-145-captures.zip `
  --use-pose-seeds --device cpu --save-session `
  --output-dir benchmark-output/chest8-rebuild
```

The numerical environment was the same Open3D 0.20.0 / OpenCV 5.0.0 / NumPy
2.5.3 server environment used in the investigation, with four OpenMP/OpenCV
threads. A separate chest-6 replay retains all 38 captures, excludes none, and
has a largest relative rotation of 13.968°. Regression coverage includes 24
fragment tests, 31 additional appearance/fusion/feature checks, and 40 research
contracts; one optional client check is skipped. New regressions exercise raw
temporal recovery, conflicting measurements, timestamp pauses, unsynchronized
RGB-D, false geometric loop ordering, and failure before fusion with diagnostics
preserved.

The offline marker-insertion research adapter is explicitly rebound as policy
v2 to the changed production function. Its exact-source and insertion-removal
checks still pass. Historical v1 reports are unchanged and do not establish
numerical or timing authority for this new algorithm.

These results address the observed half-turn failure. They do not provide a
ground-truth trajectory or certify every surface in the rebuilt mesh.
