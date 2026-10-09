# Current-core GPU Finish: C5/C6 results and C7 qualification failure

The bridge-only GPU candidate reduced the observed C5 Finish median modestly,
but produced no useful C6 improvement. These are three matched repeats per
session, with fresh raw Live checkpoints and independent original native
controls. They establish a finite-session result, not a production/default
promotion or a scanner-wide speed claim. C7's exhaustive audit and three timing
pairs have now closed. Two C7 pairs fail the existing strict bridge/witness
membership comparison, so their timing median is not a qualified speed result.

The candidate replaces only registration calls inside the original bridge
verifier. Original proposals, independent witnesses, acceptance tests,
reconnection, non-bridge registration, refinement, Bundle and weighted automatic
Final allocation retain their current production paths and settings. Finish
walls include candidate qualification loading, context entry, target/template
setup, actual GPU trajectories, original verifier work and full selected-device
cleanup. Checkpoint materialization/export is separately recorded. Measure mode
omits exhaustive CPU NN and terminal shadows; separate audit runs perform them.

Each session's paired reports use the same current source, raw inputs, runtime,
settings and checkpoint. C5 records the e251 source inventory
`4de862eebc6485464d781e38390c4e8dbc55122401bb55d45587e945b4647792`;
C6 records the later 84114d2 inventory
`8d02d1e5deef671ccce31661f03b6abbece176347eae20446033aac61f32f7de`.
No cross-session/source speed ratio is computed. Original Open3D/OpenCV thread
counts are 20, OMP is 8, and the recorded GPU is an RTX 3080 Ti. Raw sessions supplied
no archived pose seeds. `KINECT_CUDA_CONFIDENCE=off` selects CPU confidence
processing; it does not disable the enabled `confidence_fusion` algorithm.

## Matched Finish walls

Seconds. Reduction is `100 * (1 - GPU/native)`; negative values mean more GPU
wall time. Run order rotates native/GPU, GPU/native, native/GPU.

| Session / pair | Native report | GPU report | Native | GPU | Paired reduction |
| --- | --- | --- | ---: | ---: | ---: |
| C5 /1 | `native-2.json` | `measure-1.json` |11.6944872|10.9647393|6.2401%|
| C5 /2 | `native-3.json` | `measure-2.json` |11.5527405|10.7844483|6.6503%|
| C5 /3 | `native-4.json` | `measure-3.json` |11.6524955|10.7877276|7.4213%|
| C6 /1 | `native-3.json` | `measure-2.json` |25.3214064|25.7713991|−1.7771%|
| C6 /2 | `native-4.json` | `measure-3.json` |25.5168143|25.8695915|−1.3825%|
| C6 /3 | `native-5.json` | `measure-4.json` |25.9905506|25.5672684|1.6286%|

For C5, native/GPU medians are 11.6524955/10.7877276 s. Their ratio is
**1.0801622**, or **7.4213% less wall time from the medians**. The **median paired
reduction is 6.6503%**; median paired speed ratio is 1.0712408. These are different
summaries and are not interchangeable.

For C6, native/GPU medians are 25.5168143/25.7713991 s. Their ratio is 0.9901214,
or **0.9977% more GPU wall time**. Median paired reduction is −1.3825%.
The mixed individual changes provide no useful improvement or robust gain.
Three repeats of one checkpoint per session do not establish statistical
significance or general field performance.

All twelve included workers record actual exit 0, parent wait, closed worker
tree, empty cleanup failures and successful sampled process isolation. The
monitor checks named Windows jobs and port 8000 at sampled intervals; it does
not attest every desktop/GPU process or activity between samples.

## Correctness evidence and exclusions

C5's matched runs retain the same 7 Final views from 27 raw captures/21 accepted
Live views. The three independent physical and temporal comparisons pass.
Fixed-coordinate surface p95 is 6.00685e−8 m, with precision/completeness 1 at 5 mm;
Final translation differences are 0 and the maximum reported rotation difference
is 1.2074e−6 degrees.

C6's matched runs retain the same 38 Final views from 38 raw captures/33 accepted
Live views. All three physical and temporal comparisons pass. Fixed-coordinate
surface p95 is 1.1920929e−7 m, precision/completeness 1 at 5 mm; maximum reported Final
translation difference is 2.68487e−14 m and rotation difference 2.95756e−6 degrees.
Surface comparisons sample 30,000 points per surface without fitting scale or
trajectory. These are observable coverage/pose/surface comparisons, not ordered
mesh-array byte identity, ground truth or equality of hidden optimizer histories.

C5 audit 0/1 separately pass 89/90 actual GPU calls and 49,390,419/49,988,154 actual
NN rows, with fresh original nearest checks and original CPU terminal shadows.
C6 audit 1 passes 84 calls and 28,821,277 rows: 21,964,785 audited hits and 6,856,492
audited misses, with 84 original CPU terminal shadows. Their physical/temporal
comparisons also pass. External monitoring observed foreign CPU-test activity
during these audits, so their walls are validation-only and excluded from
performance results even though their workers exited 0.

Preserved exclusions:

- C5 `native-0`: preflight refused a foreign named Python worker. No native
  benchmark was launched; no worker exit or newer tree-closure claim is invented.
- C5 `measure-0`: a passed unpaired preliminary measure, explicitly excluded.
- C6 `measure-1`: actual worker exit 0, but foreign Python samples broke isolation;
  excluded. Its paired-run replacements are the three rows above. Earlier
  native controls are not substituted into those pairs.

Published bridge/witness/frontier memberships and temporal diagnostic fields
agree within each accepted comparison. Hidden retained graphs and ordered native
gate histories are unavailable. Temporal bridge counts are explicitly 0 and
ambiguous temporal lists explicitly empty on C5/C6. Conditional rejected
optimized/fallback boundary fields are missing, not proof of no conflicts;
C6's optional per-edge temporal flags are also unreported. Imported
`pose_candidates` loaded bodies lack an independent body guard in the frozen
controller; recorded file/source/runtime/checkpoint closure remains in force.
No bitwise RANSAC/history replay or general-domain GPU authority is claimed.

## What the component records suggest

- C6's 84 measured GPU calls consume 1.797–1.848 s inclusive. Nested advance walls
  consume 1.195–1.268 s, including kernels and 1,103–1,109 blocking 256-byte control
  copies. There are no ambiguity packets/CPU NN corrections in these measures.
  Graph capture totals only 27.91–29.63 ms; lane close totals 5.46–6.54 ms. The
  earlier ~15 ms fresh-constructor figure cannot be projected onto this candidate,
  which already shares a pristine compiled template.
- Exact whole-call keys have no duplicates within any included C5/C6 Finish.
  Excluding seed, C6 has 21 identical geometry/criteria groups of four calls,
  potentially 63 graph/buffer reuses. Graph capture alone is small. Allocation,
  start-time shared-array verification downloads, hashes, terminal copies and
  authorization are not separately timed, so these records do not establish a
  worthwhile retained-lane gain. Nested timers are not added to inclusive walls
  or subtracted to project scanner speed.
- The separate cached-target CPU pilots pass 66 scalar NN parity cases at all
  five declared thread counts. C6 prepared one-thread warm lookups are
  0.6218/0.7377 ms for identity/original transform, but cold first queries cost
  22.6987/22.2682 ms and whole four-query trials 25.7763/25.6641 ms. They require
  target reuse to amortize setup and omit original registration metric reduction
  and transform preparation from warm search. The baseline C5 geometry differs;
  no cross-report speed ratio or Finish/FPS claim follows. See
  [cached-target results](CACHED_TARGET_GEOMETRY_RESULTS.md).
- Earlier independent-seed streams run separate loops and CPU boundaries,
  rather than vectorized GPU batch kernels, and often cost more than native.
  Their negative result does not justify a new complete two-job kernel rewrite
  without actual representative inputs and charged component measurements. See
  [seed-batch results](GPU_ICP_SEED_BATCH_RESULTS.md).

## C7: promising timing, unresolved membership differences

C7 uses the same 84114d2 core inventory as C6. Its original native control and
audited GPU Finish both retain 110 of 115 raw views. The exhaustive audit passes
2,213 actual GPU calls, 159,125 query batches and 1,610,255,296 actual NN rows:
785,640,169 audited hits and 824,615,126 audited misses. It also records one CPU
ambiguity-resolution row and 2,213 original CPU terminal shadows. Its independent
physical and temporal comparisons pass. The actual audit worker exits 0 with a
closed tree; foreign stdlib validation activity makes the supervisor exit 3.
The approximately 106-minute audit wall is excluded from performance results.

Three subsequent timing pairs use separate fresh processes, rotating run order.
All six timing workers and supervisors exit 0 with clean sampled isolation.
Measure mode collects no exhaustive CPU NN or complete terminal shadows. Each
measure still records one CPU ambiguity-resolution row. Seconds below include
the same whole-Finish costs described above.

| Pair | Native report | GPU report | Native | GPU | Existing strict quality guard |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `native-2.json` | `measure-1.json` |213.8167421|152.4320022|FAIL: bridge/witness membership|
| 2 | `native-3.json` | `measure-2.json` |243.8902041|165.8823942|PASS: physical and temporal|
| 3 | `native-4.json` | `measure-3.json` |225.6603381|155.4147505|FAIL: bridge/witness membership|

The raw timing medians are 225.6603381/155.4147505 s, an observed 31.1289%
reduction. **This mixed-quality cohort does not establish a qualified 31% gain.**
The one passing pair is insufficient to establish repeatable quality and speed.
The failed comparisons and their original controls remain preserved; controls
are not substituted and the existing comparator has not been relaxed.

A separate descriptive investigation finds one additional verified fragment
edge, 1→10, in native 2 and measure 3. Their independent supporting raw-view pairs
differ. The edge is finally marked `connected_to_scan=false`; all other extracted
bridge/frontier memberships match, and every run accepts the same 110
raw IDs. Native-only repetitions therefore also vary in this published field.
This does not establish the cause of the variation or prove that the extra edge
was unused: production code admits verified edges into reachability, spanning
tree and optimizer construction before assigning that final flag.

Descriptive fixed-coordinate surface checks compare all three native/GPU pairs
and three successive native-only repeats using the same 30,000-point, 5 mm
procedure. All six have p95 1.1920929e−7 m and precision/completeness 1. Paired
translation differences are at most 7.32606e−15 m and rotation differences at
most 4.67632e−6 degrees. These observations find no geometric damage in these
runs, but do not override the failed membership checks or reconstruct hidden
optimizer histories. Prospective graph observations remain separate research.

## CPU thread-count pilot

A subsequent fresh-raw C6 pilot changes only the explicit Open3D TBB cap,
keeping OpenCV at 20, OMP at 8 and the same saved settings/CUDA pipeline.
Each count runs in a separate monitored process with original CPU registration.
This is one pilot per count, in order 20, 1, 8, 2, 4; no replicated advantage or
physical pose/surface qualification is claimed.

| Open3D threads | Live processing (s) | Finish (s) | Combined processing (s) |
| ---: | ---: | ---: | ---: |
| 20 |11.9531734|25.0936600|37.0468334|
| 1 |30.4759033|122.9281671|153.4040704|
| 8 |11.4756252|29.6698773|41.1455025|
| 2 |15.5854892|51.9025841|67.4880733|
| 4 |12.2881058|35.7461315|48.0342373|

All five actual workers and supervisors exit 0 with clean sampled isolation,
closed trees and restored thread setters. Their source, raw ZIP, settings and
accepted indices agree: 33 Live and 38 Final views. Those checks establish
inventory closure and coverage, not independent geometric equivalence.
The current 20-thread setting has the lowest observed combined and Finish
walls. Lowering the cap supplies no useful improvement in this pilot. Eight
threads slightly reduce Live wall but increase Finish and combined wall.

The private wrapper records explicit thread getters at phase boundaries and
separately charges boundary/source checks in whole-worker wall. Its source
checks include named fragment/proposal functions and two anonymous default
factory slots. Generated dataclass method slots are not independently guarded;
the frozen wrapper's broader docstring phrase about their identity is not an
evidence claim. Earlier launcher/preflight failures are retained separately.
Evidence is under `native-tbb-threads-v2/chest6-nN-r0.*`; this experiment supplies
no GPU-method permission and changes no production default.

## Actual refinement-input pilot

The first five completed direct C6 refinement calls taking at least 40 ms were
captured prospectively from the unchanged original CPU path. Their observed
CPU walls were 55–87 ms. Input capture occurs outside that call clock; the
observer still adds tracing overhead, so those selection walls are descriptive.
The fixture owns exact point/normal/color arrays and unrounded seeds. Original
arrays remain read-only; replay makes independently owned writable copies for
the installed Open3D binding and checks copied/native/original bytes.

A separate fresh graph-4 audit passes all five original CPU terminal shadows
and 4,538,266 actual NN rows: 4,126,564 hits and 411,702 misses. It performs 246
actual queries and 231 updates. Three subsequent fresh native/GPU timing pairs
all exit 0 with clean external isolation, closed trees, matching their own
audited terminal results and query/update counts, and comparisons against
each separately fresh native control. These timings contain
no exhaustive NN oracle or complete CPU terminal shadow work.

| Pair | Native five-call inclusive sum (s) | GPU five-call inclusive sum (s) |
| --- | ---: | ---: |
| 1 |0.4633538|0.5556187|
| 2 |0.4632283|0.5927686|
| 3 |0.4735759|0.5756264|

All pairs run native then GPU; they are repeated component pilots, not rotated
whole-Finish experiments. Each sum charges cloud preparation, source/owner
boundaries, actual matching, GPU retrieval/lane/grid/graph setup where used,
terminal validation and selected cleanup. Separate producer wall also includes
common imports and reference validation. The GPU is slower in all three cold
five-call sums. Neither nested kernel/advance times nor removed validation costs
are used to infer a Finish speedup. This fixture contains five selected directed
calls and supplies no new-input, whole-Finish or production-method permission.

Evidence is under `84114d2-refinement-calls-v1/`, including the passed writable
copy v2 audit and `time-native-N`/`time-gpu-N` reports and monitor receipts. The
first read-only binding failure is preserved. Nsight attempts are separate
diagnostics: two early traces complete the mathematical calls but fail the
original Windows shutdown because Nsight leaves `sys.stderr` unset. Their
workers are correctly demoted to failed; neither trace supplies performance or
qualification evidence.

## Stopped research checkpoint: 9 October 2026

Research stops here at the user's request. The original 145-capture C8 ZIP
completed a fresh raw Live replay on the 84114d2 source: 103 views accepted,
without archived pose seeds. Actual worker and supervisor exits are 0;
sampled isolation, source/owner closure, thread/environment restoration and
worker-tree cleanup pass. The saved logical checkpoint has SHA256
`edec8111f2870e6e51270157f439e8b4547d68bf28d0acb3a8fbc2618cb3a2e2`.
This is capture evidence only. No C8 native Finish, prospective candidate-job
census, GPU audit, allocation experiment or quality comparison has run.

The reusable public helpers and scalar result notes are ready for publication.
Fixture-coupled refinement/census/graph-observer and Nsight preparation stays
under ignored local output with its exact sources and failed attempts retained.
The graph observer and census have source/mock checks but no numerical field
run. The replacement Nsight worker/launcher has source review with 25 tests
unrun; its trace reader is an unreviewed draft. No new experiment is queued.

Local continuation state is in
`benchmark-output/field-cuda-study/gpu-icp-candidate-v1/STOP_CHECKPOINT_20261009.md`.
Publishing the additive inventory and line-ending declarations changes source
metadata bound by older reports. Do not repin measured manifests or reuse old
proofs on the published checkout: use a fresh installed-source capture/audit or
the exact preserved source snapshot. These experiments change no scanner ICP,
fusion or thread default.

## Private evidence references

All local receipt paths below are relative to
`benchmark-output/field-cuda-study/gpu-icp-candidate-v1/`; raw images, clouds,
pose arrays and inventories remain private.

- C5: `e251-chest5/closure-c5.json`, SHA256
  `f16d59a3356ccbe88dc5735c19de872fa90708f464dd75e5a92b636252abbb3d`.
  It binds all three timing pairs, monitor receipts, physical/temporal comparisons
  and preserved exclusions. It replays saved scalar/pure-guard evidence without
  new numerical work or an authority mint.
- C6: `84114d2-chest6/pair-1-quality.json`, `pair-2-quality.json`,
  `pair-3-quality.json`, plus corresponding `pair-N-temporal-quality.json`.
  Their report references identify native 3/measure 2, native 4/measure 3 and
  native 5/measure 4 exactly. Current checkpoint SHA256 is
  `d8f626885c342c944513cf6230b870b8cbd4388ee21e0c386493e938f7f1d48a`.

- C7: `84114d2-chest7/audit-1.json`, SHA256
  `eaba501910dcc2b1ee24296a9c3ca763629e7f6d8a58e0c9f9d88cc82be4a076`;
  original audit quality comparison SHA256
  `64141ef2f3d24c2aca1ad2d635b56e70a2003b62b66af5dc82888ecb5f409888`.
  `quality-measure-1.json` and `quality-measure-3.json` retain the strict failures;
  `quality-measure-2.json` retains the passing comparison. The separate published
  membership diagnostic SHA256 is
  `cd061832e4e655f96c561e173e5dd6d223c176bd9a947f3cdcf76f83a910c625`.
  `diagnostic-surfaces-v1.json` is descriptive and has qualification disabled.
