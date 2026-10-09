# GPU ICP proposal batches and complete device iterations

These are executable research prototypes. The installed scanner still uses
legacy CPU ICP. No experiment here grants production or full-Finish authority.
The original [architecture](GPU_ICP_MICROBATCH_ARCHITECTURE.md) describes the
proposal-order, independent-witness and graph requirements for later integration.

## First completed experiments

On 9 October 2026, the RTX 3080 Ti executed three new exact-neighbour pilots:
centre-first cell pruning, four warp-sized queries per block, and their combination.
Each tested 30,734 rows: twelve historical real query batches used only as input
data, three fresh random clouds, and six focused tie/boundary/subnormal/domain
cases. Fresh exhaustive CUDA first/second IDs and distance bits agreed in every
case; original CPU hit/miss queries were audited again. Pruning reduced visits
but generally added more synchronization than it saved. The warp-only real-row
sum of per-case median lookup walls fell from 0.923 to 0.893 ms (about 3.3%).
This is a small sampled raw-lookup result, not whole ICP or scanning performance.
No neighbour variant is enabled in production.

Current-source fixtures were rebuilt from the raw calibrated views in sessions
6 and 7, using the previously measured fragment-local poses to reproduce
component inputs. Nine pairs contain thirty original deduplicated proposals.
Insufficient proposal counts are not padded to manufacture a four-lane test.
The first capture rejected a legitimate column-major seed; its failed report
was preserved and capture v2 records original layout and original value bytes.

The first complete-device probe ran two genuine session-6 proposals, with
8,203 source and 6,235 target points. All queries were checked against the
original CPU search. The step-by-step GPU lane and four-step CUDA graph produced
identical final pose bytes, correspondence IDs, fitness/RMSE and query/update
counts: 58/55 and 65/62 respectively. The original CPU ICP shadows differed by
at most 1.01e-15 m in translation, zero reported rotation, zero fitness and
1.04e-17 m in RMSE. Across both execution styles, 2,017,938 actual query rows
were CPU-audited with no ambiguous rows. These diagnostic walls include audits,
setup and graph capture; they do not establish an unaudited speed gain.

The owner-bound v2 lane then passed all four session-6 proposals. Both execution
styles audited 3,937,440 actual query rows in total. Graph and ordinary execution
again produced identical terminal results and counts; the maximum original-CPU
translation difference was 8.59e-16 m. This version binds original cloud/seed
owners and the declared chunk before authorizing a trajectory.

## Independent seeds with the CPU solver boundary

The first separately authorized unaudited timings used genuine two/four-seed
prefixes of one current pair each in sessions 6 and 7. Fresh audits checked
11,484,960 actual nearest rows. Every final original correspondence ID set,
normal-diversity gate, complete original bridge proposal and competing-proposal
ambiguity verdict passed. Reverse, independent-camera, held-out, visual and
information calls stayed on the original CPU path; only initial forward ICP
results were supplied by the batch adapter.

Three contemporary rounds rotated native CPU, serial GPU lanes and concurrent
GPU lanes. Median batch host walls in milliseconds were:

| Pair | Seeds | Native CPU | Serial GPU lanes | Concurrent GPU lanes |
|---|---:|---:|---:|---:|
| Session 6, 0/2 | 2 | 66.43 | 191.56 | 159.20 |
| Session 6, 0/2 | 4 | 130.30 | 295.82 | 255.33 |
| Session 7, 0/11 | 2 | 65.68 | 101.66 | 202.58 |
| Session 7, 0/11 | 4 | 131.98 | 337.17 | 276.42 |

The batch timer includes preparation, input/permit preflight, worker joins and
selected-stream completion. Helper construction/close and complete bridge
quality shadows are separate. GPU walls vary substantially across rounds;
individual favourable timings do not establish a gain. This adapter is slower
overall and is retained as a reproducible negative result. Its per-iteration
CPU Eigen solve and nearest-counter synchronization remain. Complete device
iterations and graph execution are the separately tested route to removing
those boundaries.

## Doing fewer iterations

A separate CPU experiment changed convergence explicitly while retaining every
original bridge check and all four genuine session-6 proposals. Three rotated
rounds measured the entire proposal verification work, with comparisons outside
the timers. The original median was 2.317 s. Raising fitness/RMSE epsilon to
1e-5 reduced it to 1.935 s but verified three proposals instead of four and
changed witness support. Epsilon 1e-4 took 1.652 s but moved an accepted bridge
by 19.28 mm and changed its validation scope. Halving iteration budgets took
2.147 s; short 12/8/6 budgets took 1.760 s. Both verified only two proposals
and moved accepted bridges by as much as 23.83 mm. These methods failed the
declared support/scope and 0.5 mm / 0.1 degree comparison. They are retained
as negative evidence, not a field speed setting. The device-loop experiments
retain the original convergence and iteration budgets.

## Complete device iteration timing

The distinct v3 protocol audited all four session-6 proposals and two genuine
proposals each from session-7 pairs 0/11 and 10/11. It checked 7,957,500 actual
query rows. Ordinary steps and four-step graphs matched every audited query
packet, terminal pose/correspondence bytes, metric and query/update count.
Fresh original CPU shadows had identical canonical correspondence IDs and
maximum pose-entry differences of 4.44e-16 and 8.74e-16 respectively.

Separate three-round timing omitted query audits under a fresh device-loop
permit and required exact agreement with those terminal references. Median
per-trajectory host walls, including fresh construction, target setup, capture,
execution, result validation and cleanup, were:

| Selected inputs | Native CPU | Ordinary device steps | Four-step CUDA graph |
|---|---:|---:|---:|
| Session 6, four proposals | 34.85 ms | 45.28 ms | 41.11 ms |
| Session 7, two pairs / four proposals | 34.69 ms | 47.42 ms | 44.50 ms |

The graph reduces repeated iteration/control-copy work, but rebuilding the
execution objects and target indexes absorbs that saving. Median construction
and target-start costs were about 14.9 + 7.3 ms for session 6 and 14.6 + 10.4 ms
for session 7. Median graph advance/control-sync walls were 14.7 and 13.9 ms;
capture is included there. These nested figures cannot be subtracted to claim
an unmeasured production speedup. Every trajectory constructs fresh arrays,
indexes and a fresh graph; later repeats warm only process compiler/allocator
caches. Result quality shadows are charged in the GPU all-in columns. Native
controls precede each seed's GPU variants, with variant order reversing on
alternate repeats. Process-level input/binary checks are outside these timers.
Persistent execution setup was subsequently measured through complete original
proposal verification, as described below.

## Persistent setup through complete proposals

The separate complete-bridge v1 experiment shares compiled execution setup and
immutable train-cloud grids across every alignment in a prepared pair. Each
alignment still owns fresh buffers, state and a four-step graph. All four genuine
proposals are exhausted through the original forward, reverse, camera, witness,
information and ambiguity checks; no proposal is padded or accepted early.

Fresh session-6 and session-7 audits covered 84 and 112 alignments and 28,766,412
and 39,011,598 actual nearest-neighbor rows respectively. Every canonical CPU
correspondence set, native result shadow, complete original gate, support set
and pair verdict passed. Both pairs accepted proposal zero with four verified
proposals and no ambiguity. These are prepared-pair checks, without adaptive
frontier, optimizer, fusion, mesh or whole-Finish authority.

Three separate timing processes per pair omitted only observational shadows
under their own fresh complete-bridge permit. Session 6's median native proposal
wall was 2.889 s; GPU cold setup, proposals and final owner cleanup took 5.909 s.
Session 7's corresponding medians were 4.296 s and 9.509 s.
The research harness remains slower despite setup reuse. Its clocks include
original gates, actual input/result checks, immutable-cache setup, graph capture,
selected-stream completion, metadata and progress publication. Final producer
closure and preflight are separately recorded. The
[session-6 scalar receipt](../../docs/benchmarks/field-study-gpu-icp-v1/complete-bridge-session6-summary.json)
and [session-7 scalar receipt](../../docs/benchmarks/field-study-gpu-icp-v1/complete-bridge-session7-summary.json)
contain every matched sample and the declared timing limits.

An explicitly instrumented Python-boundary run found 421 rereads of session 6's
7.47 MB audit report, plus repeated decoding of the same ordered references.
Those overlapping clocks are diagnostics, not speed measurements. The separately
scoped v2 method holds a Windows read-only file handle, binds its exact initial
and final bytes, and caches immutable ordered references. Every actual input,
terminal, original gate and ambiguity check still runs. Old permits cannot
authorize this new method; both pairs received fresh complete audits.

Three fresh timing processes per pair charged cold setup, all original proposal
checks and final GPU owner cleanup. Session 6 took median 2.863 s on CPU and
3.164 s on GPU (median matched ratio 1.105). Session 7 took 4.281 s on CPU and
4.366 s on GPU (matched ratio 1.010). Removing repeated research-report I/O
brought complete GPU verification close to CPU, but did not establish a speedup.
Each timing receipt closed its held handle after exactly two file reads with
identical hashes. The
[session-6 owned-proof receipt](../../docs/benchmarks/field-study-gpu-icp-v1/complete-bridge-owned-session6-summary.json)
and [session-7 receipt](../../docs/benchmarks/field-study-gpu-icp-v1/complete-bridge-owned-session7-summary.json)
record all samples and unchanged quality checks. This remains one prepared pair
per session, without whole-Finish or production backend authority. Old v1 reports
and failed/aborted profiling attempts remain private and unchanged; instrumented
samples are excluded from performance publication.

## Conservative float neighbor screening

A separate two-pass raw lookup keeps the original dyadic cell candidate set,
then screens candidates using float distances and a rigorously upward-rounded
error enclosure. The final first-two distinct neighbor IDs and distances are
computed with the original double expression, including all possible ties.
Unsupported or declined screening uses the full original double candidate scan.
Float arithmetic is only a conservative screening tool; it never supplies an
ICP correspondence distance or pose update.

The v2 physical pilot passed all five raw output columns bit-for-bit against the
original shader across 23 cases and 30,735 rows, with fresh original CPU hit,
miss, ambiguity and unsupported-query resolution. Cases include 12 historical
real query arrays used only as inputs, three fresh random clouds, and focused
duplicate, subnormal, float-rounding, cell-boundary, insufficient-neighbor,
declined-bound, edge-cell and unsupported-query inputs. All source, shadow,
input, loaded backend and selected-stream cleanup checks closed successfully.

The retained double evaluations fell to 0.44–20.69% of original candidate visits
on the real batches. Nevertheless, the sum of their twelve median kernel-enqueue
and selected-stream-completion walls increased from 0.8474 to 1.0644 ms. One real
coarse batch improved 1.158x; the random coarse batch improved 1.656x. Extra passes,
block reductions and per-query bound calculation offset the arithmetic saving.
These are preuploaded, preallocated raw lookup timers, with grid/shadow/query
setup and output copies separate; they establish no ICP or scanner speed gain.

The first v1 attempt failed before kernel launch because CuPy already appends
its FTZ compiler option. Its failed report and source snapshot are preserved.
The distinct v2 driver omits the duplicate option and binds the actual CuPy
compiler wrapper. The enclosure includes FTZ and gradual-underflow behavior.
Warp-level screening was subsequently measured using four queries per block,
warp reductions and a conservative bound computed once per case. Queries beyond
the declared coordinate cap use the full original double lookup. Two fresh runs
each passed all five raw output columns bit-for-bit and fresh CPU query checks
across 24 cases and 30,737 rows, including an explicit cap-overflow fallback.

The balanced twelve-repeat run rotated all three controls equally. Across the
twelve real batches, the sums of per-batch median host walls were 0.8557 ms for
the original 128-thread lookup, 0.8048 ms for an exhaustive warp lookup, and
0.7477 ms for warp screening: **12.6% less raw lookup wall time** than the original
and 7.1% less than exhaustive warp lookup. Corresponding GPU event totals were
0.693216, 0.650000 and 0.593216 ms. These timers include kernel enqueue and
selected-stream completion with preuploaded inputs; grid/shadow/query setup,
host-bound calculation and output copies are separately recorded. This gain
does not establish complete ICP, Finish or scanning FPS improvement. Production
lookup remains unchanged.

## Measuring opportunities to avoid complete alignments

A separate original-CPU census exhausted all 30 genuine proposals across nine
current session-6/7 pairs. It returned every original registration result and
ran all original proposal checks, with **714 calls and zero skipped calls**.
Exact full-precision cloud values and direction formed 292 cloud buckets.
There were no byte-identical repeated seeds. Comparing each unrounded 4x4 seed
directly to a fixed first representative found 172 entries within a maximum
matrix-entry difference of 1e-10 (24.1% of calls), or 184 within 1e-8.

For those census groups, every canonical correspondence set was identical;
fitness differences were zero, maximum transformation-entry differences were
6.31e-15, and maximum RMSE differences were 3.04e-16. This is an observation
under the original CPU trajectory, not proof that reusing a result is safe or
faster. A changed-method reuse trial must audit every actual hit against a fresh
original CPU call on its new trajectory, preserve all full proposal checks and
measure hashing, cache storage, result reconstruction and cleanup. The census
never rounds a seed or grants timing, whole-Finish or production authority.
The [compact census](../../docs/benchmarks/field-study-gpu-icp-v1/seed-reuse-census-summary.json)
contains all five thresholds and native-result maxima without private clouds,
seeds or per-call records.

### Executed bounded near-seed reuse trial

The changed-method CPU trial now executes the same nine fixed pairs and all
30 genuine proposals. It uses exact ordered point, normal and colour bytes and
direction as the cloud key. A seed can reuse only the first immutable result
whose unrounded matrix entries differ by at most 1e-10. There is no seed rounding
or transitive chaining. Each pair starts with a cold, bounded cache, and a hit
returns fresh writable result arrays to the original proposal verifier.

The fresh audit checked all 172 actual hits against a new original CPU call on
the changed trajectory. All canonical correspondence IDs and original complete
proposal, witness, support and ambiguity checks passed. Maximum audited
transformation-entry difference was 6.03e-15; fitness differences were zero and
maximum RMSE difference was 2.85e-16. Peak owned payload was 4,497,557 bytes;
cleanup released all owned payload.

The strict v1 timing attempt refused its first call, a **cache miss**, before
any reuse. The actual seed and canonical correspondence IDs matched the audit,
but an independently repeated original CPU solve differed by 1.29e-16 in a
transformation entry and 1.74e-18 in RMSE. This failed report is preserved. The
separate v2 experiment declares a 1e-12 comparison for repeated actual seeds,
first representatives and result values, while preserving exact cloud bytes,
call order, classes and canonical correspondence IDs. Every v2 actual hit has
its own fresh audit; no old proof unlocks it. This is empirical conformance of
these fixed trajectories, not a proof for every seed in a neighbourhood.

All three separately timed v2 rounds passed the own actual-input comparison,
independent full original-CPU controls and original final proposal checks.
Their cold complete-phase walls were:

| Round | Original CPU (s) | CPU with reuse (s) | Reuse / original |
|---|---:|---:|---:|
| 0 | 43.704 | 40.490 | 0.9265 |
| 1 | 43.871 | 40.795 | 0.9299 |
| 2 | 43.392 | 40.872 | 0.9419 |

The median matched ratio is 0.9299: **7.0% less complete-phase wall time**.
Median walls are 43.704 and 40.795 seconds. Each timed phase makes 542 original
CPU calls and reuses 172 of the original 714 results. The denominator includes
all nine pairs, not just the pairs that improve. The complete-phase clocks
charge cold setup, exact hashing, fragment reconstruction, lookup, copying,
all original gates, reference checks and cleanup. Imports, raw-resource
preflight and final producer closure are separate; these are not whole-Finish,
camera-FPS or field scanning measurements.

| Session / pair | Calls | Reused | Median original (s) | Median reuse (s) |
|---|---:|---:|---:|---:|
| 6 / 0,2 | 84 | 60 | 2.926 | 1.817 |
| 7 / 0,11 | 112 | 18 | 4.437 | 4.490 |
| 7 / 10,11 | 195 | 71 | 11.557 | 8.821 |
| 7 / 0,12 | 36 | 12 | 1.392 | 1.233 |
| 7 / 1,12 | 36 | 8 | 1.889 | 1.886 |
| 7 / 0,3 | 117 | 0 | 11.651 | 11.891 |
| 7 / 0,4 | 3 | 0 | 0.772 | 0.815 |
| 7 / 0,5 | 1 | 0 | 0.199 | 0.217 |
| 7 / 0,6 | 130 | 3 | 9.317 | 9.811 |

These inclusive pair timers are nested within the complete phase and must not
be subtracted to claim a gain. Reuse helps two expensive accepted pairs, but
the expensive rejected pairs have almost no reuse and pay cache overhead.
Production ICP remains unchanged.

The [compact reuse receipt](../../docs/benchmarks/field-study-gpu-icp-v1/near-seed-conformance-summary.json)
contains all three complete-phase samples, nested pair walls, actual numerical
maxima, source/report hashes and the preserved strict-v1 refusal. The scalar
publisher recomputes every recorded timing comparison and full gate check;
it does not independently rescan raw archives or claim a process exit.

Private reports are in `benchmark-output/field-cuda-study/`, including
`gpu-icp-pruned-nn-v1`, `gpu-icp-warp-nn-v1`,
`gpu-icp-warp-pruned-nn-v1`, `gpu-icp-microbatch-v2`, and
`gpu-icp-device-loop-v1`. Their source hashes and runtime scopes remain distinct.

## Reusable entry points

- [gpu_icp_experiment_capture.py](gpu_icp_experiment_capture.py): derive genuine
  local pair fixtures from a closed current-source unseeded raw replay.
- [gpu_icp_experiment_loop_probe.py](gpu_icp_experiment_loop_probe.py): compare
  actual-input original CPU ICP with exhaustive-audited GPU steps and graphs.
- [gpu_icp_experiment_driver.py](gpu_icp_experiment_driver.py) and
  [gpu_icp_experiment_protocol.py](gpu_icp_experiment_protocol.py): fresh
  independent-seed native/gate audits, exact registered timing scope and
  separately timed runs.
- [gpu_icp_device_loop_experiment.py](gpu_icp_device_loop_experiment.py) and
  [gpu_icp_device_loop_protocol.py](gpu_icp_device_loop_protocol.py): current
  complete-device query/terminal audits and separately authorized setup-charged
  graph timing; distinct from seed batching and whole-Finish authority.
- [microbatch_icp.py](microbatch_icp.py): original mathematical operations with
  explicit independent streams, shared immutable pair buffers and bounded
  private lane state; retains the CPU solver boundary.
- [benchmark_icp_convergence_tradeoff.py](benchmark_icp_convergence_tradeoff.py):
  explicit changed-method CPU budget/epsilon experiments through original gates.
- [device_loop_icp.py](device_loop_icp.py) and
  [device_loop_control.cu](device_loop_control.cu): complete device iterations
  with explicit stream ownership, bounded buffers and blocked exception states.
- [device_loop_workspace.py](device_loop_workspace.py),
  [microbatch_bridge_driver.py](microbatch_bridge_driver.py) and
  [microbatch_bridge_protocol.py](microbatch_bridge_protocol.py): separately
  scoped persistent setup through all original proposal calls, with fresh
  actual-input CPU/query audits before timing.
- [summarize_gpu_icp_complete_bridge.py](summarize_gpu_icp_complete_bridge.py):
  publish small scalar receipts from those closed audit and timing reports.
- [device_loop_owned_workspace.py](device_loop_owned_workspace.py),
  [microbatch_bridge_owned_driver.py](microbatch_bridge_owned_driver.py) and
  [microbatch_bridge_owned_protocol.py](microbatch_bridge_owned_protocol.py):
  separately audited execution with owned read-only proof handles and immutable
  indexed references.
- [summarize_gpu_icp_owned_bridge.py](summarize_gpu_icp_owned_bridge.py): publish
  scalar matched timings only after the new held-handle receipts close.
- [profile_complete_bridge.py](profile_complete_bridge.py): observational Python
  boundary attribution; its instrumented runs cannot establish a speed gain.
- [benchmark_flat_grid_filtered_nn.py](benchmark_flat_grid_filtered_nn.py),
  [research_flat_grid_filtered_nn.cu](research_flat_grid_filtered_nn.cu) and
  [flat_grid_filter_bound.py](flat_grid_filter_bound.py): conservative float
  candidate screening with exact original double outputs and fresh CPU audits.
- [benchmark_warp_flat_grid_filtered_nn.py](benchmark_warp_flat_grid_filtered_nn.py)
  and [research_warp_flat_grid_filtered_nn.cu](research_warp_flat_grid_filtered_nn.cu):
  exact-output warp screening with charged host bounds and full fallback.
- [benchmark_icp_seed_reuse_census.py](benchmark_icp_seed_reuse_census.py): original
  CPU call census and direct first-representative seed/result comparisons.
- [summarize_icp_seed_reuse_census.py](summarize_icp_seed_reuse_census.py): recompute
  and publish only closed original census scalar counts and result differences.
- [near_seed_icp.py](near_seed_icp.py) and
  [benchmark_near_seed_reuse.py](benchmark_near_seed_reuse.py): bounded changed-
  method CPU reuse, fresh original-CPU hit shadows and strict timing refusal.
- [benchmark_near_seed_conformance.py](benchmark_near_seed_conformance.py):
  separately scoped actual-input numerical comparison with complete original
  controls, gates and charged cold-phase timing.
- [summarize_near_seed_conformance.py](summarize_near_seed_conformance.py):
  publish fixed-pair scalar receipts after independently replaying all recorded
  actual-input comparisons; preserves the strict-v1 counterexample.
- [benchmark_microbatch_pruned_nn.py](benchmark_microbatch_pruned_nn.py),
  [benchmark_microbatch_warp_nn.py](benchmark_microbatch_warp_nn.py), and
  [benchmark_microbatch_warp_pruned_nn.py](benchmark_microbatch_warp_pruned_nn.py):
  reproduce the three raw-neighbour experiments, including negative results.

All numerical runs require exclusive hardware outside active scanning. Original
CPU gates, actual proposal/graph outcomes and Final pose/surface comparisons are
required before a component experiment can support a whole-scanner change.

The detailed independent-seed report, cold setup costs and publication command
are in [GPU_ICP_SEED_BATCH_RESULTS.md](GPU_ICP_SEED_BATCH_RESULTS.md).
