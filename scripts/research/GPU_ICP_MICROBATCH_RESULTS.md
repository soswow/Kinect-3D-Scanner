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
- [microbatch_icp.py](microbatch_icp.py): original mathematical operations with
  explicit independent streams, shared immutable pair buffers and bounded
  private lane state; retains the CPU solver boundary.
- [benchmark_icp_convergence_tradeoff.py](benchmark_icp_convergence_tradeoff.py):
  explicit changed-method CPU budget/epsilon experiments through original gates.
- [device_loop_icp.py](device_loop_icp.py) and
  [device_loop_control.cu](device_loop_control.cu): complete device iterations
  with explicit stream ownership, bounded buffers and blocked exception states.
- [benchmark_microbatch_pruned_nn.py](benchmark_microbatch_pruned_nn.py),
  [benchmark_microbatch_warp_nn.py](benchmark_microbatch_warp_nn.py), and
  [benchmark_microbatch_warp_pruned_nn.py](benchmark_microbatch_warp_pruned_nn.py):
  reproduce the three raw-neighbour experiments, including negative results.

All numerical runs require exclusive hardware outside active scanning. Original
CPU gates, actual proposal/graph outcomes and Final pose/surface comparisons are
required before a component experiment can support a whole-scanner change.

The detailed independent-seed report, cold setup costs and publication command
are in [GPU_ICP_SEED_BATCH_RESULTS.md](GPU_ICP_SEED_BATCH_RESULTS.md).
