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

Private reports are in `benchmark-output/field-cuda-study/`, including
`gpu-icp-pruned-nn-v1`, `gpu-icp-warp-nn-v1`,
`gpu-icp-warp-pruned-nn-v1`, `gpu-icp-microbatch-v2`, and
`gpu-icp-device-loop-v1`. Their source hashes and runtime scopes remain distinct.

## Reusable entry points

- [gpu_icp_experiment_capture.py](gpu_icp_experiment_capture.py): derive genuine
  local pair fixtures from a closed current-source unseeded raw replay.
- [gpu_icp_experiment_loop_probe.py](gpu_icp_experiment_loop_probe.py): compare
  actual-input original CPU ICP with exhaustive-audited GPU steps and graphs.
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
