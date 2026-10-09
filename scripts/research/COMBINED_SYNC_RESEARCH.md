# Original flat NN and normal terms with one synchronization

Status: the fresh synthetic and complete original nine-proposal audits passed,
including all 104,123,989 query rows, exact nearest/distance/filter bits and
common 30-term equation bits, plus all 236 original CPU ICP result shadows.
The separate three-round timing also passed every original bridge gate. It
initially measured a regression: median native CPU 11.4178 seconds, original
Device resident 7.6794 seconds and combined single-copy resident 9.4858 seconds.
No whole-Finish or production registration authority is created.

A subsequent separately pinned causal experiment found repeated host AST source
checks caused material overhead. Both variants retained identical CUDA math,
configuration guards and nested clocks. Full original source hashing plus the
unchanged cached contract and immutable generator state reduced component wall
from 9.51528 seconds to 7.17259 seconds. The latter's contemporary native CPU and
Device controls were 11.13006 and 7.61491 seconds. This is a 1.55 CPU-to-GPU
component ratio and a 5.8% advantage over that Device control. Full details and
scope limits are in [COMBINED_SOURCE_GUARD.md](COMBINED_SOURCE_GUARD.md). The two
mode runs were successive, with three rotating native/Device/combined rounds
each. This modest component result establishes no complete Finish speedup;
the original measured sources and failed field trajectory authorities remain
unchanged.

`research_combined_sync_icp.py` inherits the existing resident transform, CPU
Eigen solve, pose multiplication, stage order, strict radius, convergence and
final correspondence handling. The first experiment keeps the original selected
device and null stream. It adds uniform guards to the existing normal/collapse
kernel signatures; reversing the four source edits recovers the original GPU
source bytes exactly. Both new kernels retain `--fmad=false`.

The original raw flat NN and original classifier enqueue first. Provisional
classification requests no audit packets, while keeping the original ambiguity,
boundary, unsupported and miss policies. The following equation kernels return
without reading candidates whenever any CPU-resolution flag or malformed count
exists. Guarded collapse writes zeros in that case. A single 320-byte host copy
then receives ten uint64 counters and thirty FP64 equation terms.

Malformed output is a hard failure. Flagged queries discard the zero placeholders,
use the unchanged original CPU resolver, and only then run the original equations.
Unflagged queries use the unchanged equation arithmetic and reduction order. This
first helper is audit-only: before the CPU solve, it additionally CPU-shadows all
queries and checks every common ID, squared-distance bit, filtered ID and equation
term bit against the original methods. The component producer also runs the
original native three-stage ICP on each actual call, requires equal correspondence
mapping and pose/fitness/RMSE deltas at most `1e-8`, and returns the identical
resident result. Shadow poses never replace a candidate pose.

The paired accepted/rejected fixture contains nine original proposals. The new
producer retains their original verification loop byte-independent AST exactly,
checks all original witness/pose/held-out/information gates and requires all
104,123,989 actual queries, 11,569 nearest calls, 10,861 pose iterations and 236
full original ICP result shadows. Archive poses authorize no new Live tracking.
The existing validated scalar/bulk CPU-auditor token permits only the CPU shadow
implementation; a new independent `CombinedSyncProofAuthority` records this
orchestration's evidence and cannot satisfy old Device or Finish token checks.
The separate registry-owned component timing authority validates both this new
proof and the original Device proof before omitting observational shadows. It
retains original CPU resolution of ambiguous rows and all original bridge gates.

Persistent fast buffers cost `20*N+320` bytes per match. The raw `40*N` output is
released after the combined copy completes and before the original audit starts.
The proof's conservative peak query-array bound is `156*N+400`, additional to
the original resident arrays and the separately bounded target cache. Both
configured query and resident byte caps apply; one million queries exceeds the
existing 136 MiB query cap and must reject that regime rather than hide the added
allocation. Host proof copies and allocator pools are outside these logical
array bounds. Buffers are released only after selected-stream completion; no
global CuPy pool is purged.

Run only in a root-allocated exclusive hardware slot. From the repository root,
with the existing native interpreter, pinned Eigen DLL, real local fixture and
closed current bulk auditor proof:

```powershell
python -s scripts/research/benchmark_combined_sync_icp.py `
  --run-allocated --synthetic-only --audit-nearest --audit-misses `
  --check-device-adapter --miss-policy direct-miss-research-v1 --repeats 1 `
  --bulk-audit-proof benchmark-output/field-cuda-study/bulk-nn-audit/old-trajectory-dual-proof.json `
  --output benchmark-output/field-cuda-study/combined-sync-v1/synthetic-proof.json

python -s scripts/research/benchmark_combined_sync_icp.py `
  --run-allocated --audit-nearest --audit-misses --check-device-adapter `
  --miss-policy direct-miss-research-v1 --repeats 1 `
  --bulk-audit-proof benchmark-output/field-cuda-study/bulk-nn-audit/old-trajectory-dual-proof.json `
  --output benchmark-output/field-cuda-study/combined-sync-v1/bridge-audit.json
```

Defaults are the unchanged chest-3 dynamic fixture/reference and pinned solver
under `benchmark-output/cuda-pipeline`. Every output must be fresh. The synthetic
phase exercises common hits/misses, a 129-row tail, duplicate ties, strict radius
boundary, unsupported coordinates, malformed output, byte caps and policy
mutation. Each focused test owns a separate cache; deliberate failures do not
contaminate real proof counters. The real phase additionally uses the ordinary
current NN/cache/classifier tests. Source/runtime/native binaries, compiler
contract, raw ZIP/fixture/proposals, actual device, original math/DLL and auditor
authority are checked before and after. Numerical faults are latched outside
ordinary graph rejection, and failed reports are preserved.

Use base Python without site packages for source contracts:

```powershell
python -S -m unittest tests.test_combined_sync_contract -v
```

Proof wall time contains repeated NN/equation CPU shadows, comparisons, extra
downloads and original ICP work. It is diagnostic. The unaudited primary copy
wait includes queued GPU work; the earlier `system_copy_s` cannot be treated as
removable memory transfer time. The separate matched timings above retain fresh
typed proof validation and complete original component gates.

A CUDA graph is a later possibility. The existing CPU Eigen/convergence and
exceptional-query resolver are host decisions, the existing methods use the
legacy null stream, and CuPy stream capture prohibits synchronous device-to-host
copies. Those constraints require a separate dispatch rather than wrapping the
current loop in capture. The single-copy baseline first measures whether reducing
one host synchronization matters. Primary references:
[CuPy 13.6 stream capture](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.cuda.Stream.html),
[CUDA 12.4 stream capture and graphs](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#cuda-graphs).
