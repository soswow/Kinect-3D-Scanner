# Independent-seed microbatch result

The explicit-stream prototype preserved the original CPU correspondence and
proposal decisions on the two selected field pairs. It did not accelerate these
batches. This is a prepared-input component result; tracking, the adaptive
fragment frontier, Finish and the final mesh were not replayed by this experiment.

The current-production raw replays supplied one genuine four-seed group from
chest-6 and one from chest-7. Both two- and four-seed ordered prefixes were tested
serially and concurrently. Every actual nearest-neighbour query was audited
against the original Open3D CPU query before correction: 5,955,378 rows for
chest-6 and 5,529,582 for chest-7, with zero mismatches or whole-call fallbacks.
Each final ICP result also passed a fresh original CPU shadow, and the unchanged
reciprocal, independent camera, held-out, visual and information gates consumed
each scoped result. All genuine competing proposals were retained, and each
complete pair verdict agreed. No shorter iteration budget or looser convergence
criterion was used.

Three nonaudit timing rounds used the exact audited arrays, ordered seeds,
source/runtime and scheduling scope. Median enclosing driver-call walls are:

| Selected field pair | Seeds | Native CPU | Serial GPU with CPU solve | Concurrent GPU with CPU solve |
|---|---:|---:|---:|---:|
| chest-6 | 2 | 66.432 ms | 191.562 ms | 159.197 ms |
| chest-6 | 4 | 130.296 ms | 295.823 ms | 255.334 ms |
| chest-7 | 2 | 65.681 ms | 101.664 ms | 202.582 ms |
| chest-7 | 4 | 131.985 ms | 337.175 ms | 276.420 ms |

The GPU columns include permit/source checks, shared input preparation, target
cache misses, lane iterations, original CPU Eigen solves, result transfers and
stream/thread completion. The nested adapter timer is smaller; its difference
from the enclosing timer is reported separately. Cold unaudited helper
construction was 84.364 ms and 80.377 ms, outside these batch walls. Original
CPU result shadows and complete dependent proposal gates were checked outside
the timed batch. The target cache persisted, while shared source points and
normals were uploaded for each batch. Later observations were not uniformly
faster: the compact data retains first samples, all three samples and ranges.
GPU event timers were disabled; inherited zero-initialized event fields are
explicitly labelled uncollected in the compact summary.

Concurrency occasionally improved over the serial GPU prototype but did not beat
the original CPU controls. Three small rounds on two selected pairs are too
limited to establish a general throughput result. The separate complete-device
loop experiment targets repeated host synchronization; it requires its own
fresh query/result proof and timing authority. This microbatch proof cannot
authorize that method or a whole-Finish replacement.

The local compact output is
[research-summary.json](../../benchmark-output/field-cuda-study/gpu-icp-microbatch-v2/research-summary.json).
It contains relative proof references and hashes, scalar coverage, complete
timing samples, source/runtime binding hashes and owned GPU memory, without
private images, point/pose arrays, graph inventories or absolute raw paths.
The publisher verifies current helper bytes and matching measured before/after
resource manifests; it does not rescan raw ZIPs or binary files and does not
create an authority token. The original numerical producers performed those
resource hashes outside the timed batches. Actual process exit completion is
recorded by the root controller separately from report validation.

Regenerate a **fresh** compact publication with stdlib Python after the four
numerical reports close:

```powershell
F:/ProgramData/anaconda3/python.exe -S scripts/research/summarize_gpu_icp_microbatch.py `
  --case chest-6 benchmark-output/field-cuda-study/gpu-icp-microbatch-v2/chest-6-seed-audit.json benchmark-output/field-cuda-study/gpu-icp-microbatch-v2/chest-6-seed-timing.json `
  --case chest-7 benchmark-output/field-cuda-study/gpu-icp-microbatch-v2/chest-7-seed-audit.json benchmark-output/field-cuda-study/gpu-icp-microbatch-v2/chest-7-seed-timing.json `
  --output benchmark-output/field-cuda-study/gpu-icp-microbatch-v2/research-summary-new.json
```

The raw-derived fixtures are private ignored files. Recreating numerical proofs
requires the original ZIPs, calibrated/current unseeded replay profiles, the
same current core, original native/Eigen binaries and library/thread policy.
Changing or relocating source helpers changes their hashes and requires fresh
proofs. Historical or hand-constructed permits cannot bypass these checks.

The first capture's contiguous-layout assumption rejected an original
Fortran-layout seed after preparation; its failed report and source snapshot
remain in `gpu-icp-microbatch-v1`. The corrected v2 descriptor records original
strides and hashes C-value order without changing values. Both CPU and GPU use
the same explicitly bit-preserving contiguous seed copy. Two initial scalar
publication attempts also failed conservatively on reporting schema assumptions
(the timing-only audit-file resource and the nested timer); their failed JSONs
remain alongside the successful compact output. The numerical sources and
proofs were unchanged by those publisher repairs.
