# Installed weighted Final allocation validation

The installed change allocates the exact required weighted Final block capacity
and activates only missing keys at the original CPU, tensor and fused CUDA
callers. Original full-frustum lookup, voxel updates, confidence weights and
frame order remain the measured numerical path. This note describes installed
core `07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c`.
The frozen historical controls use core
`9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a`.

Fresh full raw replays retain the same Final views as the historical CUDA
controls. The earlier raw reports recorded configured allowance, not actual
native allocation. The following comparison must retain that distinction.
Attribute figures include five float32 values per voxel in a 16³-voxel block;
they are an allocation floor, excluding indices, scratch, caches and allocator
overhead, and are not peak GPU memory or process RSS.

| Raw archive | Final views | Historical configured Final allowance | Installed measured capacity | Installed attribute floor |
| --- | ---: | ---: | ---: | ---: |
| chest-5 | 7 | 10,000 blocks / 781.25 MiB | 3,328 blocks | 260.00 MiB |
| chest-6 | 38 | 10,000 blocks / 781.25 MiB | 2,860 blocks | 223.4375 MiB |
| chest-7, common 20k logical budget | 108 | 20,000 blocks / 1,562.50 MiB | 13,302 blocks | 1,039.21875 MiB |

All installed initial and completed capacities equal the required block count.
The fresh installed physical smoke compares all three original weighted
backends: ordinary activation grows capacity 4→8, while missing-key activation
retains 4. All 81,920 key-mapped float32 TSDF/weight/color values per backend
match exactly, including nonzero and fractional confidence contributions.

Fixed-frame surface comparisons use 30,000 triangle samples per mesh and a
5 mm correspondence threshold, with no alignment or scale fitting. All three
have precision and completeness 1.0; p95 surface differences are about
0.00006 mm for chest-5 and 0.00012 mm for chest-6/7. Final poses also satisfy
the declared 0.5 mm / 0.1° bounds. Ordered point/face arrays differ due to mesh
buffer ordering; this is equivalent surface evidence, not ordered-array or
bit-identical trajectory proof.

The corrected current-profile allocation probe uses chest-7 poses reconstructed
by a completed current-core unseeded full raw replay solely to call the Final
allocation boundary. A logical 10,000-block limit rejects the required 13,302
blocks after one native scratch block, before any Final candidate allocation or
fusion. It does not rerun tracking and never reads archived ZIP poses as seeds.
Raw input/pose owners, hooks, environment, source, loaded native bytes and GPU
identity close unchanged. It creates no registration or mesh authority.

The first installed smoke failed on a native input representation mismatch.
The first budget probe failed on an observer comparison of Python tuples with
JSON lists. Their failed reports and measured source snapshots remain intact;
fresh corrected reports are separate. The earlier research chest-7 strict pose
identity failure is retained in the frozen field-study summary and is not
relabeled by these bounded installed comparisons.

`KINECT_CUDA_CONFIDENCE=off` selects the original CPU confidence algorithm.
`confidence_fusion=True` remains enabled in both historical and installed
replays. CUDA input preprocessing, CUDA descriptor matching and fused weighted
fusion are checked as actual backends rather than inferred from requested flags.

The new [scalar publication helper](summarize_production_allocation.py) checks
the precise source transition, same raw/settings/selection/thread/native/device
scope, actual capacities, worker completion, bound physical inputs and fixed
surface/pose limits. It emits relative content-hashed references and an explicit
false performance authority. Single-run phase walls across the source transition
are descriptive and establish no causal speed gain or continuous camera FPS.

After all local measurements close, run with base Python and no numerical imports:

```powershell
F:/ProgramData/anaconda3/python.exe -S scripts/research/summarize_production_allocation.py `
  --quality-directory production-memory-quality-v2 `
  --smoke-report production-allocation-smoke-v2/report.json `
  --probe-report production-final-budget-v2/report.json `
  --observations root-completed-exec-observations-v1.json `
  --output benchmark-output/field-cuda-study/production-allocation-summary-v1.json
```

Use a fresh output filename. The default ignored input root is
`benchmark-output/field-cuda-study`; `--study-root` can select a copied local
evidence directory, but saved private input path bindings must still identify
its actual report/payload files. The original ZIPs, logs and geometry arrays
remain local. Publish only the resulting scalar summary; hashes alone cannot
reproduce unavailable private numerical inputs. Root tool observations record
actual completed waits/exit codes without a returned PID; those records do not
claim the owned process-tree identity separately proved by raw replay workers.

This retrospective publisher also requires the measured checkout's exact core
bytes. Git newline conversion can change that fingerprint without changing the
code. The retained measurement worktree preserves those bytes; a new checkout
must produce fresh measurements rather than replace the recorded hashes.
Current-source raw replays and the installed smoke/budget tools bind their
actual source dynamically, whereas this publication describes one fixed study.

Focused reporting tests:

```powershell
F:/ProgramData/anaconda3/python.exe -S -m unittest tests.test_production_allocation_summary
```
