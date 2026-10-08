# Archived experiments and retained foundations

This directory preserves earlier measured approaches, superseded prototypes and
prepared experiments. The [active roadmap](../README.md) identifies the current
next steps; [tool-catalog.json](../../tool-catalog.json) maps all old paths.
Archive status describes research priority, not whether a module is imported.
Several files here are required by the active device-resident proof pipeline.

## Inherited foundations

| Files | Retained role |
|---|---|
| [cuda_uniform_grid_registration.py](cuda_uniform_grid_registration.py), [research_uniform_grid_nn.cu](research_uniform_grid_nn.cu), [validate_uniform_grid_proof.py](validate_uniform_grid_proof.py) | Original-double grid/cache/CPU ambiguity resolver and base proof authority |
| [cuda_flat_grid_registration.py](cuda_flat_grid_registration.py), [research_flat_grid_nn.cu](research_flat_grid_nn.cu), [validate_flat_grid_proof.py](validate_flat_grid_proof.py) | Flat candidate iterator and inherited adapter/proof hierarchy |
| [research_resident_icp.py](research_resident_icp.py) | Original resident transforms, equations, convergence and pinned Eigen bridge build |
| [benchmark_uniform_grid_nn.py](benchmark_uniform_grid_nn.py), [benchmark_flat_grid_nn.py](benchmark_flat_grid_nn.py), [benchmark_optix_nn.py](benchmark_optix_nn.py) | Shared fixture, synthetic, evidence and ownership helpers used by later experiments |
| [benchmark_selective_fragment_threads.py](benchmark_selective_fragment_threads.py) | Original fragment-copy helper and selective-thread experiment |

These foundations remain available and receive relocation/import fixes.
Numerical changes still require independent fresh proofs.

## Earlier nearest-neighbor and ICP results

OptiX conservative, staged and fused traversal preserved the tested original
nearest decisions but lost to native CPU in full-component timing. The earlier
OptiX resident loop also regressed. Serial/staged grids and the flat host bridge
showed that fewer candidate calculations or a faster sampled shader alone did
not remove host/transfer costs. Those negative results motivated the active
true-device resident path.

| Family | Notes and entry points |
|---|---|
| OptiX | [OPTIX_RESEARCH.md](OPTIX_RESEARCH.md), [benchmark_optix_nn.py](benchmark_optix_nn.py), [benchmark_optix_staged.py](benchmark_optix_staged.py), [cuda_optix_registration.py](cuda_optix_registration.py), [cuda_optix_staged_registration.py](cuda_optix_staged_registration.py), [summarize_optix_research.py](summarize_optix_research.py) |
| OptiX build/enclosure | [build_optix_nn.ps1](build_optix_nn.ps1), [build_optix_staged.ps1](build_optix_staged.ps1), [check_optix_enclosure.py](check_optix_enclosure.py), [research_optix_nn.cu](research_optix_nn.cu), [research_optix_nn_host.cpp](research_optix_nn_host.cpp), [research_optix_staged.cu](research_optix_staged.cu), [research_optix_staged_host.cpp](research_optix_staged_host.cpp) |
| Custom double KDTree | [cuda_kdtree_registration.py](cuda_kdtree_registration.py), [check_cuda_kdtree.py](check_cuda_kdtree.py) |
| Serial full grid | [UNIFORM_GRID_RESEARCH.md](UNIFORM_GRID_RESEARCH.md), [benchmark_uniform_grid_timing.py](benchmark_uniform_grid_timing.py), [summarize_uniform_grid_nn.py](summarize_uniform_grid_nn.py) |
| Staged/pruned grid | [STAGED_GRID_RESEARCH.md](STAGED_GRID_RESEARCH.md), [cuda_staged_grid_registration.py](cuda_staged_grid_registration.py), [research_staged_grid_nn.cu](research_staged_grid_nn.cu), [benchmark_staged_grid_nn.py](benchmark_staged_grid_nn.py), [benchmark_staged_grid_timing.py](benchmark_staged_grid_timing.py), [validate_staged_grid_proof.py](validate_staged_grid_proof.py), [summarize_staged_grid_nn.py](summarize_staged_grid_nn.py) |
| Flat host bridge | [FLAT_GRID_RESEARCH.md](FLAT_GRID_RESEARCH.md), [benchmark_flat_grid_timing.py](benchmark_flat_grid_timing.py), [summarize_flat_grid_research.py](summarize_flat_grid_research.py) |
| Sampled lookup and diagnostic ablations | [benchmark_grid_lookup_ablation.py](benchmark_grid_lookup_ablation.py), [benchmark_grid_host_diagnostics.py](benchmark_grid_host_diagnostics.py), [research_parallel_grid_nn.cu](research_parallel_grid_nn.cu), [summarize_grid_host_diagnostics.py](summarize_grid_host_diagnostics.py) |

## Prepared parallel-staged prototype

[PARALLEL_STAGED_GRID_RESEARCH.md](PARALLEL_STAGED_GRID_RESEARCH.md) describes
the separate [parallel-staged adapter](cuda_parallel_staged_grid_registration.py)
and [shader](research_parallel_staged_grid_nn.cu). The bounded lookup/diagnostic
ablation executed this kernel, but its dedicated complete nine-proposal proof
and timing pipeline was **prepared and unexecuted**. Its
[proof producer](benchmark_parallel_staged_grid_nn.py),
[timing driver](benchmark_parallel_staged_grid_timing.py) and
[validator](validate_parallel_staged_grid_proof.py) carry no completed full-bridge
or production authority. Sampled raw-bit parity does not replace that proof.

## Other diagnostics and superseded research

| Files | Status |
|---|---|
| [research_native_icp_signatures.py](research_native_icp_signatures.py) | Exact repeated-call signatures and measured reuse opportunity; diagnostic only |
| [summarize_fragment_verification.py](summarize_fragment_verification.py) | Summarize saved original verification-branch profiling |
| [summarize_selective_fragment_threads.py](summarize_selective_fragment_threads.py) | Selective small-cloud thread pilot; tested policies regressed |
| [benchmark_confidence_cache.py](benchmark_confidence_cache.py) | Bounded exact confidence-only reuse; small component savings did not justify production integration |
| [adaptive_visual_experiment.py](adaptive_visual_experiment.py) | Earlier visual-tracking experiment; current maintained policy is implemented in the server |
| [research_gpu_prepare.py](research_gpu_prepare.py), [benchmark_gpu_prepare.py](benchmark_gpu_prepare.py), [research_gpu_confidence_exact.py](research_gpu_confidence_exact.py) | Original input/confidence investigations; validated production modules now live in `scanner_server/` |

## Reproduction and historical authority

Read the [full experiment record](../../../docs/CUDA_EXPERIMENTS.md) and
[published result index](../../../docs/benchmarks/CUDA_REPORTS.md) for measured
status, source/runtime fingerprints, quality gates and timing scope. Commands
run from the repository root. CUDA/Open3D/CuPy are required for hardware paths;
OptiX additionally needs its documented fixed SDK headers/toolchain, and the
resident bridge needs the pinned Eigen archive/include tree and a local build.
Raw ZIPs, derived fixtures, arrays, DLLs and SDK downloads remain ignored.

Saved historical proof reports are unchanged by relocation. Current helper
paths and file hashes differ, so old proof tokens cannot authorize the relocated
sources. Rebuild local DLLs and raw-derived fixtures as needed and produce fresh
synthetic plus real hit/miss/witness proofs before timing. A frozen historical
guard may require a reviewed new experiment definition; do not rewrite an old
report's fingerprints, bypass a guard or use archived poses as live seeds.
Archive placement is neither production selection nor permission to run hardware
experiments during an active scan.
