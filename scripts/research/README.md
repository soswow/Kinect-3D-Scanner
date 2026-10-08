# Active performance research

These experiments are separate from the released scanner. They are retained
because they provide a concrete next validation step or a useful dispatcher and
architecture prototype. Run commands from the repository root. The
[catalog](../tool-catalog.json) records every relocated path; the
[archive index](archive/README.md) explains retained dependencies and earlier
negative results.

## Current field study

The chest-5/6/7 findings and field tradeoffs are in
[Field CUDA research](../../docs/FIELD_CUDA_RESEARCH.md). The existing deferred
recovery workflow reduced selected-view Live work by 47–78% with matching Final
coverage and passing physical surface comparisons, while producing a sparser
Live preview. It is available through the launcher `-LiveRecovery deferred`;
the default stays `full`. The client still shows `legacy` because production ICP
and geometric verification remain on CPU.

The [additive field-tool inventory](field-tool-catalog.json) lists the new
protocols and their supporting modules. The original relocation catalog stays
unchanged because measured earlier component proofs bind its exact bytes.

| Entry point or protocol | Future use |
|---|---|
| [analyze_field_sessions.py](analyze_field_sessions.py) | Inspect raw ZIP metadata/settings/coverage without initializing numerical libraries |
| [benchmark_field_sessions.py](benchmark_field_sessions.py) | Historical matched CPU/CUDA raw selected-view baseline producer |
| [benchmark_field_policy.py](benchmark_field_policy.py) | Compare existing full/deferred CUDA recovery with owned Windows worker cleanup and complete settings/backend closure |
| [summarize_field_latency.py](summarize_field_latency.py) | Recompute latency quantiles from closed raw profiles without importing CUDA |
| [compare_field_experiment.py](compare_field_experiment.py) | Fixed-coordinate observable mesh comparison; no backend or graph authority |
| [CHECKPOINT_RESIDENT_FINISH_RESEARCH.md](CHECKPOINT_RESIDENT_FINISH_RESEARCH.md) | Same measured Live checkpoint and strict ordered-history experiments; failed field comparisons remain evidence |
| [FIELD_FINISH_CONFORMANCE.md](FIELD_FINISH_CONFORMANCE.md) | Distinct actual-input CPU shadows, original graph/witness decisions and physical Final quality protocol |
| [BULK_LEGACY_NN_AUDIT.md](BULK_LEGACY_NN_AUDIT.md) | Full legacy CPU nearest auditing through a separately proven native bulk interface |
| [COMBINED_SYNC_RESEARCH.md](COMBINED_SYNC_RESEARCH.md) | Audited one-copy ICP orchestration and separate contemporary three-way timing; does not select a production backend |
| [DEVICE_LDLT_RESEARCH.md](DEVICE_LDLT_RESEARCH.md) | Captured original systems and CUDA solver numerical/micro-scope tests; isolated GPU solve was slower |
| [CANONICAL_FPFH_RESEARCH.md](CANONICAL_FPFH_RESEARCH.md) | Diagnose and test proposal-input stability without changing original training/verification points |
| [MARKER_PROPOSAL_RESEARCH.md](MARKER_PROPOSAL_RESEARCH.md) | Exact native marker lookup and proposal experiments; appending more seeds did not help whole Finish |
| [FINAL_HEADROOM_RESEARCH.md](FINAL_HEADROOM_RESEARCH.md) | Reproduce Open3D repeated-key reserve growth and distinguish logical limits from actual allocation |
| [benchmark_missing_activation.py](benchmark_missing_activation.py) | Compare original and missing-key activation with per-key CPU/tensor/fused voxel-bit checks |

These files have different jobs: runnable experiments, bounded adapters,
independent validators, fault contracts and negative-result diagnostics. They
are retained to reproduce a result or test a concrete next change. A component
speedup, source-only test or partial native surface reference never enables an
unaudited whole-scanner path. Private captures, checkpoint arrays and compiled
libraries stay under ignored output folders and must be rebuilt locally.

## Device-resident nearest queries and ICP

The latest fixed-component experiment measured **8.339 seconds first / 7.450
seconds warm**, against **11.209 seconds native CPU**: approximately **1.34× /
1.50× faster**. Fresh CPU audits checked 104,123,989 nearest queries with zero
changed IDs or false misses, and all nine original proposal/witness/pose gates
passed. These are two-pair component results, not complete-session Finish or
mesh measurements. The production registration path is unchanged.

Read the [device-flat research notes](DEVICE_FLAT_GRID_RESEARCH.md),
[published compact result](../../docs/benchmarks/cuda-pipeline/uniform-grid-nearest/device-resident-v1/research-summary.json)
and [full experiment report](../../docs/CUDA_EXPERIMENTS.md#true-device-flat-grid-with-resident-icp).

| File | Role |
|---|---|
| [benchmark_device_grid_resident.py](benchmark_device_grid_resident.py) | Fresh synthetic tests and original nine-proposal trajectory with CPU hit/miss shadows |
| [benchmark_device_grid_resident_timing.py](benchmark_device_grid_resident_timing.py) | Separate native/first/warm timing after distinct proofs validate |
| [cuda_device_flat_grid_registration.py](cuda_device_flat_grid_registration.py) | Bounded original XYZ/index ownership and device query/result transport |
| [research_device_flat_grid_nn.cu](research_device_flat_grid_nn.cu) | Exact ambiguity classification, flagged-row packing and correction scatter |
| [research_device_grid_resident_icp.py](research_device_grid_resident_icp.py) | Resource/provenance wrapper around the inherited resident math |
| [device_flat_grid_synthetic.py](device_flat_grid_synthetic.py) | Focused classifier, transport, failure and cache-lifetime tests |
| [validate_device_flat_grid_proof.py](validate_device_flat_grid_proof.py) | Distinct immutable authority for a freshly audited resident trajectory |
| [check_device_flat_proof_contract.py](check_device_flat_proof_contract.py) | Artificial stdlib contract/fault tests; no numerical authority |
| [summarize_device_grid_resident_research.py](summarize_device_grid_resident_research.py) | Closed-report provenance, execution and disjoint host-wall summary |

The next step is to regenerate proofs under this layout, then decide whether a
controlled complete-session experiment is justified. Such an experiment must
retain original raw/live tracking inputs, all final acceptance gates and mesh
comparison. Only complete quality, memory and latency evidence could support a
future opt-in production integration. No automatic promotion follows from the
component result.

## Other active prototypes

| Entry point | Status and next evidence |
|---|---|
| [benchmark_parallel_fragments.py](benchmark_parallel_fragments.py) | Bounded CPU worker/proposal dispatcher and local fixture builder. Measured gains were modest; it remains research-only and must preserve ordered authoritative acceptance. |
| [benchmark_anchored_gpu_model.py](benchmark_anchored_gpu_model.py) | Checked anchors and periodic local-map rebuilds under synthetic known motion. Needs continuous real-camera inputs and quality validation. |
| [benchmark_dense_model.py](benchmark_dense_model.py) | Dense GPU preview/odometry architecture measurements. Synthetic capacity is not end-to-end scanning FPS. |
| [plot_gpu_model_experiments.py](plot_gpu_model_experiments.py) | Plot saved synthetic model experiments; requires Matplotlib and existing reports. |

## Running a new experiment

Use the scanner/native environment plus CUDA-enabled Open3D and CuPy. The
resident method also needs the exact pinned Eigen CPU bridge, built locally
with Visual Studio using
[research_resident_icp.py](archive/research_resident_icp.py). Derive fixtures
locally from the original raw views and measured fragment reports; private
arrays and DLLs are not shipped with these scripts. CPU audits and timing must
run separately, in an exclusive hardware slot outside active scanning.

Archived baselines remain executable dependencies: the active adapter inherits
the [flat adapter](archive/cuda_flat_grid_registration.py),
[uniform-grid foundation](archive/cuda_uniform_grid_registration.py) and
[flat shader](archive/research_flat_grid_nn.cu); the resident wrapper inherits
the [original resident math/Eigen bridge](archive/research_resident_icp.py).
Archived producer, evidence and validator helpers are also imported by the
active proof pipeline. Moving them to `archive/` does not make them unused.

Historical proof reports and published snapshots preserve their measured paths
and hashes. Relocation, source bytes, line endings, platform, library binaries
and resource policy can invalidate that authority. Generate fresh synthetic
and real-trajectory proofs for the current helpers and raw fixture, rebuild the
pinned DLL as needed, and preserve the new reports. Historical measured-source
guards may require a separately reviewed experiment update before a new layout
can execute; a guard rejection is not permission to replace old hashes or skip
an audit. Published compact summaries cannot unlock unaudited timing.
