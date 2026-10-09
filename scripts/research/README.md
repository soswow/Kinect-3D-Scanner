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
| [benchmark_production_missing_activation.py](benchmark_production_missing_activation.py) | Check installed weighted Final allocation with native capacity and per-key voxel-bit comparisons |
| [probe_final_budget.py](probe_final_budget.py) | Verify an insufficient Final budget using a fresh current-source raw replay; allocation only |
| [summarize_production_allocation.py](summarize_production_allocation.py) | Publish bounded scalar production validation from closed local reports |
| [GPU_ICP_MICROBATCH_RESULTS.md](GPU_ICP_MICROBATCH_RESULTS.md) | Executed proposal batching, complete GPU iterations, CUDA graphs and exact-neighbour variants; correctness and timing scopes |
| [gpu_icp_experiment_capture.py](gpu_icp_experiment_capture.py) | Rebuild genuine current field-pair proposals from raw calibrated views |
| [gpu_icp_experiment_driver.py](gpu_icp_experiment_driver.py) | Audit original ICP results and all original proposal gates, then separately time two/four independent seeds |
| [gpu_icp_experiment_loop_probe.py](gpu_icp_experiment_loop_probe.py) | Exhaustive actual-query audit of complete device iterations against CPU and CUDA graph execution |
| [gpu_icp_device_loop_experiment.py](gpu_icp_device_loop_experiment.py) | Own current-input complete-device audit and separately authorized graph timing, including construction and target setup |
| [summarize_gpu_icp_microbatch.py](summarize_gpu_icp_microbatch.py) | Publish bounded scalar samples and fresh original gate/query coverage from closed batch reports |
| [summarize_gpu_icp_device_loop.py](summarize_gpu_icp_device_loop.py) | Publish current-input device-loop query audits and separately measured setup-charged graph timing |
| [device_loop_workspace.py](device_loop_workspace.py) | Reuse pristine compiled execution setup while each complete proposal call owns fresh buffers, state and graph |
| [microbatch_bridge_driver.py](microbatch_bridge_driver.py) | Exhaust all genuine proposals through original complete bridge gates, then separately time only its fresh audited dynamic trajectory |
| [microbatch_bridge_protocol.py](microbatch_bridge_protocol.py) and [device_loop_workspace_protocol.py](device_loop_workspace_protocol.py) | Distinct complete-proposal timing authority, preserving actual unrounded inputs, query shadows, terminals and ambiguity |
| [microbatch_bridge_scope.py](microbatch_bridge_scope.py) | Private unchanged original verifier namespaces and match caches; preserve actual return objects and ordered gate evidence |
| [summarize_gpu_icp_complete_bridge.py](summarize_gpu_icp_complete_bridge.py) | Publish small scalar receipts from closed complete-proposal audit and matched timing runs |
| [profile_complete_bridge.py](profile_complete_bridge.py) | Observational Python boundary attribution; profiling overhead prevents speed claims from those runs |
| [device_loop_owned_workspace.py](device_loop_owned_workspace.py), [microbatch_bridge_owned_driver.py](microbatch_bridge_owned_driver.py) and [microbatch_bridge_owned_protocol.py](microbatch_bridge_owned_protocol.py) | Fresh complete-proposal audit with owned read-only proof handles and immutable indexed references; removes measured harness I/O overhead |
| [summarize_gpu_icp_owned_bridge.py](summarize_gpu_icp_owned_bridge.py) | Publish matched complete-proposal timings after closed owned-proof receipts |
| [benchmark_warp_flat_grid_filtered_nn.py](benchmark_warp_flat_grid_filtered_nn.py) and [research_warp_flat_grid_filtered_nn.cu](research_warp_flat_grid_filtered_nn.cu) | Conservative warp float screening with exact original double outputs; measured 12.6% raw lookup wall reduction, without scanner speed authority |
| [benchmark_icp_seed_reuse_census.py](benchmark_icp_seed_reuse_census.py) | Observe exact-cloud and near-seed repetition while executing every original CPU alignment and proposal check |
| [summarize_icp_seed_reuse_census.py](summarize_icp_seed_reuse_census.py) | Publish closed census counts and result differences without private clouds, seeds or performance claims |
| [near_seed_icp.py](near_seed_icp.py) and [benchmark_near_seed_reuse.py](benchmark_near_seed_reuse.py) | Bounded changed-method CPU result reuse with fresh actual-hit CPU audits; preserves the failed strict timing attempt |
| [benchmark_near_seed_conformance.py](benchmark_near_seed_conformance.py) | Independent actual-input numerical conformance and all original controls/gates; measured 7.0% median matched cold-phase reduction across nine fixed pairs |
| [summarize_near_seed_conformance.py](summarize_near_seed_conformance.py) | Recompute closed fixed-pair timing references and publish scalar receipts, including strict-v1 refusal; no production or whole-Finish authority |
| [benchmark_icp_convergence_tradeoff.py](benchmark_icp_convergence_tradeoff.py) | Explicit changed-budget/epsilon pilot through original proposal gates; measured shortcuts failed pose/witness comparison |
| [GPU_ICP_MICROBATCH_ARCHITECTURE.md](GPU_ICP_MICROBATCH_ARCHITECTURE.md) | Ordering, memory ownership and integration requirements for proposal batching and complete device iterations |

These files have different jobs: runnable experiments, bounded adapters,
independent validators, fault contracts and negative-result diagnostics. They
are retained to reproduce a result or test a concrete next change. A component
speedup, source-only test or partial native surface reference never enables an
unaudited whole-scanner path. Private captures, checkpoint arrays and compiled
libraries stay under ignored output folders and must be rebuilt locally.

The [executed complete-pair results](GPU_ICP_MICROBATCH_RESULTS.md#complete-gpu-verification-on-expensive-pairs)
now distinguish the earlier easy cases from expensive accepted and rejected
pairs. Complete GPU iterations improved the latter by 25–39% in the finished
cases, with fresh original-CPU query/result/gate audits and cold setup/cleanup
included. This is a stronger component result, still requiring whole-Finish
quality and timing before production integration.

## Device-resident nearest queries and ICP

An earlier fixed-component experiment measured **8.339 seconds first / 7.450
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

Current field-pair experiments above have separate fresh proofs and measured
setup costs. A controlled complete-session experiment must
retain original raw/live tracking inputs, all final acceptance gates and mesh
comparison. Only complete quality, memory and latency evidence could support a
future opt-in production integration. No automatic promotion follows from the
component result.

## Current-core whole-Finish GPU candidate

[profile_gpu_icp_finish_candidate.py](profile_gpu_icp_finish_candidate.py) is
the lower-overhead successor to the frozen strict Finish study. It creates
its own fresh raw-Live checkpoint on the actual current production core,
including automatic fusion allocation. The original fragment verifier and
all reconstruction checks still run. One bounded GPU workspace persists
across Finish; immutable cloud buffers persist across competing proposals
for a directed pair. Detailed gate observers are omitted, and source/runtime
inventory checks run at cold boundaries. This is a research prototype;
production ICP selection is unchanged.

The modes make their evidence and cost explicit:

- `capture` replays all raw views without archived pose seeds and stores a
  source-bound logical Live checkpoint.
- `native` calls original Finish directly and exports its actual Final poses
  and geometry. Internal registration, gate and optimizer traces are
  explicitly uncollected.
- `audit` checks every actual GPU nearest query and complete registration
  against the original CPU on the same inputs.
- `shadow` calls original CPU ICP before every GPU call on identical clouds
  and the unrounded seed, then compares returned results. It does not claim
  an exhaustive nearest-query audit.
- `measure` requires a fresh positive exhaustive audit of these actual
  source/runtime/configuration bytes. Each subsequent call binds its own
  newly consumed inputs; it does not claim to replay an old RANSAC history
  or prove a general input domain. CPU shadows and exhaustive query checks
  are omitted. Setup, hashing, original build, selected-device completion,
  allocation validation and cleanup remain charged inside Finish.

Run only with exclusive hardware and the field server stopped after the scan
is safe. From the main Windows checkout, use the CUDA environment and a fresh
private directory; never overwrite previous measurements:

```powershell
$cudaStudyPython = '.venv/Scripts/python.exe'
$cudaStudyRunner = 'scripts/research/profile_gpu_icp_finish_candidate.py'
$rawStudySession = 'export/chest-5-scan-session.zip'
$cudaStudyDirectory = 'benchmark-output/field-cuda-study/gpu-icp-candidate-v1/chest-5'
& $cudaStudyPython -s -u $cudaStudyRunner $rawStudySession --mode capture --checkpoint-directory "$cudaStudyDirectory/live" --output "$cudaStudyDirectory/capture.json" --run-allocated
& $cudaStudyPython -s -u $cudaStudyRunner $rawStudySession --mode native --checkpoint "$cudaStudyDirectory/live/checkpoint.json" --output "$cudaStudyDirectory/native.json" --run-allocated
& $cudaStudyPython -s -u $cudaStudyRunner $rawStudySession --mode audit --checkpoint "$cudaStudyDirectory/live/checkpoint.json" --output "$cudaStudyDirectory/audit.json" --run-allocated
& $cudaStudyPython -s -u scripts/research/compare_gpu_icp_candidate_finishes.py "$cudaStudyDirectory/native.json" "$cudaStudyDirectory/audit.json" --output "$cudaStudyDirectory/audit-quality.json" --run-allocated
& $cudaStudyPython -s -u $cudaStudyRunner $rawStudySession --mode measure --checkpoint "$cudaStudyDirectory/live/checkpoint.json" --candidate-proof "$cudaStudyDirectory/audit.json" --output "$cudaStudyDirectory/measure-0.json" --run-allocated
& $cudaStudyPython -s -u scripts/research/compare_gpu_icp_candidate_finishes.py "$cudaStudyDirectory/native.json" "$cudaStudyDirectory/measure-0.json" --output "$cudaStudyDirectory/measure-0-quality.json" --run-allocated
```

Check each command's exit before continuing. The independent comparator uses
actual accepted views, unrounded Final poses within 0.5 mm / 0.1 degrees, and
30,000-point physical surfaces in the original coordinates, with p95 at most
0.5 mm and precision/completeness at least .999 within 5 mm. Published bridge
and witness memberships are compared separately from unavailable optimizer
traces. Compare every measured Finish against an independent native control;
rotate multiple fresh native/GPU runs before reporting a speed gain. Audit
and shadow walls include CPU validation and cannot establish GPU performance.
Old strict failures and fixed-pair permits cannot authorize this new study.

At source preparation, full-Finish candidate performance and physical Final
quality on the current core remain unmeasured. See the
[historical complete-Finish results](GPU_ICP_MICROBATCH_RESULTS.md#complete-finish-strict-audit-and-observer-overhead)
for actual query/ICP evidence and the measured observer cost that motivated
this separate experiment.

## Other active prototypes

| Entry point | Status and next evidence |
|---|---|
| [benchmark_parallel_fragments.py](benchmark_parallel_fragments.py) | Bounded CPU worker/proposal dispatcher and local fixture builder. Measured gains were modest; it remains research-only and must preserve ordered authoritative acceptance. |
| [benchmark_anchored_gpu_model.py](benchmark_anchored_gpu_model.py) | Checked anchors and periodic local-map rebuilds under synthetic known motion. Needs continuous real-camera inputs and quality validation. |
| [benchmark_dense_model.py](benchmark_dense_model.py) | Dense GPU preview/odometry architecture measurements. Synthetic capacity is not end-to-end scanning FPS. |
| [plot_gpu_model_experiments.py](plot_gpu_model_experiments.py) | Plot saved synthetic model experiments; requires Matplotlib and existing reports. |

## Running a new experiment

Historical source-bound experiments use research baseline commit
`fb9069d33cd12efb3b305054934fea28ffbc1959` and the
[measured-byte restoration instructions](MEASURED_SOURCE_REPRODUCIBILITY.md).
Their old core checks correctly reject the subsequent production allocation
change. Current-source raw workflow and installed allocation tools explicitly
check the source they actually measure.

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
