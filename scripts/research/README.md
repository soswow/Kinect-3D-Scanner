# Scanner research

Offline experiments, retained prototypes and their evidence live here. Start
with the family below that answers your question; the catalogs list individual
tools. These experiments do not select a released scanner backend.

- [Field study](../../docs/FIELD_CUDA_RESEARCH.md): raw chest-5/6/7 controls,
  recovery tradeoffs and quality limits.
- [Current Finish results](GPU_ICP_CURRENT_FINISH_RESULTS.md): C5 improved
  modestly, C6 showed no useful gain, and C7 has unresolved strict quality
  failures. C8 has a saved raw replay but no Finish qualification.
- [Published report index](../../docs/benchmarks/CUDA_REPORTS.md) and
  [experiment record](../../docs/CUDA_EXPERIMENTS.md): measured history.
- [Field tool catalog](field-tool-catalog.json): current tools, protocols and
  supporting modules; [original catalog](../tool-catalog.json): relocated paths.
- [Archive](archive/README.md): earlier negative results and imported foundations.

Production ICP and geometric verification remain on CPU. The existing
`-LiveRecovery deferred` launcher option reduced selected-view Live work by
47–78% in the field study with matching Final coverage and passing physical
surface comparisons, at the cost of a sparser Live preview; `full` is the default.

## Find an experiment

| Family | Start here | Scope or status |
|---|---|---|
| Raw field sessions and recovery | [Field study](../../docs/FIELD_CUDA_RESEARCH.md), [analyze_field_sessions.py](analyze_field_sessions.py), [benchmark_field_policy.py](benchmark_field_policy.py), [summarize_field_latency.py](summarize_field_latency.py) | Read ZIP metadata without numerical imports; compare policies and recompute closed-profile latency |
| Current whole-Finish candidate | [Current results](GPU_ICP_CURRENT_FINISH_RESULTS.md), [profile_gpu_icp_finish_candidate.py](profile_gpu_icp_finish_candidate.py) | Fresh current-core capture/native/audit/shadow/measure modes; no production qualification |
| Checkpoint and field conformance | [Checkpoint study](CHECKPOINT_RESIDENT_FINISH_RESEARCH.md), [field conformance](FIELD_FINISH_CONFORMANCE.md), [analyze_finish_trace_differences.py](analyze_finish_trace_differences.py) | Preserve failed strict histories, actual-input shadows and independent Final quality |
| Exact device nearest queries | [Device-flat notes](DEVICE_FLAT_GRID_RESEARCH.md), [benchmark_device_grid_resident.py](benchmark_device_grid_resident.py) | Fixed-component proofs; active adapters inherit archived grid and resident code |
| CPU bulk shadows | [Bulk audit](BULK_LEGACY_NN_AUDIT.md) | Complete original nearest-query audit; auditor throughput is separate from scanning speed |
| One-copy orchestration and source checks | [Combined synchronization](COMBINED_SYNC_RESEARCH.md), [source-guard experiment](COMBINED_SOURCE_GUARD.md) | Separate audit/timing authorities; retained regression and causal component result |
| Device solver | [LDLT study](DEVICE_LDLT_RESEARCH.md) | Captured original systems; isolated GPU solve was slower |
| Proposal batching, graphs and bridge ownership | [Microbatch results](GPU_ICP_MICROBATCH_RESULTS.md), [architecture](GPU_ICP_MICROBATCH_ARCHITECTURE.md) | Complete iterations, exact neighbours, setup reuse and owned proof handles |
| Seed reuse and convergence | [Current results](GPU_ICP_CURRENT_FINISH_RESULTS.md), [benchmark_icp_seed_reuse_census.py](benchmark_icp_seed_reuse_census.py), [benchmark_near_seed_conformance.py](benchmark_near_seed_conformance.py), [benchmark_icp_convergence_tradeoff.py](benchmark_icp_convergence_tradeoff.py) | Census, bounded changed-method reuse and failed pose/witness shortcuts |
| Canonical features and markers | [FPFH proposal notes](CANONICAL_FPFH_RESEARCH.md), [marker proposal notes](MARKER_PROPOSAL_RESEARCH.md), [current results](GPU_ICP_CURRENT_FINISH_RESULTS.md) | Separate proposal policies; additional marker seeds did not improve whole Finish |
| Final allocation | [Headroom investigation](FINAL_HEADROOM_RESEARCH.md), [installed allocation validation](PRODUCTION_ALLOCATION_VALIDATION.md) | Preserve failed reserve experiment; missing-key production validation is a distinct study |
| Cached native target geometry | [Closed results](CACHED_TARGET_GEOMETRY_RESULTS.md), [source/build contract](cached_target_geometry_native.md) | Exact finite-fixture lookup parity; no complete ICP or Finish gain |
| Routing and parallel/model prototypes | [cpu_first_icp_routing.py](cpu_first_icp_routing.py), [benchmark_parallel_fragments.py](benchmark_parallel_fragments.py), [benchmark_anchored_gpu_model.py](benchmark_anchored_gpu_model.py), [benchmark_dense_model.py](benchmark_dense_model.py), [plot_gpu_model_experiments.py](plot_gpu_model_experiments.py) | Source-only routing, modest dispatcher gains and synthetic model architecture; fixed saved-report plotter |

Protocol documents may preserve their original preparation status. Read the
linked closed results and report index for later execution outcomes. Keep failed
and unexecuted experiments distinguishable from successful component results.

## Reproduce safely

Run commands from the repository root in the scanner/native environment.
Hardware paths require CUDA-enabled Open3D and CuPy; plotting needs Matplotlib.
Private raw ZIPs, checkpoints, fixture arrays, SDK downloads and compiled DLLs
are ignored local inputs and must be obtained or rebuilt locally. A public
scalar summary cannot reproduce unavailable private numerical inputs.

CPU/native audits and timing run separately in an exclusively allocated hardware
slot, outside active scanning. Stop the field server only after the scan is safe.
Use fresh output paths, check every command's exit and preserve failed reports.
An exit of zero can still accompany a failed comparison; inspect the report's
explicit quality/pass fields before treating an experiment as accepted.
Older OptiX producers can overwrite outputs and may record `passed` during an
interrupt; reject interrupted runs regardless of that field. Preserve their
frozen source and use a fresh path rather than rerunning over evidence.
Follow the selected protocol's exact input, thread, device, cache and runtime
requirements; CLI help alone is not evidence that a run is qualified.

Historical source-bound experiments use baseline
`fb9069d33cd12efb3b305054934fea28ffbc1959` in a separate checkout. Read
[measured-byte restoration](MEASURED_SOURCE_REPRODUCIBILITY.md): Git newline
conversion can change authority even when normalized code is unchanged. Later
production allocation changes correctly invalidate historical core guards.
Current-source tools bind the source they actually measure.

The resident method needs the pinned Eigen CPU bridge, built locally with
Visual Studio using [research_resident_icp.py](archive/research_resident_icp.py).
Active modules also import archived adapters, shaders, producers and validators.
Archive placement does not make a dependency disposable.

Source, catalog, document, line-ending, platform, binary and resource-policy
changes can invalidate measured proofs and checkpoints. The original catalog
is byte-preserved because older proofs bind it. Do not replace saved hashes,
rewrite old reports or bypass a guard to run a new layout. Rebuild required
artifacts and generate fresh synthetic plus actual-trajectory evidence, or use
the exact frozen baseline.

### Current-core whole-Finish GPU candidate

For the current whole-Finish candidate, the order is fresh `capture`, independent
`native`, exhaustive `audit`, independent physical comparison, then `measure`
with the fresh positive audit and comparison against the native control.
`shadow` compares complete ICP results but is not an exhaustive nearest audit.
Audit/shadow wall time includes CPU validation and cannot establish GPU speed.
Compare every measured Finish; rotate fresh native/GPU runs before reporting a
gain. The original-coordinate criteria include 0.5 mm / 0.1 degree Final poses,
30,000 surface samples, p95 at most 0.5 mm, and precision/completeness at least
.999 within 5 mm. Old fixed-pair permits and strict failures cannot authorize
this current-source study.

## AprilTag session diagnosis

[`analyze_apriltag_session.py`](analyze_apriltag_session.py) reads an explicit saved
session ZIP and writes pair timings, duplicate-ID evidence and optional compact
tag/depth experiments to `--output`; use ignored `benchmark-output/`. It requires
the scanner numerical environment and OpenCV ArUco, and does not contact the
running server. `--compact-only --depth-refinement` selects the compact experiment;
`--exclude-id` is an explicit experimental exclusion, not production policy.
See [retained results and current reconstruction order](../../docs/FRAGMENT_RECONNECTION.md#apriltag-priority).
