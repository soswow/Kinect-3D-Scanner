# Maintained component benchmarks

These entry points measure individual costs and correctness alongside the
complete-session tools in [scripts/](../README.md). They do not enable a backend
in the running scanner. Run them from the repository root, in an isolated slot,
against local exported data or an explicitly prepared fixture.

| Entry point | Measurement |
|---|---|
| [benchmark_cuda_fusion.py](benchmark_cuda_fusion.py) | Confidence-weighted fusion timing and CPU/tensor/fused agreement |
| [benchmark_cuda_matching.py](benchmark_cuda_matching.py) | Descriptor retrieval and matching |
| [benchmark_cuda_threads.py](benchmark_cuda_threads.py) | Backend/thread policy comparisons |
| [benchmark_matching_exhaustive.py](benchmark_matching_exhaustive.py) | Exhaustive descriptor decisions and tie cases |
| [benchmark_opencv_threads.py](benchmark_opencv_threads.py) | OpenCV feature-processing thread policies |
| [benchmark_registration_options.py](benchmark_registration_options.py) | Registration alternatives against original pose/quality gates |
| [benchmark_sift_distance_options.py](benchmark_sift_distance_options.py) | SIFT distance arithmetic/precision alternatives |
| [benchmark_visual_refinement.py](benchmark_visual_refinement.py) | Visual refinement cost and original decision/pose agreement |
| [profile_fragment_verification.py](profile_fragment_verification.py) | Original camera/union verification branches, native calls and target reuse |
| [benchmark_offline_geometry.py](benchmark_offline_geometry.py) | Depth-only registration qualification across distinct local recordings |
| [audit_depth_visibility.py](audit_depth_visibility.py) | All-view depth visibility/consistency at measured poses |
| [render_depth_graph.py](render_depth_graph.py) | Render depth geometry and camera paths without RGB |
| [replay_normal_capture.py](replay_normal_capture.py) | Offline replay through ordinary client selection/payload behavior |

Find these CLIs and their supporting source without importing numerical packages:
`python scripts/list_tools.py --group benchmarks --json`.

Start with the CLI's `--help`; output and input options differ. For a full
workflow comparison use [profile_cuda_pipeline.py](../profile_cuda_pipeline.py),
then [compare_session_profiles.py](../compare_session_profiles.py) and the
appropriate summarizer. Component gains alone cannot predict live capture FPS,
Finish latency or final mesh quality.

Keep hardware workloads, source/runtime binaries, thread resources, raw views,
calibration and settings matched. Separate CPU-shadow correctness runs from
unaudited timing. Record first-call setup, warm work, memory and actual device
selection. Nested stage timers require subtraction before summing them into a
wall-time partition.

CUDA/custom-kernel benchmarks require the installed scanner/native environment,
CUDA-enabled Open3D and optional CuPy. Some comparison methods are experiments;
their presence in a maintained measurement driver is not a recommendation for
production. The [experiment report](../../docs/CUDA_EXPERIMENTS.md) records which
approaches passed, regressed or remain unvalidated.

The fragment fixture is generated locally from raw ZIP data and a measured
Finish report by
[benchmark_parallel_fragments.py](../research/benchmark_parallel_fragments.py).
Its fragment-local poses reproduce a fixed component, not live tracking seeds.
Fixtures, point clouds and logs remain ignored. Historical reports are unchanged
by this organization; relocated helper hashes require fresh proof artifacts.
