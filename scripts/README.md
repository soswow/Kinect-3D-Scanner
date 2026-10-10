# Scanner tools

Run commands from the repository root using the scanner's Python environment.
The [tool catalog](tool-catalog.json) maps the previous CUDA-study paths to the
current layout. Existing general scanner tools keep their original paths.

| Location | Purpose |
|---|---|
| This directory | Maintained launch, replay, profiling, validation and report tools |
| [benchmarks/](benchmarks/README.md) | Maintained component measurements |
| [research/](research/README.md) | Active experiments and the next validation steps |
| [research/archive/](research/archive/README.md) | Earlier experiments, negative results and inherited research dependencies |

## Everyday use

For the measured Windows CUDA preview recipe:

```powershell
./scripts/start_cuda_server.ps1 -Recipe adaptive -CudaInput auto
```

Select **Live voxel 10 mm / Final voxel 5 mm**, a **10,000-block final budget**
and **Final surface confidence 2** in the client. CUDA confidence defaults off.
See the [scanner README](../README.md) and
[current performance report](../docs/CUDA_EXPERIMENTS.md) for requirements,
quality limits and the contribution from the preview/tracking changes.

For a more responsive field trial, add `-LiveRecovery deferred`. In three new
raw session replays this reduced Live processing by 47–78% and retained the same
Final views with passing surface comparisons. It fuses fewer views during Live,
so the preview can be sparser; Finish still runs the original recovery and
verification. The default remains `full`. For the larger chest-7 scan, select
a **20,000-block final budget**: its exact 5 mm volume needed 13,302 unique
blocks and could not fit the original 10,000-block limit. Read the
[field measurements and tradeoffs](../docs/FIELD_CUDA_RESEARCH.md).

Weighted Final fusion now plans its required blocks and allocates that exact
capacity. It reports both the configured limit and actual allocation, and rejects
an insufficient limit before allocating the Final candidate. The limit still
needs to cover the complete reconstruction; it does not limit a scan's coverage.

The launcher checks ownership of an existing listener. Server validation tools
perform synthetic captures and settings/reset operations; use them only with
an empty test server. They refuse an existing scan and restore test settings.
Backend probes and offline benchmarks also consume hardware, so schedule them
outside active scanning.

| Entry point | Use |
|---|---|
| [start_cuda_server.ps1](start_cuda_server.ps1) | Start a server with an explicit baseline, hybrid or adaptive recipe |
| [replay_scan.py](replay_scan.py) | Replay public or recorded RGB-D data |
| [profile_capture.py](profile_capture.py) | Measure the capture path |
| [profile_session.py](profile_session.py) | Profile raw sessions; `--finish --finish-only` measures cold depth Finish without archived pose seeds |
| [profile_session_backends.py](profile_session_backends.py) | Compare complete session backend recipes |
| [profile_cuda_pipeline.py](profile_cuda_pipeline.py) | Run controlled pipeline matrices and quality comparisons |
| [compare_session_profiles.py](compare_session_profiles.py) | Compare completed reconstructions and triangle surfaces |
| [check_cuda_backend.py](check_cuda_backend.py) | Run backend regression tests, including hardware checks where available |
| [check_cuda_server.py](check_cuda_server.py) | Validate synthetic server requests on an empty test server |
| [check_cuda_native_server.py](check_cuda_native_server.py) | Validate calibrated native input on an empty test server |
| [summarize_cuda_study.py](summarize_cuda_study.py) | Summarize the original fusion/backend study |
| [summarize_cuda_pipeline.py](summarize_cuda_pipeline.py) | Summarize controlled pipeline results and provenance |
| [publish_cuda_reports.py](publish_cuda_reports.py) | Copy an explicit list of saved summaries/charts into documentation |
| [http_check_safety.py](http_check_safety.py) | Shared empty-session/settings/failure guards; imported by validation tools |
| [process_metrics.py](process_metrics.py) | Shared process, memory and isolated CUDA-worker reporting helpers |

Use each executable's `--help` for its input and output options. PowerShell
launchers expose their parameters in the script header.

## Prerequisites and saved evidence

Install the normal scanner/native dependencies first. CUDA experiments need a
compatible NVIDIA driver, CUDA-enabled Open3D and, for custom kernels, the
optional [CuPy requirements](../requirements-cuda-fusion.txt). Keep the selected
device, native binaries, threads, calibration, raw input and reconstruction
settings fixed when comparing results. Install the optional
[benchmark dependencies](../requirements-benchmarks.txt) for Matplotlib chart
generation and HTTP checks.

Raw captures, meshes, fixtures, downloaded SDKs, DLLs, logs and runtime snapshots
stay local under ignored output directories. Published summaries and charts are
indexed in [Saved CUDA results](../docs/benchmarks/CUDA_REPORTS.md).

The measured historical proof reports retain their original paths and hashes.
Moving a helper changes its fingerprint even when its numerical body is the
same. Research timing requires fresh synthetic and real-trajectory proofs for
the relocated sources and actual environment, plus locally derived raw fixtures
and rebuilt pinned DLLs where required. Published summaries are evidence to
read; they do not authorize an unaudited adapter. Do not edit an old report's
hashes, disable a guard or reuse archived poses as live tracking seeds.
