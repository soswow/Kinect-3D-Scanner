# Scanner tools

Run commands from the repository root in the configured scanner environment.
Start with the workflows below. For every CLI, helper, native source and research
protocol, use the standard-library finder; it reads source without importing or
executing tools and works from any current directory:

```powershell
python scripts/list_tools.py --search session --kind cli
python scripts/list_tools.py --group benchmarks
python scripts/list_tools.py --group research --search marker --json
```

Directory groups describe intended use. Research protocols, rather than a
catalog's historical `active` label, establish what was measured, rejected or
never executed. See [documentation](../docs/README.md) for current behavior.

| Location | Reader and purpose |
|---|---|
| This directory | Users/developers: launch, calibration, replay, diagnostics and complete-session evaluation |
| [benchmarks/](benchmarks/README.md) | Developers: reusable component measurements and offline geometry checks |
| [research/](research/README.md) | Researchers: experiment families, protocols and retained result producers |
| [research/archive/](research/archive/README.md) | Researchers: previous approaches and foundations still imported by later experiments |

## Launch, calibration and packaging

| Entry point | Inputs, effects and prerequisites |
|---|---|
| [start_server.ps1](start_server.ps1) | Windows source server in `.venv`; forwards server arguments |
| [start_cuda_server.ps1](start_cuda_server.ps1) | Windows CUDA server with explicit recipe/options; starts a background supervisor and writes `logs/` |
| [allow_server_firewall.ps1](allow_server_firewall.ps1) | Administrator action: permit the configured TCP port for LAN clients |
| [check_camera.py](check_camera.py) | Own the connected Kinect briefly and check frames/shutdown; close other camera users first |
| [calibrate_camera.py](calibrate_camera.py) | Fit intrinsics or inspect saved registered RGB-D target evidence; explicit output path |
| [calibrate_accelerometer.py](calibrate_accelerometer.py) | Guided physical sensor measurements or offline fitting; see [sensor reference](../docs/KINECT_ACCELEROMETER.md) |
| [build_macos_client.py](build_macos_client.py), [build_macos_icon.py](build_macos_icon.py) | macOS/Xcode packaging environment; build/check `dist/` bundle or regenerate icon assets |

The CUDA preview recipe and supported environment are in the
[project README](../README.md); measured tradeoffs and limits are in
[Field CUDA research](../docs/FIELD_CUDA_RESEARCH.md). Launchers expose PowerShell
parameters in their headers. Python CLIs expose `--help`; older tools may need
the numerical/UI dependencies even for help. The finder needs none of them.

## Recorded data, diagnostics and evaluation

| Entry points | Purpose |
|---|---|
| [download_dataset.py](download_dataset.py), [convert_dataset.py](convert_dataset.py), [replay_scan.py](replay_scan.py) | Download public data into `datasets/`, convert to a local recording or replay; reference poses are scoring inputs only. `replay_scan.py --server` resets the selected server: use a dedicated empty instance |
| [inspect_session.py](inspect_session.py), [diagnose_session.py](diagnose_session.py) | Read saved session ZIPs; write summaries/contact sheets or isolated reconstruction diagnostics |
| [reconnect_session.py](reconnect_session.py) | Rebuild a saved session with verified fragment reconnection; writes local output |
| [profile_capture.py](profile_capture.py), [profile_session.py](profile_session.py) | Measure recorded capture or an isolated session replay; `profile_session.py --finish --finish-only` measures cold depth Finish without archived pose seeds |
| [profile_backends.py](profile_backends.py), [profile_session_backends.py](profile_session_backends.py), [profile_cuda_pipeline.py](profile_cuda_pipeline.py) | Sequential matched backend/workflow comparisons in separate processes |
| [compare_session_profiles.py](compare_session_profiles.py), [evaluate_surface.py](evaluate_surface.py) | Fixed-coordinate reconstruction/triangle comparisons; matched references do not establish absolute accuracy |
| [evaluate_session_tracking.py](evaluate_session_tracking.py), [evaluate_archived_visual_tracking.py](evaluate_archived_visual_tracking.py) | Recorded camera/server tracking comparisons with explicit sparse-capture limitations |
| [evaluate_session_confidence.py](evaluate_session_confidence.py) | Held-out depth consistency at saved poses; no ground-truth accuracy claim |
| `benchmark_{quality,bundle_adjustment,pose_refinement,confidence_fusion,icp_source_cache,live_tracking,native_fusion,native_kernels}.py` | Reusable quality, pose, confidence, tracking and native-kernel measurements; find exact inputs/options with `list_tools.py --search benchmark --group workflow` |

Keep generated outputs under ignored `benchmark-output/<investigation>/`.
Captures, meshes, arrays, SDKs, native libraries and full execution proofs stay
local. See [benchmark evidence](../docs/benchmarks/README.md) for selected public
snapshots; reading them does not require running their hardware experiments.

## Validation and historical report formatters

[check_scanner.py](check_scanner.py), [check_cuda_server.py](check_cuda_server.py)
and [check_cuda_native_server.py](check_cuda_native_server.py) mutate a **dedicated
empty test server** through synthetic or bounded archived captures, build/export
and reset/settings requests. They check initial state and restore test settings;
do not run them while another client is using that server.
[check_cuda_backend.py](check_cuda_backend.py) runs backend tests and hardware
probes in the configured development environment. All numerical benchmarks can
consume substantial CPU/GPU resources; run outside active scanning.

[summarize_cuda_study.py](summarize_cuda_study.py) and
[summarize_cuda_pipeline.py](summarize_cuda_pipeline.py) reproduce the original
October study: fixed hardware/date labels and known mode matrices are historical
formatting assumptions. Adapt them for a new study before publication. Outputs
default to ignored storage. [publish_cuda_reports.py](publish_cuda_reports.py)
copies an explicit selection of closed summaries/charts into documentation;
review that selection before running it. It never runs a benchmark.
Publication writes into `docs/benchmarks/` by default; inspect the diff afterward.
The pipeline and session studies publish sequentially, so a second-stage failure
can leave the first study published.

[http_check_safety.py](http_check_safety.py), [process_metrics.py](process_metrics.py)
and [tool_paths.py](tool_paths.py) are shared helpers. The proof-pinned
[relocation catalog](tool-catalog.json) maps old resource paths; the additive
[field catalog](research/field-tool-catalog.json) covers later research resources.
Source moves/edits can invalidate historical proofs. Follow the research
protocol to rebuild local inputs/libraries and generate fresh proofs; do not
rewrite old fingerprints or use archived poses as live tracking seeds.
