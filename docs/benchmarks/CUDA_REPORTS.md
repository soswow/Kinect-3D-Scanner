# Saved CUDA results

The [current pipeline report](cuda-pipeline/REPORT.md) contains the final CPU/CUDA
archive comparisons and charts. The [initial fusion report](cuda-study/REPORT.md)
is historical. [CUDA_EXPERIMENTS.md](../CUDA_EXPERIMENTS.md) explains the field
configuration, rejected approaches, remaining research and reproduction commands.

The October field study adds a
[research summary](field-study-v1/research-summary.json) and
[installed Final allocation validation](field-study-production-v1/validation-summary.json).
[Field measurements](../FIELD_CUDA_RESEARCH.md) distinguish selected-view
processing gains, Live preview tradeoffs, and allocation payload sizes.

The [script index](../../scripts/README.md) distinguishes maintained tools,
active research and archived experiments. The catalog maps old report paths to
their current locations; source relocation requires fresh execution proofs.

The JSON summaries under `cuda-pipeline/` preserve component results and measured
source/runtime fingerprints. Each `report-manifest.json` hashes the published
files and their original sources. Published text uses LF line endings; the
ignored originals remain unchanged.
Reported absolute paths and process details describe the measurement machine;
they are provenance, not portable instructions or currently running services.
The original reports' runtime references describe their measurement-time servers.

These are report snapshots, not sufficient authority for an unaudited research
adapter. Full prerequisite proofs must be generated locally before research
timing. Raw camera captures, point-cloud arrays, meshes, fixtures, downloaded
SDKs, DLLs, logs and runtime ZIP snapshots remain in ignored directories.

Historical source fingerprints bind the measured file bytes and Windows path
conventions. A different checkout, line-ending policy, platform or source edit
can invalidate a prerequisite proof. Generate fresh proofs for that environment;
do not change an old report's fingerprints to bypass a research guard.

After generating the local summaries and charts, refresh these snapshots with:

```powershell
python scripts/publish_cuda_reports.py
```

The publisher uses an explicit file list and only copies existing results. It
does not run numerical work or contact the scanning server.

To generate plots, install `python -m pip install -r requirements-benchmarks.txt`
in the existing scanner development environment. CUDA measurements additionally
need the supported CUDA/Open3D setup and `requirements-cuda-fusion.txt`. Report
snapshot directories retain LF text through Git so their manifest hashes remain
valid across checkout line-ending settings.
