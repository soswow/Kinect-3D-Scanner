# Repository working agreements

## Documentation and investigation artifacts

- Start with `docs/README.md` and `scripts/README.md`; read the linked material
  for the task rather than ingesting every report. Code and tests establish
  current behavior; dated measurements describe only their recorded revision.
- Keep scratch notes, agent handoffs, review ledgers, logs, captures, fixtures,
  generated profiles, downloaded dependencies and build products in ignored
  `benchmark-output/<investigation>/` or outside the checkout. Add an ignore rule
  before using a new scratch directory. Do not commit them as documentation.
- Before retaining a document, identify its reader, concrete future use and
  unique knowledge. Prefer updating an existing guide. Put current behavior and
  decisions first; move detailed evidence behind links. Mark historical,
  experimental, rejected and unexecuted work explicitly. There is no arbitrary
  length limit, but a long reference needs a short summary and section navigation.
- Close an investigation by preserving the result, its scope, evidence source,
  failure/negative result and remaining uncertainty in the relevant existing
  guide or research protocol. Omit the work diary. Consolidate superseded reports
  once their unique findings are preserved; use a commit/path link for older
  detail that has no current consumer. Do not add a new dated report by default.
- Publish benchmark evidence selectively: bounded scalar summaries and useful
  charts, with input/source/environment provenance and limitations. Full run
  matrices, per-frame/query traces and duplicate snapshots stay local. Inspect
  public artifacts for private captures, credentials and personal paths; preserve
  existing historical fingerprints without treating their paths as instructions.
- Review all incoming links, imports, test fixtures, catalog entries, native
  resources and source fingerprints before deleting or moving a tool or record.
  Several archived research modules are active dependencies. Never rewrite old
  measurement hashes or bypass proof guards to make a cleanup pass.

## Retaining scripts

- An investigation tool belongs in the repository only when it reproduces
  retained evidence, supports a named next experiment, provides a reusable
  diagnostic, or is a required dependency. Remove disposable orchestration and
  ad hoc extractors after folding their useful behavior into an existing tool.
- Use `scripts/` for supported workflows, `scripts/benchmarks/` for reusable
  component measurements and `scripts/research/` for bounded experiments.
  `scripts/research/archive/` retains required foundations and useful earlier
  experiments; placement alone does not mean a file is unused or validated.
- Give a retained tool a clear purpose, input/output contract, prerequisites and
  side effects. New Python CLIs must provide `--help` before importing optional
  numerical/UI packages or doing work, accept explicit input/output paths, and
  default generated artifacts to ignored directories. Avoid developer-specific
  absolute paths and implicit contact with a running server. Importable helpers
  must not launch work at import time. Preserve frozen experiments unless an
  explicit new protocol and fresh proofs justify changing them.
- Make every retained entry point discoverable in the relevant README; shared
  helpers and experiment families may be documented together. Use
  `python scripts/list_tools.py --search <topic>` for the complete source
  inventory. Keep new research resources in `field-tool-catalog.json`; preserve
  the original proof-pinned `tool-catalog.json` bytes.
- Validate the tool's interface and appropriate dependency/layout contracts.
  Run numerical or HTTP checks only in their documented isolated environment;
  preserve active captures and unsaved server work. Performance claims require
  matched inputs/settings, correctness evidence and clearly stated timing scope.

## Finishing a task

Unless the user explicitly requests otherwise, a task is complete only after
its validated changes are committed, integrated, pushed, and installed:

1. Run the checks appropriate to the changes and commit the completed task's
   changes. Preserve unrelated work, including changes in other worktrees.
2. When working in a Git worktree or task branch, merge the task into the local
   default branch (`master` in this repository). Use its existing checkout,
   preserve unrelated uncommitted changes, and resolve routine conflicts within
   the task's scope.
3. Push the integrated default branch to `origin`. Verify that the remote branch
   contains the completed changes. This is standing authorization to commit,
   merge locally, and push; no separate confirmation is needed.
4. On macOS, build the client from the final integrated source using
   `scripts/build_macos_client.py` and the configured packaging environment.
   Install the entire resulting `Kinect 3D Scanner.app` bundle into
   `~/Applications/Kinect 3D Scanner.app`, replacing the previous installed
   build. A build left only in a worktree's `dist` folder is unfinished work.
5. Run the installed app's `--check` from outside the checkout and verify that
   the installed executable matches the checked build. If the old app is
   running, restart it with the installed build when it is idle. Preserve active
   captures and unsaved work; do not discard them to restart. For UI changes,
   verify the changed behavior in the installed app.
6. Report the pushed commit and installed app path. If a concrete blocker
   prevents pushing, building, installing, or safely restarting, report the
   blocker and remaining steps instead of declaring completion.

Installing the Mac client does not update a separately running reconstruction
server. State clearly when that server still needs to pull the changes and
restart.
