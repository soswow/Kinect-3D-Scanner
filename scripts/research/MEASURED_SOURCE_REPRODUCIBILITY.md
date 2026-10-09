# Measured Windows source bytes

The preserved research baseline uses normal Git text blobs. Git checkout can
change their line endings, while the numerical and component proof manifests
bind the measured **raw** bytes. Nine measured core files contain mixed LF and
CRLF lines. `measured-source-layout.json` records the raw and normalized hashes,
byte counts and compact one-based LF line ranges for 74 exact source paths,
including the complete 48-file core inventory.

Use research baseline commit `fb9069d33cd12efb3b305054934fea28ffbc1959`
in a separate checkout for these historical experiments. The subsequent
production allocation change deliberately invalidates the old core guards.
On that preserved baseline, first check:

```powershell
python -S scripts/research/restore_measured_source.py
```

If only newline bytes differ, explicitly restore them and check again:

```powershell
python -S scripts/research/restore_measured_source.py --restore
python -S scripts/research/restore_measured_source.py
```

The helper accepts only its SHA-pinned bundled manifest. It validates every
normalized source and the exact core inventory, constructs and hashes all
target bytes before writing, and rejects changed production sources or added
core files such as `fusion_allocation.py`. It changes newline bytes only. It
never accepts a user-supplied manifest, source archive, arbitrary target root
or replacement code. Default checks are read-only; `--restore` has not been
run against the measured checkout while experiments are active.

Each write uses an atomic same-directory replacement. If a write fails, the
helper rolls back completed replacements and retains the original exception;
rollback failures are reported explicitly. Concurrent external edits are
preserved rather than overwritten. Recovery requires exclusive ownership of
the source files; this is not a transaction over other processes or hardware.

Restoring source bytes does not recreate installed binaries, GPU/driver state,
raw session data or private fixtures. The original runtime/input/proof guards
still apply. This manifest does not authorize new production source or relabel
historical measurements. Run historical experiments from their preserved
baseline commit; generate fresh evidence for subsequent production changes.
