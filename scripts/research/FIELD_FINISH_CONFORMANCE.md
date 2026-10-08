# Actual-input field Finish conformance

The current field and quality schema is **v2**. The preserved v1 C6 native/audit
quality passed, but its first timing attempt failed before registration: graph
edge triples were computed as Python tuples and saved as JSON lists. V2 creates
those triples as lists, so a freshly computed record equals its closed JSON
record exactly. The equality guard, all numerical/discrete criteria and the
original 15 checkpoint sources remain unchanged. V1 reports and their exact
21 measured sources remain immutable under `field-conformance-v1`; v2 needs
fresh native, audit, quality and timing outputs and cannot reuse v1 authority.

This is a new offline research protocol. It does not repair, relax or relabel
the failed strict native-versus-resident Finish histories. The production
server, original acceptance bodies, geometric settings, NN mathematics and
Eigen pose solver remain unchanged.

Every audited resident call has complete original CPU nearest hit/miss
shadows and a complete original CPU ICP result shadow on the **same actual**
source/target point and normal bytes, unrounded initial matrix, stages and
thread policy. Original CPU ambiguity resolution remains authoritative.
Malformed results, full-call CPU retry, missing evidence or failed cleanup
invalidate the experiment. A verified bulk CPU auditor may accelerate the
shadows; its independent closed compatibility proof remains required.

An independent native control materializes the same freshly measured private
Live checkpoint, with its raw/prepared inputs, CPU/cache state and logical VBG
contents. No archived exported trajectory is used as a Live seed. Both Finish
phases use the same declared proposal policy and source/runtime pins.

The new comparison asks whether the actual accepted reconstruction agrees:
final view/fragment membership, retained graph connectivity and independent
camera witnesses, original stage acceptance, and physical final geometry.
An output-only observer captures each original optimizer's node/edge identity,
uncertainty, confidence, information and poses before and after the unchanged
original optimizer. A second observer copies the unrounded FP64 engine poses
after the original transactional final build succeeds; it returns the same
original result object. Applied bundle adjustment is outside the initial
scope until its complete tracked-landmark witness inventory is recorded.

The fixed per-view final bounds are **0.5 mm translation and 0.1 degree
rotation**. Fixed-coordinate triangle surfaces use **30,000 samples per
surface, p95 at most 0.5 mm, and precision/completeness at least 99.9% within
5 mm**, without scale or trajectory fitting. Matching an incomplete native
reconstruction is conformance to that control, not proof of complete capture
coverage or ground truth.

Intermediate native proposal order, RANSAC outputs, discarded candidates and
information-matrix memberships can differ and are retained as diagnostics.
Their history is not certified as equivalent. The old strict reports remain
failed under their original thresholds.

Unaudited timing requires a new, independently closed field quality proof and
the exact **audited resident trajectory**: original source/target/normal/seed
signatures before GPU work; its own ordered result, gate and graph events;
original correspondence mapping; terminal event counts; and final unrounded
pose evidence. Missing/reordered events or a seed/input drift stop the run.
An old component token or old strict Finish authority cannot authorize timing.

The default `--proposal-policy original` preserves original proposal inputs.
`--proposal-policy canonical-fresh-1um` is a separate declared proposal policy:
only private coarse points are quantized to a micrometre and have fresh
normals/FPFH built before the original global proposal function. Original
train/held-out data, proposals/checkers/caps, ICP and acceptance bodies remain
in place. Both native and resident controls must declare that same option;
it is never silently combined with marker proposals or Final allocation.

All observation, optional canonical preparation, resident setup and field
timing-authority validation are charged inside `finish_s`. Checkpoint loading,
post-Finish source closure and report/geometry writing are outside it.
Nested stage timers are explanatory and are not subtracted from measured wall.
Audit wall includes CPU shadows and is diagnostic rather than a speed result.

Run only with an allocated hardware slot, frozen original and new sources,
closed current-layout component and bulk proofs, and fresh ignored output
paths. The comparator and validator are part of the new source inventory.
These commands are templates for the preserved C5 serial checkpoint; use the
checkpoint's exact archive/settings/seed/RANSAC/preplan policy.

```powershell
$env:OMP_NUM_THREADS='8'
$py='python' # Active CUDA-enabled environment selected by the parent runner.
$raw='F:/Projects/coding/Kinect-3D-Scanner/export/chest-5-scan-session.zip'
$checkpoint='benchmark-output/field-cuda-study/resident-finish-v4/chest-5-serial-checkpoint/checkpoint.json'
$proofs='benchmark-output/field-cuda-study/device-layout-v2'
$bulk='benchmark-output/field-cuda-study/bulk-nn-audit/old-trajectory-dual-proof.json'
$out='benchmark-output/field-cuda-study/field-conformance-v2'

& $py scripts/research/profile_field_finish_conformance.py --proposal-policy original -- $raw --run-allocated --mode native --checkpoint $checkpoint --component-synthetic "$proofs/synthetic-proof.json" --component-bridge "$proofs/bridge-audit.json" --ransac-threads 1 --final-preplan --output "$out/chest-5-native.json"
& $py scripts/research/profile_field_finish_conformance.py --proposal-policy original -- $raw --run-allocated --mode audit --checkpoint $checkpoint --component-synthetic "$proofs/synthetic-proof.json" --component-bridge "$proofs/bridge-audit.json" --bulk-audit-proof $bulk --ransac-threads 1 --final-preplan --output "$out/chest-5-audit.json"
```

The separate comparator must close `chest-5-quality.json` from the new native
and audit `.conformance.json` envelopes, their raw trace/profile/geometry and
the declared graph/pose/surface gates. Only after it passes can timing run:

```powershell
& $py scripts/research/profile_field_finish_conformance.py --proposal-policy original -- $raw --run-allocated --mode timing --checkpoint $checkpoint --component-synthetic "$proofs/synthetic-proof.json" --component-bridge "$proofs/bridge-audit.json" --bulk-audit-proof $bulk --finish-audit "$out/chest-5-audit.conformance.json" --quality-proof "$out/chest-5-quality.json" --ransac-threads 1 --final-preplan --output "$out/chest-5-timing-0.json"
```

No full field timing has been established by preparing these tools. Even a
successful offline conformance experiment needs a separate production design
and field validation before it becomes a scanning backend.

The C6 original-policy v2 native/audit comparison passed its declared output
criteria, but its first timing run stopped before GPU registration call 228.
Only the unrounded initial matrix differed; the source/target points and
normals, stage recipe, thread policy and call context matched. The preceding
228 completed registration calls had identical input signatures and returned
transform values. The earliest recorded original CPU information-matrix
difference was about 4.55e-13; fragment graph optimization then produced poses
differing by at most 3.33e-16. The failed refinement loop used the original
relative fragment poses as its seed, without a feature proposal. The exact
consumed-input guard correctly rejected that seed drift.

The closed failed timing report and log remain evidence of this limitation.
Canonical coarse FPFH preparation alone does not address this relative-pose
seed path. Any future seed canonicalization or reduction/thread policy would
be a separately declared experiment with fresh controls and proof; the exact
timing guard and failed original native-history claims remain unchanged.
