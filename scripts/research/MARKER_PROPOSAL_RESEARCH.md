# Native marker proposals through original geometry gates

This is a **source-only, unexecuted prototype**. It changes only a proposal
policy inside an offline Finish scope. Production registration, SIFT features,
matching, geometric acceptance and graph search remain unchanged.

`marker_proposal_scope.py` decodes `DICT_4X4_250` on the original native RGB.
The feature identity is `(decoded ID, canonical corner index)`. It makes no
assumption about board dimensions, square size, layout or world coordinates.
For every corner it searches all original calibrated, visible depth-grid
projections in their original row-major order, accepts association only within
3 native RGB pixels, and applies the original measured-center/3x3 depth
support. Repeated decoded IDs and corners reusing a depth pixel are rejected.
The unchanged CUDA lookup fully CPU-shadows every query and exact distance
bits; that audit work is included in Finish.

`MarkerSeedScope` first checks the original SIFT proposal. When it is absent,
the marker identity correspondences may supply an original PnP proposal as
`initial` to the captured original `_local_match`. The delegated SIFT check
must remain absent exactly once. Thus marker PnP cannot take the original
direct visual-proposal acceptance branch: the returned local pose must come
from original reciprocal ICP, normal conditioning, motion and held-out gates.
Original SIFT features, descriptor caches and visual witnesses retain their
identities and values. Global proposals, independent-camera bridge checks,
information matrices and competing-proposal ambiguity rules are unchanged in
the local policy.
Existing sequential-camera edges retain the original, separately reported
single-camera authority; they are not relabeled independent loop witnesses.

The separate `--seed-scope global` policy inserts at most four marker seeds
after the original TOP4 visual and two global RANSAC proposals, before
original uniqueness and verification. Shared ID/corner support ranks only the
additional seeds; it does not change pair ranking, frontier, graph budgets or
the original proposals. A camera-to-camera seed is converted by
`target_camera_local_pose @ seed @ inverse(source_camera_local_pose)` into the
original fragment coordinate frames. Every candidate still needs the original
full bridge, independent camera/SIFT witnesses, competing-proposal ambiguity
and information-matrix gates. The entire original function AST is pinned;
removing the one insertion reconstructs its original body. This scope enters
before the original output observers, which retain their original shared globals.

`profile_marker_finish.py` materializes the same closed fresh raw Live
checkpoint for original and marker Finish. It validates the current component
preflight, checkpoint payloads, archive, runtime, all fifteen original
checkpoint artifacts, and four separate marker artifacts. It emits complete
original registration/gate traces, marker provenance/dispatch traces, actual
surface geometry and Final allocation planning when the checkpoint enabled it.
Proposal transport, evidence or trace failures are latched and re-raised after
restoration even if ordinary Finish exception handling hides them.

After explicit hardware allocation and source review, the first C5 local pair is:

```powershell
python scripts/research/profile_marker_finish.py --mode original --seed-scope local --lookup cuda-audit --checkpoint benchmark-output/field-cuda-study/resident-finish-v4/chest-5-serial-checkpoint/checkpoint.json --output benchmark-output/field-cuda-study/marker-finish-v2/chest-5-local-original.json --component-synthetic benchmark-output/field-cuda-study/device-layout-v2/synthetic-proof.json --component-bridge benchmark-output/field-cuda-study/device-layout-v2/bridge-audit.json --run-allocated

python scripts/research/profile_marker_finish.py --mode marker --seed-scope local --lookup cuda-audit --checkpoint benchmark-output/field-cuda-study/resident-finish-v4/chest-5-serial-checkpoint/checkpoint.json --output benchmark-output/field-cuda-study/marker-finish-v2/chest-5-local-marker-audit.json --component-synthetic benchmark-output/field-cuda-study/device-layout-v2/synthetic-proof.json --component-bridge benchmark-output/field-cuda-study/device-layout-v2/bridge-audit.json --run-allocated
```

For a separately declared global pair, use `--seed-scope global` in both
commands and fresh `chest-5-global-original.json` / `chest-5-global-marker-audit.json`
outputs. Both controls retain the checkpoint's explicitly bound scoped RANSAC
thread policy. Older checkpoints whose fifteen producer pins changed are
rejected; they cannot be silently rehashed or promoted.

Marker Finish includes lazy decoder creation, CUDA compilation/self-checks,
calibrated projection, every corner CPU shadow, depth support, duplicate SIFT
absence checks and all original geometric work. Nested provider timers cannot
be added together. Imports, checkpoint materialization, surface serialization
and final provenance/report closure are outside `finish_s`.

Changed fragment coverage, call order and accepted poses are possible by
design. The original C5 partial surface is a comparison control rather than
ground truth for newly recovered views. Inspection must distinguish common
coverage consistency from newly connected, independently verified geometry.
No existing resident/checkpoint proof authorizes this proposal policy, no
quality authority is minted here, and marker audit timing is not a production
speedup or camera throughput result.

A future faster CPU mapping or token-gated unaudited CUDA mapping needs a
separate declared policy and source/runtime pins. Before timing that route,
fresh same-policy CPU/CUDA mapping and whole-Finish quality audits must prove
the exact associations, original gate decisions, coverage and surface. The
current fully shadowed helper/report cannot unlock that future route.
