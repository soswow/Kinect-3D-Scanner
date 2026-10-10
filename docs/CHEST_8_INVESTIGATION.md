# Chest 8: false half-turn and measured boundary protection

Historical audit: 9 October 2026, `db0db2d`, unchanged
`chest-8-scan-20261009-145-captures.zip` (145 captures). No running server was
contacted. Capture numbers below are one-based; JSON indices are zero-based.

## Failure mechanism

| Boundary | Archived final rotation / displacement | Original live comparison |
| --- | --- | --- |
| 57 → 58 | 155.366° / 2.069 m | 57 → 60: 16.622° / 0.296 m (58 lacked a live pose) |
| 128 → 129 | 175.280° / 2.079 m | 10.915° / 0.107 m |

Middle fragments 9-15 acquired roughly 174-177° corrections. All image
orientations were 0°, matrices were proper rotations and final poses equalled
`fragment_to_world @ camera_to_fragment`: reconstruction introduced the jumps.
Reconnection retained 134 versus 103 live views. Refinement found no loops and
the old bundle pass refused over 128 views; neither created the flip.

Geometric-only bridge 1→14 (support `[18,116]`, `[20,106]`) anchored the false
orientation. Visual/depth bridge 15→16 had five witnesses but was disconnected;
final placement disagreed by 179.889°/0.833 m. Geometric 0→16/1→16 put the last
section near its old orientation. Fallback retained the initial spanning tree,
discarded the conflicting visual edge and reported no ambiguity. Multiple depth
witnesses could still match repeated box/floor geometry at a false orientation.

Saved 57→58 overlap was 0.205/0.189 and 128→129 0.102/0.103, both failing visual
witnesses. Fresh local 57→60 passed overlap 0.706/0.729, residual 11.98/11.96 mm
and 97/119 visual inliers. Fresh 128→129 passed 0.699/0.769, 12.16/14.73 mm and
163/175 inliers at 10.958°/0.111 m. The wrong bridge's depth passed but visual
identity failed. The old live 128→129 transform narrowly failed reverse RMSE
(15.30 mm): archived poses were not authority either. Symmetry is a supported
failure explanation, not an independent ground-truth trajectory measurement.

## Correction and checked reconstruction

Legacy fragments now preserve short-range RGB-D ties across boundaries, build
the forest in temporal/visual evidence order and check output boundaries even
after pruning their edges. Conflicting fallback fails before fusion, preserving
prior state/diagnostics. [Final registration](FRAGMENT_RECONNECTION.md) describes
these rules and the separate experimental depth mode.

Eight added temporal bridges included `[56,59]`/`[127,128]`, alongside three
storage bridges. Seven global pairs were tested versus 88 archived; fresh CPU
and archived CUDA timing cannot establish a kernel speedup. Replays from original
and already-flipped seeds both retained 143/145 (indices 25/42 unconnected).
Already-flipped rebuild recovered nine, corrected 82, excluded none of the
original 134 and required 15,926 blocks at 5 mm.

Rebuilt boundaries were **23.490°/0.435 m** and **10.958°/0.111 m**. All 11
retained temporal/storage boundaries passed post-export visual/depth checks.
Largest rotation was 38.518° across a missing capture. All 290 image CRCs/sizes
and timestamps matched; the new ZIP passed full CRC checks. This repairs the
observed half-turn without certifying every pose or surface.

## Loops found versus corrections applied

[Camera refinement](POSE_REFINEMENT.md) selected 64 keyframes/22 loops: nine
fragment supports, eight appearance, five geometry; `[0,140]` passed. All 11
boundaries were protected. Independent loss improved 1.984983%
(`0.000803339241838256` → `0.000787393090907004` m²), below the unchanged 2%
minimum, so refinement was refused. Earlier interpolation failures motivated
all-output validation and anchored world-frame correction interpolation.

[Bundle adjustment](JOINT_RGBD_REFINEMENT.md) removed the total-view cutoff while
bounding the solver at 24 cameras/800 landmarks/96 pairs and depth cache at eight
views. It found 24 pairs/483 three-view landmarks, but 13 selected cameras,
including the anchor, lacked 24 observations; no correction applied. Finish
still retained 143/145 and the boundary repair.

## Provenance

[Compact loop evidence](benchmarks/chest8-loop-refinement.json) is public.
Ignored `benchmark-output/chest8-investigation`, `chest8-improved` and
`chest8-refinement` retain audits, checks, logs and plots; `temporal-fixed` and
`refinement-checked` session ZIPs remain local exports. Environment:
Open3D 0.20.0/OpenCV 5.0.0/NumPy 2.5.3, four threads. The marker-insertion adapter
was rebound as policy v2; old v1 proofs were unchanged and do not authorize it.

```powershell
$env:OMP_NUM_THREADS = '4'
$env:KINECT_NATIVE = 'off'
$env:KINECT_CUDA_INPUT = 'off'
$env:KINECT_CUDA_REGISTRATION = 'cpu'
python scripts/reconnect_session.py /path/to/original.zip --use-pose-seeds --device cpu --save-session --output-dir benchmark-output/chest8-rebuild
```

No available saved trajectory is independent ground truth.

[Full historical record](https://github.com/soswow/Kinect-3D-Scanner/blob/762a6dac3b5a48b5865d382bbf18cd1746c37999/docs/CHEST_8_INVESTIGATION.md) preserves the complete tables, old
thresholds, exact fingerprints and validation chronology.
