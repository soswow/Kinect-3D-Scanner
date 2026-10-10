# Chest 4: recovery and returning-view evidence

Historical October 2026 result: 164 raw captures, 114 originally accepted live,
four retained by original Finish. Updated rebuild retained **162/164** with five
independent cycles in the surviving verified graph. Captures 52 and 110 remained
disconnected (one-based).

## Durable changes

Camera tracking kept five good references across bad/mistimed images; gaps over
0.75 seconds reset the chain. Automatic capture chose a paired observation among
the last five arrivals within 300 ms, preferring measured motion then sharpness.
This ranks candidates, not guarantees sharp images. Recovery allowed one fresh
outstanding probe with 100 ms minimum spacing.

Live retrieval ranked 40 accepted views (initial five, recent eight and spaced
landmarks), verified the best five, and raw recovery tried five recent accepted
observations. Finish tried five recent views, retained boundary witnesses and
used SIFT. Nonadjacent local motion allowances covered at most three capture
steps without relaxing residual/overlap/cycle thresholds. Feature identity,
reciprocal geometry, held-out depth and distinct camera witnesses authorize
legacy connections. See [continuous tracking](CONTINUOUS_VISUAL_TRACKING.md) and
[final registration](FRAGMENT_RECONNECTION.md) for current mode distinctions.

## Measured rebuild

| Measurement | Original | Updated saved-pose Finish |
| --- | ---: | ---: |
| Retained views | 4 | 162 |
| Vertices / triangles | 1,843 / 2,615 | 301,668 / 581,767 |
| Surface area | 0.0229 m² | 4.8427 m² |
| Voxel / confidence | 5 mm / 2.0 | 5 mm / 2.0 |

Fifteen of 17 fragments connected, nine links used earlier references and 28/98
pairs were tested without caps. Four returning bridges joined captures near
22-27 to 142-147 with independent color/depth witnesses. Optimization exceeded
raw checks, so measured-pose fallback revalidated surviving links and connectivity.
Five cycles survived; separate refinement found no further loops. This shows
coverage and returning evidence, not corrected accumulated drift or absolute accuracy.

Full server replay accepted 130 live (versus 114), reducing rejected runs 16 to
14. Finish recovered 32, corrected 129 and excluded none, reaching the same 162,
five cycles and two missing captures. Its 301,667 vertices/581,764 triangles used
9,120/10,000 blocks. The lower basket still had holes; added floor/wall coverage
contributed to mesh size. Missing intermediate images prevent archived validation
of sharp selection/reference retention on the original physical stream.

[Compact evidence](benchmarks/chest-4-tracking-recovery.json) records unchanged
ZIP SHA-256 `ab0465d56d05c172074075b056c22f10a3337c83b2e564c148d4f10ea64ce49f`.
The checkpoint ran 326 tests (324 passed/two CUDA skips). CPU runs overlapped,
so timing is not controlled performance evidence. Local
`benchmark-output/chest-4-improvement/` retains `sift-finish` saved-pose and
`final-after` live replay reports, meshes and matched comparisons.

[Full historical record](https://github.com/soswow/Kinect-3D-Scanner/blob/762a6dac3b5a48b5865d382bbf18cd1746c37999/docs/CHEST_4_TRACKING_RECOVERY.md) preserves the complete tables, old
thresholds, exact fingerprints and validation chronology.
