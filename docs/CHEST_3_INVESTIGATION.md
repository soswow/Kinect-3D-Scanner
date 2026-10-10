# Chest 3: tracking and lost final coverage

Historical investigation: 6 October 2026, revision `030486c`. The unchanged
126-capture archive is `chest-3-scan-session_20261006_215208.zip` (SHA-256
`08b8f2615348dfd6ceb9a14b8c9b964ea05efbc3ca0351b8f3d94783f9c4a3e4`).
Capture numbers here are one-based; JSON indices are zero-based.

## Findings that matter beyond this recording

- Live accepted 68 and rejected 58 captures over 563.92 seconds, with 15 loss
  episodes. Median stored interval was 4.65 seconds and processing 2.47 seconds.
  Camera input at 10 fps did not mean server tracking at 10 fps.
- All 126 pairs passed the 20 ms guard (-16.51 to +16.59 ms), and selected frames
  had 93-97% valid depth. Packet timestamps do not establish exposure timing;
  missing pixels did not explain these failures.
- Historical model ICP could look convincing while disagreeing with raw-camera
  alignment. Capture 89 had 96.4% overlap/9.1 mm RMSE but differed by
  272 mm/14.75 degrees from raw alignment whose reciprocal cycle was 1.3 mm.
  Broad planes/repeated texture can produce low-residual wrong poses.
- Replay reproduced 13 transition rejections and two pose jumps. Ten checks
  failed model/raw agreement, two overlap and two cycle translation (overlapping
  categories); none failed gap/normal diversity. Recomputed/archived poses are
  not truth. Loosening guards would also admit these large disagreements.
- Logs had 70 invalid-magic, 27 inconsistent-flag and 19 resynchronization
  warnings. The archive cannot link them to observations; USB faults were not
  established as the cause of reproduced pose disagreements.

## Why Finish removed surfaces

Original Finish exactly reproduced the downloaded 52,722-vertex/96,845-triangle
mesh. It retained captures 1-16, corrected 13, excluded 52 live-accepted views
and recovered none. Twenty of 21 fragments lacked an anchor connection despite
37 bridges among later fragments. All 91 eligible pairs were tested and required
fusion was 2,333/10,000 blocks: connectivity loss, not search or memory exhaustion.

Display also required less evidence than final extraction: live weight 0.01,
inspection 0.5, final 2.0, then cleanup below 30 triangles. Fractional confidence
is not a frame count. With the same 68 saved poses, weight 2 gave 332,064 triangles
versus 2,141,548 at 0.01; cleanup removed only 5,088 at weight 2. The further
reduction to 96,845 came from changed views/poses. More faces (including floor
and drift artifacts) do not prove accuracy.

The follow-up separated camera tracking from fusion, retained visual identities,
recent references and storage-boundary continuity, and reported final coverage.
[Continuous tracking](CONTINUOUS_VISUAL_TRACKING.md) and
[algorithm review](ALGORITHM_REVIEW.md) later retained 120/126. Their different
source/seed runs are separate experiments; this report describes the old failure.

## Evidence and reproduction

[Compact measurements](benchmarks/chest-3-summary.json) are public. Detailed
artifacts originally lived in `/Users/sasha/hobby/xbox360/chest-3-session-analysis/`
and are absent from a checkout. Use your own input path:

```sh
python scripts/diagnose_session.py /path/to/session.zip --output-dir benchmark-output/chest-3 --export-meshes
python scripts/reconnect_session.py /path/to/session.zip --use-pose-seeds --output-dir benchmark-output/chest-3/final
```

Diagnosis compares unvalidated saved poses; reconnection revalidates guesses
against raw measurements. This investigation changed no production policy and
has no independent reference trajectory/surface.

[Full historical record](https://github.com/soswow/Kinect-3D-Scanner/blob/762a6dac3b5a48b5865d382bbf18cd1746c37999/docs/CHEST_3_INVESTIGATION.md) preserves the complete tables, old
thresholds, exact fingerprints and validation chronology.
