# Scanner usability implementation

Implement the accepted review in dependency order. Preserve calibrated capture,
fresh-frame cadence, command barriers, and experimental defaults. Each completed
stage is a tested Git snapshot.

1. Capture workflow and layout: fixed actions, Automatic/Manual selection,
   Pause/Resume separate from Finish, persistent state, movement guidance,
   camera readiness and keyboard actions.
2. Session safety and recovery: protect unsaved captures, save before reset/close,
   restore server sessions, retry/resume after failure, retrieve the final mesh,
   responsive connection and operation-specific errors.
3. Views: reconstruction prominence, compact camera preview, visible Follow/Orbit,
   Color/Shape and Fit controls, matching depth inclusion, crop outline and legend.
4. Setup and output: collapsed advanced/experimental settings and connection
   details, input validation, one export dialog, one Open Model action, clear
   progress phases and nonmodal success notifications.
5. Integration: regression tests, local client/server synthetic scan, responsive
   layout checks at 960×600 and 1280×800, documentation and independent review.

Deliberate exclusions from the review remain exclusions: decorative themes,
unmeasured quality presets, invented completeness percentages and turntable
tracking. The capture model requires moving the Kinect around a stationary subject.

Baseline: `fdd978f` preserves the pre-existing measured calibration and capture work.
Baseline verification: 95 tests run, with two unavailable-hardware skips.

## Completed snapshots

| Stage | Snapshot | Result |
| --- | --- | --- |
| 1 | `86603cb` | Fixed primary actions, explicit capture modes, Pause/Resume, readiness, shortcuts and guidance. Collapsed setup sections and the export dialog landed here because the layout depended on them. |
| 2 | `ac5464e` | Save/discard/cancel protection, paused reconnect, retry/resume, asynchronous connection, atomic exports and final-mesh inspection. |
| 3 | `a044201` | Prominent reconstruction, camera aspect fitting, visible view controls, depth palette, crop outline and legend. |
| 4 | `56da9c8` | Clear operation phases, build-timeout reconciliation, retained-session recording safeguards, shortcut guards during connection/build, persistent disconnected guidance and exact restored crop display. |
| 5 | Integration/documentation snapshot | Updated workflow documentation and the synthetic client/server check; verified both window sizes and independently reviewed workflow/view behavior. |

## Verification

The final regression suite and end-to-end check run with an offscreen Qt backend,
four OpenMP threads, and a 5000-block reconstruction budget:

- Final suite: 144 tests run; 142 passed and 2 CUDA tests skipped.
- Synthetic client/server scan: passed, including final inspection and all export formats.
- Ruff and Git whitespace checks: passed on changed Python files.

```sh
QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=4 KINECT_BLOCK_COUNT=5000 python -m unittest discover -s tests -v
QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=4 KINECT_BLOCK_COUNT=5000 python scripts/check_scanner.py
```

The end-to-end check covers real loopback HTTP/WebSocket capture, reconstruction,
preview, final rebuild, PLY/OBJ exports, textured GLB/OBJ ZIP, lossless session
export, and Qt final-model inspection. It uses synthetic input and no Kinect.

Native Qt renderings were inspected at 960×600 and 1280×800: ready, capture,
paused, depth/crop, expanded diagnostics, scrolled setup, export and session
protection. Primary actions stay outside the settings scroll area; no horizontal
scrolling is required. Review images are generated locally in `logs/ux-review/`
and remain outside Git. The minimum-size action layout is also regression-tested.

Independent reviews covered worker/session recovery, workflow protection and
view/state consistency. Live Kinect ergonomics and scan quality still require
a physical scan; these checks do not measure them. CUDA checks depend on NVIDIA
hardware and are skipped on this Mac.

## Follow-up: cancel without finishing

**Cancel Scan** sits beside **Finish Scan** and works for active or paused
capture, even when camera frames stop. Save Session / Discard / Cancel protects
unsaved captures. It resets the server without building a mesh and returns to
setup after acknowledgement; **Start Scan** begins the next scan explicitly.
If cancellation times out, capture remains paused while the client checks the
server, retries failed status requests, and restores the confirmed state.

Verification: 49 workflow/dialog/cadence/worker tests passed; the loopback
client/server check passed cancellation followed by a fresh scan, final build
and export. The new button fits the 960×600 layout. An independent workflow
review found a status-polling recovery issue, which was fixed and covered by
regression tests.

## Follow-up: stacked live cameras

Scan view keeps the fused cloud on the left and stacks color above depth on the
right. Both previews retain their aspect ratios and show camera delay/failure
overlays. The depth preview uses the same calibrated clipping, restored crop
and legend as the full Depth view. Color and Depth views still use a single
large preview.

Verification: 49 existing workflow, depth, cadence and calibration tests passed.
Native Qt renders were inspected at 960×600 and 1280×800, including expanded
cloud details, paused capture, full Depth view and scrolled setup.

## Follow-up: capture sound

A bundled 70 ms cue confirms successful captured-frame upload in Automatic and
Manual modes. **Capture sound** in the toolbar toggles it and remembers mute
across launches. Failed/rejected uploads and old-session acknowledgements stay
silent. Batches use one cue, rapid confirmations do not overlap, and mute,
reset, cancellation and shutdown clear any cue waiting for its audio file to load.

Verification: 48 feedback/workflow/cadence/worker tests passed. Qt loaded the
bundled PCM WAV successfully. The loopback check verifies that accepted uploads
request sound confirmation, with playback suppressed during synthetic scans.
