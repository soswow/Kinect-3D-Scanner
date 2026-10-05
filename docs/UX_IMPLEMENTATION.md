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
Baseline verification: 95 tests passed, with two unavailable-hardware skips.
