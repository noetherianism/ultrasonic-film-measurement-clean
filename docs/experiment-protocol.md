# Experiment protocol

This procedure is a starting point for the next physical validation session. The apparatus was stored and was not connected during the refactor. Offline work can check replay and software behavior; it cannot certify a new installation's coupling or measurement accuracy.

## Before connecting

Select one software revision and record it in the session notes. Choose an independent thickness reference and the accuracy/repeatability needed for the experiment. Keep one operator or agent process in control of both instruments; stop other acquisition programs before starting.

Record the transducer and mounting position, substrate geometry, coupling method, cable and probe connections, reference measurement method, water temperature if available, and instrument configuration. A photo or sketch can help reproduce geometry, but numerical observations and waveform files are the measurement record.

The original hardware commands use a specific generator amplitude and mux profile. Confirm that these match the apparatus before starting `film-live`, which turns TX on. Begin with `film-diagnostics` if the receiving chain needs inspection with TX off.

## Establish the session reference

1. Assemble the setup consistently and inspect raw signals for clipping, unstable timing, or an absent expected echo. Read back the actual sample rate and relevant instrument settings.
2. Measure a sufficiently thick water layer with an independent reference so that the acrylic and water echoes are distinguishable. Record the reference value, method, and uncertainty.
3. Compare the measured echo delay and derived thickness against that reference. State the sound speed or temperature used; do not automatically reuse the inherited sound speed for new conditions.
4. Record a fresh thick-water baseline without moving the transducer. The current baseline builder requires a thick-water reference of at least 10 mm and checks that its water echo lies outside the fitted baseline window. Confirm those conditions using the new data.
5. Save all session folders and note which baseline belongs to which test series. Baseline construction is a signal-reference operation; it does not replace independent thickness calibration or verification.

The baseline must include enough stable frames for template construction and residual-threshold estimation. A short multi-second capture is a practical initial recording, but inspect its completeness and variation before accepting it.

## Measure a test series

Change water depth while preserving mounting and coupling. For each condition, record the independently measured reference, settling state, temperature if available, raw frames, and the analysis command/settings. Do not identify ground truth from the same echo measurement being evaluated.

Volume divided by wetted area describes a mean depth. If used as a reference, document the area/volume measurement and whether local depth at the sensor can differ because of tilt, wetting, or surface shape. Keep uncertainty from the physical reference separate from frame-to-frame variation in the ultrasonic result.

For thin-layer processing, use the fresh baseline from the same installation. Retain invalid detections and `BASELINE_TIDAK_COCOK` results when evaluating performance. Disabling a mismatch threshold may be useful for diagnosis but is not evidence that a thickness estimate is correct.

If mounting, coupling, acquisition settings, or relevant geometry changes, start a new series and acquire a fresh baseline. Alignment and amplitude fitting cannot be assumed to correct every change in echo shape.

## Check drift and close the session

Return to the same thick-water reference condition and record it again. Compare the pre/post reference echo shape, timing, fitted residuals, and thickness estimate. If they differ materially relative to the preselected acceptance criteria, investigate the affected series rather than treating the initial baseline as valid throughout.

Stop live acquisition and verify output state. Confirm that every recording has `complete: true`, the expected binary size, valid timing rows, and a matching DSP result count. Back up the raw files and session notes. Keep interrupted recordings distinguishable from completed runs.

If scope waveforms are needed as evidence, arrange explicit waveform export with its channel, sample interval, scale, trigger settings, and acquisition identity. The built-in scope preview does not provide that archive or synchronize scope frames with Red Pitaya frames.

## Suggested private session notes

```text
Session ID and date:
Software revision and exact commands:
Transducer / mounting / coupling:
Substrate and vessel geometry:
Water temperature and sound speed used:
Reference thickness, measurement method, and uncertainty:
Actual sample rate / trigger / generator / mux settings:
Baseline-before folder:
Test recording folders and reference conditions:
Baseline-after folder:
Frame counts / completeness / detection and mismatch rates:
Observed changes, interruptions, and exclusions with reasons:
```

When comparing another week's assembly, report between-session differences separately from within-session repeatability. A new baseline is part of the measurement procedure, not proof that installation variability has disappeared.
